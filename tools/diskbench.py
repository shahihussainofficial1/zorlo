"""Disk benchmark for expert streaming: random, expert-sized, unbuffered reads at several queue depths.

The OS page cache is bypassed (FILE_FLAG_NO_BUFFERING on Windows, O_DIRECT on Linux), so the numbers
are what the SSD really delivers, the same way the engine will read experts.

    python tools/diskbench.py .bench --size-gb 4 --block-mb 2.75 --qd 1 2 4 8 16

Writes a test file of --size-gb into the directory (reused if already there) and deletes it at the
end unless --keep.
"""
import argparse
import ctypes
import os
import random
import sys
import threading
import time

MB = 1024 * 1024
ALIGN = 4096


def _align(n):
    return -(-n // ALIGN) * ALIGN


if sys.platform == "win32":
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class OVERLAPPED(ctypes.Structure):
        _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t),
                    ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD),
                    ("hEvent", wintypes.HANDLE)]

    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    k32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                             ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(OVERLAPPED)]
    k32.VirtualAlloc.restype = ctypes.c_void_p
    k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
    k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    INVALID = wintypes.HANDLE(-1).value

    class Reader:
        def __init__(self, path, block):
            self.h = k32.CreateFileW(path, 0x80000000, 1, None, 3, 0x20000000, None)  # NO_BUFFERING
            if self.h in (None, INVALID):
                raise ctypes.WinError(ctypes.get_last_error())
            self.block = block
            self.buf = k32.VirtualAlloc(None, block, 0x3000, 0x04)
            self.ov, self.got = OVERLAPPED(), wintypes.DWORD()

        def read(self, offset):
            self.ov.Offset, self.ov.OffsetHigh = offset & 0xFFFFFFFF, offset >> 32
            if not k32.ReadFile(self.h, self.buf, self.block, ctypes.byref(self.got), ctypes.byref(self.ov)):
                raise ctypes.WinError(ctypes.get_last_error())
            return self.got.value

        def close(self):
            k32.VirtualFree(self.buf, 0, 0x8000)
            k32.CloseHandle(self.h)
else:
    import mmap

    class Reader:
        def __init__(self, path, block):
            flags = os.O_RDONLY | getattr(os, "O_DIRECT", 0)
            if not hasattr(os, "O_DIRECT"):
                print("warning: no O_DIRECT here, numbers may include the page cache")
            self.fd, self.block = os.open(path, flags), block
            self.buf = mmap.mmap(-1, block)  # page aligned

        def read(self, offset):
            return os.preadv(self.fd, [self.buf], offset)

        def close(self):
            self.buf.close()
            os.close(self.fd)


def make_file(path, size):
    if os.path.exists(path) and os.path.getsize(path) >= size:
        print("reusing %s" % path)
        return
    print("writing %.1f GB test file %s ..." % (size / 1024 ** 3, path), flush=True)
    chunk = os.urandom(64 * MB)
    t0 = time.perf_counter()
    with open(path, "wb") as f:
        for _ in range(size // len(chunk)):
            f.write(chunk)
        f.flush()
        os.fsync(f.fileno())
    print("  write %.0f MB/s" % (size / MB / (time.perf_counter() - t0)))


def run(path, size, block, qd, seconds, seed=0):
    slots = (size - block) // ALIGN
    stop, done, lat, errs = threading.Event(), [0] * qd, [0.0] * qd, []

    def worker(i):
        rng = random.Random(seed * 1000 + i)
        r = Reader(path, block)
        try:
            while not stop.is_set():
                t = time.perf_counter()
                if r.read(rng.randrange(slots) * ALIGN) != block:
                    raise IOError("short read")
                lat[i] += time.perf_counter() - t
                done[i] += 1
        except Exception as e:  # surfaced after the run
            errs.append(e)
        finally:
            r.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(qd)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    time.sleep(seconds)
    stop.set()
    for t in threads:
        t.join()
    if errs:
        raise errs[0]
    dt, n = time.perf_counter() - t0, sum(done)
    return n * block / MB / dt, n / dt, 1000 * sum(lat) / max(1, n)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir", help="directory on the SSD to test (models will live there)")
    ap.add_argument("--size-gb", type=float, default=4.0)
    ap.add_argument("--block-mb", type=float, default=2.75, help="one expert's bytes (Qwen3-30B-A3B Q4_K ~2.6-3.1)")
    ap.add_argument("--qd", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--experts-per-token", type=int, default=384, help="layers x top_k (48 x 8 for Qwen3-30B-A3B)")
    ap.add_argument("--keep", action="store_true", help="keep the test file for later runs")
    a = ap.parse_args()

    os.makedirs(a.dir, exist_ok=True)
    path = os.path.join(a.dir, "zorlo_diskbench.bin")
    size, block = _align(int(a.size_gb * 1024 ** 3)), _align(int(a.block_mb * MB))
    make_file(path, size)
    try:
        print("random %.2f MB unbuffered reads, %.0fs per queue depth" % (block / MB, a.seconds))
        print("%4s %10s %10s %12s %22s" % ("QD", "MB/s", "reads/s", "ms/read", "no-cache tok/s ceiling"))
        for qd in a.qd:
            mbps, rps, ms = run(path, size, block, qd, a.seconds)
            print("%4d %10.0f %10.1f %12.2f %22.2f" % (qd, mbps, rps, ms, rps / a.experts_per_token),
                  flush=True)
    finally:
        if not a.keep:
            os.remove(path)


if __name__ == "__main__":
    main()
