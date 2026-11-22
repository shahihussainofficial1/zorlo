#include "expert_cache.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    define NOMINMAX
#    include <windows.h>
#else
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <unistd.h>
#endif

namespace zorlo {

static constexpr int MIN_SLOTS = 16;
static constexpr uint64_t DECAY_EVERY = 4096;   // lookups between halving all use counts
static constexpr uint64_t REFRESH_EVERY = 256;  // lookups between hot-set recomputations

#ifdef _WIN32
static void * open_unbuffered(const std::string & path) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, w.data(), n);
    HANDLE h = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (h == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + path);
    return h;
}
static void close_file(void * f) { CloseHandle((HANDLE) f); }
static bool read_at(void * f, uint64_t off, uint8_t * dst, size_t n) {
    OVERLAPPED ov = {};
    ov.Offset = (DWORD) (off & 0xFFFFFFFFu);
    ov.OffsetHigh = (DWORD) (off >> 32);
    DWORD got = 0;
    return ReadFile((HANDLE) f, dst, (DWORD) n, &got, &ov) && got == n;
}
static uint8_t * reserve_arena(size_t n) {
    void * p = VirtualAlloc(nullptr, n, MEM_RESERVE, PAGE_NOACCESS);
    if (!p) throw std::runtime_error("cannot reserve expert cache address space");
    return (uint8_t *) p;
}
static void commit(uint8_t * p, size_t n) {
    if (n && !VirtualAlloc(p, n, MEM_COMMIT, PAGE_READWRITE)) throw std::runtime_error("cannot allocate expert cache");
}
static void decommit(uint8_t * p, size_t n) { if (n) VirtualFree(p, n, MEM_DECOMMIT); }
static void free_arena(uint8_t * p, size_t) { VirtualFree(p, 0, MEM_RELEASE); }
size_t available_ram_bytes() {
    MEMORYSTATUSEX m = { sizeof(m) };
    return GlobalMemoryStatusEx(&m) ? (size_t) m.ullAvailPhys : 0;
}
#else
static void * open_unbuffered(const std::string & path) {
    int flags = O_RDONLY;
#    ifdef O_DIRECT
    flags |= O_DIRECT;
#    endif
    const int fd = open(path.c_str(), flags);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    return (void *) (intptr_t) (fd + 1);
}
static void close_file(void * f) { close((int) (intptr_t) f - 1); }
static bool read_at(void * f, uint64_t off, uint8_t * dst, size_t n) {
    size_t done = 0;
    while (done < n) {
        const ssize_t r = pread((int) (intptr_t) f - 1, dst + done, n - done, (off_t) (off + done));
        if (r <= 0) return false;
        done += (size_t) r;
    }
    return true;
}
static uint8_t * reserve_arena(size_t n) {
    void * p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) throw std::runtime_error("cannot reserve expert cache address space");
    return (uint8_t *) p;
}
static void commit(uint8_t *, size_t) {}  // pages are backed on first touch
static void decommit(uint8_t * p, size_t n) { if (n) madvise(p, n, MADV_DONTNEED); }
static void free_arena(uint8_t * p, size_t n) { munmap(p, n); }
size_t available_ram_bytes() {
    FILE * f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    size_t kb = 0;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "MemAvailable: %zu kB", &kb) == 1) break;
    }
    fclose(f);
    return kb * 1024;
}
#endif

ExpertCache::ExpertCache(const Pack & pack, size_t budget_bytes, int io_threads, float protect, size_t max_bytes)
    : pack_(pack), n_expert_(pack.hp().n_expert) {
    slot_bytes_ = pack.max_stride();
    const size_t all = (size_t) pack.hp().n_layer * n_expert_;  // more slots than experts is pointless
    const size_t max_n = std::clamp<size_t>(std::max(max_bytes, budget_bytes) / slot_bytes_, MIN_SLOTS, all);
    arena_ = reserve_arena(max_n * slot_bytes_);
    slots_.resize(max_n);
    protect_auto_ = protect < 0;
    protect_ = protect;
    count_.assign(all, 0.0f);
    hot_.assign(count_.size(), 0);
    resize(budget_bytes);

    for (int i = 0; i < std::max(1, io_threads); ++i) {
        files_.push_back(open_unbuffered(pack.experts_path()));
        lo_files_.push_back(pack.has_lowbit() ? open_unbuffered(pack.lo_experts_path()) : nullptr);
        workers_.emplace_back(&ExpertCache::worker, this, i);
    }
}

ExpertCache::~ExpertCache() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_work_.notify_all();
    for (auto & t : workers_) t.join();
    for (void * f : files_) close_file(f);
    for (void * f : lo_files_) if (f) close_file(f);
    if (arena_) free_arena(arena_, slots_.size() * slot_bytes_);
}

void ExpertCache::touch(int64_t key) {
    ++lookups_;
    count_[key] += 1.0f;
    if (lookups_ % DECAY_EVERY == 0) {
        for (float & c : count_) c *= 0.5f;
    }
    if (lookups_ % REFRESH_EVERY == 0) refresh_hot();
}

void ExpertCache::refresh_hot() {
    if (n_hot_ <= 0) return;
    std::vector<float> c = count_;
    const int k = std::min<int>(n_hot_, (int) c.size()) - 1;
    std::nth_element(c.begin(), c.begin() + k, c.end(), std::greater<float>());
    const float thr = std::max(c[k], 1e-6f);
    for (size_t i = 0; i < count_.size(); ++i) hot_[i] = count_[i] >= thr;
}

std::vector<float> ExpertCache::profile() const {
    std::lock_guard<std::mutex> lk(mu_);
    return count_;
}

void ExpertCache::set_profile(const std::vector<float> & counts) {
    size_t n;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (counts.size() != count_.size()) return;  // another model's profile
        count_ = counts;
        refresh_hot();
        n = std::min(counts.size(), (size_t) active_);
    }
    std::vector<int64_t> keys(counts.size());
    for (size_t i = 0; i < keys.size(); ++i) keys[i] = (int64_t) i;
    std::partial_sort(keys.begin(), keys.begin() + n, keys.end(), [&](int64_t a, int64_t b) { return counts[a] > counts[b]; });
    for (size_t i = 0; i < n && counts[keys[i]] > 0; ++i) prefetch((int) (keys[i] / n_expert_), (int) (keys[i] % n_expert_));
}

int ExpertCache::victim(bool for_prefetch) const {
    int cold = -1, any = -1;
    for (int i = 0; i < active_; ++i) {
        const Slot & s = slots_[i];
        if (s.pinned || s.state == QUEUED || s.state == LOADING) continue;
        if (s.state == EMPTY) return i;
        if (any < 0 || s.last_use < slots_[any].last_use) any = i;
        if (!hot_[s.key] && (cold < 0 || s.last_use < slots_[cold].last_use)) cold = i;
    }
    return cold >= 0 ? cold : for_prefetch ? -1 : any;
}

int ExpertCache::assign(int64_t key, bool for_prefetch, bool lo_ok) {
    const int v = victim(for_prefetch);
    if (v < 0) return -1;
    Slot & s = slots_[v];
    if (s.key >= 0) where_.erase(s.key);
    s.key = key;
    s.state = QUEUED;
    s.prefetched = for_prefetch;
    s.lo = lowbit_ && lo_ok && !hot_[key];  // hot experts are worth the full read
    s.want_full = false;
    where_[key] = v;
    return v;
}

int ExpertCache::request(int il, int e, bool lo_ok) {
    const int64_t key = (int64_t) il * n_expert_ + e;
    std::lock_guard<std::mutex> lk(mu_);
    touch(key);
    int slot;
    const auto it = where_.find(key);
    if (it != where_.end()) {
        slot = it->second;
        Slot & s = slots_[slot];
        const bool full = s.lo && (hot_[key] || !lo_ok);  // turned hot, or an exact request: needs the full copy
        if (full && s.state == READY) {
            s.state = QUEUED;  // re-read it in full, once
            s.lo = false;
            demand_q_.push_back({ slot, key });
            ++stats_.upgrades;
        } else if (full && !lo_ok && s.state == QUEUED) {
            s.lo = false;  // not started yet: read the full copy instead
            demand_q_.push_front({ slot, key });
            ++stats_.late;
        } else if (full && !lo_ok && s.state == LOADING) {
            s.want_full = true;  // the worker re-queues it in full when this read lands
            ++stats_.late;
        } else if (s.state == READY) {
            ++stats_.hits;
        } else {
            ++stats_.late;
            if (s.state == QUEUED) demand_q_.push_front({ slot, key });  // promote the queued prefetch
        }
        if (s.prefetched) {
            ++stats_.prefetch_used;
            s.prefetched = false;
        }
    } else {
        slot = assign(key, false, lo_ok);
        if (slot < 0) throw std::runtime_error("expert cache: every slot is pinned or loading");
        demand_q_.push_back({ slot, key });
        ++stats_.misses;
    }
    slots_[slot].pinned = true;
    slots_[slot].last_use = ++tick_;
    cv_work_.notify_one();
    return slot;
}

void ExpertCache::prefetch(int il, int e, bool lo_ok) {
    const int64_t key = (int64_t) il * n_expert_ + e;
    std::lock_guard<std::mutex> lk(mu_);
    if (where_.count(key)) return;
    const int slot = assign(key, true, lo_ok);
    if (slot < 0) return;
    slots_[slot].last_use = ++tick_;
    prefetch_q_.push_back({ slot, key });
    ++stats_.prefetch_issued;
    cv_work_.notify_one();
}

void ExpertCache::cancel_prefetch() {
    std::lock_guard<std::mutex> lk(mu_);
    cancel_prefetch_locked();
}

void ExpertCache::cancel_prefetch_locked() {
    for (const Job & j : prefetch_q_) {
        Slot & s = slots_[j.slot];
        if (s.key == j.key && s.state == QUEUED && s.prefetched) {
            where_.erase(s.key);
            s = Slot();
        }
    }
    prefetch_q_.clear();
}

void ExpertCache::worker(int id) {
    for (;;) {
        Job job;
        bool lo = false;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_work_.wait(lk, [&] { return stop_ || !demand_q_.empty() || !prefetch_q_.empty(); });
            if (stop_) return;
            std::deque<Job> & q = !demand_q_.empty() ? demand_q_ : prefetch_q_;
            job = q.front();
            q.pop_front();
            Slot & s = slots_[job.slot];
            if (s.key != job.key || s.state != QUEUED) continue;  // cancelled, or promoted and already taken
            s.state = LOADING;
            lo = s.lo;
        }
        const int il = (int) (job.key / n_expert_);
        const ExpertLayout & L = lo ? pack_.lo_layer(il) : pack_.layer(il);
        const uint64_t off = L.offset + (uint64_t) (job.key % n_expert_) * L.stride;
        const bool ok = read_at(lo ? lo_files_[id] : files_[id], off, const_cast<uint8_t *>(data(job.slot)), L.stride);
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!ok) {
                fprintf(stderr, "zorlo: expert read failed at offset %llu\n", (unsigned long long) off);
                std::abort();
            }
            Slot & s = slots_[job.slot];
            stats_.bytes_read += L.stride;
            stats_.lo_reads += lo;
            if (s.want_full) {  // an exact request arrived during a low-bit read
                s.want_full = false;
                s.lo = false;
                s.state = QUEUED;
                demand_q_.push_front(job);
                ++stats_.upgrades;
                cv_work_.notify_one();
                continue;
            }
            s.state = READY;
        }
        cv_done_.notify_all();
    }
}

std::vector<int> ExpertCache::wait_any(std::vector<int> & pending) {
    std::unique_lock<std::mutex> lk(mu_);
    auto ready = [&] {
        for (int s : pending) {
            if (slots_[s].state == READY) return true;
        }
        return false;
    };
    if (!ready()) {
        const auto t0 = std::chrono::steady_clock::now();
        cv_done_.wait(lk, ready);
        stats_.wait_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    std::vector<int> out, rest;
    for (int s : pending) (slots_[s].state == READY ? out : rest).push_back(s);
    pending.swap(rest);
    return out;
}

void ExpertCache::set_protect() {
    // share of slots kept for the learned hot set, tuned on a real routing trace (tools/cachesim.py):
    // below one token's working set recency is useless, so protect almost everything
    if (protect_auto_) {
        const int ws = pack_.hp().n_layer * pack_.hp().n_expert_used;
        protect_ = active_ < ws ? 0.9f : active_ < 2 * ws ? 0.5f : 0.75f;
    }
    n_hot_ = (int) (active_ * protect_);
}

int ExpertCache::resize(size_t budget_bytes) {
    const int n = (int) std::clamp<size_t>(budget_bytes / slot_bytes_, MIN_SLOTS, slots_.size());
    std::unique_lock<std::mutex> lk(mu_);
    if (n == active_) return n;
    if (n > active_) {
        commit(arena_ + (size_t) active_ * slot_bytes_, (size_t) (n - active_) * slot_bytes_);
    } else {
        // let reads into the dropped slots finish before their memory goes away
        cancel_prefetch_locked();
        cv_done_.wait(lk, [&] {
            for (int i = n; i < active_; ++i) {
                if (slots_[i].state == QUEUED || slots_[i].state == LOADING) return false;
            }
            return true;
        });
        for (int i = n; i < active_; ++i) {
            if (slots_[i].pinned) throw std::runtime_error("expert cache: resize while experts are pinned");
            if (slots_[i].key >= 0) where_.erase(slots_[i].key);
            slots_[i] = Slot();
        }
        decommit(arena_ + (size_t) n * slot_bytes_, (size_t) (active_ - n) * slot_bytes_);
    }
    active_ = n;
    set_protect();
    return n;
}

bool ExpertCache::is_lo(int slot) const {
    std::lock_guard<std::mutex> lk(mu_);
    return slots_[slot].lo;
}

void ExpertCache::set_lowbit(bool on) {
    if (on && !pack_.has_lowbit()) throw std::runtime_error("--lowbit needs experts.lo.bin in the pack (zorlo pack --lowbit, or zorlo pull --lowbit)");
    std::lock_guard<std::mutex> lk(mu_);
    lowbit_ = on;
}

bool ExpertCache::resident(int il, int e) const {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = where_.find((int64_t) il * n_expert_ + e);
    return it != where_.end() && slots_[it->second].state == READY;
}

size_t ExpertCache::resident_bytes() const {
    std::lock_guard<std::mutex> lk(mu_);
    size_t n = 0;
    for (int i = 0; i < active_; ++i) n += slots_[i].key >= 0;
    return n * slot_bytes_;
}

void ExpertCache::release_all() {
    std::lock_guard<std::mutex> lk(mu_);
    for (Slot & s : slots_) s.pinned = false;
}

ExpertCache::Stats ExpertCache::stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    return stats_;
}

void ExpertCache::reset_stats() {
    std::lock_guard<std::mutex> lk(mu_);
    stats_ = Stats();
}

}  // namespace zorlo
