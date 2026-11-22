"""Expert-cache simulator over a real routing trace.

A trace is an int array [tokens, layers, top_k] of the experts each token used (collected from the
real model). For a cache of N expert slots it answers the question the whole project hinges on:
how many experts must be read from disk per token, and so how fast can decoding go.

Policies:
  lru        least recently used, one cache shared by all layers
  lru-layer  one LRU per layer, slots split evenly
  hybrid     LRU that never evicts a learned hot set; protected share by budget (what the engine uses)
  lfu     least frequently used, counts halved every `decay` lookups so old popularity fades
  static  pin the N most frequent experts of the first `calib` fraction of the trace, never evict;
          misses are read and dropped (models a precomputed hot set with no dynamic cache)

`chunk` > 1 models speculative verification: the union of a chunk's experts is needed per layer.

    python tools/cachesim.py trace.npy --budget-gb 1.5 3 5 --expert-mb 2.6 --disk-mbps 2500
"""
import argparse
import collections
import heapq

import numpy as np


class LRU:
    def __init__(self, slots):
        self.slots, self.d = slots, collections.OrderedDict()

    def lookup(self, key):
        if key in self.d:
            self.d.move_to_end(key)
            return True
        if self.slots:
            if len(self.d) >= self.slots:
                self.d.popitem(last=False)
            self.d[key] = None
        return False


class LFU:
    """Lazy min-heap of (count, last_use, key); stale entries are skipped on pop."""

    def __init__(self, slots, decay=4096):
        self.slots, self.decay = slots, decay
        self.count, self.last, self.heap, self.tick = {}, {}, [], 0

    def _rebuild(self):
        for k in self.count:
            self.count[k] *= 0.5
        self.heap = [(c, self.last[k], k) for k, c in self.count.items()]
        heapq.heapify(self.heap)

    def lookup(self, key):
        self.tick += 1
        if self.tick % self.decay == 0:
            self._rebuild()
        hit = key in self.count
        if not hit:
            if not self.slots:
                return False
            while len(self.count) >= self.slots:
                c, t, k = heapq.heappop(self.heap)
                if self.count.get(k) == c and self.last[k] == t:
                    del self.count[k], self.last[k]
            self.count[key] = 0.0
        self.count[key] += 1.0
        self.last[key] = self.tick
        heapq.heappush(self.heap, (self.count[key], self.tick, key))
        return hit


class PerLayerLRU:
    """One LRU per layer with slots split evenly. A global LRU smaller than one token's working set
    (layers x top_k) evicts every expert just before it is reused; per-layer caches keep each layer's
    most recent experts no matter how small the budget is."""

    def __init__(self, slots, n_layers):
        self.caches = [LRU(slots // n_layers + (l < slots % n_layers)) for l in range(n_layers)]

    def lookup(self, key):
        return self.caches[key[0]].lookup(key)


class Hybrid:
    """LRU whose eviction skips 'hot' experts: the `protect` fraction of slots with the highest decayed
    use counts (recomputed every `refresh` lookups). protect=0 is plain LRU; near 1 it approaches an
    online-learned static hot set, which is what tiny budgets need."""

    def __init__(self, slots, protect, decay=4096, refresh=256):
        self.slots, self.n_hot = slots, int(slots * protect)
        self.decay, self.refresh = decay, refresh
        self.lru, self.count, self.hot, self.tick = collections.OrderedDict(), collections.Counter(), set(), 0

    def lookup(self, key):
        self.tick += 1
        self.count[key] += 1
        if self.tick % self.decay == 0:
            for k in self.count:
                self.count[k] *= 0.5
        if self.n_hot and self.tick % self.refresh == 0:
            self.hot = {k for k, _ in self.count.most_common(self.n_hot)}
        if key in self.lru:
            self.lru.move_to_end(key)
            return True
        if not self.slots:
            return False
        if len(self.lru) >= self.slots:
            victim = next((k for k in self.lru if k not in self.hot), None)
            if victim is None:
                victim = next(iter(self.lru))
            del self.lru[victim]
        self.lru[key] = None
        return False


class Static:
    def __init__(self, slots, trace, calib):
        n = max(1, int(len(trace) * calib))
        keys, freq = collections.Counter(), trace[:n]
        for l in range(trace.shape[1]):
            keys.update((l, int(e)) for e in freq[:, l, :].ravel())
        self.pinned = {k for k, _ in keys.most_common(slots)}

    def lookup(self, key):
        return key in self.pinned


def auto_protect(slots, working_set):
    """Hot-set share the engine uses (src/engine/expert_cache.cpp), picked from a sweep on a real trace."""
    return 0.9 if slots < working_set else 0.5 if slots < 2 * working_set else 0.75


def simulate(trace, slots, policy="lfu", chunk=1, warm=0, calib=0.25):
    """Returns hit rate and expert reads per token, measured after the first `warm` tokens."""
    trace = np.asarray(trace)
    T, L, K = trace.shape
    cache = {"lru": lambda: LRU(slots), "lfu": lambda: LFU(slots),
             "lru-layer": lambda: PerLayerLRU(slots, L),
             "hybrid": lambda: Hybrid(slots, auto_protect(slots, L * K)),
             "static": lambda: Static(slots, trace, calib)}[policy]()
    hits = misses = 0
    for t0 in range(0, T, chunk):
        block = trace[t0:t0 + chunk]
        count = t0 >= warm
        for l in range(L):
            for e in np.unique(block[:, l, :]):
                hit = cache.lookup((l, int(e)))
                if count:
                    hits += hit
                    misses += not hit
    n_tok = max(1, T - min(warm, T))
    n = hits + misses
    return dict(hit_rate=hits / n if n else 0.0, reads_per_token=misses / n_tok, tokens=n_tok)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("trace", help=".npy int array [tokens, layers, top_k]")
    size = ap.add_mutually_exclusive_group(required=True)
    size.add_argument("--slots", type=int, nargs="+", help="cache sizes in experts")
    size.add_argument("--budget-gb", type=float, nargs="+", help="cache sizes in GB (needs --expert-mb)")
    ap.add_argument("--expert-mb", type=float, default=None, help="bytes of one expert (gate+up+down), MB")
    ap.add_argument("--policy", nargs="+", default=["lru", "hybrid", "static"],
                    choices=["lru", "lru-layer", "lfu", "hybrid", "static"])
    ap.add_argument("--chunk", type=int, default=1, help="tokens verified per pass (speculative)")
    ap.add_argument("--warm", type=int, default=64, help="tokens excluded from stats (cold start)")
    ap.add_argument("--calib", type=float, default=0.25, help="trace fraction used to pick the static set")
    ap.add_argument("--disk-mbps", type=float, default=None, help="measured disk MB/s for a tok/s estimate")
    a = ap.parse_args()

    trace = np.load(a.trace)
    T, L, K = trace.shape
    total = L * (int(trace.max()) + 1)
    if a.budget_gb:
        if not a.expert_mb:
            ap.error("--budget-gb needs --expert-mb")
        sizes = [(int(g * 1024 / a.expert_mb), "%.1f GB" % g) for g in a.budget_gb]
    else:
        sizes = [(s, "") for s in a.slots]
    print("trace: %d tokens, %d layers, top-%d, ~%d experts seen as max id" % (T, L, K, total))
    print("%-10s %-8s %8s %7s %7s %12s %10s" % ("policy", "budget", "slots", "cached", "hit", "reads/token",
                                             "MB/token" if a.expert_mb else ""))
    for slots, label in sizes:
        for pol in a.policy:
            r = simulate(trace, slots, pol, a.chunk, a.warm, a.calib)
            line = "%-10s %-8s %8d %6.1f%% %6.1f%% %12.1f" % (
                pol, label, slots, 100 * min(1, slots / total), 100 * r["hit_rate"], r["reads_per_token"])
            if a.expert_mb:
                mb = r["reads_per_token"] * a.expert_mb
                line += " %10.0f" % mb
                if a.disk_mbps and mb:
                    line += "   ~%.1f tok/s (disk only)" % (a.disk_mbps / mb)
            print(line)


if __name__ == "__main__":
    main()
