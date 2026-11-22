import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from cachesim import LFU, LRU, Hybrid, PerLayerLRU, simulate  # noqa: E402


class CacheSimTest(unittest.TestCase):
    def test_lru_evicts_oldest(self):
        c = LRU(2)
        self.assertEqual([c.lookup(k) for k in "abacb"], [False, False, True, False, False])

    def test_lfu_keeps_frequent(self):
        c = LFU(2)
        for k in "aaab":
            c.lookup(k)
        self.assertFalse(c.lookup("c"))  # evicts b (count 1), not a (count 3)
        self.assertTrue(c.lookup("a"))
        self.assertFalse(c.lookup("b"))

    def test_lfu_stays_within_slots(self):
        c = LFU(5, decay=7)
        rng = np.random.default_rng(0)
        for k in rng.integers(0, 50, 2000):
            c.lookup(int(k))
            self.assertLessEqual(len(c.count), 5)

    def test_per_layer_lru_isolates_layers(self):
        c = PerLayerLRU(4, 2)  # 2 slots per layer
        for k in [(0, 1), (0, 2), (1, 1), (1, 2), (1, 3)]:  # layer 1 churns
            c.lookup(k)
        self.assertTrue(c.lookup((0, 1)))  # layer 0 untouched by layer 1's evictions
        self.assertFalse(c.lookup((1, 1)))

    def test_hybrid_keeps_hot_expert_through_a_scan(self):
        c = Hybrid(4, protect=0.5, refresh=1)
        for _ in range(10):
            c.lookup("hot")
        for k in range(20):  # a scan of one-off experts would flush plain LRU
            c.lookup(k)
        self.assertTrue(c.lookup("hot"))

    def test_zero_slots_never_hits(self):
        tr = np.zeros((10, 2, 2), np.int16)
        for pol in ("lru", "lru-layer", "lfu", "hybrid", "static"):
            r = simulate(tr, 0, pol, warm=0)
            self.assertEqual(r["hit_rate"], 0.0)
            self.assertEqual(r["reads_per_token"], 2.0)  # one unique expert per layer

    def test_full_cache_only_cold_misses(self):
        rng = np.random.default_rng(1)
        tr = rng.integers(0, 8, (200, 3, 2)).astype(np.int16)
        for pol in ("lru", "lfu"):
            r = simulate(tr, 24, pol, warm=0)
            self.assertLessEqual(r["reads_per_token"] * 200, 24)

    def test_chunk_unions_experts(self):
        tr = np.zeros((4, 1, 2), np.int16)
        tr[:, 0, 1] = [1, 2, 3, 4]
        r = simulate(tr, 0, "lru", chunk=4, warm=0)
        self.assertEqual(r["reads_per_token"], 5 / 4)  # union {0,1,2,3,4} once for 4 tokens


if __name__ == "__main__":
    unittest.main()
