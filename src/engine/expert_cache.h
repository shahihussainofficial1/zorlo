#pragma once

// Expert cache: a fixed arena of 4 KB-aligned slots, each holding one expert blob read straight from
// experts.bin with unbuffered I/O (the OS page cache is bypassed, so RAM use is exactly the budget).
//
// Reads run on a pool of I/O threads, several in flight at once (NVMe needs queue depth >= 2).
// Demand reads go before prefetches. Eviction is LRU, except that the `protect` share of slots
// with the highest decayed use counts is never evicted: tiny budgets need a learned hot set
// (plain LRU smaller than one token's working set always misses), large ones mostly recency.
//
// Usage per layer: request() every needed expert (pins it), then wait_any() repeatedly and compute
// the experts as they land, then release_all().

#include "pack.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace zorlo {

// physical RAM free for new allocations, 0 if unknown
size_t available_ram_bytes();

class ExpertCache {
public:
    // protect < 0 picks the share from the budget. Address space is reserved for max_bytes (0 = budget)
    // so resize() can grow up to it; only the active slots hold RAM.
    ExpertCache(const Pack & pack, size_t budget_bytes, int io_threads = 4, float protect = -1.0f, size_t max_bytes = 0);
    ~ExpertCache();
    ExpertCache(const ExpertCache &) = delete;
    ExpertCache & operator=(const ExpertCache &) = delete;

    // start loading expert e of layer il if needed, pin it, return its slot
    int request(int il, int e, bool lo_ok = false);
    // load in the background if a slot can be freed without touching hot or pinned experts
    void prefetch(int il, int e, bool lo_ok = false);
    // drop queued (not yet started) prefetches
    void cancel_prefetch();
    // block until at least one of `slots` is loaded; returns the loaded ones (removed from `slots`)
    std::vector<int> wait_any(std::vector<int> & slots);
    const uint8_t * data(int slot) const { return arena_ + (size_t) slot * slot_bytes_; }
    // the slot holds the pack's low-bit copy (layout pack.lo_layer); stable while pinned
    bool is_lo(int slot) const;
    // lossy: allow reading the low-bit copy (pack must have one); a read uses it only when the request passes
    // lo_ok and the expert is outside the hot set. A request without lo_ok gets a full copy, re-read if needed
    void set_lowbit(bool on);
    void release_all();
    // expert loaded and usable right now (no read needed)
    bool resident(int il, int e) const;
    // change the RAM used to about budget_bytes (clamped to [16 slots, max_bytes]); evicted slots' memory
    // goes back to the OS. Call between evals (nothing pinned). Returns the new slot count.
    int resize(size_t budget_bytes);

    int n_slots() const { return active_; }
    size_t bytes() const { return (size_t) active_ * slot_bytes_; }
    // RAM actually backed: committed pages count against free RAM only once an expert is written into them
    size_t resident_bytes() const;
    size_t slot_bytes() const { return slot_bytes_; }
    float protect() const { return protect_; }

    // learned per-expert use counts, kept across runs so the hot set needs no relearning
    std::vector<float> profile() const;
    // restore counts from profile(), then read the most used experts into free slots in the background
    void set_profile(const std::vector<float> & counts);

    struct Stats {
        uint64_t hits = 0, misses = 0, late = 0;  // late = requested while its prefetch was still in flight
        uint64_t bytes_read = 0, prefetch_issued = 0, prefetch_used = 0;
        uint64_t lo_reads = 0, upgrades = 0;  // low-bit reads; low-bit experts re-read in full once hot
        double   wait_s = 0;  // time compute spent blocked on reads
    };
    Stats stats() const;
    void reset_stats();

private:
    enum State : uint8_t { EMPTY, QUEUED, LOADING, READY };
    struct Slot {
        int64_t  key = -1;
        uint64_t last_use = 0;
        State    state = EMPTY;
        bool     pinned = false;
        bool     prefetched = false;  // loaded by prefetch and not requested yet
        bool     lo = false;          // holds (or is loading) the low-bit copy
        bool     want_full = false;   // loading the low-bit copy, but an exact request needs it: re-read after
    };
    struct Job {
        int     slot;
        int64_t key;
    };

    int  victim(bool for_prefetch) const;  // -1 if none
    void set_protect();
    void cancel_prefetch_locked();
    int  assign(int64_t key, bool for_prefetch, bool lo_ok);
    void touch(int64_t key);
    void refresh_hot();
    void worker(int id);

    const Pack & pack_;
    const int n_expert_;
    size_t slot_bytes_ = 0;
    uint8_t * arena_ = nullptr;
    std::vector<Slot> slots_;  // all reserved slots; only the first active_ are in use
    int active_ = 0;
    std::unordered_map<int64_t, int> where_;
    uint64_t tick_ = 0;

    // hot set: decayed use counts per expert, top n_hot_ protected from eviction
    float protect_ = 0;
    bool protect_auto_ = false;
    int n_hot_ = 0;
    std::vector<float> count_;
    std::vector<uint8_t> hot_;
    uint64_t lookups_ = 0;

    mutable std::mutex mu_;
    std::condition_variable cv_work_, cv_done_;
    std::deque<Job> demand_q_, prefetch_q_;
    std::vector<std::thread> workers_;
    std::vector<void *> files_, lo_files_;
    bool lowbit_ = false;
    bool stop_ = false;
    Stats stats_;
};

}  // namespace zorlo
