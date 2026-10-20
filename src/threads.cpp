#include "threads.h"
#include <algorithm>

ThreadPool::ThreadPool(std::size_t hash_mb) : tt_(hash_mb) {
    searchers_.push_back(std::make_unique<Searcher>(tt_));
}

void ThreadPool::set_threads(int n) {
    const auto want = static_cast<std::size_t>(std::max(n, 1));
    wait_helpers();
    // Helpers are made afresh; threads are kept (they idle between searches).
    workers_.resize(want - 1);
    searchers_.resize(1);
    helper_pos_.resize(want - 1);
    for (std::size_t i = 1; i < want; ++i) {
        searchers_.push_back(std::make_unique<Searcher>(tt_));
        if (!workers_[i - 1]) workers_[i - 1] = std::make_unique<Worker>();
    }
}

void ThreadPool::new_game() {
    wait_helpers();
    tt_.clear();
    for (auto& s : searchers_) s->clear_history();
}

Move ThreadPool::search(Position& pos, const SearchLimits& limits,
                        const std::vector<std::uint64_t>& history, bool verbose) {
    Searcher& lead = main();
    wait_helpers();  // the last search's, which may still be finishing
    if (searchers_.size() == 1) return lead.search(pos, limits, history, verbose);

    // The helpers search without limits of their own (a depth limit
    // aside), until the main searcher is done.
    SearchLimits helper_limits;
    helper_limits.infinite = true;
    helper_limits.depth = limits.depth;
    lead.begin_search(pos, limits);
    std::vector<const Searcher*> helpers;
    for (std::size_t i = 1; i < searchers_.size(); ++i) {
        Searcher& h = *searchers_[i];
        h.clear_stop();
        h.adopt_root(lead);
        helper_pos_[i - 1] = pos;
        helpers.push_back(&h);
        // `history` by value: the caller's may be gone before a helper starts.
        workers_[i - 1]->start([&h, &p = helper_pos_[i - 1], history, helper_limits] {
            h.search(p, helper_limits, history, false);
        });
    }
    lead.set_helpers(helpers);
    const Move best = lead.search(pos, limits, history, verbose);
    for (std::size_t i = 1; i < searchers_.size(); ++i) searchers_[i]->request_stop();
    lead.set_helpers({});  // it reports no more
    return best;           // the helpers may still be stopping: see threads.h
}
