#pragma once
#include "search.h"
#include <condition_variable>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#ifndef _WIN32
#include <pthread.h>
#endif

// A thread that lives as long as its owner and runs one job at a time,
// sleeping in between. Starting a new thread for each "go" would be
// simpler, but on a busy machine a new thread can wait tens of
// milliseconds for its first time slice, all of it on the engine's clock:
// measured with 24 match games on 24 hardware threads, a median of 6 ms
// and up to 47 ms, where waking a waiting thread took 15 us.
class Worker {
public:
    Worker() { launch(); }
    ~Worker() {
        {
            std::lock_guard lock(mutex_);
            quit_=true;
        }
        cv_.notify_all();
        join();  // after the job in hand, if any
    }

    // Starts `job` on the thread. The previous job must have finished.
    void start(std::function<void()> job) {
        {
            std::lock_guard lock(mutex_);
            job_=std::move(job);
            busy_=true;
        }
        cv_.notify_all();
    }
    bool busy() {
        std::lock_guard lock(mutex_);
        return busy_;
    }
    void wait() {
        std::unique_lock lock(mutex_);
        cv_.wait(lock,[this] { return !busy_; });
    }

private:
    // The search recurses deeply (singular-extension probes nest frames
    // within a ply), so its thread gets a 16 MB stack everywhere, not the
    // platform default (512 KB for macOS secondary threads). std::thread
    // can't set a stack size: MinGW and MSVC builds set it at link time
    // (CMakeLists.txt), which covers every thread, and POSIX builds
    // create the thread with pthreads.
    static constexpr std::size_t STACK_SIZE = 16 << 20;
#ifdef _WIN32
    void launch() { thread_ = std::thread([this] { run(); }); }
    void join() { thread_.join(); }
#else
    void launch() {
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, STACK_SIZE);
        const int err = pthread_create(
            &thread_, &attr,
            [](void* self) -> void* {
                static_cast<Worker*>(self)->run();
                return nullptr;
            },
            this);
        pthread_attr_destroy(&attr);
        if (err) {
            std::cerr << "cannot start the search thread\n";
            std::abort();
        }
    }
    void join() { pthread_join(thread_, nullptr); }
#endif

    void run() {
        std::unique_lock lock(mutex_);
        while(true) {
            cv_.wait(lock,[this] { return job_ || quit_; });
            if(!job_) return;  // quitting, with nothing left to do
            auto job=std::move(job_);
            job_=nullptr;
            lock.unlock();
            job();
            lock.lock();
            busy_=false;
            cv_.notify_all();
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::function<void()> job_;
    bool busy_=false, quit_=false;
#ifdef _WIN32
    std::thread thread_;
#else
    pthread_t thread_{};
#endif
};

// Lazy SMP: `threads` searchers share one hash table and search the same
// root at once. The first, the main searcher, keeps time, reports and
// picks the move, exactly as a single-threaded search would; the helpers
// search until it is done, each on its own persistent thread, and reach it
// only through the hash table, where their results change which moves and
// bounds it finds. With one thread this is the plain search.
inline constexpr int MAX_THREADS = 256;

class ThreadPool {
public:
    explicit ThreadPool(std::size_t hash_mb = 64);
    // Helpers may still be stopping after search() returns: they must finish
    // before their positions (helper_pos_) and searchers go.
    ~ThreadPool() { wait_helpers(); }
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Call only while no search runs.
    void set_threads(int n);
    int threads() const { return static_cast<int>(searchers_.size()); }
    void resize_hash(std::size_t megabytes) {
        wait_helpers();
        tt_.resize(megabytes);
    }
    void new_game();

    // The main searcher: the single-threaded search (bench, statistics,
    // options that concern it alone).
    Searcher& main() { return *searchers_[0]; }

    // As Searcher::search, with every thread. Returns as soon as the main
    // searcher is done, so that the move can be sent at once: the helpers
    // have been told to stop but may still be finishing (a helper in a
    // tablebase probe under load can take tens of milliseconds to notice).
    // wait_helpers() before anything else touches the searchers or the
    // table; the next search and the calls below do it themselves.
    Move search(Position& pos, const SearchLimits& limits,
                const std::vector<std::uint64_t>& history, bool verbose = true);
    void wait_helpers() {
        for (auto& w : workers_) w->wait();
    }

    // As Searcher's; the helpers stop when the main searcher does.
    void request_stop() { main().request_stop(); }
    void clear_stop() { main().clear_stop(); }
    bool stop_requested() { return main().stop_requested(); }
    // As Searcher's: pondering concerns the main searcher alone.
    void ponderhit() { main().ponderhit(); }
    void clear_ponderhit() { main().clear_ponderhit(); }
    bool ponderhit_received() { return main().ponderhit_received(); }
    // Pruning and feature counters of all searchers together.
    prune::Stats total_stats() const {
        prune::Stats s;
        for (const auto& x : searchers_) s.add(x->stats());
        return s;
    }

private:
    TranspositionTable tt_;
    std::vector<std::unique_ptr<Searcher>> searchers_;  // [0]: main
    std::vector<std::unique_ptr<Worker>> workers_;      // one per helper
    std::vector<Position> helper_pos_;                  // each helper's copy of the root
};
