#pragma once
#include "eval.h"
#include "position.h"
#include "pruning.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

struct SearchLimits {
    int depth = 0;           // 0 = no depth limit
    int movetime = 0;        // ms; 0 = not given
    int time[2] = {-1, -1};  // wtime, btime in ms; -1 = not given
    int inc[2] = {0, 0};
    int movestogo = 0;
    std::uint64_t nodes = 0; // 0 = no node limit
    // Soft node limit (data generation): no new iteration starts once the
    // search has this many nodes; the one in progress finishes, unless
    // `nodes` cuts it short. 0 = none.
    std::uint64_t soft_nodes = 0;
    // The soft node limit scaled as the soft time limit is (tm_* params: a
    // best move that keeps changing or a falling score earns more, one that
    // holds or took most of the nodes less; the author, 2026-10-07, datagen).
    bool soft_scale = false;
    // A floor under the soft node limit (the author, 2026-10-25, datagen: no
    // shallow labels by accident): no soft node stop until this depth has
    // completed; `nodes` still cuts it short. 0 = none.
    int min_depth = 0;
    // "go searchmoves": only these root moves (legal ones; with a tablebase
    // ranking, those of them it keeps, or its own if none). Empty: all.
    std::vector<Move> search_moves;
    bool infinite = false;
    // "go ponder": the position after the move we expect the opponent to
    // play. Searched without limits until "ponderhit" (then the clock
    // limits above apply, from that moment) or "stop".
    bool ponder = false;
};

// The key the hash table keeps `pos` under: its Zobrist key with the
// halfmove clock's bucket mixed in (params::fifty_hash_width; the key
// itself while that is 0). Repetition detection uses pos.key().
std::uint64_t tt_key(const Position& pos);

// Transposition table shared by all searches of a game, and by all search
// threads. Entries store the best move found, a score that is exact or a
// bound on the true score, and the node's static evaluation so it needn't
// be recomputed on a revisit.
//
// Entries come in clusters of four, one 64-byte cache line: a position can
// live in any entry of its cluster, so a deep entry and a fresh shallow one
// needn't evict each other. Each entry records the generation (the search)
// that wrote it, and whether the position was on or near the principal
// variation (see Searcher::StackEntry::tt_pv). A position already in the
// cluster is updated in place,
// unless the stored entry is from this search and much deeper
// (params::tt_replace_margin) and the new score isn't exact. Otherwise the
// entry with the lowest depth - params::tt_age_weight * age (age: searches since it was
// written) makes room: stale entries first, then shallow ones.
//
// Threads read and write entries without locks. An entry is two 64-bit
// words, each read and written whole: the data, and the key XOR the data.
// If a read catches an entry half-written by another thread (one word old,
// one new), the key no longer checks out and the read is a miss.
class TranspositionTable {
public:
    enum Bound : std::uint8_t { NONE = 0, UPPER = 1, LOWER = 2, EXACT = 3 };

    // An entry as read from the table: a copy, unpacked.
    struct Entry {
        std::uint64_t key = 0;
        Move move{};
        std::int16_t score = 0;
        std::int16_t eval = 0;  // static eval, or Searcher::NO_EVAL in check
        std::int8_t depth = 0;
        std::uint8_t gen_bound = 0;  // generation << 3 | pv << 2 | bound

        Bound bound() const { return static_cast<Bound>(gen_bound & 3); }
        int generation() const { return gen_bound >> 3; }
        bool is_pv() const { return gen_bound & 4; }
    };

    static constexpr int CLUSTER_SIZE = 4;

    explicit TranspositionTable(std::size_t megabytes = 64) { resize(megabytes); }
    void resize(std::size_t megabytes);
    void clear();
    // Call at the start of each search: entries written before become stale.
    void new_search() { generation_ = (generation_ + 1) & 31; }
    // Permille of the first 1000 entries written in the current search
    // (UCI "hashfull").
    int hashfull() const;

    // Starts fetching a cluster into cache: call as soon as a key is known,
    // so the memory access overlaps with other work before the probe.
    void prefetch(std::uint64_t key) const { __builtin_prefetch(&clusters_[key & mask_]); }

    // Copies the position's entry into `out`, if the table has one.
    bool probe(std::uint64_t key, Entry& out) const {
        // The key is checked on the raw words; only a hit is unpacked.
        for (const Slot& s : clusters_[key & mask_].slot) {
            const std::uint64_t d = s.data.load(std::memory_order_relaxed);
            if ((s.check.load(std::memory_order_relaxed) ^ d) == key && (d >> 56 & 3) != NONE) {
                out = Slot::unpack(key, d);
                return true;
            }
        }
        return false;
    }
    void store(std::uint64_t key, Move move, int score, int eval, int depth, Bound bound,
               bool pv);

private:
    // An entry as stored: move 16 bits, score 16, eval 16, depth 8 and
    // gen_bound 8 in `data`; `check` is the key XOR data. Relaxed atomics:
    // plain loads and stores on x86, but defined behaviour between threads.
    struct Slot {
        std::atomic<std::uint64_t> check{0}, data{0};

        static Entry unpack(std::uint64_t key, std::uint64_t d) {
            Entry e;
            e.key = key;
            e.move.data = static_cast<std::uint16_t>(d);
            e.score = static_cast<std::int16_t>(d >> 16);
            e.eval = static_cast<std::int16_t>(d >> 32);
            e.depth = static_cast<std::int8_t>(d >> 48);
            e.gen_bound = static_cast<std::uint8_t>(d >> 56);
            return e;
        }
        Entry read() const {
            const std::uint64_t d = data.load(std::memory_order_relaxed);
            return unpack(check.load(std::memory_order_relaxed) ^ d, d);
        }
        void write(const Entry& e) {
            const std::uint64_t d =
                std::uint64_t{e.move.data} |
                std::uint64_t{static_cast<std::uint16_t>(e.score)} << 16 |
                std::uint64_t{static_cast<std::uint16_t>(e.eval)} << 32 |
                std::uint64_t{static_cast<std::uint8_t>(e.depth)} << 48 |
                std::uint64_t{e.gen_bound} << 56;
            data.store(d, std::memory_order_relaxed);
            check.store(e.key ^ d, std::memory_order_relaxed);
        }
    };
    struct alignas(64) Cluster {
        Slot slot[CLUSTER_SIZE];
    };
    static_assert(sizeof(Cluster) == 64);

    std::unique_ptr<Cluster[]> clusters_;
    std::size_t size_ = 0;  // clusters
    std::size_t mask_ = 0;
    int generation_ = 0;  // 5 bits, wrapping
};

class ContLog;  // cont_log.h
class TreeSampler;  // tree_sample.h
// Continuation logging is compiled in only with -DCONT_LOG=ON (CMake).
#ifdef CONT_LOG
inline constexpr bool CONT_LOG_BUILD = true;
#else
inline constexpr bool CONT_LOG_BUILD = false;
#endif
struct ContNode;

// A node picked for label logging: the position, and what the search knew
// about it when it got there. None of the context can be recovered from
// the FEN afterwards. Scores are from the side to move's point of view.
struct LabelSample {
    std::string fen;
    int depth;               // depth the node was about to be searched to
    int ply;                 // distance from the root
    int static_eval;         // as the search used it: corrected
    int correction;          // static_eval minus the raw evaluation
    int search_eval;         // what pruning decided with: static_eval, or the
                             // hash score when its bound makes it sharper
    int alpha, beta;         // the node's window in the tree
    bool improving;
    prune::NodeType node;
    bool under_null;         // a null move lies between the root and here
    bool iir;                // searched one ply shallower than `depth` (internal
                             // iterative reduction, after node-level pruning)
    // The hash entry the node found, if any (one that cut it off returned
    // before sampling): much of why one position costs little in one tree
    // and a lot in another.
    bool tt_hit = false;
    int tt_depth = -1;       // -1 without an entry
    int tt_bound = 0;        // TT::Bound
    // Set when the node returns, by whichever path (SubtreeMeter, search.cpp):
    std::uint64_t tree_nodes = 0;  // its subtree's nodes, itself included
    bool reached_moves = false;    // got past node-level pruning to the move loop
};

class Searcher {
public:
    static constexpr int INF = 32000;
    static constexpr int MATE = 30000;
    static constexpr int MAX_PLY = 128;
    static constexpr int MATE_BOUND = MATE - MAX_PLY;  // scores beyond this are mates
    // A tablebase win, less the distance from the root: below every mate,
    // so a mate the search finds still counts for more.
    static constexpr int TB_WIN = MATE_BOUND - 1;
    // Scores beyond this are decisive: mates or tablebase wins. Evals stay
    // inside it.
    static constexpr int DECISIVE_BOUND = TB_WIN - MAX_PLY;
    static constexpr int NO_EVAL = 32001;              // "no static eval" (in check)

    // With its own hash table.
    explicit Searcher(std::size_t hash_mb = 64)
        : own_tt_(std::make_unique<TranspositionTable>(hash_mb)), tt_(*own_tt_) {}
    // Sharing `tt` with other searchers: one per thread of a parallel search
    // (see threads.h).
    explicit Searcher(TranspositionTable& tt) : tt_(tt) {}

    void resize_hash(std::size_t megabytes) { tt_.resize(megabytes); }
    // Clears the hash table and everything learned (histories, corrections).
    void new_game();
    // A root move the coming searches leave out (Move{}: none), for
    // gendata's castlelong; with no other legal move, the result may be it.
    void avoid_at_root(Move m) { avoid_root_ = m; }
    // Clears what this searcher learned, but not the hash table.
    void clear_history();

    // For tests: every move the move picker hands out at the root of `pos`,
    // in order, given a hash move and two killers (any of which may be
    // illegal, as a hash collision could supply, or Move{}).
    std::vector<Move> picker_order(const Position& pos, Move tt_move, Move killer1, Move killer2);

    // Iterative-deepening search. `history` holds the Zobrist keys of every
    // position of the game before `pos` (oldest first), for repetition
    // detection. With `verbose`, prints UCI "info" lines per iteration.
    // Returns Move{} only if `pos` has no legal moves.
    Move search(Position& pos, const SearchLimits& limits,
                const std::vector<std::uint64_t>& history, bool verbose = true);

    // Parallel search (threads.h). The main searcher starts each search
    // with begin_search, which search() otherwise does itself: a new hash
    // table generation and the root's tablebase ranking. Each helper then
    // takes the root from it (adopt_root) and searches the same position
    // until stopped, sharing the hash table; the main searcher alone keeps
    // time and reports, counting the helpers' nodes (set_helpers).
    // `limits` decide whether there is time to rank a tablebase root by DTZ
    // (see tm_tb_dtz_min_ms).
    void begin_search(const Position& pos, const SearchLimits& limits);
    void adopt_root(const Searcher& main);
    void set_helpers(std::vector<const Searcher*> helpers) { helpers_ = std::move(helpers); }

    // Asks a running search, from another thread, to stop as soon as it
    // can; search() then returns the best move found so far. The request
    // stays set until clear_stop(), which the caller does before starting
    // a search (not search() itself: a stop sent just after "go" must not
    // be lost).
    void request_stop() { stop_request_.store(true, std::memory_order_relaxed); }
    void clear_stop() { stop_request_.store(false, std::memory_order_relaxed); }
    bool stop_requested() const { return stop_request_.load(std::memory_order_relaxed); }

    // Pondering (SearchLimits::ponder): "ponderhit", from another thread,
    // turns the running search into a timed one. The flag stays set until
    // clear_ponderhit(), which the caller does before starting a search.
    void ponderhit() { ponderhit_.store(true, std::memory_order_relaxed); }
    void clear_ponderhit() { ponderhit_.store(false, std::memory_order_relaxed); }
    bool ponderhit_received() const { return ponderhit_.load(std::memory_order_relaxed); }
    // The reply the hash table expects to `best` from `pos` (the position
    // just searched), for "bestmove ... ponder ..."; Move{} if none.
    Move ponder_move(Position& pos, Move best);

    std::uint64_t nodes() const { return nodes_; }
    // The last search was cut short by its exact node limit (SearchLimits::
    // nodes), in the middle of an iteration.
    bool node_limit_hit() const { return node_limit_ && nodes_ >= node_limit_; }
    // The last search's completed iterations, in order: depth, nodes searched
    // by the end of it (from the start of the search), score, the deepest
    // ply it reached and its best move.
    struct Iteration {
        int depth;
        std::uint64_t nodes;
        int score;
        int seldepth;
        Move best{};
    };
    const std::vector<Iteration>& iterations() const { return iterations_; }
    // Tablebase probes that found the position, in the last search.
    std::uint64_t tb_hits() const { return tb_hits_; }
    // The tables' verdict on the last search's root, when they ranked it:
    // TB_WIN, -TB_WIN or 0 for the side to move; NO_EVAL otherwise.
    int root_tablebase_score() const { return root_tb_score_; }
    // The verdict root_tablebase_score() would give after a search of `pos`
    // with these limits, without searching (data generation's cheaper
    // budget for positions the export drops: a won or lost tablebase root
    // scores as decisive).
    int tablebase_verdict(const Position& pos, const SearchLimits& limits);
    // A "go nodes" search at data generation's limits when the frame options
    // ask for it (params::frame_soft_nodes; frame L2): sets the soft limit,
    // softscale, and the drop budget for a root in check or a tablebase root
    // won or lost. Leaves limits alone otherwise.
    void apply_frame_limits(const Position& pos, SearchLimits& limits);
    // Both counts as last published for other threads to read: at least
    // every 2048 nodes during a search, and exactly at its end.
    std::uint64_t published_nodes() const { return published_nodes_.load(std::memory_order_relaxed); }
    std::uint64_t published_tb_hits() const {
        return published_tb_hits_.load(std::memory_order_relaxed);
    }
    // Score of the last completed iteration of the last search.
    int last_score() const { return last_score_; }
    // Data generation. root_eval: the static eval a search of `pos` would
    // start from at its root (correction history and fifty-move scaling
    // applied) and the correction in it; NO_EVAL for both in check.
    // quiescence: a full-window quiescence search of `pos` (`history` as for
    // search()), on the current hash table and histories; it resets nodes().
    // With `plain`, the hash neither cuts nor sharpens a stand-pat (the
    // corrected static eval as it is): what the captures alone make of the
    // position, the quiet test of datatext qsquiet=. Without it, after a
    // search of `pos` the stand-pat takes that search's score.
    struct RootEval {
        int static_eval, correction;
    };
    RootEval root_eval(const Position& pos);
    int quiescence(Position& pos, const std::vector<std::uint64_t>& history, bool plain = false);
    // For tests: nodes spent under root move `m` in the last search, over
    // all its iterations, and under all root moves together.
    std::uint64_t root_move_nodes(Move m) const {
        return root_move_nodes_[static_cast<std::size_t>(m.from() * 64 + m.to())];
    }
    std::uint64_t root_nodes_total() const { return root_nodes_total_; }

    // Pruning statistics, accumulated over searches until cleared.
    const prune::Stats& stats() const { return stats_; }
    void clear_stats() { stats_.clear(); }
    // Verify 1 in `rate` pruning decisions by searching anyway (0 = off).
    // For diagnostics only: verification searches share the hash table,
    // killers and history with the real search, so they change its moves,
    // scores and node counts, and the statistics gathered alongside them
    // describe that altered search.
    void set_verify_rate(int rate) {
        verify_rate_ = rate > 0 ? static_cast<std::uint64_t>(rate) : 0;
    }

    // Label logging: while `sink` is set, a main-search node at depth d
    // (not in check, not the root) is appended to it with probability
    // 1 / rate_by_depth[d]. A rate of 0, or a depth past the end of the
    // list, means never.
    void set_label_sampling(std::vector<LabelSample>* sink,
                            std::vector<std::uint64_t> rate_by_depth, std::uint64_t seed) {
        label_sink_ = sink;
        label_rates_ = std::move(rate_by_depth);
        rng_ = seed | 1;
    }

    // Continuation logging (cont_log.h): sampled nodes record their moves
    // and outcomes in `log`; null turns it off.
    void set_cont_log(ContLog* log) { cont_log_ = log; }
    // Tree sampling (tree_sample.h): positions where the search reads its static evaluation, written by
    // `sampler`; null turns it off. Research builds only (CONT_LOG), as the continuation log.
    void set_tree_sample(TreeSampler* sampler) { tree_sample_ = sampler; }

private:
    using Clock = std::chrono::steady_clock;

    // Per-ply search state. static_eval is recorded at every main-search
    // node (NO_EVAL in check) for pruning decisions to build on.
    struct StackEntry {
        int static_eval = NO_EVAL;
        Move move{};         // move being searched from this ply
        Piece piece = NO_PIECE;  // the piece making `move`; NO_PIECE for a null move
        bool quiet = false;      // `move` is a quiet move (no capture or promotion)
        int history = 0;         // quiet_history() of `move` when chosen (0 unless quiet)
        Move killers[2]{};   // quiet moves that recently caused cutoffs here
        bool no_prune = false;  // next entry is a verification re-search: don't prune
        // Set when this node's null move failed low: the opponent's best
        // reply to the pass (from the null position's hash entry, Move{} if
        // it has none) and the null search's fail-soft score. threat_score
        // is NO_EVAL when no null move failed low here. Late move reductions
        // go easier on moves that address the threat.
        Move threat{};
        int threat_score = NO_EVAL;
        // Set around a singular-extension search of this node: the move to
        // leave out. Such a search is only a probe of the other moves, so it
        // neither cuts on nor writes to the hash table and doesn't prune.
        Move excluded{};
        // Singular extensions on the path from the root to this node, and
        // how many of them were double.
        int extensions = 0;
        int doubles = 0;
        // This node is on the principal variation, or its hash entry says it
        // was when it was stored (the entry's PV bit). Such nodes are
        // reduced less: they are where the score is decided.
        bool tt_pv = false;
        // `move`'s position in this node's move order (dropped quiet moves
        // counted), and whether the child search now running for it is
        // provisional: a reduced or null-window test that is searched again
        // if the child fails low. For the prior-countermove bonus.
        int move_index = 0;
        bool provisional = false;
        // This node's children that failed high in their own move loops, so
        // far (params::lmr_child_cutoffs).
        int child_cutoffs = 0;
        // The LMR reduction of the child search now running for `move` (0 when
        // it runs at full depth): for hindsight depth (params::hindsight_red)
        // and continuation logging.
        int reduction = 0;
        int logged_reduction() const { return reduction; }
    };

    // Continuation history: how well a quiet move (by its piece and
    // destination) has done in reply to an earlier move (also by piece and
    // destination), one ply back (countermove history) or two (follow-up
    // history). One table serves both. Same units and update rule as
    // history_. On the heap: 12 * 64 * 12 * 64 entries.
    using ContTable = std::array<std::int16_t, 12 * 64>;  // indexed by the reply's piece * 64 + to
    std::vector<ContTable> cont_hist_ = std::vector<ContTable>(12 * 64, ContTable{});
    // The table for replies to the move made `back` plies before `ply`, or
    // nullptr if there is none (the root, or a null move).
    ContTable* cont_table(int ply, int back);
    static std::size_t piece_to(Piece p, int to) {
        return static_cast<std::size_t>(p) * 64 + static_cast<std::size_t>(to);
    }

    // Capture history: how well a capture has done, by moving piece,
    // destination and captured piece type. Same units and update rule as
    // history_.
    std::vector<std::int16_t> capt_hist_ = std::vector<std::int16_t>(12 * 64 * 6, 0);
    // Pawn history: quiet history by pawn structure (a slot of the pawn
    // key), moving piece and destination; same units and update rule as
    // history_. It joins quiet_history() weighted by params::pawn_hist_weight.
    static constexpr std::size_t PAWN_HIST_SIZE = 1024;
    std::vector<std::int16_t> pawn_hist_ =
        std::vector<std::int16_t>(PAWN_HIST_SIZE * 12 * 64, 0);
    static std::size_t pawn_hist_index(const Position& pos, Move m) {
        return (pos.pawn_key() & (PAWN_HIST_SIZE - 1)) * 12 * 64 +
               piece_to(pos.piece_at(m.from()), m.to());
    }
    static std::size_t capture_index(const Position& pos, Move m) {
        const Piece victim = m.is_en_passant() ? WP : pos.piece_at(m.to());
        return piece_to(pos.piece_at(m.from()), m.to()) * 6 + static_cast<std::size_t>(type_of(victim));
    }

    std::unique_ptr<TranspositionTable> own_tt_;  // none when shared
    TranspositionTable& tt_;
    Evaluator eval_;
    std::array<StackEntry, MAX_PLY + 1> stack_{};
    // Quiet-move history, [side to move][from][to]. This and the other
    // history tables hold 16-bit values (see gravity()): half the cache
    // footprint of ints.
    std::array<std::array<std::array<std::int16_t, 64>, 64>, 2> history_{};
    // Pawn correction history (see params.h): per side to move and pawn key
    // slot, the average of searched score minus raw static eval, in units
    // of 1/CORR_GRAIN centipawn. On the heap, being too big for a stack
    // that already holds a Searcher or two.
    static constexpr std::size_t CORR_SIZE = 16384;
    static constexpr int CORR_GRAIN = 256;
    std::vector<int> pawn_corr_ = std::vector<int>(2 * CORR_SIZE, 0);
    // Non-pawn correction history, the same per side to move and each
    // colour's non-pawn key: [side to move][colour][slot].
    std::vector<int> nonpawn_corr_ = std::vector<int>(2 * 2 * CORR_SIZE, 0);
    // Continuation correction: [side to move][piece * 64 + destination] of
    // the move that led to the position.
    std::vector<int> cont_corr_ = std::vector<int>(2 * 12 * 64, 0);
    std::vector<std::uint64_t> keys_;  // game history + current search path
    std::uint64_t nodes_ = 0;
    std::uint64_t node_limit_ = 0;     // 0 = none
    std::vector<Iteration> iterations_;  // see iterations()
    bool stopped_ = false;
    std::atomic<bool> stop_request_{false};  // see request_stop()
    Clock::time_point start_;
    long long soft_ms_ = 0, hard_ms_ = 0;  // 0 = unlimited
    // The soft limit counts from limit_base_ms_ (0, the start, except after
    // a ponderhit); the hard limit is from the start, being checked often.
    long long limit_base_ms_ = 0;
    // While pondering, the time limits wait here for the ponderhit, when
    // check_ponderhit (on the search thread) applies them, counted from
    // that moment.
    std::atomic<bool> ponderhit_{false};
    bool pondering_ = false;
    long long ponder_soft_ms_ = 0, ponder_hard_ms_ = 0;
    void check_ponderhit() {
        if (pondering_ && ponderhit_received()) {
            pondering_ = false;
            limit_base_ms_ = elapsed_ms();
            soft_ms_ = ponder_soft_ms_;
            hard_ms_ = limit_base_ms_ + ponder_hard_ms_;
        }
    }
    Move root_best_{};
    int root_best_score_ = 0;  // search score of root_best_
    int last_score_ = 0;
    int root_depth_ = 0;  // depth of the current iteration
    // Nodes spent under each root move (by from and to square) and under all
    // of them, over the iterations of this search: the time management's
    // node share (params::tm_nodes_base).
    std::array<std::uint64_t, 64 * 64> root_move_nodes_{};
    std::uint64_t root_nodes_total_ = 0;
    int seldepth_ = 0;    // deepest ply reached in it (UCI "seldepth")
    // First index of keys_ after the most recent null move on the search
    // path. Repetitions are only looked for from there on: a position
    // "repeated" across a null move was never reached over the board.
    std::size_t null_barrier_ = 0;

    // Tablebases (see tablebase.h). tb_pieces_: the search probes positions
    // with at most this many pieces (0: no probing). When the root itself is
    // in the tables, root_moves_ holds the moves that keep its result, the
    // only ones searched, and root_tb_score_ what the tables say it is
    // worth (NO_EVAL otherwise), reported instead of a search score that
    // isn't a mate.
    int tb_pieces_ = 0;
    std::uint64_t tb_hits_ = 0;
    std::vector<Move> root_moves_;
    int root_tb_score_ = NO_EVAL;
    Move avoid_root_{};
    // Ranks the root in the tables and sets the three above.
    void setup_tablebases(const Position& pos, bool use_dtz);
    // begin_search (or adopt_root) was done for the coming search.
    bool root_ready_ = false;
    std::vector<const Searcher*> helpers_;  // see set_helpers
    std::atomic<std::uint64_t> published_nodes_{0}, published_tb_hits_{0};
    void publish_counts() {
        published_nodes_.store(nodes_, std::memory_order_relaxed);
        published_tb_hits_.store(tb_hits_, std::memory_order_relaxed);
    }

    prune::Stats stats_;
    std::uint64_t verify_rate_ = 0;  // 0 = no verification
    int verify_depth_ = 0;           // > 0 inside a verification search
    std::vector<LabelSample>* label_sink_ = nullptr;
    ContNode* open_cont_node();
    void close_cont_node(ContNode* n, int best, int bound);
    void drop_cont_nodes(std::size_t keep);
    [[gnu::noinline]] void log_cont_node(ContNode& cn, const Position& pos, int depth, int ply,
                                         int node_type, int alpha, int beta, int raw_eval,
                                         int static_eval, int eval, int corr, bool improving,
                                         bool in_check, int tt_depth, int tt_bound, int tt_score,
                                         bool tt_move);
    [[gnu::noinline]] void log_cont_row(ContNode& cn, const Position& pos, Move m,
                                        const prune::Context& ctx, int stage, const int* pruned,
                                        bool skip_quiets, bool any_check, bool hash_move,
                                        int alpha, int best, int searched, Move threat);
    std::vector<std::uint64_t> label_rates_;  // by depth; see set_label_sampling
    std::uint64_t rng_ = 0x9e3779b97f4a7c15ULL;

    int negamax(Position& pos, int depth, int alpha, int beta, int ply, bool cut_node);
    int qsearch(Position& pos, int alpha, int beta, int ply);
    bool improving(int ply, int eval) const;
    int& correction_entry(const Position& pos) {
        return pawn_corr_[static_cast<std::size_t>(pos.side_to_move()) * CORR_SIZE +
                          (pos.pawn_key() & (CORR_SIZE - 1))];
    }
    int& nonpawn_entry(const Position& pos, Color c) {
        return nonpawn_corr_[(static_cast<std::size_t>(pos.side_to_move()) * 2 +
                              static_cast<std::size_t>(c)) * CORR_SIZE +
                             (pos.nonpawn_key(c) & (CORR_SIZE - 1))];
    }
    // The raw static eval plus `corr`, the position's correction(), then
    // fifty-move scaling.
    int corrected_eval(const Position& pos, int raw, int corr);
    // Continuation correction's entry for the node at `ply` (see params.h),
    // or nullptr: off, at the root, after a null move, or with ply < 0
    // (quiescence, whose moves aren't on the stack).
    int* cont_corr_entry(const Position& pos, int ply);
    // The correction for the position's pawns and, weighted by
    // params::corr_nonpawn_weight, its other pieces; and at a main-search
    // node (ply >= 0), weighted by params::corr_cont_weight, the move that
    // led to it.
    int correction(const Position& pos, int ply);
    void update_correction(const Position& pos, int ply, int depth, int diff);

    // Statistics are not counted inside verification searches, which
    // would count the same decisions twice.
    std::uint64_t next_random();
    bool sample_verify();
    void count_tried(prune::Heuristic h) { if (!verify_depth_) ++stats_.c[h].tried; }
    void count_fired(prune::Heuristic h) { if (!verify_depth_) ++stats_.c[h].fired; }
    void count_feature(prune::Feature f, bool acted) {
        if (verify_depth_) return;
        ++stats_.f[f].tried;
        stats_.f[f].fired += acted;
    }
    void record_verified(prune::Heuristic h, bool wrong);
    int verify_node(Position& pos, int depth, int alpha, int beta, int ply, bool cut_node);
    void verify_pruned_move(Position& pos, Move m, prune::Heuristic h, int depth, int alpha,
                            int ply);

    bool should_stop();
    bool is_draw(const Position& pos) const;
    // One reversible move from here reaches a position earlier on the
    // current search path (see negamax).
    bool repetition_in_reach(const Position& pos, int ply) const;
    long long elapsed_ms() const;
    void set_time_limits(const Position& pos, const SearchLimits& limits);
    // Hands out a node's moves one at a time, best first, generating them
    // in stages (see search.cpp) so that a node which cuts off early never
    // pays for the moves it didn't need.
    class MovePicker;

    // `cont` holds the continuation tables for this node (either may be
    // nullptr); quiet moves are scored by history plus both.
    void score_moves(const Position& pos, const MoveList& moves, int count, Move tt_move,
                     int ply, const ContTable* const cont[2],
                     std::array<int, 256>& scores) const;
    int quiet_history(const Position& pos, Move m, const ContTable* const cont[2]) const;
    void update_quiet_stats(const Position& pos, Move best, const Move* tried, int n_tried,
                            int depth, int ply);
    // A quiet hash move that cut a node off: its history bonus at `pct`
    // percent (params::tt_cut_hist_pct).
    void reward_hash_move(const Position& pos, Move m, int depth, int ply, int pct);
    // After a cutoff: reward `best` if it is a capture (Move{} otherwise),
    // and penalize the captures searched before it.
    void update_capture_stats(const Position& pos, Move best, const Move* tried, int n_tried,
                              int depth);
    std::vector<Move> principal_variation(Position& pos, int max_len);
    // The line reported for a root move: `first` (the move this searcher
    // will play), then the hash moves after it. Not the hash move at the
    // root itself: with threads, a helper may have stored its own there.
    std::vector<Move> reported_line(Position& pos, Move first, int max_len);
    // ProbCut's move loop (see negamax): the score of the first good
    // capture that beats pc_beta in a reduced search, and the move in
    // cut_move; below pc_beta if none does.
    int probcut(Position& pos, int depth, int pc_beta, int eval, Move tt_move, int ply,
                bool cut_node, Move& cut_move);

    // Continuation logging (cont_log.h), last so that it shifts no other
    // member: the installed log (null = off), and the sampled nodes still
    // open, innermost last. A node that returns early (time, multi-cut) is
    // left behind and freed when an ancestor closes or the search ends: no
    // cleanup on every exit of negamax.
    ContLog* cont_log_ = nullptr;
    bool qs_plain_ = false;  // quiescence(..., plain): no hash cutoffs or sharpened stand-pats
    TreeSampler* tree_sample_ = nullptr;
    std::vector<ContNode*> cont_open_;
};
