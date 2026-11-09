#include "search.h"
#include "cont_log.h"
#include "tree_sample.h"
#include "eval.h"
#include "movegen.h"
#include "params.h"
#include "see.h"
#include "tablebase.h"
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <bit>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

// ---------------------------------------------------------------------------
// Transposition table

void TranspositionTable::resize(std::size_t megabytes) {
    const std::size_t want = (megabytes << 20) / sizeof(Cluster);
    size_ = want ? std::bit_floor(want) : 1;
    clusters_ = std::make_unique<Cluster[]>(size_);  // all entries empty
    mask_ = size_ - 1;
}

void TranspositionTable::clear() {
    for (std::size_t i = 0; i < size_; ++i)
        for (Slot& s : clusters_[i].slot) s.write(Entry{});
}

void TranspositionTable::store(std::uint64_t key, Move move, int score, int eval, int depth,
                               Bound bound, bool pv) {
    Cluster& c = clusters_[key & mask_];
    Entry entries[CLUSTER_SIZE];
    for (int i = 0; i < CLUSTER_SIZE; ++i) entries[i] = c.slot[i].read();

    // The position's own entry if it has one; otherwise the least valuable
    // entry: an empty one, or the one with the lowest depth - weight * age
    // (params::tt_age_weight).
    int slot = -1;
    for (int i = 0; i < CLUSTER_SIZE; ++i)
        if (entries[i].key == key && entries[i].bound() != NONE) { slot = i; break; }
    if (slot < 0) {
        int worst = INT_MAX;
        for (int i = 0; i < CLUSTER_SIZE; ++i) {
            const Entry& e = entries[i];
            if (e.bound() == NONE) { slot = i; break; }
            const int age = (generation_ - e.generation()) & 31;
            const int value = e.depth - params::tt_age_weight * age;
            if (value < worst) { worst = value; slot = i; }
        }
    }
    Entry& e = entries[slot];

    // Keep a much deeper entry of the same position from this search: it
    // cost far more to produce. An exact score is worth having regardless.
    if (e.key == key && e.bound() != NONE && e.generation() == generation_ &&
        depth + params::tt_replace_margin + (pv ? params::tt_pv_replace_bonus : 0) < e.depth &&
        bound != EXACT)
        return;
    // Keep the old best move if this search of the same position found none
    // (every move failed low): it is still the best ordering guess we have.
    if (move == Move{} && e.key == key) move = e.move;
    e.key = key;
    e.move = move;
    e.score = static_cast<std::int16_t>(score);
    e.eval = static_cast<std::int16_t>(eval);
    e.depth = static_cast<std::int8_t>(std::clamp(depth, -128, 127));  // depth can exceed 127 with extensions
    e.gen_bound = static_cast<std::uint8_t>(generation_ << 3 | (pv ? 4 : 0) | bound);
    c.slot[slot].write(e);
}

int TranspositionTable::hashfull() const {
    int used = 0, seen = 0;
    for (std::size_t c = 0; c < size_ && seen < 1000; ++c)
        for (const Slot& s : clusters_[c].slot) {
            const Entry e = s.read();
            used += e.bound() != NONE && e.generation() == generation_;
            ++seen;
        }
    return seen ? used * 1000 / seen : 0;
}

// ---------------------------------------------------------------------------
// Helpers

namespace {
// Keys for the halfmove clock's buckets (params::fifty_hash_width), from
// xorshift64* with a seed of their own. Bucket 0 is 0: below
// params::fifty_hash_start the key is the position's own.
const std::array<std::uint64_t, 16> FIFTY_KEYS = [] {
    std::array<std::uint64_t, 16> keys{};
    std::uint64_t s = 0xc3a5c85c97cb3127ULL;
    for (std::size_t i = 1; i < keys.size(); ++i) {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        keys[i] = s * 0x2545F4914F6CDD1DULL;
    }
    return keys;
}();
}  // namespace

std::uint64_t tt_key(const Position& pos) {
    const int clock = pos.halfmove_clock();
    if (!params::fifty_hash_width || clock < params::fifty_hash_start) return pos.key();
    const int bucket =
        std::min(1 + (clock - params::fifty_hash_start) / params::fifty_hash_width, 15);
    return pos.key() ^ FIFTY_KEYS[static_cast<std::size_t>(bucket)];
}

namespace {
using TT = TranspositionTable;

// Measures a label sample's subtree for the cost head (labels.h): started
// at sampling, it records the nodes spent below the node when the node
// returns, whichever return that is (node-level prunings and hash cuts
// included). Idle (one branch at return) at every node not sampled: bench
// nps measured the same with and without it (3,229,565 against 3,230,199,
// ten runs each, spread 0.6%), so it isn't compiled out.
class SubtreeMeter {
public:
    SubtreeMeter() = default;
    SubtreeMeter(const SubtreeMeter&) = delete;
    SubtreeMeter& operator=(const SubtreeMeter&) = delete;
    // The sample just appended to `sink`; `nodes` already counts this node.
    void start(std::vector<LabelSample>* sink, const std::uint64_t* nodes) {
        sink_ = sink;
        index_ = sink->size() - 1;  // an index: deeper samples may reallocate
        nodes_ = nodes;
        start_ = *nodes;
    }
    void reached_moves() const {
        if (sink_) (*sink_)[index_].reached_moves = true;
    }
    ~SubtreeMeter() {
        if (sink_) (*sink_)[index_].tree_nodes = *nodes_ - start_ + 1;
    }

private:
    std::vector<LabelSample>* sink_ = nullptr;
    std::size_t index_ = 0;
    const std::uint64_t* nodes_ = nullptr;
    std::uint64_t start_ = 0;
};

// Mate and tablebase scores are stored relative to the node rather than the
// root, so an entry means the same thing wherever in the tree it is found
// again.
int score_to_tt(int score, int ply) {
    if (score > Searcher::DECISIVE_BOUND) return score + ply;
    if (score < -Searcher::DECISIVE_BOUND) return score - ply;
    return score;
}
int score_from_tt(int score, int ply) {
    if (score > Searcher::DECISIVE_BOUND) return score - ply;
    if (score < -Searcher::DECISIVE_BOUND) return score + ply;
    return score;
}

// Fail-firm: a fail high resting on a static eval or a shallow search
// overstates the node, so it is pulled toward beta, keeping `pct` percent
// of the excess (see params::ff_rfp_pct). Still at least beta: a valid
// lower bound. Decisive scores, and a decisive beta, are left alone.
int fail_firm(int score, int beta, int pct) {
    if (pct >= 100 || std::abs(score) >= Searcher::DECISIVE_BOUND ||
        std::abs(beta) >= Searcher::DECISIVE_BOUND)
        return score;
    return beta + (score - beta) * pct / 100;
}

// A hash score is a better estimate of the node than its static eval when
// its bound says the true score lies beyond the eval on the score's side:
// a lower bound above the eval, an upper bound below it, or an exact score.
// Pruning decisions then use it. Mate scores are left out.
int sharpen_eval(int eval, int tt_score, TT::Bound bound) {
    if (bound == TT::NONE || std::abs(tt_score) >= Searcher::DECISIVE_BOUND) return eval;
    const TT::Bound side = tt_score > eval ? TT::LOWER : TT::UPPER;
    return (bound & side) ? tt_score : eval;
}

// Move ordering: hash move, captures that don't lose material (by SEE) by
// victim value and then capture history, queen promotions, the two killer
// moves, the remaining quiet moves by history (butterfly and
// continuation), then losing captures, and underpromotions last.
constexpr int ORDER_VALUE[6] = {100, 320, 330, 500, 900, 0};
constexpr int TT_MOVE_SCORE = 10'000'000;
constexpr int CAPTURE_BASE = 1'000'000;
constexpr int QUEEN_PROMO_BONUS = 900'000;
constexpr int KILLER_1_SCORE = 800'000;
constexpr int KILLER_2_SCORE = 790'000;
constexpr int BAD_CAPTURE_BASE = -500'000;
constexpr int UNDERPROMO_SCORE = -1'000'000;

bool is_quiet(const Position& pos, Move m) {
    return pos.piece_at(m.to()) == NO_PIECE && !m.is_en_passant() &&
           m.promotion() == PieceType::None;
}

bool is_capture(const Position& pos, Move m) {
    return pos.piece_at(m.to()) != NO_PIECE || m.is_en_passant();
}

// Selection sort, one step at a time: brings the best remaining move to
// position i. Cheaper than a full sort because most nodes cut off early.
void pick_next(MoveList& moves, std::array<int, 256>& scores, int i, int count) {
    auto best = static_cast<std::size_t>(i);
    for (auto j = best + 1; j < static_cast<std::size_t>(count); ++j)
        if (scores[j] > scores[best]) best = j;
    std::swap(moves[static_cast<std::size_t>(i)], moves[best]);
    std::swap(scores[static_cast<std::size_t>(i)], scores[best]);
}

std::string score_string(int score) {
    if (score > Searcher::MATE_BOUND) return "mate " + std::to_string((Searcher::MATE - score + 1) / 2);
    if (score < -Searcher::MATE_BOUND) return "mate " + std::to_string(-(Searcher::MATE + score) / 2);
    return "cp " + std::to_string(score);
}
}

// ---------------------------------------------------------------------------
// Staged move picking
//
// The same order as scoring every legal move up front (see ORDER_VALUE
// above), produced in stages: the hash move before anything is generated;
// then captures and promotions are generated and scored, and handed out
// while they win material or promote to a queen; then the killers; then
// quiet moves are generated and handed out by history; then the losing
// captures and underpromotions left over from the capture stage. The hash
// move and killers are tried unseen, so each is checked for legality
// first, and skipped when it turns up again in its own stage.

class Searcher::MovePicker {
public:
    MovePicker(const Searcher& s, const Position& pos, Move tt_move, int ply,
               const ContTable* const cont[2], const ContTable* cont4)
        : s_(s), pos_(pos), tt_move_(tt_move), ply_(ply), cont_{cont[0], cont[1]}, cont4_(cont4) {
        const StackEntry& e = s.stack_[static_cast<std::size_t>(ply)];
        killers_[0] = e.killers[0];
        killers_[1] = e.killers[1];
    }

    // The move last handed out is a capture already known not to lose
    // material (its SEE >= 0 was computed when scoring it), so tests
    // against a threshold at or below 0 needn't run SEE again.
    bool good_capture() const { return good_capture_; }
    // The stage the last move came from (continuation logging).
    int stage() const { return static_cast<int>(stage_); }

    // Late move pruning has fired: the remaining quiet moves are dropped,
    // except those that give check (by `checks`, which must outlive the
    // picker). The survivors keep their order.
    void drop_quiets(const Position::CheckInfo& checks) {
        drop_quiets_ = true;
        checks_ = &checks;
        if (stage_ == Stage::Quiets) filter_quiets();
    }
    // How many dropped quiet moves would have come before the move last
    // handed out, had they been handed out too.
    int dropped_before() const { return dropped_before_; }

    // The next move, or Move{} when there are no more.
    Move next() {
        good_capture_ = false;
        while (true) {
            switch (stage_) {
            case Stage::HashMove:
                stage_ = Stage::GenCaptures;
                if (tt_move_ != Move{} && pos_.is_legal(tt_move_)) return tt_move_;
                break;

            case Stage::GenCaptures:
                generate_captures(pos_, caps_, n_caps_);
                s_.score_moves(pos_, caps_, n_caps_, Move{}, ply_, cont_, cap_scores_);
                stage_ = Stage::GoodCaptures;
                break;

            case Stage::GoodCaptures:
                while (cap_i_ < n_caps_) {
                    pick_next(caps_, cap_scores_, cap_i_, n_caps_);
                    // Everything from here on loses material or underpromotes.
                    if (cap_scores_[static_cast<std::size_t>(cap_i_)] <= KILLER_1_SCORE) break;
                    const Move m = caps_[static_cast<std::size_t>(cap_i_++)];
                    if (m != tt_move_) {
                        // Scored above CAPTURE_BASE / 2: a capture that passed
                        // SEE (losing captures, even promoting, stay far below;
                        // a quiet queen promotion scores high without SEE).
                        good_capture_ =
                            is_capture(pos_, m) &&
                            cap_scores_[static_cast<std::size_t>(cap_i_ - 1)] >= CAPTURE_BASE / 2;
                        return m;
                    }
                }
                stage_ = Stage::Killer1;
                break;

            case Stage::Killer1:
            case Stage::Killer2: {
                const Move k = killers_[stage_ == Stage::Killer1 ? 0 : 1];
                stage_ = stage_ == Stage::Killer1 ? Stage::Killer2 : Stage::GenQuiets;
                if (k != Move{} && k != tt_move_ && is_quiet(pos_, k) && pos_.is_legal(k)) return k;
                break;
            }

            case Stage::GenQuiets: {
                generate_quiets(pos_, quiets_, n_quiets_);
                // Quiet moves giving direct check sort earlier
                // (params::quiet_check_bonus).
                const Position::CheckInfo ci =
                    params::quiet_check_bonus ? pos_.check_info() : Position::CheckInfo{};
                for (int i = 0; i < n_quiets_; ++i) {
                    const Move m = quiets_[static_cast<std::size_t>(i)];
                    int score = s_.quiet_history(pos_, m, cont_);
                    if (cont4_)
                        score += (*cont4_)[piece_to(pos_.piece_at(m.from()), m.to())] *
                                 params::cont_hist4_weight / 256;
                    if (ci.checking_from[type_of(pos_.piece_at(m.from()))] & bit(m.to()))
                        score += params::quiet_check_bonus;
                    quiet_scores_[static_cast<std::size_t>(i)] = score;
                }
                if (drop_quiets_) filter_quiets();
                stage_ = Stage::Quiets;
                break;
            }

            case Stage::Quiets:
                while (quiet_i_ < n_quiets_) {
                    pick_next(quiets_, quiet_scores_, quiet_i_, n_quiets_);
                    const Move m = quiets_[static_cast<std::size_t>(quiet_i_++)];
                    if (m != tt_move_ && m != killers_[0] && m != killers_[1]) {
                        if (n_dropped_)
                            dropped_before_ = count_dropped_above(
                                quiet_scores_[static_cast<std::size_t>(quiet_i_ - 1)]);
                        return m;
                    }
                }
                stage_ = Stage::BadCaptures;
                break;

            case Stage::BadCaptures:
                while (cap_i_ < n_caps_) {
                    pick_next(caps_, cap_scores_, cap_i_, n_caps_);
                    const Move m = caps_[static_cast<std::size_t>(cap_i_++)];
                    if (m != tt_move_) {
                        dropped_before_ = n_dropped_;
                        return m;
                    }
                }
                stage_ = Stage::Done;
                break;

            case Stage::Done:
                return Move{};
            }
        }
    }

private:
    enum class Stage {
        HashMove, GenCaptures, GoodCaptures, Killer1, Killer2, GenQuiets, Quiets,
        BadCaptures, Done
    };

    // Keeps, of the quiet moves not yet handed out, the checks (if checks_
    // is set), in order. The hash move and killers go without counting:
    // they would have been skipped anyway. The scores of the dropped moves
    // are kept for dropped_before().
    void filter_quiets() {
        int kept = quiet_i_;
        for (int i = quiet_i_; i < n_quiets_; ++i) {
            const auto from = static_cast<std::size_t>(i);
            const Move m = quiets_[from];
            if (m == tt_move_ || m == killers_[0] || m == killers_[1]) continue;
            if (pos_.gives_check(m, *checks_)) {
                const auto to = static_cast<std::size_t>(kept++);
                quiets_[to] = m;
                quiet_scores_[to] = quiet_scores_[from];
            } else {
                dropped_scores_[static_cast<std::size_t>(n_dropped_++)] = quiet_scores_[from];
            }
        }
        n_quiets_ = kept;
    }

    // Dropped moves that outscore `score` (equals are taken to sort after).
    int count_dropped_above(int score) const {
        int n = 0;
        for (int d = 0; d < n_dropped_; ++d) n += dropped_scores_[static_cast<std::size_t>(d)] > score;
        return n;
    }

    const Searcher& s_;
    const Position& pos_;
    Move tt_move_;
    int ply_;
    const ContTable* cont_[2];
    const ContTable* cont4_;  // replies to the move 4 plies back; ordering only
    Move killers_[2];
    bool good_capture_ = false;
    bool drop_quiets_ = false;
    const Position::CheckInfo* checks_ = nullptr;
    std::array<int, 256> dropped_scores_;
    int n_dropped_ = 0;
    int dropped_before_ = 0;
    Stage stage_ = Stage::HashMove;

    MoveList caps_;
    std::array<int, 256> cap_scores_;
    int n_caps_ = 0, cap_i_ = 0;
    MoveList quiets_;
    std::array<int, 256> quiet_scores_;
    int n_quiets_ = 0, quiet_i_ = 0;
};

// ---------------------------------------------------------------------------
// Search

void Searcher::new_game() {
    tt_.clear();
    clear_history();
}

void Searcher::clear_history() {
    for (auto& side : history_)
        for (auto& from : side) from.fill(0);
    std::fill(pawn_corr_.begin(), pawn_corr_.end(), 0);
    std::fill(nonpawn_corr_.begin(), nonpawn_corr_.end(), 0);
    std::fill(cont_corr_.begin(), cont_corr_.end(), 0);
    for (auto& t : cont_hist_) t.fill(0);
    std::fill(capt_hist_.begin(), capt_hist_.end(), 0);
    std::fill(pawn_hist_.begin(), pawn_hist_.end(), 0);
}

int* Searcher::cont_corr_entry(const Position& pos, int ply) {
    if (!params::corr_cont_weight || ply < 1) return nullptr;
    const StackEntry& prev = stack_[static_cast<std::size_t>(ply - 1)];
    if (prev.piece == NO_PIECE || prev.move == Move{}) return nullptr;
    return &cont_corr_[static_cast<std::size_t>(pos.side_to_move()) * 12 * 64 +
                       piece_to(prev.piece, prev.move.to())];
}

int Searcher::correction(const Position& pos, int ply) {
    const long long nonpawn =
        nonpawn_entry(pos, Color::White) + nonpawn_entry(pos, Color::Black);
    int corr = correction_entry(pos) / CORR_GRAIN +
               static_cast<int>(nonpawn * params::corr_nonpawn_weight / (256LL * CORR_GRAIN));
    if (const int* cont = cont_corr_entry(pos, ply))
        corr += static_cast<int>(static_cast<long long>(*cont) * params::corr_cont_weight /
                                 (256LL * CORR_GRAIN));
    return corr;
}

int Searcher::corrected_eval(const Position& pos, int raw, int corr) {
    int corrected = raw + corr;
    // Fifty-move scaling (see params.h): here rather than in the raw eval,
    // which the hash table keeps and whose key doesn't include the clock.
    if (params::fifty_scale)
        corrected = corrected * std::max(params::fifty_scale - pos.halfmove_clock(), 0) /
                    params::fifty_scale;
    return std::clamp(corrected, -DECISIVE_BOUND, DECISIVE_BOUND);
}

// Moves the corrections toward `diff` (searched score minus raw static
// eval), faster for deeper, more trustworthy searches. The tables add up,
// so each moves by the error that remains after all of them (not by its
// own distance to `diff`: then every table would learn the whole gap, and
// their sum would overshoot it). With the non-pawn weight at 0 this is the
// pawn table's plain running average.
void Searcher::update_correction(const Position& pos, int ply, int depth, int diff) {
    const int weight = std::min(depth + 1, params::corr_weight_max);
    const int limit = params::corr_limit * CORR_GRAIN;
    const long long target = static_cast<long long>(diff) * CORR_GRAIN;
    int& pawn = correction_entry(pos);
    int& white = nonpawn_entry(pos, Color::White);
    int& black = nonpawn_entry(pos, Color::Black);
    int* const cont = cont_corr_entry(pos, ply);
    const long long total =
        pawn + (static_cast<long long>(white) + black) * params::corr_nonpawn_weight / 256 +
        (cont ? static_cast<long long>(*cont) * params::corr_cont_weight / 256 : 0);
    const long long error = target - total;
    auto update = [&](int& entry) {
        const long long updated = (static_cast<long long>(entry) * 256 + error * weight) / 256;
        entry = static_cast<int>(std::clamp<long long>(updated, -limit, limit));
    };
    update(pawn);
    update(white);
    update(black);
    if (cont) update(*cont);
}

Searcher::ContTable* Searcher::cont_table(int ply, int back) {
    if (ply < back) return nullptr;
    const StackEntry& e = stack_[static_cast<std::size_t>(ply - back)];
    if (e.piece == NO_PIECE || e.move == Move{}) return nullptr;
    return &cont_hist_[piece_to(e.piece, e.move.to())];
}

// Butterfly history, the continuation histories and pawn history: the
// quiet move's standing in general, in reply to the last two moves, and in
// this pawn structure.
int Searcher::quiet_history(const Position& pos, Move m, const ContTable* const cont[2]) const {
    int s = history_[static_cast<std::size_t>(pos.side_to_move())]
                    [static_cast<std::size_t>(m.from())][static_cast<std::size_t>(m.to())];
    const std::size_t reply = piece_to(pos.piece_at(m.from()), m.to());
    for (int k = 0; k < 2; ++k)
        if (cont[k]) s += (*cont[k])[reply];
    if (params::pawn_hist_weight)
        s += pawn_hist_[pawn_hist_index(pos, m)] * params::pawn_hist_weight / 256;
    return s;
}

void Searcher::score_moves(const Position& pos, const MoveList& moves, int count, Move tt_move,
                           int ply, const ContTable* const cont[2],
                           std::array<int, 256>& scores) const {
    const Move* killers = stack_[static_cast<std::size_t>(ply)].killers;
    for (int i = 0; i < count; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        const Move m = moves[idx];
        if (m == tt_move) { scores[idx] = TT_MOVE_SCORE; continue; }

        const PieceType promo = m.promotion();
        const Piece victim = m.is_en_passant() ? WP : pos.piece_at(m.to());
        int s;
        if (victim != NO_PIECE) {
            // Least valuable attacker only breaks ties, until capture
            // history has something to say.
            const int attacker = type_of(pos.piece_at(m.from()));
            s = (see_at_least(pos, m, 0) ? CAPTURE_BASE : BAD_CAPTURE_BASE) +
                8 * ORDER_VALUE[type_of(victim)] +
                capt_hist_[capture_index(pos, m)] * params::capt_hist_mult / 1024 - attacker;
        } else if (promo != PieceType::None) {
            s = 0;
        } else if (m == killers[0]) {
            s = KILLER_1_SCORE;
        } else if (m == killers[1]) {
            s = KILLER_2_SCORE;
        } else {
            s = quiet_history(pos, m, cont);
        }
        if (promo == PieceType::Queen) s += QUEEN_PROMO_BONUS;
        else if (promo != PieceType::None) s = UNDERPROMO_SCORE;
        scores[idx] = s;
    }
}

// History gravity: moves a history score by `delta` while pulling it toward
// +-hist_max, so the tables never saturate. With |delta| <= hist_max and
// |h| <= hist_max before, |h| <= hist_max after (rounding included), so
// the clamp keeps every history score within 16 bits whatever the
// parameters (it doesn't act at the tuned values).
static void gravity(std::int16_t& h, int delta) {
    delta = std::clamp(delta, -params::hist_max, params::hist_max);
    h = static_cast<std::int16_t>(h + delta - h * std::abs(delta) / params::hist_max);
}

// A quiet move caused a beta cutoff: make it a killer at this ply, reward
// it in the history and continuation history tables, and penalize the
// quiet moves tried before it. The updates use history
// gravity.
void Searcher::update_quiet_stats(const Position& pos, Move best, const Move* tried, int n_tried,
                                  int depth, int ply) {
    Move* killers = stack_[static_cast<std::size_t>(ply)].killers;
    if (killers[0] != best) {
        killers[1] = killers[0];
        killers[0] = best;
    }

    auto& hist = history_[static_cast<std::size_t>(pos.side_to_move())];
    ContTable* cont[3] = {cont_table(ply, 1), cont_table(ply, 2),
                          params::cont_hist4_weight ? cont_table(ply, 4) : nullptr};
    const int bonus = std::min(params::hist_bonus_mult * depth * depth, params::hist_bonus_max);
    const int malus = std::min(params::hist_malus_mult * depth * depth, params::hist_malus_max);
    auto apply = [&](Move m, int delta) {
        gravity(hist[static_cast<std::size_t>(m.from())][static_cast<std::size_t>(m.to())], delta);
        const std::size_t reply = piece_to(pos.piece_at(m.from()), m.to());
        for (ContTable* t : cont)
            if (t) gravity((*t)[reply], delta);
        if (params::pawn_hist_weight) gravity(pawn_hist_[pawn_hist_index(pos, m)], delta);
    };
    apply(best, bonus);
    for (int i = 0; i < n_tried; ++i) apply(tried[i], -malus);
}

void Searcher::reward_hash_move(const Position& pos, Move m, int depth, int ply, int pct) {
    const int bonus =
        std::min(params::hist_bonus_mult * depth * depth, params::hist_bonus_max) * pct / 100;
    gravity(history_[static_cast<std::size_t>(pos.side_to_move())]
                    [static_cast<std::size_t>(m.from())][static_cast<std::size_t>(m.to())],
            bonus);
    const std::size_t reply = piece_to(pos.piece_at(m.from()), m.to());
    ContTable* cont[3] = {cont_table(ply, 1), cont_table(ply, 2),
                          params::cont_hist4_weight ? cont_table(ply, 4) : nullptr};
    for (ContTable* t : cont)
        if (t) gravity((*t)[reply], bonus);
    if (params::pawn_hist_weight) gravity(pawn_hist_[pawn_hist_index(pos, m)], bonus);
}

void Searcher::update_capture_stats(const Position& pos, Move best, const Move* tried,
                                    int n_tried, int depth) {
    const int bonus = std::min(params::capt_bonus_mult * depth * depth, params::capt_bonus_max);
    const int malus = std::min(params::capt_malus_mult * depth * depth, params::capt_malus_max);
    auto apply = [&](Move m, int delta) { gravity(capt_hist_[capture_index(pos, m)], delta); };
    if (best != Move{}) apply(best, bonus);
    for (int i = 0; i < n_tried; ++i) apply(tried[i], -malus);
}


long long Searcher::elapsed_ms() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start_).count();
}

// Counts the node and reports whether the search must stop. The node limit
// is checked exactly, so "go nodes N" is reproducible; the clock and a
// stop request from another thread only every 2048 nodes, since reading
// the clock is comparatively slow.
bool Searcher::should_stop() {
    if (stopped_) return true;  // unwinding: calls made after a stop aren't nodes
    ++nodes_;
    if (node_limit_ && nodes_ >= node_limit_) {
        stopped_ = true;
    } else if ((nodes_ & 2047) == 0) {
        publish_counts();
        check_ponderhit();
        if ((hard_ms_ && elapsed_ms() >= hard_ms_) || stop_requested()) stopped_ = true;
    }
    return stopped_;
}

void Searcher::set_time_limits(const Position& pos, const SearchLimits& limits) {
    soft_ms_ = hard_ms_ = 0;
    if (limits.infinite) return;

    const long long overhead = params::move_overhead;  // GUI and OS latency, ms
    if (limits.movetime > 0) {
        hard_ms_ = soft_ms_ = std::max(1LL, limits.movetime - overhead);
        return;
    }

    const int side = static_cast<int>(pos.side_to_move());
    if (limits.time[side] < 0) return;  // no clock given: depth-limited or unbounded

    const long long left = limits.time[side];
    const long long inc = limits.inc[side];
    const long long mtg = limits.movestogo > 0 ? limits.movestogo : params::tm_moves_to_go;

    // Target time for this move, and a hard ceiling for when an iteration
    // runs long. A new iteration is only started while under a share of
    // the target (half by default), since each one takes roughly two to
    // three times the last. The shares are percentages (see params.h).
    const long long target = left / mtg + inc * params::tm_inc_pct / 100;
    const long long ceiling = std::max(1LL, left - overhead);
    // At least 1 ms: with a millisecond or less left and no increment the
    // share of the clock rounds to 0, and 0 would mean no limit at all.
    const long long cap =
        std::max(1LL, std::min(ceiling, left * params::tm_hard_cap_pct / 100 + inc));
    hard_ms_ = std::clamp(target * params::tm_hard_pct / 100, 1LL, cap);
    soft_ms_ = std::min(target * params::tm_soft_pct / 100, hard_ms_);
    if (soft_ms_ <= 0) soft_ms_ = 1;
}

bool Searcher::is_draw(const Position& pos) const {
    // Fifty moves without a capture or pawn move: a draw, unless the move
    // that reached the limit gave mate: checkmate takes precedence.
    if (pos.halfmove_clock() >= 100) {
        if (!pos.in_check(pos.side_to_move())) return true;
        MoveList moves;
        int count = 0;
        generate_legal(pos, moves, count);
        return count > 0;
    }

    // keys_.back() is the position one ply ago. The same side is to move
    // every second ply, and nothing before the last capture or pawn move
    // (the halfmove clock) can repeat. A single repetition inside the tree
    // is scored as a draw, the usual simplification: if repeating is good
    // for one side, the opponent would repeat again.
    const std::uint64_t key = pos.key();
    const int n = static_cast<int>(keys_.size());
    const int limit = std::min(pos.halfmove_clock(), n - static_cast<int>(null_barrier_));
    for (int d = 4; d <= limit; d += 2)
        if (keys_[static_cast<std::size_t>(n - d)] == key) return true;
    return false;
}

bool Searcher::repetition_in_reach(const Position& pos, int ply) const {
    // Positions an odd number of plies back have the other side to move, so
    // one move by either side can join them to this one. Only those on the
    // search path (i < ply) count: before the root, a single repetition
    // isn't a draw, and repetition flags aren't kept.
    const int n = static_cast<int>(keys_.size());
    const int end = std::min(pos.halfmove_clock(), n - static_cast<int>(null_barrier_));
    for (int i = 3; i <= end && i < ply; i += 2)
        if (pos.reversible_move_to(keys_[static_cast<std::size_t>(n - i)])) return true;
    return false;
}

void Searcher::setup_tablebases(const Position& pos, bool use_dtz) {
    tb_pieces_ = std::min(tb::largest(), tb::probe_limit);
    root_moves_.clear();
    root_tb_score_ = NO_EVAL;
    // Not for an illegal root (the side that just moved in check), which
    // Fathom doesn't survive; a GUI might send one.
    if (!tb_pieces_ || pos.castling_rights() ||
        std::popcount(pos.occupied()) > tb_pieces_ ||
        pos.in_check(opposite(pos.side_to_move())))
        return;

    tb::RootRanking ranking;
    // A mated or stalemated root ranks successfully with no moves.
    if (!tb::rank_root(pos, ranking, use_dtz) || ranking.moves.empty()) return;
    int best = ranking.moves.front().rank;
    for (const auto& rm : ranking.moves) best = std::max(best, rm.rank);
    for (const auto& rm : ranking.moves)
        if (rm.rank == best) root_moves_.push_back(rm.move);
    root_tb_score_ = ranking.win ? TB_WIN : ranking.loss ? -TB_WIN : 0;
    // With the root ranked by distance to zeroing, every move left keeps the
    // result, and a winning one makes the fastest progress: the search
    // needn't probe, and without tablebase scores in the way it plays for
    // the fastest mate it can see. WDL alone gives no such guarantee, so
    // then the search keeps probing to find its way into won positions.
    if (ranking.dtz) tb_pieces_ = 0;
}

// Time to rank a tablebase root by DTZ? Not with little on the clock (nor
// with none: node-limited searches rank by WDL).
static bool rank_by_dtz(const Position& pos, const SearchLimits& limits) {
    const long long budget = limits.infinite        ? -1
                             : limits.movetime > 0  ? limits.movetime
                                                    : limits.time[static_cast<int>(pos.side_to_move())];
    return budget < 0 || budget >= params::tm_tb_dtz_min_ms;
}

void Searcher::begin_search(const Position& pos, const SearchLimits& limits) {
    // The clock runs from here: ranking a tablebase root by DTZ can take
    // tens of milliseconds (cold tables, a busy disk), and with threads
    // this runs before search() does.
    start_ = Clock::now();
    prune::init();  // the tables it fills are shared by all threads
    tt_.new_search();
    if (params::pure_search) {  // the evaluation alone: no tablebase verdicts
        tb_pieces_ = 0;
        root_moves_.clear();
        root_tb_score_ = NO_EVAL;
    } else {
        setup_tablebases(pos, rank_by_dtz(pos, limits));
    }
    if (!limits.search_moves.empty()) {
        std::vector<Move> keep;
        for (const Move m : limits.search_moves)
            if (root_moves_.empty() || std::find(root_moves_.begin(), root_moves_.end(), m) != root_moves_.end())
                keep.push_back(m);
        if (!keep.empty()) root_moves_ = std::move(keep);
    }
    root_ready_ = true;
}

void Searcher::apply_frame_limits(const Position& pos, SearchLimits& limits) {
    if (!params::frame_soft_nodes || !limits.nodes || limits.soft_nodes) return;
    limits.soft_nodes = static_cast<std::uint64_t>(params::frame_soft_nodes);
    limits.soft_scale = params::frame_soft_scale != 0;
    if (!params::frame_drop_soft_nodes) return;
    const bool in_check = pos.in_check(pos.side_to_move());
    const int verdict = in_check ? NO_EVAL : tablebase_verdict(pos, limits);
    if (in_check || (verdict != NO_EVAL && verdict != 0)) {
        limits.soft_nodes = static_cast<std::uint64_t>(params::frame_drop_soft_nodes);
        if (params::frame_drop_hard_nodes) limits.nodes = static_cast<std::uint64_t>(params::frame_drop_hard_nodes);
    }
}

int Searcher::tablebase_verdict(const Position& pos, const SearchLimits& limits) {
    if (params::pure_search) return NO_EVAL;
    // The same ranking begin_search makes; it leaves the root's tablebase
    // state as begin_search would, and the search's own begin_search
    // makes it again.
    setup_tablebases(pos, rank_by_dtz(pos, limits));
    return root_tb_score_;
}

void Searcher::adopt_root(const Searcher& main) {
    start_ = main.start_;
    tb_pieces_ = main.tb_pieces_;
    root_moves_ = main.root_moves_;
    root_tb_score_ = main.root_tb_score_;
    root_ready_ = true;
    // Before this helper's thread starts: the main searcher's first report
    // mustn't count the last search's nodes.
    nodes_ = tb_hits_ = 0;
    publish_counts();
}

// ---------------------------------------------------------------------------
// Pruning support

std::uint64_t Searcher::next_random() {
    // xorshift64*
    rng_ ^= rng_ >> 12;
    rng_ ^= rng_ << 25;
    rng_ ^= rng_ >> 27;
    return rng_ * 0x2545F4914F6CDD1DULL;
}

bool Searcher::sample_verify() {
    return verify_rate_ && !verify_depth_ && next_random() % verify_rate_ == 0;
}

void Searcher::record_verified(prune::Heuristic h, bool wrong) {
    if (stopped_) return;  // the verification search was cut short
    ++stats_.c[h].verified;
    if (wrong) ++stats_.c[h].wrong;
}

ContNode* Searcher::open_cont_node() {
    cont_open_.push_back(new ContNode());
    return cont_open_.back();
}

// Writes a finished sampled node and frees it, with any nodes opened after
// it that returned early (they are dead: their search is over).
void Searcher::close_cont_node(ContNode* n, int best, int bound) {
    const auto it = std::find(cont_open_.begin(), cont_open_.end(), n);
    drop_cont_nodes(static_cast<std::size_t>(it - cont_open_.begin()) + 1);
    cont_log_->write_node(*n, best, bound, nodes_ - n->nodes_start);
    drop_cont_nodes(cont_open_.size() - 1);
}

void Searcher::drop_cont_nodes(std::size_t keep) {
    while (cont_open_.size() > keep) {
        delete cont_open_.back();
        cont_open_.pop_back();
    }
}

// Continuation logging's records (cont_log.h), out of line: the search
// only tests a pointer while logging is off.
void Searcher::log_cont_node(ContNode& cn, const Position& pos, int depth, int ply, int node_type,
                             int alpha, int beta, int raw_eval, int static_eval, int eval, int corr,
                             bool improving, bool in_check, int tt_depth, int tt_bound,
                             int tt_score, bool tt_move) {
    const StackEntry& ss = stack_[static_cast<std::size_t>(ply)];
    const StackEntry& prev = stack_[static_cast<std::size_t>(ply - 1)];
    cn.fen = pos.fen();
    cn.depth = depth;
    cn.ply = ply;
    cn.root_depth = root_depth_;
    cn.seldepth = seldepth_;
    cn.parent_reduction = prev.logged_reduction();
    cn.parent_provisional = prev.provisional;
    cn.extensions = ss.extensions;
    cn.node_type = node_type;
    cn.alpha = alpha;
    cn.beta = beta;
    cn.raw_eval = raw_eval;
    cn.static_eval = static_eval;
    cn.eval = eval;
    cn.correction = corr;
    cn.improving = improving;
    cn.in_check = in_check;
    cn.tt_hit = tt_depth >= 0;
    cn.tt_depth = tt_depth;
    cn.tt_bound = tt_bound;
    cn.tt_score = tt_score;
    cn.tt_move = tt_move;
    const Color us = pos.side_to_move();
    for (int pt = 0; pt < 5; ++pt) {
        const int own = std::popcount(pos.pieces(static_cast<Piece>(static_cast<int>(us) * 6 + pt)));
        const int theirs =
            std::popcount(pos.pieces(static_cast<Piece>(static_cast<int>(opposite(us)) * 6 + pt)));
        cn.material += (own - theirs) * SEE_VALUE[pt];
        cn.phase += (own + theirs) * (pt == 4 ? 4 : pt == 3 ? 2 : pt >= 1 ? 1 : 0);
    }
    cn.halfmove = pos.halfmove_clock();
    cn.parent_index = prev.move_index;
    cn.parent_quiet = prev.quiet && prev.move != Move{};
    cn.parent_history = prev.history;
    MoveList legal;
    int count = 0;
    generate_legal(pos, legal, count);
    cn.legal = count;
    for (int k = 0; k < count; ++k)
        (is_capture(pos, legal[static_cast<std::size_t>(k)]) ? cn.legal_captures : cn.legal_quiets)++;
    cn.nodes_start = nodes_;
}

void Searcher::log_cont_row(ContNode& cn, const Position& pos, Move m, const prune::Context& ctx,
                            int stage, const int* pruned, bool skip_quiets, bool any_check,
                            bool hash_move, int alpha, int best, int searched, Move threat) {
    // The move as it starts, with what the loop has seen so far.
    if (cn.rows.empty()) cn.threat = threat != Move{};
    ContRow r;
    r.index = ctx.move_index;
    r.searched_before = searched;
    r.stage = stage;
    std::copy(pruned, pruned + 6, std::begin(r.pruned));
    r.lmp_skipping = skip_quiets;
    r.piece = pos.piece_at(m.from());
    r.from = m.from();
    r.to = m.to();
    r.promotion = static_cast<int>(m.promotion());
    r.capture = is_capture(pos, m);
    r.gives_check = any_check;
    r.killer = ctx.killer;
    r.hash_move = hash_move;
    r.see = see_value(pos, m);
    r.history = ctx.history;
    r.capture_history = ctx.capture_history;
    r.lmr_reduction = prune::lmr_reduction(ctx);
    r.alpha = alpha;
    r.best_before = best == -INF ? ContLog::NO_SCORE : best;
    int first = ContLog::NO_SCORE, second = ContLog::NO_SCORE;
    for (const ContRow& p : cn.rows) {
        if (p.score > first) {
            second = first;
            first = p.score;
        } else if (p.score > second) {
            second = p.score;
        }
        r.near_best += p.score >= best - 25;
        r.raised_before += p.raised;
    }
    r.second_before = second;
    r.researches_before = cn.researches;
    r.nodes_before = nodes_ - cn.nodes_start;
    cn.rows.push_back(r);
}

// The node's static eval is better than the last one for the same side to
// move, two plies up (four, if that node was in check). Positions getting
// better are less likely to fail low, so the search prunes their moves
// less and cuts their nodes more readily. With nothing to compare against,
// assume improving.
bool Searcher::improving(int ply, int eval) const {
    for (int back = 2; back <= 4 && back <= ply; back += 2) {
        const int prev = stack_[static_cast<std::size_t>(ply - back)].static_eval;
        if (prev != NO_EVAL) return eval > prev;
    }
    return true;
}

// Searches a node again, with node-level pruning disabled, to check a
// decision that pruned it. Same ply, same window.
int Searcher::verify_node(Position& pos, int depth, int alpha, int beta, int ply,
                          bool cut_node) {
    ++verify_depth_;
    stack_[static_cast<std::size_t>(ply)].no_prune = true;
    const int v = negamax(pos, depth, alpha, beta, ply, cut_node);
    --verify_depth_;
    return v;
}

// Searches a pruned move anyway, at full depth with a null window, and
// records whether it would have raised alpha.
void Searcher::verify_pruned_move(Position& pos, Move m, prune::Heuristic h, int depth,
                                  int alpha, int ply) {
    ++verify_depth_;
    StateInfo st;
    keys_.push_back(pos.key());
    stack_[static_cast<std::size_t>(ply)].move = m;
    stack_[static_cast<std::size_t>(ply)].piece = pos.piece_at(m.from());
    stack_[static_cast<std::size_t>(ply)].quiet = is_quiet(pos, m);
    stack_[static_cast<std::size_t>(ply)].history = 0;  // neutral for params::lmr_parent_hist
    stack_[static_cast<std::size_t>(ply + 1)].extensions = stack_[static_cast<std::size_t>(ply)].extensions;
    stack_[static_cast<std::size_t>(ply + 1)].doubles = stack_[static_cast<std::size_t>(ply)].doubles;
    pos.make_move(m, st);
    eval_.push(st.changes);
    const int score = -negamax(pos, depth - 1, -alpha - 1, -alpha, ply + 1, true);
    eval_.pop();
    pos.unmake_move(m, st);
    keys_.pop_back();
    --verify_depth_;
    record_verified(h, score > alpha);
}

// ---------------------------------------------------------------------------
// Alpha-beta

int Searcher::qsearch(Position& pos, int alpha, int beta, int ply) {
    if (should_stop()) return 0;
    seldepth_ = std::max(seldepth_, ply);
    if (ply >= MAX_PLY - 1) return eval_.evaluate(pos);

    const bool pv_node = beta - alpha > 1;
    const int alpha_orig = alpha;
    const bool pure = params::pure_search;  // the evaluation alone (params.h)

    // Hash table. Any entry is deep enough to cut with, quiescence being
    // depth 0; as in the main search, not at PV nodes (nor in a pure search,
    // where it only orders).
    Move tt_move{};
    int tt_eval = NO_EVAL;
    int tt_depth = -1;
    int tt_score = 0;
    TT::Bound tt_bound = TT::NONE;
    bool tt_pv = pv_node;
    if (TT::Entry e; tt_.probe(tt_key(pos), e)) {
        tt_move = e.move;
        tt_eval = e.eval;
        tt_depth = e.depth;
        tt_score = score_from_tt(e.score, ply);
        tt_bound = e.bound();
        tt_pv = tt_pv || e.is_pv();
        if (!pv_node && !pure && !qs_plain_) {
            if (tt_bound == TT::EXACT ||
                (tt_bound == TT::LOWER && tt_score >= beta) ||
                (tt_bound == TT::UPPER && tt_score <= alpha))
                return tt_score;
        }
    }
    // Store a result, unless the table already holds a deeper search of
    // this position: a quiescence result would only replace it with less.
    // A plain quiescence (a measurement) leaves the table as it found it.
    auto store = [&](int score, Move best_move, int raw_eval) {
        if (tt_depth > 0 || qs_plain_) return;
        const auto bound = score >= beta ? TT::LOWER : score > alpha_orig ? TT::EXACT : TT::UPPER;
        tt_.store(tt_key(pos), best_move, score_to_tt(score, ply), raw_eval, 0, bound, tt_pv);
    };

    const bool in_check = pos.in_check(pos.side_to_move());
    MoveList moves;
    int count = 0;
    int best;
    int raw_eval = NO_EVAL;

    if (in_check) {
        // No standing pat in check: every evasion is searched, so mates at
        // the horizon are seen.
        generate_legal(pos, moves, count);
        if (count == 0) return -MATE + ply;
        best = -INF;
    } else {
        // Stand pat: the side to move can usually do at least as well as
        // the static evaluation by declining every capture.
        raw_eval = tt_eval != NO_EVAL ? tt_eval : eval_.evaluate(pos);
        if (CONT_LOG_BUILD && tree_sample_) tree_sample_->maybe(pos, 'q', 0, ply, raw_eval);
        best = pure        ? raw_eval
               : qs_plain_ ? corrected_eval(pos, raw_eval, correction(pos, -1))
                           : sharpen_eval(corrected_eval(pos, raw_eval, correction(pos, -1)), tt_score, tt_bound);
        if (best >= beta) {
            const int firm = pure ? best : fail_firm(best, beta, params::ff_stand_pat_pct);
            count_feature(prune::FailFirm, firm != best);
            best = firm;
            store(best, Move{}, raw_eval);
            return best;
        }
        alpha = std::max(alpha, best);
        generate_captures(pos, moves, count);
    }

    std::array<int, 256> scores;
    // No continuation context: quiescence moves don't record theirs on the
    // stack, so the entries a ply or two up may belong to another line.
    const ContTable* const no_cont[2] = {nullptr, nullptr};
    score_moves(pos, moves, count, tt_move, ply, no_cont, scores);
    Move best_move{};

    prune::Context ctx;
    ctx.pos = &pos;
    ctx.ply = ply;
    ctx.static_eval = best;
    ctx.beta = beta;
    ctx.in_check = in_check;
    ctx.node = beta - alpha > 1 ? prune::NodeType::PV : prune::NodeType::All;
    // Built when delta pruning first needs it.
    Position::CheckInfo check_info;
    bool have_check_info = false;
    int quiet_evasions = 0;  // quiet evasions searched, in check

    for (int i = 0; i < count; ++i) {
        pick_next(moves, scores, i, count);
        const Move m = moves[static_cast<std::size_t>(i)];
        if (!in_check) {
            if (m.promotion() != PieceType::None && m.promotion() != PieceType::Queen)
                continue;  // underpromotions never matter for tactics outside check

            ctx.move = m;
            ctx.move_index = i;
            ctx.alpha = alpha;

            // Delta pruning: even winning the captured piece outright, plus a
            // margin for what else the capture might gain, leaves the score
            // below alpha. Not for promotions, nor for checks, which can win
            // more than the piece.
            if (!pure && m.promotion() == PieceType::None && std::abs(alpha) < DECISIVE_BOUND) {
                const Piece victim = m.is_en_passant() ? WP : pos.piece_at(m.to());
                const int optimistic =
                    ctx.static_eval + ORDER_VALUE[type_of(victim)] + params::qs_delta_margin;
                count_tried(prune::QsDelta);
                if (optimistic <= alpha) {
                    if (!have_check_info) {
                        check_info = pos.check_info();
                        have_check_info = true;
                    }
                    if (!pos.gives_check(m, check_info)) {
                        count_fired(prune::QsDelta);
                        if (sample_verify())
                            verify_pruned_move(pos, m, prune::QsDelta, 1, alpha, ply);
                        best = std::max(best, optimistic);
                        continue;
                    }
                }
            }

            // A capture that loses material in the exchange can't raise a
            // score that already stands pat.
            count_tried(prune::QsSee);
            // Scoring already ran SEE against 0 for captures other than the
            // hash move: reuse it where it settles the test either way.
            const int threshold = prune::qs_see_threshold(ctx);
            const bool scored = m != tt_move && is_capture(pos, m);
            const bool good = scores[static_cast<std::size_t>(i)] >= CAPTURE_BASE / 2;
            const bool passes = scored && good && threshold <= 0    ? true
                                : scored && !good && threshold >= 0 ? false
                                                                     : see_at_least(pos, m, threshold);
            if (!passes) {
                count_fired(prune::QsSee);
                if (sample_verify()) verify_pruned_move(pos, m, prune::QsSee, 1, alpha, ply);
                continue;
            }
        } else if (!pure && is_quiet(pos, m) && best > -DECISIVE_BOUND) {
            // In check, once a line that isn't mated is known and enough
            // quiet evasions have been searched, the rest are skipped: the
            // captures that evade come first, and the point here is only to
            // see mates at the horizon.
            count_tried(prune::QsEvasion);
            if (quiet_evasions >= params::qs_quiet_evasions) {
                count_fired(prune::QsEvasion);
                if (sample_verify()) verify_pruned_move(pos, m, prune::QsEvasion, 1, alpha, ply);
                continue;
            }
        }
        if (in_check && is_quiet(pos, m)) ++quiet_evasions;

        StateInfo st;
        pos.make_move(m, st);
        tt_.prefetch(tt_key(pos));
        eval_.push(st.changes);
        const int score = -qsearch(pos, -beta, -alpha, ply + 1);
        eval_.pop();
        pos.unmake_move(m, st);
        if (stopped_) return 0;

        if (score > best) {
            best = score;
            if (score > alpha) {
                alpha = score;
                best_move = m;
                if (alpha >= beta) break;
            }
        }
    }
    store(best, best_move, raw_eval);
    return best;
}

int Searcher::probcut(Position& pos, int depth, int pc_beta, int eval, Move tt_move, int ply,
                      bool cut_node, Move& cut_move) {
    // Captures (and queen promotions) whose static exchange alone makes up
    // the gap from the eval to pc_beta: the hash move first, then by the
    // value of the piece taken.
    MoveList moves;
    int count = 0;
    generate_captures(pos, moves, count);  // captures and promotions, legal
    std::array<int, 256> scores;
    int n = 0;
    for (int i = 0; i < count; ++i) {
        const Move m = moves[static_cast<std::size_t>(i)];
        if (!is_capture(pos, m) && m.promotion() != PieceType::Queen) continue;
        if (!see_at_least(pos, m, pc_beta - eval)) continue;
        const Piece victim = m.is_en_passant() ? WP : pos.piece_at(m.to());
        moves[static_cast<std::size_t>(n)] = m;
        scores[static_cast<std::size_t>(n)] =
            m == tt_move ? INT_MAX : victim == NO_PIECE ? 0 : ORDER_VALUE[type_of(victim)];
        ++n;
    }

    // Each is checked by quiescence first, which is cheap and usually
    // enough to refute it, then by a search params::probcut_reduction
    // plies shallower.
    StackEntry& ss = stack_[static_cast<std::size_t>(ply)];
    for (int i = 0; i < n; ++i) {
        pick_next(moves, scores, i, n);
        const Move m = moves[static_cast<std::size_t>(i)];
        StateInfo st;
        keys_.push_back(pos.key());
        ss.move = m;
        ss.piece = pos.piece_at(m.from());
        ss.quiet = false;  // captures and promotions only
        stack_[static_cast<std::size_t>(ply + 1)].extensions = ss.extensions;
        stack_[static_cast<std::size_t>(ply + 1)].doubles = ss.doubles;
        pos.make_move(m, st);
        tt_.prefetch(tt_key(pos));
        eval_.push(st.changes);
        int v = -qsearch(pos, -pc_beta, -pc_beta + 1, ply + 1);
        if (v >= pc_beta && !stopped_)
            v = -negamax(pos, depth - params::probcut_reduction, -pc_beta, -pc_beta + 1, ply + 1,
                         !cut_node);
        eval_.pop();
        pos.unmake_move(m, st);
        keys_.pop_back();
        if (stopped_) return 0;
        if (v >= pc_beta) {
            cut_move = m;
            return v;
        }
    }
    return -INF;
}

// `cut_node` is the expectation that a non-PV node fails high (a "cut"
// node) rather than low (an "all" node): the children of an all node are
// expected to be cut nodes, and the first child of a cut node an all node.
int Searcher::negamax(Position& pos, int depth, int alpha, int beta, int ply, bool cut_node) {
    if (should_stop()) return 0;
    seldepth_ = std::max(seldepth_, ply);

    const bool root = ply == 0;
    const bool pv_node = beta - alpha > 1;
    const bool pure = params::pure_search;  // the evaluation alone (params.h)
    StackEntry& ss = stack_[static_cast<std::size_t>(ply)];
    const bool no_prune = std::exchange(ss.no_prune, false);
    const Move excluded = ss.excluded;  // set by a singular-extension probe
    ss.threat = Move{};
    ss.threat_score = NO_EVAL;
    if (!root && is_draw(pos)) return 0;
    if (ply >= MAX_PLY - 1) return eval_.evaluate(pos);

    // Upcoming repetition: if one reversible move can bring back a position
    // from earlier in this line, a draw is at hand, so the node is worth at
    // least that to the side to move.
    if (!root && params::rep_in_reach && !pure && alpha < 0) {
        const bool in_reach = repetition_in_reach(pos, ply);
        count_feature(prune::RepInReach, in_reach);
        if (in_reach) {
            alpha = 0;
            if (alpha >= beta) return alpha;
        }
    }

    // Mate distance pruning: from here, the best conceivable result is
    // mating on the next move and the worst is being mated right now. If
    // the window lies outside that range, no search can get into it.
    if (!root) {
        alpha = std::max(alpha, -MATE + ply);
        beta = std::min(beta, MATE - ply - 1);
        if (alpha >= beta) return alpha;
    }

    // In check, quiescence searches every evasion, so a node in check can
    // safely drop into it. Checks are extended where they are given, and
    // only when they don't lose material (see CheckExtSee).
    const bool in_check = pos.in_check(pos.side_to_move());
    if (depth <= 0) return qsearch(pos, alpha, beta, ply);

    Move tt_move{};
    int tt_eval = NO_EVAL;
    int tt_score = 0;
    int tt_depth = -1;
    TT::Bound tt_bound = TT::NONE;
    bool tt_was_pv = false;
    count_feature(prune::FiftyKey,
                  params::fifty_hash_width && pos.halfmove_clock() >= params::fifty_hash_start);
    if (TT::Entry e; tt_.probe(tt_key(pos), e)) {
        tt_move = e.move;
        tt_eval = e.eval;
        tt_score = score_from_tt(e.score, ply);
        tt_depth = e.depth;
        tt_bound = e.bound();
        tt_was_pv = e.is_pv();
        // No cutoffs in PV nodes: they would cut the principal variation
        // short and hide how the score was reached. Nor in a probe that
        // leaves a move out: the entry's score counts that move.
        if (excluded != Move{}) {
            tt_score = 0;
            tt_bound = TT::NONE;
        } else if (!pv_node && !pure && e.depth >= depth) {
            if (tt_bound == TT::EXACT ||
                (tt_bound == TT::LOWER && tt_score >= beta) ||
                (tt_bound == TT::UPPER && tt_score <= alpha)) {
                // A quiet hash move that fails high again: rewarded as if
                // searched (params::tt_cut_hist_pct).
                const bool rewarded = params::tt_cut_hist_pct && tt_score >= beta &&
                                      tt_move != Move{} && pos.is_legal(tt_move) &&
                                      is_quiet(pos, tt_move);
                count_feature(prune::TtCutHist, rewarded);
                if (rewarded) reward_hash_move(pos, tt_move, depth, ply, params::tt_cut_hist_pct);
                return tt_score;
            }
        }
    }

    // Tablebases: right after a capture or pawn move (the tables assume a
    // fresh fifty-move count), with few enough pieces, the result is known.
    // It ends the search here when it settles the window; otherwise (a win
    // below beta at a PV node, say) the node is searched as usual.
    if (tb_pieces_ && !pure && !root && excluded == Move{} && pos.halfmove_clock() == 0 &&
        !pos.castling_rights()) {
        const int pieces = std::popcount(pos.occupied());
        if (pieces < tb_pieces_ || (pieces == tb_pieces_ && depth >= tb::probe_depth)) {
            const tb::Wdl wdl = tb::probe_wdl(pos);
            if (wdl != tb::Wdl::Failed) {
                ++tb_hits_;
                const int score = wdl == tb::Wdl::Win    ? TB_WIN - ply
                                  : wdl == tb::Wdl::Loss ? -TB_WIN + ply
                                                         : 0;
                const TT::Bound bound = wdl == tb::Wdl::Win    ? TT::LOWER
                                        : wdl == tb::Wdl::Loss ? TT::UPPER
                                                               : TT::EXACT;
                if (bound == TT::EXACT || (bound == TT::LOWER && score >= beta) ||
                    (bound == TT::UPPER && score <= alpha)) {
                    tt_.store(tt_key(pos), Move{}, score_to_tt(score, ply), NO_EVAL, depth, bound,
                              pv_node || tt_was_pv);
                    return score;
                }
            }
        }
    }

    // A singular-extension probe re-enters this ply: it keeps the flag of
    // the node it probes.
    if (excluded == Move{}) ss.tt_pv = pv_node || tt_was_pv;

    // The hash table keeps the raw eval, so the correction applied is always
    // the current one. static_eval (corrected) is the node's own estimate,
    // which `improving` and correction history compare against; `eval` may
    // be sharpened by the hash score, and is what pruning decides with.
    // A pure search uses the raw eval as it is (it only orders moves there).
    const int raw_eval = in_check ? NO_EVAL : tt_eval != NO_EVAL ? tt_eval : eval_.evaluate(pos);
    if (CONT_LOG_BUILD && tree_sample_ && !in_check && ply >= 1 && excluded == Move{})
        tree_sample_->maybe(pos, 'm', depth, ply, raw_eval);
    const int corr = in_check || pure ? 0 : correction(pos, ply);
    ss.static_eval = in_check ? NO_EVAL : pure ? raw_eval : corrected_eval(pos, raw_eval, corr);
    const int static_eval = ss.static_eval;
    const int eval = in_check ? NO_EVAL : pure ? static_eval : sharpen_eval(static_eval, tt_score, tt_bound);

    // Hindsight depth (see params.h): judge the parent's reduction of this
    // node by what its move did to the static eval.
    if (params::hindsight_red && !in_check && excluded == Move{} && ply >= 1) {
        const StackEntry& prev = stack_[static_cast<std::size_t>(ply - 1)];
        if (prev.reduction >= params::hindsight_red && prev.static_eval != NO_EVAL) {
            const int cost = static_eval + prev.static_eval;
            const int change = cost <= 0 ? 1 : cost > params::hindsight_margin && depth >= 2 ? -1 : 0;
            count_feature(prune::Hindsight, change != 0);
            depth += change;
        }
    }

    // Eval-change history: how much the opponent's last quiet move changed
    // the static eval, from its own side, is a hint about the move; it goes
    // into that side's butterfly history (params::eval_hist_mult). Only on a
    // first visit, or at low depth (a revisit would count it again).
    if (params::eval_hist_mult && !in_check && excluded == Move{} && ply >= 1 &&
        (tt_depth < 0 || depth <= params::eval_hist_max_depth)) {
        const StackEntry& prev = stack_[static_cast<std::size_t>(ply - 1)];
        if (prev.quiet && prev.piece != NO_PIECE && prev.move != Move{} &&
            prev.static_eval != NO_EVAL) {
            const int delta = -(static_eval + prev.static_eval);
            const int bonus = std::clamp(delta * params::eval_hist_mult / 16,
                                         -params::eval_hist_malus_max, params::eval_hist_bonus_max);
            gravity(history_[static_cast<std::size_t>(opposite(pos.side_to_move()))]
                            [static_cast<std::size_t>(prev.move.from())]
                            [static_cast<std::size_t>(prev.move.to())],
                    bonus);
            count_feature(prune::EvalHist, bonus != 0);
        }
    }

    prune::Context ctx;
    ctx.pos = &pos;
    ctx.depth = depth;
    ctx.ply = ply;
    ctx.static_eval = eval;
    ctx.correction = std::abs(corr);
    ctx.alpha = alpha;
    ctx.beta = beta;
    ctx.improving = !in_check && improving(ply, static_eval);
    ctx.tt_pv = ss.tt_pv;
    ctx.tt_capture = tt_move != Move{} && is_capture(pos, tt_move);
    if (ply >= 1) {
        const StackEntry& prev = stack_[static_cast<std::size_t>(ply - 1)];
        ctx.parent_quiet = prev.quiet && prev.move != Move{};
        ctx.parent_history = ctx.parent_quiet ? prev.history : 0;
    }
    ctx.in_check = in_check;
    ctx.node = pv_node ? prune::NodeType::PV : cut_node ? prune::NodeType::Cut
                                                        : prune::NodeType::All;

    // Internal iterative reduction (see params.h). Decided here, applied
    // after node-level pruning, which works with the full depth.
    const bool probing = excluded != Move{};  // a singular-extension probe
    const bool iir = !root && !probing && !pure && depth >= params::iir_min_depth && tt_move == Move{} &&
                     (pv_node || cut_node);
    if (!root && !probing && depth >= params::iir_min_depth) count_feature(prune::IIR, iir);

    // Continuation logging (cont_log.h): a sampled node records its moves.
    // Nodes that end before their move loop (node-level pruning, a hash or
    // tablebase cut) write nothing; they never ask the question.
    ContNode* cn = nullptr;
    if (CONT_LOG_BUILD && cont_log_ && !root && !verify_depth_ && !probing &&
        cont_log_->sample(depth)) {
        cn = open_cont_node();
        log_cont_node(*cn, pos, depth, ply, pv_node ? 0 : cut_node ? 1 : 2, alpha, beta, raw_eval,
                      static_eval, eval, corr, ctx.improving, in_check, tt_depth, tt_bound,
                      tt_score, tt_move != Move{});
    }

    SubtreeMeter meter;
    if (label_sink_ && !in_check && !root && !verify_depth_ && !probing &&
        static_cast<std::size_t>(depth) < label_rates_.size()) {
        const std::uint64_t rate = label_rates_[static_cast<std::size_t>(depth)];
        if (rate && next_random() % rate == 0) {
            label_sink_->push_back({pos.fen(), depth, ply, static_eval, static_eval - raw_eval,
                                    eval, alpha, beta, ctx.improving, ctx.node,
                                    null_barrier_ != 0, iir, tt_depth >= 0, tt_depth,
                                    static_cast<int>(tt_bound)});
            meter.start(label_sink_, &nodes_);
        }
    }

    // Node-level pruning, only at non-PV nodes: their window is null, so
    // all that's asked is whether the node fails high or low, and a good
    // enough guess will do. Never in check, where the static eval means
    // little.
    if (!pv_node && !in_check && !no_prune && !probing && !pure) {
        // Alpha-side hash cut: a search a few plies shallower already put an
        // upper bound on this node well below alpha. Fail low without
        // searching (the mirror of ProbCut's hash-table test; see params.h).
        if (depth >= params::alpha_tt_min_depth && std::abs(alpha) < DECISIVE_BOUND &&
            (tt_bound == TT::UPPER || tt_bound == TT::EXACT) &&
            tt_depth >= depth - params::alpha_tt_depth_margin && tt_depth < depth) {
            count_tried(prune::AlphaTtCut);
            if (tt_score <= alpha - params::alpha_tt_margin) {
                count_fired(prune::AlphaTtCut);
                if (sample_verify())
                    record_verified(prune::AlphaTtCut,
                                    verify_node(pos, depth, alpha, beta, ply, cut_node) > alpha);
                return tt_score;
            }
        }

        // Reverse futility pruning: the eval is so far above beta that a
        // shallow search is very unlikely to bring it back below.
        if (depth <= params::rfp_max_depth && std::abs(beta) < DECISIVE_BOUND) {
            count_tried(prune::RFP);
            if (eval - prune::rfp_margin(ctx) >= beta) {
                count_fired(prune::RFP);
                if (sample_verify())
                    record_verified(prune::RFP,
                                    verify_node(pos, depth, alpha, beta, ply, cut_node) < beta);
                const int firm = fail_firm(eval, beta, params::ff_rfp_pct);
                count_feature(prune::FailFirm, firm != eval);
                return firm;
            }
        }

        // Razoring: the eval is so far below alpha that only a tactic could
        // save the node. Quiescence looks for one; if it fails low too,
        // believe it. Not against a mate score: quiescence failing to reach
        // a mate is no proof that the node can't.
        if (depth <= params::razor_max_depth && std::abs(alpha) < DECISIVE_BOUND) {
            count_tried(prune::Razor);
            if (eval + prune::razor_margin(ctx) < alpha) {
                const int v = qsearch(pos, alpha, alpha + 1, ply);
                if (stopped_) return 0;
                if (v <= alpha) {
                    count_fired(prune::Razor);
                    if (sample_verify())
                        record_verified(prune::Razor,
                                        verify_node(pos, depth, alpha, beta, ply, cut_node) > alpha);
                    return v;
                }
            }
        }

        // Null-move pruning: let the opponent move twice. If a reduced
        // search still fails high, a real move would almost surely do
        // better still. Not after another null move, and not without
        // pieces, where zugzwang makes passing a real advantage.
        if (depth >= params::nmp_min_depth && eval >= beta && std::abs(beta) < DECISIVE_BOUND &&
            ply > 0 && stack_[static_cast<std::size_t>(ply - 1)].move != Move{} &&
            pos.has_non_pawn_material(pos.side_to_move())) {
            count_tried(prune::NullMove);
            const int r = std::max(prune::nmp_reduction(ctx), 0);

            StateInfo st;
            keys_.push_back(pos.key());
            pos.make_null_move(st);
            tt_.prefetch(tt_key(pos));
            eval_.push(st.changes);
            ss.move = Move{};
            ss.piece = NO_PIECE;
            ss.quiet = false;
            stack_[static_cast<std::size_t>(ply + 1)].extensions = ss.extensions;
            stack_[static_cast<std::size_t>(ply + 1)].doubles = ss.doubles;
            const std::size_t barrier = std::exchange(null_barrier_, keys_.size());
            const std::uint64_t null_key = tt_key(pos);  // the null position's clock
            int v = -negamax(pos, depth - 1 - r, -beta, -beta + 1, ply + 1, !cut_node);
            null_barrier_ = barrier;
            eval_.pop();
            pos.unmake_null_move(st);
            keys_.pop_back();
            if (stopped_) return 0;

            if (v >= beta) {
                count_fired(prune::NullMove);
                if (v > DECISIVE_BOUND) v = beta;  // a win found by passing isn't proven
                if (sample_verify())
                    record_verified(prune::NullMove,
                                    verify_node(pos, depth, alpha, beta, ply, cut_node) < beta);
                return v;
            }

            // The pass failed: whatever refuted it is a threat.
            TT::Entry e;
            ss.threat = tt_.probe(null_key, e) ? e.move : Move{};
            ss.threat_score = v;
        }

        // ProbCut: if a good capture beats beta by a wide margin in a search
        // a few plies shallower, the full-depth search would almost surely
        // fail high too. Not when the hash table already says the node falls
        // short of the raised bound at nearly this depth.
        if (depth >= params::probcut_min_depth && std::abs(beta) < DECISIVE_BOUND) {
            const int pc_beta =
                beta + params::probcut_margin - (ctx.improving ? params::probcut_improving : 0);
            if (!(tt_bound != TT::NONE && tt_depth >= depth - params::probcut_tt_margin &&
                  tt_score < pc_beta)) {
                count_tried(prune::ProbCut);
                Move cut_move{};
                const int v = probcut(pos, depth, pc_beta, eval, tt_move, ply, cut_node, cut_move);
                if (stopped_) return 0;
                if (v >= pc_beta) {
                    count_fired(prune::ProbCut);
                    const int firm = fail_firm(v, beta, params::ff_probcut_pct);
                    count_feature(prune::FailFirm, firm != v);
                    tt_.store(tt_key(pos), cut_move, score_to_tt(firm, ply), raw_eval,
                              depth - params::probcut_tt_margin,
                              TT::LOWER, ss.tt_pv);
                    if (sample_verify())
                        record_verified(prune::ProbCut,
                                        verify_node(pos, depth, alpha, beta, ply, cut_node) < beta);
                    return firm;
                }
            }
        }
    }

    if (iir) {
        // Never below one ply. The node is past the "depth <= 0 is
        // quiescence" test, and the tunables' ranges overlap (IirMinDepth
        // from 2, IirReduction to 3): with the minimum depth at or under the
        // reduction a node went on at depth 0 or -1, storing that depth, and
        // with FfMainDepth on the fail-firm blend below divided by depth + 1
        // = 0 (found by FunHouse's tune, 2026-10-11; not reachable at the
        // defaults, 8 and 2).
        depth = std::max(depth - params::iir_reduction, 1);
        ctx.depth = depth;  // move-level pruning and LMR work with the reduced depth
    }

    const ContTable* const cont[2] = {cont_table(ply, 1), cont_table(ply, 2)};
    // Continuation history 4 plies back only orders quiet moves.
    const ContTable* const cont4 = params::cont_hist4_weight ? cont_table(ply, 4) : nullptr;
    MovePicker picker(*this, pos, tt_move, ply, cont, cont4);
    int legal_moves = 0;
    ss.child_cutoffs = 0;
    meter.reached_moves();

    const int alpha_orig = alpha;
    int best = -INF;
    Move best_move{};
    std::array<Move, 256> quiets_tried;
    int n_quiets = 0;
    std::array<Move, 256> captures_tried;
    int n_captures = 0;
    int searched = 0;
    bool skip_quiets = false;
    // Our pieces attacked by cheaper ones, computed at the first quiet move.
    Bitboard static_threats = 0;
    bool have_static_threats = false;
    // Built on the first move: every move is tested for check.
    Position::CheckInfo check_info;
    bool have_check_info = false;
    int skipped_excluded = 0;
    // Moves pruned so far by LMP, history, futility, SEE (quiet, capture)
    // and underpromotion pruning, for continuation logging.
    int pruned[6] = {};

    for (Move m; (m = picker.next()) != Move{};) {
        // At a root in the tablebases, only the moves that keep its result.
        if (root && !root_moves_.empty() &&
            std::find(root_moves_.begin(), root_moves_.end(), m) == root_moves_.end())
            continue;
        if (root && m == avoid_root_) continue;  // avoid_at_root (gendata)
        // Position in the move order, counting quiet moves the picker dropped.
        // In a probe, the excluded move may be left out of it (after it).
        const int i = legal_moves++ + picker.dropped_before() - skipped_excluded;
        if (m == excluded) {
            skipped_excluded = params::se_probe_skip_index;
            continue;
        }
        const bool quiet = is_quiet(pos, m);
        // Only quiet moves are ever pruned or reduced on account of checks.
        if (!have_check_info) {
            check_info = pos.check_info();
            have_check_info = true;
        }
        const bool any_check = pos.gives_check(m, check_info);
        // Pruning and reductions only consider checks by quiet moves.
        const bool gives_check = quiet && any_check;

        // Once late move pruning has triggered, the remaining quiet moves
        // go without further ado, checks aside.
        if (skip_quiets && quiet && !gives_check) {
            count_tried(prune::LMP);
            count_fired(prune::LMP);
            if (cn) ++pruned[0];
            if (sample_verify()) verify_pruned_move(pos, m, prune::LMP, depth, alpha, ply);
            continue;
        }

        ctx.move = m;
        ctx.move_index = i;
        ctx.alpha = alpha;
        ctx.child_cutoffs = ss.child_cutoffs;
        ctx.quiet = quiet;
        ctx.killer = quiet && (m == ss.killers[0] || m == ss.killers[1]);
        ctx.history = quiet ? quiet_history(pos, m, cont) : 0;
        ctx.capture_history = !quiet && is_capture(pos, m) ? capt_hist_[capture_index(pos, m)] : 0;
        ctx.lmr_depth = std::max(depth - 1 - prune::lmr_base_reduction(depth, i), 0);
        ctx.gives_check = gives_check;
        if (quiet)
            count_feature(prune::QuietCheck,
                          params::quiet_check_bonus &&
                              (check_info.checking_from[type_of(pos.piece_at(m.from()))] & bit(m.to())));
        // Static threats: only computed for LmrStaticThreat or a guide.
        if (quiet && !have_static_threats && (params::lmr_static_threat || prune::guide)) {
            static_threats = pos.threatened_by_lesser(pos.side_to_move());
            have_static_threats = true;
        }
        ctx.escapes_threat = quiet && (static_threats & bit(m.from()));
        ctx.threatened = ss.threat != Move{};
        ctx.evades_threat = ctx.threatened &&
                            (m.from() == ss.threat.to() ||
                             (squares_between(ss.threat.from(), ss.threat.to()) & bit(m.to())));

        // Move-level pruning, anywhere but the root, once a move has been
        // searched and didn't lose to a mate: there is then a real score
        // to fall back on if everything else is skipped. Checks are spared
        // futility pruning and late move pruning (pruning them too lost 42
        // Elo, before the 2.0 tune and after it): a checking sacrifice is
        // exactly the quiet move that a bad static eval says to skip.
        if (!root && !in_check && !pure && best > -DECISIVE_BOUND) {
            // Underpromotions: a queen does at least as well but for
            // stalemate tricks and knight forks (checks are kept).
            if (params::underpromo_prune && m.promotion() != PieceType::None &&
                m.promotion() != PieceType::Queen) {
                count_tried(prune::Underpromo);
                if (m.promotion() != PieceType::Knight || !any_check) {
                    count_fired(prune::Underpromo);
                    if (cn) ++pruned[5];
                    if (sample_verify())
                        verify_pruned_move(pos, m, prune::Underpromo, depth, alpha, ply);
                    continue;
                }
            }
            if (quiet) {
                // Late move pruning: quiet moves this far down the ordering
                // hardly ever matter at low depth.
                if (depth <= params::lmp_max_depth && !gives_check) {
                    count_tried(prune::LMP);
                    if (i >= prune::lmp_threshold(ctx)) {
                        skip_quiets = true;
                        // Not while verifying pruning: each skipped move is
                        // a sample.
                        if (params::lmp_in_picker && !verify_rate_)
                            picker.drop_quiets(check_info);
                        count_fired(prune::LMP);
                        if (cn) ++pruned[0];
                        if (sample_verify())
                            verify_pruned_move(pos, m, prune::LMP, depth, alpha, ply);
                        continue;
                    }
                }
                // History pruning: at low depth, a quiet move whose history
                // says it keeps failing isn't worth a search.
                if (ctx.lmr_depth <= params::hist_prune_max_depth && !ctx.gives_check) {
                    count_tried(prune::HistoryPrune);
                    if (ctx.history < prune::history_prune_threshold(ctx)) {
                        count_fired(prune::HistoryPrune);
                        if (cn) ++pruned[1];
                        if (sample_verify())
                            verify_pruned_move(pos, m, prune::HistoryPrune, depth, alpha, ply);
                        continue;
                    }
                }
                // Futility pruning: a quiet move can't gain enough to reach
                // alpha from this static eval.
                if (ctx.lmr_depth <= params::fp_max_depth && !ctx.gives_check) {
                    count_tried(prune::Futility);
                    if (eval + prune::futility_margin(ctx) <= alpha) {
                        count_fired(prune::Futility);
                        if (cn) ++pruned[2];
                        if (sample_verify())
                            verify_pruned_move(pos, m, prune::Futility, depth, alpha, ply);
                        continue;
                    }
                }
                // SEE pruning: the moved piece just gets taken.
                if (ctx.lmr_depth <= params::see_max_depth) {
                    count_tried(prune::SeeQuiet);
                    if (!see_at_least(pos, m, prune::see_threshold(ctx))) {
                        count_fired(prune::SeeQuiet);
                        if (cn) ++pruned[3];
                        if (sample_verify())
                            verify_pruned_move(pos, m, prune::SeeQuiet, depth, alpha, ply);
                        continue;
                    }
                }
            } else if (depth <= params::see_max_depth) {
                // SEE pruning of captures that lose too much material.
                count_tried(prune::SeeCapture);
                const int threshold = prune::see_threshold(ctx);
                if (!(picker.good_capture() && threshold <= 0) && !see_at_least(pos, m, threshold)) {
                    count_fired(prune::SeeCapture);
                    if (cn) ++pruned[4];
                    if (sample_verify())
                        verify_pruned_move(pos, m, prune::SeeCapture, depth, alpha, ply);
                    continue;
                }
            }
        }

        // Singular extension (see params.h): is the hash move the only
        // good move here? Search the others, shallower, against a bound
        // just below its score. If none reaches it, extend the hash move
        // (by two if they all fall well short, see SeDoubleMargin);
        // if even that bound beats beta, some other move does too: cut.
        // If the others reach it but it is below beta, the hash move isn't
        // special: when it was expected to fail high anyway (its score is
        // at least beta, or this is a cut node), search it shallower.
        int extension = 0;
        if (!root && !probing && !pure && m == tt_move && depth >= params::se_min_depth &&
            ply < params::se_ply_mult * root_depth_ && ss.extensions < params::se_max_extensions &&
            tt_depth >= depth - params::se_tt_depth_margin &&
            (tt_bound == TT::LOWER || tt_bound == TT::EXACT) &&
            std::abs(tt_score) < DECISIVE_BOUND) {
            const int s_beta = tt_score - params::se_margin * depth / 16;
            const Move threat = ss.threat;  // the probe re-enters this ply
            const int child_cutoffs = ss.child_cutoffs;
            const int threat_score = ss.threat_score;
            ss.excluded = m;
            const int v = negamax(pos, (depth - 1) * params::se_probe_depth_pct / 100, s_beta - 1,
                                  s_beta, ply, cut_node);
            ss.excluded = Move{};
            ss.threat = threat;
            ss.threat_score = threat_score;
            ss.child_cutoffs = child_cutoffs;
            if (stopped_) return 0;
            if (v < s_beta)
                extension = !pv_node && v < s_beta - params::se_double_margin &&
                                    ss.doubles < params::se_max_doubles
                                ? 2
                                : 1;
            else if (tt_score >= beta || cut_node)
                extension = -params::se_negative;
            const bool multi_cut = v >= s_beta && s_beta >= beta;
            count_feature(prune::Singular, extension > 0);
            count_feature(prune::DoubleExt, extension == 2);
            count_feature(prune::NegativeExt, extension < 0 && !multi_cut);
            count_feature(prune::MultiCut, multi_cut);
            if (multi_cut) return s_beta;
        }

        // A check that doesn't lose material is searched a ply deeper (not on
        // top of a singular extension, nor after a negative one; only singular
        // extensions count towards the per-path cap).
        const int check_ext = (!pure || params::pure_search == 2) && any_check && see_at_least(pos, m, params::check_ext_see) ? 1 : 0;
        if (any_check) count_feature(prune::CheckExt, check_ext > 0 && extension == 0);

        // A capture that loses material by static exchange (and neither
        // checks nor promotes) may be reduced like a quiet move
        // (LmrCaptures). Decided before the move is made, which SEE needs.
        const bool losing_capture =
            params::lmr_captures && !quiet && !any_check && m.promotion() == PieceType::None &&
            depth >= params::lmr_min_depth && i >= params::lmr_min_moves && !see_at_least(pos, m, 0);

        if (cn)
            log_cont_row(*cn, pos, m, ctx, picker.stage(), pruned, skip_quiets, any_check,
                         m == tt_move, alpha, best, searched, ss.threat);
        StateInfo st;
        keys_.push_back(pos.key());
        ss.move = m;
        ss.piece = pos.piece_at(m.from());
        ss.quiet = quiet;
        ss.history = ctx.history;
        ss.move_index = i;
        const std::uint64_t nodes_before = nodes_;
        pos.make_move(m, st, any_check);
        tt_.prefetch(tt_key(pos));
        eval_.push(st.changes);
        ++searched;

        // Principal variation search: the first move gets the full window.
        // Every later one is first tested with a null window around alpha,
        // which only answers "is it better than what we have?"; the rare
        // move that says yes is searched again with the full window.
        int new_depth =
            depth - 1 + (extension < 0 ? extension : std::max(extension, check_ext));
        stack_[static_cast<std::size_t>(ply + 1)].extensions = ss.extensions + (extension > 0);
        stack_[static_cast<std::size_t>(ply + 1)].doubles = ss.doubles + (extension == 2);
        int score;
        if (searched == 1) {
            ss.provisional = false;
            score = -negamax(pos, new_depth, -beta, -alpha, ply + 1, !pv_node && !cut_node);
        } else {
            // Late move reductions: quiet moves late in the ordering are
            // searched shallower first, and at full depth only if the
            // reduced search says they beat alpha.
            int r = 0;
            if (depth >= params::lmr_min_depth && !in_check && !pure &&
                i >= params::lmr_min_moves + (pv_node ? 1 : 0) &&
                (quiet ? !ctx.gives_check : losing_capture)) {
                count_tried(prune::LMR);
                if (!quiet) count_tried(prune::LmrCapture);
                // At least one ply left to search (and no clamp with lo > hi
                // if LmrMinDepth is tuned down to 1).
                r = std::max(std::min(prune::lmr_reduction(ctx), new_depth - 1), 0);
                count_feature(prune::LmrChildCut, params::lmr_child_cutoffs &&
                                                      ctx.child_cutoffs >= params::lmr_child_cutoffs);
                if (quiet && ctx.parent_quiet)
                    count_feature(prune::LmrParentHist,
                                  params::lmr_parent_hist &&
                                      ((ctx.history > 0 && ctx.parent_history < 0) ||
                                       (ctx.history < 0 && ctx.parent_history > 0)));
            }
            if (r > 0) {
                count_fired(prune::LMR);
                if (!quiet) count_fired(prune::LmrCapture);
                ss.provisional = true;  // searched again at full depth if it fails low
                ss.reduction = r;
                score = -negamax(pos, new_depth - r, -alpha - 1, -alpha, ply + 1, true);
                ss.reduction = 0;
                if (score > alpha) {
                    if (cn) ++cn->researches;
                    if (!verify_depth_) {
                        ++stats_.lmr_researches;
                        if (!quiet) ++stats_.lmr_capture_researches;
                    }
                    // How far above alpha the reduced search landed says how
                    // much to trust it: well above, re-search a ply deeper
                    // than planned; barely above the best score so far, a ply
                    // shallower (see params.h). The PV re-search below keeps
                    // the adjusted depth.
                    const int reduced = new_depth - r;
                    const bool deeper =
                        score > alpha + params::lmr_deeper_base + params::lmr_deeper_mult * r;
                    const bool shallower = score < best + params::lmr_shallower;
                    count_feature(prune::LmrDeeper, deeper);
                    count_feature(prune::LmrShallower, shallower);
                    new_depth += deeper - shallower;
                    ss.provisional = pv_node;  // a PV node searches it again, full window
                    if (new_depth > reduced)
                        score = -negamax(pos, new_depth, -alpha - 1, -alpha, ply + 1, !cut_node);
                } else if (sample_verify()) {
                    ++verify_depth_;
                    const int full =
                        -negamax(pos, new_depth, -alpha - 1, -alpha, ply + 1, !cut_node);
                    --verify_depth_;
                    record_verified(prune::LMR, full > alpha);
                    if (!quiet) record_verified(prune::LmrCapture, full > alpha);
                }
            } else {
                ss.provisional = pv_node;
                score = -negamax(pos, new_depth, -alpha - 1, -alpha, ply + 1, !cut_node);
            }
            if (pv_node && score > alpha && score < beta) {
                ss.provisional = false;
                score = -negamax(pos, new_depth, -beta, -alpha, ply + 1, false);
            }
        }

        eval_.pop();
        pos.unmake_move(m, st);
        keys_.pop_back();
        if (root) {
            const std::uint64_t spent = nodes_ - nodes_before;
            root_move_nodes_[static_cast<std::size_t>(m.from() * 64 + m.to())] += spent;
            root_nodes_total_ += spent;
        }
        if (stopped_) return 0;  // result is incomplete; don't store it
        if (cn) {
            ContRow& r = cn->rows.back();
            r.score = score;
            r.raised = score > alpha;
        }

        if (score > best) {
            best = score;
            if (score > alpha) {
                alpha = score;
                best_move = m;
                // Only a move that beat alpha is known to be good; at the
                // root this is what gets played if time runs out.
                if (root) {
                    root_best_ = m;
                    root_best_score_ = score;
                }
                if (alpha >= beta) {
                    if (quiet) update_quiet_stats(pos, m, quiets_tried.data(), n_quiets, depth, ply);
                    if (ply >= 1 && excluded == Move{})
                        ++stack_[static_cast<std::size_t>(ply - 1)].child_cutoffs;
                    update_capture_stats(pos, is_capture(pos, m) ? m : Move{}, captures_tried.data(),
                                         n_captures, depth);
                    break;
                }
            }
        }
        if (quiet) quiets_tried[static_cast<std::size_t>(n_quiets++)] = m;
        else if (is_capture(pos, m)) captures_tried[static_cast<std::size_t>(n_captures++)] = m;
    }

    if (legal_moves == 0) return in_check ? -MATE + ply : 0;

    // A singular-extension probe only reports: nothing is learned from or
    // stored for a node searched without one of its moves. With no other
    // move at all, the left-out one is singular by definition.
    if (probing) return searched ? best : alpha;

    const auto bound = best >= beta ? TT::LOWER : best > alpha_orig ? TT::EXACT : TT::UPPER;
    if (cn) close_cont_node(cn, best, bound);

    // Prior countermove: every move here failed low, so the opponent's quiet
    // move that led here held against all of them. Reward it in its side's
    // butterfly and continuation histories (see params::prior_cm_pct).
    if (bound == TT::UPPER && params::prior_cm_pct && ply >= 1) {
        const StackEntry& prev = stack_[static_cast<std::size_t>(ply - 1)];
        // The gates (see params.h): not a provisional search the parent
        // repeats, not an expected all-node, and in proportion to how late
        // the move came in the parent's order (a first move was expected to
        // refute, and the parent rewards it anyway when it cuts).
        int pct = params::prior_cm_pct;
        if ((params::prior_cm_final_only && prev.provisional) ||
            (params::prior_cm_node_gate && !pv_node && !cut_node))
            pct = 0;
        if (params::prior_cm_index_cap)
            pct = pct * std::min(prev.move_index, params::prior_cm_index_cap) /
                  params::prior_cm_index_cap;
        const bool bonus_given =
            pct && prev.quiet && prev.move != Move{} && prev.piece != NO_PIECE;
        count_feature(prune::PriorCm, bonus_given);
        if (bonus_given) {
            const int bonus =
                std::min(params::hist_bonus_mult * depth * depth, params::hist_bonus_max) * pct /
                100;
            const Color side = opposite(pos.side_to_move());
            gravity(history_[static_cast<std::size_t>(side)][static_cast<std::size_t>(
                        prev.move.from())][static_cast<std::size_t>(prev.move.to())],
                    bonus);
            const std::size_t reply = piece_to(prev.piece, prev.move.to());
            for (int back = 1; back <= 2; ++back)
                if (ContTable* t = cont_table(ply - 1, back)) gravity((*t)[reply], bonus);
        }
    }
    // A node that fails low below a flagged parent keeps the flag when it
    // was searched deep enough: it is the parent's refutation line, likely
    // to come back to the PV (see params::tt_pv_inherit_depth).
    if (bound == TT::UPPER && !ss.tt_pv && ply >= 1 &&
        stack_[static_cast<std::size_t>(ply - 1)].tt_pv) {
        const bool inherit = depth > params::tt_pv_inherit_depth;
        count_feature(prune::TtPvInherit, inherit);
        if (inherit) ss.tt_pv = true;
    }

    // Teach the correction history what the static eval got wrong. Only
    // where the score says something about it: not in check (no eval),
    // not when a capture or promotion was best (the eval never claims to
    // see tactics), not a mate, and not a bound on the wrong side of the
    // eval (a fail high below it, or a fail low above it, says nothing
    // about how far off it was).
    if (!in_check && (best_move == Move{} || is_quiet(pos, best_move)) &&
        std::abs(best) < DECISIVE_BOUND && !(bound == TT::LOWER && best <= static_eval) &&
        !(bound == TT::UPPER && best >= static_eval)) {
        // The score carries the fifty-move scaling, which is applied after
        // the correction: undo it first (CorrUnscaleFifty), or the table
        // learns the scaling as eval error. Only while the scale is at
        // least 1/2, where undoing it amplifies noise at most twofold.
        int target = best;
        const int clock = pos.halfmove_clock();
        const int left = params::fifty_scale - clock;
        const bool scaled = params::corr_unscale_fifty && params::fifty_scale && clock > 0;
        if (!scaled || 2 * left >= params::fifty_scale) {
            if (scaled) target = best * params::fifty_scale / left;
            update_correction(pos, ply, depth, target - raw_eval);
        }
    }

    // A fail high from a shallow search overstates the node more than a
    // deep one: pulled toward beta, trusted in proportion to depth
    // (params::ff_main_depth). After the correction history, which learns
    // what the search found; never at the root.
    if (bound == TT::LOWER && !root) {
        const int before = best;
        if (params::ff_main_depth && !pure && std::abs(best) < DECISIVE_BOUND &&
            std::abs(beta) < DECISIVE_BOUND)
            best = (best * depth + beta) / (depth + 1);
        count_feature(prune::FailFirm, best != before);
    }

    tt_.store(tt_key(pos), best_move, score_to_tt(best, ply), raw_eval, depth, bound, ss.tt_pv);
    return best;
}

std::vector<Move> Searcher::principal_variation(Position& pos, int max_len) {
    // Follows hash moves from the root, up to the first position after it
    // that the search scores as a draw (fifty moves, or a repetition of the
    // game or of the line itself): the game would end there, and the score
    // says so. Each move is checked against the legal moves, since a hash
    // collision could supply a bogus one. keys_ holds the game history, as
    // is_draw expects.
    std::vector<Move> pv;
    std::vector<StateInfo> states(static_cast<std::size_t>(max_len));
    const std::size_t base = keys_.size();
    while (static_cast<int>(pv.size()) < max_len && (pv.empty() || !is_draw(pos))) {
        TT::Entry e;
        if (!tt_.probe(tt_key(pos), e) || e.move == Move{}) break;

        MoveList moves;
        int count = 0;
        generate_legal(pos, moves, count);
        if (std::find(moves.begin(), moves.begin() + count, e.move) == moves.begin() + count) break;

        keys_.push_back(pos.key());
        pos.make_move(e.move, states[pv.size()]);
        pv.push_back(e.move);
    }
    keys_.resize(base);
    for (auto i = pv.size(); i-- > 0;) pos.unmake_move(pv[i], states[i]);
    return pv;
}

std::vector<Move> Searcher::reported_line(Position& pos, Move first, int max_len) {
    std::vector<Move> pv{first};
    StateInfo st;
    keys_.push_back(pos.key());
    pos.make_move(first, st);
    if (!is_draw(pos))
        for (const Move m : principal_variation(pos, max_len - 1)) pv.push_back(m);
    pos.unmake_move(first, st);
    keys_.pop_back();
    return pv;
}

Move Searcher::ponder_move(Position& pos, Move best) {
    if (best == Move{}) return Move{};
    StateInfo st;
    pos.make_move(best, st);
    TT::Entry e;
    // A colliding entry could hold any move: only a legal one will do.
    const Move reply = tt_.probe(tt_key(pos), e) && pos.is_legal(e.move) ? e.move : Move{};  // stored under tt_key
    pos.unmake_move(best, st);
    return reply;
}

Searcher::RootEval Searcher::root_eval(const Position& pos) {
    if (pos.in_check(pos.side_to_move())) return {NO_EVAL, NO_EVAL};
    const int corr = correction(pos, 0);
    const int raw = evaluate(pos);
    const int static_eval = corrected_eval(pos, raw, corr);
    return {static_eval, static_eval - raw};
}

int Searcher::quiescence(Position& pos, const std::vector<std::uint64_t>& history, bool plain) {
    nodes_ = 0;
    node_limit_ = 0;
    soft_ms_ = hard_ms_ = 0;
    pondering_ = false;
    stopped_ = false;
    eval_.reset(pos);
    for (auto& e : stack_) e = StackEntry{};
    keys_ = history;
    null_barrier_ = 0;
    verify_depth_ = 0;
    tb_pieces_ = 0;  // the score of the position's captures, not a tablebase verdict
    qs_plain_ = plain;
    const int score = qsearch(pos, -INF, INF, 0);
    qs_plain_ = false;
    return score;
}

Move Searcher::search(Position& pos, const SearchLimits& limits,
                      const std::vector<std::uint64_t>& history, bool verbose) {
    if (!root_ready_) begin_search(pos, limits);  // which starts the clock
    root_ready_ = false;
    nodes_ = 0;
    node_limit_ = limits.nodes;
    stopped_ = false;
    eval_.reset(pos);
    for (auto& e : stack_) e = StackEntry{};
    keys_ = history;
    keys_.reserve(history.size() + MAX_PLY);
    null_barrier_ = 0;
    verify_depth_ = 0;
    last_score_ = 0;
    iterations_.clear();
    set_time_limits(pos, limits);
    limit_base_ms_ = 0;
    // Pondering: the limits wait for the ponderhit (check_ponderhit).
    pondering_ = limits.ponder;
    if (pondering_) {
        ponder_soft_ms_ = soft_ms_;
        ponder_hard_ms_ = hard_ms_;
        soft_ms_ = hard_ms_ = 0;
    }

    MoveList root_moves;
    int root_count = 0;
    generate_legal(pos, root_moves, root_count);
    if (root_count == 0) return Move{};
    tb_hits_ = 0;
    root_move_nodes_.fill(0);
    root_nodes_total_ = 0;
    Move best = root_moves_.empty() ? root_moves[0] : root_moves_.front();
    const int choices = root_moves_.empty() ? root_count : static_cast<int>(root_moves_.size());

    // Only one move to choose (legal, or keeping a tablebase result): no
    // point thinking long on the clock, but a short search still reports a
    // score (match runners adjudicate on it, and GUIs show it).
    const bool forced = choices == 1 && hard_ms_;

    const int max_depth = forced                ? std::min(4, MAX_PLY - 1)
                          : limits.depth > 0 ? std::min(limits.depth, MAX_PLY - 1)
                                             : MAX_PLY - 1;

    int prev_score = 0;
    int stability = 0;   // iterations the best move has held
    Move prev_best{};
    // Continuation logging's own record of the iterations.
    int log_stability = 0;
    Move log_prev_best{};
    std::uint64_t log_nodes = 0;
    for (int depth = 1; depth <= max_depth; ++depth) {
        root_best_ = Move{};
        seldepth_ = 0;

        // Aspiration window: expect the score to stay near the last one and
        // search a narrow window around it, which prunes more. If the score
        // lands outside, widen that side and search again. Mate scores
        // jump too far for a window to help.
        int delta = params::asp_delta;
        int alpha = -INF, beta = INF;
        if (depth >= params::asp_min_depth && std::abs(prev_score) < DECISIVE_BOUND) {
            alpha = std::max(prev_score - delta, -INF);
            beta = std::min(prev_score + delta, INF);
        }
        int score;
        while (true) {
            root_depth_ = depth;
            score = negamax(pos, depth, alpha, beta, 0, false);
            if (stopped_) break;
            if (score <= alpha) {
                beta = (alpha + beta) / 2;
                alpha = std::max(score - delta, -INF);
            } else if (score >= beta) {
                beta = std::min(score + delta, INF);
            } else {
                break;
            }
            delta += delta * params::asp_widen_pct / 100;
        }

        // One "info" line: depth, score, counters and a principal variation.
        // Written in one piece: the UCI thread may be answering "isready".
        // At a root in the tablebases, their verdict is the score, unless
        // the search found a mate.
        auto report = [&](int d, int s, const std::vector<Move>& pv) {
            const long long ms = elapsed_ms();
            if (root_tb_score_ != NO_EVAL && std::abs(s) <= MATE_BOUND) s = root_tb_score_;
            std::uint64_t nodes = nodes_, tb_hits = tb_hits_;  // over all threads
            for (const Searcher* h : helpers_) {
                nodes += h->published_nodes();
                tb_hits += h->published_tb_hits();
            }
            std::ostringstream line;
            line << "info depth " << d << " seldepth " << std::max(seldepth_, d)
                 << " score " << score_string(s)
                 << " nodes " << nodes << " nps "
                 << (ms > 0 ? nodes * 1000 / static_cast<std::uint64_t>(ms) : nodes)
                 << " hashfull " << tt_.hashfull() << " tbhits " << tb_hits << " time " << ms
                 << " pv";
            for (const Move m : pv) line << ' ' << pos.move_to_uci(m);
            line << '\n';
            std::cout << line.str() << std::flush;
        };

        if (stopped_) {
            // A partial iteration still searched the previous best move
            // first (it is the hash move). root_best_ is only set by a move
            // that beat alpha, so if one did, it is a real improvement and
            // safe to play; otherwise keep the last completed result.
            if (root_best_ != Move{} && root_best_ != best) {
                best = root_best_;
                // Report it, so that the move played heads the last PV the
                // GUI saw. The rest of the line comes from the hash table,
                // where the partial iteration left it.
                if (verbose) report(depth, root_best_score_, reported_line(pos, best, depth));
            }
            break;
        }
        best = root_best_;
        const int score_before = prev_score;
        prev_score = score;
        last_score_ = score;
        iterations_.push_back({depth, nodes_, score, std::max(seldepth_, depth), best});

        if (verbose) report(depth, score, best != Move{} ? reported_line(pos, best, depth)
                                                         : principal_variation(pos, depth));
        if (CONT_LOG_BUILD && cont_log_) {
            // Continuation logging: one row per completed iteration.
            log_stability = best == log_prev_best ? log_stability + 1 : 0;
            log_prev_best = best;
            ContRoot r;
            r.depth = depth;
            r.best = pos.move_to_uci(best);
            r.score = score;
            r.score_change = depth > 1 ? score - score_before : 0;
            r.stability = log_stability;
            r.nodes = nodes_;
            r.iteration_nodes = nodes_ - log_nodes;
            log_nodes = nodes_;
            r.elapsed_ms = elapsed_ms();
            r.soft_ms = soft_ms_;
            r.hard_ms = hard_ms_;
            r.node_share = root_nodes_total_
                               ? static_cast<int>(root_move_nodes_[static_cast<std::size_t>(
                                                      best.from() * 64 + best.to())] *
                                                  100 / root_nodes_total_)
                               : 0;
            cont_log_->write_root(r);
        }

        // Time: stop before a new iteration once past the soft limit, scaled
        // by how settled the search looks (see params.h). A best move that
        // keeps changing, or a falling score, earns more time; one that has
        // held for several iterations, less; so does a best move that took
        // most of the search's nodes. With soft_scale, a soft node limit is
        // scaled the same way (data generation: the nodes go where the
        // position is unsettled).
        check_ponderhit();
        const bool scaled_nodes = limits.soft_nodes && limits.soft_scale;
        if (soft_ms_ || scaled_nodes) {
            stability = best == prev_best ? stability + 1 : 0;
            prev_best = best;
            const int drop = depth > 1 ? std::max(score_before - score, 0) : 0;
            long long pct =
                std::max(params::tm_stable_start - params::tm_stable_step * stability,
                         params::tm_stable_min) +
                std::min(params::tm_drop_scale * drop / 100, params::tm_drop_max);
            const long long pct_before_nodes = pct;
            if (depth >= params::tm_nodes_min_depth && root_nodes_total_) {
                const long long frac = static_cast<long long>(
                    root_move_nodes_[static_cast<std::size_t>(best.from() * 64 + best.to())] *
                    100 / root_nodes_total_);
                pct = pct *
                      std::max(params::tm_nodes_base - params::tm_nodes_slope * frac / 100,
                               static_cast<long long>(params::tm_nodes_min)) /
                      100;
                if (soft_ms_) {
                    // Acted: the node share changed whether this iteration stops.
                    const long long elapsed = elapsed_ms();
                    auto stops = [&](long long p) {
                        return elapsed >= std::min(limit_base_ms_ + soft_ms_ * p / 100, hard_ms_);
                    };
                    count_feature(prune::TmNodes, stops(pct) != stops(pct_before_nodes));
                }
            }
            if (soft_ms_ && elapsed_ms() >= std::min(limit_base_ms_ + soft_ms_ * pct / 100, hard_ms_)) break;
            if (scaled_nodes && depth >= limits.min_depth &&
                nodes_ >= limits.soft_nodes * static_cast<std::uint64_t>(pct) / 100)
                break;
        }
        // Nodes: likewise past a soft node limit, unscaled without soft_scale.
        if (limits.soft_nodes && !limits.soft_scale && depth >= limits.min_depth && nodes_ >= limits.soft_nodes)
            break;
        // A forced mate found within this depth can't get any shorter.
        if (score > MATE_BOUND && MATE - score <= depth) break;
    }
    drop_cont_nodes(0);  // sampled nodes cut short by the end of the search
    publish_counts();
    return best;
}

std::vector<Move> Searcher::picker_order(const Position& pos, Move tt_move, Move killer1,
                                         Move killer2) {
    stack_[0].killers[0] = killer1;
    stack_[0].killers[1] = killer2;
    const ContTable* const cont[2] = {nullptr, nullptr};
    MovePicker picker(*this, pos, tt_move, 0, cont, nullptr);
    std::vector<Move> order;
    for (Move m; (m = picker.next()) != Move{};) order.push_back(m);
    return order;
}
