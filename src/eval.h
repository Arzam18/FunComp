#pragma once
#include "nnue.h"
#include "position.h"
#include <array>
#include <vector>

// Static evaluation in centipawns from the side to move's point of view.
//
// Tapered PeSTO evaluation (Ronald Friederich's tuned piece values and
// piece-square tables, as published on the Chess Programming Wiki). Each
// piece has a middlegame and an endgame score; the two totals are blended
// by game phase, which runs from 24 (all minor and major pieces on the
// board) down to 0 (pawns and kings only).
//
// Full recomputation from the board. The search uses Evaluator below; this
// stays as the reference it is tested against.
int evaluate(const Position& pos);

// The evaluation in its parts (data generation records them): the PeSTO
// score, the initiative and mop-up terms, and the percent of their sum kept
// in drawish endings; total is evaluate(pos), their sum scaled by it.
struct EvalParts {
    int pesto, initiative, mop_up, drawish_pct, total;
};
EvalParts eval_parts(const Position& pos);

// Incrementally updated evaluation. The search calls push() after every
// make_move with the move's piece changes and pop() after unmake_move, so
// each node's evaluation costs a few table lookups instead of a pass over
// the board. An NNUE accumulator will slot into the same push/pop scheme.
class Evaluator {
public:
    static constexpr int STACK_SIZE = 256;  // > maximum search ply

    // Recomputes from scratch; call at the root of every search.
    void reset(const Position& pos);
    void push(const MoveChanges& changes);
    void pop() { --top_; }

    // Same result as evaluate(pos) for the position the stack describes.
    int evaluate(const Position& pos) const;

private:
    friend int evaluate(const Position& pos);
    friend EvalParts eval_parts(const Position& pos);

    struct Accumulator {
        int mg[2];  // middlegame score per color (White, Black)
        int eg[2];  // endgame score per color
        int phase;  // 0..24+, see above
    };

    static Accumulator compute(const Position& pos);
    static int score(const Accumulator& acc, Color side_to_move);

    std::array<Accumulator, STACK_SIZE> stack_{};
    // The network's accumulators, one per ply as above: allocated when a
    // network is first used (nnue.h), empty otherwise.
    // Mutable: evaluating may rebuild a stale view (nnue.h).
    mutable std::vector<nnue::Accumulator> nnue_;
    // Their extra-input lists, for networks with extra families (nnue.h),
    // linked to the accumulators at each reset.
    std::vector<nnue::ExtraLists> nnue_extras_;
    // Stale views are rebuilt through this thread's refresh cache (nnue.h).
    // Updates stay eager: lazy ones (applied only when a node evaluates)
    // measured 3-5% slower on every net, since the search evaluates at
    // nearly every node (ROADMAP 5d, 2026-10-06).
    mutable nnue::RefreshCache cache_;
    int top_ = 0;
    // A network was loaded at the last reset: stack_ is then left as it was.
    bool network_ = false;
};
