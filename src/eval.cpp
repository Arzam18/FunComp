#include "eval.h"
#include "initiative.h"
#include "params.h"
#include "see.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>

namespace {
// Tables are laid out as printed on the CPW page: first row is rank 8,
// a-file first, from White's point of view. See the index mapping below.
using Table = std::array<int, 64>;

constexpr int MG_VALUE[6] = {82, 337, 365, 477, 1025, 0};
constexpr int EG_VALUE[6] = {94, 281, 297, 512, 936, 0};
constexpr int PHASE_INC[6] = {0, 1, 1, 2, 4, 0};

constexpr Table MG_PAWN = {
      0,   0,   0,   0,   0,   0,  0,   0,
     98, 134,  61,  95,  68, 126, 34, -11,
     -6,   7,  26,  31,  65,  56, 25, -20,
    -14,  13,   6,  21,  23,  12, 17, -23,
    -27,  -2,  -5,  12,  17,   6, 10, -25,
    -26,  -4,  -4, -10,   3,   3, 33, -12,
    -35,  -1, -20, -23, -15,  24, 38, -22,
      0,   0,   0,   0,   0,   0,  0,   0,
};
constexpr Table EG_PAWN = {
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
};
constexpr Table MG_KNIGHT = {
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23,
};
constexpr Table EG_KNIGHT = {
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
};
constexpr Table MG_BISHOP = {
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
};
constexpr Table EG_BISHOP = {
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
     -8,  -4,   7, -12,  -3, -13,  -4, -14,
      2,  -8,   0,  -1,  -2,   6,   0,   4,
     -3,   9,  12,   9,  14,  10,   3,   2,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -23,  -9, -23,  -5,  -9, -16,  -5, -17,
};
constexpr Table MG_ROOK = {
     32,  42,  32,  51,  63,   9,  31,  43,
     27,  32,  58,  62,  80,  67,  26,  44,
     -5,  19,  26,  36,  17,  45,  61,  16,
    -24, -11,   7,  26,  24,  35,  -8, -20,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -19, -13,   1,  17,  16,   7, -37, -26,
};
constexpr Table EG_ROOK = {
     13,  10,  18,  15,  12,  12,   8,   5,
     11,  13,  13,  11,  -3,   3,   8,   3,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
      4,   3,  13,   1,   2,   1,  -1,   2,
      3,   5,   8,   4,  -5,  -6,  -8, -11,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -9,   2,   3,  -1,  -5, -13,   4, -20,
};
constexpr Table MG_QUEEN = {
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
};
constexpr Table EG_QUEEN = {
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
};
constexpr Table MG_KING = {
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
};
constexpr Table EG_KING = {
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
};

constexpr const Table* MG_TABLES[6] = {&MG_PAWN, &MG_KNIGHT, &MG_BISHOP, &MG_ROOK, &MG_QUEEN, &MG_KING};
constexpr const Table* EG_TABLES[6] = {&EG_PAWN, &EG_KNIGHT, &EG_BISHOP, &EG_ROOK, &EG_QUEEN, &EG_KING};

// Combined value + table, indexed [piece][square] with our a1 = 0 squares.
// The printed tables start at a8, so a White piece on square s reads entry
// s ^ 56 (mirror the rank). Black reads its own square directly, which is
// the same entry for the vertically mirrored square.
struct Tables {
    int mg[12][64];
    int eg[12][64];
};

constexpr Tables make_tables() {
    Tables t{};
    for (int pt = 0; pt < 6; ++pt) {
        for (int s = 0; s < 64; ++s) {
            t.mg[pt][s]     = MG_VALUE[pt] + (*MG_TABLES[pt])[static_cast<std::size_t>(s ^ 56)];
            t.eg[pt][s]     = EG_VALUE[pt] + (*EG_TABLES[pt])[static_cast<std::size_t>(s ^ 56)];
            t.mg[pt + 6][s] = MG_VALUE[pt] + (*MG_TABLES[pt])[static_cast<std::size_t>(s)];
            t.eg[pt + 6][s] = EG_VALUE[pt] + (*EG_TABLES[pt])[static_cast<std::size_t>(s)];
        }
    }
    return t;
}
constexpr Tables TABLES = make_tables();

// King moves between two squares.
int king_distance(int a, int b) {
    return std::max(std::abs(file_of(a) - file_of(b)), std::abs(rank_of(a) - rank_of(b)));
}

// Mop-up: against a bare king, the side with mating material is rewarded
// for driving that king to the edge and bringing its own king closer. The
// piece-square tables do this too weakly for fast games to convert: the
// mate sits beyond the horizon, and each drawing move scores the same, so
// at 1+0.01 queen against king was drawn by the fifty-move rule. With a
// bishop and a knight, the target is a corner of the bishop's colour,
// where the mate is. (A first version fired whenever the weaker side had
// no pawns, pieces or not, and lost 7.6 Elo.)
int mop_up(const Position& pos) {
    if (!params::mop_edge && !params::mop_kings) return 0;
    for (int strong = 0; strong < 2; ++strong) {
        const Color weak = static_cast<Color>(strong ^ 1);
        if (std::popcount(pos.occupancy(weak)) != 1) continue;  // a bare king

        const int base = strong == 0 ? WP : BP;
        const Bitboard knights = pos.pieces(static_cast<Piece>(base + 1));
        const Bitboard bishops = pos.pieces(static_cast<Piece>(base + 2));
        const bool majors = pos.pieces(static_cast<Piece>(base + 3)) | pos.pieces(static_cast<Piece>(base + 4));
        // Mating material: a queen or rook, or a bishop with another minor.
        if (!majors && !(bishops && std::popcount(knights | bishops) >= 2)) continue;

        const int wk = std::countr_zero(pos.pieces(static_cast<Piece>((strong ^ 1) == 0 ? WK : BK)));
        const int sk = std::countr_zero(pos.pieces(static_cast<Piece>(base + 5)));
        int edge;
        if (!majors && !pos.pieces(static_cast<Piece>(base)) && std::popcount(bishops) == 1 &&
            std::popcount(knights) == 1) {
            // Bishop and knight: the corners of the bishop's colour (a1 is dark).
            const int b = std::countr_zero(bishops);
            const bool light = (file_of(b) + rank_of(b)) % 2 == 1;
            const int d = light ? std::min(king_distance(wk, 7), king_distance(wk, 56))
                                : std::min(king_distance(wk, 0), king_distance(wk, 63));
            edge = 7 - d;
        } else {
            // Distance from the centre: 0 on d4-e5, 6 in a corner.
            edge = 3 - std::min(file_of(wk), 7 - file_of(wk)) + 3 - std::min(rank_of(wk), 7 - rank_of(wk));
        }
        const int bonus = params::mop_edge * edge + params::mop_kings * (7 - king_distance(wk, sk));
        return static_cast<int>(pos.side_to_move()) == strong ? bonus : -bonus;
    }
    return 0;
}

// Drawishness (params::draw_ocb_pct): the percent of the eval to keep in
// endings where an edge rarely converts; 100 elsewhere.
int drawish_pct(const Position& pos) {
    if (params::draw_ocb_pct == 100 && params::draw_pawnless_pct == 100) return 100;
    int count[2][6];
    int material[2] = {0, 0};
    for (int c = 0; c < 2; ++c)
        for (int pt = 0; pt < 5; ++pt) {
            count[c][pt] = std::popcount(pos.pieces(static_cast<Piece>(c * 6 + pt)));
            material[c] += count[c][pt] * SEE_VALUE[pt];
        }
    const int pawns = count[0][0] + count[1][0];
    if (pawns == 0 && std::abs(material[0] - material[1]) < params::draw_pawnless_margin)
        return params::draw_pawnless_pct;
    // Opposite-coloured bishops and nothing else but pawns.
    bool ocb = true;
    for (int c = 0; c < 2; ++c)
        ocb = ocb && count[c][1] == 0 && count[c][2] == 1 && count[c][3] == 0 && count[c][4] == 0;
    if (ocb) {
        const int w = std::countr_zero(pos.pieces(WB)), b = std::countr_zero(pos.pieces(BB));
        if ((file_of(w) + rank_of(w) + file_of(b) + rank_of(b)) % 2 == 1) return params::draw_ocb_pct;
    }
    return 100;
}
}

Evaluator::Accumulator Evaluator::compute(const Position& pos) {
    Accumulator acc{{0, 0}, {0, 0}, 0};
    for (int p = WP; p <= BK; ++p) {
        const int side = p < BP ? 0 : 1;
        for (Bitboard b = pos.pieces(static_cast<Piece>(p)); b; b &= b - 1) {
            const int s = std::countr_zero(b);
            acc.mg[side] += TABLES.mg[p][s];
            acc.eg[side] += TABLES.eg[p][s];
            acc.phase += PHASE_INC[type_of(p)];
        }
    }
    return acc;
}

int Evaluator::score(const Accumulator& acc, Color side_to_move) {
    const int us = static_cast<int>(side_to_move), them = us ^ 1;
    const int mg_phase = acc.phase > 24 ? 24 : acc.phase;  // early promotions can exceed 24
    const int mg_score = acc.mg[us] - acc.mg[them];
    const int eg_score = acc.eg[us] - acc.eg[them];
    return (mg_score * mg_phase + eg_score * (24 - mg_phase)) / 24;
}

void Evaluator::reset(const Position& pos) {
    top_ = 0;
    stack_[0] = compute(pos);
    network_ = nnue::active();
    if (network_) {
        if (nnue_.empty()) nnue_.resize(STACK_SIZE);
        if (nnue::extra_families() && nnue_extras_.empty()) nnue_extras_.resize(STACK_SIZE);
        for (std::size_t i = 0; i < nnue_.size(); ++i)
            nnue_[i].extras = nnue::extra_families() ? &nnue_extras_[i] : nullptr;
        nnue::refresh(pos, nnue_[0]);
        cache_.reset();
    }
}

void Evaluator::push(const MoveChanges& changes) {
    // With a network the hand-written sums go unread (evaluate() returns the
    // network's score, eval_parts() computes its own): only the network's
    // accumulator moves.
    if (network_) {
        nnue::update(nnue_[static_cast<std::size_t>(top_)], nnue_[static_cast<std::size_t>(top_) + 1], changes);
        ++top_;
        return;
    }
    Accumulator acc = stack_[static_cast<std::size_t>(top_)];
    for (int i = 0; i < changes.count; ++i) {
        const PieceChange& d = changes.list[i];
        const int p = d.piece;
        const int side = p < BP ? 0 : 1;
        if (d.from >= 0) {
            acc.mg[side] -= TABLES.mg[p][d.from];
            acc.eg[side] -= TABLES.eg[p][d.from];
        } else {
            acc.phase += PHASE_INC[type_of(p)];  // piece appears (promotion)
        }
        if (d.to >= 0) {
            acc.mg[side] += TABLES.mg[p][d.to];
            acc.eg[side] += TABLES.eg[p][d.to];
        } else {
            acc.phase -= PHASE_INC[type_of(p)];  // piece leaves the board
        }
    }
    stack_[static_cast<std::size_t>(++top_)] = acc;
}

int Evaluator::evaluate(const Position& pos) const {
    // With a network loaded, it is the evaluation. Otherwise the combining
    // point for the hand-written terms: the PeSTO score, the mop-up term and
    // the initiative term, scaled toward a draw in drawish endings.
    if (nnue::active())
        return nnue::evaluate(nnue_[static_cast<std::size_t>(top_)], pos, &cache_);
    const int v = score(stack_[static_cast<std::size_t>(top_)], pos.side_to_move()) +
                  mop_up(pos) + initiative::evaluate(pos);
    const int pct = drawish_pct(pos);
    return pct == 100 ? v : v * pct / 100;
}

int evaluate(const Position& pos) {
    if (nnue::active()) return nnue::evaluate(pos);
    const int v = Evaluator::score(Evaluator::compute(pos), pos.side_to_move()) + mop_up(pos) +
                  initiative::evaluate(pos);
    const int pct = drawish_pct(pos);
    return pct == 100 ? v : v * pct / 100;
}

EvalParts eval_parts(const Position& pos) {
    EvalParts p;
    p.pesto = Evaluator::score(Evaluator::compute(pos), pos.side_to_move());
    p.initiative = initiative::evaluate(pos);
    p.mop_up = mop_up(pos);
    p.drawish_pct = drawish_pct(pos);
    const int v = p.pesto + p.mop_up + p.initiative;
    p.total = p.drawish_pct == 100 ? v : v * p.drawish_pct / 100;
    if (nnue::active()) p.total = nnue::evaluate(pos);
    return p;
}
