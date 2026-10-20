#include "pawn_features.h"
#include "attacks.h"
#include <algorithm>
#include <bit>

namespace pawnfeat {

namespace {

Bitboard file_bb(int f) { return FILE_A_BB << f; }
Bitboard adjacent_files(int f) { return (f > 0 ? file_bb(f - 1) : 0) | (f < 7 ? file_bb(f + 1) : 0); }

// Ranks strictly ahead of (or behind) `sq`'s rank, from `c`'s side.
Bitboard ranks_ahead(Color c, int sq) {
    const int r = rank_of(sq);
    if (c == Color::White) return r == 7 ? 0 : ~Bitboard{0} << (8 * (r + 1));
    return r == 0 ? 0 : (Bitboard{1} << (8 * r)) - 1;
}
Bitboard ranks_behind(Color c, int sq) { return ranks_ahead(opposite(c), sq); }

bool sentry(Bitboard own, Bitboard opp, Color c, int s) {
    const Color o = opposite(c);
    for (Bitboard b = opp & adjacent_files(file_of(s)); b; b &= b - 1) {
        const int q = std::countr_zero(b);
        const Bitboard path = (file_bb(file_of(q)) | adjacent_files(file_of(q))) & ranks_ahead(o, q);
        if ((path & bit(s)) && !(own & path & ~bit(s))) return true;
    }
    return false;
}

}  // namespace

int features(Bitboard white_pawns, Bitboard black_pawns, std::uint16_t* out) {
    int n = 0;
    for (int ci = 0; ci < 2; ++ci) {
        const Color c = static_cast<Color>(ci), o = opposite(c);
        const Bitboard own = ci == 0 ? white_pawns : black_pawns;
        const Bitboard opp = ci == 0 ? black_pawns : white_pawns;
        const Bitboard opp_attacks = pawn_attack_set(o, opp);
        for (Bitboard b = own; b; b &= b - 1) {
            const int s = std::countr_zero(b), f = file_of(s);
            const Bitboard here = bit(s), stop = pawn_push_set(c, here);
            const Bitboard ahead = ranks_ahead(c, s), beside = RANK_BB[rank_of(s)];
            const bool passed = !(opp & (file_bb(f) | adjacent_files(f)) & ahead);
            const int defenders = std::popcount(pawn_attacks(o, s) & own);
            const Bitboard helpers = own & adjacent_files(f) & (beside | ranks_behind(c, s));
            const bool active[COUNT] = {
                !(own & adjacent_files(f)),
                (own & file_bb(f) & ~here) != 0,
                !helpers && (stop & opp_attacks) != 0,
                passed,
                !passed && !(opp & file_bb(f) & ahead) &&
                    std::popcount(helpers) >= std::popcount(opp & adjacent_files(f) & ahead),
                defenders >= 1,
                defenders >= 2,
                (own & adjacent_files(f) & beside) != 0,
                (stop & opp) != 0,
                (pawn_attacks(c, s) & opp) != 0,
                sentry(own, opp, c, s),
            };
            for (int p = 0; p < COUNT; ++p)
                if (active[p]) out[n++] = static_cast<std::uint16_t>((p * 2 + ci) * 64 + s);
        }
    }
    std::sort(out, out + n);
    return n;
}

}  // namespace pawnfeat

namespace pawnfeat {

Bitboard pawn_span(Bitboard own, Bitboard all_pawns, Color side) {
    Bitboard span = 0;
    for (Bitboard b = own; b; b &= b - 1) {
        const int s = std::countr_zero(b);
        const int f = file_of(s), r = rank_of(s);
        const int step = side == Color::White ? 1 : -1;
        // The ranks this pawn attacks: one ahead of each rank it can stand on.
        for (int q = r + step; q >= 0 && q <= 7; q += step) {
            if (f > 0) span |= bit(q * 8 + f - 1);
            if (f < 7) span |= bit(q * 8 + f + 1);
            if (all_pawns & bit(q * 8 + f)) break;  // blocked: it stops just below
        }
    }
    return span;
}

Bitboard passed_pawns(Bitboard own, Bitboard opp, Color side) {
    Bitboard passed = 0;
    for (Bitboard b = own; b; b &= b - 1) {
        const int s = std::countr_zero(b), f = file_of(s);
        if (!(opp & (file_bb(f) | adjacent_files(f)) & ranks_ahead(side, s))) passed |= bit(s);
    }
    return passed;
}

int passer_features(Bitboard white_pawns, Bitboard black_pawns, std::uint16_t* out) {
    int n = 0;
    for (int ci = 0; ci < 2; ++ci) {
        const Color c = static_cast<Color>(ci), o = opposite(c);
        const Bitboard own = ci == 0 ? white_pawns : black_pawns;
        const Bitboard opp = ci == 0 ? black_pawns : white_pawns;
        const Bitboard passed = passed_pawns(own, opp, c);
        for (Bitboard b = passed; b; b &= b - 1) {
            const int s = std::countr_zero(b), f = file_of(s), r = rank_of(s);
            const Bitboard near_ranks = RANK_BB[r] | (r > 0 ? RANK_BB[r - 1] : 0) | (r < 7 ? RANK_BB[r + 1] : 0);
            Bitboard near_files = file_bb(f) | adjacent_files(f);
            if (f > 1) near_files |= file_bb(f - 2);
            if (f < 6) near_files |= file_bb(f + 2);
            const bool active[PASSER_LABELS] = {
                (pawn_attacks(o, s) & own) != 0,
                (passed & adjacent_files(f) & near_ranks) != 0,
                opp != 0 && !((own | opp) & near_files & ~bit(s)),
            };
            for (int p = 0; p < PASSER_LABELS; ++p)
                if (active[p]) out[n++] = static_cast<std::uint16_t>((p * 2 + ci) * 64 + s);
        }
    }
    std::sort(out, out + n);
    return n;
}

}  // namespace pawnfeat
