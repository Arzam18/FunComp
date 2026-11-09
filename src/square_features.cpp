#include "square_features.h"
#include "attacks.h"
#include "pawn_features.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>

namespace sqfeat {

namespace {

int own_square(Color view, int square) { return view == Color::White ? square : square ^ 56; }

Piece piece_of(Color c, int type) { return static_cast<Piece>(piece_index(c, static_cast<PieceType>(type))); }

// A side's attacks by piece type, with the board's blockers.
std::array<Bitboard, 6> attacks_by_type(const Position& pos, Color c) {
    std::array<Bitboard, 6> by{};
    const Bitboard occ = pos.occupied();
    by[0] = pawn_attack_set(c, pos.pieces(piece_of(c, 0)));
    for (int t = 1; t <= 5; ++t)
        for (Bitboard b = pos.pieces(piece_of(c, t)); b; b &= b - 1) {
            const int s = std::countr_zero(b);
            by[static_cast<std::size_t>(t)] |= t == 1   ? knight_attacks(s)
                                               : t == 2 ? bishop_attacks(s, occ)
                                               : t == 3 ? rook_attacks(s, occ)
                                               : t == 4 ? queen_attacks(s, occ)
                                                        : king_attacks(s);
        }
    return by;
}

Bitboard piece_attacks(int type, Color c, int s, Bitboard occ) {
    switch (type) {
        case 0: return pawn_attack_set(c, bit(s));
        case 1: return knight_attacks(s);
        case 2: return bishop_attacks(s, occ);
        case 3: return rook_attacks(s, occ);
        case 4: return queen_attacks(s, occ);
        default: return king_attacks(s);
    }
}

// Pieces of `c` other than pawns attacking `s`.
Bitboard piece_attackers(const Position& pos, Color c, int s) {
    const Bitboard occ = pos.occupied();
    const Bitboard diag = pos.pieces(piece_of(c, 2)) | pos.pieces(piece_of(c, 4));
    const Bitboard orth = pos.pieces(piece_of(c, 3)) | pos.pieces(piece_of(c, 4));
    return (knight_attacks(s) & pos.pieces(piece_of(c, 1))) | (bishop_attacks(s, occ) & diag) |
           (rook_attacks(s, occ) & orth) | (king_attacks(s) & pos.pieces(piece_of(c, 5)));
}

// `c`'s pieces pinned to their king by an enemy slider.
Bitboard pinned(const Position& pos, Color c) {
    const Bitboard king = pos.pieces(piece_of(c, 5));
    if (!king) return 0;
    const int k = std::countr_zero(king);
    const Color o = opposite(c);
    const Bitboard occ = pos.occupied();
    const Bitboard diag = pos.pieces(piece_of(o, 2)) | pos.pieces(piece_of(o, 4));
    const Bitboard orth = pos.pieces(piece_of(o, 3)) | pos.pieces(piece_of(o, 4));
    Bitboard result = 0;
    for (Bitboard b = (bishop_attacks(k, 0) & diag) | (rook_attacks(k, 0) & orth); b; b &= b - 1) {
        const Bitboard between = squares_between(k, std::countr_zero(b)) & occ;
        if (std::popcount(between) == 1 && (between & pos.occupancy(c))) result |= between;
    }
    return result;
}

}  // namespace

int block_size(unsigned families) {
    return (families & Attacked ? ATTACKED_FEATURES : 0) +
           (families & LeastAttacker ? LEAST_ATTACKER_FEATURES : 0) + (families & KingZone ? KING_ZONE_FEATURES : 0) +
           (families & PawnPieces ? PAWN_PIECE_FEATURES : 0) +
           (families & ControlCounts ? CONTROL_COUNT_FEATURES : 0) + (families & Mobility ? MOBILITY_FEATURES : 0) +
           (families & Passers ? PASSER_PIECE_FEATURES : 0);
}

int inputs(const Position& pos, unsigned families, Color view, int flip, std::uint32_t* out) {
    int n = 0, offset = 0;
    const int us = static_cast<int>(view);
    const auto sq = [&](int s) { return own_square(view, s) ^ flip; };
    const auto family = [&](int bound, auto&& each) {
        const int first = n;
        bool full = false;
        each([&](int input) {
            if (n - first == bound) full = true;
            else out[n++] = static_cast<std::uint32_t>(offset + input);
        });
        if (full) n = first;
    };
    const Bitboard occ = pos.occupied();

    if (families & (Attacked | LeastAttacker | KingZone)) {
        std::array<std::array<Bitboard, 6>, 2> by;
        std::array<Bitboard, 2> all{};
        for (int c = 0; c < 2; ++c) {
            by[static_cast<std::size_t>(c)] = attacks_by_type(pos, static_cast<Color>(c));
            for (const Bitboard b : by[static_cast<std::size_t>(c)]) all[static_cast<std::size_t>(c)] |= b;
        }
        if (families & Attacked) {
            family(SQUARE_MAX, [&](auto&& add) {
                for (int c = 0; c < 2; ++c)
                    for (Bitboard b = all[static_cast<std::size_t>(c)]; b; b &= b - 1)
                        add((c != us) * 64 + sq(std::countr_zero(b)));
            });
            offset += ATTACKED_FEATURES;
        }
        if (families & LeastAttacker) {
            family(SQUARE_MAX, [&](auto&& add) {
                for (int c = 0; c < 2; ++c) {
                    Bitboard done = 0;
                    for (int t = 0; t < 6; ++t) {
                        const Bitboard b0 = by[static_cast<std::size_t>(c)][static_cast<std::size_t>(t)];
                        for (Bitboard b = b0 & ~done; b; b &= b - 1)
                            add(((c != us) * 64 + sq(std::countr_zero(b))) * 6 + t);
                        done |= b0;
                    }
                }
            });
            offset += LEAST_ATTACKER_FEATURES;
        }
        if (families & KingZone) {
            family(SQUARE_MAX, [&](auto&& add) {
                for (int c = 0; c < 2; ++c) {
                    const Color o = opposite(static_cast<Color>(c));
                    const Bitboard king = pos.pieces(piece_of(o, 5));
                    if (!king) continue;
                    const Bitboard ring = king_attacks(std::countr_zero(king)) | king;
                    const Bitboard zone = ring | pawn_push_set(o, ring);
                    for (Bitboard b = all[static_cast<std::size_t>(c)] & zone; b; b &= b - 1)
                        add((c != us) * 64 + sq(std::countr_zero(b)));
                }
            });
            offset += KING_ZONE_FEATURES;
        }
    }

    if (families & PawnPieces) {
        family(SQUARE_MAX, [&](auto&& add) {
            const Bitboard pawns = pos.pieces(WP) | pos.pieces(BP);
            for (int c = 0; c < 2; ++c) {
                const Color cc = static_cast<Color>(c), o = opposite(cc);
                const Bitboard own = pos.pieces(piece_of(cc, 0));
                const Bitboard pins = pinned(pos, cc);
                for (Bitboard b = own; b; b &= b - 1) {
                    const int s = std::countr_zero(b);
                    const Bitboard defenders = pawn_attack_set(o, bit(s)) & own;
                    const bool on[4] = {
                        piece_attackers(pos, cc, s) != 0,
                        defenders && !(defenders & ~pins),
                        piece_attackers(pos, o, s) != 0,
                        (pawn_push_set(cc, bit(s)) & occ & ~pawns) != 0,
                    };
                    for (int p = 0; p < 4; ++p)
                        if (on[p]) add((p * 2 + (c != us)) * 64 + sq(s));
                }
            }
        });
        offset += PAWN_PIECE_FEATURES;
    }

    if (families & ControlCounts) {
        family(CONTROL_COUNT_MAX, [&](auto&& add) {
            for (int c = 0; c < 2; ++c) {
                // A saturating count of attackers per square, bit-sliced:
                // level[k] holds the squares attacked at least k + 1 times.
                std::array<Bitboard, 3> level{};
                const auto count = [&](Bitboard b) {
                    level[2] |= level[1] & b;
                    level[1] |= level[0] & b;
                    level[0] |= b;
                };
                const Color cc = static_cast<Color>(c);
                const Bitboard pawns = pos.pieces(piece_of(cc, 0));
                // The two diagonals separately: two pawns may hit one square.
                count(c == 0 ? (pawns & ~FILE_A_BB) << 7 : (pawns & ~FILE_A_BB) >> 9);
                count(c == 0 ? (pawns & ~FILE_H_BB) << 9 : (pawns & ~FILE_H_BB) >> 7);
                for (int t = 1; t <= 5; ++t)
                    for (Bitboard b = pos.pieces(piece_of(cc, t)); b; b &= b - 1)
                        count(piece_attacks(t, cc, std::countr_zero(b), occ));
                for (int k = 0; k < 3; ++k)
                    for (Bitboard b = level[static_cast<std::size_t>(k)]; b; b &= b - 1)
                        add(((c != us) * 64 + sq(std::countr_zero(b))) * 3 + k);
            }
        });
        offset += CONTROL_COUNT_FEATURES;
    }

    if (families & Mobility) {
        family(MOBILITY_MAX, [&](auto&& add) {
            constexpr std::array<int, 4> first{0, 8, 21, 35}, cap{8, 13, 14, 24};
            for (int c = 0; c < 2; ++c) {
                const Color cc = static_cast<Color>(c), o = opposite(cc);
                const Bitboard safe = ~pos.occupancy(cc) & ~pawn_attack_set(o, pos.pieces(piece_of(o, 0)));
                for (int t = 1; t <= 4; ++t) {
                    // Saturating counts per step, bit-sliced: bit k - 1 of
                    // level[l] is on when at least l + 1 pieces have k.
                    std::array<std::uint32_t, 2> level{};
                    const auto tt = static_cast<std::size_t>(t - 1);
                    for (Bitboard b = pos.pieces(piece_of(cc, t)); b; b &= b - 1) {
                        const Bitboard moves = piece_attacks(t, cc, std::countr_zero(b), occ) & safe;
                        const std::uint32_t reach = (1u << std::min(std::popcount(moves), cap[tt])) - 1;
                        level[1] |= level[0] & reach;
                        level[0] |= reach;
                    }
                    for (int l = 0; l < 2; ++l)
                        for (std::uint32_t b = level[static_cast<std::size_t>(l)]; b; b &= b - 1)
                            add(((c != us) * 2 + l) * MOBILITY_STEPS + first[tt] + std::countr_zero(b));
                }
            }
        });
        offset += MOBILITY_FEATURES;
    }

    if (families & Passers) {
        family(SQUARE_MAX, [&](auto&& add) {
            const Bitboard wp = pos.pieces(WP), bp = pos.pieces(BP);
            const bool pawn_ending = !(occ & ~(wp | bp | pos.pieces(WK) | pos.pieces(BK)));
            for (int c = 0; c < 2; ++c) {
                const Color cc = static_cast<Color>(c), o = opposite(cc);
                const Bitboard passed = pawnfeat::passed_pawns(c == 0 ? wp : bp, c == 0 ? bp : wp, cc);
                if (!passed) continue;
                Bitboard enemy_attacks = 0;
                for (const Bitboard b : attacks_by_type(pos, o)) enemy_attacks |= b;
                const Bitboard own_majors = pos.pieces(piece_of(cc, 3)) | pos.pieces(piece_of(cc, 4));
                const Bitboard opp_majors = pos.pieces(piece_of(o, 3)) | pos.pieces(piece_of(o, 4));
                const Bitboard opp_king = pos.pieces(piece_of(o, 5));
                for (Bitboard b = passed; b; b &= b - 1) {
                    const int s = std::countr_zero(b), f = s & 7, own_rank = c == 0 ? s >> 3 : 7 - (s >> 3);
                    const Bitboard file = FILE_A_BB << f;
                    const Bitboard ahead = file & (c == 0 ? ~Bitboard{0} << (s + 1) : bit(s) - 1);
                    const Bitboard behind = file & ~ahead & ~bit(s);
                    const Bitboard stop = pawn_push_set(cc, bit(s));
                    const Piece blocker = stop & occ ? pos.piece_at(std::countr_zero(stop)) : NO_PIECE;
                    const Bitboard back = rook_attacks(s, occ) & behind & occ;  // the first piece behind
                    bool unstoppable = false;
                    if (pawn_ending && !(ahead & occ) && opp_king) {
                        const int promo = c == 0 ? 56 + f : f, k = std::countr_zero(opp_king);
                        const int steps = 7 - own_rank - (own_rank == 1 ? 1 : 0);
                        const int distance = std::max(std::abs((k & 7) - f), std::abs((k >> 3) - (promo >> 3)));
                        unstoppable = distance - (pos.side_to_move() == o ? 1 : 0) > steps;
                    }
                    const bool on[PASSER_PIECE_LABELS] = {
                        blocker == piece_of(o, 1),
                        blocker == piece_of(o, 2),
                        blocker != NO_PIECE && blocker != piece_of(o, 1) && blocker != piece_of(o, 2),
                        !(ahead & occ),
                        !(ahead & enemy_attacks),
                        (back & own_majors) != 0,
                        (back & opp_majors) != 0,
                        unstoppable,
                    };
                    for (int p = 0; p < PASSER_PIECE_LABELS; ++p)
                        if (on[p]) add((p * 2 + (c != us)) * 64 + sq(s));
                }
            }
        });
        offset += PASSER_PIECE_FEATURES;
    }

    std::sort(out, out + n);
    return n;
}

}  // namespace sqfeat
