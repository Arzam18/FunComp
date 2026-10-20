#include "guide_features.h"
#include "attacks.h"
#include "movegen.h"
#include "see.h"
#include <algorithm>
#include <bit>
#include <cstdlib>

namespace features {

namespace {
constexpr int PHASE_INC[6] = {0, 1, 1, 2, 4, 0};

Bitboard pieces_of(const Position& pos, Color c, int pt) {
    return pos.pieces(static_cast<Piece>(piece_index(c, static_cast<PieceType>(pt))));
}

// Most valuable piece of `c` (king aside) that is attacked by a cheaper
// piece, or attacked and not defended.
int threatened(const Position& pos, Color c) {
    const Color them = opposite(c);
    int worst = 0;
    for (int pt = 0; pt < 5; ++pt) {
        for (Bitboard b = pieces_of(pos, c, pt); b; b &= b - 1) {
            const int s = std::countr_zero(b);
            const Bitboard attackers = pos.attackers_to(s, pos.occupied());
            const Bitboard theirs = attackers & pos.occupancy(them);
            if (!theirs) continue;
            // The king counts as the most expensive attacker: it can only
            // take an undefended piece.
            int cheapest = 1 << 20;
            for (int a = 0; a < 5; ++a)
                if (theirs & pieces_of(pos, them, a)) {
                    cheapest = SEE_VALUE[a];
                    break;
                }
            const bool defended = (attackers & pos.occupancy(c)) != 0;
            if (cheapest < SEE_VALUE[pt] || !defended) worst = std::max(worst, SEE_VALUE[pt]);
        }
    }
    return worst;
}

// Number of pieces of the other side attacking `c`'s king or a square next
// to it.
int king_attackers(const Position& pos, Color c) {
    const Bitboard king = pieces_of(pos, c, static_cast<int>(PieceType::King));
    if (!king) return 0;
    const int ksq = std::countr_zero(king);
    Bitboard attackers = 0;
    for (Bitboard zone = king_attacks(ksq) | king; zone; zone &= zone - 1)
        attackers |= pos.attackers_to(std::countr_zero(zone), pos.occupied());
    return std::popcount(attackers & pos.occupancy(opposite(c)));
}

// Pawns of `c` with no enemy pawn ahead of them on their own or an
// adjacent file.
int passed_pawns(const Position& pos, Color c) {
    const Bitboard enemy = pieces_of(pos, opposite(c), static_cast<int>(PieceType::Pawn));
    int n = 0;
    for (Bitboard b = pieces_of(pos, c, static_cast<int>(PieceType::Pawn)); b; b &= b - 1) {
        const int s = std::countr_zero(b);
        bool passed = true;
        for (Bitboard e = enemy; e && passed; e &= e - 1) {
            const int t = std::countr_zero(e);
            const bool ahead = c == Color::White ? rank_of(t) > rank_of(s) : rank_of(t) < rank_of(s);
            if (ahead && std::abs(file_of(t) - file_of(s)) <= 1) passed = false;
        }
        n += passed;
    }
    return n;
}
}  // namespace

Vector extract(const Position& pos) {
    const Color us = pos.side_to_move();
    const Color them = opposite(us);

    int phase = 0;
    int material[2] = {0, 0}, pieces[2] = {0, 0}, pawns[2] = {0, 0};
    for (const Color c : {Color::White, Color::Black}) {
        const auto ci = static_cast<std::size_t>(c);
        for (int pt = 0; pt < 5; ++pt) {
            const int n = std::popcount(pieces_of(pos, c, pt));
            phase += n * PHASE_INC[pt];
            material[ci] += n * SEE_VALUE[pt];
            if (pt == 0) pawns[ci] = n;
            else pieces[ci] += n * SEE_VALUE[pt];
        }
    }

    int best_capture = 0;
    MoveList moves;
    int count = 0;
    generate_captures(pos, moves, count);
    for (int i = 0; i < count; ++i) {
        const Move m = moves[static_cast<std::size_t>(i)];
        const Piece victim = m.is_en_passant() ? WP : pos.piece_at(m.to());
        if (victim != NO_PIECE && see_at_least(pos, m, 0))
            best_capture = std::max(best_capture, SEE_VALUE[type_of(victim)]);
    }

    const auto u = static_cast<std::size_t>(us), t = static_cast<std::size_t>(them);
    Vector v{};
    v[0] = static_cast<float>(std::min(phase, 24));
    v[1] = static_cast<float>(material[u] - material[t]);
    v[2] = static_cast<float>(pieces[u]);
    v[3] = static_cast<float>(pieces[t]);
    v[4] = static_cast<float>(pawns[u]);
    v[5] = static_cast<float>(pawns[t]);
    v[6] = static_cast<float>(threatened(pos, us));
    v[7] = static_cast<float>(threatened(pos, them));
    v[8] = static_cast<float>(best_capture);
    v[9] = static_cast<float>(king_attackers(pos, us));
    v[10] = static_cast<float>(king_attackers(pos, them));
    v[11] = static_cast<float>(passed_pawns(pos, us) - passed_pawns(pos, them));
    v[12] = static_cast<float>(count_legal(pos));
    v[13] = static_cast<float>(pos.halfmove_clock());
    return v;
}

}  // namespace features
