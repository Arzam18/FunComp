#include "see.h"
#include "attacks.h"
#include "params.h"
#include <algorithm>
#include <bit>

namespace {
bool is_castling(const Position& pos, Move m) {
    const Piece p = pos.piece_at(m.from());
    return (p == WK || p == BK) && (m.to() - m.from() == 2 || m.from() - m.to() == 2);
}

// Material won by the side to move through the exchange that `m` starts,
// by the gain-array swap algorithm (Chess Programming Wiki, "SEE - The
// Swap Algorithm"). The captures are first played out to the end with
// each side's least valuable attacker, recording the running material
// balance. A minimax pass from the end then lets each side stop at the
// best point for it. The whole sequence is always played out: exchanges
// are short, and stopping early is easy to get subtly wrong.
int see(const Position& pos, Move m) {
    const int to = m.to();
    const Piece victim = pos.piece_at(to);
    // Every pawn capture onto the first or last rank promotes (to a queen,
    // for the recaptures); the promoting side gains the difference.
    const bool promo_square = params::see_promo && (rank_of(to) == 0 || rank_of(to) == 7);
    // The search's values (params.h); the pawn is never tuned, so its
    // constant serves where only the pawn can stand.
    const auto& value = params::see_piece_value;

    // gain[d]: material for the side making capture d (the move itself is
    // capture 0), if the exchange stops right after it.
    int gain[32];
    int d = 0;
    gain[0] = victim == NO_PIECE ? 0 : value[type_of(victim)];
    int on_square = value[type_of(pos.piece_at(m.from()))];  // piece now standing on `to`
    if (m.promotion() != PieceType::None) {
        const int promoted = value[static_cast<int>(m.promotion())];
        gain[0] += promoted - SEE_VALUE[0];
        on_square = promoted;
    }

    Bitboard occupied = pos.occupied() ^ bit(m.from());
    Bitboard attackers = pos.attackers_to(to, occupied) & occupied;
    const Bitboard diagonal = pos.pieces(WB) | pos.pieces(BB) | pos.pieces(WQ) | pos.pieces(BQ);
    const Bitboard orthogonal = pos.pieces(WR) | pos.pieces(BR) | pos.pieces(WQ) | pos.pieces(BQ);
    Color side = opposite(pos.side_to_move());

    while (d < 31) {
        const Bitboard ours = attackers & pos.occupancy(side);
        if (!ours) break;

        // Least valuable attacker.
        int pt = 0;
        Bitboard from_set = 0;
        for (; pt < 6; ++pt) {
            from_set = ours & pos.pieces(static_cast<Piece>(piece_index(side, static_cast<PieceType>(pt))));
            if (from_set) break;
        }

        occupied ^= bit(std::countr_zero(from_set));
        // Sliders lined up behind the piece that just left join in. A pawn,
        // bishop or queen can only have hidden a diagonal slider, a rook or
        // queen an orthogonal one; a knight or king hides nothing.
        if (pt == 0 || pt == 2 || pt == 4) attackers |= bishop_attacks(to, occupied) & diagonal;
        if (pt == 3 || pt == 4) attackers |= rook_attacks(to, occupied) & orthogonal;
        attackers &= occupied;

        // A king may only capture onto a square the other side no longer
        // attacks, and nothing can answer it; either way the exchange ends.
        if (pt == static_cast<int>(PieceType::King)) {
            if (!(attackers & pos.occupancy(opposite(side)))) {
                ++d;
                gain[d] = on_square - gain[d - 1];
            }
            break;
        }

        ++d;
        gain[d] = on_square - gain[d - 1];
        on_square = value[pt];
        if (pt == 0 && promo_square) {
            gain[d] += value[4] - SEE_VALUE[0];
            on_square = value[4];
        }
        side = opposite(side);
    }

    while (d > 0) {
        gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
        --d;
    }
    return gain[0];
}
}  // namespace

bool see_at_least(const Position& pos, Move m, int threshold) {
    const PieceType promo = m.promotion();
    if (m.is_en_passant() || is_castling(pos, m) || (promo != PieceType::None && !params::see_promo))
        return 0 >= threshold;

    // The exchange wins at most the victim, plus what a promotion adds (the
    // other side may decline to recapture), and at least that less the
    // piece left on the square (we may stop after one recapture), less a
    // promotion's worth more on our own back rank, where the other side's
    // first recapture can be a pawn promoting. Most calls are settled by
    // these bounds.
    const auto& value = params::see_piece_value;
    const Piece victim = pos.piece_at(m.to());
    int most = victim == NO_PIECE ? 0 : value[type_of(victim)];
    int on_square = value[type_of(pos.piece_at(m.from()))];
    if (promo != PieceType::None) {
        most += value[static_cast<int>(promo)] - SEE_VALUE[0];
        on_square = value[static_cast<int>(promo)];
    }
    const int own_back_rank = pos.side_to_move() == Color::White ? 0 : 7;
    const bool enemy_promotes = params::see_promo && rank_of(m.to()) == own_back_rank;
    const int least = most - on_square - (enemy_promotes ? value[4] - SEE_VALUE[0] : 0);
    if (most < threshold) return false;
    if (least >= threshold) return true;
    return see(pos, m) >= threshold;
}

int see_value(const Position& pos, Move m) {
    if (m.is_en_passant() || is_castling(pos, m) ||
        (m.promotion() != PieceType::None && !params::see_promo))
        return 0;
    return see(pos, m);
}
