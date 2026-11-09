#pragma once
#include "position.h"

// Static exchange evaluation: the material outcome of the sequence of
// captures on a move's destination square, each side recapturing with its
// least valuable attacker and free to stop whenever continuing would lose
// material. X-ray attackers behind the capturing pieces join in as the
// pieces in front leave. Pins are ignored, as usual.

// Piece values for exchanges, indexed by PieceType (king last, never
// captured). Close to the search's ordering values rather than PeSTO's
// tapered ones, so thresholds read as rough centipawns. Fixed: material
// counts, features and logs use these. The exchanges below count
// params::see_piece_value instead, the search's tunable copy (equal to
// these by default). Either way the least valuable attacker is found by
// type, pawn to king, so the values should keep that order (a bishop
// below the knight would have the knight spent first: slightly wrong,
// never unsafe).
inline constexpr int SEE_VALUE[6] = {100, 320, 330, 500, 900, 0};

// True if the exchange started by `m` wins at least `threshold`
// centipawns (in params::see_piece_value) for the side making it. Castling and en passant are treated
// as winning exactly 0; promotions too unless params::see_promo is set.
bool see_at_least(const Position& pos, Move m, int threshold);

// The exchange's value itself (for logging): what see_at_least compares
// with its threshold, with the same special cases scoring 0.
int see_value(const Position& pos, Move m);
