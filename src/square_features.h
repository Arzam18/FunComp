#pragma once
#include "position.h"
#include <cstdint>

// Threats and square control (ROADMAP, NNUE, "Threats and square control,
// two input families"; the author, 2026-10-04): what the pieces attack,
// as network inputs. When a network declares a family (nnue.h), its
// inputs follow the pair families (pair_features.h), in the order below,
// and go the same way: each accumulator keeps its views' active inputs and
// evaluation adds and removes only the difference. Computed from the board
// at each evaluation: correct, not yet fast; the roadmap tests these at
// fixed nodes first (do they add knowledge?) and makes them incremental
// only if they do. The trainer has the same definitions (tools/trainer).
// Squares in the view's orientation and mirroring; "own" and "other" from
// the view's side; piece types P N B R Q K = 0-5.
//
//   Bit 4 is retired: family T, every piece attacking every occupied square,
//     mirrored Stockfish 19's threat inputs (tools/nnue/originality-audit.md,
//     R2) and was never in a net that shipped; a network declaring it is
//     refused at load. Threat knowledge by chess logic is family W, on its
//     own branch (family-w).
//   S1, attacked (bit 5): squares each side attacks: colour x 64 + square,
//     128 inputs.
//   S2, least valuable attacker (bit 6): for each square a side attacks,
//     its cheapest attacker's type: (colour x 64 + square) x 6 + type, 768.
//   S3, king zone (bit 7): squares a side attacks in the other king's zone
//     (the king's square, its neighbours, and the squares in front of
//     those from that king's side): colour x 64 + square, 128.
//   A2, piece-dependent pawn predicates (bit 9; ROADMAP, Try 5): for each
//     pawn, (predicate x 2 + its colour) x 64 + its square, 512 inputs:
//     supported by a piece (an own piece other than a pawn attacks it),
//     pinned support only (own pawns defend it, all of them pinned to
//     their king), attacked by a piece (an enemy piece other than a pawn
//     attacks it), blocked by a piece (a piece other than a pawn, either
//     side's, on its stop square).
//   C, control counts (bit 10; the author's "negative board", 2026-10-04):
//     how many times each side attacks each square, occupied or empty, as
//     a thermometer: ((colour x 64 + square) x 3 + level), level 0 for at
//     least one attacker, 1 for two, 2 for three or more: 384 inputs. A
//     generalization of S1 (its level 0); every attack counted with the
//     board's blockers (a battery's rear piece doesn't count through).
//   M, mobility (bit 15; the author, 2026-10-05): each knight, bishop, rook
//     and queen's safe moves (squares it attacks, with the board's
//     blockers, not occupied by its own side and not attacked by an enemy
//     pawn; pins ignored), as a thermometer per piece type and side: the
//     input (colour, level, type, k) is on when at least level + 1 of that
//     side's pieces of that type have k or more safe moves, k from 1 to
//     the type's cap (N 8, B 13, R 14, Q 24): (colour x 2 + level) x 59 +
//     type's first + k - 1, 236 inputs. No squares: the network decides
//     where "congested" ends for each piece type.
//   P, passed-pawn quality, the piece-dependent part (bit 17, with the
//     pawn-only part in pawn_features.h; ROADMAP, Try 5, P; the author,
//     2026-10-05): for each passed pawn, (label x 2 + its colour) x 64 + its
//     square: 0 blockaded by an enemy knight (on its stop square), 1 by an
//     enemy bishop, 2 by any other piece; 3 path clear (every square ahead
//     to promotion empty); 4 path safe (none of them attacked by the other
//     side, blockers counted); 5 an own rook or queen behind it on its file
//     (the first piece behind), 6 an enemy one; 7 unstoppable (a pawn
//     ending, path clear, the enemy king outside its square, the side to
//     move counted): 1,024 inputs. Only passed pawns, so usually a few.
//
// A family past its bound in a view contributes none there (all or nothing,
// as for the pair families).
namespace sqfeat {

enum Family : unsigned {
    Attacked = 32, LeastAttacker = 64, KingZone = 128, PawnPieces = 512, ControlCounts = 1024, Mobility = 32768,
    Passers = 131072
};
inline constexpr unsigned RETIRED_THREATS = 16;  // bit 4: refused (above)
inline constexpr unsigned ALL = Attacked | LeastAttacker | KingZone | PawnPieces | ControlCounts | Mobility | Passers;

inline constexpr int ATTACKED_FEATURES = 128;
inline constexpr int LEAST_ATTACKER_FEATURES = 768;
inline constexpr int KING_ZONE_FEATURES = 128;
inline constexpr int PAWN_PIECE_FEATURES = 512;
inline constexpr int CONTROL_COUNT_FEATURES = 384;
inline constexpr int CONTROL_COUNT_MAX = 384;  // can't pass it
inline constexpr int MOBILITY_STEPS = 8 + 13 + 14 + 24;  // N B R Q
inline constexpr int MOBILITY_FEATURES = 4 * MOBILITY_STEPS;
inline constexpr int MOBILITY_MAX = MOBILITY_FEATURES;  // can't pass it
inline constexpr int PASSER_PIECE_LABELS = 8;
inline constexpr int PASSER_PIECE_FEATURES = PASSER_PIECE_LABELS * 2 * 64;
inline constexpr int SQUARE_MAX = 128;  // S1-S3, A2 and P can't pass it
inline constexpr int MAX_ACTIVE = 5 * SQUARE_MAX + CONTROL_COUNT_MAX + MOBILITY_MAX;

// Inputs in the block of the families in `families` (Family bits).
int block_size(unsigned families);

// The active inputs of `families` from `view` (its squares mirrored
// left-right when flip is 7), as offsets into their block, ascending.
int inputs(const Position& pos, unsigned families, Color view, int flip, std::uint32_t* out);

}  // namespace sqfeat
