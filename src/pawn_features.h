#pragma once
#include "types.h"
#include <cstdint>

// Unitary pawn predicates (ROADMAP, Try 5, group A1): properties of each
// pawn that depend on pawns only, one feature per predicate, colour and
// square: feature (predicate x 2 + colour) x 64 + square, colour 0 White.
// Used by `featstats` (stage 1) and, when a network declares them, by the
// network's inputs (nnue.h). The trainer has the same definitions
// (tools/trainer), on boards seen from the side to move.
//
//   isolated          no own pawn on an adjacent file
//   doubled           another own pawn on its file
//   backward          no own pawn beside or behind it on an adjacent file,
//                     and its stop square attacked by an enemy pawn
//   passed            no enemy pawn ahead on its file or an adjacent one
//   candidate         not passed, no enemy pawn ahead on its file, and at
//                     least as many own pawns beside or behind it on adjacent
//                     files as enemy pawns ahead on them
//   supported         defended by an own pawn
//   double-supported  defended by two own pawns
//   phalanx           an own pawn beside it
//   ram               an enemy pawn on its stop square
//   lever             it attacks an enemy pawn
//   sentry            it alone stops an enemy pawn on an adjacent file from
//                     being passed
namespace pawnfeat {

enum Predicate {
    Isolated, Doubled, Backward, Passed, Candidate, Supported, DoubleSupported, Phalanx, Ram, Lever,
    Sentry, COUNT
};
inline constexpr const char* NAMES[COUNT] = {"isolated", "doubled", "backward", "passed",
                                             "candidate", "supported", "double-supported",
                                             "phalanx", "ram", "lever", "sentry"};
inline constexpr int FEATURES = COUNT * 2 * 64;
inline constexpr int MAX_ACTIVE = COUNT * 16;

// Writes the active features of these pawns to `out` (room for MAX_ACTIVE),
// in increasing order; returns how many.
int features(Bitboard white_pawns, Bitboard black_pawns, std::uint16_t* out);

// The pawn-control map (Try 5, group A; the author, 2026-10-04): for each
// side, the squares its pawns attack now or can attack after advancing:
// for every pawn, the squares diagonally ahead on the adjacent files, up to
// and including the rank of the first pawn (either side's) on its own file
// ahead of it (beyond a blocker it can't advance; captures that change
// files ignored). Pawn-only, so a network caches it with the predicates.
// As network inputs (header bit 11): colour x 64 + square, 128 inputs.
inline constexpr int SPAN_FEATURES = 128;
Bitboard pawn_span(Bitboard own, Bitboard all_pawns, Color side);

// `side`'s passed pawns: no enemy pawn ahead on its file or the adjacent ones.
Bitboard passed_pawns(Bitboard own, Bitboard opp, Color side);

// Passed-pawn quality, the pawn-only part (ROADMAP, Try 5, P; the author,
// 2026-10-05). For each passed pawn, (label x 2 + colour) x 64 + square:
//   protected   an own pawn defends it
//   connected   an own passed pawn on an adjacent file, at most a rank apart
//   outside     no other pawn, either side's, within two files of it, and
//               the other side has pawns (it pulls the enemy king away)
// As network inputs (header bit 17, with the piece-dependent part in
// square_features.h): after the pawn-control map, cached with the
// predicates, 384 inputs.
inline constexpr int PASSER_LABELS = 3;
inline constexpr int PASSER_FEATURES = PASSER_LABELS * 2 * 64;
inline constexpr int PASSER_MAX = PASSER_LABELS * 16;
int passer_features(Bitboard white_pawns, Bitboard black_pawns, std::uint16_t* out);

}  // namespace pawnfeat
