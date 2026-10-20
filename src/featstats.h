#pragma once
#include <cstdint>
#include <iosfwd>
#include <string>

// Try 5's stage 1, and the square-control families (ROADMAP, NNUE: Try 5
// and "Threats and square control"): candidate input-feature families for
// later networks, measured on played games before any training. For each
// family: its possible features, the distinct ones seen, active features per
// position, the share of positions with any active, and features changed per
// move (the update cost an incremental accumulator would pay), from the full
// feature sets of consecutive searched positions.
//
// Families (features in absolute terms: colour and square; perspective and
// mirroring come with the network):
//   A1 pawn-only predicates of each pawn: isolated, doubled, backward,
//      passed, candidate, supported, double supported, phalanx, ram, lever,
//      sentry (alone stops an adjacent enemy pawn from being passed)
//   A2 piece-dependent predicates of each pawn: supported by a piece,
//      supported only by pinned pawns, attacked by a piece, blocked by a
//      piece
//   E  knight pairs: same-colour knights a knight's move apart
//   B  geometric slider pairs: two sliders on a line both move along (rooks
//      and queens on a rank or file, bishops and queens on a diagonal),
//      any colours, blockers ignored
//   C  each rook or queen with each pawn (either colour) on its file
//   D  connected slider pairs: B with nothing between them
//   S1 squares attacked, by side
//   S2 each attacked square's least valuable attacker, by side
//   S3 squares attacked in the enemy king's zone, by side
//
// The "featstats" UCI command: featstats in=<data file or directory> [games=N]
void feature_stats(const std::string& path, std::uint64_t max_games, std::ostream& out);
