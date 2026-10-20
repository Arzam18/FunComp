#pragma once
#include "position.h"
#include <cstdint>

// Pair features (ROADMAP, Try 5, the shared framework): a relation between
// two pieces as one input per square pair. When a network declares a family
// (nnue.h), its inputs follow the pawn predicates, in the order below, one
// block shared by every king bucket; the accumulators keep each view's
// active pair inputs and catch up on them at evaluation, adding and removing
// only the difference (one path for every family). The trainer has the same
// definitions (tools/trainer). Square pairs are numbered by their lower
// square, then their higher one, from a1, in the view's orientation and
// mirroring; "own" and "other" are from the view's side.
//
//   E, knight pairs (feature set bit 1): two same-colour knights a knight's
//     move apart (each defends the other). (the pair is the other side's)
//     x 168 + pair: 336 inputs.
//   B, slider pairs (bit 2): two sliders sharing a line both move along,
//     blockers ignored: rooks and queens on a rank or file (448 square
//     pairs), bishops and queens on a diagonal (280). Per line kind, ((colour
//     of the lower square's slider x 2 + the higher's) x 4 + (lower is a
//     queen) x 2 + (higher is a queen)) x pairs + pair, colour 0 own;
//     orthogonal block first: 11,648 inputs.
//   C, slider x pawn (bit 3): each rook or queen with each pawn (either
//     colour) on its file. ((slider is the other side's) x 2 + (slider is a
//     queen)) x 2 + (pawn is the other side's), x 336 + the (slider square,
//     pawn square) pair, numbered by slider square, then pawn square: 2,688
//     inputs.
//   D, connected slider pairs (bit 8): slider pairs (B) with nothing
//     between the two; numbered as B: 11,648 inputs. Any piece crossing
//     the line toggles one, so they change as often as threats.
//   L, line control (bit 12; the author, 2026-10-05: "we control the open
//     e-file" as an input, one per line and state rather than per square
//     pair, so it is learned from every position where it occurs). In the
//     view's coordinates, presence encoded for each side (the network
//     weighs who dominates):
//       files: file x 16 + pawn state x 4 + majors, pawn state 0 open, 1
//         only the other side's pawns (half-open for us), 2 only ours, 3
//         both; majors (our rook or queen on it) x 2 + (theirs); a file
//         counts when it has a major or is open: 128 inputs;
//       ranks: 128 + rank x 4 + majors (as above), when it has a major: 32;
//       diagonals of length 3 or more (11 a1-h8-wise, file - rank + 7 in
//         2..12, then 11 a8-h1-wise, file + rank in 2..12): 160 +
//         diagonal x 8 + ((our bishop or queen) x 2 + (theirs)) x 2 +
//         (no pawn on it), when it has a bishop or queen: 176.
//     336 inputs, at most 38 active; they change when a pawn or slider
//     moves.
//
// A family with more active pairs in a position than its bound (possible
// only with many promoted pieces) contributes none there: all or nothing,
// whatever the order pairs are found in, so engine and trainer agree.
namespace pairfeat {

enum Family : unsigned { KnightPairs = 2, SliderPairs = 4, SliderPawn = 8, ConnectedPairs = 256, LineControl = 4096 };
inline constexpr unsigned ALL = KnightPairs | SliderPairs | SliderPawn | ConnectedPairs | LineControl;
inline constexpr int LINE_FEATURES = 336;

inline constexpr int KNIGHT_PAIRS = 168;
inline constexpr int KNIGHT_FEATURES = 2 * KNIGHT_PAIRS;
inline constexpr int ORTHO_PAIRS = 448;
inline constexpr int DIAG_PAIRS = 280;
inline constexpr int SLIDER_FEATURES = 16 * (ORTHO_PAIRS + DIAG_PAIRS);
inline constexpr int FILE_PAIRS = 336;
inline constexpr int SLIDER_PAWN_FEATURES = 8 * FILE_PAIRS;
// Active inputs of one family in one view at most (knight pairs can't pass
// it: knights a knight's move apart stand on opposite square colours, so
// ten knights of a side make at most 25 pairs).
inline constexpr int FAMILY_MAX = 64;
inline constexpr int MAX_ACTIVE = 5 * FAMILY_MAX;

// The index of the unordered pair {a, b} (lower square first in the
// numbering), or -1 if not a knight's move apart / not on one rank or file /
// not on one diagonal.
int knight_pair_index(int a, int b);
int ortho_pair_index(int a, int b);
int diag_pair_index(int a, int b);
// The index of (slider square, pawn square): same file, pawn on ranks 2-7,
// different squares; else -1.
int file_pair_index(int slider, int pawn);
// The squares of knight pair `index`, lower first.
void knight_pair_squares(int index, int& a, int& b);

// Every knight pair, as colour x 168 + index (colour 0 White), ascending.
int knight_pairs(Bitboard white, Bitboard black, std::uint16_t* out);

// Inputs in the pair block of the families in `families` (Family bits).
int block_size(unsigned families);

// The active pair inputs of `families` from `view` (its squares mirrored
// left-right when flip is 7), as offsets into the pair block, ascending.
int inputs(const Position& pos, unsigned families, Color view, int flip, std::uint32_t* out);

}  // namespace pairfeat
