#pragma once
#include "position.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// The neural network evaluation (NNUE), when a network is loaded (UCI
// option EvalFile); without one the hand-written evaluation (eval.h) is
// used and nothing here runs.
//
// Architecture: (768 x KB -> H) x 2 -> OB. Inputs are the pieces on their
// squares, seen from each side: from a side's view, feature
// 384 x (the piece is the other side's) + 64 x piece type + square, the
// square mirrored vertically for Black. King buckets (KB of them) give each
// view its own block of 768 inputs, chosen by that side's king square (in
// its own view) through the network's bucket layout; with mirroring, a king
// on files e-h flips that view's squares left-right (square ^ 7), so the
// layout covers files a-d only. One first layer (weights shared by both
// views) turns each side's features into H hidden values, kept up to date
// incrementally as moves are made: the accumulators. A king move that
// changes its side's bucket or mirroring marks that side's view stale, to
// be rebuilt at the next evaluation. The output layer reads both views, the
// side to move's first, through a squared clipped ReLU (clamp to [0, QA],
// squared), with one set of output weights per output bucket (OB of them,
// by the number of pieces on the board).
//
// File format: an optional header (ours; without one, KB = OB = 1), then
// bullet's quantised output, little-endian int16: first-layer weights
// (768 x KB columns of H, scaled by QA), first-layer biases (H, by QA),
// output weights (OB rows of 2H, by QB: side to move's half first), output
// biases (OB, by QA x QB); zero padding to a multiple of 64 bytes.
//
//   header: "FCNN", u16 version (1 or 2), u16 H, u8 KB, u8 mirrored (0/1),
//           u8 OB, u8 activation (0 = SCReLU), u8 bucket layout[64] (by
//           king square in its side's view); version 2 adds u32 extra
//           feature sets (bit 0: the pawn predicates of pawn_features.h;
//           bits 1-3 and 8: the knight pairs, slider pairs, slider x
//           pawn and connected pairs of pair_features.h; bit 4 retired
//           (the threat family: refused); bits 5-7 and 9: square control
//           and the piece-dependent pawn predicates of
//           square_features.h, bit 10 their control counts,
//           bit 15 mobility;
//           bit 11: the pawn-control map of pawn_features.h, after the
//           predicates, cached with them; bit 17: passed-pawn quality,
//           its pawn-only part after the control map, cached likewise,
//           its piece-dependent part a square family; bit 16: output
//           buckets by endgame type);
//           version 3 adds u16 layer1, u16 layer2 (activation 1)
//
// Activation 1, the pairwise multi-layer network: each side's hidden values
// clipped to [0, 1], its two halves multiplied pairwise (H / 2 a side), then
// H -> layer1 -> layer2 -> 1 with squared clipped ReLU between, every layer
// chosen by output bucket. After the first layer's int16 weights and biases
// come float32 weights: OB x layer1 rows of H, OB x layer1 biases, OB x layer2
// rows of layer1, OB x layer2 biases, OB rows of layer2, OB biases.
//
// With the pawn predicates, their 1,408 inputs follow the piece inputs (one
// block shared by every king bucket; per view, feature (predicate x 2 + the
// pawn is the other side's) x 64 + square, in the view's orientation and
// mirroring). They depend on pawns only, so each view's share of the first
// layer is cached by pawn key and added at evaluation, not kept in the
// accumulators.
//
// With pair or square-control families, their inputs follow (one
// block shared by every king bucket: the families in pair_features.h's,
// then square_features.h's order and numbering, in the view's orientation
// and mirroring). Each accumulator keeps its views' active extra inputs;
// evaluation catches up on the ones that changed since.
//
// Without a header, H follows from the file size. The output is in units of
// the trainer's sigmoid scale; params::nnue_scale (centipawns per unit, the
// trainer's eval scale) turns it into the engine's centipawns.
namespace nnue {

inline constexpr int QA = 255;
inline constexpr int QB = 64;
// 1536 since 2026-10-19 (the author: a flat 1536k7 net). The AVX2 output
// path's 32-bit lanes still fit: 1536 / 16 x 2 x 255^2 x 128 < 2^31.
inline constexpr int MAX_HIDDEN = 1536;
inline constexpr int INPUTS = 768;
inline constexpr int MAX_KING_BUCKETS = 32;
inline constexpr int MAX_OUTPUT_BUCKETS = 32;  // 8 material x 3 king states + 3 endgame types fits
inline constexpr int HEADER_BYTES = 76;     // version 1; versions 2 and 3 add 4 and 8
inline constexpr int MAX_EXTRA_INPUTS = 1580;  // pairfeat::MAX_ACTIVE + sqfeat::MAX_ACTIVE
inline constexpr std::uint32_t PAWN_SPANS_BIT = 2048;  // header feature set: the pawn-control map
// Header feature sets: the first layer's block chosen by the king's state
// (KING_STATES of them, nnue.cpp's king_state(): location, castling
// rights counting as one, x pawn shelter)
// in place of the king-square layout; output buckets split three ways by
// the side to move's shelter (material buckets x 3).
inline constexpr std::uint32_t KING_STATE_BIT = 8192;
inline constexpr std::uint32_t KING_STATE_OUTPUT_BIT = 16384;
inline constexpr int KING_STATES = 15;
// With fewer king buckets than KING_STATES, the states are merged: the
// header's layout bytes 0-14 give each state's block (every block used,
// the other layout bytes 0), e.g. the rare full-shelter states folded into
// their neighbours (the author, 2026-10-18: thin states starve).
// Header feature set: the first layer's block chosen by the bare king
// state (KING_LOCATIONS of them, nnue.cpp's king_location(): edge, c/f,
// d/e, advanced, castling rights left), without the shelter, so a pawn
// move never switches blocks (ROADMAP 5b; the pawn predicates can carry
// the shelter instead, the author).
inline constexpr std::uint32_t KING_LOCATION_BIT = 262144;
inline constexpr int KING_LOCATIONS = 5;
// Header feature set: three output buckets after the others, by endgame
// type, taking over where one applies (nnue.cpp's endgame_type(): no pawns,
// pawns only, opposite-coloured bishops only; ROADMAP, Try 5, G).
inline constexpr std::uint32_t ENDGAME_OUTPUT_BIT = 65536;
inline constexpr int ENDGAME_TYPES = 3;


// Loads a network; an empty path (or "<empty>") unloads, "<embedded>"
// loads the one compiled into the binary. Returns false, leaving no network
// loaded, if the file is missing or not a network.
bool load(const std::string& path);
// The network compiled into the binary (CMake EMBED_NET; embedded_net.cpp):
// its bytes, or nullptr and 0 in a build without one.
const unsigned char* embedded_data();
std::size_t embedded_size();
inline bool embedded() { return embedded_size() > 0; }
// Shares the loaded network's first layer with the other processes that run
// the same network file (data generation's workers, the engines of a
// match). Each process otherwise holds its own copy (1.5 MB at 1024
// hidden), and a dozen copies don't fit the processor's shared cache beside
// everything else, so update() waits on memory for its columns (profiled on
// an 8700K, 2026-10-11: 41% of the loads that went to memory were its; 12
// processes ran 6 to 8% more nodes a second with one copy). The file is
// mapped read-only and the first layer read from the mapping, once it has
// been checked to hold the values loaded: the same values, so the same
// evaluations. Returns whether the first layer is shared now; false (and
// nothing changed) without a network, when the file no longer holds what
// was loaded or can't be mapped, or on a big-endian machine. Not while a
// search runs.
bool share();
bool shared();
// Whether a shared first layer still holds the values that were loaded: a
// mapped file rewritten on disk would change them under a running engine.
// (Windows refuses writes to the file while it is shared; elsewhere nothing
// does, so data generation checks before it writes each game.) True when
// not shared.
bool shared_intact();
bool active();
int hidden();
int king_buckets();
int output_buckets();
bool pawn_features();
// Whether the network has the pawn-control map (pawn_features.h).
bool pawn_span_features();
// Whether the network chooses its first-layer block, or splits its output
// buckets, by the king's state.
bool king_state_buckets();
bool king_location_buckets();
bool king_state_output_buckets();
bool endgame_output_buckets();
bool passer_features();
// The output bucket the loaded network uses for `pos` (testing).
int output_bucket(const Position& pos);
// The king state of `side`'s king (0-14: location x 3 + shelter, as the
// king-state blocks use it) and the endgame type (0 no pawns, 1 pawns only,
// 2 opposite-coloured bishops only, -1 none), whatever network is loaded:
// the start reservoir's tags (reservoir.h).
int king_state_of(const Position& pos, Color side);
int endgame_type_of(const Position& pos);
// The pair families the network declares (pairfeat::Family bits; 0 none).
unsigned pair_families();
// Threat and square-control families (sqfeat::Family bits; 0 none).
unsigned square_families();
const std::string& loaded_path();

// Both sides' hidden values, [White][...] and [Black][...], each side's
// king square when they were computed, and whether a side's values are
// stale (its king changed bucket or mirroring) and must be rebuilt.
//
// With extra families (pair_features.h, square_features.h) an accumulator
// also needs the lists of the extra inputs included in each side's values
// (offsets into their block, ascending): kept apart, owned by whoever owns
// the accumulator and attached only for networks that declare extra
// families, so that networks without them pay nothing.
struct ExtraLists {
    std::array<std::array<std::uint32_t, MAX_EXTRA_INPUTS>, 2> v;
    std::array<std::uint16_t, 2> n{};
};
struct Accumulator {
    alignas(64) std::array<std::array<std::int16_t, MAX_HIDDEN>, 2> v;
    std::array<std::int8_t, 2> king{};
    std::array<bool, 2> stale{};
    // With king-state buckets: the state each side's values were built for,
    // and whether a move of that side's king or pawns asks for a recheck.
    std::array<std::int8_t, 2> bucket{};
    std::array<bool, 2> recheck{};
    ExtraLists* extras = nullptr;  // needed when extra_families()
};

// Whether the loaded network declares any extra family.
bool extra_families();

// The refresh cache (ROADMAP 5d): for each view and first-layer block (king
// bucket or king state, and mirroring), the piece part of the hidden values
// last built there and the pieces it includes. A stale view starts from its
// block's entry and applies only the pieces that differ, instead of every
// piece from the biases: the same sums, so the same values. One per search
// thread (the Evaluator's); reset() fits it to the loaded network.
struct RefreshCache {
    struct Entry {
        alignas(64) std::array<std::int16_t, MAX_HIDDEN> v;
        std::array<Bitboard, 12> pieces;
    };
    std::vector<Entry> entries;  // [view][block][mirrored]
    void reset();
};

// From scratch, from the board.
void refresh(const Position& pos, Accumulator& acc);
// `to` = `from` with one move's piece changes applied.
void update(const Accumulator& from, Accumulator& to, const MoveChanges& changes);
// The network's evaluation in centipawns, from the side to move's view;
// rebuilds stale views first, from `pos` (the position `acc` describes),
// through `cache` when given.
int evaluate(Accumulator& acc, const Position& pos, RefreshCache* cache = nullptr);
// From scratch, for tests and data tools.
int evaluate(const Position& pos);
// Debugging: every active input of both views (side to move's first), sorted,
// under the loaded network's layout, pawn predicates included; to compare
// with the trainer's (tools/trainer --print-features).
void active_inputs(const Position& pos, std::vector<int>& stm, std::vector<int>& ntm);
// Debugging: every later evaluation by a layered network writes its first
// dense layer's inputs (hidden bytes: the side to move's products, then the
// other side's) to `f`; nullptr stops it (nnueeval p8=, tools/nnue/permute.py).
void dump_layer1_inputs(std::FILE* f);
// The loaded network's evaluation path, for the load message: its later layers
// and which code runs each (2026-10-19: a layered net that fell back to a
// slower path silently would only show as lost games).
std::string evaluation_path();

}  // namespace nnue
