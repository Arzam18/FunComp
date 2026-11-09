#pragma once
#include "position.h"
#include "search.h"
#include <cstdint>
#include <fstream>
#include <ostream>
#include <string>

// Tree sampling (ROADMAP, "3.1's data study", step 0): the positions where the search actually uses the static
// evaluation, to compare with the training data's (played-game positions). `treesample` replays the games of a
// PGN to sampled plies (FunComp's real games: a gauntlet's, a self-play match's), searches each position as in
// play (a node budget, a fresh hash), and writes one line for the game position and one for 1 in `rate` of the
// static evaluations the search reads: at main-search nodes (what pruning decides with; not the root, not a
// singular probe) and at quiescence stand-pats. Research builds only (-DCONT_LOG=ON, as the continuation log):
// the hook sits on the search's hottest path.
//
// Output, one line each: kind;depth;ply;eval;fen, kind g (the game position: depth and ply 0, eval its static
// evaluation), m (a main-search node: the depth left), q (a stand-pat: depth 0); eval the network's raw
// evaluation in centipawns, side to move's view (before the correction history).
struct TreeSampleOptions {
    std::string in;                // PGN file
    std::string fens;              // or: root positions, one a line (a FEN, before any '|'), taken in order
    std::string out;               // the samples
    int positions = 500;           // game positions searched
    std::uint64_t nodes = 700000;  // per position (FunComp's budget in a 10+0.1 game on this PC)
    int rate = 5000;               // write 1 in `rate` of the evaluations read
    int min_ply = 16;              // skip the first plies of each game (the opening book)
    std::uint64_t seed = 1;
};

class TreeSampler {
public:
    TreeSampler(std::ostream& out, int rate, std::uint64_t seed)
        : out_(out), rate_(rate > 0 ? rate : 1), state_(seed ? seed : 0x9e3779b97f4a7c15ULL) {}
    // Called by the search where it reads a static evaluation (search.cpp).
    void maybe(const Position& pos, char kind, int depth, int ply, int eval) {
        if (next() % static_cast<std::uint64_t>(rate_) == 0) write(pos, kind, depth, ply, eval);
    }
    void write(const Position& pos, char kind, int depth, int ply, int eval);
    std::uint64_t written() const { return written_; }

private:
    std::uint64_t next() {  // xorshift64*: cheap, and only a sampling coin
        state_ ^= state_ >> 12;
        state_ ^= state_ << 25;
        state_ ^= state_ >> 27;
        return state_ * 0x2545f4914f6cdd1dULL;
    }
    std::ostream& out_;
    int rate_;
    std::uint64_t state_;
    std::uint64_t written_ = 0;
};

void tree_sample(Searcher& searcher, const TreeSampleOptions& options, std::ostream& log);

// `evalfens`: the measurements for such samples (any build). Reads lines whose last ';' field is a position (a FEN,
// or "startpos moves <uci moves...>") and writes each line back with three fields added: the network's raw
// evaluation (root_eval less its correction), a full-window quiescence search (how far the position is from quiet)
// and a search to `nodes` from a fresh hash (the reference), all in centipawns for the side to move. The first
// `skip` lines are left out and `count` lines measured (0: all): several processes can split one file.
struct EvalFensOptions {
    std::string in, out;
    std::uint64_t nodes = 400000;
    std::uint64_t skip = 0, count = 0;
};

void eval_fens(Searcher& searcher, const EvalFensOptions& options, std::ostream& log);

// The data study's step 4 (relevance: positions from different sources, labelled alike). `labelfens` labels each
// line's position (its last ';' field, or the text before a '|') with a search to `nodes` and writes the trainer's
// text, "<FEN> | <score, White's view> | 0.5" (no game behind the position: train these with result weight 0),
// leaving out what an export leaves out: a position in check, a decisive score, a noisy best move (a capture or a
// promotion). The hash is kept between positions and cleared every 1,000. `perturb` moves each position `plies`
// random legal moves away (the far control), writing the FENs reached (none where a game ends on the way).
struct LabelFensOptions {
    std::string in, out;
    std::uint64_t nodes = 6000;
    std::uint64_t skip = 0, count = 0;
};
void label_fens(Searcher& searcher, const LabelFensOptions& options, std::ostream& log);

struct PerturbOptions {
    std::string in, out;
    int plies = 6;
    std::uint64_t seed = 1;
};
void perturb_fens(const PerturbOptions& options, std::ostream& log);
