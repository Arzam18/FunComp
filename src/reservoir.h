#pragma once
#include <cstdint>
#include <iosfwd>
#include <string>

// The start position reservoir for round 3 (ROADMAP 5f; the author,
// 2026-10-05/07): positions from strong human games of the Lichess open
// database (CC0) as starting positions for FunComp's self-play. Outside
// games supply the position only: every move from there on, every label
// and every result is FunComp's.
//
// From a PGN stream (a Lichess monthly file, decompressed on the fly
// through a pipe), games are kept when both players are rated at least
// min_elo (2400: where Lichess ratings line up with other systems, the
// author), the time control isn't bullet or correspondence (estimated
// duration base + 40 x increment of at least min_seconds, Lichess's own
// classes), neither player is a bot (a "BOT" title: human games only),
// the game didn't end in a rules infraction (often an engine user caught; an
// abandoned game's positions are as good as any, the author), and it is
// standard chess from the usual start. Each kept
// game gives at most one position to each of three pools, picked at random
// among its positions from move 10 to move 60, not in check: the early
// middlegame (game phase 17 or more of 24), the late middlegame (9-16) and
// the endgame (8 or less, with 7 or more men: fewer are settled by the
// tablebases at once). Positions are deduplicated (board, side, castling,
// en passant) and written with the fifty-move clock reset. Endgame
// candidates can be searched (end_nodes) and kept only with a moderate
// imbalance either way, end_min to end_max centipawns (the author: an edge
// or a disadvantage both teach; the balance filter of data generation
// still checks every start).
struct ReservoirOptions {
    std::string in;           // PGN file or pipe (required)
    std::string out;          // output prefix: <out>-early.fen, -late.fen, -end.fen, and <out>.tags (every
                              // start with its tags, for tools/nnue/plan_starts.py)
    int min_elo = 2400;
    int min_seconds = 180;    // estimated duration; bullet is under 180
    int min_move = 10, max_move = 60;
    bool count_only = false;  // count games passing the tag filters, write nothing
    std::uint64_t max_games = 0;  // games read, 0 = all
    std::uint64_t end_nodes = 0;  // search for endgame candidates (0: keep them all)
    int end_min = 150, end_max = 300;  // |score| band kept, engine centipawns
    std::uint64_t seed = 1;
    int hash_mb = 1;  // cleared before each endgame search: 1 MB holds 5,000 nodes (same scores as 16 MB, 17% faster)
};

// Returns the number of positions written (or, counting, of games kept).
std::uint64_t build_reservoir(const ReservoirOptions& opt, std::ostream& log);
