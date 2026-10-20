#pragma once
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

// Chess324 (Larry Kaufman's variant, 2022; built here independently as
// "Backrank Bingo", hence the names): shuffled-piece starting positions
// close to normal chess, for data generation and as a second opening
// book. Pawns, king and rooks stand on their usual squares; each side's
// queen, bishops and knights are shuffled over the b, c, d, f and g
// files of its back rank, with the bishops on opposite colours. 18
// arrangements a side (the standard one among them), chosen
// independently for White and Black: 324 starts.
// Castling is ordinary, so any engine and GUI plays them.

// The 324 starting positions as FENs, White to move, in a fixed order.
std::vector<std::string> bingo_starts();

struct BingoOptions {
    std::string out;               // output EPD path (required)
    int plies = 8;                 // random plies played from each start
    int per_start = 1;             // positions tried per start
    std::uint64_t nodes = 20000;   // search per position, to filter lopsided ones
    int max_score = 150;           // keep positions with |score| <= this (cp)
    std::uint64_t seed = 1;
    int hash_mb = 16;
};

// Writes one FEN per line (the format of the UHO book): from every start,
// per_start times, `plies` random legal moves, then a search of `nodes`
// nodes; positions it scores beyond +-max_score are dropped. Returns the
// number of positions written; progress goes to `log` as "info string".
std::uint64_t generate_bingo(const BingoOptions& opt, std::ostream& log);
