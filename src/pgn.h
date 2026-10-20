#pragma once
#include "position.h"
#include "types.h"
#include <istream>
#include <string>
#include <utility>
#include <vector>

// Reading PGN files (fastchess and cutechess output, the Lichess database):
// the start position, the moves and the tags of each game, for tools that
// replay games (telemetry on game positions, data generation, the start
// position reservoir).
namespace pgn {

struct Game {
    std::string fen;                 // the FEN tag, or the standard start
    std::vector<std::string> moves;  // in SAN, as written
    std::string result;              // the Result tag
    std::vector<std::pair<std::string, std::string>> tags;  // every tag, in order
    std::string movetext;            // the movetext lines as read, when the Reader leaves them unsplit

    // The value of tag `name`, or "" if the game has none.
    const std::string& tag(const std::string& name) const;
};

// One game at a time, for files too large to hold (a month of the Lichess
// database is hundreds of gigabytes of text). Comments, variations, NAGs,
// move numbers and results in the movetext are skipped. Any movetext line
// ends the tags, a lone result too (a game with no moves: Lichess's aborted
// games, which once took the next game's moves). With `split_moves` false
// the movetext is kept whole in `movetext` and `moves` left empty, for
// split_moves(game) once the tags have chosen the game: the start reservoir
// keeps about one Lichess game in a hundred.
class Reader {
public:
    explicit Reader(std::istream& in, bool split_moves = true) : in_(in), split_(split_moves) {}
    // The next game; false at the end of the input.
    bool next(Game& game);

private:
    std::istream& in_;
    bool split_;
    std::string line_;
    std::string pending_;    // a tag line read past the end of the last game
    bool has_pending_ = false;
};

// Fills `game.moves` from `game.movetext` (a Reader's unsplit game).
void split_moves(Game& game);

// Every game in `in` (a Reader's, gathered).
std::vector<Game> read(std::istream& in);

// The legal move of `pos` that `san` names (check marks and annotations
// ignored), or Move{} if there is none or the SAN is ambiguous.
Move parse_san(const Position& pos, const std::string& san);

}  // namespace pgn
