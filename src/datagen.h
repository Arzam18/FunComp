#pragma once
#include "position.h"
#include <array>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

// Training data for the network (NNUE round 1): self-play games, each
// searched position labelled by the in-game search itself.
//
// One worker plays games one after another with a single searcher: its
// hash table and histories stay warm through a game and are cleared
// between games. Openings: every `bingo_every`-th game starts from a
// Chess324 position (bingo_starts(), dealt round-robin from an
// offset per worker), the others from the standard position; then 8 or 9
// uniformly random legal moves. Each move is a search with a soft node
// limit (no new iteration past it) and a hard cap (the exact node limit);
// the best move is played. Games play out: no adjudication.
//
// File format (ours, little-endian, version 1): a sequence of game blocks,
// each written whole and flushed, so a reader stops cleanly at a block cut
// short by a stopped worker.
//
//   "FCDG", u16 version, u32 block bytes (after this field)
//   u16 worker, u32 game index, u64 seed, u8 opening kind (0 standard,
//   1 Chess324), u16 start index, u8 FEN length + FEN, u8 random
//   plies + u16 moves, u8 result (DataResult), u8 termination
//   (DataEnd), u16 plies played (random ones included), u32 records
//   records, record_bytes(version) each (DataRecord in field order; version
//   1 without best, prev_score and stable, version 2 without the last two,
//   version 3 without the backward pass's four)
//
// A game dropped by the balance filter is kept as a block with termination
// Filtered, result None and its one searched position, so that rates and
// scores of dropped games can be counted; training skips it.
// Random-ply positions have no records: they are replayed from the moves.
// The halfmove clock and ply aren't stored: replaying gives them.

enum class DataResult : std::uint8_t { WhiteWin = 0, Draw = 1, BlackWin = 2, None = 3 };
enum class DataEnd : std::uint8_t {
    Checkmate = 0, Stalemate, Threefold, FiftyMoves, Material, PlyCap, Filtered,
    AdjScore, AdjTablebase, AdjDraw,  // adjudicated (DataOptions::adj_*)
    External,                         // a game from a PGN file, labelled by `labelpgn`
    AdjMate                           // adjudicated on a mate score (DataOptions::adj_mate)
};
inline constexpr int DATA_END_COUNT = 12;
const char* data_end_name(DataEnd e);

// Flags of a searched position.
enum : std::uint8_t {
    DF_HARD_CAP = 1,     // the hard node cap stopped the search mid-iteration
    DF_MATE = 2,         // the score is a mate score
    DF_TB_ROOT = 4,      // the root was in the tablebases (ranked by them)
    DF_TB_HITS = 8,      // the search found tablebase positions
    DF_MOP_UP = 16,      // the mop-up term was active at the root
    DF_IN_CHECK = 32,    // the side to move was in check
    DF_TB_WIN = 64,      // tablebase root, won for the side to move
    DF_TB_LOSS = 128,    // tablebase root, lost for the side to move
};

// A searched position. Scores in centipawns from the side to move's view.
// score is the last completed iteration's, fifty-move scaling included;
// move is the move played (the search's best move). static_eval and
// correction are what the search's root started from (NO_EVAL in check);
// qs_score is a quiescence search after it: through round 4 with the hash
// sharpening its stand-pats (so about the search's own score: not a quiet
// test), from round 5 plain (the captures alone; Searcher::quiescence).
struct DataRecord {
    std::uint16_t move = 0;
    std::int16_t score = 0;
    std::uint8_t flags = 0;
    std::uint8_t depth = 0;     // last completed iteration
    std::uint8_t seldepth = 0;  // in that iteration
    std::uint32_t nodes = 0;
    std::uint16_t tb_hits = 0;  // capped at 65535
    std::int16_t pesto = 0;
    std::int16_t initiative = 0;
    std::int16_t mop_up = 0;
    std::uint8_t drawish_pct = 100;
    std::int16_t static_eval = 0;
    std::int16_t correction = 0;
    std::int16_t qs_score = 0;
    // Version 2 on: the search's best move, kept apart from the move played
    // (`move`, which replays the game; they differ where gendata castlelong
    // or a rescore chose another); 0 in version 1, where they are the same.
    std::uint16_t best = 0;
    // Version 3 on (the author, 2026-10-07: hard examples, the search's own
    // stability, stored at no search cost): the score of the iteration
    // before the last completed one (the last's own when there was one),
    // and how many iterations before the last had the same best move (0:
    // it changed at the last), capped at 255.
    std::int16_t prev_score = 0;
    std::uint8_t stable = 0;
    // Version 4 on (round 4, the author 2026-10-19): the backward pass's
    // label (DataOptions::back_soft_nodes), the position searched again
    // after the game, last position first, the hash table kept: its score,
    // best move, last completed depth and nodes. back_depth 0: not searched
    // (in check, or a tablebase root won or lost: dropped whatever the label).
    std::int16_t back_score = 0;
    std::uint16_t back_best = 0;
    std::uint8_t back_depth = 0;
    std::uint32_t back_nodes = 0;
};
// Bytes a record takes: version 1 26, version 2 28 (best), version 3 31
// (best, prev_score, stable), version 4 40 (and the backward four).
inline constexpr std::size_t DATA_RECORD_BYTES = 26;
inline constexpr std::size_t record_bytes(unsigned version) {
    return version >= 4 ? 40 : version == 3 ? 31 : version == 2 ? 28 : 26;
}

struct DataGame {
    std::uint16_t version = 1;  // the block's format version (gendata and rescore write 3)
    std::uint16_t worker = 0;
    std::uint32_t index = 0;
    std::uint64_t seed = 0;
    std::uint8_t opening = 0;  // 0 standard, 1 Chess324, 2 reservoir (start_index: the pool)
    std::uint16_t start_index = 0;
    std::string fen;
    std::vector<Move> random_moves;
    DataResult result = DataResult::None;
    DataEnd end = DataEnd::PlyCap;
    std::uint16_t plies = 0;
    std::vector<DataRecord> records;
};

// Appends one game block to `out` and flushes it.
void write_game(std::ostream& out, const DataGame& g);
// Reads the next game block; false at the end of the file or at a block
// cut short (or not ours).
bool read_game(std::istream& in, DataGame& g);

// Whether a game is over at `pos`, with `keys` the positions before it in
// the game and `plies` the moves played: checkmate (taking precedence over
// the fifty-move rule), stalemate, the fifty-move rule, threefold
// repetition, insufficient material, or `max_plies` reached (no result).
bool game_over(const Position& pos, const std::vector<std::uint64_t>& keys, std::size_t plies,
               int max_plies, DataEnd& end, DataResult& result);

// Draw by insufficient material: bare kings, a lone minor piece, or only
// bishops, all on squares of one colour.
bool insufficient_material(const Position& pos);

struct DataOptions {
    std::string out;                  // output file (required); .params beside it
    int worker = 0;
    std::uint64_t seed = 1;
    double hours = 0;                 // stop after the game in progress; 0 = no limit
    std::uint64_t games = 0;          // stop after this many games; 0 = no limit
    std::uint64_t soft_nodes = 10000;
    std::uint64_t hard_nodes = 40000;
    bool soft_scale = false;          // the soft limit scaled by stability (SearchLimits::soft_scale)
    // No soft stop before this depth (SearchLimits::min_depth; round 5b, the
    // author 2026-10-25: 2,000 soft, 20,000 hard, depth 9); hard_nodes still
    // stops it. Not for the positions below at the drop budget. 0 = none.
    int min_depth = 0;
    // Positions the export drops whatever their search finds (in check; a
    // tablebase root the tables call won or lost, which scores as decisive)
    // are searched with these limits instead when drop_soft_nodes is set
    // (round 4, the author 2026-10-19: round 3's 6,000 / 48,000 for those,
    // 12,000 / 96,000 for the rest; drop_hard_nodes 0 keeps hard_nodes).
    // 0: every position alike.
    std::uint64_t drop_soft_nodes = 0, drop_hard_nodes = 0;
    // The backward pass (round 4, the author 2026-10-19; the 10-07 idea, a
    // second pass with a hot table): once a game is over, its positions are
    // searched again, last first, with these limits and the hash table kept
    // (so each finds the game's later positions already searched), and the
    // label stored beside the forward one (version 4 blocks; datatext
    // label=back trains on it). Not for positions dropped whatever the label
    // (in check, a tablebase root won or lost). back_hard_nodes 0: 8 x soft.
    // 0: off, version 3 blocks as before.
    std::uint64_t back_soft_nodes = 0, back_hard_nodes = 0;
    int bingo_every = 3;              // every n-th game Chess324; 0 = never
    // Starts from the reservoir (reservoir.h; ROADMAP 5f, round 3): in
    // starts_pct percent of games, by the seed and the game's index, the game
    // begins at a position of <starts>-early.fen, -late.fen or -end.fen, the
    // file drawn by the weights starts_mix (early, late, end), then
    // starts_plies_min..max random plies (none by default); the rest begin as
    // usual. Off (0) leaves every other game as it was. Each start is used
    // once (the author, 2026-10-07): a worker's k-th game from a pool takes
    // line k x starts_stride + worker (workers numbered below the stride, so
    // none share a line); past the end of a list, a usual start.
    std::string starts;
    int starts_pct = 0;
    std::array<int, 3> starts_mix = {40, 30, 30};
    int starts_plies_min = 0, starts_plies_max = 0;
    int starts_stride = 64;
    int random_min = 8, random_max = 9;
    int max_open_score = 525;         // balance filter on the first search, cp
    int max_plies = 1000;             // a game reaching this ends with no result
    // Adjudication, all off by default (round 1 played every game out):
    // a win once |score| >= adj_score for adj_plies searched plies running,
    // the same side ahead, at phase <= adj_phase (0 = off); a tablebase
    // root ends the game with the tables' result (adj_tb); a draw once
    // |score| <= adj_draw_score for adj_draw_plies plies, from ply
    // adj_draw_ply on (0 = off); a win once the score has been a mate score
    // for adj_mate searched plies running, the side it favours winning (0 =
    // off; round 2's games: no wrong result, about 1% of the searches saved,
    // tools/research/adj_round2.py). It applies to adj_pct percent of games, by
    // the seed; the rest are played out.
    int adj_score = 0, adj_plies = 8, adj_phase = 8;
    bool adj_tb = false;
    int adj_draw_score = 0, adj_draw_plies = 20, adj_draw_ply = 80;
    int adj_mate = 0;
    int adj_pct = 100;
    // Long castling (round 2a; the author, 2026-10-05: coverage of the
    // queenside-castled king states). In castle_long_pct percent of games,
    // by the seed, one side, the other or both (a third each) castle long
    // the first time it is legal before ply castle_long_by (random plies
    // counted), if the search's own move there is quiet: the move played
    // replaces it, the position keeps the search's label, and the record
    // stores the move played (so the game replays; as labelpgn does).
    int castle_long_pct = 0, castle_long_by = 40;
    int hash_mb = 16;
    // The network's first layer shared between the workers' processes
    // (nnue::share(): one copy in the processor's shared cache instead
    // of one a worker; the same values, so the same games). The worker
    // then checks before writing each game that the mapped file still
    // holds what was loaded, and stops without writing it otherwise. Off
    // unless asked for (sharenet=1), machine by machine where it measures
    // faster: 12 workers on an 8700K (12 MB shared cache) +4.9% positions
    // an hour, 24 on a 5900X (two 32 MB caches) none (2026-10-11). Keep
    // the network file from being rewritten while workers share it
    // (elsewhere than Windows, which refuses): make it read-only.
    bool share_net = false;
    std::string syzygy;               // for the .params record
};

// Adjudication's running state for one game (DataOptions::adj_*). After
// each searched position: its score from White's side, the phase, the ply
// in the game, and the tablebases' verdict for White (+1, 0, -1) or
// TB_NONE. Returns true, with the end and result set, when it adjudicates.
struct Adjudicator {
    static constexpr int TB_NONE = 2;
    const DataOptions& opt;
    int run_side = 0, run_len = 0, draw_len = 0, mate_side = 0, mate_len = 0;
    explicit Adjudicator(const DataOptions& o) : opt(o) {}
    bool update(int white_score, int phase, int ply, int tb_white, DataEnd& end, DataResult& result);
};

// Plays games until a limit; returns the number of game blocks written.
std::uint64_t generate_data(const DataOptions& opt, std::ostream& log);

// "labelpgn": FunComp's labels for the positions of games played elsewhere
// (e.g. against stronger engines: the positions are theirs to choose, every
// label is FunComp's; the author, 2026-10-04). Each game of the PGN file
// becomes a game block: its first `skip` plies unsearched (like the random
// plies), then every position searched as `gendata` searches, the record
// keeping the move played so the game replays; the PGN's result, end
// External. Usually searched deeper than self-play, so the labels are better
// than what the network already knows; exported with blend=0 (datatext) so
// the outside games' results don't enter the targets.
struct LabelPgnOptions {
    std::string in, out;
    std::uint64_t soft_nodes = 50000;
    std::uint64_t hard_nodes = 200000;
    int hash_mb = 16;
    int skip = 16;          // the opening book's plies
    int worker = 900;       // for the files' game ids (keep apart from gendata's)
    std::uint64_t seed = 0;
};
std::uint64_t label_pgn(const LabelPgnOptions& opt, std::ostream& log);

// "rescore": new labels for stored positions (the author's practice,
// 2026-10-03; built 2026-10-07 to test round 1's hand-eval labels, run when
// CPU time allows). Each game of the data file is replayed as gendata
// played it (a new game for the search, the random plies unsearched) and
// every record searched again in move order, the table kept within the
// game; the record takes the new search's fields and best move (`best`,
// version 3, with the stability fields) and keeps the move played, so the
// game replays. Results, endings and filtered stubs stay as they were.
// Never overwrites: the output must not exist; `<out>.params` records the
// source, the engine, the network and the budget. Returns the games written.
//
// Rolled labels (the author, 2026-10-21: "roll backwards down the tree to pull
// the additional findings down"): with `roll` (soft nodes per step, e.g.
// 1000,1000,1000), each position's search (the tip) is followed by searches
// of the positions before it, one ply at a time (the last but one with the
// first budget, ...), the table as the tip left it: each finds its successor
// fresh in the table, so what the tip found travels down the played line, and
// a position's label is its last such search's. With `back_pass`, after the
// game a backward pass as gendata's (last position first, the table as the
// game left it) overwrites the labels (today's design, for comparison). With
// either, records are written as version 4: the main fields (score, best,
// depth) are the final label, `nodes` everything spent on the position, and
// the backward fields hold the tip's own search (score, best move, depth,
// nodes): one file carries both labels (`datatext label=back` exports the
// tip's). In check, tablebase wins and losses keep the tip's label.
struct RescoreOptions {
    std::string in, out;
    std::uint64_t soft_nodes = 6000;  // round 3's budget
    std::uint64_t hard_nodes = 48000;
    bool soft_scale = true;
    int hash_mb = 4;
    std::uint64_t games = 0;  // 0 = all
    std::vector<std::uint64_t> roll;  // the roll's soft nodes, nearest position first (hard: 8 times)
    std::uint64_t back_pass = 0;      // an after-game backward pass's soft nodes (hard: 8 times); 0 none
};
std::uint64_t rescore_data(const RescoreOptions& opt, std::ostream& log);

// "nnueeval": for held-out data, one line per searched position (not in
// check, no mate or tablebase score): the loaded network's evaluation, the
// search score and the game's result, all from the side to move
// (tools/research/heldout.py fits K and the losses). With a network that
// has extra families too, which tools/research/eval_net.py doesn't know.
// With `fen`, each line also gets " | <FEN> | <game> <ply>" (the game's
// number in the file, the ply from its start; tools/research/kingstate_study.py).
std::uint64_t nnue_eval_data(const std::string& in, std::uint64_t max_positions, std::ostream& out,
                             bool fen = false);

// "datadump": one line per searched position (FEN and every column).
void dump_data(const std::string& path, std::uint64_t max_games, std::ostream& out, bool back = false);  // back: the backward pass's score, depth and nodes appended, then both searches' best moves
// A data file, or every .bin file in a directory, sorted; empty if neither.
std::vector<std::string> data_files(const std::string& path);

// "datastats": totals and distributions over a file, or every .bin file in
// a directory. `scale` is the eval scale the data was generated at (the
// engine's NnueScale; round 1 120, round 2 180): the conversion table's
// thresholds are stated on round 1's scale and converted, so rounds
// compare at the same real advantage.
void data_stats(const std::string& path, std::ostream& out, int scale = 120);

// "datasplit": worker files cut in two at a game index each, for a round
// whose later games were made differently (round 3b, the author, 2026-10-10:
// the interim net from each worker's first round 3b game on). `in` is a data
// file or a directory of .bin files; `index` a text file with a line per
// file: its name without ".bin", the first game index of the second part,
// then, if a number, the file's bytes at the switch (checked against the
// cut); the rest of a line and `#` lines are ignored (round3b-first-index.txt
// beside round 3's data, the laptop kit's first-index file). Games before the
// index go to <a>/<name>.bin, the others to <b>/<name>.bin, each block byte
// for byte; a part without games gets no file, and a block cut short at the
// end is left out, as every reader leaves it. A file is split only where
// that is one cut and nothing is in doubt: with no line for it, game indexes
// that don't rise through the file, a byte count that disagrees, or an
// output that exists (never overwritten), it is reported and left alone.
// Without `a` and `b`, a dry run: the same checks and counts, nothing
// written. Returns the files split (in a dry run: that would be).
struct SplitOptions {
    std::string in, index, a, b;
};
std::uint64_t split_data(const SplitOptions& opt, std::ostream& log);

// "datashuffle": a training file's fixed-size records (the trainer's, 32
// bytes) in a uniformly random order, within a memory limit (the move out of
// WSL, the author, 2026-10-17: bullet's shuffle tool was built only there).
// Two passes: every record goes to one of K temporary files at random (K
// chosen so that one fits in `mem_mb`), then each is shuffled in memory and
// appended (Rao and Sandelius' method: the order is uniform). The temporary
// files sit beside the output (<out>.shuffle-<k>, each deleted once used);
// the output is written as <out>.part and renamed when complete; an existing
// output is never overwritten. Reproducible for a seed. Disk: the input, the
// temporary files and the output, each the input's size, at the peak.
// Returns the records written (0 on any failure, said in the log).
struct ShuffleOptions {
    std::string in, out;
    std::uint64_t mem_mb = 4096;
    std::uint64_t seed = 1;
    std::uint64_t record = 32;
};
std::uint64_t shuffle_data(const ShuffleOptions& opt, std::ostream& log);

// "relabel" (round 6's distillation, the author, 2026-10-27: "a straight up
// relabel with 3.1's net"; ROADMAP, Round 6): a training file's records
// (the trainer's 32 bytes: bulletformat's ChessBoard, the board from the
// side to move's view, flipped when Black is to move) copied with each score
// replaced by the loaded network's static evaluation of its position at
// NnueScale `scale`, clamped to int16. At 180, the trainer's scale,
// sigmoid(score / 180) is the network's own prediction: train on it with
// wdl 0, since the network already learned the game result's share. Every
// other byte is the source's, so the positions, results and order are too.
// Evaluating the flipped board with White to move is exact: the inputs are
// each side's own view. Written as <out>.part and renamed when complete;
// an existing output is never overwritten. Returns the records written (0
// on any failure, said in the log).
struct RelabelOptions {
    std::string in, out;
    int scale = 180;
};
std::uint64_t relabel_data(const RelabelOptions& opt, std::ostream& log);

// "datatext": the training positions as text, one per line, for the
// trainer's converter: "<FEN> | <score> | <result>", the score in
// centipawns and the result (1.0, 0.5, 0.0) both from White's side. Played
// games only (not filtered, not without a result); positions in check, with
// a mate or tablebase score, or (with quiet) whose best move captures or
// promotes are left out. With unscale, the fifty-move scaling is undone
// while the scale is at least 1/2, and positions past that are left out.
// Thinning: positions at |score| >= a threshold are kept only one in
// `rate`, chosen reproducibly from the game and the position's index (the
// highest threshold reached applies): decided positions carry little
// signal and, with games played out, a large share of the data.
// Returns the number of lines written.
struct TextOptions {
    bool quiet = true;
    // The quiescence test (the author, 2026-10-20; ROADMAP, the data study's
    // filter arm): with a searcher and a network, a position is left out when
    // a plain quiescence search from its static eval (no hash: what the
    // captures alone make of it) raises the score by more than this many
    // centipawns. -1: off. Beside `quiet` (the best move) or instead of it.
    int qs_quiet = -1;
    // The label (the author, 2026-10-19): the forward search's (false) or
    // the backward pass's (true: version 4 records searched backward only;
    // its best move then decides the quiet filter).
    bool backward = false;
    // The backed-up label (the author, 2026-10-22: "search the played move
    // only one more ply", in its limit no search at all): the forward score
    // averaged with the next positions' scores from the same side, weight
    // td_lambda^j for the j-th position on (j = 1 .. td_plies), while every
    // move on the way was the search's own (a rescore's or castlelong's other
    // moves end the chain) and the scores stay under td_cap as searched (a
    // mate ends it too); each score unscaled for the fifty-move rule at its own
    // position first. 0: off. The roll screen (results.md, 10-22): 0.7 over up
    // to 4 plies as near a depth-18 search as about a 1.8x budget, for
    // nothing; past 4 plies the label leans on the game's outcome.
    double td_lambda = 0;
    int td_plies = 4;
    int td_cap = 1000;
    bool unscale = true;
    struct Thin {
        int threshold;  // |score| in centipawns, after unscaling
        int rate;       // keep one in this many
    };
    std::vector<Thin> thin;  // by increasing threshold
    // At most this many positions a game (0: no cap), the kept ones chosen
    // reproducibly at random after the filters and thinning: a game's
    // positions share one result label, so long games would weigh more.
    int cap = 0;
    // Experiments with the games' results (the author, 2026-10-04; to try
    // after round 2): keep only this percentage of drawn games, or of all
    // games (the size-matched control), chosen reproducibly by game.
    int draw_keep_pct = 100;
    int game_keep_pct = 100;
    // Hard examples, written hard_copies times instead of once: positions
    // where one side's score is at least hard_score but that side didn't win
    // the game (0: off), or where the score moves by hard_swing or more by the
    // same side's next search, two plies on (0: off). Scores as exported.
    int hard_score = 0;
    int hard_swing = 0;
    int hard_copies = 2;
    // Blended targets (the author, 2026-10-04): with blend in [0, 1], the
    // score written is the one whose win probability at blend_scale is
    // (1 - blend) x sigmoid(score / blend_scale) + blend x result, to train
    // at wdl 0: bullet's target then, exactly, with a weight that can differ
    // by source (outside games at 0). Below 0: off, scores as searched.
    double blend = -1;
    double blend_scale = 120;
    // Scores multiplied by this first (before thinning and blending), to
    // bring a round onto another round's eval scale: round 1 (NnueScale
    // 120) beside round 2 (180) takes 1.5.
    double rescale = 1;
    // FiftyScale as the data was generated (0: this engine's), for undoing
    // the fifty-move scaling: round 1 137, round 2 123 (SPSA 94).
    int fifty_scale = 0;
    // Games of these slices (SLICE_* bits) left out whole: the slice audit
    // (the author, 2026-10-15), each slice's worth by games against a
    // gamekeep control of the same size. 0 keeps every game.
    unsigned drop = 0;
    // One part of each file, where datasplit would cut it, without the
    // copies (round 3a / 3b in place): 'a' the games before the file's
    // first index in the first-index file `index`, 'b' those from it on; 0
    // every game. A file without a line there is skipped whole (logged).
    std::string index;
    char part = 0;
    // For data made with gendata castlelong (round 2a): its long castles and
    // refused short castles are moves the search didn't choose, so the
    // result of a game they came before is not the result of the position
    // as the search would play it. Positions before ply castle_clean (the
    // generation's castlelongby) where either side still has its queenside
    // castling right, so that such a move could still follow, are left out
    // (0: off).
    int castle_clean = 0;
};
class Searcher;
// `searcher` (with its network loaded) only for qs_quiet.
std::uint64_t data_text(const std::string& in, std::ostream& out, const TextOptions& opt,
                        std::ostream& log, Searcher* searcher = nullptr);

// The slices of a data set by how a game was made (datatext drop=): its
// start, from the standard position with random plies, a Chess324
// start, or the Lichess reservoir's early, late or endgame pool; and its
// end, adjudicated or played out (outside games count as played).
enum : unsigned {
    SLICE_STANDARD = 1, SLICE_BINGO = 2, SLICE_RESERVOIR_EARLY = 4, SLICE_RESERVOIR_LATE = 8,
    SLICE_RESERVOIR_END = 16, SLICE_ADJUDICATED = 32, SLICE_PLAYED = 64,
};
unsigned game_slices(const DataGame& g);
// Slice names, comma-separated (standard, bingo, reservoir, reservoir-early,
// reservoir-late, reservoir-end, adjudicated, played), as SLICE_* bits;
// throws std::invalid_argument on a name it doesn't know.
unsigned slice_bits(const std::string& names);
