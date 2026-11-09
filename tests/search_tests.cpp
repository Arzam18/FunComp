#include "bench.h"
#include "bingo.h"
#include "cont_log.h"
#include "datagen.h"
#include "eval.h"
#include "guide_features.h"
#include "initiative.h"
#include "labels.h"
#include "movegen.h"
#include "nnue.h"
#include "params.h"
#include "pair_features.h"
#include "pawn_features.h"
#include "square_features.h"
#include "pgn.h"
#include "position.h"
#include "search.h"
#include "see.h"
#include "tablebase.h"
#include "threads.h"
#include <algorithm>
#include <atomic>
#include <bit>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

Move find_move(const Position& pos, const std::string& uci) {
    MoveList moves;
    int count = 0;
    generate_legal(pos, moves, count);
    for (int i = 0; i < count; ++i)
        if (pos.move_to_uci(moves[static_cast<std::size_t>(i)]) == uci)
            return moves[static_cast<std::size_t>(i)];
    return Move{};
}

// The exchange started by `uci` in `fen` is worth exactly `value`.
void see_case(const char* name, const char* fen, const char* uci, int value) {
    Position pos;
    if (!pos.set_fen(fen)) {
        check(false, std::string(name) + ": bad FEN");
        return;
    }
    const Move m = find_move(pos, uci);
    if (m == Move{}) {
        check(false, std::string(name) + ": move " + uci + " not legal");
        return;
    }
    check(see_at_least(pos, m, value), std::string(name) + ": SEE >= " + std::to_string(value));
    check(!see_at_least(pos, m, value + 1), std::string(name) + ": SEE < " + std::to_string(value + 1));
}

// see_at_least must agree with see_value at every threshold, including
// captures on the mover's own back rank, where the other side's first
// recapture can promote (a bound in see_at_least once assumed it couldn't).
void see_agrees(const char* name, const char* fen, const char* uci) {
    Position pos;
    pos.set_fen(fen);
    const Move m = find_move(pos, uci);
    bool agrees = m != Move{};
    for (int t = -1200; t <= 1200 && agrees; t += 50)
        agrees = see_at_least(pos, m, t) == (see_value(pos, m) >= t);
    check(agrees, std::string(name) + ": see_at_least agrees with see_value");
}

void test_see() {
    // The values in params::see_piece_value, which the exchanges count (tuned:
    // 3.1 moved them off the hand-written 320, 330, 500, 900).
    const int P = params::see_piece_value[0], N = params::see_piece_value[1],
              B = params::see_piece_value[2], R = params::see_piece_value[3], Q = params::see_piece_value[4];
    see_agrees("Rxd1, the pawn recaptures and promotes", "4k3/8/8/8/3R4/8/4p3/3r2K1 w - - 0 1", "d4d1");
    see_agrees("...Rxd8, the pawn recaptures and promotes", "3R2k1/4P3/8/3r4/8/8/8/4K3 b - - 0 1", "d5d8");
    see_case("undefended pawn", "1k1r4/1pp4p/p7/4p3/8/P5P1/1PP4P/2K1R3 w - - 0 1", "e1e5", P);
    see_case("queen takes defended pawn", "4k3/8/3p4/4p3/8/8/8/4QK2 w - - 0 1", "e1e5", P - Q);
    see_case("knight takes defended pawn", "4k3/8/3p4/4p3/8/5N2/8/4K3 w - - 0 1", "f3e5", P - N);
    // White's second rook joins from behind the first; so does Black's.
    // Without x-rays this would come out at +100.
    see_case("x-ray battery", "4k3/4r3/4r3/4p3/8/8/4R3/4RK2 w - - 0 1", "e2e5", P - R);
    // Nxc4 Nxc4 Bxc4: the defender's recapture only loses more, so the
    // exchange must be played to the end, not cut off after it.
    see_case("recapture that doesn't help",
             "r2k3r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R2K3R w - - 2 1", "e2c4", -B);
    // A king can't recapture on a square the other king guards.
    see_case("king can't recapture next to king",
             "8/2p5/1p1p2p1/1P1Bpk2/p1Pr4/P3K3/8/4R3 w - - 8 1", "d5e4", -B);
    see_case("pawn guarded by king", "8/2p5/1p1p2pk/1P1B4/p1P2K2/P6q/8/2Q5 b - - 0 1", "g6g5", 0);
    see_case("quiet move to attacked square", "4k3/8/3p4/8/8/3N4/8/4K3 w - - 0 1", "d3e5", -N);
    see_case("quiet move to safe square", "4k3/8/3p4/8/8/3N4/8/4K3 w - - 0 1", "d3f4", 0);
    // The king may take only if nothing takes it back.
    see_case("king takes undefended", "4k3/8/8/8/8/8/3p4/4K3 w - - 0 1", "e1d2", P);
    see_case("en passant counts as even", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", 0);
    // Promotions gain the new piece less the pawn, and the new piece is
    // what can be taken back.
    see_case("safe promotion", "4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q", Q - P);
    see_case("promotion onto a guarded square", "7r/P3k3/8/8/8/8/8/4K3 w - - 0 1", "a7a8q", -P);
    see_case("capture-promotion, retaken", "r3k3/1P6/1n6/8/8/8/8/4K3 w - - 0 1", "b7a8q", R - P);
    // Qxb8 Rxb8 axb8=Q: the recapture would lose the rook to a promoting
    // pawn, so Black declines it (without the promotion: N - Q + R).
    see_case("pawn recapture promotes", "1n5r/P7/4k3/8/8/8/8/1Q2K3 w - - 0 1", "b1b8", N);
    // The exchanges count the search's values (params::see_piece_value), not
    // SEE_VALUE: a bishop taking a defended knight loses their difference.
    const char* bishop_takes = "4k3/8/3p4/4n3/8/8/1B6/4K3 w - - 0 1";
    see_case("bishop takes defended knight", bishop_takes, "b2e5", N - B);
    // A bishop worth a knight trades even; a queen at 1,000 promotes for 900.
    params::see_piece_value[2] = N;
    params::see_piece_value[4] = 1000;
    see_case("bishop takes defended knight, the bishop at the knight's value", bishop_takes, "b2e5", 0);
    see_case("safe promotion, the queen at 1000", "4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q", 900);
    see_agrees("...Rxd8, the pawn recaptures and promotes, the queen at 1000",
               "3R2k1/4P3/8/3r4/8/8/8/4K3 b - - 0 1", "d5d8");
    params::see_piece_value[2] = B;
    params::see_piece_value[4] = Q;
}

// The mop-up term in `fen` is worth exactly `value` to the side to move.
void mop_up_case(const char* name, const char* fen, int value) {
    Position pos;
    if (!pos.set_fen(fen)) {
        check(false, std::string(name) + ": bad FEN");
        return;
    }
    const int with = evaluate(pos);
    const int edge = params::mop_edge, kings = params::mop_kings;
    params::mop_edge = params::mop_kings = 0;
    const int without = evaluate(pos);
    params::mop_edge = edge;
    params::mop_kings = kings;
    check(with - without == value,
          std::string(name) + ": mop-up " + std::to_string(with - without) + ", expected " + std::to_string(value));
}

void test_mop_up() {
    // Lone king in the corner, kings 7 apart: the edge term alone.
    mop_up_case("rook vs cornered king", "k7/8/8/8/8/8/8/R3K3 w - - 0 1", 6 * params::mop_edge);
    mop_up_case("seen by the defender", "k7/8/8/8/8/8/8/R3K3 b - - 0 1", -6 * params::mop_edge);
    // King on d5 (a centre square), kings 2 apart: the kings term alone.
    mop_up_case("kings close", "8/8/8/3k4/8/4K3/8/R7 w - - 0 1", 5 * params::mop_kings);
    // Bishop and knight: only the corners of the bishop's colour count.
    mop_up_case("bishop's corner", "7k/8/8/8/8/8/8/1NB1K3 w - - 0 1", 7 * params::mop_edge);
    mop_up_case("wrong corner", "7k/8/8/8/8/8/8/1N1BK3 w - - 0 1", 0);
    mop_up_case("knight alone", "k7/8/8/8/8/8/8/N3K3 w - - 0 1", 0);
    mop_up_case("two knights", "k7/8/8/8/8/8/8/NN2K3 w - - 0 1", 0);
    // Only against a bare king: a rook left to the defender turns it off.
    mop_up_case("queen vs rook", "k7/r7/8/8/8/8/8/Q3K3 w - - 0 1", 0);
    // Pawns on the stronger side are fine.
    mop_up_case("rook and pawn", "k7/8/8/8/8/8/P7/R3K3 w - - 0 1", 6 * params::mop_edge);
    mop_up_case("both sides have pawns", "k7/p7/8/8/8/8/P7/R3K3 w - - 0 1", 0);
}

void test_null_move() {
    Position pos;
    pos.set_fen("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3");
    const std::uint64_t key = pos.key();
    const std::string fen = pos.fen();

    StateInfo st;
    pos.make_null_move(st);
    Position expected;
    expected.set_fen("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR b KQkq - 1 3");
    check(pos.key() == expected.key(), "null move: key matches the passed position");
    check(pos.side_to_move() == Color::Black && pos.ep_square() == -1,
          "null move: side flipped, ep square cleared");
    check(pos.is_consistent(), "null move: position consistent");

    pos.unmake_null_move(st);
    check(pos.key() == key && pos.fen() == fen, "null move: unmake restores the position");

    Position kp;
    kp.set_fen("8/5k2/8/8/8/8/4P3/4K3 w - - 0 1");
    check(!kp.has_non_pawn_material(Color::White), "king and pawn has no piece material");
    check(pos.has_non_pawn_material(Color::White), "start material has piece material");
}

// Finds the mate at `depth` with the full pruning suite on.
void mate_case(const char* name, const char* fen, const char* best_uci, int mate_in, int depth) {
    Position pos;
    pos.set_fen(fen);
    Searcher s(16);
    SearchLimits limits;
    limits.depth = depth;
    const Move best = s.search(pos, limits, {}, false);
    check(pos.move_to_uci(best) == best_uci,
          std::string(name) + ": played " + pos.move_to_uci(best) + ", expected " + best_uci);
    check(s.last_score() == Searcher::MATE - (2 * mate_in - 1),
          std::string(name) + ": score " + std::to_string(s.last_score()) + " is not mate in " +
              std::to_string(mate_in));
}

void test_mates() {
    mate_case("back rank", "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", "d1d8", 1, 6);
    mate_case("rook sacrifice", "1r4k1/5ppp/8/8/8/8/3R1PPP/3R2K1 w - - 0 1", "d2d8", 2, 8);
    // Depth 14 (10 until 2026-10-19): the 5c values reduce the queen
    // sacrifice (a losing capture by SEE, searched late at a PV node) more,
    // so the mate first shows at depth 12 (24k nodes) instead of 10; any of
    // seven old values alone (LmrPvNode, LmrBase, RazorMaxDepth, ...)
    // brings it back to 10: the search's edge, not a fault. Two plies spare.
    mate_case("smothered", "r1b3kr/ppp1Bp1p/1b6/n2P4/2p3q1/2Q2N2/P4PPP/RN2R1K1 w - - 1 0",
              "c3h8", 3, 14);

    // Fail-firm returns leave decisive scores alone: the same mates, found
    // and scored exactly, with every site on.
    const int rfp = params::ff_rfp_pct, sp = params::ff_stand_pat_pct,
              pc = params::ff_probcut_pct, md = params::ff_main_depth;
    params::ff_rfp_pct = params::ff_stand_pat_pct = params::ff_probcut_pct = 50;
    params::ff_main_depth = 1;
    mate_case("fail-firm: rook sacrifice", "1r4k1/5ppp/8/8/8/8/3R1PPP/3R2K1 w - - 0 1", "d2d8",
              2, 8);
    mate_case("fail-firm: smothered",
              "r1b3kr/ppp1Bp1p/1b6/n2P4/2p3q1/2Q2N2/P4PPP/RN2R1K1 w - - 1 0", "c3h8", 3, 14);
    params::ff_rfp_pct = rfp;
    params::ff_stand_pat_pct = sp;
    params::ff_probcut_pct = pc;
    params::ff_main_depth = md;

    // Internal iterative reduction never takes a node below one ply: with
    // the minimum depth at or under the reduction (the tunables' ranges
    // allow it) and the main search's fail-firm blend on, the blend divided
    // by zero. The search runs, and still finds a mate.
    const int imd = params::iir_min_depth, ir = params::iir_reduction;
    params::iir_min_depth = 2;
    params::iir_reduction = 3;
    params::ff_main_depth = 1;
    mate_case("iir floor: rook sacrifice", "1r4k1/5ppp/8/8/8/8/3R1PPP/3R2K1 w - - 0 1", "d2d8", 2, 10);
    {
        Position pos;
        pos.set_fen("r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 3 9");
        Searcher s(16);
        SearchLimits limits;
        limits.depth = 12;
        const Move best = s.search(pos, limits, {}, false);
        check(best != Move{} && std::abs(s.last_score()) < Searcher::DECISIVE_BOUND,
              "iir floor: a middlegame searched with the reduction above the minimum depth");
    }
    params::iir_min_depth = imd;
    params::iir_reduction = ir;
    params::ff_main_depth = md;
}

// Pure search (params::pure_search, ROADMAP step 4b): nothing pruned,
// reduced or extended, so a fixed-depth search's root score is exactly the
// plain minimax value over the same quiescence: this mode's perft. The
// reference searches every legal move to the depth and calls quiescence at
// the leaves (fresh positions: no repetition or fifty-move draw can arise
// within the depth). Mate scores count plies from where they were found, so
// a leaf's quiescence (which counts from its own root) is shifted by the
// leaf's ply. Also: the switch changes the result where pruning acts.
int plain_minimax(Searcher& s, Position& pos, int depth, int ply) {
    if (depth == 0) {
        int v = s.quiescence(pos, {});
        if (v > Searcher::MATE_BOUND) v -= ply;
        if (v < -Searcher::MATE_BOUND) v += ply;
        return v;
    }
    MoveList moves;
    int count = 0;
    generate_legal(pos, moves, count);
    if (count == 0) return pos.in_check(pos.side_to_move()) ? -Searcher::MATE + ply : 0;
    int best = -Searcher::INF;
    for (int i = 0; i < count; ++i) {
        StateInfo st;
        pos.make_move(moves[static_cast<std::size_t>(i)], st);
        best = std::max(best, -plain_minimax(s, pos, depth - 1, ply + 1));
        pos.unmake_move(moves[static_cast<std::size_t>(i)], st);
    }
    return best;
}

void test_pure_search() {
    const char* fens[] = {
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",  // an opening
        "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 3 9",      // a sharp middlegame
        "2kr3r/ppq2ppp/2n1bn2/2b1p3/4P3/2N1BN2/PPPQBPPP/2KR3R b - - 6 12",       // castled opposite ways
        "6k1/5ppp/8/3N4/8/8/5PPP/3R2K1 w - - 0 30",                              // a back-rank mate in reach
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",      // checks, captures, promotions
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",                             // an endgame
        "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",                                       // king and pawn
        "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2",         // mate in one for Black
    };
    const int saved = params::pure_search;
    params::pure_search = 1;
    bool exact = true;
    for (const char* fen : fens) {
        Position pos;
        pos.set_fen(fen);
        Searcher s(16);
        SearchLimits limits;
        limits.depth = 3;
        s.search(pos, limits, {}, false);
        const int searched = s.last_score();
        Searcher r(16);
        const int reference = plain_minimax(r, pos, 3, 0);
        if (searched != reference) {
            exact = false;
            check(false, std::string("pure search at depth 3: ") + std::to_string(searched) + ", minimax " +
                             std::to_string(reference) + " in " + fen);
        }
    }
    check(exact, "pure search: the root score is the plain minimax value at depth 3");

    // The switch acts: at depth 6 a pure search visits more nodes than the
    // full suite, which prunes and reduces.
    Position pos;
    pos.set_fen(fens[1]);
    SearchLimits limits;
    limits.depth = 6;
    Searcher p(16);
    p.search(pos, limits, {}, false);
    const std::uint64_t pure_nodes = p.nodes();
    params::pure_search = 0;
    Searcher f(16);
    f.search(pos, limits, {}, false);
    check(pure_nodes > 2 * f.nodes(), "pure search: more nodes than the pruned search at depth 6");
    params::pure_search = saved;
}

// The PV printed after the last iteration of a search of `fen` after
// `moves` (UCI), as a list of moves.
std::vector<std::string> printed_pv(Searcher& s, const char* fen,
                                    const std::vector<std::string>& moves, int depth) {
    Position pos;
    pos.set_fen(fen);
    std::vector<std::uint64_t> history;
    std::vector<StateInfo> states(moves.size());
    for (std::size_t i = 0; i < moves.size(); ++i) {
        history.push_back(pos.key());
        pos.make_move(find_move(pos, moves[i]), states[i]);
    }
    SearchLimits limits;
    limits.depth = depth;
    std::ostringstream out;
    auto* const old = std::cout.rdbuf(out.rdbuf());
    s.search(pos, limits, history, true);
    std::cout.rdbuf(old);

    std::string line, last;
    std::istringstream in(out.str());
    while (std::getline(in, line))
        if (line.starts_with("info depth")) last = line;
    std::vector<std::string> pv;
    std::istringstream words(last.substr(last.find(" pv") + 3));
    for (std::string w; words >> w;) pv.push_back(w);
    return pv;
}

// The PV stops where the game would be drawn, even when the hash table
// holds moves from there on (from an earlier search, as in a game).
void test_pv_stops_at_draws() {
    // Perpetual check, the cycle already played once: the line ends at the
    // first repetition, not at the third.
    const char* perpetual = "4Q1k1/6p1/8/8/8/2rr4/1q3PPP/6K1 b - - 0 1";
    Searcher s(16);
    printed_pv(s, perpetual, {"g8h7"}, 10);
    const auto rep = printed_pv(s, perpetual, {"g8h7", "e8h5", "h7g8", "h5e8", "g8h7"}, 10);
    check(rep == std::vector<std::string>{"e8h5"}, "pv: stops at a repetition");

    // Four plies from the fifty-move rule: at most four moves.
    Searcher t(16);
    printed_pv(t, "8/8/4k3/8/8/3K4/8/R7 w - - 0 80", {}, 12);
    const auto fifty = printed_pv(t, "8/8/4k3/8/8/3K4/8/R7 w - - 96 80", {}, 12);
    check(!fifty.empty() && fifty.size() <= 4, "pv: stops at the fifty-move rule");
}

// A stop requested (as from the UCI thread) ends the search at once with a
// legal move, and stays in force until cleared.
void test_stop_request() {
    Position pos;
    Searcher s(16);
    SearchLimits limits;
    limits.depth = 64;
    s.request_stop();
    const Move best = s.search(pos, limits, {}, false);
    check(find_move(pos, pos.move_to_uci(best)) != Move{}, "stop: legal move");
    check(s.nodes() <= 2048, "stop: search ended at the first check");
    s.clear_stop();
    limits.depth = 10;
    s.search(pos, limits, {}, false);
    check(s.nodes() > 2048, "stop: cleared, the next search runs");
}

// PGN reading and SAN: tags, comments, variations, castling both ways,
// disambiguation, promotion, and a second game in the same file.
void test_pgn() {
    std::istringstream in(
        "[Event \"t\"]\n"
        "[FEN \"r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1\"]\n"
        "[Result \"*\"]\n"
        "\n"
        "1. O-O {+0.10/10 0.1s} O-O-O 2. Rfd1 (2. Ra2 Kb8) Rhe8 3. Kg2 *\n"
        "[Event \"t2\"]\n"
        "[Result \"1/2-1/2\"]\n"
        "\n"
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Bxc6 dxc6 5. O-O f6 1/2-1/2\n");
    const std::vector<pgn::Game> games = pgn::read(in);
    check(games.size() == 2, "pgn: two games");
    if (games.size() != 2) return;
    auto replay = [](const pgn::Game& g, Position& pos) {
        pos.set_fen(g.fen);
        for (const std::string& san : g.moves) {
            const Move m = pgn::parse_san(pos, san);
            if (m == Move{}) return false;
            StateInfo st;
            pos.make_move(m, st);
        }
        return true;
    };
    Position pos;
    check(games[0].moves.size() == 5 && replay(games[0], pos) &&
              pos.fen().rfind("2krr3/8/8/8/8/8/6K1/R2R4 b", 0) == 0,
          "pgn: castling and disambiguation replayed (" + pos.fen() + ")");
    check(games[1].moves.size() == 10 && games[1].result == "1/2-1/2" && replay(games[1], pos),
          "pgn: second game from the start position");

    // The streaming reader: one game at a time, every tag kept, the tag
    // line that ends a game read into the next one.
    std::istringstream lichess(
        "[Event \"Rated Blitz game\"]\n[WhiteElo \"2450\"]\n[WhiteTitle \"BOT\"]\n"
        "[TimeControl \"180+2\"]\n\n1. e4 e5 1-0\n\n"
        "[Event \"Rated Rapid game\"]\n[BlackElo \"2510\"]\n\n1. d4 d5 2. c4 0-1\n");
    pgn::Reader reader(lichess);
    pgn::Game a, b, c;
    const bool got_a = reader.next(a), got_b = reader.next(b), got_c = reader.next(c);
    check(got_a && got_b && !got_c && a.tag("WhiteElo") == "2450" && a.tag("WhiteTitle") == "BOT" &&
              a.tag("TimeControl") == "180+2" && a.tag("BlackElo").empty() && a.moves.size() == 2 &&
              b.tag("Event") == "Rated Rapid game" && b.tag("BlackElo") == "2510" && b.moves.size() == 3,
          "pgn: the streaming reader, game by game with tags");

    // A game with no moves (Lichess's aborted games: the result alone, after
    // a space) is a game of its own, not the next game's tags; unsplit, the
    // movetext waits for split_moves, comments and variations as before.
    const std::string aborted =
        "[Event \"a\"]\n[WhiteElo \"2500\"]\n\n 0-1\n\n"
        "[Event \"b\"]\n[WhiteElo \"2300\"]\n\n1. e4 { [%clk 0:03:00] } 1... e5 (1... c5 2. Nf3) 2. Nf3 ; note\nNc6 1-0\n\n"
        "[Event \"c\"]\n\n1. d4 1/2-1/2\n";
    for (const bool split : {true, false}) {
        std::istringstream text(aborted);
        pgn::Reader r(text, split);
        std::vector<pgn::Game> got;
        pgn::Game g;
        while (r.next(g)) {
            if (!split) {
                check(g.moves.empty(), "pgn: unsplit, no moves until asked");
                pgn::split_moves(g);
            }
            got.push_back(g);
        }
        const std::string mode = split ? "split" : "unsplit";
        check(got.size() == 3 && got[0].tag("Event") == "a" && got[0].moves.empty() &&
                  got[1].tag("Event") == "b" && got[1].tag("WhiteElo") == "2300" &&
                  got[1].moves == std::vector<std::string>{"e4", "e5", "Nf3", "Nc6"} &&
                  got[2].tag("Event") == "c" && got[2].moves.size() == 1,
              "pgn: a game with no moves stays its own game (" + mode + ")");
    }
    pos.set_fen("8/P6k/8/8/8/8/8/K7 w - - 0 1");
    const Move promo = pgn::parse_san(pos, "a8=Q+");
    check(promo != Move{} && promo.promotion() == PieceType::Queen, "pgn: promotion");
    check(pgn::parse_san(pos, "Nf3") == Move{}, "pgn: no such move");
}

// The hash-table key carries the halfmove clock's bucket when
// FiftyHashWidth is set; the position's own key (repetitions) never does.
void test_tt_key() {
    Position a, b, c;
    a.set_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
    b.set_fen("4k3/8/8/8/8/8/8/R3K3 w - - 50 1");
    c.set_fen("4k3/8/8/8/8/8/8/R3K3 w - - 55 1");
    const int start = params::fifty_hash_start, width = params::fifty_hash_width;
    params::fifty_hash_width = 0;
    check(tt_key(a) == a.key() && tt_key(b) == b.key(), "tt_key: off, the position's key");
    params::fifty_hash_start = 10;
    params::fifty_hash_width = 10;
    check(a.key() == b.key(), "tt_key: the position's key ignores the clock");
    check(tt_key(a) == a.key(), "tt_key: below the start, the position's key");
    check(tt_key(a) != tt_key(b), "tt_key: clock 0 and 50 differ");
    check(tt_key(b) == tt_key(c), "tt_key: one bucket for 50 and 55");
    params::fifty_hash_start = start;
    params::fifty_hash_width = width;
}

// Clearing empties every entry.
void test_tt_clear() {
    TranspositionTable tt(1);
    std::vector<std::uint64_t> keys;
    for (std::uint64_t i = 1; i <= 5000; ++i) keys.push_back(i * 0x9E3779B97F4A7C15ull);
    for (std::uint64_t k : keys) tt.store(k, Move{}, 10, 20, 5, TranspositionTable::EXACT, false);
    TranspositionTable::Entry e;
    int stored = 0;
    for (std::uint64_t k : keys) stored += tt.probe(k, e);
    check(stored > 4000, "tt clear: entries stored");
    tt.clear();
    int left = 0;
    for (std::uint64_t k : keys) left += tt.probe(k, e);
    check(left == 0, "tt clear: no entry left");
    tt.store(keys[0], Move{}, 10, 20, 5, TranspositionTable::EXACT, false);
    check(tt.probe(keys[0], e) && e.score == 10 && e.depth == 5, "tt clear: stores again");
}

// Time management's node share: the nodes counted under each root move add
// up to the root total, which is every node but the root visits.
void test_root_move_nodes() {
    Position pos;
    Searcher s(16);
    SearchLimits limits;
    limits.depth = 9;
    s.search(pos, limits, {}, false);
    MoveList moves;
    int count = 0;
    generate_legal(pos, moves, count);
    std::uint64_t sum = 0;
    for (int i = 0; i < count; ++i) sum += s.root_move_nodes(moves[static_cast<std::size_t>(i)]);
    check(sum == s.root_nodes_total(), "root nodes: the moves add up to the total");
    check(s.root_nodes_total() < s.nodes() && s.root_nodes_total() + 100 > s.nodes(),
          "root nodes: all nodes but the root visits (" + std::to_string(s.root_nodes_total()) +
              " of " + std::to_string(s.nodes()) + ")");
}

// Lazy SMP: helpers search alongside the main searcher and stop with it;
// with one thread the pool is the plain search (same nodes as a Searcher).
void test_thread_pool() {
    Position pos;
    SearchLimits limits;
    limits.depth = 10;
    ThreadPool pool(16);
    Searcher alone(16);
    pool.search(pos, limits, {}, false);
    alone.search(pos, limits, {}, false);
    check(pool.main().nodes() == alone.nodes(), "threads: one thread searches as a Searcher");

    pool.set_threads(4);
    pool.new_game();
    for (int i = 0; i < 3; ++i) {
        const Move best = pool.search(pos, limits, {}, false);
        check(find_move(pos, pos.move_to_uci(best)) != Move{}, "threads: legal move");
    }
    pos.set_fen("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1");
    limits.depth = 8;
    check(pos.move_to_uci(pool.search(pos, limits, {}, false)) == "d1d8",
          "threads: back-rank mate");
    // A stop before the search: the helpers stop with the main searcher.
    pos.set_startpos();
    limits.depth = 64;
    pool.request_stop();
    pool.search(pos, limits, {}, false);
    pool.clear_stop();
    pool.set_threads(1);
    check(pool.threads() == 1, "threads: back to one");
}

// Pondering: a "go ponder" search ignores its clock until the ponderhit,
// then keeps to it, counted from the ponderhit; a stop ends it too.
void test_ponder() {
    using namespace std::chrono;
    Position pos;
    Searcher s(16);
    SearchLimits limits;
    limits.ponder = true;
    limits.movetime = 50;  // + Move Overhead: well under 100 ms once on the clock
    std::atomic<bool> done{false};
    Move best{};
    std::thread t([&] {
        best = s.search(pos, limits, {}, false);
        done = true;
    });
    std::this_thread::sleep_for(milliseconds(300));
    check(!done, "ponder: no limits before the ponderhit");
    const auto hit = steady_clock::now();
    s.ponderhit();
    while (!done && steady_clock::now() - hit < seconds(5))
        std::this_thread::sleep_for(milliseconds(1));
    t.join();
    const auto after = duration_cast<milliseconds>(steady_clock::now() - hit).count();
    check(after < 1000, "ponder: on the clock after the ponderhit (" + std::to_string(after) +
                            " ms)");
    // The move to ponder on next is legal after the best move.
    const Move reply = s.ponder_move(pos, best);
    StateInfo st;
    pos.make_move(best, st);
    check(reply != Move{} && pos.is_legal(reply), "ponder: a legal ponder move");
    pos.unmake_move(best, st);
    {
        // The same with the fifty-move clock past FiftyHashStart, where the
        // hash key differs from the position key (a ponder move was once
        // looked up under the wrong one and never found).
        Position late;
        late.set_fen("4k3/8/3r4/8/8/3R4/8/4K3 w - - 30 60");
        Searcher ls(16);
        SearchLimits ll;
        ll.depth = 8;
        const Move lb = ls.search(late, ll, {}, false);
        const Move lr = ls.ponder_move(late, lb);
        StateInfo lst;
        late.make_move(lb, lst);
        check(lr != Move{} && late.is_legal(lr), "ponder: a ponder move at fifty-move clock 30");
    }

    // Pondering on a move the opponent doesn't play: "stop".
    s.clear_ponderhit();
    done = false;
    std::thread t2([&] {
        s.search(pos, limits, {}, false);
        done = true;
    });
    std::this_thread::sleep_for(milliseconds(100));
    s.request_stop();
    t2.join();
    s.clear_stop();
    check(done.load(), "ponder: stopped");
}

// A clock with a millisecond or none left (or run below zero, which the
// UCI layer turns into 0) still means a clock: the search stops almost at
// once with a legal move, instead of running without a limit.
void test_empty_clock() {
    for (const int left : {0, 1}) {
        Position pos;
        Searcher s(16);
        SearchLimits limits;
        limits.time[0] = limits.time[1] = left;
        limits.movestogo = 1;
        const auto start = std::chrono::steady_clock::now();
        const Move best = s.search(pos, limits, {}, false);
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count();
        const std::string what = "clock " + std::to_string(left) + " ms";
        check(find_move(pos, pos.move_to_uci(best)) != Move{}, what + ": legal move");
        check(ms < 1000, what + ": search stopped (took " + std::to_string(ms) + " ms)");
    }
}

// FENs without the two clock fields (as in EPD) default them to 0 1; a
// clock field that isn't a number is an invalid FEN.
void test_fen_clocks() {
    Position pos;
    check(pos.set_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq -"),
          "fen: 4 fields accepted");
    check(pos.fen() == "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
          "fen: missing clocks default to 0 1");
    check(pos.set_fen("8/8/8/8/8/8/8/K1k5 w - - 12"), "fen: 5 fields accepted");
    check(pos.halfmove_clock() == 12, "fen: halfmove clock read without full-move number");
    Position other;
    check(!other.set_fen("8/8/8/8/8/8/8/K1k5 w - - moves e2e4"), "fen: non-numeric clock rejected");
}

// One reversible move joins two positions (upcoming-repetition detection).
void test_reversible_move() {
    auto key = [](const char* fen) {
        Position p;
        p.set_fen(fen);
        return p;
    };
    // A black rook between a8 and d8, the other side to move: a match while
    // the path is clear, none with a piece on b8 in between.
    const Position a = key("r5k1/8/8/8/8/8/8/6K1 w - - 0 1");
    const Position b = key("3r2k1/8/8/8/8/8/8/6K1 b - - 0 1");
    check(a.reversible_move_to(b.key()) && b.reversible_move_to(a.key()),
          "repetition: rook move, clear path");
    const Position c = key("rN4k1/8/8/8/8/8/8/6K1 w - - 0 1");
    const Position d = key("1N1r2k1/8/8/8/8/8/8/6K1 b - - 0 1");
    check(!c.reversible_move_to(d.key()), "repetition: rook move, blocked path");
    // Two moves apart, or a pawn move: no match.
    const Position e = key("6k1/1r6/8/8/8/8/8/6K1 b - - 0 1");
    check(!a.reversible_move_to(e.key()), "repetition: not one move");
    const Position f = key("6k1/8/8/8/8/8/4P3/6K1 w - - 0 1");
    const Position g = key("6k1/8/8/8/8/4P3/8/6K1 b - - 0 1");
    check(!f.reversible_move_to(g.key()), "repetition: pawn moves aren't reversible");

    // In a game: Kg1-h1, Kg8-h8, Kh1-g1 leaves Black a move (Kh8-g8) from
    // the position three plies back.
    Position p;
    p.set_fen("6k1/8/8/8/8/8/8/R5K1 w - - 0 1");
    const std::uint64_t start = p.key();
    StateInfo st[3];
    const char* moves[3] = {"g1h1", "g8h8", "h1g1"};
    for (int i = 0; i < 3; ++i) p.make_move(find_move(p, moves[i]), st[i]);
    check(p.reversible_move_to(start), "repetition: king shuffle");
}

// The move picker hands out every legal move exactly once, whatever the
// hash move and killers are: legal ones, illegal ones from another
// position (as a hash collision could supply), or none. Perft never uses
// the picker, so it wouldn't notice a picker that drops a move or yields
// one twice (at the boundary between good and bad captures, say).
void test_picker_completeness() {
    const char* fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p2/4P3/6P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",
        "1k6/8/8/3q4/8/2N1B3/8/1K6 w - - 0 1",
    };
    auto sorted = [](std::vector<Move> v) {
        std::sort(v.begin(), v.end(), [](Move a, Move b) { return a.data < b.data; });
        return v;
    };
    auto legal_of = [](const Position& p) {
        MoveList moves;
        int count = 0;
        generate_legal(p, moves, count);
        return std::vector<Move>(moves.begin(), moves.begin() + count);
    };
    Searcher s(1);
    Position other;
    other.set_fen(fens[1]);
    const std::vector<Move> foreign = legal_of(other);  // mostly illegal elsewhere
    for (const char* fen : fens) {
        Position pos;
        pos.set_fen(fen);
        const std::vector<Move> legal = legal_of(pos);
        const std::vector<Move> expected = sorted(legal);
        std::vector<Move> quiets;
        for (Move m : legal)
            if (pos.piece_at(m.to()) == NO_PIECE && m.promotion() == PieceType::None) quiets.push_back(m);
        const Move k1 = quiets.size() > 0 ? quiets[0] : Move{};
        const Move k2 = quiets.size() > 1 ? quiets[1] : Move{};
        bool ok = sorted(s.picker_order(pos, Move{}, Move{}, Move{})) == expected;
        for (Move tt : legal) ok &= sorted(s.picker_order(pos, tt, k1, k2)) == expected;
        for (std::size_t i = 0; i + 2 < foreign.size(); i += 3)
            ok &= sorted(s.picker_order(pos, foreign[i], foreign[i + 1], foreign[i + 2])) == expected;
        check(ok, std::string("picker: every legal move exactly once in ") + fen);
    }

    // With QuietCheckBonus, and no history yet (every quiet scores alike),
    // the quiet checks come before the other quiets.
    const int bonus = params::quiet_check_bonus;
    params::quiet_check_bonus = 8000;
    Position pos;
    pos.set_fen("4k3/8/8/8/8/8/3N4/R3K3 w - - 0 1");
    const Position::CheckInfo ci = pos.check_info();
    bool checks_first = true, seen_plain = false;
    for (const Move m : Searcher(16).picker_order(pos, Move{}, Move{}, Move{})) {
        if (pos.piece_at(m.to()) != NO_PIECE) continue;
        const bool direct = ci.checking_from[pos.piece_at(m.from()) % 6] & bit(m.to());
        if (direct && seen_plain) checks_first = false;
        seen_plain |= !direct;
    }
    check(checks_first, "picker: quiet checks first with QuietCheckBonus");
    params::quiet_check_bonus = bonus;
}

void test_verification() {
    // Verify every pruning decision: counters must be consistent, and the
    // search must still return a legal move.
    Position pos;
    pos.set_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    Searcher s(16);
    s.set_verify_rate(1);
    SearchLimits limits;
    limits.depth = 9;
    // Null move must get a chance at this depth whatever the tuned
    // minimum depth is (the tuned pruning left no null-move cutoffs at 7).
    const int nmp_min_depth = params::nmp_min_depth;
    params::nmp_min_depth = 3;
    const Move best = s.search(pos, limits, {}, false);
    params::nmp_min_depth = nmp_min_depth;
    check(find_move(pos, pos.move_to_uci(best)) != Move{}, "verification: legal best move");

    std::uint64_t verified = 0;
    for (int h = 0; h < prune::COUNT; ++h) {
        const auto& c = s.stats().c[static_cast<std::size_t>(h)];
        check(c.fired <= c.tried, std::string(prune::NAMES[h]) + ": fired <= tried");
        check(c.verified <= c.fired, std::string(prune::NAMES[h]) + ": verified <= fired");
        check(c.wrong <= c.verified, std::string(prune::NAMES[h]) + ": wrong <= verified");
        verified += c.verified;
    }
    check(s.stats().c[prune::LMR].fired > 0, "verification: LMR fired");
    check(s.stats().c[prune::NullMove].fired > 0, "verification: null move fired");
    check(verified > 0, "verification: decisions verified");
    // Capture reductions are a subset of all reductions.
    const auto& lmr = s.stats().c[prune::LMR];
    const auto& lmr_capture = s.stats().c[prune::LmrCapture];
    check(lmr_capture.tried <= lmr.tried && lmr_capture.fired <= lmr.fired &&
              lmr_capture.verified <= lmr.verified,
          "verification: lmr-capture within lmr");
    for (int k = 0; k < prune::FEATURE_COUNT; ++k) {
        const auto& f = s.stats().f[static_cast<std::size_t>(k)];
        check(f.fired <= f.tried, std::string(prune::FEATURE_NAMES[k]) + ": acted <= considered");
    }

    s.clear_stats();
    check(s.stats().c[prune::LMR].tried == 0, "stats cleared");
}

void test_label_sampling() {
    Position pos;
    pos.set_fen("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10");
    Searcher s(16);
    std::vector<LabelSample> samples;
    s.set_label_sampling(&samples, {0, 4, 4, 4}, 12345);  // depths 1-3
    SearchLimits limits;
    limits.depth = 12;  // null-move subtrees reach depth 1 only from depth ~7 up
    s.search(pos, limits, {}, false);
    check(!samples.empty(), "labels: nodes sampled");
    bool any_null = false, any_pv = false;
    for (const LabelSample& ls : samples) {
        Position p;
        check(p.set_fen(ls.fen), "labels: sampled FEN parses");
        check(!p.in_check(p.side_to_move()), "labels: sample not in check");
        check(ls.depth >= 1 && ls.depth <= 3, "labels: depth within limit");
        check(ls.ply >= 1 && ls.alpha < ls.beta, "labels: ply and window");
        check((ls.node == prune::NodeType::PV) == (ls.beta - ls.alpha > 1),
              "labels: node type matches window");
        check(ls.static_eval - ls.correction == evaluate(p), "labels: raw eval matches the position");
        any_null |= ls.under_null;
        any_pv |= ls.node == prune::NodeType::PV;
    }
    check(any_null, "labels: some samples below a null move");
    check(any_pv, "labels: some PV samples");
}

// A label depends only on its position and depth, not on what the
// labeller searched before.
void test_label_rows() {
    const LabelSample a{"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
                        7, 3, 0, 0, 0, -10, 10, true, prune::NodeType::PV, false, false};
    const LabelSample b{"4rrk1/2p1b1p1/p1p3q1/4p3/2P2n1p/1P1NR2P/PB3PP1/3R1QK1 b - - 2 24",
                        5, 4, 0, 0, 0, 20, 21, false, prune::NodeType::Cut, true, false};
    Searcher fresh(2), used(2);
    const std::string alone = label_row(fresh, b);
    label_row(used, a);
    check(!alone.empty() && label_row(used, b) == alone, "labels: row independent of order");

    // Mate scores are kept, clipped, and flagged.
    const LabelSample m{"6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", 3, 2, 0, 0, 0, -5, 5, true,
                        prune::NodeType::PV, false, false};
    const std::string row = label_row(fresh, m);
    std::vector<std::string> fields;
    for (std::size_t start = 0, end; start <= row.size(); start = end + 1) {
        end = row.find(',', start);
        if (end == std::string::npos) end = row.size();
        fields.push_back(row.substr(start, end - start));
    }
    check(fields.size() > 13 && fields[12] == std::to_string(LABEL_SCORE_CLIP) && fields[13] == "1",
          "labels: mate clipped and flagged");

    // Try 6's null probe: one more column, a curve as long as score_curve;
    // the rest of the row unchanged.
    const std::string plain = label_row(fresh, a);
    const std::string probed = label_row(fresh, a, 0, 0, 0, true);
    const auto last = probed.rfind(',');
    const auto split = [](const std::string& s, char c) {
        std::vector<std::string> out;
        for (std::size_t start = 0, end; start <= s.size(); start = end + 1) {
            end = s.find(c, start);
            if (end == std::string::npos) end = s.size();
            out.push_back(s.substr(start, end - start));
        }
        return out;
    };
    const std::string score_curve = plain.substr(plain.rfind(',') + 1);
    check(probed.substr(0, last) == plain &&
              split(probed.substr(last + 1), ';').size() == split(score_curve, ';').size() &&
              label_header(true) == label_header() + ",null_curve",
          "labels: the null probe's column");
}

void test_features() {
    Position pos;
    const features::Vector f = features::extract(pos);
    check(f[0] == 24, "features: start phase");
    check(f[1] == 0, "features: start material balance");
    check(f[2] == 3200 && f[3] == 3200, "features: start piece material");
    check(f[4] == 8 && f[5] == 8, "features: start pawns");
    check(f[6] == 0 && f[7] == 0 && f[8] == 0, "features: nothing hanging at the start");
    check(f[12] == 20, "features: start mobility");

    // Black's queen is attacked by a pawn; White can win it.
    pos.set_fen("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1");
    const features::Vector g = features::extract(pos);
    check(g[7] == 900, "features: their queen threatened");
    check(g[8] == 900, "features: queen capture available");
    check(g[11] == 1, "features: passed pawn");
}

// Syzygy tablebases. Without tables (an empty path) nothing is probed; with
// the 3-4-5-piece tables in FUNCOMP_SYZYGY (CMake: -DSYZYGY_PATH=...),
// known results.
void test_tablebases() {
    Position pos;
    check(tb::init("") == 0 && tb::largest() == 0, "tablebases: empty path opens none");
    pos.set_fen("8/8/8/4k3/8/8/8/4KQ2 w - - 0 1");
    Searcher plain(16);
    SearchLimits limits;
    limits.depth = 6;
    plain.search(pos, limits, {}, false);
    check(plain.tb_hits() == 0, "tablebases: no probes without tables");

    const char* path = std::getenv("FUNCOMP_SYZYGY");
    if (!path || !*path) {
        std::cout << "Tablebase tests with tables skipped (FUNCOMP_SYZYGY not set).\n";
        return;
    }
    if (tb::init(path) < 5) {
        check(false, std::string("tablebases: no 5-piece tables in ") + path);
        return;
    }
    auto wdl = [&](const char* fen) {
        pos.set_fen(fen);
        return tb::probe_wdl(pos);
    };
    check(wdl("8/8/8/4k3/8/8/8/4KQ2 w - - 0 1") == tb::Wdl::Win, "tablebases: KQK won");
    check(wdl("8/8/8/4k3/8/8/8/4KQ2 b - - 0 1") == tb::Wdl::Loss, "tablebases: KQK lost");
    check(wdl("8/8/8/4k3/8/8/2b5/4KR2 w - - 0 1") == tb::Wdl::Draw, "tablebases: KRKB drawn");
    check(wdl("8/8/8/8/8/2k5/8/KBN5 w - - 0 1") == tb::Wdl::Win, "tablebases: KBNK won");
    check(wdl("8/8/8/4k3/8/8/2b5/4KR2 w - - 3 1") == tb::Wdl::Failed,
          "tablebases: no WDL probe with a running fifty-move count");

    // The same KBNK win with 90 half-moves gone: too slow for the rule.
    tb::RootRanking ranking;
    pos.set_fen("8/8/8/8/8/2k5/8/KBN5 w - - 90 1");
    check(tb::rank_root(pos, ranking) && !ranking.win && !ranking.loss,
          "tablebases: KBNK at 90 half-moves is a cursed win");
    pos.set_fen("8/8/8/8/8/2k5/8/KBN5 w - - 0 1");
    check(tb::rank_root(pos, ranking) && ranking.win && ranking.dtz,
          "tablebases: KBNK ranked a win by DTZ");
    // Every move matched, en passant included.
    pos.set_fen("8/8/8/k7/2Pp4/8/8/K7 b - c3 0 1");
    check(tb::rank_root(pos, ranking) &&
              std::any_of(ranking.moves.begin(), ranking.moves.end(),
                          [&](const tb::RankedMove& rm) { return rm.move.is_en_passant(); }),
          "tablebases: root ranking with an en passant capture");

    // Only one move wins (taking the bishop): the search plays it.
    pos.set_fen("8/8/8/4k3/8/8/3b4/4KR2 w - - 0 1");
    Searcher s(16);
    const Move best = s.search(pos, limits, {}, false);
    check(pos.move_to_uci(best) == "e1d2", "tablebases: the only winning move is played");

    // Six pieces at the root: the search probes once a capture leaves five.
    pos.set_fen("8/2p5/3k4/8/2r5/8/1Q2N3/K7 w - - 0 1");
    s.search(pos, limits, {}, false);
    check(s.tb_hits() > 0, "tablebases: probed inside the search");
    tb::init("");
}

// Data generation (datagen.h).
void test_datagen() {
    // The soft node limit ends the search at the end of an iteration: the
    // one before it hadn't reached the limit, the last one had, and the
    // score is the last completed iteration's.
    {
        Position pos;
        Searcher s(16);
        SearchLimits limits;
        limits.soft_nodes = 5000;
        s.search(pos, limits, {}, false);
        const auto& it = s.iterations();
        check(it.size() >= 2 && it.back().nodes >= 5000 && it[it.size() - 2].nodes < 5000,
              "datagen: soft limit stops after the iteration that reaches it");
        check(s.last_score() == it.back().score && !s.node_limit_hit(),
              "datagen: soft limit, the last iteration's score");
        check(s.nodes() == it.back().nodes, "datagen: soft limit, no iteration started past it");
        limits.nodes = 6000;
        limits.soft_nodes = 1000000;
        s.search(pos, limits, {}, false);
        check(s.node_limit_hit() && s.last_score() == s.iterations().back().score &&
                  s.iterations().back().nodes < 6000,
              "datagen: the hard cap keeps the last completed iteration");

        // soft_scale: the soft node limit scaled by stability, still only
        // between iterations; somewhere it stops at another iteration than
        // the plain limit does (the scale: from 51% of the limit, a best move
        // held and dominant, to over 5x, a changing one with a falling score).
        limits.nodes = 0;
        limits.soft_nodes = 5000;
        limits.soft_scale = true;
        bool ends_iterations = true, differs = false;
        for (const char* fen : {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                                "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                                "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"}) {
            Position p;
            p.set_fen(fen);
            limits.soft_scale = true;
            s.new_game();
            s.search(p, limits, {}, false);
            const std::uint64_t scaled = s.nodes();
            ends_iterations = ends_iterations && scaled == s.iterations().back().nodes &&
                              s.last_score() == s.iterations().back().score && scaled >= 5000 * 51 / 100;
            limits.soft_scale = false;
            s.new_game();
            s.search(p, limits, {}, false);
            differs = differs || s.nodes() != scaled;
        }
        check(ends_iterations, "datagen: soft_scale stops between iterations, the last one's score");
        check(differs, "datagen: soft_scale moves the stop");
    }

    // Game endings.
    {
        Position pos;
        DataEnd end{};
        DataResult result{};
        auto over = [&](const char* fen, std::vector<std::uint64_t> keys = {},
                        std::size_t plies = 0) {
            pos.set_fen(fen);
            return game_over(pos, keys, plies, 1000, end, result);
        };
        check(over("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1") && end == DataEnd::Checkmate &&
                  result == DataResult::WhiteWin,
              "datagen: checkmate");
        check(over("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1") && end == DataEnd::Stalemate &&
                  result == DataResult::Draw,
              "datagen: stalemate");
        check(over("R5k1/5ppp/8/8/8/8/8/6K1 b - - 100 80") && end == DataEnd::Checkmate &&
                  result == DataResult::WhiteWin,
              "datagen: mate on the hundredth ply wins");
        check(over("6k1/5ppp/8/8/8/8/8/R5K1 b - - 100 80") && end == DataEnd::FiftyMoves &&
                  result == DataResult::Draw,
              "datagen: fifty-move rule");
        check(!over("6k1/5ppp/8/8/8/8/8/R5K1 b - - 99 80"),
              "datagen: ninety-nine half-moves play on");
        pos.set_fen("6k1/5ppp/8/8/8/8/8/R5K1 b - - 0 1");
        const std::uint64_t key = pos.key();
        check(over("6k1/5ppp/8/8/8/8/8/R5K1 b - - 8 5", {key, 1, 2, 3, key, 4, 5, 6}) &&
                  end == DataEnd::Threefold && result == DataResult::Draw,
              "datagen: threefold repetition");
        check(!over("6k1/5ppp/8/8/8/8/8/R5K1 b - - 4 3", {key, 1, 2, 3}),
              "datagen: twice is not a draw");
        check(over("8/8/4k3/8/8/2K5/8/8 w - - 0 1") && end == DataEnd::Material,
              "datagen: bare kings");
        check(over("8/8/4k3/8/8/2K5/8/6N1 w - - 0 1") && end == DataEnd::Material,
              "datagen: a lone knight");
        check(over("8/8/4k3/8/8/2K1b3/8/2B5 w - - 0 1") && end == DataEnd::Material,
              "datagen: bishops on one colour");
        check(!over("8/8/4k3/8/8/2K2b2/8/2B5 w - - 0 1"),
              "datagen: bishops on both colours play on");
        check(!over("8/8/4k3/8/8/2K5/8/5NN1 w - - 0 1"), "datagen: two knights play on");
        check(over("6k1/5ppp/8/8/8/8/8/R5K1 b - - 0 1", {}, 1000) && end == DataEnd::PlyCap &&
                  result == DataResult::None,
              "datagen: the ply cap");
    }

    // A game survives writing and reading, field for field.
    {
        DataGame g;
        g.worker = 7;
        g.index = 123456;
        g.seed = 0x0123456789ABCDEFULL;
        g.opening = 1;
        g.start_index = 321;
        g.fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        g.random_moves = {Move(12, 28), Move(52, 36)};
        g.result = DataResult::BlackWin;
        g.end = DataEnd::Checkmate;
        g.plies = 99;
        DataRecord r;
        r.move = Move(6, 21).data;
        r.score = -1234;
        r.flags = DF_MATE | DF_TB_LOSS;
        r.depth = 17;
        r.seldepth = 33;
        r.nodes = 4000000000u;
        r.tb_hits = 65535;
        r.pesto = -30000;
        r.initiative = 45;
        r.mop_up = -7;
        r.drawish_pct = 64;
        r.static_eval = static_cast<std::int16_t>(Searcher::NO_EVAL);
        r.correction = -12;
        r.qs_score = 31000;
        g.records = {r, DataRecord{}};
        std::stringstream buf;
        write_game(buf, g);
        check(buf.str().size() == 10 + 2 + 4 + 8 + 1 + 2 + 1 + g.fen.size() + 1 + 4 + 1 + 1 + 2 +
                                      4 + 2 * DATA_RECORD_BYTES,
              "datagen: block size");
        DataGame h;
        check(read_game(buf, h), "datagen: block read back");
        const DataRecord& q = h.records.front();
        check(h.worker == 7 && h.index == 123456 && h.seed == g.seed && h.opening == 1 &&
                  h.start_index == 321 && h.fen == g.fen && h.random_moves == g.random_moves &&
                  h.result == g.result && h.end == g.end && h.plies == 99 && h.records.size() == 2,
              "datagen: game header read back");
        check(q.move == r.move && q.score == r.score && q.flags == r.flags && q.depth == 17 &&
                  q.seldepth == 33 && q.nodes == r.nodes && q.tb_hits == 65535 &&
                  q.pesto == r.pesto && q.initiative == 45 && q.mop_up == -7 &&
                  q.drawish_pct == 64 && q.static_eval == r.static_eval && q.correction == -12 &&
                  q.qs_score == 31000,
              "datagen: record read back");
        {
            // Version 3: the best move and the stability fields after each
            // record (31 bytes); version 1 reads them as zero.
            DataGame g3 = g;
            g3.version = 3;
            g3.records.front().best = Move(12, 28).data;
            g3.records.front().prev_score = -987;
            g3.records.front().stable = 4;
            std::stringstream b3;
            write_game(b3, g3);
            DataGame h3;
            check(b3.str().size() == buf.str().size() + 2 * (record_bytes(3) - record_bytes(1)) &&
                      read_game(b3, h3) && h3.version == 3 && h3.records.front().best == Move(12, 28).data &&
                      h3.records.front().prev_score == -987 && h3.records.front().stable == 4 &&
                      h3.records.front().qs_score == 31000,
                  "datagen: version 3 record read back");
            std::stringstream b1(buf.str());
            DataGame h1;
            check(read_game(b1, h1) && h1.version == 1 && h1.records.front().best == 0 &&
                      h1.records.front().prev_score == 0 && h1.records.front().stable == 0,
                  "datagen: version 1 reads without the newer fields");
        }
        std::string cut = buf.str();
        cut.pop_back();
        std::stringstream part(cut);
        check(!read_game(part, h), "datagen: a truncated block is ignored");
    }

    // datasplit: a worker file cut at a game index, each part byte for byte
    // and a block cut short left out; nothing written in a dry run, over an
    // existing output, without a line for the file, where the recorded bytes
    // disagree, or where the indexes don't rise.
    {
        namespace fs = std::filesystem;
        const auto bytes = [](const char* file) {
            std::ifstream in(file, std::ios::binary);
            return std::string(std::istreambuf_iterator<char>(in), {});
        };
        const auto put = [](const char* file, const std::string& s) {
            std::ofstream out(file, std::ios::binary | std::ios::trunc);
            out << s;
        };
        // Games 3, 4 (a filtered stub), 7 and 9, with one to four records.
        const std::uint32_t index[4] = {3, 4, 7, 9};
        std::string block[4];
        for (int i = 0; i < 4; ++i) {
            DataGame g;
            g.version = 3;
            g.worker = 5;
            g.index = index[i];
            g.fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
            g.result = DataResult::Draw;
            g.end = i == 1 ? DataEnd::Filtered : DataEnd::Threefold;
            g.records.assign(static_cast<std::size_t>(i + 1), DataRecord{});
            std::stringstream buf;
            write_game(buf, g);
            block[i] = buf.str();
        }
        const std::string head = block[0] + block[1], tail = block[2] + block[3];
        const char *a = "datasplit_test_a/w.bin", *b = "datasplit_test_b/w.bin";
        SplitOptions so;
        so.in = "datasplit_test";
        so.index = "datasplit_test/index.txt";
        std::ostringstream log;
        // A fresh run: the index file's text, the outputs gone.
        const auto run = [&](const std::string& lines) {
            fs::remove_all("datasplit_test_a");
            fs::remove_all("datasplit_test_b");
            put("datasplit_test/index.txt", lines);
            return split_data(so, log);
        };
        fs::remove_all("datasplit_test");
        fs::create_directory("datasplit_test");
        put("datasplit_test/w.bin", head + tail + block[0].substr(0, 20));  // and a block cut short
        const std::string recorded = "# file, first index, bytes, when\r\nw 7 " + std::to_string(head.size()) +
                                     " 2026-10-10 10:22\r\n";
        check(run(recorded) == 1 && !fs::exists("datasplit_test_a") &&
                  log.str().find("before it 2 games, 1 positions; from it on 2 games, 7 positions; 20 bytes") !=
                      std::string::npos,
              "datasplit: a dry run counts and writes nothing");
        so.a = "datasplit_test_a";
        so.b = "datasplit_test_b";
        check(run(recorded) == 1 && bytes(a) == head && bytes(b) == tail,
              "datasplit: each part byte for byte, the block cut short left out");
        put(b, "kept");
        check(split_data(so, log) == 0 && bytes(a) == head && bytes(b) == "kept", "datasplit: never overwrites");
        check(run("w 8 2026-10-10 10:26\n") == 1 && bytes(a) == head + block[2] && bytes(b) == block[3],
              "datasplit: an index between two games, no bytes recorded");
        check(run("w 0 0\n") == 1 && !fs::exists(a) && bytes(b) == head + tail,
              "datasplit: a part without games gets no file");
        check(run("w 7 " + std::to_string(head.size() + 1) + "\n") == 0 && !fs::exists(a) && !fs::exists(b),
              "datasplit: recorded bytes that disagree, not split");
        check(run("v 7\nw ? 2026-10-10 10:26\n") == 0 && !fs::exists(a) && !fs::exists(b),
              "datasplit: no line with an index for the file, not split");
        put("datasplit_test/w.bin", tail + head);
        check(run("w 7\n") == 0 && !fs::exists(a) && !fs::exists(b),
              "datasplit: indexes that don't rise, not split");
        for (const char* d : {"datasplit_test", "datasplit_test_a", "datasplit_test_b"}) fs::remove_all(d);
    }

    // datashuffle: every record once and whole, in a mixed order, the same
    // for a seed; in memory and in two parts; its scratch files gone; never
    // over an existing output; a file of partial records refused.
    {
        namespace fs = std::filesystem;
        const auto make = [](const char* file, std::uint64_t n) {  // record i: i, then i's bytes again
            std::ofstream out(file, std::ios::binary | std::ios::trunc);
            for (std::uint64_t i = 0; i < n; ++i) {
                const std::uint64_t v[4] = {i, ~i, i * 3, i ^ 0x5555};
                out.write(reinterpret_cast<const char*>(v), sizeof v);
            }
        };
        const auto read = [](const char* file) {
            std::ifstream in(file, std::ios::binary);
            std::vector<std::uint64_t> v;
            std::uint64_t r[4];
            while (in.read(reinterpret_cast<char*>(r), sizeof r)) {
                if (r[1] != ~r[0] || r[2] != r[0] * 3 || r[3] != (r[0] ^ 0x5555)) v.push_back(~std::uint64_t{0});
                else v.push_back(r[0]);
            }
            return v;
        };
        const auto every_once = [](std::vector<std::uint64_t> v, std::uint64_t n) {
            std::sort(v.begin(), v.end());
            for (std::uint64_t i = 0; i < n; ++i)
                if (i >= v.size() || v[i] != i) return false;
            return v.size() == n;
        };
        for (const std::uint64_t n : {std::uint64_t{1000}, std::uint64_t{600000}}) {  // 32 KB in memory; 19 MB in 2 parts
            ShuffleOptions so;
            so.in = "shuffle_test.in";
            so.out = "shuffle_test.out";
            so.mem_mb = 16;
            std::ostringstream log;
            fs::remove(so.out);
            make(so.in.c_str(), n);
            check(shuffle_data(so, log) == n, "datashuffle: every record written (" + std::to_string(n) + ")");
            const auto v = read(so.out.c_str());
            check(every_once(v, n), "datashuffle: every record once and whole (" + std::to_string(n) + ")");
            std::uint64_t moved = 0, head = 0;
            for (std::uint64_t i = 0; i < v.size(); ++i) moved += v[i] != i;
            for (std::uint64_t i = 0; i < v.size() / 10; ++i) head += v[i];
            check(moved > n * 99 / 100 && head / (v.size() / 10) > n * 4 / 10 && head / (v.size() / 10) < n * 6 / 10,
                  "datashuffle: mixed, the first tenth from all over (" + std::to_string(n) + ")");
            check(log.str().find(n > 1000 ? "in 2 parts" : "in 1 part,") != std::string::npos,
                  "datashuffle: " + std::string(n > 1000 ? "two parts" : "in memory") + " as the limit says");
            bool scratch = false;
            for (const auto& e : fs::directory_iterator("."))
                scratch = scratch || e.path().filename().string().rfind("shuffle_test.out.", 0) == 0;
            check(!scratch, "datashuffle: no scratch files left (" + std::to_string(n) + ")");
            check(shuffle_data(so, log) == 0 && read(so.out.c_str()) == v, "datashuffle: never overwrites");
            fs::remove(so.out);
            check(shuffle_data(so, log) == n && read(so.out.c_str()) == v, "datashuffle: the same for a seed");
            fs::remove(so.out);
            so.seed = 2;
            check(shuffle_data(so, log) == n && read(so.out.c_str()) != v, "datashuffle: another seed, another order");
            fs::remove(so.out);
        }
        {
            std::ofstream out("shuffle_test.in", std::ios::binary | std::ios::trunc);
            out << std::string(100, 'x');  // three records and four bytes
        }
        ShuffleOptions so;
        so.in = "shuffle_test.in";
        so.out = "shuffle_test.out";
        std::ostringstream log;
        check(shuffle_data(so, log) == 0 && !fs::exists(so.out) && !fs::exists(so.out + ".part"),
              "datashuffle: partial records refused, nothing written");
        fs::remove(so.in);
    }

    // Generated games: reproducible from the seed, Backrank Bingo every
    // third game with its start index cycling, moves that replay legally,
    // eval columns as the evaluation gives them, and a clean resume after
    // a block cut short.
    {
        auto options = [](const char* file, std::uint64_t games) {
            DataOptions opt;
            opt.out = file;
            opt.worker = 1;
            opt.seed = 99;
            opt.games = games;
            opt.soft_nodes = 300;
            opt.hard_nodes = 1200;
            opt.max_plies = 40;
            opt.max_open_score = 100000;
            return opt;
        };
        auto bytes = [](const char* file) {
            std::ifstream in(file, std::ios::binary);
            return std::string(std::istreambuf_iterator<char>(in), {});
        };
        std::ostringstream log;
        std::remove("datagen_test_a.bin");
        std::remove("datagen_test_b.bin");
        check(generate_data(options("datagen_test_a.bin", 6), log) == 6 &&
                  generate_data(options("datagen_test_b.bin", 6), log) == 6 &&
                  bytes("datagen_test_a.bin") == bytes("datagen_test_b.bin"),
              "datagen: the same seed gives the same file");
        const std::vector<std::string> starts = bingo_starts();
        bool openings = true, legal = true, columns = true;
        int n = 0;
        {
            std::ifstream in("datagen_test_a.bin", std::ios::binary);
            DataGame g;
            for (; read_game(in, g); ++n) {
                const bool bingo = n % 3 == 2;
                openings = openings && g.index == static_cast<std::uint32_t>(n) &&
                           (g.opening == 1) == bingo && g.random_moves.size() >= 8 &&
                           g.random_moves.size() <= 9;
                if (bingo)
                    openings = openings && g.start_index == (n / 3 + 108) % 324 &&
                               g.fen == starts[g.start_index];
                Position pos;
                pos.set_fen(g.fen);
                std::vector<StateInfo> st(64);
                std::size_t made = 0;
                for (const Move m : g.random_moves) pos.make_move(m, st[made++]);
                for (const DataRecord& r : g.records) {
                    const EvalParts p = eval_parts(pos);
                    columns = columns && r.pesto == p.pesto && r.initiative == p.initiative &&
                              r.mop_up == p.mop_up && r.drawish_pct == p.drawish_pct;
                    Move m;
                    m.data = r.move;
                    MoveList moves;
                    int count = 0;
                    generate_legal(pos, moves, count);
                    bool found = false;
                    for (int i = 0; i < count; ++i)
                        found = found || moves[static_cast<std::size_t>(i)] == m;
                    legal = legal && found;
                    if (!legal) break;
                    pos.make_move(m, st[made++]);
                }
            }
        }
        check(n == 6 && openings, "datagen: openings, one in three Backrank Bingo, starts cycling");
        check(legal && columns, "datagen: moves replay legally, eval columns match");
        const std::string all = bytes("datagen_test_a.bin");
        {
            std::ofstream cut("datagen_test_a.bin", std::ios::binary | std::ios::trunc);
            cut.write(all.data(), static_cast<std::streamsize>(all.size() - 5));
        }
        generate_data(options("datagen_test_a.bin", 1), log);
        check(bytes("datagen_test_a.bin") == all, "datagen: resumes after a block cut short");
        {
            // rescore: at the generation's own budget the replay is exact, so
            // every label, best move and node count comes back as generated;
            // the games replay, in version 2; never overwrites; datatext
            // reads version 2 as version 1.
            // The games stop at the ply cap with no result, which datatext
            // skips: a copy with each game drawn, so the text has positions.
            std::remove("datagen_test_r.bin");
            std::remove("datagen_test_r.bin.params");
            {
                std::ifstream src("datagen_test_a.bin", std::ios::binary);
                std::ofstream dst("datagen_test_d.bin", std::ios::binary | std::ios::trunc);
                for (DataGame g; read_game(src, g);) {
                    g.result = DataResult::Draw;
                    g.end = DataEnd::Threefold;
                    write_game(dst, g);
                }
            }
            RescoreOptions ro;
            ro.in = "datagen_test_d.bin";
            ro.out = "datagen_test_r.bin";
            ro.soft_nodes = 300;
            ro.hard_nodes = 1200;
            ro.soft_scale = false;
            ro.hash_mb = DataOptions{}.hash_mb;
            const std::uint64_t games = rescore_data(ro, log);
            std::ifstream ia("datagen_test_d.bin", std::ios::binary), ir("datagen_test_r.bin", std::ios::binary);
            DataGame ga, gr;
            int count = 0;
            bool same = games == 6, versions = true, stability = false;
            while (read_game(ia, ga) && read_game(ir, gr)) {
                ++count;
                versions = versions && ga.version == 3 && gr.version == 3;
                same = same && ga.fen == gr.fen && ga.result == gr.result && ga.records.size() == gr.records.size();
                for (std::size_t i = 0; same && i < ga.records.size(); ++i) {
                    const DataRecord &x = ga.records[i], &y = gr.records[i];
                    same = y.move == x.move && y.best == x.move && x.best == x.move && y.score == x.score &&
                           y.nodes == x.nodes && y.depth == x.depth && y.flags == x.flags &&
                           y.qs_score == x.qs_score && y.prev_score == x.prev_score && y.stable == x.stable;
                    stability = stability || (x.stable > 0 && x.prev_score != x.score);
                }
            }
            check(same && versions && count == 6, "rescore: the generation's budget gives its labels back exactly");
            check(stability, "datagen: version 3 records carry the search's stability");
            const std::string written = bytes("datagen_test_r.bin");
            check(rescore_data(ro, log) == 0 && bytes("datagen_test_r.bin") == written, "rescore: never overwrites");
            TextOptions to;
            std::ostringstream ta, tr;
            const std::uint64_t na = data_text("datagen_test_d.bin", ta, to, log);
            const std::uint64_t nr = data_text("datagen_test_r.bin", tr, to, log);
            check(na > 0 && na == nr && ta.str() == tr.str(), "rescore: datatext reads version 2 as version 1");
            std::remove("datagen_test_r.bin");
            std::remove("datagen_test_r.bin.params");
            std::remove("datagen_test_d.bin");
        }
        {
            // A file that isn't ours (a path cut short at a space once
            // pointed at an installer's log) is refused, never truncated.
            const std::string foreign = "not FunComp data, but somebody's file\n";
            {
                std::ofstream f("datagen_test_foreign.bin", std::ios::binary | std::ios::trunc);
                f << foreign;
            }
            const std::uint64_t made = generate_data(options("datagen_test_foreign.bin", 1), log);
            check(made == 0 && bytes("datagen_test_foreign.bin") == foreign,
                  "datagen: a file that isn't data is left untouched");
            std::remove("datagen_test_foreign.bin");
            std::remove("datagen_test_foreign.bin.params");
        }
        {
            // castlelong: long castles played where the option asks, the
            // games still replaying move by move from their records.
            std::remove("datagen_test_b.bin");
            DataOptions opt = options("datagen_test_b.bin", 12);
            opt.castle_long_pct = 100;
            opt.castle_long_by = 60;
            opt.max_plies = 70;
            opt.bingo_every = 0;
            generate_data(opt, log);
            std::ifstream in("datagen_test_b.bin", std::ios::binary);
            DataGame g;
            int long_castles = 0;
            bool replays = true;
            while (read_game(in, g)) {
                Position pos;
                pos.set_fen(g.fen);
                std::vector<StateInfo> st(g.random_moves.size() + g.records.size() + 1);
                std::size_t made = 0;
                for (const Move m : g.random_moves) pos.make_move(m, st[made++]);
                for (const DataRecord& r : g.records) {
                    Move m;
                    m.data = r.move;
                    MoveList moves;
                    int count = 0;
                    generate_legal(pos, moves, count);
                    bool found = false;
                    for (int i = 0; i < count; ++i) found = found || moves[static_cast<std::size_t>(i)] == m;
                    replays = replays && found;
                    if (!replays) break;
                    long_castles += (m == Move(4, 2) && pos.piece_at(4) == WK) || (m == Move(60, 58) && pos.piece_at(60) == BK);
                    pos.make_move(m, st[made++]);
                }
            }
            check(replays && long_castles > 0, "datagen: castlelong plays long castles, games replay");
            // castleclean: before the ply bound, no position where a
            // queenside castling right (so a forced move) is still ahead.
            in.close();
            TextOptions text;
            std::ostringstream all_text, clean_text;
            const std::uint64_t all_n = data_text("datagen_test_b.bin", all_text, text, log);
            text.castle_clean = 60;
            const std::uint64_t clean_n = data_text("datagen_test_b.bin", clean_text, text, log);
            bool clean = true;
            std::istringstream lines(clean_text.str());
            for (std::string line; std::getline(lines, line);) {
                std::istringstream fen(line);
                std::string board, side, castling, ep;
                int halfmove = 0, fullmove = 0;
                fen >> board >> side >> castling >> ep >> halfmove >> fullmove;
                const int ply = 2 * (fullmove - 1) + (side == "b");
                clean = clean && (ply >= 60 || castling.find_first_of("Qq") == std::string::npos);
            }
            check(clean && clean_n > 0 && clean_n < all_n, "datagen: castleclean leaves out the castling window");
        }
        {
            // Reservoir starts: each used once, a worker's k-th taking line
            // k x stride + worker, no random plies; past the end of the
            // list a usual start; a resumed run continues the walk.
            const std::vector<std::string> early = {
                "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",
                "rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 1 1",
                "rnbqkbnr/pppppppp/8/8/8/7N/PPPPPPPP/RNBQKB1R b KQkq - 1 1",
                "rnbqkbnr/pppppppp/8/8/8/N7/PPPPPPPP/R1BQKBNR b KQkq - 1 1",
                "rnbqkbnr/pppppppp/8/8/8/P7/1PPPPPPP/RNBQKBNR b KQkq - 0 1",
                "rnbqkbnr/pppppppp/8/8/8/1P6/P1PPPPPP/RNBQKBNR b KQkq - 0 1",
                "rnbqkbnr/pppppppp/8/8/8/2P5/PP1PPPPP/RNBQKBNR b KQkq - 0 1",
                "rnbqkbnr/pppppppp/8/8/8/3P4/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
                "rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                "rnbqkbnr/pppppppp/8/8/8/5P2/PPPPP1PP/RNBQKBNR b KQkq - 0 1"};
            {
                std::ofstream f("datagen_test_starts-early.fen");
                for (const std::string& s : early) f << s << '\n';
            }
            auto reservoir = [&](const char* file, std::uint64_t games, int worker) {
                DataOptions opt = options(file, games);
                opt.worker = worker;
                opt.bingo_every = 0;
                opt.starts = "datagen_test_starts";
                opt.starts_pct = 100;
                opt.starts_stride = 4;
                return opt;
            };
            auto read_all = [](const char* file) {
                std::vector<DataGame> games;
                std::ifstream in(file, std::ios::binary);
                for (DataGame g; read_game(in, g);) games.push_back(g);
                return games;
            };
            auto canonical = [](const std::string& fen) {
                Position p;
                p.set_fen(fen);
                return p.fen();
            };
            for (const char* f : {"datagen_test_c.bin", "datagen_test_d.bin", "datagen_test_e.bin"}) std::remove(f);
            generate_data(reservoir("datagen_test_c.bin", 5, 1), log);
            const std::vector<DataGame> c = read_all("datagen_test_c.bin");
            bool walk = c.size() == 5;
            for (std::size_t k = 0; walk && k < c.size(); ++k) {
                if (k < 3)  // lines 1, 5, 9 of 10
                    walk = c[k].opening == 2 && c[k].fen == canonical(early[k * 4 + 1]) && c[k].random_moves.empty();
                else
                    walk = c[k].opening == 0 && c[k].random_moves.size() >= 8;
            }
            check(walk, "datagen: reservoir starts walked once each, no random plies, then usual starts");
            generate_data(reservoir("datagen_test_d.bin", 1, 2), log);
            const std::vector<DataGame> d = read_all("datagen_test_d.bin");
            check(d.size() == 1 && d[0].fen == canonical(early[2]), "datagen: another worker, other lines");
            generate_data(reservoir("datagen_test_e.bin", 2, 1), log);
            generate_data(reservoir("datagen_test_e.bin", 3, 1), log);
            check(bytes("datagen_test_e.bin") == bytes("datagen_test_c.bin"),
                  "datagen: a resumed run continues the reservoir walk");
            for (const char* f : {"datagen_test_c.bin", "datagen_test_d.bin", "datagen_test_e.bin",
                                  "datagen_test_c.bin.params", "datagen_test_d.bin.params",
                                  "datagen_test_e.bin.params", "datagen_test_starts-early.fen"})
                std::remove(f);
        }
        for (const char* f : {"datagen_test_a.bin", "datagen_test_b.bin", "datagen_test_a.bin.params",
                              "datagen_test_b.bin.params"})
            std::remove(f);
    }

    // Adjudication rules: a score run of adj_plies at a low enough phase,
    // reset when the side ahead changes; tablebase roots at once; draws
    // late and long; all off by default.
    {
        DataEnd end{};
        DataResult result{};
        DataOptions off;
        Adjudicator none(off);
        check(!none.update(5000, 0, 200, 1, end, result), "datagen: adjudication off by default");
        DataOptions o;
        o.adj_score = 1000;
        o.adj_plies = 3;
        o.adj_phase = 8;
        Adjudicator a(o);
        check(!a.update(1200, 6, 50, Adjudicator::TB_NONE, end, result) &&
                  !a.update(1100, 6, 51, Adjudicator::TB_NONE, end, result) &&
                  a.update(1300, 6, 52, Adjudicator::TB_NONE, end, result) &&
                  end == DataEnd::AdjScore && result == DataResult::WhiteWin,
              "datagen: score adjudication after the run");
        Adjudicator b(o);
        check(!b.update(-1200, 6, 50, Adjudicator::TB_NONE, end, result) &&
                  !b.update(1200, 6, 51, Adjudicator::TB_NONE, end, result) &&
                  !b.update(-1200, 6, 52, Adjudicator::TB_NONE, end, result),
              "datagen: the run resets when the side ahead changes");
        Adjudicator c(o);
        check(!c.update(-1500, 12, 50, Adjudicator::TB_NONE, end, result) &&
                  !c.update(-1500, 12, 51, Adjudicator::TB_NONE, end, result) &&
                  !c.update(-1500, 12, 52, Adjudicator::TB_NONE, end, result) &&
                  c.update(-1500, 8, 53, Adjudicator::TB_NONE, end, result) &&
                  result == DataResult::BlackWin,
              "datagen: score adjudication waits for the phase");
        o.adj_tb = true;
        Adjudicator d(o);
        check(d.update(0, 2, 90, 0, end, result) && end == DataEnd::AdjTablebase &&
                  result == DataResult::Draw,
              "datagen: tablebase adjudication, a draw");
        o.adj_score = 0;
        o.adj_tb = false;
        o.adj_draw_score = 10;
        o.adj_draw_plies = 2;
        o.adj_draw_ply = 80;
        Adjudicator e(o);
        check(!e.update(5, 4, 70, Adjudicator::TB_NONE, end, result) &&
                  !e.update(5, 4, 71, Adjudicator::TB_NONE, end, result) &&
                  !e.update(50, 4, 80, Adjudicator::TB_NONE, end, result) &&
                  !e.update(5, 4, 81, Adjudicator::TB_NONE, end, result) &&
                  e.update(-5, 4, 82, Adjudicator::TB_NONE, end, result) &&
                  end == DataEnd::AdjDraw && result == DataResult::Draw,
              "datagen: draw adjudication, late and long");
        o.adj_draw_score = 0;
        o.adj_mate = 2;
        Adjudicator m(o);
        const int mate_white = Searcher::MATE - 9, mate_black = -(Searcher::MATE - 12);
        check(!m.update(mate_white, 20, 40, Adjudicator::TB_NONE, end, result) &&
                  !m.update(900, 20, 41, Adjudicator::TB_NONE, end, result) &&
                  !m.update(mate_black, 20, 42, Adjudicator::TB_NONE, end, result) &&
                  m.update(mate_black + 1, 20, 43, Adjudicator::TB_NONE, end, result) &&
                  end == DataEnd::AdjMate && result == DataResult::BlackWin && std::string(data_end_name(end)) == "adj-mate",
              "datagen: mate adjudication, the run restarting on a non-mate score");

        // In generation: games end by adjudication; with adj_pct 0 none do.
        auto run = [](int pct) {
            DataOptions g;
            g.out = "datagen_test_adj.bin";
            std::remove(g.out.c_str());
            g.seed = 5;
            g.games = 4;
            g.soft_nodes = 300;
            g.hard_nodes = 1200;
            g.max_plies = 200;
            g.max_open_score = 100000;
            g.adj_score = 300;
            g.adj_plies = 2;
            g.adj_phase = 24;
            g.adj_pct = pct;
            std::ostringstream log;
            generate_data(g, log);
            std::ifstream in(g.out, std::ios::binary);
            DataGame game;
            int adjudicated = 0;
            while (read_game(in, game)) adjudicated += game.end == DataEnd::AdjScore;
            in.close();
            std::remove(g.out.c_str());
            std::remove((g.out + ".params").c_str());
            return adjudicated;
        };
        check(run(100) > 0 && run(0) == 0, "datagen: adjudication in generation, and adj_pct 0");
    }

    // Text export: White's side for score and result, and the filters.
    {
        DataGame g;
        g.fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        g.result = DataResult::BlackWin;
        g.end = DataEnd::Checkmate;
        auto record = [](int from, int to, int score, std::uint8_t flags = 0) {
            DataRecord r;
            r.move = Move(from, to).data;
            r.score = static_cast<std::int16_t>(score);
            r.flags = flags;
            return r;
        };
        g.records = {record(12, 28, 50),             // e2e4, White to move
                     record(51, 35, -30),            // d7d5, Black to move
                     record(28, 35, 40),             // exd5: a capture
                     record(59, 35, 31000, DF_MATE)}; // Qxd5 with a mate score
        {
            std::ofstream f("datagen_test_c.bin", std::ios::binary);
            write_game(f, g);
            g.end = DataEnd::Filtered;  // not exported
            write_game(f, g);
        }
        std::ostringstream text, log;
        TextOptions opt;
        check(data_text("datagen_test_c.bin", text, opt, log) == 2 &&
                  text.str() ==
                      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 50 | 0.0\n"
                      "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1 | 30 | 0.0\n",
              "datagen: text export, signs and filters");
        std::ostringstream all;
        opt.quiet = false;
        check(data_text("datagen_test_c.bin", all, opt, log) == 3, "datagen: text export with captures");
        // The quiescence test (qsquiet): a huge margin keeps what quiet=0 keeps; without a searcher it refuses.
        {
            Searcher qs(16);
            TextOptions q = opt;
            q.qs_quiet = 100000;
            std::ostringstream kept, refused, qlog;
            check(data_text("datagen_test_c.bin", kept, q, qlog, &qs) == 3, "datagen: qsquiet with a huge margin keeps all");
            check(data_text("datagen_test_c.bin", refused, q, qlog) == 0 &&
                      qlog.str().find("qsquiet needs the engine's searcher") != std::string::npos,
                  "datagen: qsquiet without a searcher refuses");
            // Plain quiescence: what the captures alone make of the static eval.
            Position hang;
            hang.set_fen("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1");  // exd5 wins the queen
            check(qs.quiescence(hang, {}, true) - qs.root_eval(hang).static_eval >= 600,
                  "quiescence plain: a hanging queen raises the score");
            Position calm;
            calm.set_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");  // no capture anywhere
            check(qs.quiescence(calm, {}, true) == qs.root_eval(calm).static_eval,
                  "quiescence plain: no captures, the static eval itself");
        }
        // Thinning: a rate of one keeps everything; a huge rate, nothing.
        opt.thin = {{0, 1}};
        std::ostringstream thin_all, thin_none;
        check(data_text("datagen_test_c.bin", thin_all, opt, log) == 3, "datagen: thinning at rate 1");
        opt.thin = {{0, 1 << 30}};
        check(data_text("datagen_test_c.bin", thin_none, opt, log) == 0 &&
                  log.str().find("thinned 3") != std::string::npos,
              "datagen: thinning at a huge rate, counted");
        opt.thin = {{100, 1 << 30}};  // only scores of 100 and more: none here
        std::ostringstream thin_high;
        check(data_text("datagen_test_c.bin", thin_high, opt, log) == 3, "datagen: thinning above a threshold only");
        // The per-game cap: two of the three positions, the same two each
        // time, in game order (a subsequence of the uncapped export).
        opt.thin.clear();
        opt.cap = 2;
        std::ostringstream cap_a, cap_b;
        const std::uint64_t n_cap = data_text("datagen_test_c.bin", cap_a, opt, log);
        data_text("datagen_test_c.bin", cap_b, opt, log);
        std::vector<std::string> full_lines, cap_lines;
        std::string line;
        for (std::istringstream s(all.str()); std::getline(s, line);) full_lines.push_back(line);
        for (std::istringstream s(cap_a.str()); std::getline(s, line);) cap_lines.push_back(line);
        std::size_t at = 0;
        bool in_order = true;
        for (const std::string& l : cap_lines) {
            while (at < full_lines.size() && full_lines[at] != l) ++at;
            in_order = in_order && at < full_lines.size();
            ++at;
        }
        check(n_cap == 2 && cap_a.str() == cap_b.str() && cap_lines.size() == 2 && in_order &&
                  log.str().find("capped 1") != std::string::npos,
              "datagen: per-game cap, reproducible, in game order");
        opt.cap = 0;
        // Results experiments: no draws here (Black won), so drawkeep=0
        // keeps the game and gamekeep=0 drops it. Hard examples at 30: White is
        // +50, +30 and +40 in the three positions and didn't win, so each is
        // written twice: three positions, six lines.
        std::ostringstream no_draws, no_games, hard;
        opt.draw_keep_pct = 0;
        const std::uint64_t n_nd = data_text("datagen_test_c.bin", no_draws, opt, log);
        opt.draw_keep_pct = 100;
        opt.game_keep_pct = 0;
        const std::uint64_t n_ng = data_text("datagen_test_c.bin", no_games, opt, log);
        opt.game_keep_pct = 100;
        opt.hard_score = 30;
        const std::uint64_t n_hard = data_text("datagen_test_c.bin", hard, opt, log);
        opt.hard_score = 0;
        const auto lines = [](const std::string& s) { return std::count(s.begin(), s.end(), '\n'); };
        check(n_nd == 3 && n_ng == 0 && n_hard == 3 && lines(hard.str()) == 6 &&
                  log.str().find("extra copies of hard examples 3") != std::string::npos,
              "datagen: draw and game selection, hard examples copied");
        // Blended targets: blend 0 keeps the score; 0.5 halfway to Black's
        // win (White's +50 at 120: p = 0.5 x 0.6027 = 0.3013, -101); 1, the
        // result alone (clamped).
        const auto first_score = [&](double blend) {
            opt.blend = blend;
            std::ostringstream o;
            data_text("datagen_test_c.bin", o, opt, log);
            const std::string s = o.str();
            const auto a = s.find(" | ") + 3;
            return std::stoi(s.substr(a, s.find(" | ", a) - a));
        };
        const int b0 = first_score(0), b5 = first_score(0.5), b1 = first_score(1);
        opt.rescale = 1.5;  // round 1 onto round 2's scale
        const int r15 = first_score(-1);
        opt.rescale = 1;
        opt.blend = -1;
        check(b0 == 50 && b5 == -101 && b1 < -1500 && r15 == 75, "datagen: blended targets, rescaled scores");
        std::remove("datagen_test_c.bin");
        // Slices (the slice audit): the same game made five ways, each left
        // out whole by its start or its ending, nothing else touched.
        {
            std::ofstream f("datagen_test_s.bin", std::ios::binary);
            for (int kind = 0; kind < 5; ++kind) {  // standard, bingo, the reservoir's three pools
                g.index = static_cast<std::uint32_t>(kind);
                g.opening = static_cast<std::uint8_t>(std::min(kind, 2));
                g.start_index = static_cast<std::uint16_t>(kind < 2 ? 0 : kind - 2);
                g.end = kind == 4 ? DataEnd::AdjScore : DataEnd::Checkmate;
                write_game(f, g);
            }
        }
        const auto kept = [&](const char* drop) {
            TextOptions so;
            so.drop = slice_bits(drop);
            std::ostringstream t, l;
            return data_text("datagen_test_s.bin", t, so, l);
        };
        bool unknown = false;
        try { slice_bits("reservoir,openings"); } catch (const std::invalid_argument&) { unknown = true; }
        check(kept("") == 10 && kept("standard") == 8 && kept("bingo") == 8 && kept("reservoir") == 4 &&
                  kept("reservoir-late") == 8 && kept("adjudicated") == 8 && kept("played") == 2 &&
                  kept("standard,bingo") == 6 && unknown,
              "datagen: slices left out whole by start and ending");
        // A part in place, as datasplit would cut at game 2; a file the
        // index file doesn't name is skipped whole.
        const auto part = [&](const char* index_lines, char p) {
            {
                std::ofstream idx("datagen_test_s.idx");
                idx << index_lines;
            }
            TextOptions po;
            po.index = "datagen_test_s.idx";
            po.part = p;
            std::ostringstream t, l;
            return data_text("datagen_test_s.bin", t, po, l);
        };
        check(part("# worker, first index\ndatagen_test_s 2 0 2026-10-10\n", 'a') == 4 &&
                  part("datagen_test_s 2\n", 'b') == 6 && part("another 2\n", 'b') == 0,
              "datagen: one part of a file by its first index, in place");
        std::remove("datagen_test_s.bin");
        std::remove("datagen_test_s.idx");
    }
    {
        // labelpgn: a short game from a PGN, the first two plies unsearched,
        // the rest searched and replayable as the moves played.
        {
            std::ofstream p("labelpgn_test.pgn");
            p << "[Event \"t\"]\n[Result \"1-0\"]\n\n1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0\n";
        }
        LabelPgnOptions lo;
        lo.in = "labelpgn_test.pgn";
        lo.out = "labelpgn_test.bin";
        lo.soft_nodes = 200;
        lo.hard_nodes = 1000;
        lo.skip = 2;
        std::ostringstream log;
        const std::uint64_t n = label_pgn(lo, log);
        TextOptions to;
        to.quiet = false;
        std::ostringstream text;
        const std::uint64_t positions = data_text("labelpgn_test.bin", text, to, log);
        std::istringstream ts(text.str());
        std::string first_line, last_line, l;
        std::getline(ts, first_line);
        last_line = first_line;
        while (std::getline(ts, l)) last_line = l;
        // Five positions searched; the last (White mates next) has a mate
        // score, which the export leaves out.
        check(n == 1 && positions == 4 && log.str().find("1 games, 5 positions") != std::string::npos &&
                  first_line.rfind("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w", 0) == 0 &&
                  last_line.rfind("r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b", 0) == 0 &&
                  last_line.find("| 1.0") != std::string::npos,
              "labelpgn: positions labelled and replayed as played, the result kept");
        std::remove("labelpgn_test.pgn");
        std::remove("labelpgn_test.bin");
    }

    // The drop budget (round 4, the author 2026-10-19): a position in check,
    // which the export drops whatever its search finds, is searched within
    // dropsoftnodes / drophardnodes; the rest at the usual budget.
    {
        DataOptions opt;
        opt.out = "datagen_test_drop.bin";
        opt.worker = 1;
        opt.seed = 7;
        opt.games = 10;
        opt.soft_nodes = 2000;
        opt.hard_nodes = 8000;
        opt.drop_soft_nodes = 100;
        opt.drop_hard_nodes = 400;
        opt.max_plies = 100;
        opt.max_open_score = 100000;
        std::remove(opt.out.c_str());
        std::ostringstream log;
        generate_data(opt, log);
        int in_check = 0, above = 0;
        bool within = true;
        {
            std::ifstream in(opt.out, std::ios::binary);
            DataGame g;
            while (read_game(in, g))
                for (const DataRecord& r : g.records) {
                    if (r.flags & DF_IN_CHECK) {
                        ++in_check;
                        within = within && r.nodes <= 400;
                    } else {
                        above += r.nodes > 400;
                    }
                }
        }
        check(in_check > 0 && within, "datagen: positions in check searched within the drop budget (" +
                                          std::to_string(in_check) + ")");
        check(above > 0, "datagen: the other positions at the usual budget");
        check(log.str().find("at the drop budget " + std::to_string(in_check) + " searches") != std::string::npos,
              "datagen: the drop budget's searches counted");
        std::remove(opt.out.c_str());
        std::remove((opt.out + ".params").c_str());
    }

    // The backward pass (round 4, the author 2026-10-19): version 4 blocks,
    // the same file from the same seed, every position searched backward
    // within its limits except those in check, the forward labels as without
    // it; and the export's label=back takes the backward score and best move.
    {
        auto options = [](const char* file, std::uint64_t back) {
            DataOptions opt;
            opt.out = file;
            opt.worker = 2;
            opt.seed = 11;
            opt.games = 6;
            opt.soft_nodes = 1000;
            opt.hard_nodes = 4000;
            opt.back_soft_nodes = back;
            opt.back_hard_nodes = back ? 8 * back : 0;
            opt.hash_mb = 16;
            opt.max_plies = 80;
            opt.max_open_score = 100000;
            return opt;
        };
        auto read_all = [](const char* file) {
            std::vector<DataGame> games;
            std::ifstream in(file, std::ios::binary);
            DataGame g;
            while (read_game(in, g)) games.push_back(g);
            return games;
        };
        std::ostringstream log;
        for (const char* f : {"datagen_test_back_a.bin", "datagen_test_back_b.bin", "datagen_test_fwd.bin"})
            std::remove(f);
        generate_data(options("datagen_test_back_a.bin", 300), log);
        generate_data(options("datagen_test_back_b.bin", 300), log);
        generate_data(options("datagen_test_fwd.bin", 0), log);
        const std::vector<DataGame> back = read_all("datagen_test_back_a.bin"),
                                    again = read_all("datagen_test_back_b.bin"),
                                    fwd = read_all("datagen_test_fwd.bin");
        bool versions = !back.empty() && back.size() == fwd.size(), same = back.size() == again.size(),
             searched = true, forward_same = true;
        int in_check = 0, backward = 0, moved = 0;
        for (std::size_t i = 0; i < back.size() && i < fwd.size() && i < again.size(); ++i) {
            versions = versions && back[i].version == 4 && fwd[i].version == 3;
            same = same && back[i].records.size() == again[i].records.size();
            forward_same = forward_same && back[i].records.size() == fwd[i].records.size();
            for (std::size_t j = 0; j < back[i].records.size(); ++j) {
                const DataRecord& r = back[i].records[j];
                if (j < again[i].records.size())
                    same = same && r.back_score == again[i].records[j].back_score &&
                           r.back_best == again[i].records[j].back_best;
                if (j < fwd[i].records.size())
                    forward_same = forward_same && r.score == fwd[i].records[j].score &&
                                   r.move == fwd[i].records[j].move && fwd[i].records[j].back_depth == 0;
                if (back[i].end == DataEnd::Filtered) continue;
                if (r.flags & DF_IN_CHECK) {
                    ++in_check;
                    searched = searched && r.back_depth == 0;
                } else {
                    ++backward;
                    searched = searched && r.back_depth > 0 && r.back_nodes > 0 && r.back_nodes <= 2400 &&
                               r.back_best != 0;
                    moved += r.back_best != r.best;
                }
            }
        }
        check(versions, "datagen: the backward pass writes version 4 blocks, without it version 3");
        check(same, "datagen: the backward pass, the same file from the same seed");
        check(forward_same, "datagen: the forward labels and moves as without the backward pass");
        check(searched && backward > 0 && in_check > 0,
              "datagen: every position searched backward within its limits but those in check (" +
                  std::to_string(backward) + ", " + std::to_string(in_check) + " in check, best move changed " +
                  std::to_string(moved) + ")");
        check(log.str().find("backward " + std::to_string(backward) + " searches") != std::string::npos,
              "datagen: the backward searches counted");

        // The export: each backward score set to the forward one + 7 and its
        // best move to the forward one's, label=back writes the same positions
        // with every score 7 more for the side to move.
        {
            std::ofstream out("datagen_test_back_c.bin", std::ios::binary);
            for (DataGame g : back) {
                for (DataRecord& r : g.records)
                    if (r.back_depth) {
                        r.back_score = static_cast<std::int16_t>(r.score + 7);
                        r.back_best = r.best ? r.best : r.move;
                    }
                write_game(out, g);
            }
        }
        TextOptions plain;
        plain.unscale = false;
        TextOptions labelled = plain;
        labelled.backward = true;
        std::ostringstream ft, bt, ft3;
        const std::uint64_t nf = data_text("datagen_test_back_c.bin", ft, plain, log);
        const std::uint64_t nb = data_text("datagen_test_back_c.bin", bt, labelled, log);
        std::istringstream fl(ft.str()), bl(bt.str());
        std::string a, b;
        bool plus7 = nf == nb && nf > 0;
        while (plus7 && std::getline(fl, a) && std::getline(bl, b)) {
            const std::size_t ab = a.find(" | "), bb = b.find(" | ");
            const bool white = a.find(" w ") < ab;
            const long long sa = std::stoll(a.substr(ab + 3)), sb = std::stoll(b.substr(bb + 3));
            plus7 = a.substr(0, ab) == b.substr(0, bb) && sb - sa == (white ? 7 : -7);
        }
        check(plus7, "datatext: label=back writes the backward scores (" + std::to_string(nb) + " positions)");
        // A version 3 file has no backward labels: label=back writes nothing.
        check(data_text("datagen_test_fwd.bin", ft3, labelled, log) == 0,
              "datatext: label=back skips records never searched backward");
        for (const char* f : {"datagen_test_back_a.bin", "datagen_test_back_b.bin", "datagen_test_fwd.bin",
                              "datagen_test_back_c.bin"}) {
            std::remove(f);
            std::remove((std::string(f) + ".params").c_str());
        }
    }

    // Frame L2 (params::frame_*, the author 2026-10-19): "go nodes" at data
    // generation's limits; off by default; only for "go nodes" without a soft
    // limit of its own.
    {
        Searcher fs(16);
        const int saved[4] = {params::frame_soft_nodes, params::frame_soft_scale, params::frame_drop_soft_nodes,
                              params::frame_drop_hard_nodes};
        Position start, checked;
        start.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        checked.set_fen("rnbqkbnr/ppp2ppp/8/1B1pp3/4P3/8/PPPP1PPP/RNBQK1NR b KQkq - 1 3");
        auto limits_for = [&](const Position& p, std::uint64_t nodes, std::uint64_t soft) {
            SearchLimits l;
            l.nodes = nodes;
            l.soft_nodes = soft;
            fs.apply_frame_limits(p, l);
            return l;
        };
        params::frame_soft_nodes = 0;
        const SearchLimits off = limits_for(start, 96000, 0);
        check(off.soft_nodes == 0 && off.nodes == 96000 && !off.soft_scale, "frame L2: off by default");
        params::frame_soft_nodes = 12000;
        params::frame_soft_scale = 1;
        params::frame_drop_soft_nodes = 6000;
        params::frame_drop_hard_nodes = 48000;
        const SearchLimits a = limits_for(start, 96000, 0), b = limits_for(checked, 96000, 0),
                           c = limits_for(start, 0, 0), d = limits_for(start, 96000, 500);
        check(a.soft_nodes == 12000 && a.soft_scale && a.nodes == 96000, "frame L2: go nodes at datagen's soft limit");
        check(b.soft_nodes == 6000 && b.nodes == 48000, "frame L2: a root in check at the drop budget");
        check(c.soft_nodes == 0 && c.nodes == 0 && d.soft_nodes == 500 && d.nodes == 96000,
              "frame L2: only for go nodes without a soft limit of its own");
        params::frame_soft_nodes = saved[0];
        params::frame_soft_scale = saved[1];
        params::frame_drop_soft_nodes = saved[2];
        params::frame_drop_hard_nodes = saved[3];
    }

    // With tables: a won five-piece ending is played out to mate at the
    // data-generation budget, never drawn by the fifty-move rule.
    const char* path = std::getenv("FUNCOMP_SYZYGY");
    if (!path || !*path || tb::init(path) < 5) {
        std::cout << "Data generation tablebase test skipped (FUNCOMP_SYZYGY not set).\n";
        return;
    }
    Position pos;
    pos.set_fen("8/8/5k2/8/8/2P5/3QK3/7r w - - 0 1");
    tb::RootRanking ranking;
    check(tb::rank_root(pos, ranking) && ranking.win, "datagen: the test ending is won");
    Searcher s(16);
    std::vector<std::uint64_t> keys;
    std::vector<StateInfo> st(400);
    std::size_t made = 0;
    DataEnd end{};
    DataResult result{};
    while (!game_over(pos, keys, made, 300, end, result)) {
        SearchLimits limits;
        limits.nodes = 40000;
        limits.soft_nodes = 10000;
        const Move m = s.search(pos, limits, keys, false);
        keys.push_back(pos.key());
        pos.make_move(m, st[made++]);
    }
    check(end == DataEnd::Checkmate && result == DataResult::WhiteWin,
          std::string("datagen: the won ending is mated, ended by ") + data_end_name(end));

    // Frame L2: a tablebase root won or lost at the drop budget too.
    {
        Searcher fs(16);
        const int saved[3] = {params::frame_soft_nodes, params::frame_drop_soft_nodes, params::frame_drop_hard_nodes};
        params::frame_soft_nodes = 12000;
        params::frame_drop_soft_nodes = 6000;
        params::frame_drop_hard_nodes = 48000;
        Position won;
        won.set_fen("8/8/5k2/8/8/2P5/3QK3/7r w - - 0 1");
        SearchLimits l;
        l.nodes = 96000;
        fs.apply_frame_limits(won, l);
        check(l.soft_nodes == 6000 && l.nodes == 48000, "frame L2: a won tablebase root at the drop budget");
        params::frame_soft_nodes = saved[0];
        params::frame_drop_soft_nodes = saved[1];
        params::frame_drop_hard_nodes = saved[2];
    }

    // The tables' verdict before a search (the drop budget's) is the one the
    // search then gives: won, drawn, and not a tablebase root.
    {
        Searcher v(16);
        SearchLimits limits;
        limits.nodes = 4000;
        limits.soft_nodes = 1000;
        const char* fens[] = {"8/8/5k2/8/8/2P5/3QK3/7r w - - 0 1", "8/8/3k4/3r4/8/3R4/3K4/8 w - - 0 1",
                              "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"};
        const int expected[] = {Searcher::TB_WIN, 0, Searcher::NO_EVAL};
        for (int i = 0; i < 3; ++i) {
            Position p;
            p.set_fen(fens[i]);
            const int verdict = v.tablebase_verdict(p, limits);
            v.search(p, limits, {}, false);
            check(verdict == expected[i] && verdict == v.root_tablebase_score(),
                  std::string("search: the tablebase verdict before the search, ") + fens[i]);
        }
    }
    tb::init("");
}
}  // namespace

// LmrParentHist: a quiet reply of positive history to a quiet move of
// negative history is reduced less, the reverse more; zero is neutral, and
// the term is off at 0.
void test_lmr_parent_hist() {
    const int saved = params::lmr_parent_hist;
    prune::Context c;
    c.depth = 10;
    c.move_index = 10;
    c.quiet = true;
    c.parent_quiet = true;
    auto r = [&](int ours, int parents) {
        c.history = ours;
        c.parent_history = parents;
        return prune::lmr_reduction(c);
    };
    params::lmr_parent_hist = 0;
    const int base_pos = r(1, -1), base_neg = r(-1, 1);
    check(r(1, -1) == base_pos && r(-1, 1) == base_neg, "lmr parent hist: off, no change");
    params::lmr_parent_hist = 1024;
    check(r(1, -1) == base_pos - 1, "lmr parent hist: refutation reduced a ply less");
    check(r(-1, 1) == base_neg + 1, "lmr parent hist: poor reply reduced a ply more");
    check(r(0, -1) == r(0, 1) && r(1, 0) == base_pos && r(1, 1) == base_pos,
          "lmr parent hist: zero or same sign is neutral");
    c.parent_quiet = false;
    check(r(1, -1) == base_pos, "lmr parent hist: needs a quiet parent move");
    params::lmr_parent_hist = saved;
}

// Initiative: colour-symmetric (a position and its colour-flipped mirror
// score the same), only the tempo bonus at the start, and off at InitScale 0.
std::string flip_fen(const std::string& fen) {
    std::istringstream in(fen);
    std::string board, side, castling, ep, rest;
    in >> board >> side >> castling >> ep;
    std::getline(in, rest);
    std::vector<std::string> ranks;
    std::stringstream rs(board);
    for (std::string r; std::getline(rs, r, '/');) ranks.push_back(r);
    std::string flipped;
    for (auto it = ranks.rbegin(); it != ranks.rend(); ++it) {
        for (char ch : *it)
            flipped += std::isalpha(static_cast<unsigned char>(ch))
                           ? static_cast<char>(std::isupper(static_cast<unsigned char>(ch))
                                                   ? std::tolower(static_cast<unsigned char>(ch))
                                                   : std::toupper(static_cast<unsigned char>(ch)))
                           : ch;
        if (it + 1 != ranks.rend()) flipped += '/';
    }
    std::string cast;
    for (char ch : std::string("KQkq")) {
        const char from = std::isupper(static_cast<unsigned char>(ch))
                              ? static_cast<char>(std::tolower(static_cast<unsigned char>(ch)))
                              : static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        if (castling.find(from) != std::string::npos) cast += ch;
    }
    if (cast.empty()) cast = "-";
    if (ep != "-") ep[1] = ep[1] == '3' ? '6' : '3';
    return flipped + ' ' + (side == "w" ? "b" : "w") + ' ' + cast + ' ' + ep + rest;
}

void test_initiative() {
    const char* fens[] = {
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "r2q1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R2Q1RK1 b - - 3 10",
        "2kr3r/ppq2ppp/2n1bn2/2b1p3/4P3/2N1BN2/PPPQBPPP/2KR3R w - - 6 12",
        "6k1/5ppp/8/3N4/8/8/5PPP/3R2K1 w - - 0 30",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbqkb1r/pp1p1ppp/4pn2/2p5/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq c6 0 4",
        "4r1k1/1q3ppp/p7/1p1Q4/3n4/P4N2/1P3PPP/2R3K1 b - - 1 25",
    };
    for (const char* fen : fens) {
        Position a, b;
        a.set_fen(fen);
        b.set_fen(flip_fen(fen));
        check(initiative::evaluate(a) == initiative::evaluate(b),
              std::string("initiative: mirror-symmetric (") + fen + ")");
        check(evaluate(a) == evaluate(b), std::string("eval: mirror-symmetric (") + fen + ")");
    }
    Position start;
    check(initiative::evaluate(start) == params::init_tempo * params::init_scale / 100,
          "initiative: only the tempo bonus at the start");
    const int scale = params::init_scale;
    params::init_scale = 0;
    Position p;
    p.set_fen(fens[1]);
    check(initiative::evaluate(p) == 0, "initiative: off at InitScale 0");
    params::init_scale = scale;
    // The endgame taper: at phase 3 (a knight and a rook), 50% at phase 0
    // keeps (3 * 100 + 21 * 50) / 2400 of the untapered term.
    Position end;
    end.set_fen(fens[3]);
    const int pct = params::init_endgame_pct;
    params::init_endgame_pct = 100;
    const int full = initiative::evaluate(end);
    params::init_endgame_pct = 50;
    check(initiative::evaluate(end) == full * 1350 / 2400, "initiative: endgame taper");
    Position endm;
    endm.set_fen(flip_fen(fens[3]));
    check(initiative::evaluate(end) == initiative::evaluate(endm), "initiative: taper symmetric");
    params::init_endgame_pct = pct;
}

// Drawishness: opposite-coloured bishops and close pawnless material scale
// the eval toward a draw; same-coloured bishops and queen against rook (at
// a margin of 400, the hand-set one) don't.
void test_drawish() {
    const int ocb = params::draw_ocb_pct, pl = params::draw_pawnless_pct;
    const int margin = params::draw_pawnless_margin;
    params::draw_pawnless_margin = 400;
    auto eval_at = [](const char* fen, int pct_ocb, int pct_pl) {
        params::draw_ocb_pct = pct_ocb;
        params::draw_pawnless_pct = pct_pl;
        Position p;
        p.set_fen(fen);
        return evaluate(p);
    };
    const char* opposite = "4k3/pp3b2/8/8/8/8/PPP5/2B1K3 w - - 0 40";   // c1 dark, f7 light
    const char* same = "4k3/pp6/3b4/8/8/8/PPP5/2B1K3 w - - 0 40";       // c1 dark, d6 dark
    const char* rook_bishop = "4k3/8/8/3b4/8/8/8/R3K3 w - - 0 60";
    const char* queen_rook = "4k3/8/8/3r4/8/8/8/Q3K3 w - - 0 60";
    const int o100 = eval_at(opposite, 100, 100);
    check(eval_at(opposite, 50, 100) == o100 * 50 / 100, "drawish: opposite bishops halved");
    check(eval_at(same, 50, 100) == eval_at(same, 100, 100), "drawish: same-coloured bishops not");
    const int rb = eval_at(rook_bishop, 100, 100);
    check(eval_at(rook_bishop, 100, 50) == rb * 50 / 100, "drawish: rook vs bishop halved");
    check(eval_at(queen_rook, 100, 50) == eval_at(queen_rook, 100, 100),
          "drawish: queen vs rook not");
    params::draw_ocb_pct = ocb;
    params::draw_pawnless_pct = pl;
    params::draw_pawnless_margin = margin;
}

// Backrank Bingo: 324 distinct legal starts, the standard one among them;
// king and rooks at home, bishops on opposite colours, 20 legal moves.
void test_bingo() {
    const std::vector<std::string> starts = bingo_starts();
    check(starts.size() == 324, "bingo: 324 starts");
    std::set<std::string> distinct(starts.begin(), starts.end());
    check(distinct.size() == 324, "bingo: all distinct");
    check(distinct.count("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") == 1,
          "bingo: the standard start is one of them");
    int ok = 0;
    for (const std::string& fen : starts) {
        Position pos;
        if (!pos.set_fen(fen)) continue;
        MoveList moves;
        int count = 0;
        generate_legal(pos, moves, count);
        bool good = count == 20 && pos.fen() == fen;
        for (int c = 0; c < 2; ++c) {
            const int rank = c == 0 ? 0 : 7, base = c * 6;
            const Bitboard bishops = pos.pieces(static_cast<Piece>(base + 2));
            int light = 0;
            for (Bitboard b = bishops; b; b &= b - 1) {
                const int s = std::countr_zero(b);
                light += (file_of(s) + rank_of(s)) % 2;
                good = good && rank_of(s) == rank;
            }
            good = good && std::popcount(bishops) == 2 && light == 1;
            good = good && pos.pieces(static_cast<Piece>(base + 5)) == bit(sq(4, rank)) &&
                   pos.pieces(static_cast<Piece>(base + 3)) == (bit(sq(0, rank)) | bit(sq(7, rank)));
        }
        ok += good;
    }
    check(ok == 324, "bingo: legal, 20 moves, kings and rooks home, bishops opposite (" +
                         std::to_string(ok) + " of 324)");
}

// Continuation logging: off in normal builds (genconlog refuses); in a
// CONT_LOG build a short run writes nodes and rows that add up.
void test_cont_log() {
    ContLogOptions opt;
    opt.out = "cont_log_test";
    opt.games = 1;
    opt.random_plies = 4;
    opt.nodes = 4000;
    opt.rate = 4;
    std::ostringstream log;
    const std::uint64_t rows = generate_cont_log(opt, log);
    if (!CONT_LOG_BUILD) {
        check(rows == 0, "cont log: compiled out, genconlog refuses");
        return;
    }
    check(rows > 0, "cont log: rows written");
    std::ifstream nodes("cont_log_test.nodes.csv");
    std::string line;
    std::getline(nodes, line);
    std::uint64_t sum = 0;
    while (std::getline(nodes, line)) sum += std::stoull(line.substr(line.rfind(',') + 1));
    check(sum == rows, "cont log: the nodes' row counts add up");
}

// The learned-guide hook (prune::guide): a guide that returns the hand value
// leaves the search exactly as it is; every guided heuristic consults it,
// with a valid context; and its answers are held to the allowed band.
int guide_calls[prune::COUNT];
int guide_bad_context = 0;
int identity_guide(prune::Heuristic h, const prune::Context& c, int hand) {
    ++guide_calls[h];
    if (!c.pos || c.depth < 0) ++guide_bad_context;
    return hand;
}
int extreme_guide(prune::Heuristic, const prune::Context&, int hand) { return hand + 100000; }

// Pawn predicates (pawn_features.h): known cases, and colour symmetry (the
// flipped position gives the same predicates, colours swapped and squares
// mirrored).
void test_pawn_features() {
    auto active = [](const std::string& fen) {
        Position pos;
        pos.set_fen(fen);
        std::uint16_t f[pawnfeat::MAX_ACTIVE];
        const int n = pawnfeat::features(pos.pieces(WP), pos.pieces(BP), f);
        return std::set<int>(f, f + n);
    };
    auto feature = [](pawnfeat::Predicate p, int colour, int sq) { return (p * 2 + colour) * 64 + sq; };
    // White a2, c3, e3; Black d5 (attacking c4 and e4).
    const std::string fen = "4k3/8/8/3p4/8/2P1P3/P7/4K3 w - - 0 1";
    const std::set<int> s = active(fen);
    const int a2 = 8, c3 = 18, e3 = 20, d5 = 35;
    check(s.contains(feature(pawnfeat::Isolated, 0, a2)) && s.contains(feature(pawnfeat::Passed, 0, a2)),
          "pawn features: an isolated passed a-pawn");
    check(s.contains(feature(pawnfeat::Backward, 0, c3)) && s.contains(feature(pawnfeat::Backward, 0, e3)) &&
              s.contains(feature(pawnfeat::Isolated, 0, c3)),
          "pawn features: backward pawns, their stop squares attacked");
    check(!s.contains(feature(pawnfeat::Passed, 1, d5)) && !s.contains(feature(pawnfeat::Sentry, 0, c3)) &&
              !s.contains(feature(pawnfeat::Supported, 0, c3)),
          "pawn features: a blocked path, two sentries (neither alone), no support");
    bool symmetric = true;
    for (const std::string& f : {fen, std::string("r1bqkb1r/pp1p1ppp/2n2n2/2p1p3/2P1P3/2N2N2/PP1P1PPP/R1BQKB1R w KQkq - 0 5"),
                                 std::string("8/5pk1/6p1/1p6/1P6/P1K5/8/8 b - - 0 40")}) {
        std::set<int> mirrored;
        for (const int x : active(f)) {
            const int sq = x % 64, colour = (x / 64) % 2, p = x / 128;
            mirrored.insert((p * 2 + (1 - colour)) * 64 + (sq ^ 56));
        }
        symmetric = symmetric && mirrored == active(flip_fen(f));
    }
    check(symmetric, "pawn features: colour-symmetric");
}

void test_guide() {
    auto nodes_at = [](const char* fen) {
        Position pos;
        pos.set_fen(fen);
        Searcher s(16);
        s.new_game();
        SearchLimits limits;
        limits.depth = 10;
        s.search(pos, limits, {}, false);
        return s.nodes();
    };
    const char* fen = "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R2QKB1R w KQ - 2 8";
    const std::uint64_t plain = nodes_at(fen);
    std::fill(std::begin(guide_calls), std::end(guide_calls), 0);
    prune::guide = identity_guide;
    const std::uint64_t guided = nodes_at(fen);
    prune::guide = nullptr;
    check(guided == plain, "guide: an identity guide leaves the search unchanged (" +
                               std::to_string(guided) + " vs " + std::to_string(plain) + ")");
    for (const prune::Heuristic h : {prune::RFP, prune::Razor, prune::NullMove, prune::LMP,
                                     prune::Futility, prune::HistoryPrune, prune::SeeQuiet,
                                     prune::SeeCapture, prune::QsSee, prune::LMR})
        check(guide_calls[h] > 0, std::string("guide: consulted for ") + prune::NAMES[h]);
    check(guide_bad_context == 0, "guide: always given a position and a depth");

    Position pos;
    pos.set_fen(fen);
    prune::Context c;
    c.pos = &pos;
    c.depth = 5;
    c.move_index = 10;
    c.quiet = true;
    const int rfp = prune::rfp_margin(c);
    const int lmr = prune::lmr_reduction(c);
    prune::guide = extreme_guide;
    const int w = std::max(std::abs(rfp) * params::guide_margin_pct / 100, params::guide_margin_min);
    check(prune::rfp_margin(c) == rfp + w, "guide: a margin held to its band");
    check(prune::lmr_reduction(c) == lmr + params::guide_reduction_delta,
          "guide: a reduction held to its band");
    prune::guide = nullptr;
}

// A network of random weights for the tests: `kb` king buckets with
// `layout` (by king square in its side's view), `ob` output buckets; with
// `header`, our header in front, else bullet's bare format (kb = ob = 1).
void write_test_net(const char* file, int hidden, int kb, bool mirrored, int ob,
                    const std::array<std::uint8_t, 64>& layout, bool header, bool pawns = false,
                    int layer1 = 0, int layer2 = 0, unsigned families = 0) {
    std::uint64_t x = 12345;
    auto next = [&] {
        x = x * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<int>((x >> 33) % 101) - 50;
    };
    std::string bytes;
    auto put16 = [&](int w) {
        const auto u = static_cast<std::uint16_t>(static_cast<std::int16_t>(w));
        bytes += static_cast<char>(u & 0xFF);
        bytes += static_cast<char>(u >> 8);
    };
    auto put_float = [&](float f) {
        char b[4];
        std::memcpy(b, &f, 4);
        bytes.append(b, 4);
    };
    if (header) {
        bytes += "FCNN";
        put16(layer1 ? 3 : pawns || families ? 2 : 1);
        put16(hidden);
        bytes += static_cast<char>(kb);
        bytes += static_cast<char>(mirrored ? 1 : 0);
        bytes += static_cast<char>(ob);
        bytes += static_cast<char>(layer1 ? 1 : 0);  // activation
        for (const std::uint8_t b : layout) bytes += static_cast<char>(b);
        if (pawns || families || layer1) {  // version 2: extra feature sets (bit 0 pawn predicates, the rest the extra families)
            const unsigned bits = (pawns ? 1u : 0u) | families;
            put16(static_cast<int>(bits & 0xFFFF));
            put16(static_cast<int>(bits >> 16));
        }
        if (layer1) {  // version 3: the hidden layers' sizes
            put16(layer1);
            put16(layer2);
        }
    }
    const std::size_t start = bytes.size();
    for (int i = 0; i < (nnue::INPUTS * kb + (pawns ? pawnfeat::FEATURES : 0) + pairfeat::block_size(families & pairfeat::ALL) + sqfeat::block_size(families & sqfeat::ALL) + (families & nnue::PAWN_SPANS_BIT ? pawnfeat::SPAN_FEATURES : 0) + (families & sqfeat::Passers ? pawnfeat::PASSER_FEATURES : 0)) * hidden; ++i) put16(next());
    for (int i = 0; i < hidden; ++i) put16(next() + 60);
    if (layer1) {
        const auto f = [&] { return static_cast<float>(next()) / 50.0f; };
        for (int i = 0; i < ob * layer1 * hidden; ++i) put_float(f());
        for (int i = 0; i < ob * layer1; ++i) put_float(f() / 2);
        for (int i = 0; i < ob * layer2 * layer1; ++i) put_float(f());
        for (int i = 0; i < ob * layer2; ++i) put_float(f() / 2);
        for (int i = 0; i < ob * layer2; ++i) put_float(f());
        for (int i = 0; i < ob; ++i) put_float(f());
    } else {
        for (int i = 0; i < 2 * hidden * ob; ++i) put16(next());
        for (int i = 0; i < ob; ++i) put16(next() * 10);
    }
    while ((bytes.size() - start) % 64) bytes += '\0';
    std::ofstream(file, std::ios::binary) << bytes;
}

// A FEN mirrored left-right (only for positions without castling rights).
std::string mirror_fen(const std::string& fen) {
    std::istringstream in(fen);
    std::string board, rest;
    in >> board;
    std::getline(in, rest);
    std::string out, rank;
    std::stringstream rs(board);
    bool first = true;
    while (std::getline(rs, rank, '/')) {
        std::string expanded;
        for (char ch : rank)
            if (std::isdigit(static_cast<unsigned char>(ch))) expanded.append(static_cast<std::size_t>(ch - '0'), '.');
            else expanded += ch;
        std::reverse(expanded.begin(), expanded.end());
        std::string packed;
        int empty = 0;
        for (char ch : expanded) {
            if (ch == '.') { ++empty; continue; }
            if (empty) packed += static_cast<char>('0' + empty), empty = 0;
            packed += ch;
        }
        if (empty) packed += static_cast<char>('0' + empty);
        out += (first ? "" : "/") + packed;
        first = false;
    }
    return out + rest;
}

// The checks every network must pass: incremental updates equal to a
// refresh along a random game (captures, castling, promotions and king
// moves across buckets included) and colour symmetry.
void check_net(const std::string& label,
               const char* start = "r3k2r/pPppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1") {
    Position pos;
    pos.set_fen(start);
    Evaluator e;
    e.reset(pos);
    std::vector<StateInfo> st(200);
    std::uint64_t r = 7;
    bool same = e.evaluate(pos) == nnue::evaluate(pos);
    for (std::size_t ply = 0; ply < st.size() && same; ++ply) {
        MoveList moves;
        int count = 0;
        generate_legal(pos, moves, count);
        if (count == 0) break;
        r = r * 6364136223846793005ULL + 1;
        pos.make_move(moves[static_cast<std::size_t>((r >> 33) % static_cast<std::uint64_t>(count))], st[ply]);
        e.push(st[ply].changes);
        // Evaluate only every third ply, so stale views travel down the stack.
        if (ply % 3 == 0) same = e.evaluate(pos) == nnue::evaluate(pos) && e.evaluate(pos) == evaluate(pos);
    }
    check(same, "nnue (" + label + "): incremental updates match a refresh");

    // As a search walks: at each ply a few siblings tried and taken back
    // (some evaluated, some not, their children too), then one played on;
    // the accumulators must never keep a taken-back sibling's values.
    pos.set_fen(start);
    e.reset(pos);
    std::vector<StateInfo> sib(2);
    bool walk = true;
    for (std::size_t ply = 0; ply < 120 && walk; ++ply) {
        MoveList moves;
        int count = 0;
        generate_legal(pos, moves, count);
        if (count == 0) break;
        for (int k = 0; k < 3 && walk; ++k) {
            r = r * 6364136223846793005ULL + 1;
            const Move m = moves[static_cast<std::size_t>((r >> 33) % static_cast<std::uint64_t>(count))];
            pos.make_move(m, sib[0]);
            e.push(sib[0].changes);
            if ((r >> 20) & 1) walk = e.evaluate(pos) == nnue::evaluate(pos);
            MoveList replies;
            int n = 0;
            generate_legal(pos, replies, n);
            if (n && walk) {
                const Move m2 = replies[static_cast<std::size_t>((r >> 40) % static_cast<std::uint64_t>(n))];
                pos.make_move(m2, sib[1]);
                e.push(sib[1].changes);
                walk = e.evaluate(pos) == nnue::evaluate(pos);
                e.pop();
                pos.unmake_move(m2, sib[1]);
            }
            e.pop();
            pos.unmake_move(m, sib[0]);
        }
        r = r * 6364136223846793005ULL + 1;
        pos.make_move(moves[static_cast<std::size_t>((r >> 33) % static_cast<std::uint64_t>(count))], st[ply]);
        e.push(st[ply].changes);
        if (ply % 2 == 0 && walk) walk = e.evaluate(pos) == nnue::evaluate(pos);
    }
    check(walk, "nnue (" + label + "): updates across taken-back moves match a refresh");

    bool symmetric = true;
    for (const char* fen : {"r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
                            "6k1/5ppp/8/3N4/8/8/5PPP/3R2K1 b - - 0 30"}) {
        Position a, b;
        a.set_fen(fen);
        b.set_fen(flip_fen(fen));
        symmetric = symmetric && nnue::evaluate(a) == nnue::evaluate(b);
    }
    check(symmetric, "nnue (" + label + "): colour-symmetric");
}

// The network evaluation, on small networks of random weights written here:
// loading, the checks above for a bare file and for king and output buckets
// (with left-right mirroring), and the hand-written evaluation back once
// unloaded.
void test_nnue() {
    const int hidden = 16;
    const char* file = "nnue_test.bin";
    std::array<std::uint8_t, 64> flat{};
    write_test_net(file, hidden, 1, false, 1, flat, false);
    {
        std::ofstream bad("nnue_test_bad.bin", std::ios::binary);
        bad << std::string(5000, 'x');
    }
    check(!nnue::load("nnue_test_bad.bin") && !nnue::active(), "nnue: a file of the wrong size is refused");
    check(nnue::load(file) && nnue::active() && nnue::hidden() == hidden && nnue::king_buckets() == 1 &&
              nnue::output_buckets() == 1,
          "nnue: a bare file loaded, hidden size from its size");
    check_net("bare");

    // The first layer shared from the mapped file (nnue::share()): the same
    // values, so the same evaluations and every check again. A file changed
    // afterwards is noticed where the system allows the change at all
    // (Windows refuses writes to a shared file). The network is unloaded
    // before its file is written anew below.
    {
        Position p;
        p.set_fen("r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 3 9");
        const int before = nnue::evaluate(p);
        check(!nnue::shared() && nnue::shared_intact() && nnue::share() && nnue::shared() && nnue::share() &&
                  nnue::shared_intact() && nnue::evaluate(p) == before,
              "nnue: the first layer shared from the file, the same evaluation");
        check_net("bare, shared");
        bool written = false;
        {
            std::fstream f(file, std::ios::in | std::ios::out | std::ios::binary);
            char c = 0;
            if (f && f.seekg(10) && f.get(c)) {
                f.seekp(10);
                f.put(static_cast<char>(c ^ 0x55));
                f.flush();
                written = static_cast<bool>(f);
            }
        }
#if defined(__linux__)
        check(written && !nnue::shared_intact(), "nnue: a shared network file changed on disk is noticed");
#else
        check(!written || !nnue::shared_intact() || nnue::evaluate(p) == before,
              "nnue: a shared network file is protected from change, or the change noticed");
#endif
        nnue::load("");
        check(!nnue::shared() && !nnue::share() && nnue::shared_intact(), "nnue: nothing to share without a network");
        write_test_net(file, hidden, 1, false, 1, flat, false);
        check(nnue::load(file) && !nnue::shared(), "nnue: a network loaded anew is private again");
    }

    // The author's 7 mirrored king buckets (back rank a-b|g-h, c|f, d|e;
    // the same on rank 2; the rest), 8 output buckets.
    std::array<std::uint8_t, 64> seven{};
    for (int sq = 0; sq < 64; ++sq) {
        const int file_ = std::min(sq & 7, 7 - (sq & 7)), rank = sq >> 3;
        const int wing = file_ <= 1 ? 0 : file_ == 2 ? 1 : 2;
        seven[static_cast<std::size_t>(sq)] = static_cast<std::uint8_t>(rank == 0 ? wing : rank == 1 ? 3 + wing : 6);
    }
    write_test_net(file, hidden, 7, true, 8, seven, true);
    check(nnue::load(file) && nnue::king_buckets() == 7 && nnue::output_buckets() == 8,
          "nnue: a headed file loaded, king and output buckets");
    check_net("7 mirrored king buckets, 8 output buckets");
    bool mirrored = true;
    for (const char* fen : {"6k1/5ppp/8/3N4/8/8/5PPP/3R2K1 b - - 0 30",
                            "2kr3r/ppq2ppp/2n1bn2/2b1p3/4P3/2N1BN2/PPPQBPPP/1K1R3R w - - 6 12"}) {
        Position a, b;
        a.set_fen(fen);
        b.set_fen(mirror_fen(fen));
        mirrored = mirrored && nnue::evaluate(a) == nnue::evaluate(b);
    }
    check(mirrored, "nnue: mirrored buckets, left-right symmetric");

    // The same with the pawn predicates as extra inputs (header version 2).
    write_test_net(file, hidden, 7, true, 8, seven, true, true);
    check(nnue::load(file) && nnue::pawn_features(), "nnue: a version 2 file with pawn predicates loaded");
    check_net("pawn predicates");
    bool pawn_mirror = true;
    for (const char* fen : {"6k1/5ppp/8/3N4/8/8/5PPP/3R2K1 b - - 0 30",
                            "2kr3r/ppq2ppp/2n1bn2/2b1p3/4P3/2N1BN2/PPPQBPPP/1K1R3R w - - 6 12"}) {
        Position a, b;
        a.set_fen(fen);
        b.set_fen(mirror_fen(fen));
        pawn_mirror = pawn_mirror && nnue::evaluate(a) == nnue::evaluate(b);
    }
    check(pawn_mirror, "nnue: pawn predicates, left-right symmetric");

    // The pairwise multi-layer network (header version 3), with the pawn
    // predicates too: the same checks.
    write_test_net(file, hidden, 7, true, 8, seven, true, true, 8, 8);
    check(nnue::load(file) && nnue::pawn_features(), "nnue: a version 3 multi-layer file loaded");
    check_net("multi-layer");
    bool multi_mirror = true;
    for (const char* fen : {"6k1/5ppp/8/3N4/8/8/5PPP/3R2K1 b - - 0 30",
                            "2kr3r/ppq2ppp/2n1bn2/2b1p3/4P3/2N1BN2/PPPQBPPP/1K1R3R w - - 6 12"}) {
        Position a, b;
        a.set_fen(fen);
        b.set_fen(mirror_fen(fen));
        multi_mirror = multi_mirror && nnue::evaluate(a) == nnue::evaluate(b);
    }
    check(multi_mirror, "nnue: multi-layer, left-right symmetric");

    // layer1 in int8 and the later layers in SIMD against the plain float layers they replace (NnueLayerFloat,
    // 2026-10-18): at hidden 16 the plain layer1, at 64 the AVX2 one (32 values a half, in an AVX2 build);
    // layer2's blocks of 8, 16, 32 and 64 outputs; every position within the rounding of the stored weights
    // (check-nets' limit: 8 cp and 5% of the evaluation), and the two paths not identical. Since 2026-10-19
    // layer2 is int8 too in AVX2 builds: the limit 16 cp and 5% (these random nets reach 13; the real ones
    // gain 0.4-0.8 cp on average over int8 layer1 alone, largest differences alike).
    for (const auto [hid, l1, l2] : {std::array<int, 3>{hidden, 16, 8}, {64, 16, 8}, {64, 16, 16}, {64, 32, 32}, {64, 64, 64}}) {
        write_test_net(file, hid, 7, true, 8, seven, true, true, l1, l2);
        bool within = nnue::load(file), differ = false;
        int worst = 0;
        for (const char* fen : {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                                "6k1/5ppp/8/3N4/8/8/5PPP/3R2K1 b - - 0 30",
                                "2kr3r/ppq2ppp/2n1bn2/2b1p3/4P3/2N1BN2/PPPQBPPP/1K1R3R w - - 6 12",
                                "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R2QKB1R w KQ - 0 8",
                                "8/5k2/3p4/1p1Pp2p/pP2Pp1P/P4P1K/8/8 b - - 99 50"}) {
            Position p;
            p.set_fen(fen);
            params::nnue_layer_float = 1;
            const int f = nnue::evaluate(p);
            params::nnue_layer_float = 0;
            const int q = nnue::evaluate(p);
            worst = std::max(worst, std::abs(f - q));
            within = within && std::abs(f - q) <= 16 + std::abs(f) / 20;
            differ = differ || f != q;
        }
        check(within && differ, "nnue: layer1 in int8 within rounding of the float layer (" + std::to_string(hid) + "x" +
                                    std::to_string(l1) + "x" + std::to_string(l2) + ", largest difference " +
                                    std::to_string(worst) + " cp)");
    }

    // Pair features (pair_features.h): the pairs and their numbering, the
    // family bound, then networks with them (and the pawn predicates) along
    // random games heavy in knights and in sliders.
    int knight_n = 0, ortho_n = 0, diag_n = 0, file_n = 0;
    for (int a = 0; a < 64; ++a)
        for (int b = 0; b < 64; ++b) {
            if (a < b) {
                knight_n += pairfeat::knight_pair_index(a, b) >= 0;
                ortho_n += pairfeat::ortho_pair_index(a, b) >= 0;
                diag_n += pairfeat::diag_pair_index(a, b) >= 0;
            }
            file_n += pairfeat::file_pair_index(a, b) >= 0;
        }
    std::uint16_t kp[pairfeat::MAX_ACTIVE];
    const int b1 = 1, d2 = 11, f3 = 21, c6 = 42;
    const int n_kp = pairfeat::knight_pairs(bit(b1) | bit(d2) | bit(f3), bit(c6), kp);
    int pa = 0, pb = 0;
    pairfeat::knight_pair_squares(pairfeat::knight_pair_index(f3, d2), pa, pb);
    check(knight_n == pairfeat::KNIGHT_PAIRS && ortho_n == pairfeat::ORTHO_PAIRS && diag_n == pairfeat::DIAG_PAIRS &&
              file_n == pairfeat::FILE_PAIRS && n_kp == 2 && kp[0] == pairfeat::knight_pair_index(b1, d2) &&
              kp[1] == pairfeat::knight_pair_index(d2, f3) && pa == d2 && pb == f3 &&
              pairfeat::knight_pair_index(b1, c6) < 0 && pairfeat::file_pair_index(3, 59) < 0,
          "pair features: pairs found and numbered");
    {
        // Rooks d1, d2 and a black rook d8 (file pairs, own and mixed), a
        // queen and bishop on b7-c6 (diagonal), rook d2 and queen c3 (no
        // line both move along): seven pairs in all; pawns on the d-file.
        Position p;
        p.set_fen("3r2k1/1b3ppp/2q5/p2Pp3/P3P3/2Q2B2/3R1PPP/3R2K1 w - - 0 25");
        std::uint32_t in[pairfeat::MAX_ACTIVE];
        const int n_b = pairfeat::inputs(p, pairfeat::SliderPairs, Color::White, 0, in);
        const int d1 = 3, d8 = 59, b7 = 49, d5 = 35;
        const int rr = pairfeat::ortho_pair_index(d1, d2);  // own, own, no queens
        const int mixed = (0 * 2 + 1) * 4 * pairfeat::ORTHO_PAIRS + pairfeat::ortho_pair_index(d1, d8);
        const int qb = 16 * pairfeat::ORTHO_PAIRS + ((1 * 2 + 1) * 4 + 1 * 2 + 0) * pairfeat::DIAG_PAIRS +
                       pairfeat::diag_pair_index(b7, c6);
        const std::set<int> got(in, in + n_b);
        const int n_c = pairfeat::inputs(p, pairfeat::SliderPawn, Color::White, 0, in);
        // White rook d1 with the white pawn d5: (own, rook, own pawn).
        const std::set<int> got_c(in, in + n_c);
        // Connected (D): queens c3-c6, rooks d1-d2, bishop and queen b7-c6.
        const int n_d = pairfeat::inputs(p, pairfeat::ConnectedPairs, Color::White, 0, in);
        check(n_b == 7 && got.contains(rr) && got.contains(mixed) && got.contains(qb) && n_c == 3 &&
                  got_c.contains(pairfeat::file_pair_index(d1, d5)) && n_d == 3,
              "pair features: slider pairs and slider x pawn found");
        Position many, fewer;
        many.set_fen("k7/8/8/8/8/8/QQQQQQQQ/QQQQKQQQ b - - 0 1");   // 68 slider pairs
        fewer.set_fen("k7/8/8/8/8/8/QQQQQQQQ/4K3 b - - 0 1");       // 28
        check(pairfeat::inputs(many, pairfeat::SliderPairs, Color::White, 0, in) == 0 &&
                  pairfeat::inputs(fewer, pairfeat::SliderPairs, Color::Black, 7, in) == 28,
              "pair features: a family past its bound contributes nothing");
    }
    const char* knights = "r3k2r/1n1n1n2/2n2n2/8/8/2N2N2/1N1N1N2/R3K2R w KQkq - 0 1";
    const char* sliders = "r2qk2r/1b1r1b2/2q5/pp1PP1pp/PP1pp1PP/2Q5/1B1R1B2/R2QK2R w KQkq - 0 1";
    write_test_net(file, hidden, 7, true, 8, seven, true, true, 0, 0, pairfeat::KnightPairs);
    check(nnue::load(file) && nnue::pair_families() == pairfeat::KnightPairs && nnue::pawn_features(),
          "nnue: a version 2 file with knight pairs loaded");
    check_net("knight pairs");
    check_net("knight pairs, many knights", knights);
    write_test_net(file, hidden, 7, true, 8, seven, true, true, 0, 0, pairfeat::ALL);
    check(nnue::load(file) && nnue::pair_families() == pairfeat::ALL, "nnue: a file with every pair family loaded");
    check_net("every pair family");
    check_net("every pair family, many knights", knights);
    check_net("every pair family, many sliders", sliders);
    bool pair_mirror = true;
    for (const char* fen : {"6k1/5ppp/2n5/3Nn3/5N2/8/5PPP/3R2K1 b - - 0 30",
                            "3r2k1/1b3ppp/2q5/p2Pp3/P3P3/2Q2B2/3R1PPP/3R2K1 w - - 0 25",
                            "3r2k1/1b3ppp/2q5/p2Pp3/P3P3/2Q2B2/3R1PPP/3R2K1 b - - 0 25"}) {
        Position a, b, c;
        a.set_fen(fen);
        b.set_fen(mirror_fen(fen));
        c.set_fen(flip_fen(fen));
        pair_mirror = pair_mirror && nnue::evaluate(a) == nnue::evaluate(b) && nnue::evaluate(a) == nnue::evaluate(c);
    }
    check(pair_mirror, "nnue: pair families, left-right and colour symmetric");
    write_test_net(file, hidden, 7, true, 8, seven, true, false, 8, 8, pairfeat::ALL);
    check(nnue::load(file) && nnue::pair_families() == pairfeat::ALL && !nnue::pawn_features(),
          "nnue: a multi-layer file with pair families loaded");
    check_net("multi-layer, pair families", sliders);

    // Square control (square_features.h): the families' sizes in a quiet
    // position and a known pawn-piece predicate, a file declaring the retired
    // threat family refused, then a network with every extra family along
    // random games.
    {
        std::uint32_t in[sqfeat::MAX_ACTIVE];
        Position start;
        const int n_s1 = sqfeat::inputs(start, sqfeat::Attacked, Color::White, 0, in);
        const int n_s2 = sqfeat::inputs(start, sqfeat::LeastAttacker, Color::White, 0, in);
        // A2: the d5 pawn attacked by the c3 knight, blocked by the d4 bishop.
        Position q;
        q.set_fen("4k3/8/8/3p4/3B4/2N5/8/4K3 w - - 0 1");
        const int n_a2 = sqfeat::inputs(q, sqfeat::PawnPieces, Color::White, 0, in);
        const bool a2 = n_a2 == 2 && in[0] == (2 * 2 + 1) * 64 + 35 && in[1] == (3 * 2 + 1) * 64 + 35;
        check(n_s1 == 44 && a2 && n_s2 == 44, "square features: attacked squares, least attackers, pawn-piece predicates");
    }
    write_test_net(file, hidden, 7, true, 8, seven, true, true, 0, 0, sqfeat::RETIRED_THREATS);
    check(!nnue::load(file), "nnue: a file declaring the retired threat family refused");
    write_test_net(file, hidden, 7, true, 8, seven, true, true, 0, 0, pairfeat::ALL | sqfeat::ALL | nnue::PAWN_SPANS_BIT);
    check(nnue::load(file) && nnue::pair_families() == pairfeat::ALL && nnue::square_families() == sqfeat::ALL,
          "nnue: a file with square control loaded");
    check_net("square control");
    check_net("square control, many sliders", sliders);
    bool square_mirror = true;
    for (const char* fen : {"3r2k1/1b3ppp/2q5/p2Pp3/P3P3/2Q2B2/3R1PPP/3R2K1 w - - 0 25",
                            "6k1/5ppp/2n5/3Nn3/5N2/8/5PPP/3R2K1 b - - 0 30"}) {
        Position a, b, c;
        a.set_fen(fen);
        b.set_fen(mirror_fen(fen));
        c.set_fen(flip_fen(fen));
        square_mirror = square_mirror && nnue::evaluate(a) == nnue::evaluate(b) && nnue::evaluate(a) == nnue::evaluate(c);
    }
    check(square_mirror, "nnue: square control, left-right and colour symmetric");
    check(nnue::pawn_span_features(), "nnue: the pawn-control map loaded");
    {
        // King-state buckets (15, by location x shelter, castling rights a
        // location) and output buckets split by the side to move's shelter
        // (8 x 3): incremental updates (rechecks after king, pawn and rook
        // moves) equal a refresh along random games, colour and left-right
        // symmetric.
        write_test_net(file, hidden, nnue::KING_STATES, true, 24, seven, true, false, 0, 0,
                       nnue::KING_STATE_BIT | nnue::KING_STATE_OUTPUT_BIT);
        check(nnue::load(file) && nnue::king_state_buckets() && nnue::king_state_output_buckets() &&
                  nnue::king_buckets() == nnue::KING_STATES,
              "nnue: a file with king-state buckets and outputs loaded");
        check_net("king state");
        check_net("king state, castling and pawn storms",
                  "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1");
        bool ks_symmetric = true;
        for (const char* fen : {"r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w - - 0 1",
                                "6k1/5p1p/6p1/8/8/8/5PPP/6K1 b - - 0 30"}) {
            Position a, b, c;
            a.set_fen(fen);
            b.set_fen(mirror_fen(fen));
            c.set_fen(flip_fen(fen));
            ks_symmetric = ks_symmetric && nnue::evaluate(a) == nnue::evaluate(b) && nnue::evaluate(a) == nnue::evaluate(c);
        }
        check(ks_symmetric, "nnue: king state, left-right and colour symmetric");
        // Merged king states (fewer blocks, each state's block in the layout's first 15 bytes): the same
        // checks, and a 15-block net whose states share their block's weights evaluates exactly alike.
        const std::array<std::uint8_t, 15> merge12{0, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 10, 11};
        std::array<std::uint8_t, 64> map{};
        std::copy(merge12.begin(), merge12.end(), map.begin());
        write_test_net(file, hidden, 12, true, 8, map, true, false, 0, 0, nnue::KING_STATE_BIT);
        check(nnue::load(file) && nnue::king_state_buckets() && nnue::king_buckets() == 12,
              "nnue: a file with merged king states loaded");
        check_net("merged king states", "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1");
        std::vector<int> merged_evals;
        const char* merge_fens[] = {"r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1",
                                    "2r3k1/5ppp/8/8/8/8/3K1PPP/8 w - - 0 40",   // White's king centre, rank 2
                                    "6k1/8/8/3K4/8/8/8/8 w - - 0 60",           // advanced, no pawns
                                    "r4rk1/ppp2ppp/8/8/8/8/PPPP1PPP/2KR3R b - - 0 20",  // c-file, covered (state 3)
                                    "4k3/8/8/8/8/8/3PPP2/4K3 w - - 0 1"};       // centre, covered (state 6)
        for (const char* fen : merge_fens) {
            Position p;
            p.set_fen(fen);
            merged_evals.push_back(nnue::evaluate(p));
        }
        // The same weights spread over 15 blocks: block b's 768 x hidden piece weights copied to every state in it.
        {
            std::ifstream in(file, std::ios::binary);
            std::string bytes((std::istreambuf_iterator<char>(in)), {});
            const std::size_t head = nnue::HEADER_BYTES + 4,
                              block = static_cast<std::size_t>(nnue::INPUTS) * static_cast<std::size_t>(hidden) * 2;
            std::string out = bytes.substr(0, head);
            out[8] = static_cast<char>(nnue::KING_STATES);
            for (std::size_t s = 0; s < 15; ++s) out[12 + s] = 0;
            for (std::size_t s = 0; s < 15; ++s) out += bytes.substr(head + merge12[s] * block, block);
            out += bytes.substr(head + 12 * block);
            std::ofstream(file, std::ios::binary) << out;
        }
        bool same = nnue::load(file) && nnue::king_buckets() == nnue::KING_STATES;
        for (std::size_t i = 0; i < std::size(merge_fens); ++i) {
            Position p;
            p.set_fen(merge_fens[i]);
            same = same && nnue::evaluate(p) == merged_evals[i];
        }
        check(same, "nnue: merged king states evaluate as 15 states sharing their block's weights");
        // A merge table naming a block past the count, or leaving one unused, is refused.
        map[0] = 12;
        write_test_net(file, hidden, 12, true, 8, map, true, false, 0, 0, nnue::KING_STATE_BIT);
        check(!nnue::load(file), "nnue: a merge table past the block count is refused");
        map[0] = 0, map[14] = 10;
        write_test_net(file, hidden, 12, true, 8, map, true, false, 0, 0, nnue::KING_STATE_BIT);
        check(!nnue::load(file), "nnue: a merge table leaving a block unused is refused");
        // A file declaring king-state buckets with any other bucket count is refused (7 blocks, no table).
        write_test_net(file, hidden, 7, true, 8, seven, true, false, 0, 0, nnue::KING_STATE_BIT);
        check(!nnue::load(file), "nnue: king-state buckets need 15 blocks or a merge table");
    }
    {
        // The bare king state (5 blocks: location, castling rights one of
        // them; no shelter, so pawn moves never switch blocks): incremental
        // updates (rechecks after king and rook moves) equal a refresh,
        // with pawn storms and castling; colour and left-right symmetric.
        write_test_net(file, hidden, nnue::KING_LOCATIONS, true, 8, std::array<std::uint8_t, 64>{}, true, false, 0, 0,
                       nnue::KING_LOCATION_BIT);
        check(nnue::load(file) && nnue::king_location_buckets() && !nnue::king_state_buckets() &&
                  nnue::king_buckets() == nnue::KING_LOCATIONS,
              "nnue: a file with bare king-state buckets loaded");
        check_net("bare king state");
        check_net("bare king state, castling and pawn storms",
                  "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1");
        bool kl_symmetric = true;
        for (const char* fen : {"r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w - - 0 1",
                                "6k1/5p1p/6p1/8/8/8/5PPP/6K1 b - - 0 30"}) {
            Position a, b, c;
            a.set_fen(fen);
            b.set_fen(mirror_fen(fen));
            c.set_fen(flip_fen(fen));
            kl_symmetric = kl_symmetric && nnue::evaluate(a) == nnue::evaluate(b) && nnue::evaluate(a) == nnue::evaluate(c);
        }
        check(kl_symmetric, "nnue: bare king state, left-right and colour symmetric");
        // Any other block count, or both selectors at once, is refused.
        write_test_net(file, hidden, 7, true, 8, seven, true, false, 0, 0, nnue::KING_LOCATION_BIT);
        check(!nnue::load(file), "nnue: bare king-state buckets need 5 blocks");
        write_test_net(file, hidden, nnue::KING_LOCATIONS, true, 8, std::array<std::uint8_t, 64>{}, true, false, 0, 0,
                       nnue::KING_LOCATION_BIT | nnue::KING_STATE_BIT);
        check(!nnue::load(file), "nnue: king state and bare king state together are refused");
    }
    {
        // Output buckets by endgame type (8 material + 3): the type's bucket
        // where one applies, the material bucket elsewhere; with king-state
        // outputs too, 8 x 3 + 3.
        const auto bucket = [](const char* fen) {
            Position p;
            p.set_fen(fen);
            return nnue::output_bucket(p);
        };
        const char* start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        const char* pawnless = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1";
        const char* pawns_only = "4k3/4p3/8/8/8/8/4P3/4K3 w - - 0 1";
        const char* opposite = "4k3/4p3/4b3/8/8/2B5/4P3/4K3 w - - 0 1";  // e6 light, c3 dark
        const char* same = "4k3/4p3/3b4/8/8/2B5/4P3/4K3 w - - 0 1";      // d6 and c3 dark
        write_test_net(file, hidden, 7, true, 11, seven, true, false, 0, 0, nnue::ENDGAME_OUTPUT_BIT);
        check(nnue::load(file) && nnue::endgame_output_buckets() && bucket(start) == 7 && bucket(pawnless) == 8 &&
                  bucket(pawns_only) == 9 && bucket(opposite) == 10 && bucket(same) == 1,
              "nnue: output buckets by endgame type");
        write_test_net(file, hidden, 7, true, 27, seven, true, false, 0, 0,
                       nnue::ENDGAME_OUTPUT_BIT | nnue::KING_STATE_OUTPUT_BIT);
        check(nnue::load(file) && bucket(opposite) == 26 && bucket(start) < 24,
              "nnue: endgame types with king-state outputs");
        write_test_net(file, hidden, 7, true, 3, seven, true, false, 0, 0, nnue::ENDGAME_OUTPUT_BIT);
        check(!nnue::load(file), "nnue: endgame types need material buckets too");
    }
    {
        // Passed-pawn quality. Pawn-only: a2 outside, d5 connected, e6
        // protected and connected; Black's h7 outside.
        std::uint16_t f[pawnfeat::PASSER_MAX];
        Position p;
        p.set_fen("4k3/7p/4P3/3P4/8/8/P7/4K3 w - - 0 1");
        const int n = pawnfeat::passer_features(p.pieces(WP), p.pieces(BP), f);
        const std::set<int> pawn_only(f, f + n);
        // Pieces: e5 blockaded by a knight, a rook behind it; Black's a3 with
        // a clear path (not safe: the rook reaches a1). Both views.
        const auto pieces = [](const char* fen, Color view) {
            Position q;
            q.set_fen(fen);
            std::uint32_t in[sqfeat::MAX_ACTIVE];
            const int k = sqfeat::inputs(q, sqfeat::Passers, view, 0, in);
            return std::set<std::uint32_t>(in, in + k);
        };
        const char* blockade = "4k3/8/4n3/4P3/8/p7/8/4R1K1 w - - 0 1";
        // Unstoppable by the rule of the square: a5 against a king on e6 with
        // White to move, catchable with Black to move.
        const char* square_w = "8/8/4k3/P7/8/8/8/7K w - - 0 1";
        const char* square_b = "8/8/4k3/P7/8/8/8/7K b - - 0 1";
        check(pawn_only == std::set<int>{44, 163, 172, 264, 375} &&
                  pieces(blockade, Color::White) == std::set<std::uint32_t>{36, 464, 676} &&
                  pieces(blockade, Color::Black) == std::set<std::uint32_t>{92, 424, 732} &&
                  pieces(square_w, Color::White) == std::set<std::uint32_t>{416, 544, 928} &&
                  pieces(square_b, Color::White) == std::set<std::uint32_t>{416, 544},
              "passed-pawn quality in known positions");
        write_test_net(file, hidden, 7, true, 8, seven, true, true, 0, 0, sqfeat::Passers | nnue::PAWN_SPANS_BIT);
        check(nnue::load(file) && nnue::passer_features(), "nnue: a file with passed-pawn quality loaded");
        check_net("passed-pawn quality", "8/2p2pk1/1p4p1/1PP5/P2n3P/5PK1/3B4/8 w - - 0 1");
        bool passer_symmetric = true;
        for (const char* fen : {blockade, "8/2p2pk1/1p4p1/1PP5/P2n3P/5PK1/3B4/8 w - - 0 1"}) {
            Position a, b, c;
            a.set_fen(fen);
            b.set_fen(mirror_fen(fen));
            c.set_fen(flip_fen(fen));
            passer_symmetric = passer_symmetric && nnue::evaluate(a) == nnue::evaluate(b) && nnue::evaluate(a) == nnue::evaluate(c);
        }
        check(passer_symmetric, "nnue: passed-pawn quality, left-right and colour symmetric");
    }
    {
        // Line control: White's rook on the open e-file (both e-pawns gone),
        // no other line with a major, bishop or queen, every other file closed.
        Position p;
        p.set_fen("4k3/pppp1ppp/8/8/8/8/PPPP1PPP/4R1K1 w - - 0 1");
        std::uint32_t in[pairfeat::MAX_ACTIVE];
        const int nw = pairfeat::inputs(p, pairfeat::LineControl, Color::White, 0, in);
        const std::set<std::uint32_t> white(in, in + nw);
        const int nb = pairfeat::inputs(p, pairfeat::LineControl, Color::Black, 0, in);
        const std::set<std::uint32_t> black(in, in + nb);
        // e-file 4: open (0), our major (2) for White, theirs (1) for Black;
        // rank 1 is White's rank 0 and Black's rank 7.
        check(white == std::set<std::uint32_t>{4 * 16 + 0 + 2, 128 + 0 * 4 + 2} &&
                  black == std::set<std::uint32_t>{4 * 16 + 0 + 1, 128 + 7 * 4 + 1},
              "pair features: line control in a known position");
    }
    {
        // The pawn-control map: White e2 and Black e5 each reach the squares
        // beside the file up to the other's rank (each blocks the other).
        const int e2 = 12, e5 = 36;
        const Bitboard both = bit(e2) | bit(e5);
        const Bitboard white = pawnfeat::pawn_span(bit(e2), both, Color::White);
        const Bitboard black = pawnfeat::pawn_span(bit(e5), both, Color::Black);
        const auto squares = [](std::initializer_list<int> s) {
            Bitboard b = 0;
            for (const int x : s) b |= bit(x);
            return b;
        };
        const int f2 = 13, d3 = 19, f4 = 29, d4 = 27, d5 = 35, f5 = 37;  // d2, f3 as above
        // Control counts at the start: c3 by the b2 and d2 pawns and the b1
        // knight (three), h3 by the g2 pawn and the g1 knight (two).
        Position start;
        std::uint32_t in[sqfeat::MAX_ACTIVE];
        const int n = sqfeat::inputs(start, sqfeat::ControlCounts, Color::White, 0, in);
        const std::set<std::uint32_t> got(in, in + n);
        const int c3 = 18, h3 = 23;
        const auto level = [](int side, int square, int k) { return static_cast<std::uint32_t>((side * 64 + square) * 3 + k); };
        check(white == squares({d3, f3, d4, f4, d5, f5}) && black == squares({d2, f2, d3, f3, d4, f4}) &&
                  got.contains(level(0, c3, 2)) && got.contains(level(0, h3, 1)) && !got.contains(level(0, h3, 2)) &&
                  got.contains(level(1, 47, 1)),
              "pawn-control map and control counts in known positions");
    }
    {
        // Mobility. At the start each knight has two safe moves (its third
        // square is its own pawn's), nothing else any: steps 1-2 at both
        // levels, for each side. A lone knight on d4 against a pawn on c6:
        // 8 squares less b5 (the pawn guards it), c6 itself a capture.
        const auto mobility = [](const char* fen, Color view) {
            Position p;
            p.set_fen(fen);
            std::uint32_t in[sqfeat::MAX_ACTIVE];
            const int n = sqfeat::inputs(p, sqfeat::Mobility, view, 0, in);
            return std::set<std::uint32_t>(in, in + n);
        };
        const int s = sqfeat::MOBILITY_STEPS;
        const char* start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        const char* knight = "4k3/8/2p5/8/3N4/8/8/4K3 w - - 0 1";
        const auto range = [](std::uint32_t from, std::uint32_t to) {
            std::set<std::uint32_t> r;
            for (std::uint32_t i = from; i < to; ++i) r.insert(i);
            return r;
        };
        const std::set<std::uint32_t> both{0, 1, s + 0u, s + 1u, 2u * s, 2u * s + 1, 3u * s, 3u * s + 1};
        check(mobility(start, Color::White) == both && mobility(start, Color::Black) == both &&
                  mobility(knight, Color::White) == range(0, 7) &&
                  mobility(knight, Color::Black) == range(2 * s, 2 * s + 7),
              "square features: mobility in known positions");
    }

    std::array<std::uint8_t, 64> bad_layout = seven;
    bad_layout[20] = 7;  // past the last bucket
    write_test_net(file, hidden, 7, true, 8, bad_layout, true);
    check(!nnue::load(file), "nnue: a layout naming a missing bucket is refused");

    nnue::load("");
    Position pos;
    check(!nnue::active() && evaluate(pos) == eval_parts(pos).total &&
              eval_parts(pos).pesto == evaluate(pos) - eval_parts(pos).initiative - eval_parts(pos).mop_up,
          "nnue: unloaded, the hand-written evaluation");
    std::remove(file);
    std::remove("nnue_test_bad.bin");
}

int main(int argc, char** argv) {
    // "search_tests bench": the bench from this copy of the engine, whose
    // parameters are variables. ctest holds it to the same count as the
    // engine's own, frozen where FROZEN_PARAMS is on: both forms must search
    // alike.
    if (argc >= 2 && std::string(argv[1]) == "bench") {
        init_zobrist();
        if (nnue::embedded()) nnue::load("<embedded>");  // as the engine's bench does
        Searcher searcher;
        bench(searcher, BENCH_DEPTH, std::cout);
        return 0;
    }
    test_see();
    test_mop_up();
    test_null_move();
    test_mates();
    test_pure_search();
    test_pv_stops_at_draws();
    test_stop_request();
    test_lmr_parent_hist();
    test_initiative();
    test_drawish();
    test_bingo();
    test_cont_log();
    test_guide();
    test_thread_pool();
    test_pgn();
    test_ponder();
    test_tt_key();
    test_tt_clear();
    test_root_move_nodes();
    test_empty_clock();
    test_fen_clocks();
    test_reversible_move();
    test_picker_completeness();
    test_verification();
    test_label_sampling();
    test_label_rows();
    test_features();
    test_tablebases();
    test_datagen();
    test_pawn_features();
    test_nnue();
    if (failures) {
        std::cerr << failures << " search test(s) failed.\n";
        return 1;
    }
    std::cout << "All search tests passed.\n";
    return 0;
}
