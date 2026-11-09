#include "position.h"
#include "movegen.h"
#include "perft.h"
#include "eval.h"
#include <algorithm>
#include <cstdint>
#include <vector>
#include <iostream>

struct Case {
    const char* name;
    const char* fen;
    int depth;
    std::uint64_t expected;
};

namespace {
// Standard perft positions (chessprogramming.org "Perft Results").
constexpr const char* STARTPOS =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
// Kiwipete: castling, en passant, pins.
constexpr const char* KIWIPETE =
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
// Position 3: en passant, discovered checks along ranks.
constexpr const char* POS3 = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
// Position 4: promotions (including capture-promotions) and castling.
constexpr const char* POS4 =
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
// Position 5: promotion with check, castling rights edge cases.
constexpr const char* POS5 =
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";

// Captures and quiets must split the legal moves exactly: no move missing,
// none in both, and captures really are captures or promotions.
bool partition_ok(const Position& p, const MoveList& all, int all_count) {
    MoveList caps, quiets;
    int nc = 0, nq = 0;
    generate_captures(p, caps, nc);
    generate_quiets(p, quiets, nq);
    if (nc + nq != all_count) return false;

    std::vector<std::uint16_t> a, b;
    for (int i = 0; i < all_count; ++i) a.push_back(all[static_cast<std::size_t>(i)].data);
    for (int i = 0; i < nc; ++i) {
        const Move m = caps[static_cast<std::size_t>(i)];
        const bool is_capture = p.piece_at(m.to()) != NO_PIECE || m.is_en_passant();
        if (!is_capture && m.promotion() == PieceType::None) return false;
        b.push_back(m.data);
    }
    for (int i = 0; i < nq; ++i) {
        const Move m = quiets[static_cast<std::size_t>(i)];
        if (p.piece_at(m.to()) != NO_PIECE || m.is_en_passant() ||
            m.promotion() != PieceType::None) return false;
        b.push_back(m.data);
    }
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    return a == b;
}

// Walks the tree and checks, at every node, that count_legal agrees with
// generate_legal, that the incrementally updated mailbox, occupancy,
// Zobrist keys and evaluation match a from-scratch recomputation, that
// gives_check predicts exactly the moves that leave the opponent in check,
// and that unmake_move restores the keys exactly.
bool walk_consistent(Position& p, Evaluator& ev, int depth) {
    if (!p.is_consistent()) return false;
    if (ev.evaluate(p) != evaluate(p)) {
        std::cerr << "  incremental eval " << ev.evaluate(p) << " != full eval "
                  << evaluate(p) << " in " << p.fen() << '\n';
        return false;
    }
    if (depth == 0) return true;

    MoveList moves;
    int count = 0;
    generate_legal(p, moves, count);
    if (count_legal(p) != count) {
        std::cerr << "  count_legal disagrees with generate_legal in " << p.fen() << '\n';
        return false;
    }
    if (!partition_ok(p, moves, count)) {
        std::cerr << "  captures + quiets != legal moves in " << p.fen() << '\n';
        return false;
    }

    // is_legal accepts every generated move, and (away from the leaves,
    // where it's affordable) rejects every other one of the 65536 values a
    // hash move could take.
    for (int i = 0; i < count; ++i)
        if (!p.is_legal(moves[static_cast<std::size_t>(i)])) {
            std::cerr << "  is_legal rejects " << p.move_to_uci(moves[static_cast<std::size_t>(i)])
                      << " in " << p.fen() << '\n';
            return false;
        }
    if (depth >= 2) {
        std::vector<bool> legal(65536, false);
        for (int i = 0; i < count; ++i) legal[moves[static_cast<std::size_t>(i)].data] = true;
        for (std::uint32_t v = 0; v < 65536; ++v) {
            Move m;
            m.data = static_cast<std::uint16_t>(v);
            if (p.is_legal(m) != legal[v]) {
                std::cerr << "  is_legal wrong for move value " << v << " ("
                          << p.move_to_uci(m) << (m.is_en_passant() ? " ep" : "") << ") in "
                          << p.fen() << '\n';
                return false;
            }
        }
    }

    // Both check tests, the full one and the fast one with precomputed
    // CheckInfo, must match the truth: the opponent in check after the move.
    const Position::CheckInfo ci = p.check_info();
    for (int i = 0; i < count; ++i) {
        const std::uint64_t key_before = p.key();
        const std::uint64_t pawn_key_before = p.pawn_key();
        const bool predicted_check = p.gives_check(moves[i]);
        const bool predicted_check_fast = p.gives_check(moves[i], ci);
        StateInfo st;
        p.make_move(moves[i], st);
        const bool checks = p.in_check(p.side_to_move());
        if (predicted_check != checks || predicted_check_fast != checks) {
            p.unmake_move(moves[i], st);
            std::cerr << "  gives_check wrong for " << p.move_to_uci(moves[i]) << " in "
                      << p.fen() << '\n';
            return false;
        }
        ev.push(st.changes);
        const bool ok = walk_consistent(p, ev, depth - 1);
        ev.pop();
        p.unmake_move(moves[i], st);
        if (!ok) return false;
        if (p.key() != key_before || p.pawn_key() != pawn_key_before || !p.is_consistent()) {
            std::cerr << "  state not restored after " << p.move_to_uci(moves[i])
                      << " in " << p.fen() << '\n';
            return false;
        }
    }
    return true;
}
}

int main() {
    Position p;
    PerftTable tt(16);

    const Case cases[] = {
        {"startpos", STARTPOS, 1, 20},
        {"startpos", STARTPOS, 2, 400},
        {"startpos", STARTPOS, 3, 8902},
        {"startpos", STARTPOS, 4, 197281},
        {"startpos", STARTPOS, 5, 4865609},
        {"kiwipete", KIWIPETE, 1, 48},
        {"kiwipete", KIWIPETE, 2, 2039},
        {"kiwipete", KIWIPETE, 3, 97862},
        {"pos3",     POS3,     4, 43238},
        {"pos3",     POS3,     5, 674624},
        {"pos4",     POS4,     3, 9467},
        {"pos4",     POS4,     4, 422333},
        {"pos5",     POS5,     3, 62379},

        // Legal-generator edge cases (verified against python-chess and the
        // previous make/unmake-based generator).
        {"illegal ep: rank pin",      "3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1", 6, 1134888},
        {"illegal ep: diagonal pin",  "8/8/4k3/8/2p5/8/B2P2K1/8 w - - 0 1", 6, 1015133},
        {"ep gives discovered check", "8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1", 6, 1440467},
        {"short castle gives check",  "5k2/8/8/8/8/8/8/4K2R w K - 0 1", 6, 661072},
        {"long castle gives check",   "3k4/8/8/8/8/8/8/R3K3 w Q - 0 1", 6, 803711},
        {"castling rights",           "r3k2r/1b4bq/8/8/8/8/7B/R3K2R w KQkq - 0 1", 4, 1274206},
        {"castling prevented",        "r3k2r/8/3Q4/8/8/5q2/8/R3K2R b KQkq - 0 1", 4, 1720476},
        {"promote out of check",      "2K2r2/4P3/8/8/8/8/8/3k4 w - - 0 1", 6, 3821001},
        {"discovered check",          "8/8/1P2K3/8/2n5/1q6/8/5k2 b - - 0 1", 5, 1004658},
        {"promote to give check",     "4k3/1P6/8/8/8/8/K7/8 w - - 0 1", 6, 217342},
        {"underpromote to check",     "8/P1k5/K7/8/8/8/8/8 w - - 0 1", 6, 92683},
        {"self stalemate",            "K1k5/8/P7/8/8/8/8/8 w - - 0 1", 6, 2217},
        {"stalemate & checkmate",     "8/k1P5/8/1K6/8/8/8/8 w - - 0 1", 7, 567584},
        {"stalemate & checkmate 2",   "8/8/2k5/5q2/5n2/8/5K2/8 b - - 0 1", 4, 23527},
    };

    for (const auto& tc : cases) {
        if (!p.set_fen(tc.fen)) {
            std::cerr << tc.name << ": set_fen rejected \"" << tc.fen << "\"\n";
            return 1;
        }
        auto got = perft(p, tc.depth);
        if (got != tc.expected) {
            std::cerr << tc.name << " perft " << tc.depth << ": got "
                      << got << ", expected " << tc.expected
                      << "\n  FEN: " << p.fen() << '\n';
            return 1;
        }
        // Same count through the hash table. The table is shared across
        // cases on purpose, so stale entries from earlier positions are
        // also exercised.
        got = perft(p, tc.depth, tt);
        if (got != tc.expected) {
            std::cerr << tc.name << " hashed perft " << tc.depth << ": got "
                      << got << ", expected " << tc.expected
                      << "\n  FEN: " << p.fen() << '\n';
            return 1;
        }
    }

    const Case consistency[] = {
        {"startpos", STARTPOS, 3, 0},
        {"kiwipete", KIWIPETE, 3, 0},
        {"pos3",     POS3,     4, 0},
        {"pos4",     POS4,     3, 0},
        {"pos5",     POS5,     3, 0},
        // Checks given by castling and by en passant discovery.
        {"short castle gives check", "5k2/8/8/8/8/8/8/4K2R w K - 0 1", 4, 0},
        {"long castle gives check",  "3k4/8/8/8/8/8/8/R3K3 w Q - 0 1", 4, 0},
        {"ep gives discovered check", "8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1", 4, 0},
    };

    for (const auto& tc : consistency) {
        p.set_fen(tc.fen);
        Evaluator ev;
        ev.reset(p);
        if (!walk_consistent(p, ev, tc.depth)) {
            std::cerr << tc.name << ": incremental state diverged from recomputed state\n";
            return 1;
        }
    }

    std::cout << "All perft tests passed.\n";
    return 0;
}
