#include "tree_sample.h"
#include "movegen.h"
#include "pgn.h"
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <vector>

void TreeSampler::write(const Position& pos, char kind, int depth, int ply, int eval) {
    out_ << kind << ';' << depth << ';' << ply << ';' << eval << ';' << pos.fen() << '\n';
    ++written_;
}

namespace {
// A position field: a FEN, or "startpos moves <uci...>" (tools/research/collapse.py's positions).
bool read_position(const std::string& field, Position& pos, std::vector<std::uint64_t>& history) {
    history.clear();
    if (field.rfind("startpos", 0) != 0) return pos.set_fen(field);
    pos.set_startpos();
    std::istringstream ss(field);
    std::string token;
    ss >> token;  // startpos
    if (!(ss >> token)) return true;
    if (token != "moves") return false;
    while (ss >> token) {
        MoveList moves;
        int count = 0;
        generate_legal(pos, moves, count);
        bool found = false;
        for (int i = 0; i < count && !found; ++i) {
            if (pos.move_to_uci(moves[i]) == token) {
                history.push_back(pos.key());
                StateInfo st;
                pos.make_move(moves[i], st);
                found = true;
            }
        }
        if (!found) return false;
    }
    return true;
}

// A line's position: the text before a '|' (the trainer's text, datatext's), or the last ';' field (treesample's
// and evalfens' lines), or the whole line.
std::string position_field(const std::string& line) {
    std::string f;
    if (const auto bar = line.find('|'); bar != std::string::npos) f = line.substr(0, bar);
    else if (const auto cut = line.rfind(';'); cut != std::string::npos) f = line.substr(cut + 1);
    else f = line;
    while (!f.empty() && (f.back() == ' ' || f.back() == '\r')) f.pop_back();
    return f;
}

// One root searched with the sampler on: its own line first.
std::uint64_t sample_root(Searcher& searcher, TreeSampler& sampler, Position& pos,
                          const std::vector<std::uint64_t>& history, std::uint64_t nodes) {
    searcher.new_game();
    const Searcher::RootEval root = searcher.root_eval(pos);
    sampler.write(pos, 'g', 0, 0, root.static_eval - root.correction);
    searcher.set_tree_sample(&sampler);
    SearchLimits limits;
    limits.nodes = nodes;
    searcher.search(pos, limits, history, false);
    searcher.set_tree_sample(nullptr);
    return searcher.nodes();
}
}  // namespace

void tree_sample(Searcher& searcher, const TreeSampleOptions& o, std::ostream& log) {
    if (!CONT_LOG_BUILD) {
        log << "info string treesample needs a build configured with -DCONT_LOG=ON" << std::endl;
        return;
    }
    std::ifstream file(o.fens.empty() ? o.in : o.fens);
    std::ofstream out(o.out);
    if (!file || !out) {
        log << "info string treesample: cannot read " << (o.fens.empty() ? o.in : o.fens) << " or write " << o.out
            << std::endl;
        return;
    }
    TreeSampler sampler(out, o.rate, o.seed);
    int done = 0, unreadable = 0;
    std::uint64_t nodes = 0;
    if (!o.fens.empty()) {
        // Root positions in order (the caller samples them, e.g. from an export's text).
        std::string line;
        while (done < o.positions && std::getline(file, line)) {
            Position pos;
            std::vector<std::uint64_t> history;
            if (!read_position(position_field(line), pos, history)) {
                ++unreadable;
                continue;
            }
            if (pos.in_check(pos.side_to_move())) continue;  // no static evaluation at the root
            nodes += sample_root(searcher, sampler, pos, history, o.nodes);
            ++done;
        }
        log << "info string treesample: " << done << " root positions (" << unreadable << " unreadable), " << nodes
            << " nodes, " << sampler.written() << " lines written to " << o.out << std::endl;
        return;
    }
    const std::vector<pgn::Game> games = pgn::read(file);
    if (games.empty()) {
        log << "info string treesample: no games in " << o.in << std::endl;
        return;
    }
    std::mt19937_64 rng(o.seed);
    for (int tries = 0; done < o.positions && tries < 20 * o.positions; ++tries) {
        const pgn::Game& game = games[rng() % games.size()];
        const int plies = static_cast<int>(game.moves.size());
        if (plies <= o.min_ply) continue;
        const int ply = o.min_ply + static_cast<int>(rng() % static_cast<std::uint64_t>(plies - o.min_ply));
        Position pos;
        if (!pos.set_fen(game.fen)) {
            ++unreadable;
            continue;
        }
        // Replay to the sampled ply, keeping the keys for repetitions (as game_telemetry).
        std::vector<std::uint64_t> history;
        bool ok = true;
        for (int i = 0; i < ply && ok; ++i) {
            const Move m = pgn::parse_san(pos, game.moves[static_cast<std::size_t>(i)]);
            if (m == Move{}) {
                ok = false;
                break;
            }
            history.push_back(pos.key());
            StateInfo st;
            pos.make_move(m, st);
        }
        if (!ok) {
            ++unreadable;
            continue;
        }
        if (pos.in_check(pos.side_to_move())) continue;  // no static evaluation at the root
        nodes += sample_root(searcher, sampler, pos, history, o.nodes);
        ++done;
    }
    log << "info string treesample: " << done << " positions from " << games.size() << " games (" << unreadable
        << " unreadable), " << nodes << " nodes, " << sampler.written() << " lines written to " << o.out
        << std::endl;
}

void eval_fens(Searcher& searcher, const EvalFensOptions& o, std::ostream& log) {
    std::ifstream in(o.in);
    std::ofstream out(o.out);
    if (!in || !out) {
        log << "info string evalfens: cannot read " << o.in << " or write " << o.out << std::endl;
        return;
    }
    std::string line;
    std::uint64_t index = 0, done = 0, bad = 0;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || index++ < o.skip) continue;
        if (o.count && done + bad >= o.count) break;
        Position pos;
        std::vector<std::uint64_t> history;
        if (!read_position(position_field(line), pos, history) || pos.in_check(pos.side_to_move())) {
            ++bad;
            continue;
        }
        searcher.new_game();
        const Searcher::RootEval root = searcher.root_eval(pos);
        const int raw = root.static_eval - root.correction;
        const int qs = searcher.quiescence(pos, history);
        searcher.new_game();
        SearchLimits limits;
        limits.nodes = o.nodes;
        searcher.search(pos, limits, history, false);
        out << line << ';' << raw << ';' << qs << ';' << searcher.last_score() << '\n';
        ++done;
    }
    log << "info string evalfens: " << done << " positions measured, " << bad
        << " left out (unreadable or in check), written to " << o.out << std::endl;
}

void label_fens(Searcher& searcher, const LabelFensOptions& o, std::ostream& log) {
    std::ifstream in(o.in);
    std::ofstream out(o.out);
    if (!in || !out) {
        log << "info string labelfens: cannot read " << o.in << " or write " << o.out << std::endl;
        return;
    }
    std::string line;
    std::uint64_t index = 0, seen = 0, written = 0, unreadable = 0, in_check = 0, decisive = 0, noisy = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line == "\r" || index++ < o.skip) continue;
        if (o.count && seen >= o.count) break;
        ++seen;
        Position pos;
        std::vector<std::uint64_t> history;
        if (!read_position(position_field(line), pos, history)) {
            ++unreadable;
            continue;
        }
        if (pos.in_check(pos.side_to_move())) {
            ++in_check;
            continue;
        }
        if ((seen - 1) % 1000 == 0) searcher.new_game();
        const std::string fen = pos.fen();
        const bool white = pos.side_to_move() == Color::White;
        SearchLimits limits;
        limits.nodes = o.nodes;
        const Move bm = searcher.search(pos, limits, history, false);
        const int score = searcher.last_score();
        if (std::abs(score) >= Searcher::DECISIVE_BOUND) {
            ++decisive;
            continue;
        }
        // As datatext's quiet filter: a capture, en passant or a promotion for the best move.
        if (pos.piece_at(bm.to()) != NO_PIECE || bm.is_en_passant() || bm.promotion() != PieceType::None) {
            ++noisy;
            continue;
        }
        out << fen << " | " << (white ? score : -score) << " | 0.5\n";
        ++written;
    }
    log << "info string labelfens: " << written << " written of " << seen << " (left out: unreadable " << unreadable
        << ", in check " << in_check << ", decisive " << decisive << ", noisy best move " << noisy << ") to " << o.out
        << std::endl;
}

void perturb_fens(const PerturbOptions& o, std::ostream& log) {
    std::ifstream in(o.in);
    std::ofstream out(o.out);
    if (!in || !out) {
        log << "info string perturb: cannot read " << o.in << " or write " << o.out << std::endl;
        return;
    }
    std::mt19937_64 rng(o.seed);
    std::string line;
    std::uint64_t written = 0, ended = 0, unreadable = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line == "\r") continue;
        Position pos;
        std::vector<std::uint64_t> history;
        if (!read_position(position_field(line), pos, history)) {
            ++unreadable;
            continue;
        }
        bool ok = true;
        for (int i = 0; i < o.plies && ok; ++i) {
            MoveList moves;
            int count = 0;
            generate_legal(pos, moves, count);
            if (count == 0) {
                ok = false;
                break;
            }
            StateInfo st;
            pos.make_move(moves[static_cast<int>(rng() % static_cast<std::uint64_t>(count))], st);
        }
        if (!ok) {
            ++ended;
            continue;
        }
        out << pos.fen() << '\n';
        ++written;
    }
    log << "info string perturb: " << written << " positions written (" << ended << " games ended on the way, "
        << unreadable << " unreadable) to " << o.out << std::endl;
}
