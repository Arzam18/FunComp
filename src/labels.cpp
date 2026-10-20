#include "labels.h"
#include "git_hash.h"
#include "guide_features.h"
#include "movegen.h"
#include "params.h"
#include "search.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <ostream>
#include <sstream>
#include <vector>

namespace {

std::uint64_t splitmix64(std::uint64_t& x) {
    x += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

const char* node_name(prune::NodeType n) {
    switch (n) {
        case prune::NodeType::PV: return "pv";
        case prune::NodeType::Cut: return "cut";
        default: return "all";
    }
}

const char* bound_name(int b) {
    switch (b) {
        case TranspositionTable::UPPER: return "upper";
        case TranspositionTable::LOWER: return "lower";
        case TranspositionTable::EXACT: return "exact";
        default: return "none";
    }
}

// 1-in-N sampling rate for each depth; see labels.h.
std::vector<std::uint64_t> sampling_rates(const LabelOptions& opt) {
    std::vector<std::uint64_t> rates(static_cast<std::size_t>(std::max(opt.max_label_depth, 0) + 1), 0);
    const double scale = std::max(opt.depth_scale, 1) / 100.0;
    for (int d = 1; d <= opt.max_label_depth; ++d) {
        const double r = std::max(opt.sample_rate, 1) / std::pow(scale, d - 1);
        rates[static_cast<std::size_t>(d)] = static_cast<std::uint64_t>(std::max(1.0, std::round(r)));
    }
    return rates;
}

// Reads "FEN" or EPD ("4 FEN fields, then operations") from a line.
bool parse_fen_line(const std::string& line, std::string& fen) {
    std::istringstream ss(line);
    std::string f[6];
    for (int i = 0; i < 4; ++i)
        if (!(ss >> f[i])) return false;
    fen = f[0] + ' ' + f[1] + ' ' + f[2] + ' ' + f[3];
    const auto numeric = [](const std::string& s) {
        return !s.empty() && s.find_first_not_of("0123456789") == std::string::npos;
    };
    if (ss >> f[4] >> f[5] && numeric(f[4]) && numeric(f[5])) fen += ' ' + f[4] + ' ' + f[5];
    else fen += " 0 1";
    return true;
}

bool threefold(const Position& pos, const std::vector<std::uint64_t>& history) {
    return std::count(history.begin(), history.end(), pos.key()) >= 2;
}

// The parameters the labels were made with, for telling rounds of data
// apart once the margins have been retuned.
void write_params(const LabelOptions& opt) {
    std::ofstream out(opt.out + ".params");
    out << "# genlabels options\n"
        << "git=" << GIT_HASH << '\n'
        << "in=" << opt.in << "\ngames=" << opt.games << "\nrandomplies=" << opt.random_plies
        << "\ndepth=" << opt.depth << "\nsample=" << opt.sample_rate
        << "\ndepthscale=" << opt.depth_scale << "\nmaxdepth=" << opt.max_label_depth
        << "\nseed=" << opt.seed << "\nhash=" << opt.hash_mb << "\nlabelhash=" << opt.label_hash_mb
        << "\nlabelnodes=" << opt.label_node_cap << "\nnullprobe=" << opt.null_probe
        << "\n# search parameters\n";
    for (const auto& t : params::tunables()) out << t.name << '=' << *t.value << '\n';
}

}  // namespace

std::string label_header(bool null_probe) {
    std::string h =
        "fen,depth,ply,static_eval,correction,search_eval,alpha,beta,improving,node,under_null,iir,"
        "score,mate";
    for (const char* name : features::NAMES) (h += ',') += name;
    h += ",group,rate,tt_hit,tt_depth,tt_bound,tree_nodes,reached_moves,fresh_nodes,censored,"
         "fresh_curve,score_curve";
    if (null_probe) h += ",null_curve";
    return h;
}

std::string label_row(Searcher& labeller, const LabelSample& s, std::uint64_t group,
                      std::uint64_t rate, std::uint64_t node_cap, bool null_probe) {
    Position pos;
    if (!pos.set_fen(s.fen) || pos.in_check(pos.side_to_move())) return {};

    // A fresh table for every label: otherwise entries from earlier labels,
    // some searched deeper, leak into this one and make it depend on the
    // order the samples were processed in.
    labeller.new_game();
    SearchLimits limits;
    limits.depth = s.depth;
    limits.nodes = node_cap;
    labeller.search(pos, limits, {}, false);
    const auto& iters = labeller.iterations();
    // Stopped by the cap (a mate or stalemate at the root has no iterations
    // and isn't).
    const bool censored = node_cap && labeller.nodes() >= node_cap;
    const int raw = labeller.last_score();
    const bool mate = std::abs(raw) > Searcher::MATE_BOUND;
    const int score = std::clamp(raw, -LABEL_SCORE_CLIP, LABEL_SCORE_CLIP);

    std::ostringstream row;
    row << s.fen << ',' << s.depth << ',' << s.ply << ',' << s.static_eval << ',' << s.correction
        << ',' << s.search_eval << ',' << s.alpha << ',' << s.beta << ',' << s.improving << ','
        << node_name(s.node) << ',' << s.under_null << ',' << s.iir << ',' << score << ','
        << mate;
    for (const float f : features::extract(pos)) row << ',' << f;

    row << ',' << group << ',' << rate << ',' << s.tt_hit << ',' << s.tt_depth << ','
        << bound_name(s.tt_bound) << ',' << s.tree_nodes << ',' << s.reached_moves << ','
        << labeller.nodes() << ',' << censored << ',';
    for (std::size_t i = 0; i < iters.size(); ++i) row << (i ? ";" : "") << iters[i].nodes;
    row << ',';
    for (std::size_t i = 0; i < iters.size(); ++i)
        row << (i ? ";" : "") << std::clamp(iters[i].score, -LABEL_SCORE_CLIP, LABEL_SCORE_CLIP);
    if (null_probe) {
        // Try 6: the null position's scores at every depth, from a fresh
        // table (its side to move is the other one: T = score + null).
        StateInfo st;
        Position null = pos;
        null.make_null_move(st);
        labeller.new_game();
        labeller.search(null, limits, {}, false);
        row << ',';
        const auto& null_iters = labeller.iterations();
        for (std::size_t i = 0; i < null_iters.size(); ++i)
            row << (i ? ";" : "") << std::clamp(null_iters[i].score, -LABEL_SCORE_CLIP, LABEL_SCORE_CLIP);
    }
    return row.str();
}

std::uint64_t generate_labels(const LabelOptions& opt, std::ostream& log) {
    std::ofstream out(opt.out);
    if (!out) {
        log << "info string genlabels: cannot open " << opt.out << std::endl;
        return 0;
    }
    out << label_header(opt.null_probe) << '\n';
    write_params(opt);

    Searcher driver(static_cast<std::size_t>(opt.hash_mb));
    Searcher labeller(static_cast<std::size_t>(opt.label_hash_mb));
    const std::vector<std::uint64_t> rates = sampling_rates(opt);
    std::vector<std::uint64_t> per_depth(rates.size(), 0);
    std::vector<LabelSample> samples;
    std::uint64_t rng = opt.seed;
    std::uint64_t written = 0;
    std::uint64_t group = 0;  // games or input lines so far; see labels.h
    const auto done = [&] { return opt.count && written >= opt.count; };

    SearchLimits limits;
    limits.depth = opt.depth;

    // One driver search with sampling on, then its samples labelled.
    auto drive = [&](Position& pos, const std::vector<std::uint64_t>& history) {
        samples.clear();
        driver.set_label_sampling(&samples, rates, splitmix64(rng));
        const Move best = driver.search(pos, limits, history, false);
        driver.set_label_sampling(nullptr, {}, 0);
        for (const LabelSample& s : samples) {
            if (done()) break;
            const std::string row = label_row(labeller, s, group,
                                              rates[static_cast<std::size_t>(s.depth)],
                                              opt.label_node_cap, opt.null_probe);
            if (row.empty()) continue;
            out << row << '\n';
            ++written;
            ++per_depth[static_cast<std::size_t>(s.depth)];
        }
        return best;
    };

    if (!opt.in.empty()) {
        std::ifstream in(opt.in);
        if (!in) {
            log << "info string genlabels: cannot open " << opt.in << std::endl;
            return 0;
        }
        std::string line, fen;
        std::uint64_t n = 0;
        while (!done() && std::getline(in, line)) {
            Position pos;
            if (!parse_fen_line(line, fen) || !pos.set_fen(fen)) continue;
            drive(pos, {});
            ++group;
            if (++n % 100 == 0)
                log << "info string genlabels: " << n << " positions, " << written << " rows"
                    << std::endl;
        }
    } else {
        for (int g = 0; g < opt.games && !done(); ++g) {
            Position pos;
            std::vector<std::uint64_t> history;
            // Room for the random plies (unclamped) and a game after them.
            std::vector<StateInfo> states(static_cast<std::size_t>(std::max(opt.random_plies, 0)) + 1024);
            std::size_t made = 0;
            auto play = [&](Move m) {
                history.push_back(pos.key());
                pos.make_move(m, states[made++]);
            };

            // Random opening, so that games differ.
            bool alive = true;
            for (int i = 0; i < opt.random_plies && alive; ++i) {
                MoveList moves;
                int count = 0;
                generate_legal(pos, moves, count);
                if (count == 0) alive = false;
                else play(moves[static_cast<std::size_t>(splitmix64(rng) % static_cast<std::uint64_t>(count))]);
            }
            if (!alive) continue;

            driver.new_game();
            while (!done() && made < states.size()) {
                const Move best = drive(pos, history);
                if (best == Move{}) break;                           // mate or stalemate
                if (std::abs(driver.last_score()) >= 2000) break;    // decided
                play(best);
                if (pos.halfmove_clock() >= 100 || threefold(pos, history)) break;
            }
            ++group;
            log << "info string genlabels: game " << g + 1 << '/' << opt.games << ", " << written
                << " rows" << std::endl;
        }
    }

    log << "info string genlabels: rows by depth:";
    for (std::size_t d = 1; d < per_depth.size(); ++d) log << ' ' << d << ':' << per_depth[d];
    log << "\ninfo string genlabels: wrote " << written << " rows to " << opt.out << std::endl;
    return written;
}
