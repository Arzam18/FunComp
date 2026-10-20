#include "cont_log.h"
#include "git_hash.h"
#include "movegen.h"
#include "params.h"
#include "position.h"
#include "search.h"
#include <algorithm>
#include <cmath>
#include <ostream>

namespace {

void header_nodes(std::ostream& o) {
    o << "node,game,game_ply,fen,depth,ply,root_depth,seldepth,parent_reduction,"
         "parent_provisional,extensions,node_type,alpha,beta,raw_eval,static_eval,eval,"
         "correction,improving,in_check,threat,tt_hit,tt_depth,tt_bound,tt_score,tt_move,phase,"
         "material,halfmove,parent_index,parent_quiet,parent_history,legal,legal_captures,"
         "legal_quiets,final_best,bound,nodes_total,rows\n";
}

void header_rows(std::ostream& o) {
    o << "node,row,index,searched_before,stage,pruned_lmp,pruned_history,pruned_futility,"
         "pruned_see_quiet,pruned_see_capture,pruned_underpromo,lmp_skipping,piece,from,to,"
         "promotion,capture,gives_check,killer,hash_move,see,history,capture_history,"
         "lmr_reduction,alpha,best_before,second_before,near_best,raised_before,"
         "researches_before,nodes_before,score,raised,gain,later_raise_index,later_fail_high,"
         "rest_nodes,final_best,bound\n";
}

void header_root(std::ostream& o) {
    o << "game,game_ply,depth,best,score,score_change,stability,nodes,iteration_nodes,"
         "elapsed_ms,soft_ms,hard_ms,node_share\n";
}

bool threefold(const Position& pos, const std::vector<std::uint64_t>& history) {
    return std::count(history.begin(), history.end(), pos.key()) >= 2;
}

}  // namespace

ContLog::ContLog(const std::string& prefix, int min_depth, int base_rate, int depth_scale,
                 std::uint64_t seed)
    : nodes_(prefix + ".nodes.csv"),
      rows_(prefix + ".rows.csv"),
      root_(prefix + ".root.csv"),
      min_depth_(min_depth),
      rates_(64, 0),
      rng_(seed) {
    header_nodes(nodes_);
    header_rows(rows_);
    header_root(root_);
    for (int d = min_depth; d < 64; ++d) {
        const double r = base_rate / std::pow(depth_scale / 100.0, d - min_depth);
        rates_[static_cast<std::size_t>(d)] = static_cast<std::uint64_t>(std::max(1.0, std::round(r)));
    }
}

bool ContLog::sample(int depth) {
    if (depth < min_depth_) return false;
    const std::uint64_t rate = rates_[static_cast<std::size_t>(std::min(depth, 63))];
    return rate && rng_() % rate == 0;
}

void ContLog::write_node(const ContNode& n, int final_best, int bound, std::uint64_t nodes_total) {
    const std::uint64_t id = next_node_++;
    nodes_ << id << ',' << game_ << ',' << game_ply_ << ',' << n.fen << ',' << n.depth << ','
           << n.ply << ',' << n.root_depth << ',' << n.seldepth << ',' << n.parent_reduction << ','
           << n.parent_provisional << ',' << n.extensions << ',' << n.node_type << ',' << n.alpha
           << ',' << n.beta << ',' << n.raw_eval << ',' << n.static_eval << ',' << n.eval << ','
           << n.correction << ',' << n.improving << ',' << n.in_check << ',' << n.threat << ','
           << n.tt_hit << ',' << n.tt_depth << ',' << n.tt_bound << ',' << n.tt_score << ','
           << n.tt_move << ',' << n.phase << ',' << n.material << ',' << n.halfmove << ','
           << n.parent_index << ',' << n.parent_quiet << ',' << n.parent_history << ',' << n.legal
           << ',' << n.legal_captures << ',' << n.legal_quiets << ',' << final_best << ',' << bound
           << ',' << nodes_total << ',' << n.rows.size() << '\n';
    // Outcome labels, from the end backwards: the first later row that
    // raised alpha; a fail high is the last row searched (the loop breaks).
    const int count = static_cast<int>(n.rows.size());
    std::vector<int> later_raise(n.rows.size(), -1);
    for (int k = count - 1, next = -1; k >= 0; --k) {
        if (n.rows[static_cast<std::size_t>(k)].raised) next = n.rows[static_cast<std::size_t>(k)].index;
        later_raise[static_cast<std::size_t>(k)] = next;
    }
    auto score_or_empty = [](int s) { return s == NO_SCORE ? std::string() : std::to_string(s); };
    for (int k = 0; k < count; ++k) {
        const ContRow& r = n.rows[static_cast<std::size_t>(k)];
        rows_ << id << ',' << k << ',' << r.index << ',' << r.searched_before << ',' << r.stage;
        for (int p : r.pruned) rows_ << ',' << p;
        rows_ << ',' << r.lmp_skipping << ',' << r.piece << ',' << r.from << ',' << r.to << ','
              << r.promotion << ',' << r.capture << ',' << r.gives_check << ',' << r.killer << ','
              << r.hash_move << ',' << r.see << ',' << r.history << ',' << r.capture_history << ','
              << r.lmr_reduction << ',' << r.alpha << ',' << score_or_empty(r.best_before) << ','
              << score_or_empty(r.second_before) << ',' << r.near_best << ',' << r.raised_before
              << ',' << r.researches_before << ',' << r.nodes_before << ',' << r.score << ','
              << r.raised << ','
              << (r.best_before == NO_SCORE ? std::string() : std::to_string(final_best - r.best_before))
              << ',' << later_raise[static_cast<std::size_t>(k)] << ','
              << (bound == TranspositionTable::LOWER ? 1 : 0) << ',' << nodes_total - r.nodes_before << ','
              << final_best << ',' << bound << '\n';
        ++rows_written_;
    }
}

void ContLog::write_root(const ContRoot& r) {
    root_ << game_ << ',' << game_ply_ << ',' << r.depth << ',' << r.best << ',' << r.score << ','
          << r.score_change << ',' << r.stability << ',' << r.nodes << ',' << r.iteration_nodes
          << ',' << r.elapsed_ms << ',' << r.soft_ms << ',' << r.hard_ms << ',' << r.node_share
          << '\n';
}

std::uint64_t generate_cont_log(const ContLogOptions& opt, std::ostream& log) {
    if (!CONT_LOG_BUILD) {
        log << "info string genconlog needs a build configured with -DCONT_LOG=ON" << std::endl;
        return 0;
    }
    ContLog cl(opt.out, opt.min_depth, opt.rate, opt.depth_scale, opt.seed ^ 0x9e3779b97f4a7c15ULL);
    if (!cl.ok()) {
        log << "info string genconlog: cannot write " << opt.out << ".*.csv" << std::endl;
        return 0;
    }
    {
        std::ofstream p(opt.out + ".params");
        p << "# genconlog options\ngit=" << GIT_HASH << "\ngames=" << opt.games
          << "\nrandomplies=" << opt.random_plies << "\nnodes=" << opt.nodes
          << "\nmindepth=" << opt.min_depth << "\nrate=" << opt.rate
          << "\ndepthscale=" << opt.depth_scale << "\nseed=" << opt.seed << "\nhash=" << opt.hash_mb
          << "\nthreads=1\n# sampling, 1 in N nodes by depth\n";
        for (int d = opt.min_depth; d < 64; ++d)
            p << "rate" << d << '=' << cl.rates()[static_cast<std::size_t>(d)] << '\n';
        p << "# search parameters\n";
        for (const auto& t : params::tunables()) p << t.name << '=' << *t.value << '\n';
    }
    std::mt19937_64 rng(opt.seed);
    Searcher searcher(static_cast<std::size_t>(opt.hash_mb));
    searcher.set_cont_log(&cl);
    for (int g = 0; g < opt.games; ++g) {
        Position pos;
        std::vector<std::uint64_t> history;
        // Room for the random plies (unclamped) and a game after them.
        std::vector<StateInfo> states(static_cast<std::size_t>(std::max(opt.random_plies, 0)) + 1024);
        std::size_t made = 0;
        auto play = [&](Move m) {
            history.push_back(pos.key());
            pos.make_move(m, states[made++]);
        };
        bool alive = true;
        for (int i = 0; i < opt.random_plies && alive; ++i) {
            MoveList moves;
            int count = 0;
            generate_legal(pos, moves, count);
            if (count == 0) alive = false;
            else play(moves[static_cast<std::size_t>(rng() % static_cast<std::uint64_t>(count))]);
        }
        if (!alive) continue;
        searcher.new_game();
        while (made < states.size()) {
            cl.set_position(g, static_cast<int>(made));
            SearchLimits limits;
            limits.nodes = opt.nodes;
            const Move best = searcher.search(pos, limits, history, false);
            if (best == Move{}) break;                             // mate or stalemate
            if (std::abs(searcher.last_score()) >= 2000) break;    // decided
            play(best);
            if (pos.halfmove_clock() >= 100 || threefold(pos, history)) break;
        }
        log << "info string genconlog: game " << g + 1 << '/' << opt.games << ", "
            << cl.nodes_written() << " nodes, " << cl.rows_written() << " rows" << std::endl;
    }
    searcher.set_cont_log(nullptr);
    log << "info string genconlog: wrote " << cl.rows_written() << " rows from "
        << cl.nodes_written() << " nodes to " << opt.out << ".*.csv" << std::endl;
    return cl.rows_written();
}
