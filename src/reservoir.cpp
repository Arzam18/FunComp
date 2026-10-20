#include "reservoir.h"
#include "movegen.h"
#include "nnue.h"
#include "pgn.h"
#include "position.h"
#include "search.h"
#include <array>
#include <bit>
#include <cstdlib>
#include <fstream>
#include <ostream>
#include <random>
#include <sstream>
#include <unordered_set>
#include <vector>

namespace {

enum Pool { Early, Late, End, POOLS };
constexpr std::array<const char*, POOLS> POOL_NAME = {"early", "late", "end"};

// Why a game was left out (the tag filters, then replay).
enum Drop { Elo, Speed, Bot, Ending, Variant, Illegal, DROPS };
constexpr std::array<const char*, DROPS> DROP_NAME = {
    "rating", "bullet or correspondence", "bot", "rules infraction", "not standard", "illegal move"};

int rating(const pgn::Game& g, const char* tag) {
    const std::string& v = g.tag(tag);
    if (v.empty() || v[0] < '0' || v[0] > '9') return 0;
    return std::atoi(v.c_str());
}

// Lichess's estimated duration, base + 40 x increment (seconds); -1 for none
// ("-": correspondence) or an unreadable tag.
int estimated_seconds(const pgn::Game& g) {
    const std::string& tc = g.tag("TimeControl");
    const auto plus = tc.find('+');
    if (tc.empty() || plus == std::string::npos || tc[0] < '0' || tc[0] > '9') return -1;
    return std::atoi(tc.c_str()) + 40 * std::atoi(tc.c_str() + plus + 1);
}

// The tag filters: -1 kept, else the reason.
int tag_drop(const pgn::Game& g, const ReservoirOptions& opt) {
    if (g.tag("WhiteTitle") == "BOT" || g.tag("BlackTitle") == "BOT") return Bot;
    if (rating(g, "WhiteElo") < opt.min_elo || rating(g, "BlackElo") < opt.min_elo) return Elo;
    if (estimated_seconds(g) < opt.min_seconds) return Speed;
    const std::string& term = g.tag("Termination");
    if (term == "Rules infraction") return Ending;  // often an engine user caught: not a human game
    const std::string& variant = g.tag("Variant");
    if ((!variant.empty() && variant != "Standard") || !g.tag("FEN").empty()) return Variant;
    return -1;
}

// Game phase as the evaluation counts it: knights and bishops 1, rooks 2,
// queens 4, both sides, at most 24.
int phase_of(const Position& pos) {
    const int minors = std::popcount(pos.pieces(WN) | pos.pieces(BN) | pos.pieces(WB) | pos.pieces(BB));
    const int rooks = std::popcount(pos.pieces(WR) | pos.pieces(BR));
    const int queens = std::popcount(pos.pieces(WQ) | pos.pieces(BQ));
    return std::min(minors + 2 * rooks + 4 * queens, 24);
}

// The FEN with the fifty-move clock reset (the start's own clock).
std::string start_fen(const Position& pos) {
    std::istringstream ss(pos.fen());
    std::array<std::string, 6> f;
    for (std::string& x : f) ss >> x;
    return f[0] + ' ' + f[1] + ' ' + f[2] + ' ' + f[3] + " 0 " + f[5];
}

// A start's tags for the planner (tools/nnue/plan_starts.py), the
// properties that persist through the game it begins: the pool, each king's
// state (0-14, nnue.h) and wing (q: files a-d, k: e-h), the men on the board,
// the material bucket (of 8, as the networks' outputs), the endgame type (0
// no pawns, 1 pawns only, 2 opposite bishops, -1 none) and the phase; then
// the game's ratings and estimated duration and, for an endgame start, its
// search score (side to move's view; "-" if not searched), so a later plan
// can choose another rating floor or endgame band without the raw file (the
// author, 2026-10-07: never download a month twice).
std::string tags(const char* pool, const std::string& fen, int white_elo, int black_elo, int seconds,
                 const std::string& score) {
    Position pos;
    pos.set_fen(fen);
    const int men = std::popcount(pos.occupied());
    auto wing = [&](Piece king) { return (std::countr_zero(pos.pieces(king)) & 7) < 4 ? 'q' : 'k'; };
    std::ostringstream t;
    t << pool << ' ' << nnue::king_state_of(pos, Color::White) << ' ' << nnue::king_state_of(pos, Color::Black) << ' '
      << wing(WK) << ' ' << wing(BK) << ' ' << men << ' ' << std::min((men - 2) / 4, 7) << ' '
      << nnue::endgame_type_of(pos) << ' ' << phase_of(pos) << ' ' << white_elo << ' ' << black_elo << ' '
      << seconds << ' ' << score << " | " << fen;
    return t.str();
}

}  // namespace

std::uint64_t build_reservoir(const ReservoirOptions& opt, std::ostream& log) {
    std::ifstream in(opt.in, std::ios::binary);
    if (!in) {
        log << "info string reservoir: cannot read " << opt.in << std::endl;
        return 0;
    }
    std::array<std::ofstream, POOLS> out;
    std::ofstream tagged;  // every written start with its tags, for the planner
    if (!opt.count_only) {
        for (int p = 0; p < POOLS; ++p) {
            out[static_cast<std::size_t>(p)].open(opt.out + "-" + POOL_NAME[static_cast<std::size_t>(p)] + ".fen");
            if (!out[static_cast<std::size_t>(p)]) {
                log << "info string reservoir: cannot write " << opt.out << "-*.fen" << std::endl;
                return 0;
            }
        }
        tagged.open(opt.out + ".tags");
        if (!tagged) {
            log << "info string reservoir: cannot write " << opt.out << ".tags" << std::endl;
            return 0;
        }
    }
    std::mt19937_64 rng(opt.seed);
    Searcher searcher(static_cast<std::size_t>(opt.hash_mb));
    std::unordered_set<std::uint64_t> seen;
    std::array<std::uint64_t, DROPS> dropped{};
    std::array<std::uint64_t, POOLS> candidates{}, duplicates{}, written{};
    std::uint64_t games = 0, kept = 0, end_searched = 0, end_out_of_band = 0;

    auto report = [&](const char* when) {
        log << "info string reservoir " << when << ": games " << games << ", kept " << kept;
        for (int d = 0; d < DROPS; ++d) log << ", " << DROP_NAME[static_cast<std::size_t>(d)] << ' ' << dropped[static_cast<std::size_t>(d)];
        if (!opt.count_only) {
            for (int p = 0; p < POOLS; ++p) {
                const auto i = static_cast<std::size_t>(p);
                log << "; " << POOL_NAME[i] << ' ' << written[i] << " written (" << candidates[i]
                    << " candidates, " << duplicates[i] << " duplicates)";
            }
            if (opt.end_nodes) log << "; end searched " << end_searched << ", outside the band " << end_out_of_band;
        }
        log << std::endl;
    };

    pgn::Reader reader(in, false);  // the moves split only for games the tags keep
    pgn::Game g;
    while ((!opt.max_games || games < opt.max_games) && reader.next(g)) {
        ++games;
        if (games % 1000000 == 0) report("so far");
        const int d = tag_drop(g, opt);
        if (d >= 0) {
            ++dropped[static_cast<std::size_t>(d)];
            continue;
        }
        if (opt.count_only) {
            ++kept;
            continue;
        }
        pgn::split_moves(g);
        // Replay, gathering each pool's positions in the move window.
        Position pos;
        pos.set_fen(g.fen);
        std::vector<StateInfo> states(g.moves.size() + 1);
        std::array<std::vector<std::string>, POOLS> found;
        std::array<std::vector<std::uint64_t>, POOLS> keys;
        bool legal = true;
        for (std::size_t i = 0; i < g.moves.size(); ++i) {
            const Move m = pgn::parse_san(pos, g.moves[i]);
            if (m == Move{}) {
                legal = false;
                break;
            }
            pos.make_move(m, states[i]);
            const int move_no = static_cast<int>(i / 2) + 1;  // of the move just played
            if (move_no < opt.min_move) continue;
            if (move_no > opt.max_move) break;
            if (pos.in_check(pos.side_to_move())) continue;
            const int phase = phase_of(pos);
            const int men = std::popcount(pos.occupied());
            const int pool = phase >= 17 ? Early : phase >= 9 ? Late : men >= 7 ? End : -1;
            if (pool < 0) continue;
            found[static_cast<std::size_t>(pool)].push_back(start_fen(pos));
            keys[static_cast<std::size_t>(pool)].push_back(pos.key());
        }
        if (!legal) {
            ++dropped[Illegal];
            continue;
        }
        ++kept;
        // One position per pool, at random.
        for (int p = 0; p < POOLS; ++p) {
            const auto i = static_cast<std::size_t>(p);
            if (found[i].empty()) continue;
            ++candidates[i];
            const std::size_t k = static_cast<std::size_t>(rng() % found[i].size());
            if (!seen.insert(keys[i][k]).second) {
                ++duplicates[i];
                continue;
            }
            std::string score = "-";
            bool in_band = true;
            if (p == End && opt.end_nodes) {
                ++end_searched;
                Position e;
                e.set_fen(found[i][k]);
                searcher.new_game();
                SearchLimits limits;
                limits.nodes = opt.end_nodes;
                searcher.search(e, limits, {}, false);
                const int s = searcher.last_score();
                score = std::to_string(s);
                in_band = std::abs(s) >= opt.end_min && std::abs(s) <= opt.end_max;
                if (!in_band) ++end_out_of_band;
            }
            // Every start goes into the tags (an endgame outside the band too,
            // with its score: the band can change at planning time); the pool
            // files keep the band.
            tagged << tags(POOL_NAME[i], found[i][k], rating(g, "WhiteElo"), rating(g, "BlackElo"),
                           estimated_seconds(g), score)
                   << '\n';
            if (!in_band) continue;
            out[i] << found[i][k] << '\n';
            ++written[i];
        }
    }
    report("done");
    return opt.count_only ? kept : written[Early] + written[Late] + written[End];
}
