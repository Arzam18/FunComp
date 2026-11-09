#include "datagen.h"
#include "bingo.h"
#include "eval.h"
#include "git_hash.h"
#include "movegen.h"
#include "nnue.h"
#include "pgn.h"
#include "params.h"
#include "search.h"
#include "tablebase.h"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace {

constexpr char MAGIC[4] = {'F', 'C', 'D', 'G'};
// The newest block version: 1, round 1 and 2's gendata; 2, best move after
// each record (rescore's first version); 3, gendata and rescore since
// 2026-10-07 (the previous iteration's score and the best move's stability
// too). A game is written in its own version; every version is read.
constexpr std::uint16_t VERSION = 4;
// What gendata and rescore write: 3, or 4 with the backward pass's label
// (gendata with back_soft_nodes; 2026-10-19).
constexpr std::uint16_t FORWARD_VERSION = 3, BACKWARD_VERSION = 4;

std::uint64_t splitmix64(std::uint64_t& x) {
    x += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

std::int16_t to_i16(long long v) {
    return static_cast<std::int16_t>(std::clamp(v, -32767LL, 32767LL));
}

// Little-endian encoding into a byte string, and back.
struct Writer {
    std::string b;
    void u8(unsigned v) { b.push_back(static_cast<char>(v & 0xFF)); }
    void u16(unsigned v) {
        u8(v);
        u8(v >> 8);
    }
    void u32(std::uint32_t v) {
        u16(v & 0xFFFF);
        u16(v >> 16);
    }
    void u64(std::uint64_t v) {
        u32(static_cast<std::uint32_t>(v));
        u32(static_cast<std::uint32_t>(v >> 32));
    }
    void i16(std::int16_t v) { u16(static_cast<std::uint16_t>(v)); }
};

struct Reader {
    const std::string& b;
    std::size_t p = 0;
    bool ok = true;
    unsigned u8() {
        if (p >= b.size()) {
            ok = false;
            return 0;
        }
        return static_cast<unsigned char>(b[p++]);
    }
    std::uint16_t u16() {
        const unsigned lo = u8();
        return static_cast<std::uint16_t>(lo | u8() << 8);
    }
    std::uint32_t u32() {
        const std::uint32_t lo = u16();
        return lo | static_cast<std::uint32_t>(u16()) << 16;
    }
    std::uint64_t u64() {
        const std::uint64_t lo = u32();
        return lo | static_cast<std::uint64_t>(u32()) << 32;
    }
    std::int16_t i16() { return static_cast<std::int16_t>(u16()); }
};

bool is_legal_move(const Position& pos, Move m) {
    MoveList moves;
    int count = 0;
    generate_legal(pos, moves, count);
    for (int i = 0; i < count; ++i)
        if (moves[static_cast<std::size_t>(i)] == m) return true;
    return false;
}

// Game phase as the evaluation counts it: 24 with every minor and major
// piece on the board, 0 with none.
int phase_of(const Position& pos) {
    int phase = 0;
    for (int c = 0; c < 2; ++c)
        for (int pt = 1; pt <= 4; ++pt)
            phase += std::popcount(pos.pieces(static_cast<Piece>(c * 6 + pt))) *
                     (pt == 4 ? 4 : pt == 3 ? 2 : 1);
    return std::min(phase, 24);
}

}  // namespace

std::vector<std::string> data_files(const std::string& path) {
    std::vector<std::string> files;
    if (std::filesystem::is_directory(path)) {
        for (const auto& e : std::filesystem::directory_iterator(path))
            if (e.is_regular_file() && e.path().extension() == ".bin")
                files.push_back(e.path().string());
        std::sort(files.begin(), files.end());
    } else if (std::filesystem::exists(path)) {
        files.push_back(path);
    }
    return files;
}

namespace {

// The complete blocks at the start of an existing file: where they end,
// and the index of the last game. Anything after them is a block cut short.
std::uint64_t scan_existing(const std::string& path, std::uint32_t& next_index) {
    std::ifstream in(path, std::ios::binary);
    std::uint64_t good = 0;
    DataGame g;
    while (read_game(in, g)) {
        good = static_cast<std::uint64_t>(in.tellg());
        next_index = std::max(next_index, g.index + 1);
    }
    return good;
}

void write_params(const DataOptions& opt, std::uint32_t first_index) {
    std::ofstream out(opt.out + ".params");
    out << "# gendata options\n"
        << "git=" << GIT_HASH << "\nworker=" << opt.worker << "\nseed=" << opt.seed
        << "\nhours=" << opt.hours << "\ngames=" << opt.games << "\nsoftnodes=" << opt.soft_nodes
        << "\nhardnodes=" << opt.hard_nodes << "\nsoftscale=" << opt.soft_scale
        << "\nmindepth=" << opt.min_depth << "\ndropsoftnodes=" << opt.drop_soft_nodes << "\ndrophardnodes=" << opt.drop_hard_nodes
        << "\nbacksoftnodes=" << opt.back_soft_nodes << "\nbackhardnodes=" << opt.back_hard_nodes
        << "\nbingoevery=" << opt.bingo_every
        << "\nrandomplies=" << opt.random_min << '-' << opt.random_max
        << "\nmaxopenscore=" << opt.max_open_score << "\nmaxplies=" << opt.max_plies
        << "\nhash=" << opt.hash_mb << "\nsyzygy=" << opt.syzygy
        << "\nsyzygy_largest=" << tb::largest() << "\nfirst_index=" << first_index
        << "\nadjscore=" << opt.adj_score << "\nadjplies=" << opt.adj_plies
        << "\nadjphase=" << opt.adj_phase << "\nadjtb=" << opt.adj_tb
        << "\nadjdraw=" << opt.adj_draw_score << "\nadjdrawplies=" << opt.adj_draw_plies
        << "\nadjdrawply=" << opt.adj_draw_ply << "\nadjpct=" << opt.adj_pct
        << "\ncastlelong=" << opt.castle_long_pct << "\ncastlelongby=" << opt.castle_long_by
        << "\nadjmate=" << opt.adj_mate << "\nstarts=" << opt.starts << "\nstartspct=" << opt.starts_pct
        << "\nstartmix=" << opt.starts_mix[0] << ':' << opt.starts_mix[1] << ':' << opt.starts_mix[2]
        << "\nstartplies=" << opt.starts_plies_min << '-' << opt.starts_plies_max
        << "\nstartstride=" << opt.starts_stride << "\nsharenet=" << nnue::shared()
        << "\nformat_version=" << (opt.back_soft_nodes ? BACKWARD_VERSION : FORWARD_VERSION)
        << "\n# search parameters\n";
    for (const auto& t : params::tunables()) out << t.name << '=' << *t.value << '\n';
}

}  // namespace

const char* data_end_name(DataEnd e) {
    switch (e) {
        case DataEnd::Checkmate: return "checkmate";
        case DataEnd::Stalemate: return "stalemate";
        case DataEnd::Threefold: return "threefold";
        case DataEnd::FiftyMoves: return "fifty-moves";
        case DataEnd::Material: return "material";
        case DataEnd::PlyCap: return "ply-cap";
        case DataEnd::Filtered: return "filtered";
        case DataEnd::AdjScore: return "adj-score";
        case DataEnd::AdjTablebase: return "adj-tablebase";
        case DataEnd::AdjDraw: return "adj-draw";
        case DataEnd::External: return "external";
        case DataEnd::AdjMate: return "adj-mate";
    }
    return "?";
}

void write_game(std::ostream& out, const DataGame& g) {
    Writer w;
    w.u16(g.worker);
    w.u32(g.index);
    w.u64(g.seed);
    w.u8(g.opening);
    w.u16(g.start_index);
    w.u8(static_cast<unsigned>(g.fen.size()));
    w.b += g.fen;
    w.u8(static_cast<unsigned>(g.random_moves.size()));
    for (const Move m : g.random_moves) w.u16(m.data);
    w.u8(static_cast<unsigned>(g.result));
    w.u8(static_cast<unsigned>(g.end));
    w.u16(g.plies);
    w.u32(static_cast<std::uint32_t>(g.records.size()));
    for (const DataRecord& r : g.records) {
        w.u16(r.move);
        w.i16(r.score);
        w.u8(r.flags);
        w.u8(r.depth);
        w.u8(r.seldepth);
        w.u32(r.nodes);
        w.u16(r.tb_hits);
        w.i16(r.pesto);
        w.i16(r.initiative);
        w.i16(r.mop_up);
        w.u8(r.drawish_pct);
        w.i16(r.static_eval);
        w.i16(r.correction);
        w.i16(r.qs_score);
        if (g.version >= 2) w.u16(r.best);
        if (g.version >= 3) {
            w.i16(r.prev_score);
            w.u8(r.stable);
        }
        if (g.version >= 4) {
            w.i16(r.back_score);
            w.u16(r.back_best);
            w.u8(r.back_depth);
            w.u32(r.back_nodes);
        }
    }
    Writer head;
    for (const char c : MAGIC) head.u8(static_cast<unsigned char>(c));
    head.u16(g.version);
    head.u32(static_cast<std::uint32_t>(w.b.size()));
    out.write(head.b.data(), static_cast<std::streamsize>(head.b.size()));
    out.write(w.b.data(), static_cast<std::streamsize>(w.b.size()));
    out.flush();
}

bool read_game(std::istream& in, DataGame& g) {
    std::string head(10, '\0');
    if (!in.read(head.data(), 10)) return false;
    if (head.compare(0, 4, MAGIC, 4) != 0) return false;
    Reader h{head, 4};
    g.version = h.u16();
    if (g.version < 1 || g.version > VERSION) return false;
    const std::uint32_t size = h.u32();
    std::string body(size, '\0');
    if (!in.read(body.data(), static_cast<std::streamsize>(size))) return false;

    Reader r{body};
    g.worker = r.u16();
    g.index = r.u32();
    g.seed = r.u64();
    g.opening = static_cast<std::uint8_t>(r.u8());
    g.start_index = r.u16();
    const unsigned fen_len = r.u8();
    if (r.p + fen_len > body.size()) return false;
    g.fen = body.substr(r.p, fen_len);
    r.p += fen_len;
    g.random_moves.resize(r.u8());
    for (Move& m : g.random_moves) m.data = r.u16();
    g.result = static_cast<DataResult>(r.u8());
    g.end = static_cast<DataEnd>(r.u8());
    g.plies = r.u16();
    const std::uint32_t n = r.u32();
    if (!r.ok || body.size() - r.p != static_cast<std::size_t>(n) * record_bytes(g.version)) return false;
    g.records.resize(n);
    for (DataRecord& d : g.records) {
        d.move = r.u16();
        d.score = r.i16();
        d.flags = static_cast<std::uint8_t>(r.u8());
        d.depth = static_cast<std::uint8_t>(r.u8());
        d.seldepth = static_cast<std::uint8_t>(r.u8());
        d.nodes = r.u32();
        d.tb_hits = r.u16();
        d.pesto = r.i16();
        d.initiative = r.i16();
        d.mop_up = r.i16();
        d.drawish_pct = static_cast<std::uint8_t>(r.u8());
        d.static_eval = r.i16();
        d.correction = r.i16();
        d.qs_score = r.i16();
        d.best = g.version >= 2 ? static_cast<std::uint16_t>(r.u16()) : 0;
        d.prev_score = g.version >= 3 ? static_cast<std::int16_t>(r.i16()) : 0;
        d.stable = g.version >= 3 ? static_cast<std::uint8_t>(r.u8()) : 0;
        if (g.version >= 4) {
            d.back_score = static_cast<std::int16_t>(r.i16());
            d.back_best = static_cast<std::uint16_t>(r.u16());
            d.back_depth = static_cast<std::uint8_t>(r.u8());
            d.back_nodes = r.u32();
        } else {
            d.back_score = 0;
            d.back_best = 0;
            d.back_depth = 0;
            d.back_nodes = 0;
        }
    }
    return r.ok;
}

bool insufficient_material(const Position& pos) {
    for (const Piece p : {WP, WR, WQ, BP, BR, BQ})
        if (pos.pieces(p)) return false;
    const Bitboard knights = pos.pieces(WN) | pos.pieces(BN);
    const Bitboard bishops = pos.pieces(WB) | pos.pieces(BB);
    if (!bishops) return std::popcount(knights) <= 1;
    if (knights) return false;
    constexpr Bitboard DARK = 0xAA55AA55AA55AA55ULL;  // a1 is dark
    return !(bishops & DARK) || !(bishops & ~DARK);
}

bool game_over(const Position& pos, const std::vector<std::uint64_t>& keys, std::size_t plies,
               int max_plies, DataEnd& end, DataResult& result) {
    const Color stm = pos.side_to_move();
    MoveList moves;
    int count = 0;
    generate_legal(pos, moves, count);
    if (count == 0) {
        const bool mated = pos.in_check(stm);
        end = mated ? DataEnd::Checkmate : DataEnd::Stalemate;
        result = !mated ? DataResult::Draw
                 : stm == Color::White ? DataResult::BlackWin
                                       : DataResult::WhiteWin;
    } else if (pos.halfmove_clock() >= 100) {
        end = DataEnd::FiftyMoves;
        result = DataResult::Draw;
    } else if (std::count(keys.begin(), keys.end(), pos.key()) >= 2) {
        end = DataEnd::Threefold;
        result = DataResult::Draw;
    } else if (insufficient_material(pos)) {
        end = DataEnd::Material;
        result = DataResult::Draw;
    } else if (plies >= static_cast<std::size_t>(max_plies)) {
        end = DataEnd::PlyCap;
        result = DataResult::None;
    } else {
        return false;
    }
    return true;
}

bool Adjudicator::update(int white_score, int phase, int ply, int tb_white, DataEnd& end,
                         DataResult& result) {
    auto win = [&](int side) {
        result = side > 0 ? DataResult::WhiteWin : DataResult::BlackWin;
    };
    if (opt.adj_tb && tb_white != TB_NONE) {
        end = DataEnd::AdjTablebase;
        if (tb_white == 0) result = DataResult::Draw;
        else win(tb_white);
        return true;
    }
    if (opt.adj_mate) {
        const int side = white_score > Searcher::MATE_BOUND ? 1 : white_score < -Searcher::MATE_BOUND ? -1 : 0;
        if (side && side == mate_side) ++mate_len;
        else mate_side = side, mate_len = side ? 1 : 0;
        if (mate_len >= opt.adj_mate) {
            end = DataEnd::AdjMate;
            win(mate_side);
            return true;
        }
    }
    if (opt.adj_score) {
        const int side = white_score >= opt.adj_score ? 1 : white_score <= -opt.adj_score ? -1 : 0;
        if (side && side == run_side) ++run_len;
        else run_side = side, run_len = side ? 1 : 0;
        if (run_len >= opt.adj_plies && phase <= opt.adj_phase) {
            end = DataEnd::AdjScore;
            win(run_side);
            return true;
        }
    }
    if (opt.adj_draw_score) {
        draw_len = std::abs(white_score) <= opt.adj_draw_score ? draw_len + 1 : 0;
        if (draw_len >= opt.adj_draw_plies && ply >= opt.adj_draw_ply) {
            end = DataEnd::AdjDraw;
            result = DataResult::Draw;
            return true;
        }
    }
    return false;
}

// One searched position as a record: the search's score, depth and flags and
// the eval terms at the root, with `best` the search's move (stored as the
// record's move; `labelpgn` replaces it with the move played), `score` and
// `tb` (the root's tablebase score, or NO_EVAL) for the caller.
static DataRecord search_record(Searcher& s, Position& pos, const std::vector<std::uint64_t>& keys,
                                std::uint64_t soft_nodes, std::uint64_t hard_nodes, Move& best, int& score,
                                int& tb, bool soft_scale = false, std::uint64_t drop_soft = 0,
                                std::uint64_t drop_hard = 0, bool* cheap = nullptr, int min_depth = 0) {
    const bool in_check = pos.in_check(pos.side_to_move());
    DataRecord r;
    const EvalParts parts = eval_parts(pos);
    const Searcher::RootEval root = s.root_eval(pos);
    SearchLimits limits;
    limits.nodes = hard_nodes;
    limits.soft_nodes = soft_nodes;
    limits.soft_scale = soft_scale;
    limits.min_depth = min_depth;
    // A position the export will drop whatever the search finds gets the
    // cheaper budget (DataOptions::drop_soft_nodes), and no depth floor.
    if (drop_soft) {
        const int verdict = in_check ? Searcher::NO_EVAL : s.tablebase_verdict(pos, limits);
        if (in_check || (verdict != Searcher::NO_EVAL && verdict != 0)) {
            limits.soft_nodes = drop_soft;
            limits.min_depth = 0;
            if (drop_hard) limits.nodes = drop_hard;
            if (cheap) *cheap = true;
        }
    }
    best = s.search(pos, limits, keys, false);
    score = s.last_score();
    r.move = best.data;
    r.score = to_i16(score);
    r.nodes = static_cast<std::uint32_t>(std::min<std::uint64_t>(s.nodes(), UINT32_MAX));
    r.tb_hits = static_cast<std::uint16_t>(std::min<std::uint64_t>(s.tb_hits(), 65535));
    r.best = best.data;
    r.prev_score = r.score;
    if (!s.iterations().empty()) {
        const auto& it = s.iterations();
        r.depth = static_cast<std::uint8_t>(std::min(it.back().depth, 255));
        r.seldepth = static_cast<std::uint8_t>(std::min(it.back().seldepth, 255));
        // The search's own stability (version 3): the iteration before the
        // last, and how long the last one's best move had held.
        if (it.size() >= 2) r.prev_score = to_i16(it[it.size() - 2].score);
        int held = 0;
        for (std::size_t i = it.size() - 1; i > 0 && it[i - 1].best == it.back().best; --i) ++held;
        r.stable = static_cast<std::uint8_t>(std::min(held, 255));
    }
    tb = s.root_tablebase_score();
    r.flags = static_cast<std::uint8_t>(
        (s.node_limit_hit() ? DF_HARD_CAP : 0) | (std::abs(score) > Searcher::MATE_BOUND ? DF_MATE : 0) |
        (tb != Searcher::NO_EVAL ? DF_TB_ROOT : 0) | (s.tb_hits() ? DF_TB_HITS : 0) |
        (parts.mop_up ? DF_MOP_UP : 0) | (in_check ? DF_IN_CHECK : 0) |
        (tb != Searcher::NO_EVAL && tb > 0 ? DF_TB_WIN : 0) | (tb != Searcher::NO_EVAL && tb < 0 ? DF_TB_LOSS : 0));
    r.pesto = to_i16(parts.pesto);
    r.initiative = to_i16(parts.initiative);
    r.mop_up = to_i16(parts.mop_up);
    r.drawish_pct = static_cast<std::uint8_t>(std::clamp(parts.drawish_pct, 0, 255));
    r.static_eval = to_i16(root.static_eval);
    r.correction = to_i16(root.correction);
    r.qs_score = to_i16(s.quiescence(pos, keys, true));  // plain: the captures alone, not the search just made (round 5 on)
    return r;
}

std::uint64_t generate_data(const DataOptions& opt, std::ostream& log) {
    using Clock = std::chrono::steady_clock;
    const auto started = Clock::now();
    auto hours_up = [&] {
        return opt.hours > 0 &&
               std::chrono::duration<double>(Clock::now() - started).count() >= opt.hours * 3600;
    };

    // Resume after the last complete game of an existing file. A file that
    // has content but no game at its start isn't ours (a mistyped path, a
    // path cut at a space): refused and left untouched, never truncated.
    std::uint32_t first = 0;
    if (std::filesystem::exists(opt.out)) {
        std::error_code ec;
        const std::uintmax_t size = std::filesystem::file_size(opt.out, ec);
        const std::uint64_t good = scan_existing(opt.out, first);
        if (ec || (size > 0 && good == 0)) {
            log << "info string gendata: " << opt.out
                << " exists and isn't FunComp data: left untouched, nothing generated" << std::endl;
            return 0;
        }
        if (good < size) {
            std::filesystem::resize_file(opt.out, good, ec);
            if (ec) {
                log << "info string gendata: cannot trim " << opt.out << " to its last complete game: "
                    << ec.message() << std::endl;
                return 0;
            }
        }
    }
    if (opt.share_net && nnue::active())
        log << "info string gendata: network "
            << (nnue::share() ? "shared between processes" : "not shared (a copy of its own)") << std::endl;
    write_params(opt, first);
    std::ofstream out(opt.out, std::ios::binary | std::ios::app);
    if (!out) {
        log << "info string gendata: cannot write " << opt.out << std::endl;
        return 0;
    }

    const std::vector<std::string> starts = bingo_starts();
    // The reservoir's three pools (early, late, end), when asked for. Each
    // worker keeps only its own lines, k x starts_stride + worker (2026-10-08:
    // 24 workers each holding all 8.7M starts took 26 GB of this PC's 32);
    // `total` counts every line, so a pool weighs the same for every worker.
    std::array<std::vector<std::string>, 3> pools;
    std::array<std::uint64_t, 3> total{};
    int pool_weight = 0;
    if (opt.starts_pct > 0) {
        static constexpr const char* POOL[3] = {"early", "late", "end"};
        for (std::size_t p = 0; p < 3; ++p) {
            std::ifstream f(opt.starts + "-" + POOL[p] + ".fen");
            for (std::string line; std::getline(f, line);) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.empty()) continue;
                if (opt.worker >= 0 && total[p] % static_cast<std::uint64_t>(opt.starts_stride) ==
                                           static_cast<std::uint64_t>(opt.worker))
                    pools[p].push_back(line);
                ++total[p];
            }
            if (total[p]) pool_weight += std::max(opt.starts_mix[p], 0);
            log << "info string gendata: reservoir " << POOL[p] << ' ' << total[p] << " starts (this worker's "
                << pools[p].size() << ")" << std::endl;
        }
        if (pool_weight == 0) {
            log << "info string gendata: no reservoir starts in " << opt.starts << "-*.fen" << std::endl;
            return 0;
        }
    }
    // The reservoir draw: whether game `index` starts there and from which
    // pool, by the seed and the index alone (the rest of the game's draws
    // continue from `rng`). -1 for a usual start.
    auto reservoir_pool = [&](std::uint32_t index, std::uint64_t& rng) -> int {
        rng = opt.seed * 0x9E3779B97F4A7C15ULL ^ index;
        splitmix64(rng);
        // Drawn only when the reservoir is on, so other runs' games stay as
        // they were.
        if (pool_weight == 0 || splitmix64(rng) % 100 >= static_cast<std::uint64_t>(opt.starts_pct)) return -1;
        // A pool by weight (empty pools weigh nothing; r < their sum, so the
        // walk stops at a pool with starts).
        int r = static_cast<int>(splitmix64(rng) % static_cast<std::uint64_t>(pool_weight));
        int p = 0;
        for (; p < 3; ++p) {
            const int w = total[static_cast<std::size_t>(p)] == 0 ? 0 : std::max(opt.starts_mix[static_cast<std::size_t>(p)], 0);
            if (r < w) break;
            r -= w;
        }
        return p;
    };
    // Each start is used once (the author, 2026-10-07: no reuse, no random
    // plies): this worker's k-th game from a pool takes line
    // k x starts_stride + worker, so workers never share a line; the count
    // so far is replayed on resume. Past the end of the list a game starts
    // as usual (more starts, or a new plan, then).
    std::array<std::uint64_t, 3> pool_used{};
    std::uint64_t exhausted = 0;
    if (pool_weight > 0) {
        if (opt.worker < 0 || opt.worker >= opt.starts_stride) {
            log << "info string gendata: worker " << opt.worker << " outside the start stride "
                << opt.starts_stride << std::endl;
            return 0;
        }
        for (std::uint32_t i = 0; i < first; ++i) {
            std::uint64_t r = 0;
            const int p = reservoir_pool(i, r);
            if (p >= 0) ++pool_used[static_cast<std::size_t>(p)];
        }
    }
    // This worker's k-th draw from a pool: its k-th kept line, the list's
    // line k x starts_stride + worker.
    auto start_line = [&](std::size_t p) -> std::int64_t {
        const std::uint64_t k = pool_used[p]++;
        return k < pools[p].size() ? static_cast<std::int64_t>(k) : -1;
    };

    Searcher s(static_cast<std::size_t>(opt.hash_mb));
    // Room for the random plies too: they may outnumber max_plies.
    std::vector<StateInfo> states(static_cast<std::size_t>(std::max(opt.max_plies, opt.random_max)) + 16);
    std::uint64_t written = 0, filtered = 0, discarded = 0, positions = 0;
    std::uint64_t cheap_searches = 0;  // at the drop budget (DataOptions::drop_soft_nodes)
    std::uint64_t back_searches = 0, back_nodes = 0, forward_nodes = 0;  // the backward pass's, beside the forward's
    std::array<std::uint64_t, 4> results{};

    for (std::uint32_t index = first; (!opt.games || index - first < opt.games) && !hours_up();
         ++index) {
        DataGame g;
        g.version = opt.back_soft_nodes ? BACKWARD_VERSION : FORWARD_VERSION;
        g.worker = static_cast<std::uint16_t>(opt.worker);
        g.index = index;
        g.seed = opt.seed;
        // The opening depends only on the seed and the game index.
        std::uint64_t rng = 0;
        Position pos;
        const int pool = reservoir_pool(index, rng);
        const std::int64_t line = pool >= 0 ? start_line(static_cast<std::size_t>(pool)) : -1;
        if (pool >= 0 && line < 0) ++exhausted;
        int random_min = opt.random_min, random_max = opt.random_max;
        if (line >= 0) {
            const auto p = static_cast<std::size_t>(pool);
            g.opening = 2;
            g.start_index = static_cast<std::uint16_t>(p);
            pos.set_fen(pools[p][static_cast<std::size_t>(line)]);
            random_min = opt.starts_plies_min;
            random_max = std::max(opt.starts_plies_max, opt.starts_plies_min);
        } else if (opt.bingo_every > 0 && index % static_cast<std::uint32_t>(opt.bingo_every) ==
                                              static_cast<std::uint32_t>(opt.bingo_every - 1)) {
            g.opening = 1;
            g.start_index = static_cast<std::uint16_t>(
                (index / static_cast<std::uint32_t>(opt.bingo_every) +
                 108 * static_cast<std::uint32_t>(opt.worker)) %
                starts.size());
            pos.set_fen(starts[g.start_index]);
        }
        g.fen = pos.fen();

        std::vector<std::uint64_t> keys;
        std::size_t made = 0;
        auto play = [&](Move m) {
            keys.push_back(pos.key());
            pos.make_move(m, states[made++]);
        };

        const int span = random_max - random_min + 1;
        const int random_plies =
            random_min + static_cast<int>(splitmix64(rng) % static_cast<std::uint64_t>(span));
        bool alive = true;
        for (int i = 0; i < random_plies && alive; ++i) {
            MoveList moves;
            int count = 0;
            generate_legal(pos, moves, count);
            if (count == 0) {
                alive = false;
            } else {
                const Move m = moves[static_cast<std::size_t>(
                    splitmix64(rng) % static_cast<std::uint64_t>(count))];
                g.random_moves.push_back(m);
                play(m);
            }
        }
        if (!alive) {
            ++discarded;
            continue;
        }

        // Adjudication applies to adj_pct percent of games, drawn after the
        // opening (which stays as it was), and only when some rule is on.
        const bool adjudicate =
            (opt.adj_score || opt.adj_tb || opt.adj_draw_score || opt.adj_mate) &&
            splitmix64(rng) % 100 < static_cast<std::uint64_t>(std::max(opt.adj_pct, 0));
        Adjudicator adj(opt);
        // Long castling (see datagen.h): bit 0 White, bit 1 Black. Drawn only
        // when the option is on, so other runs' games stay as they were.
        unsigned castle_sides = 0;
        if (opt.castle_long_pct > 0 &&
            splitmix64(rng) % 100 < static_cast<std::uint64_t>(opt.castle_long_pct))
            castle_sides = 1u + static_cast<unsigned>(splitmix64(rng) % 3);
        s.new_game();
        // For the backward pass: each searched position and how much of the
        // game's history (keys) it had.
        std::vector<std::string> searched_fens;
        std::vector<std::size_t> searched_keys;
        while (!game_over(pos, keys, made, opt.max_plies, g.end, g.result)) {
            Move best;
            int score = 0, tb = Searcher::NO_EVAL;
            bool cheap = false;
            if (opt.back_soft_nodes) {
                searched_fens.push_back(pos.fen());
                searched_keys.push_back(keys.size());
            }
            const DataRecord r = search_record(s, pos, keys, opt.soft_nodes, opt.hard_nodes, best, score, tb,
                                               opt.soft_scale, opt.drop_soft_nodes, opt.drop_hard_nodes, &cheap,
                                               opt.min_depth);
            g.records.push_back(r);
            cheap_searches += cheap;
            forward_nodes += r.nodes;

            // Balance filter: a game whose first search is lopsided is kept
            // only as a stub (see datagen.h).
            if (g.records.size() == 1 && std::abs(score) > opt.max_open_score) {
                g.end = DataEnd::Filtered;
                g.result = DataResult::None;
                break;
            }
            // Adjudication (off unless set; see datagen.h). The tables'
            // verdict counts only at a tablebase root.
            if (adjudicate) {
                const bool white = pos.side_to_move() == Color::White;
                const int tb_white = tb == Searcher::NO_EVAL ? Adjudicator::TB_NONE
                                     : tb == 0                ? 0
                                     : (tb > 0) == white      ? 1
                                                              : -1;
                if (adj.update(white ? score : -score, phase_of(pos), static_cast<int>(made),
                               tb_white, g.end, g.result))
                    break;
            }
            if (best == Move{} || !is_legal_move(pos, best)) {
                log << "info string gendata: error, illegal move from " << pos.fen() << std::endl;
                return written;
            }
            Move played = best;
            const unsigned side = pos.side_to_move() == Color::White ? 1u : 2u;
            if ((castle_sides & side) && static_cast<int>(made) < opt.castle_long_by) {
                const bool quiet = pos.piece_at(best.to()) == NO_PIECE && !best.is_en_passant() &&
                                   best.promotion() == PieceType::None;
                const int king = side == 1 ? 4 : 60;  // e1, e8
                const Move long_castle(king, king - 2);
                const bool home = pos.piece_at(king) == (side == 1 ? WK : BK);
                const int queenside = side == 1 ? Position::WQ_CASTLE : Position::BQ_CASTLE;
                if (quiet && home && is_legal_move(pos, long_castle)) {
                    played = long_castle;
                    castle_sides &= ~side;
                } else if (home && best == Move(king, king + 2) && (pos.castling_rights() & queenside)) {
                    // It would castle short, giving up the long castle: its
                    // best other move instead (the label stays the full search's).
                    Move other;
                    int other_score = 0, other_tb = Searcher::NO_EVAL;
                    s.avoid_at_root(best);
                    search_record(s, pos, keys, opt.soft_nodes, opt.hard_nodes, other, other_score, other_tb,
                                  opt.soft_scale);
                    s.avoid_at_root(Move{});
                    if (other != Move{} && other != best && is_legal_move(pos, other)) played = other;
                }
                g.records.back().move = played.data;  // the move played, so the game replays
            }
            play(played);
        }
        if (g.records.empty()) {  // over before the first search
            ++discarded;
            continue;
        }
        g.plies = static_cast<std::uint16_t>(made);
        // The backward pass (DataOptions::back_soft_nodes): the game's
        // positions again, last first, the hash table as the game left it.
        if (opt.back_soft_nodes && g.end != DataEnd::Filtered) {
            SearchLimits limits;
            limits.soft_nodes = opt.back_soft_nodes;
            limits.nodes = opt.back_hard_nodes ? opt.back_hard_nodes : 8 * opt.back_soft_nodes;
            limits.soft_scale = opt.soft_scale;
            for (std::size_t i = g.records.size(); i-- > 0;) {
                DataRecord& r = g.records[i];
                if (r.flags & (DF_IN_CHECK | DF_TB_WIN | DF_TB_LOSS)) continue;
                Position bp;
                bp.set_fen(searched_fens[i]);
                const std::vector<std::uint64_t> history(
                    keys.begin(), keys.begin() + static_cast<std::ptrdiff_t>(searched_keys[i]));
                const Move b = s.search(bp, limits, history, false);
                r.back_score = to_i16(s.last_score());
                r.back_best = b.data;
                r.back_depth = static_cast<std::uint8_t>(
                    std::clamp(s.iterations().empty() ? 1 : s.iterations().back().depth, 1, 255));
                r.back_nodes = static_cast<std::uint32_t>(std::min<std::uint64_t>(s.nodes(), UINT32_MAX));
                ++back_searches;
                back_nodes += r.back_nodes;
            }
        }
        if (!nnue::shared_intact()) {
            log << "info string gendata: error, the shared network file changed on disk: stopped, this game not written"
                << std::endl;
            return written;
        }
        write_game(out, g);
        ++written;
        if (g.end == DataEnd::Filtered) {
            ++filtered;
        } else {
            positions += g.records.size();
            ++results[static_cast<std::size_t>(g.result)];
        }
        if (written % 100 == 0) {
            log << "info string gendata: worker " << opt.worker << ", " << written << " games ("
                << filtered << " filtered, " << discarded << " discarded), " << positions
                << " positions, W/D/B/none " << results[0] << '/' << results[1] << '/'
                << results[2] << '/' << results[3];
            // How far this worker has walked each pool (a re-plan keeps that
            // prefix of the lists, so no start is used twice).
            if (pool_weight > 0)
                log << ", reservoir walked " << pool_used[0] << '/' << pool_used[1] << '/' << pool_used[2];
            if (opt.drop_soft_nodes) log << ", at the drop budget " << cheap_searches;
            log << std::endl;
        }
    }
    log << "info string gendata: done, worker " << opt.worker << ", " << written << " games ("
        << filtered << " filtered, " << discarded << " discarded), " << positions
        << " positions";
    if (pool_weight > 0)
        log << ", reservoir starts used " << pool_used[0] << '/' << pool_used[1] << '/' << pool_used[2]
            << " (early/late/end, this worker's count since game 0), " << exhausted
            << " past the end of a list (started as usual)";
    if (opt.drop_soft_nodes) log << ", at the drop budget " << cheap_searches << " searches";
    if (opt.back_soft_nodes)
        log << ", backward " << back_searches << " searches, " << back_nodes << " nodes beside " << forward_nodes
            << " forward";
    log << std::endl;
    return written;
}

std::uint64_t label_pgn(const LabelPgnOptions& opt, std::ostream& log) {
    std::ifstream in(opt.in);
    std::ofstream out(opt.out, std::ios::binary);
    if (!in || !out) {
        log << "info string labelpgn: cannot read " << opt.in << " or write " << opt.out << std::endl;
        return 0;
    }
    const std::vector<pgn::Game> games = pgn::read(in);
    Searcher s(static_cast<std::size_t>(opt.hash_mb));
    std::uint64_t written = 0, positions = 0, no_result = 0, unparsed = 0;
    for (std::size_t gi = 0; gi < games.size(); ++gi) {
        const pgn::Game& pg = games[gi];
        DataGame g;
        g.result = pg.result == "1-0"       ? DataResult::WhiteWin
                   : pg.result == "0-1"     ? DataResult::BlackWin
                   : pg.result == "1/2-1/2" ? DataResult::Draw
                                            : DataResult::None;
        Position pos;
        if (g.result == DataResult::None || !pos.set_fen(pg.fen)) {
            ++no_result;
            continue;
        }
        g.worker = static_cast<std::uint16_t>(opt.worker);
        g.index = static_cast<std::uint32_t>(gi);
        g.seed = opt.seed;
        g.fen = pos.fen();
        g.end = DataEnd::External;
        std::vector<StateInfo> states(pg.moves.size() + 1);
        std::vector<std::uint64_t> keys;
        std::size_t made = 0;
        s.new_game();
        for (const std::string& san : pg.moves) {
            const Move m = pgn::parse_san(pos, san);
            if (m == Move{}) {
                ++unparsed;
                break;
            }
            if (static_cast<int>(made) < opt.skip) {
                g.random_moves.push_back(m);
            } else {
                Move best;
                int score = 0, tb = Searcher::NO_EVAL;
                DataRecord r = search_record(s, pos, keys, opt.soft_nodes, opt.hard_nodes, best, score, tb);
                r.move = m.data;  // the move played, so the game replays
                g.records.push_back(r);
            }
            keys.push_back(pos.key());
            pos.make_move(m, states[made++]);
        }
        if (g.records.empty()) continue;
        g.plies = static_cast<std::uint16_t>(std::min<std::size_t>(made, 65535));
        write_game(out, g);
        ++written;
        positions += g.records.size();
        if (written % 50 == 0)
            log << "info string labelpgn: " << written << " games, " << positions << " positions" << std::endl;
    }
    log << "info string labelpgn: done, " << written << " games, " << positions << " positions ("
        << no_result << " games without a result or start, " << unparsed << " cut at an unreadable move)"
        << std::endl;
    return written;
}

std::uint64_t nnue_eval_data(const std::string& in, std::uint64_t max_positions, std::ostream& out, bool fen) {
    std::uint64_t written = 0, game = 0;
    for (const std::string& f : data_files(in)) {
        std::ifstream file(f, std::ios::binary);
        DataGame g;
        while (read_game(file, g) && (!max_positions || written < max_positions)) {
            ++game;
            if (g.end == DataEnd::Filtered || g.result == DataResult::None) continue;
            Position pos;
            pos.set_fen(g.fen);
            std::vector<StateInfo> states(g.random_moves.size() + g.records.size() + 1);
            std::size_t made = 0;
            for (const Move m : g.random_moves) pos.make_move(m, states[made++]);
            for (const DataRecord& r : g.records) {
                Move m;
                m.data = r.move;
                if (!(r.flags & (DF_IN_CHECK | DF_MATE | DF_TB_ROOT)) && std::abs(r.score) < Searcher::DECISIVE_BOUND &&
                    (!max_positions || written < max_positions)) {
                    const bool white = pos.side_to_move() == Color::White;
                    const double res = g.result == DataResult::Draw ? 0.5 : (g.result == DataResult::WhiteWin) == white ? 1.0 : 0.0;
                    out << evaluate(pos) << ' ' << r.score << ' ' << res;
                    if (fen) out << " | " << pos.fen() << " | " << game << ' ' << made;
                    out << '\n';
                    ++written;
                }
                if (!is_legal_move(pos, m)) break;
                pos.make_move(m, states[made++]);
            }
        }
    }
    return written;
}

void dump_data(const std::string& path, std::uint64_t max_games, std::ostream& out, bool back) {
    std::ifstream in(path, std::ios::binary);
    DataGame g;
    for (std::uint64_t n = 0; (!max_games || n < max_games) && read_game(in, g); ++n) {
        out << "# game worker=" << g.worker << " index=" << g.index
            << " opening=" << (g.opening == 2 ? "reservoir" : g.opening ? "bingo" : "standard") << " start=" << g.start_index
            << " result=";
        static constexpr const char* RESULT[4] = {"1-0", "1/2-1/2", "0-1", "*"};
        out << RESULT[static_cast<int>(g.result) & 3] << " end=" << data_end_name(g.end)
            << " plies=" << g.plies << " fen=" << g.fen << " random=";
        Position pos;
        pos.set_fen(g.fen);
        std::vector<StateInfo> states(g.random_moves.size() + g.records.size() + 1);
        std::size_t made = 0;
        for (const Move m : g.random_moves) {
            out << pos.move_to_uci(m) << ' ';
            pos.make_move(m, states[made++]);
        }
        out << '\n'
            << "# fen | move score flags depth seldepth nodes tbhits pesto initiative mopup "
               "drawish static_eval correction qs" << (back ? " back_score back_depth back_nodes best back_best" : "")
            << '\n';
        for (const DataRecord& r : g.records) {
            Move m;
            m.data = r.move;
            out << pos.fen() << " | " << pos.move_to_uci(m) << ' ' << r.score << ' '
                << static_cast<int>(r.flags) << ' ' << static_cast<int>(r.depth) << ' '
                << static_cast<int>(r.seldepth) << ' ' << r.nodes << ' ' << r.tb_hits << ' '
                << r.pesto << ' ' << r.initiative << ' ' << r.mop_up << ' '
                << static_cast<int>(r.drawish_pct) << ' ' << r.static_eval << ' ' << r.correction
                << ' ' << r.qs_score;
            if (back) {
                // The searches' own best moves (a rescore's may differ from the move played); "-" none.
                auto uci = [&](std::uint16_t d) {
                    Move b;
                    b.data = d;
                    return d ? pos.move_to_uci(b) : std::string("-");
                };
                out << ' ' << r.back_score << ' ' << static_cast<int>(r.back_depth) << ' ' << r.back_nodes << ' '
                    << uci(r.best) << ' ' << uci(r.back_best);
            }
            out << '\n';
            if (&r == &g.records.back() && g.end == DataEnd::Filtered) break;
            if (!is_legal_move(pos, m)) {
                out << "# error: illegal move\n";
                break;
            }
            pos.make_move(m, states[made++]);
        }
    }
}

void data_stats(const std::string& path, std::ostream& out, int scale) {
    const std::vector<std::string> files = data_files(path);
    if (files.empty()) {
        out << "info string datastats: no data in " << path << std::endl;
        return;
    }

    std::uint64_t bytes = 0, games = 0, records = 0, labelled = 0, nodes = 0, hard_cap = 0,
                  mates = 0, in_check = 0, tb_root = 0, tb_won_drawn_fifty = 0,
                  tb_won_drawn_any = 0, illegal = 0;
    // Version 4: the backward pass's labels beside the forward ones.
    std::uint64_t back = 0, back_nodes = 0, back_moved = 0, back_both = 0, back_over50 = 0;
    double back_diff = 0;
    std::array<std::uint64_t, DATA_END_COUNT> ends{};
    std::uint64_t by_kind[3][4] = {};      // [opening][result] of played games
    std::uint64_t filtered_kind[3] = {}, games_kind[3] = {};
    std::map<int, std::uint64_t> length_hist, searched_hist;  // buckets of 20 plies / 20 positions
    // Completed depth by phase bucket: 17-24, 9-16, 0-8.
    std::uint64_t depth_sum[3] = {}, depth_n[3] = {};
    std::map<int, std::uint64_t> depth_hist[3];
    // Conversion: the first time one side's search score held >= CONV_T[t]
    // for CONV_PLIES plies in a row, by phase bucket then; the game's outcome
    // for that side [won, drawn, lost]; [0] every played game (what the
    // labels say), [1] games that ended without adjudication (what FunComp
    // converts).
    // Thresholds on round 1's scale (120), in this data's units.
    static constexpr int CONV_T120[3] = {200, 400, 800};
    int CONV_T[3];
    for (int t = 0; t < 3; ++t) CONV_T[t] = CONV_T120[t] * std::max(scale, 1) / 120;
    static constexpr int CONV_PLIES = 4;
    std::uint64_t conv[2][3][3][3] = {};

    for (const std::string& f : files) {
        bytes += std::filesystem::file_size(f);
        std::ifstream in(f, std::ios::binary);
        DataGame g;
        while (read_game(in, g)) {
            ++games;
            ++ends[static_cast<std::size_t>(g.end) % DATA_END_COUNT];
            const int kind = std::min<int>(g.opening, 2);  // standard, bingo, reservoir
            ++games_kind[kind];
            records += g.records.size();
            if (g.end == DataEnd::Filtered) {
                ++filtered_kind[kind];
                continue;
            }
            ++by_kind[kind][static_cast<int>(g.result) & 3];
            ++length_hist[g.plies / 20 * 20];
            ++searched_hist[static_cast<int>(g.records.size()) / 20 * 20];
            labelled += g.records.size();

            Position pos;
            pos.set_fen(g.fen);
            std::vector<StateInfo> states(g.random_moves.size() + g.records.size() + 1);
            std::size_t made = 0;
            for (const Move m : g.random_moves) pos.make_move(m, states[made++]);
            bool tb_won = false;
            int run_side[3] = {0, 0, 0}, run_len[3] = {0, 0, 0};
            int reached[3] = {0, 0, 0}, reached_bucket[3] = {0, 0, 0};  // side +1 White, -1 Black
            for (const DataRecord& r : g.records) {
                nodes += r.nodes;
                hard_cap += (r.flags & DF_HARD_CAP) != 0;
                mates += (r.flags & DF_MATE) != 0;
                in_check += (r.flags & DF_IN_CHECK) != 0;
                tb_root += (r.flags & DF_TB_ROOT) != 0;
                tb_won = tb_won || (r.flags & (DF_TB_WIN | DF_TB_LOSS));
                if (r.back_depth) {
                    ++back;
                    back_nodes += r.back_nodes;
                    back_moved += r.back_best != (r.best ? r.best : r.move);
                    if (std::abs(r.score) < Searcher::DECISIVE_BOUND && std::abs(r.back_score) < Searcher::DECISIVE_BOUND) {
                        const int d = std::abs(r.back_score - r.score);
                        ++back_both;
                        back_diff += d;
                        back_over50 += d > 50;
                    }
                }
                const int phase = phase_of(pos);
                const int bucket = phase >= 17 ? 0 : phase >= 9 ? 1 : 2;
                const int white_score = pos.side_to_move() == Color::White ? r.score : -r.score;
                for (int t = 0; t < 3; ++t) {
                    const int side = white_score >= CONV_T[t] ? 1 : white_score <= -CONV_T[t] ? -1 : 0;
                    run_len[t] = side != 0 && side == run_side[t] ? run_len[t] + 1 : side != 0 ? 1 : 0;
                    run_side[t] = side;
                    if (!reached[t] && run_len[t] >= CONV_PLIES) {
                        reached[t] = side;
                        reached_bucket[t] = bucket;
                    }
                }
                depth_sum[bucket] += r.depth;
                ++depth_n[bucket];
                ++depth_hist[bucket][r.depth];
                Move m;
                m.data = r.move;
                if (!is_legal_move(pos, m)) {
                    ++illegal;
                    break;
                }
                pos.make_move(m, states[made++]);
            }
            const bool adjudicated = g.end == DataEnd::AdjScore || g.end == DataEnd::AdjTablebase || g.end == DataEnd::AdjDraw ||
                                     g.end == DataEnd::AdjMate;
            const int res = static_cast<int>(g.result) & 3;  // 0 White won, 1 draw, 2 Black won
            for (int t = 0; t < 3; ++t) {
                if (!reached[t] || res == 3) continue;
                const int outcome = res == 1 ? 1 : (res == 0) == (reached[t] == 1) ? 0 : 2;
                ++conv[0][t][reached_bucket[t]][outcome];
                if (!adjudicated) ++conv[1][t][reached_bucket[t]][outcome];
            }
            if (tb_won && g.result == DataResult::Draw) {
                ++tb_won_drawn_any;
                if (g.end == DataEnd::FiftyMoves) ++tb_won_drawn_fifty;
            }
        }
    }

    auto pct = [](std::uint64_t a, std::uint64_t b) {
        std::ostringstream s;
        s << std::fixed << std::setprecision(1) << (b ? 100.0 * static_cast<double>(a) / static_cast<double>(b) : 0.0) << '%';
        return s.str();
    };
    const std::uint64_t played = games - ends[static_cast<std::size_t>(DataEnd::Filtered)];
    out << "files " << files.size() << ", bytes " << bytes << ", games " << games
        << " (played " << played << ", filtered " << games - played << ")\n";
    out << "positions " << labelled << " in played games (" << records << " records), bytes/record "
        << std::fixed << std::setprecision(2)
        << (records ? static_cast<double>(bytes) / static_cast<double>(records) : 0.0) << '\n';
    out << "searched positions per played game " << std::setprecision(1)
        << (played ? static_cast<double>(labelled) / static_cast<double>(played) : 0.0)
        << ", nodes per position "
        << (labelled ? static_cast<double>(nodes) / static_cast<double>(labelled) : 0.0) << '\n';
    out << "hard-cap hits " << hard_cap << " (" << pct(hard_cap, labelled) << "), mate scores "
        << mates << ", in check " << in_check << ", tablebase roots " << tb_root << '\n';
    if (back)
        out << "backward pass: " << back << " positions (" << pct(back, labelled) << "), nodes per position "
            << static_cast<double>(back_nodes) / static_cast<double>(back) << ", best move changed "
            << pct(back_moved, back) << ", score moved " << (back_both ? back_diff / static_cast<double>(back_both) : 0.0)
            << " cp on average, more than 50 cp " << pct(back_over50, back_both) << '\n';
    out << "tablebase-decided games drawn: " << tb_won_drawn_any << " (by the fifty-move rule "
        << tb_won_drawn_fifty << ")\nillegal moves " << illegal << '\n';
    out << "terminations:";
    for (int e = 0; e < DATA_END_COUNT; ++e)
        out << ' ' << data_end_name(static_cast<DataEnd>(e)) << ' ' << ends[static_cast<std::size_t>(e)]
            << " (" << pct(ends[static_cast<std::size_t>(e)], games) << ')';
    out << '\n';
    static constexpr const char* KIND[3] = {"standard", "bingo", "reservoir"};
    for (int k = 0; k < 3; ++k) {
        if (k == 2 && !games_kind[k]) continue;  // only data with reservoir starts
        const std::uint64_t n = by_kind[k][0] + by_kind[k][1] + by_kind[k][2] + by_kind[k][3];
        out << KIND[k] << ": games " << games_kind[k] << ", filtered " << filtered_kind[k] << " ("
            << pct(filtered_kind[k], games_kind[k]) << "), W/D/B " << pct(by_kind[k][0], n) << " / "
            << pct(by_kind[k][1], n) << " / " << pct(by_kind[k][2], n) << ", no result "
            << by_kind[k][3] << '\n';
    }
    static constexpr const char* PHASE[3] = {"phase 17-24", "phase 9-16", "phase 0-8"};
    for (int b = 0; b < 3; ++b) {
        out << PHASE[b] << ": positions " << depth_n[b] << ", mean depth "
            << (depth_n[b] ? static_cast<double>(depth_sum[b]) / static_cast<double>(depth_n[b]) : 0.0)
            << ", depths";
        for (const auto& [d, n] : depth_hist[b]) out << ' ' << d << ':' << n;
        out << '\n';
    }
    for (int v = 0; v < 2; ++v) {
        out << "conversion, " << (v == 0 ? "every played game" : "games ended without adjudication")
            << " (one side's score >= T for " << CONV_PLIES << " plies in a row, first time; that side's outcome):\n";
        for (int t = 0; t < 3; ++t)
            for (int b = 0; b < 3; ++b) {
                const std::uint64_t* c = conv[v][t][b];
                const std::uint64_t n = c[0] + c[1] + c[2];
                out << "  T " << CONV_T120[t] << (scale != 120 ? " (" + std::to_string(CONV_T[t]) + " here)" : std::string()) << ", " << PHASE[b] << ": games " << n << ", won " << pct(c[0], n)
                    << ", drawn " << pct(c[1], n) << ", lost " << pct(c[2], n) << '\n';
            }
    }
    out << "game length (plies, by 20):";
    for (const auto& [l, n] : length_hist) out << ' ' << l << ':' << n;
    out << "\nsearched positions per game (by 20):";
    for (const auto& [l, n] : searched_hist) out << ' ' << l << ':' << n;
    out << '\n';
}

namespace {

// A first-index file (datasplit, datatext part=): for each data file's name
// without ".bin", the first game index of its second part, and the file's
// bytes then if they were recorded.
struct Cut {
    std::uint64_t index, bytes;
    bool has_bytes;
};

std::map<std::string, Cut> read_cuts(const std::string& path) {
    const auto number = [](const std::string& s) {
        return !s.empty() && s.size() <= 18 &&
               std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; });
    };
    std::map<std::string, Cut> cuts;
    std::ifstream f(path);
    for (std::string line; std::getline(f, line);) {
        if (line.compare(0, 3, "\xEF\xBB\xBF") == 0) line.erase(0, 3);  // a byte order mark (PowerShell)
        std::istringstream ss(line);
        std::string name, index, bytes;
        if (!(ss >> name >> index) || name[0] == '#' || !number(index)) continue;
        ss >> bytes;
        cuts[name] = {std::stoull(index), number(bytes) ? std::stoull(bytes) : 0, number(bytes)};
    }
    return cuts;
}

}  // namespace

std::uint64_t split_data(const SplitOptions& opt, std::ostream& log) {
    if (opt.a.empty() != opt.b.empty()) {
        log << "info string datasplit: a=<dir> and b=<dir> go together (neither: a dry run)" << std::endl;
        return 0;
    }
    const bool dry = opt.a.empty();
    const std::map<std::string, Cut> cuts = read_cuts(opt.index);
    const std::vector<std::string> files = data_files(opt.in);
    if (files.empty() || cuts.empty()) {
        log << "info string datasplit: no data in " << opt.in << ", or no lines in the index file " << opt.index
            << std::endl;
        return 0;
    }
    std::error_code ec;
    if (!dry) {
        std::filesystem::create_directories(opt.a, ec);
        std::filesystem::create_directories(opt.b, ec);
    }

    std::uint64_t split = 0, refused = 0, left_out = 0;
    std::uint64_t games[2] = {}, positions[2] = {}, bytes[2] = {};  // [before the cut, from it on]
    for (const std::string& f : files) {
        const std::string name = std::filesystem::path(f).stem().string();
        const auto refuse = [&](const std::string& why) {
            log << "info string datasplit: " << name << ": not split: " << why << std::endl;
            ++refused;
        };
        const auto c = cuts.find(name);
        if (c == cuts.end()) {
            refuse("no line with a game index for it in the index file");
            continue;
        }
        const std::filesystem::path out[2] = {std::filesystem::path(opt.a) / (name + ".bin"),
                                              std::filesystem::path(opt.b) / (name + ".bin")};
        if (!dry && (std::filesystem::exists(out[0]) || std::filesystem::exists(out[1]))) {
            refuse("its output exists: never overwritten");
            continue;
        }

        // The file's complete blocks: where the second part begins and where
        // they end, and the two parts' counts as gendata's log counts them.
        std::ifstream in(f, std::ios::binary);
        std::uint64_t cut = 0, end = 0, g_n[2] = {}, p_n[2] = {};
        std::uint32_t last = 0;
        bool rising = true;
        DataGame g;
        while (read_game(in, g)) {
            rising = rising && (end == 0 || g.index > last);
            last = g.index;
            end = static_cast<std::uint64_t>(in.tellg());
            const int part = g.index >= c->second.index;
            if (!part) cut = end;
            ++g_n[part];
            if (g.end != DataEnd::Filtered) p_n[part] += g.records.size();
        }
        if (end == 0) {
            refuse("no game in it");
            continue;
        }
        if (!rising) {
            refuse("its game indexes don't rise through the file");
            continue;
        }
        if (c->second.has_bytes && c->second.bytes != cut) {
            refuse("game " + std::to_string(c->second.index) + " begins at byte " + std::to_string(cut) +
                   ", the index file says " + std::to_string(c->second.bytes));
            continue;
        }

        if (!dry) {
            const std::uint64_t range[2][2] = {{0, cut}, {cut, end}};
            bool ok = true;
            std::vector<char> buf(1 << 20);
            for (int part = 0; part < 2 && ok; ++part) {
                std::uint64_t left = range[part][1] - range[part][0];
                if (!left) continue;
                std::ofstream to(out[part], std::ios::binary);
                in.clear();
                in.seekg(static_cast<std::streamoff>(range[part][0]));
                while (left && in && to) {
                    const std::size_t n = static_cast<std::size_t>(std::min<std::uint64_t>(left, buf.size()));
                    in.read(buf.data(), static_cast<std::streamsize>(n));
                    to.write(buf.data(), in.gcount());
                    left -= static_cast<std::uint64_t>(in.gcount());
                }
                to.close();
                ok = !left && !to.fail() &&
                     std::filesystem::file_size(out[part], ec) == range[part][1] - range[part][0] && !ec;
            }
            if (!ok) {  // half a part must not pass for a whole one
                std::filesystem::remove(out[0], ec);
                std::filesystem::remove(out[1], ec);
                refuse("cannot write " + out[0].string() + " or " + out[1].string() + ": removed");
                continue;
            }
        }
        ++split;
        log << "info string datasplit: " << name << ": cut at game " << c->second.index << ", byte " << cut
            << (c->second.has_bytes ? " (as recorded)" : "") << ": before it " << g_n[0] << " games, " << p_n[0]
            << " positions; from it on " << g_n[1] << " games, " << p_n[1] << " positions";
        const std::uint64_t size = std::filesystem::file_size(f, ec);
        if (!ec && size > end) {
            log << "; " << size - end << " bytes after the last complete game left out";
            left_out += size - end;
        }
        log << std::endl;
        for (int part = 0; part < 2; ++part) {
            games[part] += g_n[part];
            positions[part] += p_n[part];
        }
        bytes[0] += cut;
        bytes[1] += end - cut;
    }
    log << "info string datasplit: " << split << " files "
        << (dry ? "would be split (a dry run: nothing written)" : "split") << ", " << refused
        << " not split; before the cuts " << games[0] << " games, " << positions[0] << " positions, " << bytes[0]
        << " bytes; from them on " << games[1] << " games, " << positions[1] << " positions, " << bytes[1]
        << " bytes; after the files' last complete games " << left_out << " bytes left out" << std::endl;
    return split;
}

std::uint64_t shuffle_data(const ShuffleOptions& opt, std::ostream& log) {
    namespace fs = std::filesystem;
    const auto start = std::chrono::steady_clock::now();
    const std::uint64_t rec = opt.record;
    std::vector<std::string> made;  // files to remove if anything fails
    const auto fail = [&](const std::string& why) {
        std::error_code ec;
        for (const std::string& f : made) fs::remove(f, ec);
        log << "info string datashuffle: " << why << "; nothing written" << std::endl;
        return std::uint64_t{0};
    };
    std::error_code ec;
    if (rec == 0 || rec > 4096) return fail("a record of " + std::to_string(rec) + " bytes");
    const std::uint64_t size = fs::file_size(opt.in, ec);
    if (ec) return fail("no file " + opt.in);
    if (size % rec) return fail(opt.in + " is not a whole number of " + std::to_string(rec) + "-byte records");
    if (opt.out.empty() || fs::exists(opt.out)) return fail("the output " + opt.out + " exists (never overwritten)");
    const std::uint64_t n = size / rec;
    const std::uint64_t mem = std::max<std::uint64_t>(opt.mem_mb, 16) << 20;
    // Parts sized to a tenth below the limit, so the largest (a few records over the mean) still fits.
    const std::uint64_t parts = size + size / 10 <= mem ? 1 : (size + size / 10) / mem + 1;
    std::uint64_t rng = opt.seed * 0x9E3779B97F4A7C15ULL ^ size;
    const auto below = [&](std::uint64_t bound) {  // uniform in [0, bound), Lemire's multiply-shift
        return static_cast<std::uint64_t>((static_cast<unsigned __int128>(splitmix64(rng)) * bound) >> 64);
    };
    const auto shuffle_block = [&](std::vector<char>& block) {  // Fisher-Yates on the records
        std::vector<char> tmp(rec);
        for (std::uint64_t i = block.size() / rec; i > 1; --i) {
            const std::uint64_t j = below(i);
            if (j == i - 1) continue;
            char* a = block.data() + (i - 1) * rec;
            char* b = block.data() + j * rec;
            std::memcpy(tmp.data(), a, rec);
            std::memcpy(a, b, rec);
            std::memcpy(b, tmp.data(), rec);
        }
    };
    const std::string part = opt.out + ".part";
    std::FILE* out = std::fopen(part.c_str(), "wb");
    if (!out) return fail("cannot write " + part);
    made.push_back(part);
    std::FILE* in = std::fopen(opt.in.c_str(), "rb");
    if (!in) { std::fclose(out); return fail("cannot read " + opt.in); }
    std::uint64_t written = 0;
    bool ok = true;
    if (parts == 1) {  // all of it in memory
        std::vector<char> block(size);
        ok = std::fread(block.data(), 1, size, in) == size;
        if (ok) {
            shuffle_block(block);
            ok = std::fwrite(block.data(), 1, size, out) == size;
            written = n;
        }
    } else {
        // Pass 1: every record to a part at random, through a buffer per part.
        std::vector<std::FILE*> tmp(parts, nullptr);
        std::vector<std::string> names(parts);
        std::vector<std::vector<char>> buffers(parts);
        const std::size_t flush_at = (std::size_t{1} << 20) / rec * rec;
        for (std::uint64_t p = 0; p < parts && ok; ++p) {
            names[p] = opt.out + ".shuffle-" + std::to_string(p);
            tmp[p] = std::fopen(names[p].c_str(), "wb");
            ok = tmp[p] != nullptr;
            if (ok) made.push_back(names[p]);
            buffers[p].reserve(flush_at);
        }
        std::vector<char> chunk((std::size_t{64} << 20) / rec * rec);
        std::size_t got;
        while (ok && (got = std::fread(chunk.data(), 1, chunk.size(), in)) > 0) {
            if (got % rec) { ok = false; break; }
            for (std::size_t off = 0; off < got; off += rec) {
                std::vector<char>& b = buffers[below(parts)];
                b.insert(b.end(), chunk.data() + off, chunk.data() + off + rec);
                if (b.size() >= flush_at) {
                    const std::uint64_t p = static_cast<std::uint64_t>(&b - buffers.data());
                    ok = ok && std::fwrite(b.data(), 1, b.size(), tmp[p]) == b.size();
                    b.clear();
                }
            }
        }
        for (std::uint64_t p = 0; p < parts; ++p) {
            if (ok && !buffers[p].empty()) ok = std::fwrite(buffers[p].data(), 1, buffers[p].size(), tmp[p]) == buffers[p].size();
            if (tmp[p] && std::fclose(tmp[p]) != 0) ok = false;
        }
        // Pass 2: each part shuffled in memory and appended; a part goes once it is in the output.
        for (std::uint64_t p = 0; p < parts && ok; ++p) {
            const std::uint64_t bytes = fs::file_size(names[p], ec);
            if (ec) { ok = false; break; }
            std::vector<char> block(bytes);
            std::FILE* f = std::fopen(names[p].c_str(), "rb");
            ok = f && std::fread(block.data(), 1, bytes, f) == bytes;
            if (f) std::fclose(f);
            if (!ok) break;
            shuffle_block(block);
            ok = std::fwrite(block.data(), 1, bytes, out) == bytes;
            written += bytes / rec;
            fs::remove(names[p], ec);
        }
    }
    std::fclose(in);
    if (std::fclose(out) != 0) ok = false;
    if (!ok || written != n) return fail("a read or a write failed (the disk full?) after " + std::to_string(written) + " of " + std::to_string(n) + " records");
    fs::rename(part, opt.out, ec);
    if (ec) return fail("cannot rename " + part + " to " + opt.out);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    log << "info string datashuffle: " << n << " records of " << rec << " bytes in " << parts << " part"
        << (parts == 1 ? "" : "s") << ", " << std::fixed << std::setprecision(0) << secs << " s, to " << opt.out << std::endl;
    return n;
}

std::uint64_t relabel_data(const RelabelOptions& opt, std::ostream& log) {
    namespace fs = std::filesystem;
    const auto start = std::chrono::steady_clock::now();
    constexpr std::uint64_t REC = 32;
    const std::string part = opt.out + ".part";
    const auto fail = [&](const std::string& why) {
        std::error_code ec;
        fs::remove(part, ec);
        log << "info string relabel: " << why << "; nothing written" << std::endl;
        return std::uint64_t{0};
    };
    if (!nnue::active()) return fail("no network loaded (EvalFile)");
    std::error_code ec;
    const std::uint64_t size = fs::file_size(opt.in, ec);
    if (ec) return fail("no file " + opt.in);
    if (size % REC) return fail(opt.in + " is not a whole number of 32-byte records");
    if (opt.out.empty() || fs::exists(opt.out)) return fail("the output " + opt.out + " exists (never overwritten)");
    std::FILE* in = std::fopen(opt.in.c_str(), "rb");
    std::FILE* out = std::fopen(part.c_str(), "wb");
    if (!in || !out) {
        if (in) std::fclose(in);
        if (out) std::fclose(out);
        return fail("cannot read " + opt.in + " or write " + part);
    }
    const int saved_scale = params::nnue_scale;
    params::nnue_scale = opt.scale;
    // ChessBoard: occupancy (u64), then a nibble a piece in square order (bit 3 the side not to move, bits 0-2
    // the type, pawn to king), the score (i16, side to move's), the result, the king squares and 3 spare bytes.
    static constexpr char PIECE_CHARS[2][6] = {{'P', 'N', 'B', 'R', 'Q', 'K'}, {'p', 'n', 'b', 'r', 'q', 'k'}};
    std::vector<unsigned char> block(REC * 65536);
    std::uint64_t written = 0, kept = 0, signed_pairs = 0, same_sign = 0;
    bool ok = true;
    Position pos;
    std::string fen;
    for (std::size_t got; ok && (got = std::fread(block.data(), 1, block.size(), in)) > 0;) {
        if (got % REC) {
            ok = false;
            break;
        }
        for (std::size_t at = 0; at < got; at += REC) {
            unsigned char* r = block.data() + at;
            std::uint64_t occ;
            std::memcpy(&occ, r, 8);  // little-endian, as bullet writes it
            char board[64] = {};
            int kings[2] = {0, 0};
            bool valid = std::popcount(occ) <= 32;
            int idx = 0;
            for (Bitboard b = occ; b && valid; b &= b - 1, ++idx) {
                const int pc = (r[8 + idx / 2] >> (4 * (idx & 1))) & 0xF;
                const int side = pc >> 3, type = pc & 7;
                valid = type <= 5;
                if (!valid) break;
                kings[side] += type == 5;
                board[std::countr_zero(b)] = PIECE_CHARS[side][type];
            }
            fen.clear();
            for (int rank = 7; rank >= 0 && valid; --rank) {
                int empty = 0;
                for (int file = 0; file < 8; ++file) {
                    const char c = board[rank * 8 + file];
                    if (!c) {
                        ++empty;
                        continue;
                    }
                    if (empty) fen += static_cast<char>('0' + empty);
                    empty = 0;
                    fen += c;
                }
                if (empty) fen += static_cast<char>('0' + empty);
                if (rank) fen += '/';
            }
            fen += " w - - 0 1";
            // A record that isn't a position keeps its score: the positions stay the source's.
            if (!valid || kings[0] != 1 || kings[1] != 1 || !pos.set_fen(fen)) {
                ++kept;
                continue;
            }
            std::int16_t old_score;
            std::memcpy(&old_score, r + 24, 2);
            const auto score = static_cast<std::int16_t>(std::clamp(nnue::evaluate(pos), -32767, 32767));
            std::memcpy(r + 24, &score, 2);
            if (old_score != 0 && score != 0) {
                ++signed_pairs;
                same_sign += (old_score > 0) == (score > 0);
            }
        }
        ok = std::fwrite(block.data(), 1, got, out) == got;
        written += got / REC;
    }
    params::nnue_scale = saved_scale;
    std::fclose(in);
    if (std::fclose(out) != 0) ok = false;
    if (!ok || written != size / REC)
        return fail("a read or a write failed (the disk full?) after " + std::to_string(written) + " of " +
                    std::to_string(size / REC) + " records");
    fs::rename(part, opt.out, ec);
    if (ec) return fail("cannot rename " + part + " to " + opt.out);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    log << "info string relabel: " << written << " records, " << kept << " kept as they were (not a position); "
        << "the new score's sign the old one's in " << std::fixed << std::setprecision(1)
        << (signed_pairs ? 100.0 * static_cast<double>(same_sign) / static_cast<double>(signed_pairs) : 0.0)
        << "%; NnueScale " << opt.scale << ", "
        << std::setprecision(0) << secs << " s, to " << opt.out << std::endl;
    return written;
}

std::uint64_t rescore_data(const RescoreOptions& opt, std::ostream& log) {
    if (std::filesystem::exists(opt.out)) {
        log << "info string rescore: " << opt.out << " exists: never overwritten" << std::endl;
        return 0;
    }
    std::ifstream in(opt.in, std::ios::binary);
    std::ofstream out(opt.out, std::ios::binary);
    if (!in || !out) {
        log << "info string rescore: cannot read " << opt.in << " or write " << opt.out << std::endl;
        return 0;
    }
    const bool two_labels = !opt.roll.empty() || opt.back_pass;  // the final label and the tip's (version 4)
    {
        const std::time_t now = std::time(nullptr);
        std::ofstream p(opt.out + ".params");
        p << "# rescore\nsource=" << opt.in << "\ngit=" << GIT_HASH << "\nnet=" << nnue::loaded_path()
          << "\nsoftnodes=" << opt.soft_nodes << "\nhardnodes=" << opt.hard_nodes << "\nsoftscale=" << opt.soft_scale
          << "\nhash=" << opt.hash_mb << "\nroll=";
        for (std::size_t k = 0; k < opt.roll.size(); ++k) p << (k ? "," : "") << opt.roll[k];
        p << "\nbackpass=" << opt.back_pass << "\ndate=" << std::put_time(std::gmtime(&now), "%Y-%m-%d %H:%M UTC")
          << "\nformat_version=" << (two_labels ? BACKWARD_VERSION : FORWARD_VERSION) << "\n# search parameters\n";
        for (const auto& t : params::tunables()) p << t.name << '=' << *t.value << '\n';
    }
    Searcher s(static_cast<std::size_t>(opt.hash_mb));
    std::uint64_t games = 0, positions = 0, stubs = 0;
    std::uint64_t tip_nodes = 0, roll_nodes = 0, roll_searches = 0, back_nodes = 0, back_searches = 0;
    // A later search of a record's position overwrites its label (the roll's and the backward pass's), adding its
    // nodes to the record's.
    auto relabel = [&](DataRecord& r, const std::string& fen, const std::vector<std::uint64_t>& keys,
                       std::size_t key_count, std::uint64_t soft) {
        Position q;
        q.set_fen(fen);
        const std::vector<std::uint64_t> history(keys.begin(), keys.begin() + static_cast<std::ptrdiff_t>(key_count));
        SearchLimits limits;
        limits.soft_nodes = soft;
        limits.nodes = 8 * soft;
        limits.soft_scale = opt.soft_scale;
        const Move b = s.search(q, limits, history, false);
        r.score = to_i16(s.last_score());
        r.best = b.data;
        r.depth = static_cast<std::uint8_t>(
            std::clamp(s.iterations().empty() ? 1 : s.iterations().back().depth, 1, 255));
        r.flags = static_cast<std::uint8_t>((r.flags & ~DF_MATE) |
                                            (std::abs(s.last_score()) > Searcher::MATE_BOUND ? DF_MATE : 0));
        r.nodes = static_cast<std::uint32_t>(std::min<std::uint64_t>(std::uint64_t{r.nodes} + s.nodes(), UINT32_MAX));
        return s.nodes();
    };
    auto keeps_tip = [](const DataRecord& r) { return (r.flags & (DF_IN_CHECK | DF_TB_WIN | DF_TB_LOSS)) != 0; };
    DataGame g;
    while ((!opt.games || games < opt.games) && read_game(in, g)) {
        g.version = two_labels ? BACKWARD_VERSION : FORWARD_VERSION;
        if (g.end == DataEnd::Filtered) {  // a stub: no game to replay
            write_game(out, g);
            ++games;
            ++stubs;
            continue;
        }
        Position pos;
        pos.set_fen(g.fen);
        std::vector<StateInfo> states(g.random_moves.size() + g.records.size() + 1);
        std::vector<std::uint64_t> keys;
        std::size_t made = 0;
        auto play = [&](Move m) {
            keys.push_back(pos.key());
            pos.make_move(m, states[made++]);
        };
        for (const Move m : g.random_moves) play(m);
        s.new_game();
        std::vector<std::string> fens;         // each record's position, for the roll and the backward pass
        std::vector<std::size_t> key_counts;  // and its history's length
        for (std::size_t i = 0; i < g.records.size(); ++i) {
            DataRecord& r = g.records[i];
            Move played;
            played.data = r.move;
            if (!is_legal_move(pos, played)) {
                log << "info string rescore: game " << g.worker << ':' << g.index << " doesn't replay at "
                    << pos.fen() << ": stopped" << std::endl;
                return games;
            }
            fens.push_back(pos.fen());
            key_counts.push_back(keys.size());
            Move best;
            int score = 0, tb = Searcher::NO_EVAL;
            DataRecord nr = search_record(s, pos, keys, opt.soft_nodes, opt.hard_nodes, best, score, tb,
                                          opt.soft_scale);
            nr.best = best.data;
            nr.move = r.move;
            tip_nodes += nr.nodes;
            if (two_labels) {  // the tip's own label, kept beside the final one
                nr.back_score = nr.score;
                nr.back_best = nr.best;
                nr.back_depth = nr.depth;
                nr.back_nodes = nr.nodes;
            }
            r = nr;
            ++positions;
            // The roll: the positions before this one, nearest first, each finding its successor fresh.
            for (std::size_t k = 1; k <= opt.roll.size() && k <= i; ++k) {
                DataRecord& earlier = g.records[i - k];
                if (keeps_tip(earlier)) continue;
                roll_nodes += relabel(earlier, fens[i - k], keys, key_counts[i - k], opt.roll[k - 1]);
                ++roll_searches;
            }
            play(played);
        }
        if (opt.back_pass) {  // gendata's backward pass, for comparison: last position first
            for (std::size_t i = g.records.size(); i-- > 0;) {
                if (keeps_tip(g.records[i])) continue;
                back_nodes += relabel(g.records[i], fens[i], keys, key_counts[i], opt.back_pass);
                ++back_searches;
            }
        }
        write_game(out, g);
        ++games;
        if (games % 1000 == 0)
            log << "info string rescore: " << games << " games, " << positions << " positions" << std::endl;
    }
    log << "info string rescore: done, " << games << " games (" << stubs << " filtered stubs copied), "
        << positions << " positions rescored; nodes: tips " << tip_nodes << ", roll " << roll_nodes << " in "
        << roll_searches << " searches, backward pass " << back_nodes << " in " << back_searches << " searches"
        << std::endl;
    return games;
}

unsigned game_slices(const DataGame& g) {
    const unsigned start = g.opening == 1   ? SLICE_BINGO
                           : g.opening == 2 ? (g.start_index == 0   ? SLICE_RESERVOIR_EARLY
                                               : g.start_index == 1 ? SLICE_RESERVOIR_LATE
                                                                    : SLICE_RESERVOIR_END)
                                            : SLICE_STANDARD;
    const bool adjudicated = g.end == DataEnd::AdjScore || g.end == DataEnd::AdjTablebase ||
                             g.end == DataEnd::AdjDraw || g.end == DataEnd::AdjMate;
    return start | (adjudicated ? SLICE_ADJUDICATED : SLICE_PLAYED);
}

unsigned slice_bits(const std::string& names) {
    static const std::map<std::string, unsigned> NAMES = {
        {"standard", SLICE_STANDARD}, {"bingo", SLICE_BINGO},
        {"reservoir", SLICE_RESERVOIR_EARLY | SLICE_RESERVOIR_LATE | SLICE_RESERVOIR_END},
        {"reservoir-early", SLICE_RESERVOIR_EARLY}, {"reservoir-late", SLICE_RESERVOIR_LATE},
        {"reservoir-end", SLICE_RESERVOIR_END}, {"adjudicated", SLICE_ADJUDICATED}, {"played", SLICE_PLAYED}};
    unsigned bits = 0;
    std::stringstream list(names);
    for (std::string name; std::getline(list, name, ',');) {
        const auto it = NAMES.find(name);
        if (it == NAMES.end()) throw std::invalid_argument("unknown slice " + name);
        bits |= it->second;
    }
    return bits;
}

std::uint64_t data_text(const std::string& in, std::ostream& out, const TextOptions& opt,
                        std::ostream& log, Searcher* searcher) {
    static constexpr const char* RESULT[3] = {"1.0", "0.5", "0.0"};  // by DataResult
    std::uint64_t written = 0, skipped = 0;
    // Why positions were left out, by the first filter each failed.
    std::uint64_t in_check = 0, decisive = 0, noisy_moves = 0, fifty = 0, thinned = 0, capped = 0, castle_window = 0;
    std::uint64_t qs_moved = 0;  // qs_quiet: the captures raised the static eval by more than the margin
    if (opt.qs_quiet >= 0 && !searcher) {
        log << "info string datatext: qsquiet needs the engine's searcher" << std::endl;
        return 0;
    }
    const std::vector<std::uint64_t> no_history;
    std::uint64_t no_backward = 0;  // with the backward label: records not searched backward
    std::uint64_t td_backed = 0, td_reach = 0;  // backed-up labels: those with a chain, the plies they reached
    std::uint64_t draws_out = 0, games_out = 0, hard_extra = 0, sliced_out = 0, other_part = 0, unindexed = 0;
    const std::map<std::string, Cut> cuts = opt.part ? read_cuts(opt.index) : std::map<std::string, Cut>{};
    struct Kept {
        std::uint64_t key;  // for the cap
        std::string line;
        int copies;
    };
    std::vector<Kept> kept;  // one game's lines
    // A game's reproducible draw in [0, 100), for the game-level selections.
    const auto game_draw = [](const DataGame& g, std::uint64_t salt) {
        std::uint64_t h = g.seed ^ (std::uint64_t{g.worker} << 48) ^ (std::uint64_t{g.index} << 16) ^ salt;
        return static_cast<int>(splitmix64(h) % 100);
    };
    for (const std::string& f : data_files(in)) {
        std::uint64_t first = 0;  // with a part: where the file's second part begins
        if (opt.part) {
            const auto c = cuts.find(std::filesystem::path(f).stem().string());
            if (c == cuts.end()) {
                log << "info string datatext: " << f << " has no line in " << opt.index << ": skipped" << std::endl;
                ++unindexed;
                continue;
            }
            first = c->second.index;
        }
        std::ifstream file(f, std::ios::binary);
        DataGame g;
        while (read_game(file, g)) {
            // No result, or a corrupt byte (RESULT[] has three entries).
            if (g.end == DataEnd::Filtered || static_cast<int>(g.result) > 2) continue;
            if (opt.part && (g.index >= first) != (opt.part == 'b')) {
                ++other_part;
                continue;
            }
            if (game_slices(g) & opt.drop) {
                ++sliced_out;
                continue;
            }
            if (opt.game_keep_pct < 100 && game_draw(g, 0xA0761D6478BD642FULL) >= opt.game_keep_pct) {
                ++games_out;
                continue;
            }
            if (opt.draw_keep_pct < 100 && g.result == DataResult::Draw &&
                game_draw(g, 0xE7037ED1A0B428DBULL) >= opt.draw_keep_pct) {
                ++draws_out;
                continue;
            }
            kept.clear();
            std::uint64_t index = 0;
            Position pos;
            pos.set_fen(g.fen);
            std::vector<StateInfo> states(g.random_moves.size() + g.records.size() + 1);
            std::size_t made = 0;
            for (const Move m : g.random_moves) pos.make_move(m, states[made++]);
            // The backed-up label's inputs (TextOptions::td_lambda): every record's score as the export
            // reads it (unscaled for the fifty-move rule at its own position) and whether it may join a chain.
            std::vector<double> td_score;
            std::vector<char> td_ok;
            if (opt.td_lambda > 0 && !opt.backward) {
                Position q;
                q.set_fen(g.fen);
                std::vector<StateInfo> qstates(g.random_moves.size() + g.records.size() + 1);
                std::size_t qmade = 0;
                for (const Move m : g.random_moves) q.make_move(m, qstates[qmade++]);
                const int fs = opt.fifty_scale ? opt.fifty_scale : params::fifty_scale;
                for (const DataRecord& r : g.records) {
                    long long s = r.score;
                    bool ok = !(r.flags & DF_MATE) && std::abs(s) < opt.td_cap;
                    if (opt.unscale && fs && q.halfmove_clock() > 0) {
                        const int left = fs - q.halfmove_clock();
                        if (2 * left >= fs)
                            s = s * fs / left;
                        else
                            ok = false;
                    }
                    td_score.push_back(static_cast<double>(s));
                    td_ok.push_back(ok);
                    Move m;
                    m.data = r.move;
                    if (!is_legal_move(q, m)) break;
                    q.make_move(m, qstates[qmade++]);
                }
            }
            // Record ri's backed-up label from its own unscaled score.
            const auto td_label = [&](std::size_t ri) {
                double sum = td_score[ri], weights = 1, w = 1;
                int reach = 0;
                for (std::size_t j = 1; j <= static_cast<std::size_t>(opt.td_plies) && ri + j < td_score.size(); ++j) {
                    const DataRecord& before = g.records[ri + j - 1];
                    if ((before.best && before.best != before.move) || !td_ok[ri + j]) break;
                    w *= opt.td_lambda;
                    sum += w * (j % 2 ? -td_score[ri + j] : td_score[ri + j]);
                    weights += w;
                    ++reach;
                }
                td_reach += static_cast<std::uint64_t>(reach);
                td_backed += reach > 0;
                return sum / weights;
            };
            for (std::size_t ri = 0; ri < g.records.size(); ++ri) {
                const DataRecord& r = g.records[ri];
                Move m;
                m.data = r.move;
                // The search's best move: a rescored record keeps the move played
                // apart (version 2); otherwise they are the same move. With the
                // backward label (TextOptions::backward), that search's move.
                Move bm;
                bm.data = opt.backward ? r.back_best : r.best ? r.best : r.move;
                const bool noisy = pos.piece_at(bm.to()) != NO_PIECE || bm.is_en_passant() ||
                                   bm.promotion() != PieceType::None;
                long long score = opt.backward ? r.back_score : r.score;
                bool keep = true;
                if (opt.castle_clean && static_cast<int>(made) < opt.castle_clean &&
                    (pos.castling_rights() & (Position::WQ_CASTLE | Position::BQ_CASTLE))) {
                    keep = false;
                    ++castle_window;
                } else if (r.flags & DF_IN_CHECK) {
                    keep = false;
                    ++in_check;
                } else if (opt.backward && r.back_depth == 0) {
                    keep = false;  // not searched backward: a tablebase root won or lost, or version 3
                    ++no_backward;
                } else if ((!opt.backward && (r.flags & DF_MATE)) || std::abs(score) >= Searcher::DECISIVE_BOUND) {
                    keep = false;
                    ++decisive;
                } else if (opt.quiet && noisy) {
                    keep = false;
                    ++noisy_moves;
                } else if (opt.qs_quiet >= 0) {
                    // What the side to move's captures alone make of the static eval (no hash, no search).
                    const int stand = searcher->root_eval(pos).static_eval;
                    if (searcher->quiescence(pos, no_history, true) - stand > opt.qs_quiet) {
                        keep = false;
                        ++qs_moved;
                    }
                }
                const int fs = opt.fifty_scale ? opt.fifty_scale : params::fifty_scale;
                const int left = fs - pos.halfmove_clock();
                if (keep && opt.unscale && fs && pos.halfmove_clock() > 0) {
                    if (2 * left >= fs) {
                        score = score * fs / left;
                    } else {
                        keep = false;
                        ++fifty;
                    }
                }
                if (keep && ri < td_score.size()) score = std::llround(td_label(ri));  // unscaled already, as `score`
                if (keep && opt.rescale != 1) score = std::llround(static_cast<double>(score) * opt.rescale);
                if (keep) {
                    // Thinning: the highest threshold reached sets the rate.
                    int rate = 1;
                    for (const TextOptions::Thin& t : opt.thin)
                        if (std::abs(score) >= t.threshold) rate = t.rate;
                    if (rate > 1) {
                        std::uint64_t h = g.seed ^ (std::uint64_t{g.worker} << 48) ^
                                          (std::uint64_t{g.index} << 16) ^ index;
                        if (splitmix64(h) % static_cast<std::uint64_t>(rate) != 0) {
                            keep = false;
                            ++thinned;
                        }
                    }
                }
                if (keep) {
                    // Hard examples (scores from the side to move).
                    int copies = 1;
                    const int res = static_cast<int>(g.result);  // 0 White won, 1 draw, 2 Black won
                    const bool stm_won = (res == 0) == (pos.side_to_move() == Color::White) && res != 1;
                    const bool stm_lost = res != 1 && !stm_won;
                    if (opt.hard_score && ((score >= opt.hard_score && !stm_won) || (score <= -opt.hard_score && !stm_lost)))
                        copies = opt.hard_copies;
                    if (opt.hard_swing && ri + 2 < g.records.size() &&
                        std::abs(g.records[ri + 2].score) < Searcher::DECISIVE_BOUND &&
                        std::abs(g.records[ri + 2].score - r.score) >= opt.hard_swing)
                        copies = opt.hard_copies;
                    if (pos.side_to_move() == Color::Black) score = -score;
                    if (opt.blend >= 0) {
                        const double outcome = g.result == DataResult::WhiteWin ? 1.0 : g.result == DataResult::Draw ? 0.5 : 0.0;
                        const double p = std::clamp((1 - opt.blend) / (1 + std::exp(-static_cast<double>(score) / opt.blend_scale)) +
                                                        opt.blend * outcome,
                                                    1e-6, 1 - 1e-6);
                        score = std::llround(opt.blend_scale * std::log(p / (1 - p)));
                    }
                    std::ostringstream line;
                    line << pos.fen() << " | " << score << " | " << RESULT[static_cast<int>(g.result)] << '\n';
                    // The cap's key: a second hash of the position's place.
                    std::uint64_t h = (g.seed ^ (std::uint64_t{g.worker} << 48) ^ (std::uint64_t{g.index} << 16) ^
                                       index) + 0x5851F42D4C957F2DULL;
                    kept.push_back({splitmix64(h), line.str(), copies});
                } else {
                    ++skipped;
                }
                if (!is_legal_move(pos, m)) break;  // a corrupt file: stop this game cleanly
                pos.make_move(m, states[made++]);
                ++index;
            }
            // The cap: the `cap` smallest keys stay, in game order.
            std::uint64_t limit = ~std::uint64_t{0};
            if (opt.cap > 0 && kept.size() > static_cast<std::size_t>(opt.cap)) {
                std::vector<std::uint64_t> keys;
                for (const Kept& k : kept) keys.push_back(k.key);
                std::nth_element(keys.begin(), keys.begin() + (opt.cap - 1), keys.end());
                limit = keys[static_cast<std::size_t>(opt.cap - 1)];
            }
            for (const Kept& k : kept) {
                if (k.key > limit) {
                    ++capped;
                    ++skipped;
                    continue;
                }
                for (int c = 0; c < k.copies; ++c) out << k.line;
                ++written;
                hard_extra += static_cast<std::uint64_t>(k.copies - 1);
            }
        }
    }
    log << "info string datatext: wrote " << written << " positions, left out " << skipped
        << " (in check " << in_check << ", mate or tablebase score " << decisive
        << ", noisy best move " << noisy_moves << ", fifty-move clock " << fifty << ", thinned "
        << thinned << ", capped " << capped << "); drawn games left out " << draws_out
        << ", games left out " << games_out << ", extra copies of hard examples " << hard_extra
        << ", castling window " << castle_window << ", games of dropped slices " << sliced_out
        << ", games of the other part " << other_part << ", files without an index line " << unindexed;
    if (opt.backward) log << "; the backward label: records without one " << no_backward;
    if (opt.td_lambda > 0 && !opt.backward)
        log << "; backed up (lambda " << opt.td_lambda << ", up to " << opt.td_plies << " plies): " << td_backed
            << " labels, " << std::fixed << std::setprecision(2)
            << (td_backed ? static_cast<double>(td_reach) / static_cast<double>(td_backed) : 0.0)
            << " plies on average" << std::defaultfloat;
    if (opt.qs_quiet >= 0) log << "; not quiet by quiescence (margin " << opt.qs_quiet << ") " << qs_moved;
    log << std::endl;
    return written;
}
