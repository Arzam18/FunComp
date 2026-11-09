#include "uci.h"
#include "bench.h"
#include "bingo.h"
#include "reservoir.h"
#include "cont_log.h"
#include "datagen.h"
#include "featstats.h"
#include "eval.h"
#include "initiative.h"
#include "labels.h"
#include "position.h"
#include "movegen.h"
#include "nnue.h"
#include "pair_features.h"
#include "square_features.h"
#include "perft.h"
#include "search.h"
#include "params.h"
#include "tablebase.h"
#include "telemetry.h"
#include "threads.h"
#include "tree_sample.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
void print_board(const Position& p) {
    for(int r=7;r>=0;--r) {
        std::cout << r+1 << " ";
        for(int f=0;f<8;++f) {
            Piece pc=p.piece_at(sq(f,r));
            static constexpr char chars[]="PNBRQKpnbrqk";
            std::cout << (pc==NO_PIECE ? '.' : chars[pc]) << ' ';
        }
        std::cout << '\n';
    }
    std::cout << "  a b c d e f g h\n";
    std::cout << "FEN: " << p.fen() << '\n';
}

bool apply_uci_move(Position& pos, const std::string& uci, std::vector<std::uint64_t>& history) {
    MoveList moves;
    int count=0;
    generate_legal(pos,moves,count);
    for(int i=0;i<count;++i) {
        if(pos.move_to_uci(moves[i])==uci) {
            StateInfo st;
            history.push_back(pos.key());
            pos.make_move(moves[i],st);
            return true;
        }
    }
    return false;
}

// Match telemetry (TelemetryDir): this process's counters over all its
// searches, in a file of its own (many engine processes write at once),
// summed across files by tools/telemetry-sum.sh.
void write_telemetry(const std::string& dir, const prune::Stats& stats) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::string name = "telemetry-" + std::to_string(stamp) + "-" +
                             std::to_string(std::random_device{}()) + ".txt";
    std::ofstream out(std::filesystem::path(dir) / name);
    stats.write(out);
}
}

void uci_loop() {
    if (nnue::embedded()) nnue::load("<embedded>");  // the binary's own network, EvalFile's default
    Position pos;
    ThreadPool pool;  // the searchers, one per thread
    std::string telemetry_dir;  // TelemetryDir: empty, no match telemetry
    std::string syzygy_path;    // SyzygyPath as last set, for data records
    std::vector<std::uint64_t> history;  // keys of the game's earlier positions
    std::string line;

    // The search runs on its own thread, so that "stop" and "isready" are
    // answered while it thinks. Any other command first stops a running
    // search and waits for its bestmove (a GUI only sends them between
    // searches): so the search thread has the position, the history and
    // the searchers to themselves, and "go infinite" can't be left waiting for
    // a "stop" that sits behind the command being handled.
    Worker search_thread;
    auto finish_search=[&] {
        if(search_thread.busy()) {
            pool.request_stop();
            search_thread.wait();
            pool.clear_stop();  // or it would cut short a later "bench"
        }
    };

    while(std::getline(std::cin,line)) try {
        // Some Windows tools (PowerShell's pipe, .NET process streams) start
        // their input with a UTF-8 byte-order mark; it would hide the command.
        if(line.starts_with("\xEF\xBB\xBF")) line.erase(0,3);
        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;

        if(cmd=="isready") {
            std::cout << "readyok\n" << std::flush;  // one write: a search may be printing
            continue;
        }
        // The opponent played the move we were pondering on: the search goes
        // on, now on the clock.
        if(cmd=="ponderhit") { pool.ponderhit(); continue; }
        finish_search();  // "stop" is exactly this
        if(cmd=="quit") break;

        if(cmd=="uci") {
            std::cout << "id name FunComp 3.1\n";
            std::cout << "id author Moxnar\n";
            std::cout << "option name Hash type spin default 64 min 1 max 4096\n";
            std::cout << "option name Threads type spin default 1 min 1 max " << MAX_THREADS << '\n';
            // Only tells the GUI it may send "go ponder"; nothing to set.
            std::cout << "option name Ponder type check default false\n";
            std::cout << "option name Move Overhead type spin default " << params::move_overhead
                      << " min 0 max 5000\n";
            std::cout << "option name SyzygyPath type string default <empty>\n";
            // The network compiled into the binary when there is one (3.0 on).
            std::cout << "option name EvalFile type string default "
                      << (nnue::embedded() ? "<embedded>" : "<empty>") << '\n';
            // Counters over the whole session, written at exit (for matches).
            std::cout << "option name TelemetryDir type string default <empty>\n";
            std::cout << "option name SyzygyProbeLimit type spin default " << tb::probe_limit
                      << " min 0 max 7\n";
            std::cout << "option name SyzygyProbeDepth type spin default " << tb::probe_depth
                      << " min 1 max 100\n";
#ifdef TUNE
            // Diagnostics and tuning only; still settable in any build.
            std::cout << "option name PruneVerify type spin default 0 min 0 max 1000000\n";
            for(const auto& t : params::tunables())
                std::cout << "option name " << t.name << " type spin default " << *t.value
                          << " min " << t.min << " max " << t.max << '\n';
#endif
            std::cout << "uciok" << std::endl;
        } else if(cmd=="setoption") {
            // setoption name <id> [value <x>]; the id may contain spaces
            // ("Move Overhead"), so it runs up to "value".
            std::string tok, name, value;
            std::string* field=nullptr;
            while(ss>>tok) {
                if(tok=="name") field=&name;
                else if(tok=="value") field=&value;
                else if(field) {
                    if(!field->empty()) *field += ' ';
                    *field += tok;
                }
            }
            if(name=="Move Overhead" && !value.empty()) {
                params::move_overhead=std::clamp(std::stoi(value),0,5000);
            } else if(name=="Threads" && !value.empty()) {
                pool.set_threads(std::clamp(std::stoi(value),1,MAX_THREADS));
            } else if(name=="Hash" && !value.empty()) {
                const long mb=std::stol(value);
                pool.resize_hash(static_cast<std::size_t>(std::clamp(mb,1L,4096L)));
            } else if(name=="TelemetryDir") {
                telemetry_dir = value=="<empty>" ? std::string() : value;
            } else if(name=="EvalFile") {
                // A network replaces the hand-written evaluation (nnue.h). The
                // hash table keeps raw evals, so it starts afresh.
                finish_search();
                if(nnue::load(value))
                    std::cout << "info string network loaded: " << value << ", "
                              << nnue::hidden() << " hidden, " << nnue::king_buckets()
                              << " king buckets, " << nnue::output_buckets() << " output buckets"
                              << (nnue::pawn_features() ? ", pawn predicates" : "")
                              << (nnue::pawn_span_features() ? ", pawn-control map" : "")
                              << (nnue::passer_features() ? ", passed-pawn quality" : "")
                              << (nnue::king_state_buckets() ? ", king-state buckets" : "")
                              << (nnue::king_location_buckets() ? ", bare king-state buckets" : "")
                              << (nnue::king_state_output_buckets() ? ", output by king state" : "")
                              << (nnue::endgame_output_buckets() ? ", output by endgame type" : "")
                              << (nnue::pair_families() & pairfeat::KnightPairs ? ", knight pairs" : "")
                              << (nnue::pair_families() & pairfeat::SliderPairs ? ", slider pairs" : "")
                              << (nnue::pair_families() & pairfeat::SliderPawn ? ", slider x pawn" : "")
                              << (nnue::pair_families() & pairfeat::ConnectedPairs ? ", connected pairs" : "")
                              << (nnue::pair_families() & pairfeat::LineControl ? ", line control" : "")
                              << (nnue::square_families() & sqfeat::Attacked ? ", attacked squares" : "")
                              << (nnue::square_families() & sqfeat::LeastAttacker ? ", least attackers" : "")
                              << (nnue::square_families() & sqfeat::KingZone ? ", king zone" : "")
                              << (nnue::square_families() & sqfeat::PawnPieces ? ", pawn-piece predicates" : "")
                              << (nnue::square_families() & sqfeat::ControlCounts ? ", control counts" : "")
                              << (nnue::square_families() & sqfeat::Mobility ? ", mobility" : "")
                              << "; " << nnue::evaluation_path() << '\n'
                              << std::flush;
                else if(!value.empty() && value!="<empty>")
                    std::cout << "info string no network in " << value
                              << ", using the hand-written evaluation\n" << std::flush;
                pool.new_game();
            } else if(name=="SyzygyPath") {
                syzygy_path = value=="<empty>" ? std::string() : value;
                const int pieces=tb::init(value);
                if(pieces)
                    std::cout << "info string Syzygy tablebases found, up to " << pieces
                              << " pieces\n" << std::flush;
                else if(!value.empty() && value!="<empty>")
                    std::cout << "info string no Syzygy tablebases found in " << value
                              << '\n' << std::flush;
            } else if(name=="SyzygyProbeLimit" && !value.empty()) {
                tb::probe_limit=std::clamp(std::stoi(value),0,7);
            } else if(name=="SyzygyProbeDepth" && !value.empty()) {
                tb::probe_depth=std::clamp(std::stoi(value),1,100);
            } else if(name=="PruneVerify" && !value.empty()) {
                // Check 1 in N pruning decisions by searching anyway; see
                // "pstats". Diagnostic only: slows and alters the search.
                pool.main().set_verify_rate(std::stoi(value));
            } else if(!value.empty()) {
                // Search parameters are settable even when a normal build
                // doesn't advertise them, except where a frozen build has
                // them as constants (params.h). Asked for another value, it
                // stops rather than play on unnoticed with the wrong one: a
                // TUNE build sets them. The value it has is accepted.
                for(const auto& t : params::tunables()) {
                    if(name!=t.name) continue;
                    const int v=std::clamp(std::stoi(value),t.min,t.max);
                    if(t.set) {
                        *t.set=v;
                    } else if(v!=*t.value) {
                        const std::string error="info string error: " + name + " is fixed at " +
                            std::to_string(*t.value) + " in this build, asked for " + value +
                            " (a TUNE build sets it)";
                        std::cout << error << std::endl;
                        std::cerr << error << std::endl;
                        std::exit(1);
                    }
                }
            }
        } else if(cmd=="ucinewgame") {
            pos.set_startpos();
            history.clear();
            pool.new_game();
        } else if(cmd=="position") {
            std::string arg;
            ss >> arg;
            history.clear();
            if(arg=="startpos") {
                pos.set_startpos();
            }
            // A FEN runs up to "moves" (its clock fields are optional: GUIs
            // and EPD books often leave them out). An invalid one leaves the
            // position unchanged, and its moves are ignored rather than
            // played on the wrong position.
            std::string token;
            bool valid=true;
            if(arg=="fen") {
                std::string fen;
                while(ss>>token && token!="moves") {
                    if(!fen.empty()) fen += ' ';
                    fen += token;
                }
                valid = pos.set_fen(fen);
                if(!valid) std::cerr << "info string invalid FEN\n";
            } else {
                ss>>token;
            }

            if(valid && token=="moves") {
                while(ss>>token) {
                    if(!apply_uci_move(pos,token,history))
                        std::cerr << "info string illegal move " << token << "\n";
                }
            }
        } else if(cmd=="go") {
            SearchLimits limits;
            std::string arg;
            bool listing=false;  // after "searchmoves": moves until the next keyword
            MoveList legal;
            int legal_count=0;
            generate_legal(pos,legal,legal_count);
            while(ss>>arg) {
                if(listing) {
                    bool found=false;
                    for(int i=0;i<legal_count && !found;++i)
                        if(pos.move_to_uci(legal[i])==arg) { limits.search_moves.push_back(legal[i]); found=true; }
                    if(found) continue;
                    listing=false;  // not a legal move here: a keyword, read as one below
                }
                if(arg=="searchmoves") listing=true;
                else if(arg=="depth") ss>>limits.depth;
                else if(arg=="movetime") ss>>limits.movetime;
                // A clock that has run below zero (some GUIs send that) is
                // no time left, not "no clock": -1 means no clock.
                else if(arg=="wtime") { ss>>limits.time[0]; limits.time[0]=std::max(limits.time[0],0); }
                else if(arg=="btime") { ss>>limits.time[1]; limits.time[1]=std::max(limits.time[1],0); }
                else if(arg=="winc") { ss>>limits.inc[0]; limits.inc[0]=std::max(limits.inc[0],0); }
                else if(arg=="binc") { ss>>limits.inc[1]; limits.inc[1]=std::max(limits.inc[1],0); }
                else if(arg=="movestogo") ss>>limits.movestogo;
                // Data generation's limits, for probes (tools/research/depth_probe.py).
                else if(arg=="softnodes") ss>>limits.soft_nodes;
                else if(arg=="softscale") { int v=0; ss>>v; limits.soft_scale=v!=0; }
                else if(arg=="mindepth") ss>>limits.min_depth;
                else if(arg=="nodes") ss>>limits.nodes;
                else if(arg=="infinite") limits.infinite=true;
                else if(arg=="ponder") limits.ponder=true;
            }
            pool.main().apply_frame_limits(pos,limits);  // frame L2's data-generation limits, when set
            // Bare "go" with no limits searches until "stop", like "go infinite".
            const bool has_clock = limits.time[static_cast<int>(pos.side_to_move())] >= 0;
            if(!limits.depth && !limits.movetime && !limits.nodes && !has_clock) limits.infinite=true;
            pool.clear_stop();
            pool.clear_ponderhit();
            search_thread.start([&pool,&pos,&history,limits] {
                const Move best=pool.search(pos,limits,history);
                // With "go infinite" the GUI decides when the search is over: no
                // bestmove before "stop", even if the search ran out first (a mate
                // found, or the maximum depth). Nor while pondering, before the
                // ponderhit (a ponderhit after a finished search answers at once).
                auto waiting=[&] {
                    if(pool.stop_requested()) return false;
                    return limits.infinite || (limits.ponder && !pool.ponderhit_received());
                };
                while(waiting()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                std::string out="bestmove "+(best==Move{} ? std::string("0000") : pos.move_to_uci(best));
                const Move reply=pool.main().ponder_move(pos,best);
                if(reply!=Move{}) {
                    // The reply is a move in the position after `best`.
                    StateInfo st;
                    pos.make_move(best,st);
                    out+=" ponder "+pos.move_to_uci(reply);
                    pos.unmake_move(best,st);
                }
                std::cout << out+"\n" << std::flush;
                // Only now: the move mustn't wait on helpers still stopping.
                // Every later command waits for this job, so for them too.
                pool.wait_helpers();
            });
        } else if(cmd=="legalmoves") {
            // "legalmoves": the position's legal moves in UCI notation, a
            // capture marked x, a check + (probes: tools/research/pair_gate.py).
            MoveList moves;
            int count=0;
            generate_legal(pos,moves,count);
            std::string out="legalmoves";
            for(int i=0;i<count;++i) {
                const Move m=moves[static_cast<std::size_t>(i)];
                out+=' '+pos.move_to_uci(m);
                if(pos.piece_at(m.to())!=NO_PIECE || m.is_en_passant()) out+='x';
                StateInfo st;
                pos.make_move(m,st);
                if(pos.in_check(pos.side_to_move())) out+='+';
                pos.unmake_move(m,st);
            }
            std::cout << out << std::endl;
        } else if(cmd=="perft") {
            // "perft <depth>" uses the hash table; "perft <depth> nohash" doesn't.
            static std::unique_ptr<PerftTable> tt;
            int depth=1;
            ss>>depth;
            std::string opt;
            const bool hashed = !(ss>>opt && opt=="nohash");
            if(hashed && !tt) tt=std::make_unique<PerftTable>(256);
            const auto start=std::chrono::steady_clock::now();
            std::uint64_t nodes;
            if(hashed) {
                nodes=perft(pos,depth,*tt);
            } else {
                nodes=perft(pos,depth);
            }
            const double secs=std::chrono::duration<double>(
                std::chrono::steady_clock::now()-start).count();
            std::cout << "nodes " << nodes << " time " << static_cast<long long>(secs*1000)
                      << "ms nps " << static_cast<long long>(secs>0 ? static_cast<double>(nodes)/secs : 0)
                      << '\n';
        } else if(cmd=="d") {
            print_board(pos);
        } else if(cmd=="nnueinputs") {
            // Debugging: the loaded network's active inputs for the current
            // position, side to move's view first (see nnue.h).
            if(!nnue::active()) {
                std::cout << "info string nnueinputs needs a network (EvalFile)" << std::endl;
            } else {
                std::vector<int> stm, ntm;
                nnue::active_inputs(pos, stm, ntm);
                std::cout << "stm";
                for(const int i : stm) std::cout << ' ' << i;
                std::cout << "\nntm";
                for(const int i : ntm) std::cout << ' ' << i;
                std::cout << "\nbucket " << nnue::output_bucket(pos) << std::endl;
            }
        } else if(cmd=="eval") {
            // Static evaluation with the initiative term broken down (not
            // corrected, not fifty-move scaled).
            initiative::print_trace(pos, std::cout);
            std::cout << "info string static eval " << evaluate(pos) << " cp (side to move)"
                      << std::endl;
        } else if(cmd=="bench") {
            // "bench [depth]"
            int depth=BENCH_DEPTH;
            ss>>depth;
            bench(pool.main(),depth,std::cout);
        } else if(cmd=="pstats") {
            // "pstats" prints pruning statistics accumulated since the last
            // "pstats reset"; "pstats reset" clears them.
            std::string arg;
            if(ss>>arg && arg=="reset") pool.main().clear_stats();
            else pool.main().stats().print(std::cout);
        } else if(cmd=="telemetry") {
            // "telemetry in=<pgn> [positions=N] [nodes=N] [verify=N]
            //  [minply=N] [seed=N]": counters on positions from a PGN's
            // games; see telemetry.h.
            TelemetryOptions opt;
            std::string tok;
            while(ss>>tok) {
                const auto eq=tok.find('=');
                if(eq==std::string::npos) continue;
                const std::string key=tok.substr(0,eq), val=tok.substr(eq+1);
                if(key=="in") opt.in=val;
                else if(key=="positions") opt.positions=std::max(1,std::stoi(val));
                else if(key=="nodes") opt.nodes=std::stoull(val);
                else if(key=="verify") opt.verify=std::max(0,std::stoi(val));
                else if(key=="minply") opt.min_ply=std::max(0,std::stoi(val));
                else if(key=="seed") opt.seed=std::stoull(val);
            }
            if(opt.in.empty()) std::cout << "info string telemetry needs in=<pgn>" << std::endl;
            else game_telemetry(pool.main(),opt,std::cout);
        } else if(cmd=="treesample") {
            // "treesample in=<pgn>|fens=<file> out=<file> [positions=N] [nodes=N]
            //  [rate=N] [minply=N] [seed=N]": the positions where the search
            // reads its static eval, on positions from a PGN's games or a list;
            // see tree_sample.h.
            TreeSampleOptions opt;
            std::string tok;
            while(ss>>tok) {
                const auto eq=tok.find('=');
                if(eq==std::string::npos) continue;
                const std::string key=tok.substr(0,eq), val=tok.substr(eq+1);
                if(key=="in") opt.in=val;
                else if(key=="fens") opt.fens=val;
                else if(key=="out") opt.out=val;
                else if(key=="positions") opt.positions=std::max(1,std::stoi(val));
                else if(key=="nodes") opt.nodes=std::stoull(val);
                else if(key=="rate") opt.rate=std::max(1,std::stoi(val));
                else if(key=="minply") opt.min_ply=std::max(0,std::stoi(val));
                else if(key=="seed") opt.seed=std::stoull(val);
            }
            if((opt.in.empty() && opt.fens.empty()) || opt.out.empty())
                std::cout << "info string treesample needs in=<pgn> or fens=<file>, and out=<file>" << std::endl;
            else tree_sample(pool.main(),opt,std::cout);
        } else if(cmd=="evalfens") {
            // "evalfens in=<file> out=<file> [nodes=N] [skip=N] [count=N]": the
            // raw eval, a quiescence search and a search to N nodes for each
            // line's position; see tree_sample.h.
            EvalFensOptions opt;
            std::string tok;
            while(ss>>tok) {
                const auto eq=tok.find('=');
                if(eq==std::string::npos) continue;
                const std::string key=tok.substr(0,eq), val=tok.substr(eq+1);
                if(key=="in") opt.in=val;
                else if(key=="out") opt.out=val;
                else if(key=="nodes") opt.nodes=std::stoull(val);
                else if(key=="skip") opt.skip=std::stoull(val);
                else if(key=="count") opt.count=std::stoull(val);
            }
            if(opt.in.empty() || opt.out.empty()) std::cout << "info string evalfens needs in=<file> and out=<file>" << std::endl;
            else eval_fens(pool.main(),opt,std::cout);
        } else if(cmd=="labelfens") {
            // "labelfens in=<file> out=<text> [nodes=N] [skip=N] [count=N]": each
            // line's position labelled by a search to N nodes, written as the
            // trainer's text; see tree_sample.h.
            LabelFensOptions opt;
            std::string tok;
            while(ss>>tok) {
                const auto eq=tok.find('=');
                if(eq==std::string::npos) continue;
                const std::string key=tok.substr(0,eq), val=tok.substr(eq+1);
                if(key=="in") opt.in=val;
                else if(key=="out") opt.out=val;
                else if(key=="nodes") opt.nodes=std::stoull(val);
                else if(key=="skip") opt.skip=std::stoull(val);
                else if(key=="count") opt.count=std::stoull(val);
            }
            if(opt.in.empty() || opt.out.empty()) std::cout << "info string labelfens needs in=<file> and out=<text>" << std::endl;
            else label_fens(pool.main(),opt,std::cout);
        } else if(cmd=="perturb") {
            // "perturb in=<file> out=<file> [plies=N] [seed=N]": each line's
            // position moved N random legal moves away; see tree_sample.h.
            PerturbOptions opt;
            std::string tok;
            while(ss>>tok) {
                const auto eq=tok.find('=');
                if(eq==std::string::npos) continue;
                const std::string key=tok.substr(0,eq), val=tok.substr(eq+1);
                if(key=="in") opt.in=val;
                else if(key=="out") opt.out=val;
                else if(key=="plies") opt.plies=std::max(1,std::stoi(val));
                else if(key=="seed") opt.seed=std::stoull(val);
            }
            if(opt.in.empty() || opt.out.empty()) std::cout << "info string perturb needs in=<file> and out=<file>" << std::endl;
            else perturb_fens(opt,std::cout);
        } else if(cmd=="genlabels") {
            // "genlabels out=<file> [in=<file>] [games=N] [randomplies=N]
            //  [depth=N] [sample=N] [depthscale=PCT] [maxdepth=N] [count=N]
            //  [seed=N] [hash=MB] [labelhash=MB] [labelnodes=N] [nullprobe=0|1]"
            // Writes pruning labels; see labels.h.
            LabelOptions opt;
            std::string tok;
            while(ss>>tok) {
                const auto eq=tok.find('=');
                if(eq==std::string::npos) continue;
                const std::string key=tok.substr(0,eq), val=tok.substr(eq+1);
                if(key=="out") opt.out=val;
                else if(key=="in") opt.in=val;
                else if(key=="games") opt.games=std::stoi(val);
                else if(key=="randomplies") opt.random_plies=std::stoi(val);
                else if(key=="depth") opt.depth=std::stoi(val);
                else if(key=="sample") opt.sample_rate=std::stoi(val);
                else if(key=="depthscale") opt.depth_scale=std::stoi(val);
                else if(key=="maxdepth") opt.max_label_depth=std::clamp(std::stoi(val),1,Searcher::MAX_PLY);
                else if(key=="count") opt.count=std::stoull(val);
                else if(key=="seed") opt.seed=std::stoull(val);
                else if(key=="hash") opt.hash_mb=std::clamp(std::stoi(val),1,4096);
                else if(key=="labelhash") opt.label_hash_mb=std::clamp(std::stoi(val),1,4096);
                else if(key=="labelnodes") opt.label_node_cap=std::stoull(val);
                else if(key=="nullprobe") opt.null_probe=val!="0";
            }
            if(opt.out.empty()) std::cout << "info string genlabels needs out=<file>" << std::endl;
            else generate_labels(opt,std::cout);
        } else if(cmd=="genbingo") {
            // "genbingo out=<file> [plies=N] [perstart=N] [nodes=N]
            //  [maxscore=CP] [seed=N] [hash=MB]": Chess324 openings;
            // see bingo.h.
            BingoOptions opt;
            std::string tok;
            while(ss>>tok) {
                const auto eq=tok.find('=');
                if(eq==std::string::npos) continue;
                const std::string key=tok.substr(0,eq), val=tok.substr(eq+1);
                if(key=="out") opt.out=val;
                else if(key=="plies") opt.plies=std::max(0,std::stoi(val));
                else if(key=="perstart") opt.per_start=std::max(1,std::stoi(val));
                else if(key=="nodes") opt.nodes=std::stoull(val);
                else if(key=="maxscore") opt.max_score=std::stoi(val);
                else if(key=="seed") opt.seed=std::stoull(val);
                else if(key=="hash") opt.hash_mb=std::clamp(std::stoi(val),1,4096);
            }
            if(opt.out.empty()) std::cout << "info string genbingo needs out=<file>" << std::endl;
            else generate_bingo(opt,std::cout);
        } else if(cmd=="reservoir") {
            // "reservoir in=<pgn> [out=<prefix>] [count=1] [minelo=N]
            //  [minseconds=N] [minmove=N] [maxmove=N] [games=N] [endnodes=N]
            //  [endmin=CP] [endmax=CP] [seed=N] [hash=MB]": round 3's start
            // positions from Lichess games; see reservoir.h.
            ReservoirOptions opt;
            std::string tok;
            while(ss>>tok) {
                const auto eq=tok.find('=');
                if(eq==std::string::npos) continue;
                const std::string key=tok.substr(0,eq), val=tok.substr(eq+1);
                if(key=="in") opt.in=val;
                else if(key=="out") opt.out=val;
                else if(key=="count") opt.count_only=val!="0";
                else if(key=="minelo") opt.min_elo=std::stoi(val);
                else if(key=="minseconds") opt.min_seconds=std::stoi(val);
                else if(key=="minmove") opt.min_move=std::max(1,std::stoi(val));
                else if(key=="maxmove") opt.max_move=std::max(1,std::stoi(val));
                else if(key=="games") opt.max_games=std::stoull(val);
                else if(key=="endnodes") opt.end_nodes=std::stoull(val);
                else if(key=="endmin") opt.end_min=std::stoi(val);
                else if(key=="endmax") opt.end_max=std::stoi(val);
                else if(key=="seed") opt.seed=std::stoull(val);
                else if(key=="hash") opt.hash_mb=std::clamp(std::stoi(val),1,4096);
            }
            if(opt.in.empty() || (opt.out.empty() && !opt.count_only))
                std::cout << "info string reservoir needs in=<pgn> and out=<prefix> (or count=1)" << std::endl;
            else build_reservoir(opt,std::cout);
        } else if(cmd=="genconlog") {
            // "genconlog out=<prefix> [games=N] [randomplies=N] [nodes=N]
            //  [mindepth=N] [rate=N] [depthscale=PCT] [seed=N] [hash=MB]":
            // continuation logs from self-play; see cont_log.h.
            ContLogOptions opt;
            std::string tok;
            while(ss>>tok) {
                const auto eq=tok.find('=');
                if(eq==std::string::npos) continue;
                const std::string key=tok.substr(0,eq), val=tok.substr(eq+1);
                if(key=="out") opt.out=val;
                else if(key=="games") opt.games=std::max(1,std::stoi(val));
                else if(key=="randomplies") opt.random_plies=std::max(0,std::stoi(val));
                else if(key=="nodes") opt.nodes=std::stoull(val);
                else if(key=="mindepth") opt.min_depth=std::clamp(std::stoi(val),1,63);
                else if(key=="rate") opt.rate=std::max(1,std::stoi(val));
                else if(key=="depthscale") opt.depth_scale=std::max(100,std::stoi(val));
                else if(key=="seed") opt.seed=std::stoull(val);
                else if(key=="hash") opt.hash_mb=std::clamp(std::stoi(val),1,4096);
            }
            if(opt.out.empty()) std::cout << "info string genconlog needs out=<prefix>" << std::endl;
            else generate_cont_log(opt,std::cout);
        } else if(cmd=="gendata") {
            // "gendata out=<file> [worker=N] [seed=N] [hours=H] [games=N]
            //  [softnodes=N] [hardnodes=N] [softscale=0|1] [mindepth=N] [dropsoftnodes=N] [drophardnodes=N]
            //  [backsoftnodes=N] [backhardnodes=N]
            //  [bingoevery=N] [randomplies=A-B]
            //  [maxopenscore=CP] [maxplies=N] [hash=MB] [adjscore=CP]
            //  [adjplies=N] [adjphase=N] [adjtb=0|1] [adjdraw=CP]
            //  [adjdrawplies=N] [adjdrawply=N] [adjpct=PCT] [adjmate=N]
            //  [sharenet=0|1]
            //  [starts=<prefix>] [startspct=PCT] [startmix=E:L:N] [startplies=A-B]
            //  [startstride=N]
            //  [castlelong=PCT] [castlelongby=PLY]": self-play
            // training data; see datagen.h.
            DataOptions opt;
            opt.syzygy=syzygy_path;
            std::string tok;
            while(ss>>tok) {
                const auto eq=tok.find('=');
                if(eq==std::string::npos) continue;
                const std::string key=tok.substr(0,eq), val=tok.substr(eq+1);
                if(key=="out") opt.out=val;
                else if(key=="worker") opt.worker=std::clamp(std::stoi(val),0,65535);
                else if(key=="seed") opt.seed=std::stoull(val);
                else if(key=="hours") opt.hours=std::stod(val);
                else if(key=="games") opt.games=std::stoull(val);
                else if(key=="softnodes") opt.soft_nodes=std::stoull(val);
                else if(key=="hardnodes") opt.hard_nodes=std::stoull(val);
                else if(key=="softscale") opt.soft_scale=val!="0";
                else if(key=="mindepth") opt.min_depth=std::clamp(std::stoi(val),0,64);
                else if(key=="dropsoftnodes") opt.drop_soft_nodes=std::stoull(val);
                else if(key=="drophardnodes") opt.drop_hard_nodes=std::stoull(val);
                else if(key=="backsoftnodes") opt.back_soft_nodes=std::stoull(val);
                else if(key=="backhardnodes") opt.back_hard_nodes=std::stoull(val);
                else if(key=="bingoevery") opt.bingo_every=std::max(0,std::stoi(val));
                else if(key=="randomplies") {
                    const auto dash=val.find('-');
                    opt.random_min=std::clamp(std::stoi(val.substr(0,dash)),0,60);
                    opt.random_max=dash==std::string::npos ? opt.random_min
                        : std::clamp(std::stoi(val.substr(dash+1)),opt.random_min,60);
                }
                else if(key=="maxopenscore") opt.max_open_score=std::stoi(val);
                else if(key=="maxplies") opt.max_plies=std::clamp(std::stoi(val),1,60000);
                else if(key=="hash") opt.hash_mb=std::clamp(std::stoi(val),1,4096);
                else if(key=="adjscore") opt.adj_score=std::max(0,std::stoi(val));
                else if(key=="adjplies") opt.adj_plies=std::max(1,std::stoi(val));
                else if(key=="adjphase") opt.adj_phase=std::clamp(std::stoi(val),0,24);
                else if(key=="adjtb") opt.adj_tb=val!="0";
                else if(key=="adjdraw") opt.adj_draw_score=std::max(0,std::stoi(val));
                else if(key=="adjmate") opt.adj_mate=std::max(0,std::stoi(val));
                else if(key=="sharenet") opt.share_net=val!="0";
                else if(key=="adjdrawplies") opt.adj_draw_plies=std::max(1,std::stoi(val));
                else if(key=="adjdrawply") opt.adj_draw_ply=std::max(0,std::stoi(val));
                else if(key=="adjpct") opt.adj_pct=std::clamp(std::stoi(val),0,100);
                else if(key=="starts") opt.starts=val;
                else if(key=="startspct") opt.starts_pct=std::clamp(std::stoi(val),0,100);
                else if(key=="startmix") {
                    // early:late:end weights
                    std::stringstream ms(val);
                    std::string w;
                    for(std::size_t i=0;i<3 && std::getline(ms,w,':');++i) opt.starts_mix[i]=std::max(0,std::stoi(w));
                }
                else if(key=="startplies") {
                    const auto dash=val.find('-');
                    opt.starts_plies_min=std::max(0,std::stoi(val.substr(0,dash)));
                    opt.starts_plies_max=dash==std::string::npos ? opt.starts_plies_min
                                                                 : std::max(opt.starts_plies_min,std::stoi(val.substr(dash+1)));
                }
                else if(key=="startstride") opt.starts_stride=std::clamp(std::stoi(val),1,4096);
                else if(key=="castlelong") opt.castle_long_pct=std::clamp(std::stoi(val),0,100);
                else if(key=="castlelongby") opt.castle_long_by=std::clamp(std::stoi(val),1,1000);
            }
            if(opt.out.empty()) std::cout << "info string gendata needs out=<file>" << std::endl;
            else generate_data(opt,std::cout);
        } else if(cmd=="nnueeval") {
            // "nnueeval in=<data file or directory> out=<file> [max=N] [fen=1]
            //  [p8=<file>]": the loaded network's evals beside search scores
            // and results; see datagen.h. With p8, a layered network's first
            // dense layer inputs too, hidden bytes a position, for
            // tools/nnue/permute.py.
            std::string in, out, p8, tok;
            std::uint64_t max = 0;
            bool fen = false;
            while(ss>>tok) {
                const auto eq=tok.find('=');
                if(eq==std::string::npos) continue;
                const std::string key=tok.substr(0,eq), val=tok.substr(eq+1);
                if(key=="in") in=val;
                else if(key=="out") out=val;
                else if(key=="max") max=std::stoull(val);
                else if(key=="fen") fen=val!="0";
                else if(key=="p8") p8=val;
            }
            std::ofstream file(out);
            std::FILE* dump = p8.empty() ? nullptr : std::fopen(p8.c_str(), "wb");
            if(in.empty() || !file || (!p8.empty() && !dump))
                std::cout << "info string nnueeval needs in=<data> and writable out=<file> (and p8=<file>)" << std::endl;
            else {
                nnue::dump_layer1_inputs(dump);
                std::cout << "info string nnueeval: " << nnue_eval_data(in,max,file,fen) << " positions" << std::endl;
                nnue::dump_layer1_inputs(nullptr);
            }
            if(dump) std::fclose(dump);
        } else if(cmd=="labelpgn") {
            // "labelpgn in=<pgn> out=<file> [softnodes=N] [hardnodes=N] [hash=MB]
            //  [skip=plies] [worker=N] [seed=N]": FunComp's labels for games
            // played elsewhere; see datagen.h.
            LabelPgnOptions opt;
            std::string tok;
            while(ss>>tok) {
                const auto eq=tok.find('=');
                if(eq==std::string::npos) continue;
                const std::string key=tok.substr(0,eq), val=tok.substr(eq+1);
                if(key=="in") opt.in=val;
                else if(key=="out") opt.out=val;
                else if(key=="softnodes") opt.soft_nodes=std::stoull(val);
                else if(key=="hardnodes") opt.hard_nodes=std::stoull(val);
                else if(key=="hash") opt.hash_mb=std::max(1,std::stoi(val));
                else if(key=="skip") opt.skip=std::max(0,std::stoi(val));
                else if(key=="worker") opt.worker=std::clamp(std::stoi(val),0,65535);
                else if(key=="seed") opt.seed=std::stoull(val);
            }
            if(opt.in.empty() || opt.out.empty())
                std::cout << "info string labelpgn needs in=<pgn> and out=<file>" << std::endl;
            else label_pgn(opt,std::cout);
        } else if(cmd=="rescore") {
            // "rescore in=<data> out=<new data> [softnodes=N] [hardnodes=N]
            //  [softscale=0|1] [hash=MB] [games=N] [roll=N,N,...] [backpass=N]":
            // new labels for stored positions with the loaded network, rolled
            // down the played line or by a backward pass; see datagen.h.
            RescoreOptions opt;
            std::string tok;
            while(ss>>tok) {
                const auto eq=tok.find('=');
                if(eq==std::string::npos) continue;
                const std::string key=tok.substr(0,eq), val=tok.substr(eq+1);
                if(key=="in") opt.in=val;
                else if(key=="out") opt.out=val;
                else if(key=="softnodes") opt.soft_nodes=std::stoull(val);
                else if(key=="hardnodes") opt.hard_nodes=std::stoull(val);
                else if(key=="softscale") opt.soft_scale=val!="0";
                else if(key=="hash") opt.hash_mb=std::max(1,std::stoi(val));
                else if(key=="games") opt.games=std::stoull(val);
                else if(key=="backpass") opt.back_pass=std::stoull(val);
                else if(key=="roll") {
                    std::stringstream list(val);
                    for(std::string item; std::getline(list,item,',');)
                        if(!item.empty()) opt.roll.push_back(std::stoull(item));
                }
            }
            if(opt.in.empty() || opt.out.empty())
                std::cout << "info string rescore needs in=<data> and out=<new data>" << std::endl;
            else rescore_data(opt,std::cout);
        } else if(cmd=="datatext") {
            // "datatext in=<file or directory> out=<file> [quiet=0|1] [label=forward|back]
            //  [unscale=0|1] [thin=T:R,...] [cap=N]
            //  [drawkeep=P] [gamekeep=P] [hardscore=T] [hardswing=S] [hardcopies=K]
            //  [blend=W] [blendscale=K] [rescale=F] [fiftyscale=N]
            //  [castleclean=PLY] [drop=SLICE,...] [index=<first-index file>
            //  part=a|b] [qsquiet=CP] [tdlambda=L tdplies=N tdcap=CP]": positions as text for the trainer; see
            //  datagen.h (qsquiet with the network to test with loaded).
            //  An unknown slice name or part ignores the command.
            std::string in, out, tok;
            TextOptions opt;
            while(ss>>tok) {
                const auto eq=tok.find('=');
                if(eq==std::string::npos) continue;
                const std::string key=tok.substr(0,eq), val=tok.substr(eq+1);
                if(key=="in") in=val;
                else if(key=="out") out=val;
                else if(key=="quiet") opt.quiet=val!="0";
                else if(key=="qsquiet") opt.qs_quiet=std::max(-1,std::stoi(val));
                else if(key=="label") opt.backward=val=="back";
                else if(key=="tdlambda") opt.td_lambda=std::clamp(std::stod(val),0.0,1.0);
                else if(key=="tdplies") opt.td_plies=std::clamp(std::stoi(val),1,32);
                else if(key=="tdcap") opt.td_cap=std::max(1,std::stoi(val));
                else if(key=="unscale") opt.unscale=val!="0";
                else if(key=="cap") opt.cap=std::max(0,std::stoi(val));
                else if(key=="drawkeep") opt.draw_keep_pct=std::clamp(std::stoi(val),0,100);
                else if(key=="gamekeep") opt.game_keep_pct=std::clamp(std::stoi(val),0,100);
                else if(key=="hardscore") opt.hard_score=std::max(0,std::stoi(val));
                else if(key=="hardswing") opt.hard_swing=std::max(0,std::stoi(val));
                else if(key=="hardcopies") opt.hard_copies=std::clamp(std::stoi(val),1,10);
                else if(key=="blend") opt.blend=std::stod(val);
                else if(key=="blendscale") opt.blend_scale=std::max(1.0,std::stod(val));
                else if(key=="rescale") opt.rescale=std::stod(val);
                else if(key=="fiftyscale") opt.fifty_scale=std::max(0,std::stoi(val));
                else if(key=="castleclean") opt.castle_clean=std::max(0,std::stoi(val));
                else if(key=="drop") opt.drop=slice_bits(val);
                else if(key=="index") opt.index=val;
                else if(key=="part") {
                    if(val!="a" && val!="b") throw std::invalid_argument("part is a or b");
                    opt.part=val[0];
                }
                else if(key=="thin") {
                    // thin=<threshold>:<rate>,... (e.g. 400:3,1000:10)
                    std::stringstream list(val);
                    for(std::string item; std::getline(list,item,',');) {
                        const auto colon=item.find(':');
                        if(colon==std::string::npos) continue;
                        opt.thin.push_back({std::stoi(item.substr(0,colon)),
                                            std::max(1,std::stoi(item.substr(colon+1)))});
                    }
                    std::sort(opt.thin.begin(),opt.thin.end(),
                              [](const TextOptions::Thin& a,const TextOptions::Thin& b){
                                  return a.threshold<b.threshold; });
                }
            }
            std::ofstream file(out);
            if(in.empty() || out.empty() || !file)
                std::cout << "info string datatext needs in=<file> and a writable out=<file>" << std::endl;
            else if(opt.part && opt.index.empty())
                std::cout << "info string datatext: part= needs index=<first-index file>" << std::endl;
            else data_text(in,file,opt,std::cout,&pool.main());
        } else if(cmd=="featstats") {
            // "featstats in=<data file or directory> [games=N]": candidate
            // input-feature families measured on played games; see featstats.h.
            std::string in, tok;
            std::uint64_t games=0;
            while(ss>>tok) {
                const auto eq=tok.find('=');
                if(eq==std::string::npos) continue;
                const std::string key=tok.substr(0,eq), val=tok.substr(eq+1);
                if(key=="in") in=val;
                else if(key=="games") games=std::stoull(val);
            }
            if(in.empty()) std::cout << "info string featstats needs in=<file>" << std::endl;
            else feature_stats(in,games,std::cout);
        } else if(cmd=="datasplit") {
            // "datasplit in=<file or directory> index=<first-index file>
            //  [a=<dir> b=<dir>]": each worker file cut in two at its game
            // index (round 3a and 3b); without a and b, a dry run. See
            // datagen.h.
            SplitOptions opt;
            std::string tok;
            while(ss>>tok) {
                const auto eq=tok.find('=');
                if(eq==std::string::npos) continue;
                const std::string key=tok.substr(0,eq), val=tok.substr(eq+1);
                if(key=="in") opt.in=val;
                else if(key=="index") opt.index=val;
                else if(key=="a") opt.a=val;
                else if(key=="b") opt.b=val;
            }
            if(opt.in.empty() || opt.index.empty())
                std::cout << "info string datasplit needs in=<data> and index=<first-index file>" << std::endl;
            else split_data(opt,std::cout);
        } else if(cmd=="datashuffle") {
            // "datashuffle in=<file> out=<file> [mem=<MB>] [seed=<n>]
            //  [record=<bytes>]": a training file's records in a random order
            // within a memory limit (the move out of WSL). See datagen.h.
            ShuffleOptions opt;
            std::string tok;
            try {
                while(ss>>tok) {
                    const auto eq=tok.find('=');
                    if(eq==std::string::npos) continue;
                    const std::string key=tok.substr(0,eq), val=tok.substr(eq+1);
                    if(key=="in") opt.in=val;
                    else if(key=="out") opt.out=val;
                    else if(key=="mem") opt.mem_mb=std::stoull(val);
                    else if(key=="seed") opt.seed=std::stoull(val);
                    else if(key=="record") opt.record=std::stoull(val);
                }
                if(opt.in.empty() || opt.out.empty())
                    std::cout << "info string datashuffle needs in=<file> and out=<file>" << std::endl;
                else shuffle_data(opt,std::cout);
            } catch(const std::exception&) {
                std::cout << "info string datashuffle: a number expected in mem=, seed= or record=" << std::endl;
            }
        } else if(cmd=="relabel") {
            // "relabel in=<file> out=<file> [scale=180]": a training file's
            // scores replaced by the loaded network's static evaluations (round
            // 6's distillation). See datagen.h.
            RelabelOptions opt;
            std::string tok;
            try {
                while(ss>>tok) {
                    const auto eq=tok.find('=');
                    if(eq==std::string::npos) continue;
                    const std::string key=tok.substr(0,eq), val=tok.substr(eq+1);
                    if(key=="in") opt.in=val;
                    else if(key=="out") opt.out=val;
                    else if(key=="scale") opt.scale=std::stoi(val);
                }
                if(opt.in.empty() || opt.out.empty())
                    std::cout << "info string relabel needs in=<file> and out=<file>" << std::endl;
                else relabel_data(opt,std::cout);
            } catch(const std::exception&) {
                std::cout << "info string relabel: a number expected in scale=" << std::endl;
            }
        } else if(cmd=="datadump" || cmd=="datastats") {
            // "datadump in=<file> [games=N] [back=1]": every searched position of the
            // games, one line each; "datastats in=<file or directory>":
            // totals and distributions; [scale=K] the data's eval scale (round 1
            // 120, round 2 180) for the conversion table. See datagen.h.
            std::string in, tok;
            std::uint64_t games=0;
            int scale=120;
            bool back=false;  // datadump: the backward pass's fields too
            while(ss>>tok) {
                const auto eq=tok.find('=');
                if(eq==std::string::npos) continue;
                const std::string key=tok.substr(0,eq), val=tok.substr(eq+1);
                if(key=="in") in=val;
                else if(key=="games") games=std::stoull(val);
                else if(key=="back") back=val!="0";
                else if(key=="scale") scale=std::stoi(val);
            }
            if(in.empty()) std::cout << "info string " << cmd << " needs in=<file>" << std::endl;
            else if(cmd=="datadump") dump_data(in,games,std::cout,back);
            else data_stats(in,std::cout,scale);
            std::cout << std::flush;
        }
    } catch(const std::logic_error&) {
        // A number that doesn't parse or doesn't fit (std::stoi and the like
        // throw): the command is ignored, the engine carries on.
        std::cout << "info string ignored (bad value): " << line << std::endl;
    }
    // End of input (the GUI went away): the search mustn't outlive the
    // searchers.
    finish_search();
    if(!telemetry_dir.empty()) write_telemetry(telemetry_dir,pool.total_stats());
}
