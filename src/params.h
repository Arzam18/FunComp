#pragma once
#include <vector>

// Search parameters, gathered in one place so they can be retuned when the
// evaluation changes (their right values depend on its scale). A tuning
// build has them as plain variables, set through UCI options: configure
// with -DTUNE=ON and every entry in tunables() is advertised as a spin
// option, ready for SPSA. Release and datagen builds (FROZEN_PARAMS, on
// unless TUNE) compile them as constants instead, which the compiler folds
// into the code: 2.3-3.9% faster on one thread and 3.0% at datagen's 32
// copies, the same search (results.md, "Frozen parameters"). The tests
// keep them variables. NnueScale and Move Overhead are variables in every
// build: each network brings its own scale, and the GUI sets the overhead.
//
// Of the pruning parameters, the search reads only the depth limits that
// decide whether a heuristic applies at all. Margins, thresholds and
// reductions are computed by the functions in pruning.h, which are where a
// learned guide takes over.
//
// Since 2.0 the defaults of the tuned parameters come from a full SPSA
// pass over 91 of them (80,000 games at 8+0.08, confirmed at 8+0.08 and
// 40+0.4; the list is tools/spsa/full.csv). Comments give the values
// features were tested at alone; the tuned values are what they settled
// at together, so the two can differ widely. Switches (on/off options)
// were not tuned.
#ifdef FUNCOMP_FROZEN_PARAMS
#define FUNCOMP_PARAM constexpr
#else
#define FUNCOMP_PARAM
#endif

namespace params {

// Time management. The soft limit (no new iteration after it) is scaled
// by how settled the search looks, in percent: tm_stable_start with a best
// move that just changed, less tm_stable_step per iteration it has held,
// down to tm_stable_min; plus tm_drop_scale per 100 cp the score fell since
// the last iteration, at most tm_drop_max. The hard limit still caps it.
// 100/0/100/0/0 turns the scaling off (always 100%). Hand-set at
// 140/15/75/50/60 it tested at +1.4 +- 7.6 Elo at 8+0.08 (4022 games); the
// time-management SPSA (at 20+0.2, all the terms below with it) turned it
// on, and the tuned set gained +64.1 +- 14.6 Elo at 40+0.4 (778 games),
// using 2% more time per move (5% more in the first 20 moves).
inline FUNCOMP_PARAM int tm_stable_start = 158;
inline FUNCOMP_PARAM int tm_stable_step = 14;
inline FUNCOMP_PARAM int tm_stable_min = 79;
inline FUNCOMP_PARAM int tm_drop_scale = 47;
inline FUNCOMP_PARAM int tm_drop_max = 56;
// And by how much of the search went into the best move: from depth
// tm_nodes_min_depth, with frac the percentage of the root moves' nodes
// (over all iterations so far) spent under the best one, the soft limit is
// scaled by max(tm_nodes_base - tm_nodes_slope * frac / 100, tm_nodes_min)
// percent: most nodes in the best move, settled, stop sooner; spread out,
// think longer. 100 / 0 turns it off. On at 230 / 180: +26.3 +- 10.4 Elo at
// 8+0.08 (1982 games) and +14.8 +- 9.4 at 40+0.4 (2018 games), using 2-3%
// less time per move.
inline FUNCOMP_PARAM int tm_nodes_base = 251;
inline FUNCOMP_PARAM int tm_nodes_slope = 151;
inline FUNCOMP_PARAM int tm_nodes_min = 65;
inline FUNCOMP_PARAM int tm_nodes_min_depth = 7;

// Time kept in hand per move for GUI and operating-system latency, in
// milliseconds. Also the UCI option "Move Overhead" in every build.
inline int move_overhead = 30;

// Ranking a tablebase root by distance to zeroing reads the DTZ tables,
// which can take 50 ms or more when they are cold and the disk is busy
// (time losses in bullet with threads). With less than tm_tb_dtz_min_ms
// on the clock (or as the move time) the root is ranked by WDL alone: the
// result is kept, and the search goes on probing to make progress. 0
// always ranks by DTZ.
inline FUNCOMP_PARAM int tm_tb_dtz_min_ms = 1000;

// The time allocated to a move: target = time left / tm_moves_to_go (when
// the GUI gives no moves-to-go) + tm_inc_pct percent of the increment. The
// hard limit is tm_hard_pct percent of the target, at most tm_hard_cap_pct
// percent of the time left plus the increment; no new iteration starts
// past tm_soft_pct percent of the target (scaled as above).
inline FUNCOMP_PARAM int tm_moves_to_go = 25;
inline FUNCOMP_PARAM int tm_inc_pct = 76;
inline FUNCOMP_PARAM int tm_hard_pct = 305;
inline FUNCOMP_PARAM int tm_hard_cap_pct = 51;
inline FUNCOMP_PARAM int tm_soft_pct = 64;

// Root aspiration windows.
inline FUNCOMP_PARAM int asp_min_depth = 7;   // first iteration searched with a window
inline FUNCOMP_PARAM int asp_delta = 13;      // initial half-width of the window, cp
inline FUNCOMP_PARAM int asp_widen_pct = 42;  // each failed window widens by this percent

// History heuristic. A quiet move that causes a cutoff at depth d gets a
// bonus of min(hist_bonus_mult * d * d, hist_bonus_max); quiets searched
// before it get the same amount as a penalty. Scores stay within
// +-hist_max, so they never outrank killers or captures. Each update is
// clamped to +-hist_max, which keeps that bound at any setting; the tables
// hold 16-bit values, so hist_max is at most 32767.
inline FUNCOMP_PARAM int hist_bonus_mult = 42;
inline FUNCOMP_PARAM int hist_bonus_max = 2079;
inline FUNCOMP_PARAM int hist_max = 6553;
// Eval-change history: after the opponent's quiet move, the change in the
// static eval from its side (delta), times eval_hist_mult / 16, clamped to
// -eval_hist_malus_max..eval_hist_bonus_max, updates that move in its
// side's butterfly history, on a first visit or at depth <=
// eval_hist_max_depth. For after the first network (search addition E);
// eval_hist_mult 0 turns it off. At 64 it lost -17.8 +- 9.3 Elo (2514
// games), updating at 98% of eligible nodes; histories sank on balance
// (history pruning fired at 46% of its tries against 38%), likely a
// negative offset in the delta (a quiet move rarely raises its own
// side's static eval): measure and centre the delta before retrying.
inline FUNCOMP_PARAM int eval_hist_mult = 25;
inline FUNCOMP_PARAM int eval_hist_bonus_max = 1018;
inline FUNCOMP_PARAM int eval_hist_malus_max = 333;
inline FUNCOMP_PARAM int eval_hist_max_depth = 9;
// The penalty for the quiet moves searched before the cutoff, likewise:
// min(hist_malus_mult * d * d, hist_malus_max). Capture history has its
// own bonus and penalty on the same pattern.
inline FUNCOMP_PARAM int hist_malus_mult = 73;
inline FUNCOMP_PARAM int hist_malus_max = 980;
inline FUNCOMP_PARAM int capt_bonus_mult = 51;
inline FUNCOMP_PARAM int capt_bonus_max = 2893;
inline FUNCOMP_PARAM int capt_malus_mult = 7;
inline FUNCOMP_PARAM int capt_malus_max = 2773;
// Prior-countermove bonus: when every move of a node fails low, the
// opponent's quiet move that led there gets prior_cm_pct percent of the
// usual cutoff bonus in its butterfly and continuation histories. 0 turns
// it off. As tested at first it fires at every fail-low, on top of the
// cutoff bonus the parent gives the same move, and again after a reduced
// or null-window search that is then repeated: at 100% it lost -29.3 +-
// 12.1 Elo (1714 games), and the tune took it down to 34. Three gates,
// each 0 = off:
// - prior_cm_final_only: only the parent's final search of the move
//   counts, not a provisional one that is repeated when it fails low;
// - prior_cm_node_gate: only where the failing node is a PV or cut node
//   (a refutation nobody expected), not an expected all-node;
// - prior_cm_index_cap: scaled by the move's place in the parent's order,
//   min(index, cap) / cap: nothing for a first move, the full share from
//   the cap on.
// prior_cm_final_only is on as a correctness fix: it drops the 7% of
// bonuses that came from repeated searches, flat at the tuned 34 (-3.5 +-
// 7.2 Elo, 4034 games; alone at 100%, -1.0 +- 7.3). The node gate (-5.9 +-
// 7.2 with the fix, at 100%) and the index cap stay off.
inline FUNCOMP_PARAM int prior_cm_pct = 3;
// Hash-cutoff history: when a non-PV node is cut off by a hash entry that
// fails high with a quiet hash move, that move gets tt_cut_hist_pct percent
// of a searched cutoff's history bonus at the node's depth (no killer, no
// maluses). 0 turns it off. Off: at 50 it lost -4.8 +- 6.8 Elo (4040 games),
// rewarding a quiet hash move at 55% of hash cutoffs.
inline FUNCOMP_PARAM int tt_cut_hist_pct = 24;
inline FUNCOMP_PARAM int prior_cm_final_only = 1;
inline FUNCOMP_PARAM int prior_cm_node_gate = 0;
inline FUNCOMP_PARAM int prior_cm_index_cap = 4;
// Capture history (same update rule) orders captures of equal victims;
// it enters the capture's score as history * capt_hist_mult / 1024 (1/8).
inline FUNCOMP_PARAM int capt_hist_mult = 192;
// Continuation history 4 plies back (same table and update rule) orders
// quiet moves, weighted cont_hist4_weight/256; the history that
// reductions and pruning read leaves it out. 0 turns it off, updates
// included. On since the 2.0 tune; alone, at 128 it leaned negative (-3.1 +- 6.4 Elo,
// 5500 games).
inline FUNCOMP_PARAM int cont_hist4_weight = 183;
// Quiet moves that give direct check get this much added to their
// ordering score (history, at most about 49,000 either way, stays well
// below the killers). Ordering only: checks are already spared LMP and
// futility pruning. 0 turns it off. Off: at 8000 it lost -13.5 +- 8.1 Elo
// (3148 games), bumping 12% of the quiet moves searched.
inline FUNCOMP_PARAM int quiet_check_bonus = 568;

// Pawn history (quiet history by pawn structure) joins the quiet history
// that orders, reduces and prunes quiet moves, weighted pawn_hist_weight/256.
// 0 turns it off, updates included. On since the 2.0 tune; alone, at 256 it leaned
// negative (-5.8 +- 8.3 Elo, 3224 games).
inline FUNCOMP_PARAM int pawn_hist_weight = 228;

// Transposition table replacement: an entry written in the current search
// is only overwritten by a search at most tt_replace_margin plies
// shallower (or an exact score for the same position).
inline FUNCOMP_PARAM int tt_replace_margin = 2;
// Replacement prefers the entry with the lowest depth - tt_age_weight * age.
inline FUNCOMP_PARAM int tt_age_weight = 10;
// A store from a node flagged ttPv may replace an entry of the same
// position up to tt_pv_replace_bonus plies deeper than others may: PV
// information is worth keeping fresh. 0 treats all stores alike (off
// before the 2.0 tune; alone at 2 it leaned negative, -4.9 +- 7.2 Elo,
// 4038 games).
inline FUNCOMP_PARAM int tt_pv_replace_bonus = 4;
// A node that fails low below a parent flagged ttPv inherits the flag
// (in its hash entry) when its depth exceeds tt_pv_inherit_depth. 64
// turns this off (as before the 2.0 tune; alone at 3 it was flat, -1.9
// +- 7.5 Elo, 4038 games).
inline FUNCOMP_PARAM int tt_pv_inherit_depth = 3;

// Pawn correction history. After each search of a node that isn't in
// check, the gap between its score and its raw static eval is folded into
// a running average kept per pawn structure (pawn key) and side to move;
// the average is added to later static evals in the same structure. Each
// update weighs min(depth + 1, corr_weight_max) / 256, and the correction
// is capped at +-corr_limit centipawns.
inline FUNCOMP_PARAM int corr_weight_max = 14;
inline FUNCOMP_PARAM int corr_limit = 24;
// Non-pawn correction history: the same per side to move and each
// colour's non-pawn pieces (king included), updated alike; their sum
// enters the eval weighted by corr_nonpawn_weight/256 (0 turns it off).
inline FUNCOMP_PARAM int corr_nonpawn_weight = 181;
// Continuation correction history: the same per side to move and the move
// that led to the position (its piece and destination), updated alike and
// weighted corr_cont_weight/256 (0 turns it off). Main-search nodes only:
// quiescence doesn't record its moves.
inline FUNCOMP_PARAM int corr_cont_weight = 120;
// Reverse futility and futility margins grow by corr_margin_pct percent of
// the correction's size: where correction history moves the eval a lot,
// the eval is less sure. 0 turns it off (as before the 2.0 tune; alone at
// 50 it was flat, +2.1 +- 7.4 Elo, 4056 games).
inline FUNCOMP_PARAM int corr_margin_pct = 60;

// Fifty-move scaling: the corrected static eval is multiplied by
// (fifty_scale - halfmove clock) / fifty_scale, so it shrinks toward a
// draw as the clock runs (to half at 64 with 128) and the engine prefers
// moves that make progress. 0 turns it off.
inline FUNCOMP_PARAM int fifty_scale = 203;

// The network evaluation (nnue.h): centipawns per unit of its output, the
// trainer's sigmoid scale (the first network: 120, fitted to self-play
// results; 3.0's network: 184 matched, 188 after the 3.0 fine tune; 3.1's:
// 185 matched, 192 after 3.1's focused tune). Used with a network loaded
// (EvalFile or the embedded one). Settable in every build (NnueScale),
// since each network has its own.
inline int nnue_scale = 192;
// Halfmove-clock buckets in the hash-table key: with fifty-move scaling a
// score depends on the clock, which the position's key leaves out. From
// fifty_hash_start on, every fifty_hash_width half-moves of clock get a
// key of their own (at most 15 buckets) mixed into the key the hash table
// uses, so a score stored at clock 10 isn't reused at clock 80. Repetition
// detection and the other keys are untouched. Width 0 turns it off. On
// since 2.0 at 10 / 10: +11.5 +- 10.3 Elo (2018 games).
inline FUNCOMP_PARAM int fifty_hash_start = 10;
inline FUNCOMP_PARAM int fifty_hash_width = 7;

// Mop-up (eval.cpp): against a bare king, the side with mating material
// earns mop_edge per step of that king's distance from the centre (0-6;
// with bishop and knight, closeness to the bishop's corner, 0-7) and
// mop_kings per step the kings are closer than 7 king moves apart. Both
// 0 turn it off. Accepted: no harm in games (+0.4 +- 6.6 Elo, 4044 games),
// and at 5 ms a move it mates 46 of 48 bare-king starts against 30.
inline FUNCOMP_PARAM int mop_edge = 20;
inline FUNCOMP_PARAM int mop_kings = 10;
// Drawishness: endings where an edge rarely converts scale the whole eval
// toward a draw. Opposite-coloured bishops with no other pieces (pawns
// allowed) keep draw_ocb_pct percent; pawnless positions where the
// material difference is under draw_pawnless_margin centipawns (SEE values:
// queen against rook is 400) keep draw_pawnless_pct percent. 100 turns
// each off. On at 50/50: +15.8 +- 8.8 Elo (2074 games), with the margin at
// 400, which left queen against rook unscaled; the second full pass took
// it to 499, which scales that ending too.
inline FUNCOMP_PARAM int draw_ocb_pct = 73;
inline FUNCOMP_PARAM int draw_pawnless_pct = 44;
inline FUNCOMP_PARAM int draw_pawnless_margin = 499;
// Initiative (initiative.cpp): who is making the opponent respond. Added to
// the PeSTO score; all terms from the side to move's point of view and
// colour-symmetric. Hand-set at first so that typical positions saw 10-30
// cp (+20.3 +- 9.0 Elo untuned); tuned by a 2,000-iteration SPSA
// (tools/spsa/initiative.csv): +64.9 +- 17.1 more. init_scale scales
// the whole term in percent (0: off, not computed), tuned with the rest to
// see whether the style survives; init_clamp caps the raw term.
inline FUNCOMP_PARAM int init_scale = 226;
inline FUNCOMP_PARAM int init_clamp = 149;
inline FUNCOMP_PARAM int init_tempo = 14;           // cp for the side to move
inline FUNCOMP_PARAM int init_mobility = 14;        // 1/16 cp per weighted safe square
inline FUNCOMP_PARAM int init_territory = 3;        // 1/4 cp per uncontested square in their half
inline FUNCOMP_PARAM int init_coordination = 3;     // cp per unit we hit twice, they defend once
inline FUNCOMP_PARAM int init_push_threat = 14;     // cp per piece a safe pawn push would attack
inline FUNCOMP_PARAM int init_safe_check = 29;      // cp per piece type with a safe check
inline FUNCOMP_PARAM int init_king_pressure = 45;   // cp * danger^2 / 256, tapered by phase
inline FUNCOMP_PARAM int init_threat_latent = 58;   // 1/256 of the stm's two largest threats
inline FUNCOMP_PARAM int init_threat_pending = 14;   // 1/256 of the largest threat against the stm
inline FUNCOMP_PARAM int init_threat_double = 60;   // 1/256 of the second largest against it
// The whole term fades toward the endgame: at phase 0 (no pieces) it keeps
// init_endgame_pct percent, linearly in the phase (24 = all pieces). Having
// the move can be a liability in endings, and activity there often doesn't
// convert. 100 turns it off. Off: at 50 it lost -11.2 +- 7.3 Elo (3592
// games): the term's endgame weight pays; the full SPSA pass tunes it from
// 100 (both directions open), and took it to 42: a taper after all, with
// the weights retuned alongside it.
inline FUNCOMP_PARAM int init_endgame_pct = 42;
// Correction history learns in unscaled units: the searched score is
// divided by the fifty-move scale before the raw eval is subtracted, so
// the scaling (applied after the correction) isn't learned as eval error.
// 0 learns the scaled score minus the raw eval. On since 2.0: with
// FiftyScale 200 it lost -9.2 +- 7.3 Elo (4064 games), but with the tuned
// FiftyScale it came out slightly ahead (+4.0 +- 7.4, 4066 games), and it
// is the correct way to learn.
inline FUNCOMP_PARAM int corr_unscale_fifty = 1;

// Upcoming-repetition detection (1: on, 0: off): a node from which one
// reversible move reaches an earlier position of its line is worth at
// least a draw.
inline FUNCOMP_PARAM int rep_in_reach = 1;

// Pure search (1: on; ROADMAP step 4b, the author 2026-10-15): the evaluation
// alone, for testing networks against each other at a fixed depth. Nothing
// is pruned, reduced or extended (no node- or move-level pruning, null move,
// ProbCut, singular or check extension, internal iterative reduction or
// late-move reduction), no hash cut, no upcoming-repetition shortcut, no
// tablebases, and the static eval is the network's own (no correction
// history, fifty-move scaling, sharpening by the hash score or fail-firm
// blend). Quiescence keeps stand pat, its captures and its SEE filter
// (material values only), without delta pruning or the quiet-evasion limit.
// Plain alpha-beta only compares, so a fixed-depth result is the same when
// every eval is multiplied by a positive constant: no matched scale needed.
// Its test: at depth 3 the root score equals a plain minimax over the same
// quiescence. Off for play and datagen. 2: the same with the check extension
// kept (frame P4c, the author 2026-10-17: does a net gain in the full search
// because the search follows checks against the king further?).
inline FUNCOMP_PARAM int pure_search = 0;

// A ladder frame at data generation's limits (L2, the author 2026-10-19: the
// latest round's datagen configuration, so a ladder match also tells how a
// net would generate): with frame_soft_nodes set, "go nodes N" searches as
// gendata does, N the hard limit, frame_soft_nodes the soft one (scaled by
// stability when frame_soft_scale), and a root in check or a tablebase root
// won or lost (positions the export drops) within frame_drop_soft_nodes /
// frame_drop_hard_nodes when set. 0: off ("go nodes" is a hard limit alone).
// Options of the frame, not tuned (as PureSearch).
inline FUNCOMP_PARAM int frame_soft_nodes = 0;
inline FUNCOMP_PARAM int frame_soft_scale = 0;
inline FUNCOMP_PARAM int frame_drop_soft_nodes = 0;
inline FUNCOMP_PARAM int frame_drop_hard_nodes = 0;

// A layered network's first dense layer in float as trained (1), or in int8
// with AVX2 (0, the default; nnue.cpp quantizes it at load, a scale a row):
// the float path is the reference the int8 one is measured against (the
// author, 2026-10-18: the float layer cost a layered net about 3x a node).
inline FUNCOMP_PARAM int nnue_layer_float = 0;

// Internal iterative reduction: a PV or cut node with no hash move and at
// least iir_min_depth plies left is searched one ply shallower. Without a
// hash move its ordering is poor, and the shallower search fills in one
// for the next iteration.
inline FUNCOMP_PARAM int iir_min_depth = 6;
inline FUNCOMP_PARAM int iir_reduction = 2;  // plies

// Check extensions: a checking move is searched a ply deeper if its static
// exchange is at least check_ext_see (0: checks that don't lose material).
inline FUNCOMP_PARAM int check_ext_see = 171;

// Singular extensions: at depth >= se_min_depth, if the hash move has a
// lower-bound or exact score from a search at most se_tt_depth_margin plies
// shallower, the other moves are searched to (depth - 1) / 2 against
// tt_score - se_margin/16 * depth. If none reaches it, the hash move is
// extended by a ply; if that bound is at least beta anyway, the node cuts.
// At most se_max_extensions singular extensions on any path from the root.
// At non-PV nodes, a hash move so singular that the others fall short by
// more than se_double_margin as well is extended by two plies instead, at
// most se_max_doubles times on a path (these count as one singular
// extension each towards se_max_extensions too).
// When the other moves do reach the bound but it is below beta, a hash move
// with a score of at least beta, or at a cut node, is searched se_negative
// plies shallower instead (a negative extension).
inline FUNCOMP_PARAM int se_min_depth = 7;
inline FUNCOMP_PARAM int se_tt_depth_margin = 5;
inline FUNCOMP_PARAM int se_margin = 31;  // in 1/16 cp per ply: 2 cp
inline FUNCOMP_PARAM int se_max_extensions = 10;
inline FUNCOMP_PARAM int se_double_margin = 12;
inline FUNCOMP_PARAM int se_max_doubles = 4;
inline FUNCOMP_PARAM int se_negative = 2;
// The probe searches the other moves (depth - 1) * se_probe_depth_pct / 100
// plies deep, and extensions stop past se_ply_mult times the root depth.
inline FUNCOMP_PARAM int se_probe_depth_pct = 36;
inline FUNCOMP_PARAM int se_ply_mult = 4;
// In the probe, the left-out hash move doesn't count towards the other
// moves' indices (1), so they are reduced and pruned as the moves they
// are; 0 counts it, one step later for every move. On since 2.0 as a
// correctness fix: flat both times (-2.6 +- 7.5 Elo before the tune,
// +2.8 +- 7.2 after it, about 4,000 games each).
inline FUNCOMP_PARAM int se_probe_skip_index = 1;

// Fail-firm returns: a fail high resting on a static eval or a shallow
// search keeps only this percentage of its excess over beta (100: all of
// it, off): reverse futility pruning, the quiescence stand pat (also what
// is stored) and ProbCut. With ff_main_depth 1 the main search's fail
// high becomes (best * depth + beta) / (depth + 1). For after the first
// network (search addition D); all off. At 50/50/50/1 it lost -20.0 +-
// 9.7 Elo (2228 games): 93% of fail highs pulled in, and the lower stored
// bounds cut singular extensions (36% of probes against 44%) and
// alpha-side hash cuts (9% against 18%), which are tuned on the old ones.
inline FUNCOMP_PARAM int ff_rfp_pct = 70;
inline FUNCOMP_PARAM int ff_stand_pat_pct = 96;
inline FUNCOMP_PARAM int ff_probcut_pct = 89;
inline FUNCOMP_PARAM int ff_main_depth = 0;

// Reverse futility pruning (non-PV, depth <= rfp_max_depth): return the
// static eval if it beats beta by rfp_margin per ply, less rfp_improving
// when the eval is improving.
inline FUNCOMP_PARAM int rfp_max_depth = 6;
inline FUNCOMP_PARAM int rfp_margin = 36;
inline FUNCOMP_PARAM int rfp_improving = 41;
// At nodes flagged ttPv, the margin grows by rfp_tt_pv_margin: they may
// well be on the principal variation, where a static-eval cut is riskiest.
// 500 all but turns RFP off there; 0 treats them like any other node
// (as before the 2.0 tune; alone at 500 it leaned negative, -5.3 +- 7.4
// Elo, 4052 games).
inline FUNCOMP_PARAM int rfp_tt_pv_margin = 77;

// Razoring (non-PV, depth <= razor_max_depth): if the static eval is
// razor_base + razor_mult * depth^2 below alpha, drop into quiescence and
// trust it when it confirms the fail low.
inline FUNCOMP_PARAM int razor_max_depth = 3;
inline FUNCOMP_PARAM int razor_base = 204;
inline FUNCOMP_PARAM int razor_mult = 53;

// Null-move pruning (non-PV, depth >= nmp_min_depth, eval >= beta).
// Reduction R, in 1/1024 ply: nmp_base + depth * nmp_depth_mult
// + min((eval - beta) * nmp_eval_mult / 100000 plies, nmp_eval_max), summed
// and rounded down to whole plies once (nmp_round_once 1, merged for the
// 2.0 tuning, whose finer steps need it; undecided as a no-harm test at
// -2.3 +- 6.1 Elo), or each term rounded on its own (0, as in 1.0).
inline FUNCOMP_PARAM int nmp_min_depth = 4;
inline FUNCOMP_PARAM int nmp_base = 2979;
inline FUNCOMP_PARAM int nmp_depth_mult = 372;  // a ply per 3 of depth (exact below depth 128)
inline FUNCOMP_PARAM int nmp_eval_mult = 753;  // a ply per 200 cp
inline FUNCOMP_PARAM int nmp_eval_max = 1518;
inline FUNCOMP_PARAM int nmp_round_once = 1;

// ProbCut (non-PV, depth >= probcut_min_depth): captures whose static
// exchange covers the gap from the eval to beta + probcut_margin (less
// probcut_improving when improving) are searched probcut_reduction plies
// shallower against that raised bound, after quiescence agrees; one that
// beats it cuts the node. probcut_min_depth 64 turns it off.
inline FUNCOMP_PARAM int probcut_min_depth = 3;
inline FUNCOMP_PARAM int probcut_margin = 95;
inline FUNCOMP_PARAM int probcut_improving = 4;
inline FUNCOMP_PARAM int probcut_reduction = 3;
// ProbCut is skipped when the hash table has a score below the raised bound
// from at least depth - probcut_tt_margin; its cuts are stored at that depth.
inline FUNCOMP_PARAM int probcut_tt_margin = 7;

// Alpha-side hash cut (non-PV, depth >= alpha_tt_min_depth): a hash entry
// from a search at most alpha_tt_depth_margin plies shallower whose upper
// bound is at least alpha_tt_margin below alpha fails the node low at once.
// alpha_tt_min_depth 64 turns it off (as before the 2.0 tune; alone at 5
// it was flat, -0.7 +- 6.1 Elo, 6016 games).
inline FUNCOMP_PARAM int alpha_tt_min_depth = 7;
inline FUNCOMP_PARAM int alpha_tt_depth_margin = 2;
inline FUNCOMP_PARAM int alpha_tt_margin = 226;

// Late move pruning (depth <= lmp_max_depth): stop trying quiet moves
// after (lmp_base + lmp_mult * depth^2) / 100 of them, and lmp_not_improving_pct
// percent of that when the eval is not improving. Moves that give check are
// spared.
inline FUNCOMP_PARAM int lmp_max_depth = 9;
inline FUNCOMP_PARAM int lmp_base = 609;  // in 1/100 move
inline FUNCOMP_PARAM int lmp_not_improving_pct = 70;  // share of the count when not improving
inline FUNCOMP_PARAM int lmp_mult = 38;

// Once late move pruning has fired, the move picker drops the remaining
// quiet moves it would skip (1: sparing the gives_check test and the
// sorting of each) instead of handing them to the search to skip (0).
// The moves searched are the same; each later move counts, in its move
// index, the dropped moves that would have sorted before it. Accepted:
// +11.5 +- 10.6 Elo (+13-15% nps).
inline FUNCOMP_PARAM int lmp_in_picker = 1;

// Underpromotions, once a move has been searched (not at the root, not in
// check): rook and bishop promotions are skipped, and knight promotions
// unless they give check (1); 0 searches them all. A queen does at least
// as well but for stalemate tricks and knight forks. Accepted as marginal:
// +3.2 +- 6.0 Elo, 6012 games.
inline FUNCOMP_PARAM int underpromo_prune = 1;

// History pruning (lmr depth <= hist_prune_max_depth): skip a quiet move
// (not a check) whose combined history is below -hist_prune_mult * depth.
// hist_prune_max_depth -1 turns it off.
inline FUNCOMP_PARAM int hist_prune_max_depth = 7;
inline FUNCOMP_PARAM int hist_prune_mult = 1699;

// Futility pruning of quiet moves (reduced depth <= fp_max_depth): skip
// the move if static eval + fp_base + fp_mult * reduced depth <= alpha.
inline FUNCOMP_PARAM int fp_max_depth = 13;
inline FUNCOMP_PARAM int fp_base = 113;
inline FUNCOMP_PARAM int fp_mult = 141;

// SEE pruning (depth <= see_max_depth): skip captures losing more than
// see_capture_margin * depth, and quiets losing more than
// see_quiet_margin * reduced depth^2. Quiescence skips captures whose
// exchange falls below qs_see_threshold.
inline FUNCOMP_PARAM int see_max_depth = 7;
inline FUNCOMP_PARAM int see_capture_margin = 44;
inline FUNCOMP_PARAM int see_quiet_margin = 17;
inline FUNCOMP_PARAM int qs_see_threshold = -12;
// The piece values the search's static exchanges count, indexed by
// PieceType (king last, never captured): see_at_least and see_value read
// these, and everything else that counts material in exchange terms keeps
// SEE_VALUE (see.h), which they start equal to. The pawn stays at 100: it
// is the unit of every threshold above, and ProbCut compares an exchange
// with the eval. The hand-written values of 09-26 (320, 330, 500, 900)
// until 3.1's focused tune, which tuned them with the thresholds they meet.
// Swapped in alone, without those thresholds, other values can lose:
// FunHouse's network ratios scored -5.2 +- 7.2 Elo and shrank its tree 44%.
inline FUNCOMP_PARAM int see_piece_value[6] = {100, 303, 326, 514, 886, 0};
// Static exchanges value promotions (1): the promoting move gains the new
// piece less the pawn, and the new piece is what can be taken back; a pawn
// recapturing on the last rank promotes (to a queen) too. 0 treats every
// promotion as winning exactly 0. Accepted as a correctness fix (flat:
// -0.6 +- 8.0 Elo, 3424 games).
inline FUNCOMP_PARAM int see_promo = 1;
// A capture's SEE pruning threshold is shifted by its capture history *
// see_capt_hist_mult / 65536 centipawns (at most see_capture_margin * depth
// either way): a capture that has often worked may lose more on paper. 0
// turns the shift off. On since the 2.0 tune; alone, at 2048 (1/32) it tested flat (-0.8
// +- 6.4 Elo, 5510 games).
inline FUNCOMP_PARAM int see_capt_hist_mult = 988;
// Delta pruning in quiescence: skip a capture (not a check or promotion)
// when the stand-pat score plus the captured piece plus qs_delta_margin
// is still at most alpha. 32000 turns it off. On since the 2.0 tune; alone, at 200 it
// tested marginal (+2.6 +- 6.4 Elo, LOS 78.5%, 5546 games).
inline FUNCOMP_PARAM int qs_delta_margin = 693;
// In check, quiescence searches at most qs_quiet_evasions quiet evasions
// once a score that isn't mated is known (captures always). 256 turns
// the limit off.
inline FUNCOMP_PARAM int qs_quiet_evasions = 2;

// Late move reductions of quiet moves (and losing captures, see
// lmr_captures), from depth lmr_min_depth and move index lmr_min_moves
// (one later at PV nodes). The reduction is computed in 1/1024 ply
// (prune::LMR_SCALE) so that each term can be tuned finely, and truncated
// to whole plies at the end. Base reduction: lmr_base/100 + ln(depth) *
// ln(index) * lmr_mult/10^6 plies. Then, in 1/1024 ply: lmr_not_improving
// more when the eval isn't improving, lmr_cut_node more at cut nodes,
// lmr_pv_node less at PV nodes, lmr_killer less for killer moves, and
// history * lmr_hist_mult / 65536 less. Both scales are multipliers rather
// than divisors, so that SPSA's steps have roughly even effects. Adjustment
// terms (here and below) may be tuned below zero: SPSA may find that a term
// wants the opposite sign, e.g. killers reduced more rather than less.
inline FUNCOMP_PARAM int lmr_min_depth = 4;
inline FUNCOMP_PARAM int lmr_min_moves = 1;
inline FUNCOMP_PARAM int lmr_base = 138;
inline FUNCOMP_PARAM int lmr_mult = 240181;  // in millionths (hand value 444444 = 1/2.25)
inline FUNCOMP_PARAM int lmr_hist_mult = 11442;  // 1/1024 ply per 65536 of history (hand value 8192)
inline FUNCOMP_PARAM int lmr_not_improving = 1143;
inline FUNCOMP_PARAM int lmr_cut_node = 1190;
inline FUNCOMP_PARAM int lmr_pv_node = 754;
inline FUNCOMP_PARAM int lmr_killer = 2741;
// With a known threat (see StackEntry::threat): reduce moves that address
// it lmr_threat_evade/1024 plies less, and moves that don't
// lmr_threat_ignore/1024 more. On since the 2.0 tune; alone, with a ply each it tested
// neutral (-2.8 +- 10.1 Elo); kept as the hand-made baseline for a learned
// threat signal, and for SPSA.
inline FUNCOMP_PARAM int lmr_threat_evade = 428;
inline FUNCOMP_PARAM int lmr_threat_ignore = 1539;
// Quiet moves that take a piece out of a static threat (attacked by a
// cheaper piece) are reduced lmr_static_threat/1024 plies less. On since
// the 2.0 tune; alone, at a ply it leaned negative (stopped early).
inline FUNCOMP_PARAM int lmr_static_threat = 1320;
// Quiet replies to a quiet move are reduced lmr_parent_hist/1024 plies less
// when their quiet history is positive and the move they answer had a
// negative one when it was chosen, and that much more the other way round.
// Zero histories (untried moves) are neutral. 0 turns it off. Off: at a
// ply it lost -9.1 +- 6.7 Elo (4330 games), acting on 68% of those moves
// with the other statistics unchanged; left to the full SPSA pass, which
// sees a gradient at 0 (both signs change the search). The second full
// pass turned it on at 634 (0.6 ply), acting on about 73% of those moves.
inline FUNCOMP_PARAM int lmr_parent_hist = 714;
// Non-PV nodes whose hash entry says they were on the principal variation
// (ttPv) are reduced lmr_tt_pv/1024 plies less: they may well return to it.
inline FUNCOMP_PARAM int lmr_tt_pv = 565;
// When the hash move is a capture, quiet moves are reduced
// lmr_tt_capture/1024 plies more: the position's best line is likely
// tactical, and a quiet move is unlikely to beat it. At one ply it tested
// flat (-0.6 +- 6.5 Elo, 5558 games); the LMR SPSA moved it from 0 to 43,
// the 2.0 tune further.
inline FUNCOMP_PARAM int lmr_tt_capture = 1132;
// After a reduced search fails high, the full re-search goes a ply deeper
// when it landed more than lmr_deeper_base + lmr_deeper_mult * reduction
// centipawns above alpha, and a ply shallower when it beat the best
// score so far by less than lmr_shallower (0 turns that off: the score
// is above alpha, and alpha is at least the best score). lmr_deeper_base
// 32000 (beyond any mate score) turns deepening off. At 35 + 5r and 5 it
// lost (-15.1 +- 8.9 Elo, H0, 3010 games): deepening fired on 19% of reduced
// fail-highs (fail-soft scores land far above alpha). The LMR SPSA, started
// nearly off at 300, left it there (296): rare, for huge swings only. The
// second full pass took it to 175 + 14r and turned the shallower re-search
// on at 13.
inline FUNCOMP_PARAM int lmr_deeper_base = 256;
inline FUNCOMP_PARAM int lmr_deeper_mult = 16;
inline FUNCOMP_PARAM int lmr_shallower = 22;
// Child cutoffs: once lmr_child_cutoffs of a node's children have failed
// high in their own move loops (refuted by a searched move, not by the hash
// table or a static test), the node's later moves look likely to be refuted
// too: they are reduced lmr_child_cutoff_add/1024 plies more. 0 turns it off.
inline FUNCOMP_PARAM int lmr_child_cutoffs = 0;
inline FUNCOMP_PARAM int lmr_child_cutoff_add = 562;
// Hindsight depth: at a node its parent reduced by at least hindsight_red
// plies, the static evals of the two say how the parent's move turned out.
// Their sum (each from its own side) is what the move cost its side by
// static eval: nothing or less, the move held up better than its
// reduction assumed, and the node gets a ply back; more than
// hindsight_margin, the move looks as bad as assumed, and the node loses a
// ply (from depth 2 up). 0 turns it off.
inline FUNCOMP_PARAM int hindsight_red = 2;
inline FUNCOMP_PARAM int hindsight_margin = 74;
// Captures that lose material by static exchange (and neither check nor
// promote) are reduced like quiet moves when lmr_captures is 1, with
// lmr_bad_capture/1024 plies more (their history terms are zero). Other
// captures are never reduced. With the offset at 0 it tested flat (-0.9
// +- 6.5 Elo, 5514 games); the LMR SPSA (run with it on) set the offset to
// -574, just over half a ply less than a quiet move.
inline FUNCOMP_PARAM int lmr_captures = 1;
inline FUNCOMP_PARAM int lmr_bad_capture = -1878;

// Clamps on a learned guide's output (see pruning.h). Margins may move by
// guide_margin_pct percent of the hand value, but at least guide_margin_min
// centipawns; reductions by guide_reduction_delta plies; the late-move
// pruning count by guide_margin_pct percent (at least one move). Not
// tunables: they bound a model, they don't play chess.
inline FUNCOMP_PARAM int guide_margin_pct = 50;
inline FUNCOMP_PARAM int guide_margin_min = 50;
inline FUNCOMP_PARAM int guide_reduction_delta = 2;

// A parameter by its UCI name. `set` is null where this build has it as a
// constant: the constructor taking a pointer to const is the one chosen.
struct Tunable {
    const char* name;
    const int* value;
    int* set;
    int min, max;
    Tunable(const char* n, int* v, int lo, int hi) : name(n), value(v), set(v), min(lo), max(hi) {}
    Tunable(const char* n, const int* v, int lo, int hi) : name(n), value(v), set(nullptr), min(lo), max(hi) {}
};

inline const std::vector<Tunable>& tunables() {
    static const std::vector<Tunable> list = {
        {"TmStableStart", &tm_stable_start, 50, 300},
        {"TmStableStep", &tm_stable_step, 0, 50},
        {"TmStableMin", &tm_stable_min, 25, 150},
        {"TmDropScale", &tm_drop_scale, 0, 200},
        {"TmDropMax", &tm_drop_max, 0, 200},
        {"TmNodesBase", &tm_nodes_base, 50, 400},
        {"TmNodesSlope", &tm_nodes_slope, 0, 400},
        {"TmNodesMin", &tm_nodes_min, 10, 150},
        {"TmNodesMinDepth", &tm_nodes_min_depth, 1, 20},
        {"TmTbDtzMinMs", &tm_tb_dtz_min_ms, 0, 10000},
        {"TmMovesToGo", &tm_moves_to_go, 5, 100},
        {"TmIncPct", &tm_inc_pct, 0, 200},
        {"TmHardPct", &tm_hard_pct, 100, 1000},
        {"TmHardCapPct", &tm_hard_cap_pct, 10, 90},
        {"TmSoftPct", &tm_soft_pct, 10, 200},
        {"AspMinDepth", &asp_min_depth, 1, 16},
        {"AspDelta", &asp_delta, 1, 300},
        {"AspWidenPct", &asp_widen_pct, 10, 200},
        {"HistBonusMult", &hist_bonus_mult, 1, 256},
        {"HistBonusMax", &hist_bonus_max, 64, 8192},
        {"HistMax", &hist_max, 1024, 32767},
        {"EvalHistMult", &eval_hist_mult, 0, 512},
        {"EvalHistBonusMax", &eval_hist_bonus_max, 0, 8192},
        {"EvalHistMalusMax", &eval_hist_malus_max, 0, 8192},
        {"EvalHistMaxDepth", &eval_hist_max_depth, 0, 32},
        {"HistMalusMult", &hist_malus_mult, 1, 256},
        {"HistMalusMax", &hist_malus_max, 64, 8192},
        {"CaptBonusMult", &capt_bonus_mult, 1, 256},
        {"CaptBonusMax", &capt_bonus_max, 64, 8192},
        {"CaptMalusMult", &capt_malus_mult, 1, 256},
        {"CaptMalusMax", &capt_malus_max, 64, 8192},
        {"PriorCmPct", &prior_cm_pct, 0, 400},
        {"TtCutHistPct", &tt_cut_hist_pct, 0, 400},
        {"PriorCmFinalOnly", &prior_cm_final_only, 0, 1},
        {"PriorCmNodeGate", &prior_cm_node_gate, 0, 1},
        {"PriorCmIndexCap", &prior_cm_index_cap, 0, 64},
        {"CaptHistMult", &capt_hist_mult, 0, 1024},
        {"ContHist4Weight", &cont_hist4_weight, 0, 512},
        {"QuietCheckBonus", &quiet_check_bonus, 0, 32000},
        {"PawnHistWeight", &pawn_hist_weight, 0, 512},
        {"TtReplaceMargin", &tt_replace_margin, 0, 16},
        {"TtAgeWeight", &tt_age_weight, 0, 32},
        {"TtPvReplaceBonus", &tt_pv_replace_bonus, -4, 8},
        {"TtPvInheritDepth", &tt_pv_inherit_depth, 0, 64},
        {"CorrWeightMax", &corr_weight_max, 1, 128},
        {"CorrLimit", &corr_limit, 0, 1000},
        {"CorrNonPawnWeight", &corr_nonpawn_weight, 0, 512},
        {"CorrContWeight", &corr_cont_weight, 0, 512},
        {"CorrMarginPct", &corr_margin_pct, 0, 300},
        {"FiftyScale", &fifty_scale, 0, 1000},
        {"NnueScale", &nnue_scale, 30, 400},
        {"FiftyHashStart", &fifty_hash_start, 0, 100},
        {"FiftyHashWidth", &fifty_hash_width, 0, 50},
        {"InitScale", &init_scale, 0, 400},
        {"InitClamp", &init_clamp, 0, 1000},
        {"InitTempo", &init_tempo, 0, 60},
        {"InitMobility", &init_mobility, 0, 48},
        {"InitTerritory", &init_territory, 0, 64},
        {"InitCoordination", &init_coordination, 0, 30},
        {"InitPushThreat", &init_push_threat, 0, 50},
        {"InitSafeCheck", &init_safe_check, 0, 50},
        {"InitKingPressure", &init_king_pressure, 0, 128},
        {"InitThreatLatent", &init_threat_latent, 0, 96},
        {"InitThreatPending", &init_threat_pending, 0, 128},
        {"InitThreatDouble", &init_threat_double, 0, 256},
        {"InitEndgamePct", &init_endgame_pct, 0, 200},
        {"MopEdge", &mop_edge, 0, 100},
        {"MopKings", &mop_kings, 0, 100},
        {"DrawOcbPct", &draw_ocb_pct, 0, 100},
        {"DrawPawnlessPct", &draw_pawnless_pct, 0, 100},
        {"DrawPawnlessMargin", &draw_pawnless_margin, 0, 1000},
        {"CorrUnscaleFifty", &corr_unscale_fifty, 0, 1},
        {"RepInReach", &rep_in_reach, 0, 1},
        {"PureSearch", &pure_search, 0, 2},
        {"FrameSoftNodes", &frame_soft_nodes, 0, 10000000},
        {"FrameSoftScale", &frame_soft_scale, 0, 1},
        {"FrameDropSoftNodes", &frame_drop_soft_nodes, 0, 10000000},
        {"FrameDropHardNodes", &frame_drop_hard_nodes, 0, 80000000},
        {"NnueLayerFloat", &nnue_layer_float, 0, 1},
        {"IirMinDepth", &iir_min_depth, 2, 32},
        {"IirReduction", &iir_reduction, 0, 3},
        {"CheckExtSee", &check_ext_see, -1000, 500},
        {"SeMinDepth", &se_min_depth, 4, 32},
        {"SeTtDepthMargin", &se_tt_depth_margin, 0, 8},
        {"SeMargin", &se_margin, 0, 256},
        {"SeMaxExtensions", &se_max_extensions, 0, 16},
        {"SeDoubleMargin", &se_double_margin, 0, 300},
        {"SeMaxDoubles", &se_max_doubles, 0, 32},
        {"SeNegative", &se_negative, 0, 3},
        {"SeProbeDepthPct", &se_probe_depth_pct, 25, 75},
        {"SePlyMult", &se_ply_mult, 1, 4},
        {"SeProbeSkipIndex", &se_probe_skip_index, 0, 1},
        {"FfRfpPct", &ff_rfp_pct, 0, 100},
        {"FfStandPatPct", &ff_stand_pat_pct, 0, 100},
        {"FfProbCutPct", &ff_probcut_pct, 0, 100},
        {"FfMainDepth", &ff_main_depth, 0, 1},
        {"RfpMaxDepth", &rfp_max_depth, 0, 16},
        {"RfpMargin", &rfp_margin, 10, 300},
        {"RfpImproving", &rfp_improving, -100, 200},
        {"RfpTtPvMargin", &rfp_tt_pv_margin, -100, 1000},
        {"RazorMaxDepth", &razor_max_depth, 0, 8},
        {"RazorBase", &razor_base, 0, 1000},
        {"RazorMult", &razor_mult, 0, 1000},
        {"NmpMinDepth", &nmp_min_depth, 1, 16},
        {"NmpBase", &nmp_base, 0, 8192},
        {"NmpDepthMult", &nmp_depth_mult, 0, 1024},
        {"NmpEvalMult", &nmp_eval_mult, 100, 4000},
        {"NmpEvalMax", &nmp_eval_max, 0, 8192},
        {"NmpRoundOnce", &nmp_round_once, 0, 1},
        {"ProbCutMinDepth", &probcut_min_depth, 2, 64},
        {"ProbCutMargin", &probcut_margin, 0, 500},
        {"ProbCutImproving", &probcut_improving, -200, 200},
        {"ProbCutReduction", &probcut_reduction, 2, 8},
        {"ProbCutTtMargin", &probcut_tt_margin, 0, 8},
        {"AlphaTtMinDepth", &alpha_tt_min_depth, 2, 64},
        {"AlphaTtDepthMargin", &alpha_tt_depth_margin, 1, 8},
        {"AlphaTtMargin", &alpha_tt_margin, 0, 500},
        {"LmpMaxDepth", &lmp_max_depth, 0, 16},
        {"LmpBase", &lmp_base, 100, 2000},
        {"LmpNotImprovingPct", &lmp_not_improving_pct, 10, 200},
        {"LmpMult", &lmp_mult, 10, 400},
        {"LmpInPicker", &lmp_in_picker, 0, 1},
        {"UnderpromoPrune", &underpromo_prune, 0, 1},
        {"HistPruneMaxDepth", &hist_prune_max_depth, -1, 16},
        {"HistPruneMult", &hist_prune_mult, 0, 20000},
        {"FpMaxDepth", &fp_max_depth, 0, 16},
        {"FpBase", &fp_base, 0, 500},
        {"FpMult", &fp_mult, 10, 400},
        {"SeeMaxDepth", &see_max_depth, 0, 16},
        {"SeeCaptureMargin", &see_capture_margin, 0, 400},
        {"SeeQuietMargin", &see_quiet_margin, 0, 200},
        {"QsSeeThreshold", &qs_see_threshold, -500, 200},
        {"SeeKnight", &see_piece_value[1], 200, 500},
        {"SeeBishop", &see_piece_value[2], 200, 500},
        {"SeeRook", &see_piece_value[3], 350, 900},
        {"SeeQueen", &see_piece_value[4], 600, 1600},
        {"SeePromo", &see_promo, 0, 1},
        {"SeeCaptHistMult", &see_capt_hist_mult, 0, 8192},
        {"QsDeltaMargin", &qs_delta_margin, 0, 32000},
        {"QsQuietEvasions", &qs_quiet_evasions, 0, 256},
        {"LmrMinDepth", &lmr_min_depth, 1, 16},
        {"LmrMinMoves", &lmr_min_moves, 1, 16},
        {"LmrBase", &lmr_base, 0, 300},
        {"LmrMult", &lmr_mult, 150000, 1000000},
        {"LmrHistMult", &lmr_hist_mult, 0, 65536},
        {"LmrNotImproving", &lmr_not_improving, -2048, 3072},
        {"LmrCutNode", &lmr_cut_node, -2048, 3072},
        {"LmrPvNode", &lmr_pv_node, -2048, 3072},
        {"LmrKiller", &lmr_killer, -2048, 3072},
        {"LmrThreatEvade", &lmr_threat_evade, -2048, 4096},
        {"LmrThreatIgnore", &lmr_threat_ignore, -2048, 4096},
        {"LmrStaticThreat", &lmr_static_threat, -2048, 4096},
        {"LmrParentHist", &lmr_parent_hist, -2048, 4096},
        {"LmrTtPv", &lmr_tt_pv, -2048, 4096},
        {"LmrTtCapture", &lmr_tt_capture, -2048, 3072},
        {"LmrDeeperBase", &lmr_deeper_base, 0, 32000},
        {"LmrDeeperMult", &lmr_deeper_mult, 0, 50},
        {"LmrShallower", &lmr_shallower, 0, 100},
        {"LmrChildCutoffs", &lmr_child_cutoffs, 0, 16},
        {"LmrChildCutoffAdd", &lmr_child_cutoff_add, -2048, 4096},
        {"HindsightRed", &hindsight_red, 0, 16},
        {"HindsightMargin", &hindsight_margin, 0, 1000},
        {"LmrCaptures", &lmr_captures, 0, 1},
        {"LmrBadCapture", &lmr_bad_capture, -4096, 3072},
    };
    return list;
}

}  // namespace params
