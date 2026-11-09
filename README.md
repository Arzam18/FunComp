# FunComp

A UCI chess engine written in C++23 from the ground up: a neural network
evaluation (NNUE) trained only on FunComp's own games, and a modern
alpha-beta search in which every feature was proven in games before it
was kept, then tuned as a whole. Version 3.1 is estimated at about
**3660 on the CCRL Blitz scale** (3662 ± 8 from a single-threaded
gauntlet of 2,000 games against 20 rated engines; an estimate on my
hardware, not an official CCRL rating), about 20 above version 3.0.
It searches with as many threads as you give it, on x86-64 and on
64-bit ARM (Android).

## Features

- **Board:** bitboards with magic sliders, legal move generation,
  make/unmake, Zobrist hashing; move generation verified by perft.
- **Evaluation:** an NNUE built into the executable: 768 piece-square
  inputs for each of 7 king buckets (by the king's square, mirrored so
  the king stands on the queen's side), a 1,024-wide hidden layer per
  side (clipped, its halves multiplied pairwise), then two small layers,
  1,024 → 32 → 32 → 1, each chosen by one of 8 output buckets by material.
  The hidden layer is updated incrementally with each move, and a cache of
  each king bucket's last accumulator spares a rebuild when the king
  changes bucket. Correction history (by pawn structure, by each side's
  non-pawn material, and by the previous move) learns during the search
  how far the static eval tends to be off and corrects it; scores shrink
  toward a draw as the fifty-move counter runs.
- **Training:** the network learned from 4.7 billion positions of
  FunComp's own self-play in five rounds, each labelled by FunComp's own
  search (the first with 2.0's hand-written evaluation, the later ones
  with an earlier round's network) and blended with the games' results.
  Openings came from random plies, Chess324 starts, and positions
  from the Lichess open database (as starting positions only). No other
  engine's games, evaluations or networks were used.
- **The hand-written evaluation** of 2.0 is still there (`EvalFile`
  `<empty>`): PeSTO's tapered piece-square tables, and the two terms
  below.
- **Initiative:** a term for who is making the other side respond:
  threats pending against the side to move and threats it holds, safe
  mobility, territory, coordination, pawn-push threats, safe checks,
  king pressure, and the move itself. Colour-symmetric, clamped, and
  tuned with the search.
- **Drawishness:** the eval is scaled toward a draw for opposite-coloured
  bishops alone and for pawnless endings with a small material gap; a
  bare king is driven to the edge (and, against bishop and knight, to
  the bishop's corner).
- **Search:** iterative deepening, principal variation search with
  aspiration windows, quiescence search with SEE pruning.
- **Parallel search:** Lazy SMP; threads share a lockless transposition
  table.
- **Transposition table:** four-entry buckets with aging; entries
  remember whether the position was on a principal variation.
- **Move ordering:** a staged picker: hash move, winning captures (by
  SEE, victim and capture history), killers, quiet moves (butterfly and
  two continuation histories), losing captures last.
- **Pruning and reductions:** reverse futility, razoring, null move,
  ProbCut, late move pruning, futility, history and SEE pruning, late
  move reductions in fractions of a ply (with deeper and shallower
  re-searches), internal iterative reductions, mate distance pruning.
- **Extensions:** singular extensions (double and negative, with
  multi-cut), and checks that don't lose material.
- **Draws:** repetitions and the fifty-move rule (checkmate on the
  hundredth ply still counts), and detection of a repetition one move
  away.
- **Time management:** soft and hard limits, scaled by how settled the
  search looks: how long the best move has held, whether the score
  dropped, and how much of the search went into the best move. Pondering
  is supported.
- **Endgame tablebases:** Syzygy WDL probes in the search and DTZ
  ranking at the root (through Fathom), so won endings are converted
  within the fifty-move rule.

Every change was tested in games (an SPRT against the previous best
version) before it was kept. For 2.0, all of the search's parameters
(about 110) were then tuned together by SPSA in two full passes, and the
time management in a pass of its own; for 3.0 the search was tuned again
for the network (2,000 iterations, 108 parameters), and for 3.1 the
margins nearest the evaluation for the new one. Ideas that did not pay
are still in the code, switched off. Every margin
and threshold lives in `src/params.h`, and the search reaches them through
`src/pruning.h`, which is built so that a learned model can later guide
the pruning (see "Learned pruning" below).

## Using FunComp

FunComp speaks UCI: load it in any UCI GUI (Arena, Cute Chess, BanksiaGUI,
En Croissant, ...) or match runner (fastchess, cutechess-cli). The release
has two Windows binaries and two for Android, each with the network
inside:

- `funcomp-3.1-x86-64-v3.exe` for CPUs with AVX2 and BMI2 (Intel
  Haswell, AMD Zen and later): the faster one;
- `funcomp-3.1-x86-64.exe` for any 64-bit x86 CPU;
- `funcomp-3.1-android-armv8-dotprod` for 64-bit ARM CPUs with the dot
  product instructions (most phones since about 2018): the faster one;
- `funcomp-3.1-android-armv8` for any 64-bit ARM CPU (Android 7 or
  later).

Options:

- `EvalFile` (default `<embedded>`): the network built into the binary;
  a path to a `.nnue` file loads another, `<empty>` the hand-written
  evaluation.

- `Hash` (MB, default 64, 1-4096): transposition table size.
- `Threads` (default 1, 1-256): search threads (Lazy SMP: they share
  the transposition table).
- `Ponder` (default false): the GUI may let FunComp think on the
  opponent's time (`go ponder`, `ponderhit`); `bestmove` always names the
  move it expects in reply.
- `TelemetryDir` (default `<empty>`): for testing; each engine process
  writes its pruning and feature counters over the session to a file of
  its own in this directory when it quits (`tools/telemetry-sum.sh`
  adds them up).
- `Move Overhead` (ms, default 30, 0-5000): time kept in reserve per move
  for GUI and network latency.
- `SyzygyPath` (default `<empty>`): directories holding Syzygy tables,
  separated by `;` on Windows and `:` elsewhere.
- `SyzygyProbeLimit` (pieces, default 7, 0-7): probe positions with at
  most this many pieces (and no more than the tables hold).
- `SyzygyProbeDepth` (default 1, 1-100): at the largest piece count,
  probe only at this depth or more (for tables on a slow disk).

Beyond UCI, from a console:

```text
d                     show the board
eval                  the static eval, with the initiative term broken
                      down
perft <depth>         count leaf nodes
bench [depth]         search a fixed position set; prints pruning
                      statistics, total nodes and nps
                      (also from the command line: funcomp bench)
pstats                pruning statistics since the last "pstats reset"
pstats reset
genlabels ...         write labels for the learned pruning guide (below)
genbingo out=<file> [plies=N perstart=N nodes=N maxscore=CP seed=N hash=MB]
                      Chess324 openings (Larry Kaufman's variant): kings,
                      rooks and pawns at home, queen, opposite-coloured
                      bishops and knights shuffled over b, c, d, f and g,
                      each side independently (324 starts)
genconlog ...         continuation logs (research builds only, configured
                      with -DCONT_LOG=ON; see src/cont_log.h)
gendata out=<file> ...
                      self-play training data: positions with their
                      search scores and the games' results (src/datagen.h
                      lists the options)
datatext in=<data> out=<file> ...
                      training text from it, with the export's filters
datastats in=<data>   totals and distributions of a data file
telemetry in=<pgn> [positions=N nodes=N verify=N minply=N seed=N]
                      pruning and feature counters, with verification,
                      on positions sampled from the games of a PGN file
```

The bench node count is the search's signature (1031103 for 3.1 with its
network on x86-64-v3 and ARM, 1094101 on x86-64 and x86-64-v2; 1187868
for 3.0; 1614718 for 2.0): a
change meant to alter the search changes it, and a refactor or speedup
must not.

## Build

On Windows with MSYS2 (UCRT64), or on Linux with a C++23 compiler and
CMake:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

For builds that run on other machines, `-DARCH=` sets the CPU level:
`x86-64-v2` (the default) needs POPCNT and SSE4.2, which every x86-64
CPU since about 2009 has; `x86-64-v3` needs AVX2 and BMI2 and is
faster. The network's later layers run in plain C++ on `x86-64` and
`x86-64-v2`, which rounds them differently, so those builds have their
own bench count (`BENCH_NODES_EMBEDDED_PLAIN`). `-DARCH=x86-64` runs on
any 64-bit x86 CPU but is about 5% slower, having no popcount
instruction; `-DNATIVE=ON` targets the build machine.

```bash
cmake -S . -B build-v2 -DARCH=x86-64-v2
cmake -S . -B build-v3 -DARCH=x86-64-v3
```

On 64-bit ARM, `-DARCH=armv8-a` or `-DARCH=armv8.2-a+dotprod` (NEON,
with the int8 dot product); these search exactly as `x86-64-v3`.
`tools/release-android.sh` builds both for Android with the NDK (and
checks their bench under qemu when given one).

The network `nets/funcomp-3.1.nnue` is compiled into the executable;
`-DEMBED_NET=<file>` embeds another, `-DEMBED_NET=` none (then only
`EvalFile` loads one).

`-DTUNE=ON` advertises every search parameter as a UCI option (and
`PruneVerify`, below), for tuning and for tests that change parameters.

## Testing tools

The scripts in `tools/` (PowerShell, plus awk) drive
[fastchess](https://github.com/Disservin/fastchess); they need
`fastchess` on PATH or `-Fastchess <path>`, and opening books (UHO books
for SPRTs, a balanced book such as 8moves_v3 for ratings).

- **`sprt.ps1`**: a sequential probability ratio test of a new build
  against a base, which stops as soon as the result is statistically
  clear. Default bounds `-Elo0 0 -Elo1 5` test for a gain; `-Elo0 -5
  -Elo1 0` for a change that only has to do no harm. `-NewOptions` and
  `-BaseOptions` set parameters on one side (TUNE builds only; the script
  checks).

  ```powershell
  tools\sprt.ps1 -New build\chess.exe -Base base.exe -Book books\UHO_Lichess_4852_v1.epd
  ```

- **`spsa.ps1`**: tunes parameters by SPSA (OpenBench's schedules) with a
  TUNE build playing itself. Parameters, ranges and step sizes come from
  a CSV (`tools/spsa/`); progress goes to a state CSV that also resumes a
  run. Values are rounded stochastically, so integer parameters get a
  gradient too.

  ```powershell
  tools\spsa.ps1 -Engine build-tune\chess.exe -Params tools\spsa\lmr.csv -Iterations 1000 -Book books\UHO_Lichess_4852_v1.epd -State spsa-lmr.csv
  ```

- **`stress.ps1`**: many fast self-play games at once, then a report of
  how games ended, any time losses, crashes, disconnects or illegal
  moves, and how many plies wins took from +10 pawns to mate.
- **`gauntlet.ps1`** and **`rating.awk`**: games against rated opponents
  from a balanced book, one game per physical core (hyper-threading would
  bias an absolute result), and one rating fitted to the PGN. The
  gauntlets' opponents and their CCRL Blitz ratings are in
  `tools/gauntlet-1.0.txt` and `tools/gauntlet-2.0.txt`, one per line as
  `file=rating`, optionally with `;name=...` (the name in the PGN),
  `;dir=...` (a working directory) and `;option.Name=value`;
  `tools/opponents.sha256` has the checksums of the opponents' binaries
  and source archives.

  ```powershell
  tools\gauntlet.ps1 -Engine funcomp.exe -Opponents (Get-Content tools\gauntlet-2.0.txt) -Rounds 60 -Pgn gauntlet.pgn
  ```

  ```bash
  awk -v ENGINE=FunComp -v RATINGS="stash-25.0=2933,stash-27.0=3049" -f tools/rating.awk gauntlet.pgn
  ```

- **`similarity.ps1`**: the move-similarity (clone) check: how often
  engines choose the same best move on a sample of positions.
- **`wdl.awk`**: the eval's scale, as win and draw rates by reported
  score, from self-play PGNs.
- **`telemetry-sum.sh`**: adds up the counter files that `TelemetryDir`
  collects during a match, side by side for two builds.
- **`convert.sh`**: endgame conversion, which match adjudication hides:
  an engine plays both sides of won positions (`tools/bare-king.epd`,
  `tools/tb-wins.epd`) and the plies to mate are counted.
- **`scaling.sh`**: node rate and depth by thread count.
- **`research/`**: Python scripts (numpy, scipy) for the learned-pruning
  studies, on `genlabels` and `genconlog` output.

## Learned pruning

The long-term plan: replace hand-written margins with a model that
predicts, from the node's depth, static eval and features, how far the
searched score can land from the static eval. In place today:

1. **Hooks.** `prune::Context` (in `src/pruning.h`) holds the depth, ply,
   static eval, alpha, beta, improving flag, node type (PV, cut or all),
   and for move-level decisions the move, its index, reduced depth,
   history, killer and check flags, plus a slot for a feature vector.
   Setting `prune::guide` to a function installs a model; its answers are
   clamped by the `guide_*` parameters in `params.h`.
2. **Statistics.** `pstats` and `bench` show how often each heuristic is
   tried and fires. In a TUNE build, `PruneVerify` N (0 = off) checks 1 in
   N pruning decisions by searching anyway and counts how often the
   pruning was wrong (for diagnosis only: the checks share the hash table
   and histories with the real search, so they alter it).
3. **Labels.** `genlabels` samples nodes from real searches, records the
   search's context at each, and labels each node with the score of a
   search of it, on its own, to the depth it had in the tree:

   ```text
   genlabels out=labels.csv games=200
   genlabels out=labels.csv in=positions.epd depth=12
   ```

   Options: `out=` (required), `in=` (FEN/EPD file; self-play if absent),
   `games=`, `randomplies=` (random opening moves), `depth=` (driver
   search depth, default 10), `sample=` (1 in N depth-1 nodes, default
   256), `depthscale=` (percent: each ply deeper is sampled this much more
   often, default 200, which roughly evens out the depths), `maxdepth=`
   (deepest node depth sampled, default 8), `count=` (row limit),
   `seed=`, `hash=` (driver, MB), `labelhash=` (labeller, MB; cleared
   before every label so labels don't depend on processing order),
   `labelnodes=` (a node cap per labelling search; rows it stops are
   marked censored).

   Columns: `fen, depth, ply, static_eval, correction, search_eval, alpha,
   beta, improving, node, under_null, iir, score, mate`, then the hand
   features of `src/guide_features.h` (phase, material, threatened pieces,
   best safe capture, king attackers, passed pawns, mobility, halfmove
   clock). Scores are from the side to move's view. `static_eval` includes
   correction history, and `correction` is the part that came from it.
   `search_eval` is what pruning decided with: the static eval, or the hash
   score where its bound made it a sharper estimate. `alpha`/`beta` are
   the node's window in the tree, `node` is `pv`, `cut` or `all`. `depth`
   is what node-level pruning saw; `iir` is 1 if the node's moves were then
   searched a ply shallower. Mate scores are clipped to +-2000 and flagged
   in `mate`. Then the columns for a cost model: `group` (the game or
   input line, for splitting data), `rate` (the depth's sampling rate),
   the hash entry the node found (`tt_hit, tt_depth, tt_bound`), what the
   node cost in the tree (`tree_nodes`, and `reached_moves`: 0 if
   node-level pruning ended it), and what the labelling search cost from
   a cold start (`fresh_nodes`, `censored`, and per iteration
   `fresh_curve` and `score_curve`, `;`-separated). The run's options,
   git commit and search parameters are written to `<out>.params`.

## How FunComp was written

FunComp was written with AI tools, specifically Claude Code. Writing
software today is not what it was even a few years ago, and that
deserves saying plainly.

The hypotheses are mine. The roadmap is mine. The testing pipeline is
mine. I've worked to make every piece of code original work, guided by
me. But the people who came before me worked their ideas out by hand,
and I have not. I started in computer chess around 1995, and the first
move generator I ever wrote was in Perl, so I know the difference AI
has made. Writing every line by hand is a craft I haven't spent the
years to hone, and I don't claim the craftsmanship that so many in this
hobby have earned.

What I do have is ideas. Claude helps me turn them into code, test
them, and move through the engineering far faster than I could alone.
My hope is to contribute something genuine to the chess engine
community: alternate approaches to modern problems, proven by results.
I couldn't do that without these tools. And after all, this is FunComp.

## Credits

FunComp's code is its own, but its ideas mostly are not: they come from
decades of published work by the chess-programming community.

- **Influences.** The Chess Programming Wiki was the main reference
  throughout. Stockfish influenced FunComp's development more than any
  other engine: many of the search techniques here (singular and negative
  extensions, ProbCut, continuation and correction histories, the ttPv
  flag, upcoming-repetition detection) are ideas I learned from it and
  from other open-source engines. The implementations, their structure
  and their constants are FunComp's own; nothing is copied from another
  engine.
- **Evaluation.** The piece-square tables and material values are
  PeSTO's, by Ronald Friederich, as published on the Chess Programming
  Wiki.
- **NNUE.** The efficiently updatable network was invented by Yu Nasu
  for shogi; Stockfish brought it to chess, and its use there influenced
  FunComp's (3.1's later layers among them: multiplying the hidden
  layer's halves pairwise, and skipping its zero blocks in the first
  small layer, with the hidden neurons reordered so that more blocks are
  zero). FunComp's network, its features, its data and its inference
  code are its own. The trainer is built on bullet by Jamie Whiting (MIT
  licence). Starting positions for some training games came from the
  Lichess open database (CC0); others are Chess324, Larry Kaufman's
  variant (2022), which FunComp's opening generator arrived at
  independently before I learned it already existed.
- **Tablebases.** Syzygy tables and their original probing code are by
  Ronald de Man; FunComp reads them through Fathom (basil00, Jon Dart;
  MIT licence, in `third_party/fathom` with its licence file).
- **Published algorithms.** Static exchange evaluation uses the wiki's
  gain-array swap algorithm; sliders use magic bitboards; move ordering
  values pieces as in Tomasz Michniewski's Simplified Evaluation
  Function; the sampling generator is Sebastiano Vigna's xorshift64*.
- **Testing.** Games are run with fastchess, SPRTs from the UHO opening
  books by Stefan Pohl; the SPSA driver follows OpenBench's schedule. The
  rating gauntlets are played against engines generously published by
  their authors: Stash (Morgan Houppin), Booot (Alex Morozov), Reckless,
  Akimbo (Jamie Whiting), RubiChess (Andreas Matthies), Seer (Connor
  McMonigle), Igel (V. Medvedev, V. Shcherbyna), Clover (Luca
  Metehau), PlentyChess (Yoshie2000) and Stockfish (the Stockfish
  developers).

## License

FunComp is free software under the GNU General Public License, version 3:
see `LICENSE`.
