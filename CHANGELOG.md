# Changelog

## FunComp 3.1 (2026-10-29)

**Strength.** About **3660 on the CCRL Blitz scale**: 3662 ± 8 from a
single-threaded gauntlet of 2,000 games at 10+0.1 against the 20 rated
engines of 3.0's (CCRL Blitz 3492-3793), about 20 above 3.0 (3642 ± 8).
Against 3.0 directly: +23 ± 9 Elo in 1,000 self-play games from balanced
openings (8+0.08: 144 wins, 779 draws, 77 losses). An estimate on our
hardware, not an official CCRL rating.

**Binaries.**
- `funcomp-3.1-x86-64-v3.exe`: for CPUs with AVX2 and BMI2 (Intel
  Haswell, AMD Zen and later). The faster x86 build.
- `funcomp-3.1-x86-64.exe`: for any 64-bit x86 CPU.
- `funcomp-3.1-android-armv8-dotprod`: for 64-bit ARM CPUs with the dot
  product instructions (most phones since about 2018). The faster ARM
  build.
- `funcomp-3.1-android-armv8`: for any 64-bit ARM CPU (arm64-v8a).

The Android builds run on Android 7 or later. All are statically linked
and carry the network inside: nothing to download or set. Bench: 1031103
nodes for x86-64-v3 and both ARM builds, 1094101 for x86-64 (whose later
layers run in plain C++ and round differently). The UCI options are 3.0's.

**What's new.**
- *Android and ARM:* builds for 64-bit ARM with NEON, computing exactly
  what the x86-64-v3 build does.
- *A new network,* trained on 4.7 billion positions of FunComp's own games
  (five rounds of self-play; 3.0's had three and 1.44 billion), with two
  small layers after the first: 1,024 → 32 → 32 → 1.
- *Code cleanup and optimization,* and the search re-tuned for the new
  network.

**Checked before release.**
- A stress run of 8,000 games on both x86 builds (bullet, 1 s + 0.01 s,
  and 40 moves in 10 s, 24 at a time): no crashes, disconnects or illegal
  moves; no time losses at 40 moves in 10 s (see the limitations for
  bullet).
- Threads with tablebases in bullet (4 threads, Syzygy, 2+0.02, five
  games at a time), 2,000 games: no crashes, disconnects or illegal moves.
- No time losses in the 3,000 timed games of the gauntlet and the match
  against 3.0.
- The Android builds give the x86-64-v3 build's bench under emulation
  (their speed on a phone is not measured here).

**Known limitations.**
- No Chess960.
- In bullet with every hardware thread busy, about 1 game in 1,000 was
  lost on time (6 of 6,000 at 1 s + 0.01 s and 2 s + 0.02 s; 3.0: about 3
  in 1,000), by 2 to 30 ms, most likely to the operating system's
  scheduling: raise `Move Overhead` (default 30 ms) in such setups.

## FunComp 3.0 (2026-10-20)

The first FunComp with a neural network evaluation (NNUE), trained only on
FunComp's own games.

**Strength.** About **3640 on the CCRL Blitz scale**: 3642 ± 8 from a
single-threaded gauntlet of 2,000 games at 10+0.1 against 20 rated engines
(CCRL Blitz 3492-3793), about 360 above 2.0 measured the same way (3282 ± 10). Against 2.0
directly: +484 ± 33 Elo in 1,000 self-play games from balanced openings
(8+0.08: 888 wins, 108 draws, 4 losses). An
estimate on our hardware, not an official CCRL rating.

**Binaries.**
- `funcomp-3.0-x86-64-v3.exe`: for CPUs with AVX2 and BMI2 (Intel
  Haswell, AMD Zen and later). The faster build.
- `funcomp-3.0-x86-64.exe`: for any 64-bit x86 CPU.

Both search identically (bench 1187868 nodes), are statically linked, and
carry the network inside the executable: nothing to download or set.

**New UCI option.** `EvalFile` (default `<embedded>`: the network built
into the binary). A path loads another network; `<empty>` switches to the
hand-written evaluation of 2.0.

**What's new.**
- *The network.* 768 piece-square inputs for each of 7 king buckets (by
  the king's square, mirrored so the king stands on the queen's side), a
  1,024-wide hidden layer for each side to move, squared clipped ReLU, and
  8 output buckets by material. Updated incrementally move by move, with a
  cache of each king bucket's last accumulator, so a king move doesn't
  rebuild it from scratch.
- *Its data.* 1.44 billion positions from FunComp's own self-play in
  three rounds, each labelled by FunComp's own search (round 1 with 2.0's
  hand-written evaluation, rounds 2 and 3 with the network of the round
  before), blended with the games' results. Openings: random plies from
  the standard position, Chess324 starts (Larry Kaufman's variant), and
  (round 3) positions from the Lichess open database (CC0), used as
  starting positions only.
  No other engine's games, evaluations or networks were used.
- *Its training.* Our own trainer on the bullet library (GPU), 120
  passes of 100 million positions. The width, king buckets, later layers
  and input features were chosen by games: matches at fixed nodes against
  a reference network, then timed matches, then a round robin of the three
  finalists at 2+1 (a tie; the simplest and fastest, without later layers,
  ships).
- *Search.* Two features carried over from the 2.0-era tuning branch
  and kept after tests: correction history by the previous move
  (continuation correction), and hindsight depth (a node whose parent
  was reduced gets a ply back if the parent's move cost nothing, or one
  less if it cost a lot). The whole search re-tuned by SPSA for the
  network (2,000 iterations, 108 parameters).
- *Speed.* The network's output in AVX2: about 2.7 million nodes a second
  on one thread of a Ryzen 9 5900X for the x86-64-v3 build (bench), about
  1.5 million for the portable x86-64 build; 2.0's hand-written
  evaluation searched about 3.6 million.

**Checked before release.**
- A stress run of 8,000 games on both builds (bullet, 1 s + 0.01 s, and 40
  moves in 10 s, 24 at a time): no crashes, disconnects or illegal moves;
  no time losses at 40 moves in 10 s (see the limitations for bullet).
- Threads with tablebases in bullet (4 threads, Syzygy, 2+0.02, five
  games at a time), 2,000 games: no failures.
- No time losses in the 5,000 timed games of the gauntlet, the tune's
  confirmation and the match against 2.0.

**Known limitations.**
- No Chess960.
- In bullet with every hardware thread busy (1 s + 0.01 s, 24 games on
  24 hardware threads), about 3 games in 1,000 were lost on time, by up
  to 61 ms, most likely to the operating system's scheduling: raise
  `Move Overhead` (default 30 ms) in such setups.

**Next.** Round 4's data (in generation: better labels from a deeper
search and a second, backward pass over each game; more positions), then
a network chosen by forecasts from the learning curves, and the learned
guide for the search's pruning.

## FunComp 2.0 (2026-10-02)

The complete classical engine: multi-threaded, with tablebases and
pondering, a richer evaluation, and every parameter tuned.

**Strength.** About **3280 on the CCRL Blitz scale**: 3282 ± 10 from a
single-threaded gauntlet of 2,880 games at 10+0.1 against 24 rated
engines from eight families (Stash, Booot, Reckless, Akimbo, RubiChess,
Seer, Igel, Clover; CCRL Blitz 3049-3419), about 245 above 1.0 measured
the same way. Against 1.0 directly: +306 ± 23 in 1,000 self-play games
from balanced openings at 8+0.08. An estimate on
our hardware, not an official CCRL rating.

**Binaries.**
- `funcomp-2.0-x86-64-v3.exe`: for CPUs with AVX2 and BMI2 (Intel
  Haswell, AMD Zen and later). The faster build.
- `funcomp-2.0-x86-64.exe`: for any 64-bit x86 CPU.

Both search identically (bench 1614718 nodes) and are statically linked.

**New UCI options.** `Threads` (Lazy SMP, 1-256), `Ponder`,
`SyzygyPath`, `SyzygyProbeLimit`, `SyzygyProbeDepth`, and `TelemetryDir`
(for testing). `Hash` and `Move Overhead` as before.

**What's new.**
- *Multi-threaded search* (Lazy SMP): threads share the hash table.
  +151 Elo at 4 threads and +87 at 2 in self-play.
- *Syzygy tablebases* through Fathom: WDL probes in the search, and at
  the root winning moves ranked by distance to zeroing, so won endings
  are converted inside the fifty-move rule.
- *Pondering.*
- *Evaluation:* an initiative term (threats, safe mobility, territory,
  coordination, pawn-push threats, safe checks, king pressure, tempo),
  a drawishness scale for opposite-coloured bishops and close pawnless
  endings, and driving a bare king to the edge (to the right corner for
  bishop and knight).
- *Search:* the halfmove clock in the hash key (so a score from early in
  a fifty-move count isn't reused late), a fix to the prior-countermove
  bonus, late move pruning in the move picker, underpromotion pruning,
  static exchange evaluation of promotions, and time management that
  weighs how much of the search went into the best move, how long it has
  held and whether the score dropped.
- *Tuning:* all of the search's parameters, about 110, tuned together by
  SPSA in two full passes (160,000 games), and the time
  management in a pass of its own at a longer time control.

**Checked before release.**
- A stress run of 8,000 games on both builds (bullet and 40 moves in
  10 s, 24 at a time): no time losses, crashes, disconnects or illegal
  moves; 300 games with both sides pondering, all normal.
- Threads with tablebases in bullet (4 threads, 2+0.02): the stress runs
  found and fixed three problems, none of which changes the search. The
  reported line could start with a helper thread's move; a tablebase
  root's ranking ran before the clock started; and that ranking (by
  distance to zeroing) can take 50 ms on cold tables, so with less than
  a second left the root is now ranked by win, draw or loss alone. After
  the fixes, 2,000 games with 20 search threads on 24 hardware threads:
  no failures.

**Known limitations.**
- No Chess960.
- The evaluation is hand-written; an NNUE comes in 3.0.
- With every hardware thread busy (many threaded games at once), bullet
  games can still be lost on time about once in a thousand, to the
  operating system's scheduling: raise `Move Overhead` in such setups.

**Next.** Chess960 (2.x), then an NNUE trained on FunComp's own games
(3.0), and a learned guide for the search's pruning.

## FunComp 1.0 (2026-09-28)

The first public release: a classical engine (PeSTO's tables and a modern
alpha-beta search), single-threaded.

**Strength.** About **3030 on the CCRL Blitz scale**: 3031 ± 14 from a
gauntlet of 2,040 games at 10+0.1 against 17 rated engines from eight
families (Stash, Booot, Reckless, Akimbo, RubiChess, Seer, Igel, Clover;
CCRL Blitz 2933-3345). An estimate on our hardware, not an official CCRL
rating.

**Binaries.**
- `funcomp-1.0-x86-64-v3.exe`: for CPUs with AVX2 and BMI2 (Intel
  Haswell, AMD Zen and later). The faster build.
- `funcomp-1.0-x86-64.exe`: for any 64-bit x86 CPU.

Both search identically (bench 2780319 nodes) and are statically linked.

**UCI options.** `Hash` (MB, default 64), `Move Overhead` (ms, default 30).

**What went into it.** Every search feature was tested by SPRT
against the previous best build before it was kept. Since the
last internal milestone, accepted: double and negative singular
extensions, a flag for positions on principal variations, late move
reductions in fractions of a ply, ProbCut, history pruning, a limit
on quiet evasions in quiescence, correction history by non-pawn
material, fifty-move scaling, and detection of repetitions one move
away. Together these gained +61 ± 15 Elo against that milestone in
1,000 balanced games. Tested ideas that didn't pay remain in the code,
switched off, for the tuning pass of 2.0.

**Checked before release.**
- A stress run of 8,000 games on both builds (bullet and 40 moves in
  10 s, 24 at a time): no time losses, crashes, disconnects or illegal
  moves.
- A robustness pass on the protocol: negative clocks from the GUI, a
  clock with no time left, FENs without move counters, invalid FENs.
- An originality review: FunComp's code is its own. Stockfish and the
  chess-programming community influenced its ideas (see the README's
  credits); a best-move similarity test puts FunComp no closer to
  Stockfish than unrelated engines are to each other.

**Known limitations.**
- Single-threaded (no `Threads` option); no endgame tablebases; no
  Chess960.
- At bullet speeds (about 5 ms a move) a winning side occasionally fails
  to convert before the fifty-move rule (0.3-0.65% of games at 1+0.01).

**Next.** Version 2.0: a full tuning pass and multi-threaded search.
Then Chess960 and an NNUE.
