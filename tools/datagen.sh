#!/bin/bash
# Runs NNUE data generation on Linux: one engine process per worker, each
# playing self-play games with "gendata" into its own file (the Linux
# counterpart of tools/datagen.ps1). Workers are detached (nohup), so they
# outlive the shell or SSH session that started them.
#
#   tools/datagen.sh <engine> <out dir> <hours> <first worker> <workers> <seed base> <syzygy dir> [eval file] [more gendata options]
#
# Worker w (first .. first + workers - 1) writes <out dir>/worker-<ww>.bin
# with seed <seed base> + w, logging to worker-<ww>.log. A worker whose file
# exists resumes after its last complete game. Number workers and seeds so
# they don't repeat another machine's. The eval file is a network for the
# games (empty or absent: the hand-written eval); more gendata options, e.g.
# "adjscore=800 adjplies=8 adjphase=8 adjtb=1 adjpct=90", as one argument.
# OPTIONS="Name=value ..." sets more UCI options before the games (round 4:
# NnueScale=189, its generator's scale, on a build made with -DTUNE=ON).
set -euo pipefail
engine=$(realpath "$1") out=$2 hours=$3 first=$4 count=$5 base=$6 syzygy=$7 evalfile=${8:-} extra=${9:-}
mkdir -p "$out"
out=$(realpath "$out")
for ((w = first; w < first + count; w++)); do
    name=$(printf 'worker-%02d' "$w")
    {
        echo "setoption name SyzygyPath value $syzygy"
        echo 'setoption name Hash value 1'
        [ -n "$evalfile" ] && echo "setoption name EvalFile value $evalfile"
        for kv in ${OPTIONS:-}; do echo "setoption name ${kv%%=*} value ${kv#*=}"; done
        echo "gendata out=$out/$name.bin worker=$w seed=$((base + w)) softnodes=10000 hardnodes=40000 maxopenscore=525 hash=16 hours=$hours $extra"
        echo 'quit'
    } > "$out/$name.in"
    nohup "$engine" < "$out/$name.in" > "$out/$name.log" 2> "$out/$name.err" &
done
echo "started $count workers ($first..$((first + count - 1))) into $out"
