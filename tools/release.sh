#!/bin/bash
# The release binaries (3.0 on), in Git Bash on Windows: each CPU level built with the release network embedded,
# their benches checked equal within each float path (those this machine can run), copied to
# release/funcomp-<version>-<level>.exe, and their SHA-256 sums added to release/SHA256SUMS.txt.
#
#   bash tools/release.sh <version> <network .nnue> [levels...]     (levels: x86-64 x86-64-v3 by default)
#
# x86-64: any 64-bit x86 CPU; x86-64-v2: POPCNT and SSE4.2 (about 2009 on); x86-64-v3: AVX2 and BMI2 (Haswell, Zen
# and later), the fast build; x86-64-v4: AVX-512 (Ice Lake, Zen 4 and later; about 10% faster for a flat network on
# Zen 4, measured 2026-10-19), only when tested on such a CPU. Release build type, statically linked, no LTO or PGO
# (measured to give nothing). The network's search parameters are the source's defaults (params.h, NnueScale
# included). The float paths (CMakeLists.txt, FUNCOMP_SIMD_FLOAT): AVX2 with FMA (v3, v4; Android's NEON computes
# the same) and the plain C++ fallback (x86-64, v2). A flat network has one bench count; a multi-layer network one
# for each path (the author, 2026-10-28): CMake's BENCH_NODES_EMBEDDED and BENCH_NODES_EMBEDDED_PLAIN.
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -ge 2 ] || { sed -n '2,14p' "$0"; exit 1; }
version=$1 net=$(cygpath -m "$(realpath "$2")"); shift 2
if [ $# -eq 0 ]; then levels=(x86-64 x86-64-v3); else levels=("$@"); fi
[ -f "$net" ] || { echo "no network $net"; exit 1; }
rel=${RELEASE_DIR:-release}  # another folder for a trial run
mkdir -p "$rel"
declare -A benches  # float path -> its levels' counts
for level in "${levels[@]}"; do
    dir=build-rel-$version-$level
    flags=""; [ "$level" = x86-64-v4 ] && flags="-mavx512vnni"
    cmake -S . -B "$dir" -G Ninja -DCMAKE_BUILD_TYPE=Release -DARCH="$level" -DCMAKE_CXX_FLAGS="$flags" \
        -DEMBED_NET="$net" -DTUNE=OFF > "$dir.cmake.log" 2>&1
    cmake --build "$dir" --target chess > "$dir.build.log" 2>&1
    path=plain; grep -q '^FUNCOMP_SIMD_FLOAT:INTERNAL=1$' "$dir/CMakeCache.txt" && path=simd
    out=$rel/funcomp-$version-$level.exe
    cp "$dir/chess.exe" "$out"
    if [ "$level" = x86-64-v4 ] && ! grep -q avx512f /proc/cpuinfo 2>/dev/null; then
        echo "$out: built, not run here (no AVX-512 on this CPU): test it on one before shipping"
        continue
    fi
    b=$("$out" bench 2>&1 | grep -a ' nodes ' | tail -1 | awk '{print $1}')
    echo "$out: bench $b ($path float layers), $(printf 'uci\nquit\n' | "$out" | grep -a '^id name' | cut -c9-)"
    benches[$path]+="$b "
done
for path in "${!benches[@]}"; do
    [ "$(printf '%s\n' ${benches[$path]} | sort -u | wc -l)" = 1 ] ||
        { echo "the benches differ on the $path float path: ${benches[$path]}"; exit 1; }
done
(cd "$rel" && for level in "${levels[@]}"; do sha256sum -b "funcomp-$version-$level.exe"; done) > "$rel/SHA256SUMS-$version.txt"
grep -v "funcomp-$version-" "$rel/SHA256SUMS.txt" > "$rel/SHA256SUMS.new" 2>/dev/null || true
cat "$rel/SHA256SUMS-$version.txt" >> "$rel/SHA256SUMS.new" && mv "$rel/SHA256SUMS.new" "$rel/SHA256SUMS.txt" && rm "$rel/SHA256SUMS-$version.txt"
for path in simd plain; do
    if [ -n "${benches[$path]:-}" ]; then
        echo "bench ${benches[$path]%% *} ($path float layers: BENCH_NODES_EMBEDDED$([ $path = simd ] || echo _PLAIN))"
    fi
done
echo "$rel/SHA256SUMS.txt:"; cat "$rel/SHA256SUMS.txt"
