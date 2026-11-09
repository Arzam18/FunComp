#!/bin/bash
# The Android release binaries (3.1 on; the author, 2026-10-26, for GitHub issue #1, the Mobile Chess Engine
# Circuit), on Linux with the Android NDK: each ARM level built with the release network embedded, as a static
# executable for Android 7 or later (API 24, arm64-v8a), its bench run under qemu's user-mode emulator when one is
# given and checked equal across the levels, copied to release/funcomp-<version>-android-<level>, and their SHA-256
# sums added to release/SHA256SUMS.txt.
#
#   NDK=<ndk folder> [QEMU=<qemu-aarch64>] bash tools/release-android.sh <version> <network .nnue> [levels...]
#
# Levels: armv8-a (every 64-bit ARM phone) and armv8.2-a+dotprod (the int8 dot product, SDOT: phones since about
# 2018), both by default. The NEON paths (src/nnue.cpp) compute what the AVX2 ones do in the same order, so the bench
# equals the x86-64-v3 build's: compare it with release.sh's. Emulation checks the numbers, not the speed; a phone
# shows the speed.
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -ge 2 ] || { sed -n '2,13p' "$0"; exit 1; }
[ -n "${NDK:-}" ] && [ -f "$NDK/build/cmake/android.toolchain.cmake" ] || { echo "NDK: the Android NDK's folder"; exit 1; }
version=$1 net=$(realpath "$2"); shift 2
if [ $# -eq 0 ]; then levels=(armv8-a armv8.2-a+dotprod); else levels=("$@"); fi
[ -f "$net" ] || { echo "no network $net"; exit 1; }
rel=${RELEASE_DIR:-release}  # another folder for a trial run
mkdir -p "$rel"
benches=() names=()
for level in "${levels[@]}"; do
    name=armv8; [[ $level == *dotprod* ]] && name=armv8-dotprod
    dir=build-android-rel-$version-$name
    cmake -S . -B "$dir" -G Ninja -DCMAKE_BUILD_TYPE=Release -DARCH="$level" -DEMBED_NET="$net" -DTUNE=OFF \
        -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" -DANDROID_ABI=arm64-v8a \
        -DANDROID_PLATFORM=android-24 -DANDROID_STL=c++_static > "$dir.cmake.log" 2>&1
    cmake --build "$dir" --target chess > "$dir.build.log" 2>&1
    out=$rel/funcomp-$version-android-$name
    cp "$dir/chess" "$out"
    "$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" "$out"  # symbols out: a phone app copies it whole
    names+=("funcomp-$version-android-$name")
    if [ -z "${QEMU:-}" ]; then
        echo "$out: built, not run (no QEMU): run its bench on a phone or under qemu-aarch64 before shipping"
        continue
    fi
    b=$("$QEMU" "$out" bench 2>&1 | grep -a ' nodes ' | tail -1 | awk '{print $1}')
    echo "$out: bench $b (under emulation), $(printf 'uci\nquit\n' | "$QEMU" "$out" | grep -a '^id name' | cut -c9-)"
    benches+=("$b")
done
[ "${#benches[@]}" -le 1 ] || [ "$(printf '%s\n' "${benches[@]}" | sort -u | wc -l)" = 1 ] ||
    { echo "the benches differ: ${benches[*]}"; exit 1; }
(cd "$rel" && for n in "${names[@]}"; do sha256sum -b "$n"; done) > "$rel/SHA256SUMS-$version-android.txt"
grep -v "funcomp-$version-android-" "$rel/SHA256SUMS.txt" > "$rel/SHA256SUMS.new" 2>/dev/null || true
cat "$rel/SHA256SUMS-$version-android.txt" >> "$rel/SHA256SUMS.new" && mv "$rel/SHA256SUMS.new" "$rel/SHA256SUMS.txt" &&
    rm "$rel/SHA256SUMS-$version-android.txt"
