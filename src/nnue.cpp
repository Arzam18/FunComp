#include "nnue.h"
#include "params.h"
#include "pair_features.h"
#include "pawn_features.h"
#include "square_features.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <vector>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace nnue {

namespace {

struct Network {
    int hidden = 0;
    int king_buckets = 1;
    int output_buckets = 1;
    bool mirrored = false;
    std::array<std::uint8_t, 64> layout{};  // king bucket by king square, in its side's view
    // The first layer as the accumulators read it: l0w's values, or after
    // share() the same values in the mapped network file (l0w is then
    // empty). `weights_at` is where the file's weights start; `l0_count` and
    // `l0_sum` describe a shared first layer, for shared_intact().
    const std::int16_t* l0 = nullptr;
    std::shared_ptr<void> mapping;
    std::size_t weights_at = 0, l0_count = 0;
    std::uint64_t l0_sum = 0;
    std::vector<std::int16_t> l0w;          // INPUTS x king_buckets columns of `hidden`
    std::vector<std::int16_t> l0b;          // hidden
    std::vector<std::int16_t> l1w;          // output_buckets rows of 2 x hidden
    std::vector<std::int16_t> l1b;          // output_buckets
    // Output weights small enough (|w| <= 512) for the products over a
    // chunk of 32 hidden values, both sides (64 products), to fit in 32 bits
    // (64 x 255^2 x 512 < 2^31): bullet clips weights to about
    // +-1.98 x QB = 127.
    bool small_output = false;
    // The AVX2 output path: hidden a multiple of 16 and |w| <= 128, so a
    // clamped hidden value (<= 255) times a weight fits in 16 bits and the
    // 32-bit lane sums can't overflow (per side, at most MAX_HIDDEN / 16 x 2
    // x 255 x 255 x 128 < 2^31 for MAX_HIDDEN 1536). Same results as the scalar code.
    bool simd_output = false;
    // The pawn predicates as extra inputs (pawn_features.h), after the
    // piece inputs.
    bool pawns = false;
    // The pawn-control map (pawn_features.h, feature set bit 11), after the
    // predicates, cached with them.
    bool pawn_spans = false;
    // King-state buckets (feature set bit 13: the first layer's block by
    // king_state(), 15 of them, in place of the layout) and output buckets
    // split by the side to move's shelter (bit 14: material buckets x 3).
    bool king_state = false;
    // Each king state's block: the state itself, or with merged states
    // (fewer king buckets, nnue.h) the header's layout bytes.
    std::array<std::uint8_t, KING_STATES> state_block{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    // The bare king state (bit 18: king_location(), 5 blocks), likewise.
    bool king_location = false;
    bool king_state_output = false;
    // Output buckets by endgame type (bit 16: ENDGAME_TYPES after the
    // others) and passed-pawn quality (bit 17: its pawn-only part with the
    // cached pawn inputs, its piece-dependent part a square family).
    bool endgame_output = false;
    bool passers = false;
    // The extra families after the pawn predicates, kept in the
    // accumulators' lists: pairs (pair_features.h; feature set bits 1-3),
    // then threats and square control (square_features.h; bits 4-7).
    // `extra_base` is the first input of their block.
    unsigned pair_families = 0;
    unsigned square_families = 0;
    std::size_t extra_base = 0;
    // The pairwise multi-layer architecture (header version 3, activation
    // 1): hidden layers of layer1 and layer2 (0: the single output layer
    // above), in floating point as the trainer saves them, one set per
    // output bucket: f1w (OB x layer1 rows of hidden), f2w (OB x layer2
    // rows of layer1), f3w (OB rows of layer2), biases alike.
    int layer1 = 0, layer2 = 0;
    std::vector<float> f1w, f1b, f2w, f2b, f3w, f3b;
    // layer1 quantized at load (the author, 2026-10-18: the float layer cost
    // a layered net about 3x a node): each row's weights in int8 at its own
    // scale (127 / its largest |weight|), and the factor that turns a row's
    // int32 sum over the uint8 pairwise products back into the float sum.
    std::vector<std::int8_t> q1w;
    std::vector<float> q1scale;
    // The same int8 weights block by block for the sparse path (2026-10-19): for each 4 inputs, their
    // weights for every row together (row by row, 4 bytes each), a bucket's blocks one after another.
    // When layer1 is a multiple of 8 (and hidden of 32).
    std::vector<std::int8_t> q1s;
    // layer2's weights transposed, a bucket's layer1 rows of layer2: the SIMD
    // path adds one input's weights to eight of the bucket's outputs at once.
    std::vector<float> f2t;
    // layer2 in int8 too (the author, 2026-10-19): each row's weights at its
    // own scale, block-major as q1s, and the factor that turns a row's int32
    // sum over its inputs (layer1's outputs, in [0, 1], as uint8 in [0, 127])
    // back into the float sum. When layer1 is a multiple of 4, layer2 of 8.
    std::vector<std::int8_t> q2s;
    std::vector<float> q2scale;
    std::string path;
};

Network net;
// Bumped by every load: cached pawn contributions of an earlier network
// are stale.
std::uint32_t generation = 0;

int own_square(Color view, int square) { return view == Color::White ? square : square ^ 56; }

// A view's block of inputs and its mirroring, from its king's square.
struct ViewKey {
    int base;  // first input of the king bucket's block
    int flip;  // 7 when the view is mirrored left-right, else 0
    bool operator==(const ViewKey&) const = default;
};

// With king-state or king-location buckets the block is the state's
// (`state`), not the layout's.
ViewKey view_key(Color view, int king_square, int state = 0) {
    const int k = own_square(view, king_square);
    const int bucket = net.king_state || net.king_location ? state : net.layout[static_cast<std::size_t>(k)];
    return {bucket * INPUTS, net.mirrored && (k & 7) > 3 ? 7 : 0};
}

// The king's pawn shelter from `view`'s side: of the king's file and its
// neighbours, how many have no own pawn on the two ranks in front of the
// king, at most 2 (0 intact, 1 weakened, 2 broken).
int shelter(const Position& pos, int view) {
    const Color v = static_cast<Color>(view);
    Bitboard own = pos.pieces(view == 0 ? WP : BP);
    if (view == 1) own = __builtin_bswap64(own);
    const Bitboard kb = pos.pieces(view == 0 ? WK : BK);
    const int k = own_square(v, kb ? std::countr_zero(kb) : 0);
    const int f = k & 7, r = k >> 3;
    Bitboard front = 0;
    for (int q = r + 1; q <= std::min(r + 2, 7); ++q) front |= Bitboard{0xFF} << (8 * q);
    int missing = 0;
    for (int x = std::max(f - 1, 0); x <= std::min(f + 1, 7); ++x)
        if (!(own & front & (FILE_A_BB << x))) ++missing;
    return std::min(missing, 2);
}

// The king's state from `view`'s side (pawn_features.h's sibling idea; the
// author, 2026-10-05): location x 3 + shelter, location by the king's own
// square, symmetric left-right: files a-b|g-h on its first two ranks 0
// (castled short), c|f 1 (castled long), d|e 2 (centred), further up 3
// (advanced); 4 while the side keeps a castling right (either: they swap
// under mirroring), the king still at home (ROADMAP, Try 5, K2). 15 states.
int king_location(const Position& pos, int view) {
    const Bitboard kb = pos.pieces(view == 0 ? WK : BK);
    const int k = own_square(static_cast<Color>(view), kb ? std::countr_zero(kb) : 0);
    const int edge = std::min(k & 7, 7 - (k & 7));
    const int rights = view == 0 ? Position::WK_CASTLE | Position::WQ_CASTLE : Position::BK_CASTLE | Position::BQ_CASTLE;
    return (pos.castling_rights() & rights) ? 4 : (k >> 3) >= 2 ? 3 : edge <= 1 ? 0 : edge == 2 ? 1 : 2;
}

int king_state(const Position& pos, int view) { return king_location(pos, view) * 3 + shelter(pos, view); }

// The block a view's values are built in, for nets that choose it by the
// king's state or bare location (0 otherwise: the layout's by square).
int view_block(const Position& pos, int view) {
    return net.king_state ? net.state_block[static_cast<std::size_t>(king_state(pos, view))]
                          : net.king_location ? king_location(pos, view) : 0;
}

// The feature of `p` on `square` from `view`'s side.
std::size_t feature(Color view, ViewKey key, Piece p, int square) {
    const int colour = p < BP ? 0 : 1;
    const int sq = own_square(view, square) ^ key.flip;
    return static_cast<std::size_t>(key.base + 384 * (colour != static_cast<int>(view)) +
                                    64 * type_of(p) + sq);
}

const std::int16_t* column(std::size_t f) { return net.l0 + f * static_cast<std::size_t>(net.hidden); }

// The whole file mapped read-only, or nothing if it can't be or is shorter
// than `need` bytes. The mapping lasts as long as the pointer; on Windows the
// file stays open with it, refusing writers meanwhile.
std::shared_ptr<void> map_file(const std::string& path, std::size_t need) {
#if defined(_WIN32)
    const HANDLE file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                    FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return nullptr;
    LARGE_INTEGER size;
    const HANDLE map = GetFileSizeEx(file, &size) && static_cast<std::uint64_t>(size.QuadPart) >= need
                           ? CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr)
                           : nullptr;
    void* p = map ? MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0) : nullptr;
    if (!p) {
        if (map) CloseHandle(map);
        CloseHandle(file);
        return nullptr;
    }
    return std::shared_ptr<void>(p, [file, map](void* q) {
        UnmapViewOfFile(q);
        CloseHandle(map);
        CloseHandle(file);
    });
#elif defined(__unix__) || defined(__APPLE__)
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return nullptr;
    struct stat st;
    const bool fits = ::fstat(fd, &st) == 0 && static_cast<std::uint64_t>(st.st_size) >= need;
    const std::size_t size = fits ? static_cast<std::size_t>(st.st_size) : 0;
    void* p = fits ? ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0) : MAP_FAILED;
    ::close(fd);
    if (p == MAP_FAILED) return nullptr;
    return std::shared_ptr<void>(p, [size](void* q) { ::munmap(q, size); });
#else
    (void)path;
    (void)need;
    return nullptr;
#endif
}

// A sum over the first layer that any changed value changes, eight bytes at
// a time (a pass over 1.5 MB: about a tenth of a millisecond).
std::uint64_t first_layer_sum(const std::int16_t* p, std::size_t count) {
    std::uint64_t h = 0x9E3779B97F4A7C15ULL;
    const auto* bytes = reinterpret_cast<const unsigned char*>(p);
    const std::size_t n = count * sizeof(std::int16_t);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        std::uint64_t x;
        std::memcpy(&x, bytes + i, 8);
        h = (h ^ x) * 0x100000001B3ULL;
        h ^= h >> 29;
    }
    for (; i < n; ++i) h = (h ^ bytes[i]) * 0x100000001B3ULL;
    return h;
}

// A view's active extra inputs, as offsets into their block, ascending
// (the pair families' block first).
int extra_inputs(const Position& pos, Color view, int flip, std::uint32_t* out) {
    const int n = net.pair_families ? pairfeat::inputs(pos, net.pair_families, view, flip, out) : 0;
    if (!net.square_families) return n;
    const int m = sqfeat::inputs(pos, net.square_families, view, flip, out + n);
    const auto shift = static_cast<std::uint32_t>(pairfeat::block_size(net.pair_families));
    for (int i = n; i < n + m; ++i) out[i] += shift;
    return n + m;
}

void add_column(std::int16_t* a, std::size_t f, int sign) {
    const std::int16_t* w = column(f);
    for (int i = 0; i < net.hidden; ++i) a[i] = static_cast<std::int16_t>(a[i] + sign * w[i]);
}

int king_square(const Position& pos, int view) {
    const Bitboard k = pos.pieces(view == 0 ? WK : BK);
    return k ? std::countr_zero(k) : 0;
}

void refresh_view(const Position& pos, Accumulator& acc, int view, RefreshCache* cache = nullptr) {
    const int h = net.hidden;
    const Color v = static_cast<Color>(view);
    const auto c = static_cast<std::size_t>(view);
    acc.king[c] = static_cast<std::int8_t>(king_square(pos, view));
    acc.stale[c] = false;
    acc.bucket[c] = static_cast<std::int8_t>(view_block(pos, view));
    acc.recheck[c] = false;
    const ViewKey key = view_key(v, acc.king[c], acc.bucket[c]);
    std::int16_t* a = acc.v[c].data();
    if (cache && !cache->entries.empty()) {
        // From the block's cached values: out with the pieces that left,
        // in with those that arrived since it was last built.
        const auto block = static_cast<std::size_t>(key.base / INPUTS);
        RefreshCache::Entry& e =
            cache->entries[(c * static_cast<std::size_t>(net.king_buckets) + block) * 2 + (key.flip ? 1 : 0)];
        std::int16_t* ev = e.v.data();
        for (int p = WP; p <= BK; ++p) {
            const Bitboard now = pos.pieces(static_cast<Piece>(p)), was = e.pieces[static_cast<std::size_t>(p)];
            for (Bitboard b = was & ~now; b; b &= b - 1) {
                const std::int16_t* w = column(feature(v, key, static_cast<Piece>(p), std::countr_zero(b)));
                for (int i = 0; i < h; ++i) ev[i] = static_cast<std::int16_t>(ev[i] - w[i]);
            }
            for (Bitboard b = now & ~was; b; b &= b - 1) {
                const std::int16_t* w = column(feature(v, key, static_cast<Piece>(p), std::countr_zero(b)));
                for (int i = 0; i < h; ++i) ev[i] = static_cast<std::int16_t>(ev[i] + w[i]);
            }
            e.pieces[static_cast<std::size_t>(p)] = now;
        }
        std::copy(ev, ev + h, a);
    } else {
        std::copy(net.l0b.begin(), net.l0b.end(), a);
        for (int p = WP; p <= BK; ++p)
            for (Bitboard b = pos.pieces(static_cast<Piece>(p)); b; b &= b - 1) {
                const std::int16_t* w = column(feature(v, key, static_cast<Piece>(p), std::countr_zero(b)));
                for (int i = 0; i < h; ++i) a[i] = static_cast<std::int16_t>(a[i] + w[i]);
            }
    }
    if (acc.extras) {
        ExtraLists& x = *acc.extras;
        x.n[c] = static_cast<std::uint16_t>(extra_inputs(pos, v, key.flip, x.v[c].data()));
        for (int i = 0; i < x.n[c]; ++i) add_column(a, net.extra_base + x.v[c][static_cast<std::size_t>(i)], 1);
    }
}

std::vector<std::int16_t> little_endian_int16(const std::string& bytes, std::size_t from, std::size_t n) {
    std::vector<std::int16_t> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = static_cast<std::int16_t>(static_cast<unsigned char>(bytes[from + 2 * i]) |
                                         static_cast<unsigned char>(bytes[from + 2 * i + 1]) << 8);
    return v;
}

// A view's active pawn-only inputs (the predicates, the pawn-control map,
// passed-pawn quality's pawn-only part), as input indices.
constexpr int MAX_PAWN_INPUTS = pawnfeat::MAX_ACTIVE + pawnfeat::SPAN_FEATURES + pawnfeat::PASSER_MAX;
int pawn_inputs(const Position& pos, int view, int flip, std::uint32_t* out) {
    int n = 0;
    const Color v = static_cast<Color>(view);
    std::size_t base = static_cast<std::size_t>(INPUTS * net.king_buckets);
    if (net.pawns) {
        std::uint16_t f[pawnfeat::MAX_ACTIVE];
        const int k = pawnfeat::features(pos.pieces(WP), pos.pieces(BP), f);
        for (int i = 0; i < k; ++i) {
            const int predicate = f[i] / 128, colour = (f[i] / 64) % 2, sq = f[i] % 64;
            out[n++] = static_cast<std::uint32_t>(base + static_cast<std::size_t>(
                                                             (predicate * 2 + (colour != view)) * 64 + (own_square(v, sq) ^ flip)));
        }
        base += pawnfeat::FEATURES;
    }
    if (net.pawn_spans) {
        const Bitboard all = pos.pieces(WP) | pos.pieces(BP);
        for (int c = 0; c < 2; ++c) {
            const Color side = static_cast<Color>(c);
            for (Bitboard b = pawnfeat::pawn_span(pos.pieces(c == 0 ? WP : BP), all, side); b; b &= b - 1)
                out[n++] = static_cast<std::uint32_t>(base + static_cast<std::size_t>((c != view) * 64 +
                                                                                       (own_square(v, std::countr_zero(b)) ^ flip)));
        }
        base += pawnfeat::SPAN_FEATURES;
    }
    if (net.passers) {
        std::uint16_t f[pawnfeat::PASSER_MAX];
        const int k = pawnfeat::passer_features(pos.pieces(WP), pos.pieces(BP), f);
        for (int i = 0; i < k; ++i) {
            const int label = f[i] / 128, colour = (f[i] / 64) % 2, sq = f[i] % 64;
            out[n++] = static_cast<std::uint32_t>(base + static_cast<std::size_t>(
                                                             (label * 2 + (colour != view)) * 64 + (own_square(v, sq) ^ flip)));
        }
    }
    return n;
}

// A view's share of the first layer from the pawn predicates, by pawn key
// and the view's orientation: a small cache per thread, filled on misses.
const std::int16_t* pawn_contribution(const Position& pos, int view, int flip) {
    constexpr std::size_t ENTRIES = 4096;  // 12 MB a thread (8 MB used at 1024); 2.4% faster than 1024 entries in bench, more no better (2026-10-18)
    struct Entry {
        std::uint64_t key = 0;
        std::uint32_t generation = 0;
        std::array<std::int16_t, MAX_HIDDEN> v;
    };
    thread_local std::vector<Entry> cache(ENTRIES);
    const std::uint64_t key = pos.pawn_key() ^ (0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(1 + view * 8 + flip));
    Entry& e = cache[key % ENTRIES];
    if (e.key == key && e.generation == generation) return e.v.data();
    e.key = key;
    e.generation = generation;
    const int h = net.hidden;
    std::fill(e.v.begin(), e.v.begin() + h, std::int16_t{0});
    std::uint32_t inputs[MAX_PAWN_INPUTS];
    const int n = pawn_inputs(pos, view, flip, inputs);
    for (int i = 0; i < n; ++i) {
        const std::int16_t* w = column(inputs[i]);
        for (int k = 0; k < h; ++k) e.v[static_cast<std::size_t>(k)] = static_cast<std::int16_t>(e.v[static_cast<std::size_t>(k)] + w[k]);
    }
    return e.v.data();
}

// The endgame type for the output (feature set bit 16; ROADMAP, Try 5, G):
// 0 no pawns, 1 pawns only (besides the kings), 2 opposite-coloured
// bishops only (one each, with pawns); -1 none of these.
int endgame_type(const Position& pos) {
    const Bitboard pawns = pos.pieces(WP) | pos.pieces(BP);
    if (!pawns) return 0;
    const Bitboard pieces = pos.occupied() & ~pawns & ~(pos.pieces(WK) | pos.pieces(BK));
    if (!pieces) return 1;
    const Bitboard wb = pos.pieces(WB), bb = pos.pieces(BB);
    constexpr Bitboard LIGHT = 0x55AA55AA55AA55AAULL;
    if (pieces == (wb | bb) && std::popcount(wb) == 1 && std::popcount(bb) == 1 &&
        ((wb & LIGHT) != 0) != ((bb & LIGHT) != 0))
        return 2;
    return -1;
}

}  // namespace

int king_state_of(const Position& pos, Color side) { return king_state(pos, static_cast<int>(side)); }
int endgame_type_of(const Position& pos) { return endgame_type(pos); }

// Output bucket by the pieces on the board (kings included), in
// ceil(32 / material buckets)-sized steps from 2. With output buckets by
// king state, each material bucket splits three ways by the side to move's
// shelter; with output buckets by endgame type, the last ENDGAME_TYPES take
// over where one applies.
int output_bucket(const Position& pos) {
    const int types = net.endgame_output ? ENDGAME_TYPES : 0;
    if (types) {
        const int type = endgame_type(pos);
        if (type >= 0) return net.output_buckets - types + type;
    }
    const int others = net.output_buckets - types;
    const int material_buckets = net.king_state_output ? others / 3 : others;
    const int step = (32 + material_buckets - 1) / material_buckets;
    const int material = std::min((std::popcount(pos.occupied()) - 2) / step, material_buckets - 1);
    return net.king_state_output ? material * 3 + shelter(pos, static_cast<int>(pos.side_to_move())) : material;
}

bool load(const std::string& path) {
    net = Network{};
    if (path.empty() || path == "<empty>") return false;
    std::string bytes;
    if (path == "<embedded>") {  // the binary's own (embedded_net.cpp)
        if (!embedded()) return false;
        bytes.assign(reinterpret_cast<const char*>(embedded_data()), embedded_size());
    } else {
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;
        bytes.assign(std::istreambuf_iterator<char>(in), {});
    }

    Network n;
    std::size_t start = 0;
    if (bytes.size() >= HEADER_BYTES && bytes.compare(0, 4, "FCNN") == 0) {
        const auto u8 = [&](std::size_t i) { return static_cast<unsigned char>(bytes[i]); };
        const int version = u8(4) | u8(5) << 8;
        n.hidden = u8(6) | u8(7) << 8;
        n.king_buckets = u8(8);
        n.mirrored = u8(9) != 0;
        n.output_buckets = u8(10);
        const int activation = u8(11);
        for (std::size_t s = 0; s < 64; ++s) n.layout[s] = u8(12 + s);
        std::uint32_t features = 0;
        if (version >= 2) {
            if (bytes.size() < HEADER_BYTES + (version == 3 ? 8u : 4u)) return false;
            features = static_cast<std::uint32_t>(u8(76) | u8(77) << 8 | u8(78) << 16) |
                       static_cast<std::uint32_t>(u8(79)) << 24;
        }
        n.pawns = (features & 1) != 0;
        n.pawn_spans = (features & PAWN_SPANS_BIT) != 0;
        n.king_state = (features & KING_STATE_BIT) != 0;
        n.king_location = (features & KING_LOCATION_BIT) != 0;
        n.king_state_output = (features & KING_STATE_OUTPUT_BIT) != 0;
        n.endgame_output = (features & ENDGAME_OUTPUT_BIT) != 0;
        n.passers = (features & sqfeat::Passers) != 0;
        n.pair_families = features & pairfeat::ALL;
        n.square_families = features & sqfeat::ALL;
        if (version == 3) {
            n.layer1 = u8(80) | u8(81) << 8;
            n.layer2 = u8(82) | u8(83) << 8;
        }
        const bool multi = activation == 1 && n.layer1 > 0 && n.layer1 <= 256 && n.layer2 > 0 &&
                           n.layer2 <= 256 && n.hidden % 2 == 0;
        // Merged king states: each state's block from the layout, every block used, the rest of the layout 0.
        bool merged_ok = true;
        if (n.king_state && n.king_buckets < KING_STATES && n.king_buckets >= 1) {
            std::array<bool, KING_STATES> used{};
            for (std::size_t s = 0; s < 64; ++s) {
                if (s < static_cast<std::size_t>(KING_STATES)) {
                    merged_ok = merged_ok && n.layout[s] < n.king_buckets;
                    if (merged_ok) used[n.layout[s]] = true, n.state_block[s] = n.layout[s];
                } else {
                    merged_ok = merged_ok && n.layout[s] == 0;
                }
            }
            for (int k = 0; k < n.king_buckets; ++k) merged_ok = merged_ok && used[static_cast<std::size_t>(k)];
        }
        if (version < 1 || version > 3 || (features & ~(1u | PAWN_SPANS_BIT | KING_STATE_BIT | KING_LOCATION_BIT | KING_STATE_OUTPUT_BIT | ENDGAME_OUTPUT_BIT | pairfeat::ALL | sqfeat::ALL)) ||
            (n.king_state && (n.king_buckets > KING_STATES || !merged_ok)) || (n.king_location && n.king_buckets != KING_LOCATIONS) ||
            (n.king_state && n.king_location) || (n.king_state_output && n.output_buckets % 3 != 0) ||
            (n.endgame_output && n.output_buckets <= ENDGAME_TYPES) || !(activation == 0 ? n.layer1 == 0 : multi) ||
            n.hidden < 1 ||
            n.hidden > MAX_HIDDEN ||
            n.king_buckets < 1 || n.king_buckets > MAX_KING_BUCKETS || n.output_buckets < 1 ||
            n.output_buckets > MAX_OUTPUT_BUCKETS ||
            std::any_of(n.layout.begin(), n.layout.end(),
                        [&](std::uint8_t b) { return b >= n.king_buckets; }))
            return false;
        start = HEADER_BYTES + (version >= 2 ? 4 : 0) + (version == 3 ? 4 : 0);
    } else {
        // No header: one king bucket, one output bucket; 1542 bytes per
        // hidden value (768 + 1 + 2 int16) and 2 for the output bias.
        if (bytes.size() < 1544) return false;
        n.hidden = static_cast<int>((bytes.size() - 2) / 1542);
        if (n.hidden > MAX_HIDDEN) return false;
    }
    const std::size_t h = static_cast<std::size_t>(n.hidden);
    n.extra_base = static_cast<std::size_t>(INPUTS * n.king_buckets) +
                  (n.pawns ? static_cast<std::size_t>(pawnfeat::FEATURES) : 0);
    n.extra_base += n.pawn_spans ? static_cast<std::size_t>(pawnfeat::SPAN_FEATURES) : 0;
    n.extra_base += n.passers ? static_cast<std::size_t>(pawnfeat::PASSER_FEATURES) : 0;
    const std::size_t inputs = n.extra_base + static_cast<std::size_t>(pairfeat::block_size(n.pair_families)) +
                               static_cast<std::size_t>(sqfeat::block_size(n.square_families));
    const std::size_t ob = static_cast<std::size_t>(n.output_buckets);
    const std::size_t l1 = static_cast<std::size_t>(n.layer1), l2 = static_cast<std::size_t>(n.layer2);
    // int16: the first layer, and for the single-layer network its output
    // layer; float32: the multi-layer network's later layers.
    const std::size_t count = inputs * h + h + (l1 ? 0 : 2 * h * ob + ob);
    const std::size_t floats = l1 ? ob * l1 * h + ob * l1 + ob * l2 * l1 + ob * l2 + ob * l2 + ob : 0;
    const std::size_t used = 2 * count + 4 * floats;
    // Zero padding to a multiple of 64 bytes after the weights.
    if (bytes.size() < start + used || bytes.size() - start - used >= 64) return false;
    const std::vector<std::int16_t> v = little_endian_int16(bytes, start, count);
    auto at = [&](std::size_t i) { return v.begin() + static_cast<std::ptrdiff_t>(i); };
    n.l0w.assign(at(0), at(inputs * h));
    n.l0 = n.l0w.data();  // the vector's storage moves with it, below
    n.weights_at = start;
    n.l0b.assign(at(inputs * h), at(inputs * h + h));
    if (l1) {
        std::size_t pos = start + 2 * count;
        auto take = [&](std::vector<float>& out, std::size_t k) {
            out.resize(k);
            std::memcpy(out.data(), bytes.data() + pos, 4 * k);
            pos += 4 * k;
        };
        take(n.f1w, ob * l1 * h);
        take(n.f1b, ob * l1);
        take(n.f2w, ob * l2 * l1);
        take(n.f2b, ob * l2);
        take(n.f3w, ob * l2);
        take(n.f3b, ob);
        // layer1 in int8, a scale a row. A pairwise product p in [0, 1] is
        // carried as p8 = (a * b + 256) >> 9 for a, b the clipped values in
        // [0, 255]: the nearest integer to p * 65025 / 512, so a row's float
        // sum is its int32 sum times 512 / (65025 * scale).
        n.q1w.resize(ob * l1 * h);
        n.q1scale.resize(ob * l1);
        for (std::size_t r = 0; r < ob * l1; ++r) {
            const float* w = &n.f1w[r * h];
            float m = 0.0f;
            for (std::size_t i = 0; i < h; ++i) m = std::max(m, std::abs(w[i]));
            const float s = m > 0.0f ? 127.0f / m : 1.0f;
            for (std::size_t i = 0; i < h; ++i)
                n.q1w[r * h + i] = static_cast<std::int8_t>(std::clamp(std::lround(w[i] * s), -127L, 127L));
            n.q1scale[r] = 512.0f / (65025.0f * s);
        }
        if (l1 % 8 == 0 && h % 32 == 0) {
            const std::size_t blocks = h / 4;
            n.q1s.resize(ob * blocks * l1 * 4);
            for (std::size_t k = 0; k < ob; ++k)
                for (std::size_t blk = 0; blk < blocks; ++blk)
                    for (std::size_t row = 0; row < l1; ++row)
                        for (std::size_t t = 0; t < 4; ++t)
                            n.q1s[((k * blocks + blk) * l1 + row) * 4 + t] = n.q1w[(k * l1 + row) * h + blk * 4 + t];
        }
        n.f2t.resize(ob * l2 * l1);
        for (std::size_t k = 0; k < ob; ++k)
            for (std::size_t j = 0; j < l2; ++j)
                for (std::size_t i = 0; i < l1; ++i) n.f2t[(k * l1 + i) * l2 + j] = n.f2w[(k * l2 + j) * l1 + i];
        if (l1 % 4 == 0 && l2 % 8 == 0) {
            const std::size_t blocks = l1 / 4;
            n.q2s.resize(ob * blocks * l2 * 4);
            n.q2scale.resize(ob * l2);
            for (std::size_t k = 0; k < ob; ++k)
                for (std::size_t row = 0; row < l2; ++row) {
                    const float* w = &n.f2w[(k * l2 + row) * l1];
                    float m = 0.0f;
                    for (std::size_t i = 0; i < l1; ++i) m = std::max(m, std::abs(w[i]));
                    const float s = m > 0.0f ? 127.0f / m : 1.0f;
                    n.q2scale[k * l2 + row] = 1.0f / (127.0f * s);
                    for (std::size_t i = 0; i < l1; ++i)
                        n.q2s[((k * blocks + i / 4) * l2 + row) * 4 + i % 4] =
                            static_cast<std::int8_t>(std::clamp(std::lround(w[i] * s), -127L, 127L));
                }
        }
    } else {
        n.l1w.assign(at(inputs * h + h), at(inputs * h + h + 2 * h * ob));
        n.l1b.assign(at(inputs * h + h + 2 * h * ob), at(count));
    }
    n.small_output = std::all_of(n.l1w.begin(), n.l1w.end(),
                                 [](std::int16_t w) { return std::abs(w) <= 512; });
    n.simd_output = n.hidden % 16 == 0 &&
                    std::all_of(n.l1w.begin(), n.l1w.end(),
                                [](std::int16_t w) { return std::abs(w) <= 128; });
    n.path = path;
    net = std::move(n);
    ++generation;
    return true;
}

bool share() {
    if (net.mapping) return true;
    // The file's values are little-endian int16 at an even offset: on such a
    // machine the mapped bytes are the values themselves.
    if (net.hidden <= 0 || net.l0w.empty() || std::endian::native != std::endian::little || net.weights_at % 2 != 0)
        return false;
    const std::size_t bytes = net.l0w.size() * sizeof(std::int16_t);
    std::shared_ptr<void> map = map_file(net.path, net.weights_at + bytes);
    if (!map) return false;
    const auto* p = reinterpret_cast<const std::int16_t*>(static_cast<const char*>(map.get()) + net.weights_at);
    if (std::memcmp(p, net.l0w.data(), bytes) != 0) return false;  // not the file that was loaded any more
    net.l0_count = net.l0w.size();
    net.l0_sum = first_layer_sum(p, net.l0_count);
    net.mapping = std::move(map);
    net.l0 = p;
    std::vector<std::int16_t>().swap(net.l0w);
    return true;
}

bool shared() { return net.mapping != nullptr; }

bool shared_intact() { return !net.mapping || first_layer_sum(net.l0, net.l0_count) == net.l0_sum; }

bool active() { return net.hidden > 0; }
int hidden() { return net.hidden; }
int king_buckets() { return net.king_buckets; }
int output_buckets() { return net.output_buckets; }
bool pawn_features() { return net.pawns; }
bool pawn_span_features() { return net.pawn_spans; }
bool king_state_buckets() { return net.king_state; }
bool king_location_buckets() { return net.king_location; }
bool king_state_output_buckets() { return net.king_state_output; }
bool endgame_output_buckets() { return net.endgame_output; }
bool passer_features() { return net.passers; }
unsigned pair_families() { return net.pair_families; }
unsigned square_families() { return net.square_families; }
bool extra_families() { return net.pair_families || net.square_families; }
const std::string& loaded_path() { return net.path; }

void refresh(const Position& pos, Accumulator& acc) {
    for (int view = 0; view < 2; ++view) refresh_view(pos, acc, view);
}

void RefreshCache::reset() {
    // Every entry the biases with no pieces: the empty board's values.
    entries.resize(static_cast<std::size_t>(2 * net.king_buckets * 2));
    for (Entry& e : entries) {
        std::copy(net.l0b.begin(), net.l0b.end(), e.v.begin());
        e.pieces.fill(0);
    }
}

void update(const Accumulator& from, Accumulator& to, const DirtyPieces& dirty) {
    const int h = net.hidden;
    for (int view = 0; view < 2; ++view) {
        const auto c = static_cast<std::size_t>(view);
        const Color v = static_cast<Color>(view);
        const Piece king = view == 0 ? WK : BK;
        // This side's king may have moved; a new bucket or mirroring means
        // every feature of the view changes: rebuild it when next needed.
        int king_sq = from.king[c];
        for (int i = 0; i < dirty.count; ++i)
            if (dirty.d[i].piece == king && dirty.d[i].to >= 0) king_sq = dirty.d[i].to;
        to.king[c] = static_cast<std::int8_t>(king_sq);
        // With king-state buckets the view keeps its state's block until
        // evaluate() rechecks it, after a move of this side's king or pawns,
        // or of a rook of its (moved or taken: castling rights); with the
        // bare king state, of its king or a rook only (no shelter).
        to.bucket[c] = from.bucket[c];
        to.recheck[c] = from.recheck[c];
        if (net.king_state || net.king_location) {
            const Piece pawn = net.king_state ? (view == 0 ? WP : BP) : NO_PIECE, rook = view == 0 ? WR : BR;
            for (int i = 0; i < dirty.count; ++i)
                if (dirty.d[i].piece == king || dirty.d[i].piece == pawn || dirty.d[i].piece == rook) to.recheck[c] = true;
        }
        const ViewKey key = view_key(v, king_sq, to.bucket[c]);
        to.stale[c] = from.stale[c] || !(key == view_key(v, from.king[c], from.bucket[c]));
        if (to.stale[c]) continue;
        // Extra inputs as in `from`; evaluate() catches up on them.
        if (from.extras && to.extras) {
            to.extras->n[c] = from.extras->n[c];
            std::copy_n(from.extras->v[c].begin(), from.extras->n[c], to.extras->v[c].begin());
        }

        // One pass over the hidden values, with the columns of the features
        // that leave and appear: a quiet move is one of each, a capture two
        // and one, castling two and two (a capturing promotion, two and one).
        const std::int16_t* subs[3];
        const std::int16_t* adds[3];
        int n_sub = 0, n_add = 0;
        for (int i = 0; i < dirty.count; ++i) {
            const DirtyPiece& d = dirty.d[i];
            if (d.from >= 0) subs[n_sub++] = column(feature(v, key, d.piece, d.from));
            if (d.to >= 0) adds[n_add++] = column(feature(v, key, d.piece, d.to));
        }
        const std::int16_t* in = from.v[c].data();
        std::int16_t* out = to.v[c].data();
        auto apply = [&](auto&& f) {
            for (int i = 0; i < h; ++i) out[i] = static_cast<std::int16_t>(f(i));
        };
        if (n_sub == 1 && n_add == 1)
            apply([&](int i) { return in[i] - subs[0][i] + adds[0][i]; });
        else if (n_sub == 2 && n_add == 1)
            apply([&](int i) { return in[i] - subs[0][i] - subs[1][i] + adds[0][i]; });
        else if (n_sub == 2 && n_add == 2)
            apply([&](int i) { return in[i] - subs[0][i] - subs[1][i] + adds[0][i] + adds[1][i]; });
        else {
            std::copy(in, in + h, out);
            for (int k = 0; k < n_sub; ++k)
                for (int i = 0; i < h; ++i) out[i] = static_cast<std::int16_t>(out[i] - subs[k][i]);
            for (int k = 0; k < n_add; ++k)
                for (int i = 0; i < h; ++i) out[i] = static_cast<std::int16_t>(out[i] + adds[k][i]);
        }
    }
}

#if defined(__AVX2__) && defined(__FMA__)
// layer2's outputs j0 to j0 + 8 NV of output bucket b, squared clipped ReLU
// applied: NV independent accumulators of eight outputs, each input's
// transposed weights times its value (a row's dot product would wait on
// every addition before the next).
template <int NV>
static void layer2_block(const float* h1, float* h2, std::size_t b, int j0) {
    const std::size_t l1 = static_cast<std::size_t>(net.layer1), l2 = static_cast<std::size_t>(net.layer2);
    const std::size_t first = b * l2 + static_cast<std::size_t>(j0);
    __m256 acc[NV];
    for (int k = 0; k < NV; ++k) acc[k] = _mm256_loadu_ps(&net.f2b[first + 8 * static_cast<std::size_t>(k)]);
    const float* w = &net.f2t[b * l1 * l2 + static_cast<std::size_t>(j0)];
    for (std::size_t i = 0; i < l1; ++i, w += l2) {
        const __m256 x = _mm256_set1_ps(h1[i]);
        for (int k = 0; k < NV; ++k) acc[k] = _mm256_fmadd_ps(_mm256_loadu_ps(w + 8 * k), x, acc[k]);
    }
    const __m256 zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
    for (int k = 0; k < NV; ++k) {
        const __m256 c = _mm256_min_ps(_mm256_max_ps(acc[k], zero), one);
        _mm256_storeu_ps(h2 + j0 + 8 * k, _mm256_mul_ps(c, c));
    }
}
#endif

// The pairwise multi-layer network after the accumulators, in floating
// point as trained: each side's hidden values clipped to [0, 1] (/ QA), its two
// halves multiplied pairwise (hidden / 2 values a side, the side to move's
// first), then layer1 and layer2 with squared clipped ReLU and the output,
// each from the output bucket's weights. Centipawns, from the side to move.
std::FILE* input_dump = nullptr;  // dump_layer1_inputs()

#if defined(__AVX2__)
// acc plus, in each int32 lane, the sum of its 4 uint8 x int8 products. With
// AVX-512 VNNI one instruction (vpdpbusd: a VNNI build, e.g. -march=znver4,
// 2026-10-19); else maddubs (adjacent pairs into int16) and madd with ones
// (into int32). Every caller's uint8 inputs are at most 127, so a pair is at
// most 2 x 127 x 128 and maddubs never saturates: the same sums either way.
static inline __m256i dot4_add(__m256i acc, __m256i u8, __m256i i8) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    return _mm256_dpbusd_epi32(acc, u8, i8);
#else
    return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(u8, i8), _mm256_set1_epi16(1)));
#endif
}

// For each byte, the positions of its set bits, first to last: the sparse
// layer1's list of nonzero input blocks, eight blocks at a time.
struct NonzeroTable {
    std::array<std::array<std::uint16_t, 8>, 256> idx{};
    NonzeroTable() {
        for (int m = 0; m < 256; ++m) {
            int k = 0;
            for (int bit = 0; bit < 8; ++bit)
                if (m >> bit & 1) idx[static_cast<std::size_t>(m)][static_cast<std::size_t>(k++)] = static_cast<std::uint16_t>(bit);
        }
    }
};
const NonzeroTable nonzero_table;

// layer1 over the nonzero 4-input blocks only (the author, 2026-10-19: about
// 7 in 8 inputs are zero; with the neurons grouped by tools/nnue/permute.py,
// about 2 in 3 blocks are all zero): the blocks found eight at a time (one
// compare, a mask, the table), then per block its 4 inputs broadcast against
// every row's 4 weights, RV registers of 8 rows. The same int32 sums as the
// dense loop, so the same evaluations.
template <int RV>
static void sparse_layer1(const std::int8_t* wb, const std::uint8_t* p8, int h, std::int32_t* sums) {
    alignas(32) std::array<std::uint16_t, MAX_HIDDEN / 4 + 8> nz;
    int count = 0;
    const __m256i zero = _mm256_setzero_si256();
    for (int i = 0; i < h / 4; i += 8) {
        // An input block's 4 bytes (each at most 127) as an int32 is positive exactly when one is nonzero.
        const __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(p8 + 4 * i));
        const auto mask = static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(v, zero))));
        const __m128i idx = _mm_add_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(nonzero_table.idx[mask].data())),
                                          _mm_set1_epi16(static_cast<short>(i)));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&nz[static_cast<std::size_t>(count)]), idx);
        count += std::popcount(mask);
    }
    __m256i acc[RV];
    for (int r = 0; r < RV; ++r) acc[r] = zero;
    for (int k = 0; k < count; ++k) {
        const int blk = nz[static_cast<std::size_t>(k)];
        std::int32_t four;
        std::memcpy(&four, p8 + 4 * blk, 4);
        const __m256i x = _mm256_set1_epi32(four);
        const std::int8_t* w = wb + static_cast<std::size_t>(blk) * (RV * 32);
        for (int r = 0; r < RV; ++r)
            acc[r] = dot4_add(acc[r], x, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 32 * r)));
    }
    for (int r = 0; r < RV; ++r) _mm256_storeu_si256(reinterpret_cast<__m256i*>(sums + 8 * r), acc[r]);
}
#endif

#if defined(__AVX2__)
// A dense int8 layer over uint8 inputs in [0, 127], block-major weights (each
// 4 inputs' weights for every row together): per block its 4 inputs broadcast
// against the rows, RV registers of 8 rows' int32 sums; all-zero blocks skipped.
template <int RV>
static void int8_rows(const std::int8_t* wb, const std::uint8_t* in, int blocks, std::int32_t* sums) {
    __m256i acc[RV];
    for (int r = 0; r < RV; ++r) acc[r] = _mm256_setzero_si256();
    for (int blk = 0; blk < blocks; ++blk) {
        std::int32_t four;
        std::memcpy(&four, in + 4 * blk, 4);
        if (!four) continue;
        const __m256i x = _mm256_set1_epi32(four);
        const std::int8_t* w = wb + static_cast<std::size_t>(blk) * (RV * 32);
        for (int r = 0; r < RV; ++r)
            acc[r] = dot4_add(acc[r], x, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 32 * r)));
    }
    for (int r = 0; r < RV; ++r) _mm256_storeu_si256(reinterpret_cast<__m256i*>(sums + 8 * r), acc[r]);
}
#endif

// `pu`, `pt`: each side's pawn-only share, added as the values are read (the
// AVX2 path only; null elsewhere, the caller adds them first).
static int multi_layer(const std::int16_t* us, const std::int16_t* them, const std::int16_t* pu, const std::int16_t* pt,
                       int bucket) {
    const int h = net.hidden, half = h / 2, l1 = net.layer1, l2 = net.layer2;
    const auto b = static_cast<std::size_t>(bucket);
    const auto screlu = [](float x) {
        const float c = std::clamp(x, 0.0f, 1.0f);
        return c * c;
    };
    std::array<float, 256> h1, h2;
    if (params::nnue_layer_float) {  // the reference: layer1 in float, as trained
        std::array<float, MAX_HIDDEN> in;
        const auto crelu = [](std::int16_t x) { return std::clamp(static_cast<float>(x) / QA, 0.0f, 1.0f); };
        for (int i = 0; i < half; ++i) {
            in[static_cast<std::size_t>(i)] = crelu(us[i]) * crelu(us[half + i]);
            in[static_cast<std::size_t>(half + i)] = crelu(them[i]) * crelu(them[half + i]);
        }
        for (int j = 0; j < l1; ++j) {
            const std::size_t row = b * static_cast<std::size_t>(l1) + static_cast<std::size_t>(j);
            const float* w = &net.f1w[row * static_cast<std::size_t>(h)];
            float z = net.f1b[row];
            for (int i = 0; i < h; ++i) z += w[i] * in[static_cast<std::size_t>(i)];
            h1[static_cast<std::size_t>(j)] = screlu(z);
        }
    } else {
        // layer1 in int8: each side's two halves clipped to [0, 255] and
        // multiplied, kept as (a * b + 256) >> 9 (rounded) in [0, 127] (uint8), then
        // each row's int32 sum of uint8 x int8 (q1w), rescaled (q1scale).
        alignas(32) std::array<std::uint8_t, MAX_HIDDEN> p8;
        std::array<std::int32_t, 256> sums;
        const std::int8_t* rows = &net.q1w[b * static_cast<std::size_t>(l1) * static_cast<std::size_t>(h)];
#if defined(__AVX2__)
        if (half % 32 == 0) {
            const __m256i zero = _mm256_setzero_si256(), qa = _mm256_set1_epi16(QA);
            const auto load = [](const std::int16_t* p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)); };
            const auto clip = [&](__m256i x) { return _mm256_min_epi16(_mm256_max_epi16(x, zero), qa); };
            // A value with its pawn-only share (when there is one) added as it is read: no copy of the side's values.
            const auto value = [&](const std::int16_t* v, const std::int16_t* p, int i) {
                return p ? _mm256_add_epi16(load(v + i), load(p + i)) : load(v + i);
            };
            // (a * b + 256) >> 9 in unsigned 16-bit lanes: a * b <= 65025 fits, the low half of the product is it.
            const auto prod = [&](const std::int16_t* v, const std::int16_t* p, int i) {
                const __m256i ab = _mm256_mullo_epi16(clip(value(v, p, i)), clip(value(v, p, half + i)));
                return _mm256_srli_epi16(_mm256_add_epi16(ab, _mm256_set1_epi16(256)), 9);
            };
            for (int side = 0; side < 2; ++side) {
                const std::int16_t* v = side ? them : us;
                const std::int16_t* p = side ? pt : pu;
                for (int i = 0; i < half; i += 32) {
                    // packus interleaves the two 128-bit lanes; the permute puts the 32 bytes back in order.
                    const __m256i packed = _mm256_packus_epi16(prod(v, p, i), prod(v, p, i + 16));
                    _mm256_store_si256(reinterpret_cast<__m256i*>(&p8[static_cast<std::size_t>(side * half + i)]),
                                       _mm256_permute4x64_epi64(packed, 0xD8));
                }
            }
            // Four rows at a time share each 32-byte load of the inputs (dot4_add: the int32 sums of 4
            // uint8 x int8 products).
            const auto hsum = [](__m256i x) {
                const __m128i s = _mm_add_epi32(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
                const __m128i t = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
                return _mm_cvtsi128_si32(_mm_add_epi32(t, _mm_shuffle_epi32(t, 0xB1)));
            };
            if (input_dump) std::fwrite(p8.data(), 1, static_cast<std::size_t>(h), input_dump);
            const std::int8_t* wb = net.q1s.empty() ? nullptr
                : &net.q1s[b * static_cast<std::size_t>(h / 4) * static_cast<std::size_t>(l1) * 4];
            int j = wb ? l1 : 0;  // the sparse path does every row; else the dense loops below
            if (wb) {
                switch (l1 / 8) {
                    case 1: sparse_layer1<1>(wb, p8.data(), h, sums.data()); break;
                    case 2: sparse_layer1<2>(wb, p8.data(), h, sums.data()); break;
                    case 4: sparse_layer1<4>(wb, p8.data(), h, sums.data()); break;
                    case 8: sparse_layer1<8>(wb, p8.data(), h, sums.data()); break;
                    default: j = 0; break;
                }
            }
            for (; j + 4 <= l1; j += 4) {
                __m256i acc[4] = {zero, zero, zero, zero};
                for (int i = 0; i < h; i += 32) {
                    const __m256i in = _mm256_load_si256(reinterpret_cast<const __m256i*>(&p8[static_cast<std::size_t>(i)]));
                    for (int k = 0; k < 4; ++k) {
                        const __m256i w = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(rows + static_cast<std::size_t>(j + k) * static_cast<std::size_t>(h) + static_cast<std::size_t>(i)));
                        acc[k] = dot4_add(acc[k], in, w);
                    }
                }
                for (int k = 0; k < 4; ++k) sums[static_cast<std::size_t>(j + k)] = hsum(acc[k]);
            }
            for (; j < l1; ++j) {
                __m256i acc = zero;
                for (int i = 0; i < h; i += 32) {
                    const __m256i in = _mm256_load_si256(reinterpret_cast<const __m256i*>(&p8[static_cast<std::size_t>(i)]));
                    const __m256i w = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(rows + static_cast<std::size_t>(j) * static_cast<std::size_t>(h) + static_cast<std::size_t>(i)));
                    acc = dot4_add(acc, in, w);
                }
                sums[static_cast<std::size_t>(j)] = hsum(acc);
            }
        } else
#endif
        {
            const auto clip = [](std::int16_t x) { return std::clamp(static_cast<int>(x), 0, QA); };
            for (int i = 0; i < half; ++i) {
                p8[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>((clip(us[i]) * clip(us[half + i]) + 256) >> 9);
                p8[static_cast<std::size_t>(half + i)] =
                    static_cast<std::uint8_t>((clip(them[i]) * clip(them[half + i]) + 256) >> 9);
            }
            if (input_dump) std::fwrite(p8.data(), 1, static_cast<std::size_t>(h), input_dump);
            for (int j = 0; j < l1; ++j) {
                const std::int8_t* w = rows + static_cast<std::size_t>(j) * static_cast<std::size_t>(h);
                std::int32_t s = 0;
                for (int i = 0; i < h; ++i) s += static_cast<std::int32_t>(p8[static_cast<std::size_t>(i)]) * w[i];
                sums[static_cast<std::size_t>(j)] = s;
            }
        }
        int j = 0;
#if defined(__AVX2__) && defined(__FMA__)
        const __m256 zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
        for (; j + 8 <= l1; j += 8) {
            const std::size_t row = b * static_cast<std::size_t>(l1) + static_cast<std::size_t>(j);
            const __m256 s = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(&sums[static_cast<std::size_t>(j)])));
            const __m256 z = _mm256_fmadd_ps(s, _mm256_loadu_ps(&net.q1scale[row]), _mm256_loadu_ps(&net.f1b[row]));
            const __m256 c = _mm256_min_ps(_mm256_max_ps(z, zero), one);
            _mm256_storeu_ps(&h1[static_cast<std::size_t>(j)], _mm256_mul_ps(c, c));
        }
#endif
        for (; j < l1; ++j) {
            const std::size_t row = b * static_cast<std::size_t>(l1) + static_cast<std::size_t>(j);
            h1[static_cast<std::size_t>(j)] =
                screlu(static_cast<float>(sums[static_cast<std::size_t>(j)]) * net.q1scale[row] + net.f1b[row]);
        }
    }
#if defined(__AVX2__) && defined(__FMA__)
    if (l2 % 8 == 0 && !params::nnue_layer_float) {  // NnueLayerFloat: the reference, every layer plain
        const int rv = l2 / 8;
        if (!net.q2s.empty() && (rv == 1 || rv == 2 || rv == 4 || rv == 8)) {
            // layer2 in int8: layer1's outputs as uint8 (x 127, rounded), the rows by blocks, rescaled.
            alignas(32) std::array<std::uint8_t, 256> h8;
            std::array<std::int32_t, 256> s2;
            for (int i = 0; i < l1; i += 8) {
                const __m256i q = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(&h1[static_cast<std::size_t>(i)]), _mm256_set1_ps(127.0f)));
                const __m128i w16 = _mm_packs_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(&h8[static_cast<std::size_t>(i)]), _mm_packus_epi16(w16, w16));
            }
            const std::int8_t* wb = &net.q2s[b * static_cast<std::size_t>(l1 / 4) * static_cast<std::size_t>(l2) * 4];
            switch (rv) {
                case 1: int8_rows<1>(wb, h8.data(), l1 / 4, s2.data()); break;
                case 2: int8_rows<2>(wb, h8.data(), l1 / 4, s2.data()); break;
                case 4: int8_rows<4>(wb, h8.data(), l1 / 4, s2.data()); break;
                default: int8_rows<8>(wb, h8.data(), l1 / 4, s2.data()); break;
            }
            const __m256 zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
            for (int j = 0; j < l2; j += 8) {
                const std::size_t row = b * static_cast<std::size_t>(l2) + static_cast<std::size_t>(j);
                const __m256 s = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(&s2[static_cast<std::size_t>(j)])));
                const __m256 z = _mm256_fmadd_ps(s, _mm256_loadu_ps(&net.q2scale[row]), _mm256_loadu_ps(&net.f2b[row]));
                const __m256 c = _mm256_min_ps(_mm256_max_ps(z, zero), one);
                _mm256_storeu_ps(&h2[static_cast<std::size_t>(j)], _mm256_mul_ps(c, c));
            }
        } else {
            int j = 0;
            for (; j + 64 <= l2; j += 64) layer2_block<8>(h1.data(), h2.data(), b, j);
            if (j + 32 <= l2) layer2_block<4>(h1.data(), h2.data(), b, j), j += 32;
            if (j + 16 <= l2) layer2_block<2>(h1.data(), h2.data(), b, j), j += 16;
            if (j < l2) layer2_block<1>(h1.data(), h2.data(), b, j);
        }
        const float* w = &net.f3w[b * static_cast<std::size_t>(l2)];
        __m256 acc = _mm256_setzero_ps();
        for (int i = 0; i < l2; i += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(&h2[static_cast<std::size_t>(i)]), acc);
        const __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        const __m128 t = _mm_add_ps(s, _mm_movehl_ps(s, s));
        const float out = net.f3b[b] + _mm_cvtss_f32(_mm_add_ss(t, _mm_shuffle_ps(t, t, 1)));
        return static_cast<int>(out * static_cast<float>(params::nnue_scale));
    }
#endif
    for (int j = 0; j < l2; ++j) {
        const std::size_t row = b * static_cast<std::size_t>(l2) + static_cast<std::size_t>(j);
        const float* w = &net.f2w[row * static_cast<std::size_t>(l1)];
        float z = net.f2b[row];
        for (int i = 0; i < l1; ++i) z += w[i] * h1[static_cast<std::size_t>(i)];
        h2[static_cast<std::size_t>(j)] = screlu(z);
    }
    const float* w = &net.f3w[b * static_cast<std::size_t>(l2)];
    float out = net.f3b[b];
    for (int i = 0; i < l2; ++i) out += w[i] * h2[static_cast<std::size_t>(i)];
    return static_cast<int>(out * static_cast<float>(params::nnue_scale));
}

// The extra inputs that changed since a view's values were computed: out
// with the old, in with the new (the lists are ascending).
static void catch_up_extras(const Position& pos, Accumulator& acc, int view) {
    const auto c = static_cast<std::size_t>(view);
    const Color v = static_cast<Color>(view);
    static_assert(MAX_EXTRA_INPUTS == pairfeat::MAX_ACTIVE + sqfeat::MAX_ACTIVE);
    std::uint32_t now[MAX_EXTRA_INPUTS];
    const int n = extra_inputs(pos, v, view_key(v, acc.king[c]).flip, now);
    ExtraLists& x = *acc.extras;
    const std::uint32_t* was = x.v[c].data();
    const int m = x.n[c];
    std::int16_t* a = acc.v[c].data();
    int i = 0, j = 0;
    while (i < m || j < n) {
        if (j == n || (i < m && was[i] < now[j])) add_column(a, net.extra_base + was[i++], -1);
        else if (i == m || now[j] < was[i]) add_column(a, net.extra_base + now[j++], 1);
        else ++i, ++j;
    }
    std::copy_n(now, n, x.v[c].begin());
    x.n[c] = static_cast<std::uint16_t>(n);
}

int evaluate(Accumulator& acc, const Position& pos, RefreshCache* cache) {
    for (int view = 0; view < 2; ++view) {
        const auto c = static_cast<std::size_t>(view);
        // A king-state (or bare king-state) view whose block changed is rebuilt for its new one.
        if (!acc.stale[c] && (net.king_state || net.king_location) && acc.recheck[c]) {
            if (view_block(pos, view) != acc.bucket[c]) acc.stale[c] = true;
            else acc.recheck[c] = false;
        }
        if (acc.stale[c]) refresh_view(pos, acc, view, cache);
        else if (acc.extras && (net.pair_families || net.square_families))
            catch_up_extras(pos, acc, view);
    }
    const int h = net.hidden;
    const Color stm = pos.side_to_move();
    const std::int16_t* us = acc.v[static_cast<std::size_t>(stm)].data();
    const std::int16_t* them = acc.v[static_cast<std::size_t>(opposite(stm))].data();
    // The pawn-only share (predicates, control map), from the cache, added
    // to both views: by the layered net's AVX2 path as it reads them (no
    // copies: 2026-10-18), else into copies first.
    alignas(64) std::array<std::int16_t, MAX_HIDDEN> us_pawns, them_pawns;
    const std::int16_t* pu = nullptr;
    const std::int16_t* pt = nullptr;
    if (net.pawns || net.pawn_spans || net.passers) {
        const auto share = [&](Color view) {
            const int v = static_cast<int>(view);
            return pawn_contribution(pos, v, view_key(view, acc.king[static_cast<std::size_t>(v)]).flip);
        };
        const auto add_pawns = [&](const std::int16_t* from, const std::int16_t* p, std::int16_t* to) {
            for (int i = 0; i < h; ++i) to[i] = static_cast<std::int16_t>(from[i] + p[i]);
        };
#if defined(__AVX2__)
        const bool fused = net.layer1 && !params::nnue_layer_float && (h / 2) % 32 == 0;
#else
        const bool fused = false;
#endif
        if (fused) {
            pt = share(opposite(stm));
            pu = share(stm);
            if (pu == pt) {  // both in one slot: the second lookup replaced the first; them's added into a copy
                add_pawns(them, share(opposite(stm)), them_pawns.data());
                them = them_pawns.data();
                pt = nullptr;
                pu = share(stm);
            }
        } else {
            add_pawns(us, share(stm), us_pawns.data());
            add_pawns(them, share(opposite(stm)), them_pawns.data());
            us = us_pawns.data();
            them = them_pawns.data();
        }
    }
    const int bucket = output_bucket(pos);
    if (net.layer1) return multi_layer(us, them, pu, pt, bucket);
    const std::int16_t* w = net.l1w.data() + static_cast<std::size_t>(bucket) * 2 * static_cast<std::size_t>(h);
    std::int64_t sum = 0;
#if defined(__AVX2__)
    if (net.simd_output) {
        const __m256i zero = _mm256_setzero_si256(), qa = _mm256_set1_epi16(QA);
        const auto load = [](const std::int16_t* p) {
            return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        };
        __m256i sum_us = zero, sum_them = zero;
        for (int i = 0; i < h; i += 16) {
            const __m256i x = _mm256_min_epi16(_mm256_max_epi16(load(us + i), zero), qa);
            const __m256i y = _mm256_min_epi16(_mm256_max_epi16(load(them + i), zero), qa);
            // x * w fits in 16 bits; madd then multiplies by x again and adds
            // neighbouring pairs into 32-bit lanes: x * x * w, exactly.
            sum_us = _mm256_add_epi32(sum_us, _mm256_madd_epi16(_mm256_mullo_epi16(x, load(w + i)), x));
            sum_them = _mm256_add_epi32(sum_them,
                                        _mm256_madd_epi16(_mm256_mullo_epi16(y, load(w + h + i)), y));
        }
        alignas(32) std::int32_t lanes[16];
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), sum_us);
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes + 8), sum_them);
        for (const std::int32_t l : lanes) sum += l;
    } else
#endif
    if (net.small_output) {
        // In 32 bits over chunks of 32 (see small_output), which vectorizes.
        for (int start = 0; start < h; start += 32) {
            const int end = std::min(start + 32, h);
            std::int32_t part = 0;
            for (int i = start; i < end; ++i) {
                const std::int32_t x = std::clamp<std::int32_t>(us[i], 0, QA);
                const std::int32_t y = std::clamp<std::int32_t>(them[i], 0, QA);
                part += x * x * w[i] + y * y * w[h + i];
            }
            sum += part;
        }
    } else {
        for (int i = 0; i < h; ++i) {
            const std::int64_t x = std::clamp<int>(us[i], 0, QA), y = std::clamp<int>(them[i], 0, QA);
            sum += x * x * w[i] + y * y * w[h + i];
        }
    }
    // Quantisation: QA x QA x QB on the products, QA x QB on the bias.
    const std::int64_t out = sum / QA + net.l1b[static_cast<std::size_t>(bucket)];
    return static_cast<int>(out * params::nnue_scale / (QA * QB));
}

int evaluate(const Position& pos) {
    static thread_local Accumulator acc;
    static thread_local ExtraLists extras;
    acc.extras = &extras;
    refresh(pos, acc);
    return evaluate(acc, pos);
}

void dump_layer1_inputs(std::FILE* f) { input_dump = f; }

std::string evaluation_path() {
#if defined(__AVX2__)
    constexpr bool avx2 = true;
#else
    constexpr bool avx2 = false;
#endif
#if defined(__AVX2__) && defined(__FMA__)
    constexpr bool fma = true;
#else
    constexpr bool fma = false;
#endif
    if (!net.layer1) return std::string("output layer ") + (avx2 && net.simd_output ? "AVX2" : "plain");
    const int l1 = net.layer1, l2 = net.layer2;
    const bool vector1 = avx2 && (net.hidden / 2) % 32 == 0;
    const bool sparse = vector1 && !net.q1s.empty() && (l1 / 8 == 1 || l1 / 8 == 2 || l1 / 8 == 4 || l1 / 8 == 8);
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    const std::string madd = " (VNNI)";
#else
    const std::string madd;
#endif
    return "layers " + std::to_string(l1) + "," + std::to_string(l2) + ": layer1 int8 " +
           (sparse ? "AVX2 sparse (nonzero 4-blocks)" + madd : vector1 ? "AVX2 dense" + madd : "plain") + ", layer2 " +
           (fma && l2 % 8 == 0 && !net.q2s.empty() && (l2 / 8 == 1 || l2 / 8 == 2 || l2 / 8 == 4 || l2 / 8 == 8)
                ? "int8 AVX2" + madd
                : fma && l2 % 8 == 0 ? "float AVX2/FMA" : "plain float") +
           (params::nnue_layer_float ? " (NnueLayerFloat=1: every layer plain float)" : "");
}

void active_inputs(const Position& pos, std::vector<int>& stm, std::vector<int>& ntm) {
    for (int view = 0; view < 2; ++view) {
        const Color v = static_cast<Color>(view);
        std::vector<int>& out = v == pos.side_to_move() ? stm : ntm;
        out.clear();
        const ViewKey key = view_key(v, king_square(pos, view), view_block(pos, view));
        for (int p = WP; p <= BK; ++p)
            for (Bitboard b = pos.pieces(static_cast<Piece>(p)); b; b &= b - 1)
                out.push_back(static_cast<int>(feature(v, key, static_cast<Piece>(p), std::countr_zero(b))));
        std::uint32_t pawn[MAX_PAWN_INPUTS];
        const int np = pawn_inputs(pos, view, key.flip, pawn);
        for (int i = 0; i < np; ++i) out.push_back(static_cast<int>(pawn[i]));
        std::uint32_t extras[MAX_EXTRA_INPUTS];
        const int n = extra_inputs(pos, v, key.flip, extras);
        for (int i = 0; i < n; ++i) out.push_back(static_cast<int>(net.extra_base + extras[i]));
        std::sort(out.begin(), out.end());
    }
}

}  // namespace nnue
