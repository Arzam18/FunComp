#include "featstats.h"
#include "attacks.h"
#include "datagen.h"
#include "movegen.h"
#include "pawn_features.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <ostream>
#include <vector>

namespace {

using Set = std::vector<std::uint32_t>;  // a family's active features, sorted

Bitboard file_bb(int f) { return FILE_A_BB << f; }

Piece piece_of(Color c, int type) { return static_cast<Piece>(piece_index(c, static_cast<PieceType>(type))); }

// Whether a stored move is legal here (a corrupt file may hold anything).
bool is_legal_move(const Position& pos, Move m) {
    MoveList moves;
    int count = 0;
    generate_legal(pos, moves, count);
    for (int i = 0; i < count; ++i)
        if (moves[static_cast<std::size_t>(i)] == m) return true;
    return false;
}

// Pieces of `c` (pawns excluded) attacking `sq`.
Bitboard piece_attackers(const Position& pos, Color c, int sq) {
    const Bitboard occ = pos.occupied();
    const Bitboard diag = pos.pieces(piece_of(c, 2)) | pos.pieces(piece_of(c, 4));
    const Bitboard orth = pos.pieces(piece_of(c, 3)) | pos.pieces(piece_of(c, 4));
    return (knight_attacks(sq) & pos.pieces(piece_of(c, 1))) | (bishop_attacks(sq, occ) & diag) |
           (rook_attacks(sq, occ) & orth) | (king_attacks(sq) & pos.pieces(piece_of(c, 5)));
}

// `c`'s pieces pinned to their king by an enemy slider.
Bitboard pinned(const Position& pos, Color c) {
    const Bitboard king = pos.pieces(piece_of(c, 5));
    if (!king) return 0;
    const int k = std::countr_zero(king);
    const Color o = opposite(c);
    const Bitboard occ = pos.occupied();
    const Bitboard diag = pos.pieces(piece_of(o, 2)) | pos.pieces(piece_of(o, 4));
    const Bitboard orth = pos.pieces(piece_of(o, 3)) | pos.pieces(piece_of(o, 4));
    Bitboard snipers = (bishop_attacks(k, 0) & diag) | (rook_attacks(k, 0) & orth);
    Bitboard result = 0;
    for (; snipers; snipers &= snipers - 1) {
        const Bitboard between = squares_between(k, std::countr_zero(snipers)) & occ;
        if (std::popcount(between) == 1 && (between & pos.occupancy(c))) result |= between;
    }
    return result;
}

// A1: the pawn predicates of pawn_features.h. A2: 4 more, piece-dependent.
constexpr int A1_COUNT = pawnfeat::COUNT;
enum A2 { SupportedByPiece, PinnedSupportOnly, AttackedByPiece, BlockedByPiece, A2_COUNT };
constexpr const char* A2_NAMES[A2_COUNT] = {"supported-by-piece", "pinned-support-only",
                                            "attacked-by-piece", "blocked-by-piece"};

std::uint32_t pawn_feature(int predicate, int colour, int sq) {
    return static_cast<std::uint32_t>((predicate * 2 + colour) * 64 + sq);
}

void extract_a1(const Position& pos, Set& out) {
    std::uint16_t f[pawnfeat::MAX_ACTIVE];
    const int n = pawnfeat::features(pos.pieces(WP), pos.pieces(BP), f);
    out.insert(out.end(), f, f + n);
}

void extract_a2(const Position& pos, Set& out) {
    for (int ci = 0; ci < 2; ++ci) {
        const Color c = static_cast<Color>(ci), o = opposite(c);
        const Bitboard own = pos.pieces(piece_of(c, 0));
        const Bitboard pins = pinned(pos, c);
        const Bitboard pieces = pos.occupied() & ~(own | pos.pieces(piece_of(o, 0)));
        for (Bitboard b = own; b; b &= b - 1) {
            const int s = std::countr_zero(b);
            const Bitboard defenders = pawn_attacks(o, s) & own;
            const bool add[A2_COUNT] = {
                piece_attackers(pos, c, s) != 0,
                defenders && !(defenders & ~pins),
                piece_attackers(pos, o, s) != 0,
                (pawn_push_set(c, bit(s)) & pieces) != 0,
            };
            for (int p = 0; p < A2_COUNT; ++p)
                if (add[p]) out.push_back(pawn_feature(p, ci, s));
        }
    }
}

void extract_e(const Position& pos, Set& out) {
    for (int ci = 0; ci < 2; ++ci) {
        const Bitboard knights = pos.pieces(piece_of(static_cast<Color>(ci), 1));
        for (Bitboard b = knights; b; b &= b - 1) {
            const int a = std::countr_zero(b);
            for (Bitboard m = knight_attacks(a) & knights; m; m &= m - 1) {
                const int z = std::countr_zero(m);
                if (z > a) out.push_back(static_cast<std::uint32_t>(ci * 4096 + a * 64 + z));
            }
        }
    }
}

// Whether two sliders of these types share a line both move along.
bool slider_line(int ta, int sa, int tb, int sb) {
    const bool orth_a = ta != 2, orth_b = tb != 2;  // queen or rook
    const bool diag_a = ta != 3, diag_b = tb != 3;  // queen or bishop
    const int df = file_of(sa) - file_of(sb), dr = rank_of(sa) - rank_of(sb);
    if (df == 0 || dr == 0) return orth_a && orth_b;
    if (df == dr || df == -dr) return diag_a && diag_b;
    return false;
}

std::uint32_t pair_feature(Piece pa, int sa, Piece pb, int sb) {
    if (pa * 64 + sa > pb * 64 + sb) std::swap(pa, pb), std::swap(sa, sb);
    return static_cast<std::uint32_t>(((pa * 64 + sa) * 12 + pb) * 64 + sb);
}

template <bool CONNECTED>
void extract_sliders(const Position& pos, Set& out) {
    struct S { Piece p; int sq; int type; };
    std::array<S, 32> s;
    int n = 0;
    for (int p = WP; p <= BK; ++p) {
        const int type = type_of(p);
        if (type < 2 || type > 4) continue;
        for (Bitboard b = pos.pieces(static_cast<Piece>(p)); b; b &= b - 1)
            s[static_cast<std::size_t>(n++)] = {static_cast<Piece>(p), std::countr_zero(b), type};
    }
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            const S& a = s[static_cast<std::size_t>(i)];
            const S& b = s[static_cast<std::size_t>(j)];
            if (!slider_line(a.type, a.sq, b.type, b.sq)) continue;
            if (CONNECTED && (squares_between(a.sq, b.sq) & pos.occupied())) continue;
            out.push_back(pair_feature(a.p, a.sq, b.p, b.sq));
        }
}

void extract_c(const Position& pos, Set& out) {
    const Bitboard pawns = pos.pieces(WP) | pos.pieces(BP);
    for (const Piece p : {WR, WQ, BR, BQ})
        for (Bitboard b = pos.pieces(p); b; b &= b - 1) {
            const int sq = std::countr_zero(b);
            for (Bitboard m = pawns & file_bb(file_of(sq)); m; m &= m - 1) {
                const int ps = std::countr_zero(m);
                out.push_back(static_cast<std::uint32_t>(((p * 64 + sq) * 2 + (pos.piece_at(ps) == BP)) * 64 + ps));
            }
        }
}

// Each side's attacks by piece type.
std::array<Bitboard, 6> attacks_by_type(const Position& pos, Color c) {
    std::array<Bitboard, 6> by{};
    const Bitboard occ = pos.occupied();
    by[0] = pawn_attack_set(c, pos.pieces(piece_of(c, 0)));
    for (int t = 1; t <= 5; ++t)
        for (Bitboard b = pos.pieces(piece_of(c, t)); b; b &= b - 1) {
            const int sq = std::countr_zero(b);
            by[static_cast<std::size_t>(t)] |= t == 1 ? knight_attacks(sq)
                                               : t == 2 ? bishop_attacks(sq, occ)
                                               : t == 3 ? rook_attacks(sq, occ)
                                               : t == 4 ? queen_attacks(sq, occ)
                                                        : king_attacks(sq);
        }
    return by;
}

void extract_s1(const Position& pos, Set& out) {
    for (int ci = 0; ci < 2; ++ci) {
        Bitboard all = 0;
        for (const Bitboard b : attacks_by_type(pos, static_cast<Color>(ci))) all |= b;
        for (; all; all &= all - 1) out.push_back(static_cast<std::uint32_t>(ci * 64 + std::countr_zero(all)));
    }
}

void extract_s2(const Position& pos, Set& out) {
    for (int ci = 0; ci < 2; ++ci) {
        const auto by = attacks_by_type(pos, static_cast<Color>(ci));
        Bitboard done = 0;
        for (int t = 0; t <= 5; ++t) {
            for (Bitboard b = by[static_cast<std::size_t>(t)] & ~done; b; b &= b - 1)
                out.push_back(static_cast<std::uint32_t>((ci * 64 + std::countr_zero(b)) * 6 + t));
            done |= by[static_cast<std::size_t>(t)];
        }
    }
}

void extract_s3(const Position& pos, Set& out) {
    for (int ci = 0; ci < 2; ++ci) {
        const Color c = static_cast<Color>(ci), o = opposite(c);
        const Bitboard king = pos.pieces(piece_of(o, 5));
        if (!king) continue;
        const Bitboard ring = king_attacks(std::countr_zero(king)) | king;
        const Bitboard zone = ring | pawn_push_set(o, ring);
        Bitboard all = 0;
        for (const Bitboard b : attacks_by_type(pos, c)) all |= b;
        for (Bitboard b = all & zone; b; b &= b - 1)
            out.push_back(static_cast<std::uint32_t>(ci * 64 + std::countr_zero(b)));
    }
}

struct Family {
    const char* name;
    std::uint32_t ids;  // feature ids are below this
    void (*extract)(const Position&, Set&);
    int predicates;            // A1 and A2: predicates counted separately
    const char* const* names;  // their names
};

const Family FAMILIES[] = {
    {"A1 pawn predicates", A1_COUNT * 128, extract_a1, A1_COUNT, pawnfeat::NAMES},
    {"A2 piece-dependent", A2_COUNT * 128, extract_a2, A2_COUNT, A2_NAMES},
    {"E knight pairs", 2 * 4096, extract_e, 0, nullptr},
    {"B slider pairs", 12 * 64 * 12 * 64, extract_sliders<false>, 0, nullptr},
    {"C slider x pawn", 12 * 64 * 2 * 64, extract_c, 0, nullptr},
    {"D connected pairs", 12 * 64 * 12 * 64, extract_sliders<true>, 0, nullptr},
    {"S1 attacked", 128, extract_s1, 0, nullptr},
    {"S2 least attacker", 768, extract_s2, 0, nullptr},
    {"S3 king zone", 128, extract_s3, 0, nullptr},
};

// The family's possible features: those some legal-looking placement can
// produce (by enumeration over its id space).
std::uint64_t possible(std::size_t family) {
    std::uint64_t n = 0;
    switch (family) {
        case 0: case 1:  // pawns on ranks 2-7
            return static_cast<std::uint64_t>(FAMILIES[family].predicates) * 2 * 48;
        case 2:
            for (int a = 0; a < 64; ++a) n += static_cast<std::uint64_t>(std::popcount(knight_attacks(a) & ~((bit(a) << 1) - 1)));
            return 2 * n;
        case 3: case 5:
            for (int ta = 2; ta <= 4; ++ta)
                for (int tb = 2; tb <= 4; ++tb)
                    for (int sa = 0; sa < 64; ++sa)
                        for (int sb = 0; sb < 64; ++sb)
                            if (sa != sb && slider_line(ta, sa, tb, sb)) n += 4;  // 4 colour pairs
            return n / 2;  // unordered
        case 4:
            return 4 * 64 * 2 * 6;  // slider (4 kinds) x square x pawn colour x pawn rank on the file
        case 6: case 8: return 128;
        case 7: return 768;
    }
    return 0;
}

struct Stats {
    std::uint64_t positions = 0, active = 0, any = 0, changes = 0, moves = 0;
    std::vector<char> seen;
    std::vector<std::uint64_t> by_predicate;
};

std::uint64_t symmetric_difference(const Set& a, const Set& b) {
    std::uint64_t n = 0;
    std::size_t i = 0, j = 0;
    while (i < a.size() || j < b.size()) {
        if (j == b.size() || (i < a.size() && a[i] < b[j])) ++n, ++i;
        else if (i == a.size() || b[j] < a[i]) ++n, ++j;
        else ++i, ++j;
    }
    return n;
}

}  // namespace

void feature_stats(const std::string& path, std::uint64_t max_games, std::ostream& out) {
    constexpr std::size_t F = std::size(FAMILIES);
    std::array<Stats, F> stats;
    for (std::size_t f = 0; f < F; ++f) {
        stats[f].seen.assign(FAMILIES[f].ids, 0);
        stats[f].by_predicate.assign(static_cast<std::size_t>(FAMILIES[f].predicates), 0);
    }
    std::uint64_t games = 0;
    for (const std::string& file : data_files(path)) {
        std::ifstream in(file, std::ios::binary);
        DataGame g;
        while ((!max_games || games < max_games) && read_game(in, g)) {
            if (g.end == DataEnd::Filtered) continue;
            ++games;
            Position pos;
            pos.set_fen(g.fen);
            std::vector<StateInfo> st(g.random_moves.size() + g.records.size() + 1);
            std::size_t made = 0;
            for (const Move m : g.random_moves) pos.make_move(m, st[made++]);
            std::array<Set, F> prev;
            for (std::size_t r = 0; r < g.records.size(); ++r) {
                for (std::size_t f = 0; f < F; ++f) {
                    Set cur;
                    FAMILIES[f].extract(pos, cur);
                    std::sort(cur.begin(), cur.end());
                    Stats& s = stats[f];
                    ++s.positions;
                    s.active += cur.size();
                    s.any += !cur.empty();
                    for (const std::uint32_t id : cur) {
                        s.seen[id] = 1;
                        if (FAMILIES[f].predicates) ++s.by_predicate[id / 128];
                    }
                    if (r > 0) {
                        s.changes += symmetric_difference(prev[f], cur);
                        ++s.moves;
                    }
                    prev[f] = std::move(cur);
                }
                Move m;
                m.data = g.records[r].move;
                if (!is_legal_move(pos, m)) break;  // a corrupt file: stop this game cleanly
                pos.make_move(m, st[made++]);
            }
        }
    }

    char line[200];
    out << "info string featstats: " << games << " games, " << stats[0].positions << " positions\n";
    std::snprintf(line, sizeof line, "%-20s %10s %10s %10s %10s %12s\n", "family", "possible", "seen",
                  "active", "any (%)", "changes/move");
    out << line;
    for (std::size_t f = 0; f < F; ++f) {
        const Stats& s = stats[f];
        const auto seen = static_cast<std::uint64_t>(std::count(s.seen.begin(), s.seen.end(), 1));
        std::snprintf(line, sizeof line, "%-20s %10llu %10llu %10.2f %10.1f %12.2f\n", FAMILIES[f].name,
                      static_cast<unsigned long long>(possible(f)), static_cast<unsigned long long>(seen),
                      s.positions ? static_cast<double>(s.active) / static_cast<double>(s.positions) : 0.0,
                      s.positions ? 100.0 * static_cast<double>(s.any) / static_cast<double>(s.positions) : 0.0,
                      s.moves ? static_cast<double>(s.changes) / static_cast<double>(s.moves) : 0.0);
        out << line;
    }
    for (std::size_t f = 0; f < 2; ++f) {
        out << FAMILIES[f].name << ", active per position by predicate:";
        for (int p = 0; p < FAMILIES[f].predicates; ++p) {
            std::snprintf(line, sizeof line, " %s %.3f", FAMILIES[f].names[p],
                          stats[f].positions ? static_cast<double>(stats[f].by_predicate[static_cast<std::size_t>(p)]) /
                                                   static_cast<double>(stats[f].positions)
                                             : 0.0);
            out << line;
        }
        out << '\n';
    }
    out << std::flush;
}
