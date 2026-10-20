#include "pair_features.h"
#include "attacks.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>

namespace pairfeat {

namespace {

using Index = std::array<std::array<std::int16_t, 64>, 64>;

struct Tables {
    Index knight, ortho, diag, file;
    std::array<std::array<std::uint8_t, 2>, KNIGHT_PAIRS> knight_squares;
    Tables() {
        // Unordered pairs passing `test`, by lower square then higher.
        const auto number = [](Index& t, auto&& test) {
            for (auto& row : t) row.fill(-1);
            int n = 0;
            for (int a = 0; a < 64; ++a)
                for (int b = a + 1; b < 64; ++b)
                    if (test(a, b)) {
                        t[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] = static_cast<std::int16_t>(n);
                        t[static_cast<std::size_t>(b)][static_cast<std::size_t>(a)] = static_cast<std::int16_t>(n);
                        ++n;
                    }
        };
        number(knight, [](int a, int b) { return (knight_attacks(a) & bit(b)) != 0; });
        number(ortho, [](int a, int b) { return file_of(a) == file_of(b) || rank_of(a) == rank_of(b); });
        number(diag, [](int a, int b) {
            return std::abs(file_of(a) - file_of(b)) == std::abs(rank_of(a) - rank_of(b));
        });
        for (int a = 0; a < 64; ++a)
            for (int b = a + 1; b < 64; ++b) {
                const int k = knight[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)];
                if (k >= 0)
                    knight_squares[static_cast<std::size_t>(k)] = {static_cast<std::uint8_t>(a),
                                                                   static_cast<std::uint8_t>(b)};
            }
        // Ordered (slider, pawn) pairs, by slider square then pawn square.
        for (auto& row : file) row.fill(-1);
        int n = 0;
        for (int s = 0; s < 64; ++s)
            for (int p = 0; p < 64; ++p)
                if (p != s && file_of(p) == file_of(s) && rank_of(p) >= 1 && rank_of(p) <= 6)
                    file[static_cast<std::size_t>(s)][static_cast<std::size_t>(p)] = static_cast<std::int16_t>(n++);
    }
};

const Tables& tables() {
    static const Tables t;
    return t;
}

int at(const Index& t, int a, int b) { return t[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)]; }

int own_square(Color view, int square) { return view == Color::White ? square : square ^ 56; }

// Line control's diagonals of length 3 or more: 0-10 a1-h8-wise (file -
// rank + 7 from 2 to 12), 11-21 a8-h1-wise (file + rank from 2 to 12).
Bitboard diagonal_mask(int d) {
    static const std::array<Bitboard, 22> masks = [] {
        std::array<Bitboard, 22> m{};
        for (int s = 0; s < 64; ++s) {
            const int a = file_of(s) - rank_of(s) + 7, b = file_of(s) + rank_of(s);
            if (a >= 2 && a <= 12) m[static_cast<std::size_t>(a - 2)] |= bit(s);
            if (b >= 2 && b <= 12) m[static_cast<std::size_t>(11 + b - 2)] |= bit(s);
        }
        return m;
    }();
    return masks[static_cast<std::size_t>(d)];
}

// Squares above `a` (for counting each unordered pair once).
Bitboard above(int a) { return ~((Bitboard{1} << a << 1) - 1); }

}  // namespace

int knight_pair_index(int a, int b) { return at(tables().knight, a, b); }
int ortho_pair_index(int a, int b) { return at(tables().ortho, a, b); }
int diag_pair_index(int a, int b) { return at(tables().diag, a, b); }
int file_pair_index(int slider, int pawn) { return at(tables().file, slider, pawn); }

void knight_pair_squares(int index, int& a, int& b) {
    const auto& s = tables().knight_squares[static_cast<std::size_t>(index)];
    a = s[0];
    b = s[1];
}

int knight_pairs(Bitboard white, Bitboard black, std::uint16_t* out) {
    int n = 0;
    for (int colour = 0; colour < 2; ++colour) {
        const Bitboard knights = colour == 0 ? white : black;
        for (Bitboard b = knights; b; b &= b - 1) {
            const int a = std::countr_zero(b);
            for (Bitboard m = knight_attacks(a) & knights & above(a); m; m &= m - 1)
                if (n < FAMILY_MAX)
                    out[n++] = static_cast<std::uint16_t>(colour * KNIGHT_PAIRS +
                                                          knight_pair_index(a, std::countr_zero(m)));
        }
    }
    return n;
}

int block_size(unsigned families) {
    return (families & KnightPairs ? KNIGHT_FEATURES : 0) + (families & SliderPairs ? SLIDER_FEATURES : 0) +
           (families & SliderPawn ? SLIDER_PAWN_FEATURES : 0) + (families & ConnectedPairs ? SLIDER_FEATURES : 0) +
           (families & LineControl ? LINE_FEATURES : 0);
}

int inputs(const Position& pos, unsigned families, Color view, int flip, std::uint32_t* out) {
    int n = 0, offset = 0;
    const int us = static_cast<int>(view);
    const auto sq = [&](int s) { return own_square(view, s) ^ flip; };
    // One family's inputs, or none past its bound.
    const auto family = [&](auto&& each) {
        const int first = n;
        bool full = false;
        each([&](int input) {
            if (n - first == FAMILY_MAX) full = true;
            else out[n++] = static_cast<std::uint32_t>(offset + input);
        });
        if (full) n = first;
    };

    if (families & KnightPairs) {
        family([&](auto&& add) {
            for (int c = 0; c < 2; ++c) {
                const Bitboard knights = pos.pieces(c == 0 ? WN : BN);
                for (Bitboard b = knights; b; b &= b - 1) {
                    const int a = std::countr_zero(b);
                    for (Bitboard m = knight_attacks(a) & knights & above(a); m; m &= m - 1)
                        add((c != us) * KNIGHT_PAIRS + knight_pair_index(sq(a), sq(std::countr_zero(m))));
                }
            }
        });
        offset += KNIGHT_FEATURES;
    }

    // Slider pairs (B), and connected ones (D) below: every pair of sliders
    // sharing a line both move along, `connected` only with nothing between.
    struct Slider {
        int board_sq, sq, other, queen;
        bool orth, diag;
    };
    std::array<Slider, 64> s;
    int count = 0;
    if (families & (SliderPairs | ConnectedPairs))
        for (const Piece p : {WB, WR, WQ, BB, BR, BQ}) {
            const int t = type_of(p);
            for (Bitboard b = pos.pieces(p); b; b &= b - 1) {
                const int at = std::countr_zero(b);
                s[static_cast<std::size_t>(count++)] = {at, sq(at), (p < BP ? 0 : 1) != us, t == 4, t != 2, t != 3};
            }
        }
    const auto slider_pairs = [&](bool connected) {
        family([&](auto&& add) {
            for (int i = 0; i < count; ++i)
                for (int j = i + 1; j < count; ++j) {
                    Slider lo = s[static_cast<std::size_t>(i)], hi = s[static_cast<std::size_t>(j)];
                    if (connected && (squares_between(lo.board_sq, hi.board_sq) & pos.occupied())) continue;
                    if (lo.sq > hi.sq) std::swap(lo, hi);
                    const int kind = (lo.other * 2 + hi.other) * 4 + lo.queen * 2 + hi.queen;
                    const int o = ortho_pair_index(lo.sq, hi.sq);
                    const int d = diag_pair_index(lo.sq, hi.sq);
                    if (o >= 0 && lo.orth && hi.orth) add(kind * ORTHO_PAIRS + o);
                    else if (d >= 0 && lo.diag && hi.diag) add(16 * ORTHO_PAIRS + kind * DIAG_PAIRS + d);
                }
        });
        offset += SLIDER_FEATURES;
    };
    if (families & SliderPairs) slider_pairs(false);

    if (families & SliderPawn) {
        const Bitboard pawns = pos.pieces(WP) | pos.pieces(BP);
        family([&](auto&& add) {
            for (const Piece p : {WR, WQ, BR, BQ})
                for (Bitboard b = pos.pieces(p); b; b &= b - 1) {
                    const int sl = std::countr_zero(b);
                    const int kind = ((p < BP ? 0 : 1) != us) * 2 + (type_of(p) == 4);
                    for (Bitboard q = pawns & (FILE_A_BB << file_of(sl)); q; q &= q - 1) {
                        const int ps = std::countr_zero(q);
                        const int pawn_other = (pos.piece_at(ps) == WP ? 0 : 1) != us;
                        add((kind * 2 + pawn_other) * FILE_PAIRS + file_pair_index(sq(sl), sq(ps)));
                    }
                }
        });
        offset += SLIDER_PAWN_FEATURES;
    }

    if (families & ConnectedPairs) slider_pairs(true);

    if (families & LineControl) {
        // The board in the view's coordinates: flipped for Black, mirrored
        // left-right by `flip`.
        const auto view_bb = [&](Bitboard b) {
            if (view == Color::Black) b = __builtin_bswap64(b);
            if (flip) {
                b = ((b >> 1) & 0x5555555555555555ULL) | ((b & 0x5555555555555555ULL) << 1);
                b = ((b >> 2) & 0x3333333333333333ULL) | ((b & 0x3333333333333333ULL) << 2);
                b = ((b >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((b & 0x0F0F0F0F0F0F0F0FULL) << 4);
            }
            return b;
        };
        const Color them = opposite(view);
        const Bitboard own_pawns = view_bb(pos.pieces(view == Color::White ? WP : BP));
        const Bitboard their_pawns = view_bb(pos.pieces(them == Color::White ? WP : BP));
        const auto majors = [&](Color c) {
            return view_bb(pos.pieces(c == Color::White ? WR : BR) | pos.pieces(c == Color::White ? WQ : BQ));
        };
        const auto diagonals = [&](Color c) {
            return view_bb(pos.pieces(c == Color::White ? WB : BB) | pos.pieces(c == Color::White ? WQ : BQ));
        };
        const Bitboard own_major = majors(view), their_major = majors(them);
        const Bitboard own_diag = diagonals(view), their_diag = diagonals(them);
        const Bitboard pawns = own_pawns | their_pawns;
        family([&](auto&& add) {
            for (int x = 0; x < 8; ++x) {
                const Bitboard file = FILE_A_BB << x;
                const int ps = (own_pawns & file ? 2 : 0) + (their_pawns & file ? 1 : 0);
                const int ms = (own_major & file ? 2 : 0) + (their_major & file ? 1 : 0);
                if (ms || ps == 0) add(x * 16 + ps * 4 + ms);
            }
            for (int y = 0; y < 8; ++y) {
                const Bitboard rank = Bitboard{0xFF} << (8 * y);
                const int ms = (own_major & rank ? 2 : 0) + (their_major & rank ? 1 : 0);
                if (ms) add(128 + y * 4 + ms);
            }
            for (int d = 0; d < 22; ++d) {
                const Bitboard line = diagonal_mask(d);
                const int ds = (own_diag & line ? 2 : 0) + (their_diag & line ? 1 : 0);
                if (ds) add(160 + d * 8 + ds * 2 + !(pawns & line));
            }
        });
        offset += LINE_FEATURES;
    }

    std::sort(out, out + n);
    return n;
}

}  // namespace pairfeat
