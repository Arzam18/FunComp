#include "initiative.h"
#include "attacks.h"
#include "params.h"
#include "see.h"
#include <algorithm>
#include <bit>
#include <ostream>
#include <string>

namespace initiative {

namespace {

// The half of the board beyond the centre line, from each colour's point
// of view: ranks 5-8 for White, 1-4 for Black.
constexpr Bitboard ENEMY_HALF[2] = {0xFFFFFFFF00000000ULL, 0x00000000FFFFFFFFULL};

// Mobility units per safe square, by piece type: a knight's or bishop's
// square is worth more than a queen's, which has many.
constexpr int MOB_UNIT[6] = {0, 4, 3, 2, 1, 0};
// King-attack weight of a piece reaching the king zone, by type.
constexpr int KING_ATTACK_WEIGHT[6] = {0, 2, 2, 3, 5, 0};

constexpr int P = 0, N = 1, B = 2, R = 3, Q = 4, K = 5;

Bitboard pieces_of(const Position& pos, Color c, int pt) {
    return pos.pieces(static_cast<Piece>(piece_index(c, static_cast<PieceType>(pt))));
}

// The squares around `c`'s king plus the rank in front of them: on a
// castled king, that front rank is where attacks land.
Bitboard king_zone(Color c, int ksq) {
    if (ksq < 0) return 0;
    const Bitboard ring = king_attacks(ksq) | bit(ksq);
    return ring | pawn_push_set(c, ring);
}

// One colour's attack information. `by` holds direct attacks by piece type;
// `all` and `two` also count x-rays through our own batteries (a bishop
// behind our queen, a rook behind our rook or queen): a battery is
// coordination.
struct Side {
    Bitboard by[6] = {};
    Bitboard all = 0;
    Bitboard two = 0;        // attacked at least twice
    int mobility = 0;        // MOB_UNIT-weighted safe squares
    int king_attackers = 0;  // pieces reaching the enemy king zone
    int king_weight = 0;     // their KING_ATTACK_WEIGHT sum

    void add(Bitboard a) {
        two |= all & a;
        all |= a;
    }
};

struct Features {
    int mobility[2] = {};
    int territory[2] = {};
    int coordination[2] = {};
    int push_threats[2] = {};
    int safe_checks[2] = {};
    int king_danger[2] = {};  // before squaring; 0 with fewer than two attackers
    int threat1[2] = {};      // largest threat each colour poses, SEE_VALUE cp
    int threat2[2] = {};      // second largest
    int phase = 0;
};

struct Terms {
    int mobility, territory, coordination, push, checks, king;
    int latent, pending, doubled, tempo;
    int raw, clamped, final_score;
};

void compute_side(const Position& pos, Color c, Bitboard their_pawn_attacks, int their_king,
                  Side& s) {
    const Bitboard occ = pos.occupied();
    const Bitboard own = pos.occupancy(c);
    const Bitboard queens = pieces_of(pos, c, Q);
    const Bitboard rooks = pieces_of(pos, c, R);
    const Bitboard area = ~own & ~their_pawn_attacks;
    const Bitboard zone = king_zone(opposite(c), their_king);

    // Pawns: each diagonal separately, so a square both cover counts twice.
    const Bitboard pawns = pieces_of(pos, c, P);
    const Bitboard west = c == Color::White ? (pawns & ~FILE_A_BB) << 7 : (pawns & ~FILE_A_BB) >> 9;
    const Bitboard east = c == Color::White ? (pawns & ~FILE_H_BB) << 9 : (pawns & ~FILE_H_BB) >> 7;
    s.by[P] = west | east;
    s.add(west);
    s.add(east);

    auto piece = [&](int pt, Bitboard direct, Bitboard control) {
        s.by[pt] |= direct;
        s.add(control);
        s.mobility += MOB_UNIT[pt] * std::popcount(direct & area);
        if (control & zone) {
            ++s.king_attackers;
            s.king_weight += KING_ATTACK_WEIGHT[pt];
        }
    };

    for (Bitboard b = pieces_of(pos, c, N); b; b &= b - 1) {
        const Bitboard a = knight_attacks(std::countr_zero(b));
        piece(N, a, a);
    }
    for (Bitboard b = pieces_of(pos, c, B); b; b &= b - 1) {
        const int sq = std::countr_zero(b);
        const Bitboard a = bishop_attacks(sq, occ);
        const Bitboard x = queens ? bishop_attacks(sq, occ ^ queens) : a;
        piece(B, a, x);
    }
    for (Bitboard b = rooks; b; b &= b - 1) {
        const int sq = std::countr_zero(b);
        const Bitboard a = rook_attacks(sq, occ);
        const Bitboard behind = (queens | rooks) & ~bit(sq);
        const Bitboard x = behind ? rook_attacks(sq, occ ^ behind) : a;
        piece(R, a, x);
    }
    for (Bitboard b = queens; b; b &= b - 1) {
        const Bitboard a = queen_attacks(std::countr_zero(b), occ);
        piece(Q, a, a);
    }
    const Bitboard king = pieces_of(pos, c, K);
    if (king) {
        s.by[K] = king_attacks(std::countr_zero(king));
        s.add(s.by[K]);
    }
}

// The two largest gains `c` threatens against the other side's pieces: a
// piece attacked and undefended is worth its value; a defended one
// attacked by a cheaper piece, its value less the cheapest attacker's.
// Pieces are taken a set at a time: within a type, all the undefended ones
// gain the same, and so do all the defended ones with the same cheapest
// attacker, and only two of a kind can matter.
void threats(const Position& pos, Color c, const Side& us, const Side& them, int& t1, int& t2) {
    t1 = t2 = 0;
    auto add = [&](int gain, Bitboard set) {
        for (int k = 0; k < 2 && set; ++k, set &= set - 1) {
            if (gain > t1) {
                t2 = t1;
                t1 = gain;
            } else if (gain > t2) {
                t2 = gain;
            }
        }
    };
    const Color o = opposite(c);
    Bitboard cheaper = 0;  // attacked by our pieces of a type below pt
    for (int pt = P; pt <= Q; ++pt) {
        const Bitboard attacked = pieces_of(pos, o, pt) & us.all;
        if (attacked) {
            const int value = SEE_VALUE[pt];
            add(value, attacked & ~them.all);
            Bitboard rest = attacked & them.all & cheaper;
            for (int a = P; rest; ++a) {
                add(value - SEE_VALUE[a], rest & us.by[a]);
                rest &= ~us.by[a];
            }
        }
        cheaper |= us.by[pt];
    }
}

Features extract(const Position& pos) {
    Features f;
    const Bitboard occ = pos.occupied();
    Side side[2];
    int king_sq[2];
    Bitboard pawn_att[2];
    for (int c = 0; c < 2; ++c) {
        const Color col = static_cast<Color>(c);
        const Bitboard k = pieces_of(pos, col, K);
        king_sq[c] = k ? std::countr_zero(k) : -1;
        pawn_att[c] = pawn_attack_set(col, pieces_of(pos, col, P));
        for (int pt = N; pt <= Q; ++pt)
            f.phase += std::popcount(pieces_of(pos, col, pt)) * (pt == Q ? 4 : pt == R ? 2 : 1);
    }
    f.phase = std::min(f.phase, 24);
    for (int c = 0; c < 2; ++c)
        compute_side(pos, static_cast<Color>(c), pawn_att[c ^ 1], king_sq[c ^ 1], side[c]);

    for (int c = 0; c < 2; ++c) {
        const Color col = static_cast<Color>(c), opp = opposite(col);
        const Side& us = side[c];
        const Side& them = side[c ^ 1];
        const Bitboard their_units = pos.occupancy(opp) & ~pieces_of(pos, opp, K);
        // Squares we outnumber: attacked by us and not by them at all, or
        // attacked twice by us and at most once by them. One notion, used
        // for coordination, safe checks and the king zone alike.
        const Bitboard outnumbered = us.all & (~them.all | (us.two & ~them.two));

        f.mobility[c] = us.mobility;

        // Territory: squares in their half that we control and they don't
        // contest at all.
        f.territory[c] = std::popcount(us.all & ~them.all & ENEMY_HALF[c]);

        // Coordination: their pieces and pawns we hit at least twice while
        // they defend them at most once.
        f.coordination[c] = std::popcount(us.two & ~them.two & their_units);

        // Pawn-push threats: safe single or double pushes after which the
        // pawn attacks a piece (not a pawn, not the king) that no pawn of
        // ours attacks already.
        {
            const Bitboard pawns = pieces_of(pos, col, P);
            const Bitboard empty = ~occ;
            const Bitboard one = pawn_push_set(col, pawns) & empty;
            const Bitboard two =
                pawn_push_set(col, one & RANK_BB[c == 0 ? 2 : 5]) & empty;
            const Bitboard safe = (one | two) & ~them.by[P] & (us.all | ~them.all);
            const Bitboard targets = their_units & ~pieces_of(pos, opp, P) & ~us.by[P];
            f.push_threats[c] = std::popcount(pawn_attack_set(col, safe) & targets);
        }

        const int ksq = king_sq[c ^ 1];
        if (ksq >= 0) {
            // Safe checks: piece types that can check from an empty or enemy
            // square we outnumber. Counted once per type.
            const Bitboard safe = ~pos.occupancy(col) & outnumbered;
            const Bitboard diag = bishop_attacks(ksq, occ), orth = rook_attacks(ksq, occ);
            f.safe_checks[c] = ((knight_attacks(ksq) & us.by[N] & safe) != 0) +
                               ((diag & us.by[B] & safe) != 0) + ((orth & us.by[R] & safe) != 0) +
                               (((diag | orth) & us.by[Q] & safe) != 0);

            // King danger: only once at least two pieces take part, growing
            // with how many, plus the zone squares we outnumber; halved
            // without our queen.
            if (us.king_attackers >= 2) {
                const Bitboard zone = king_zone(opp, ksq);
                int danger = us.king_weight * (us.king_attackers - 1) +
                             std::popcount(zone & outnumbered);
                if (!pieces_of(pos, col, Q)) danger /= 2;
                f.king_danger[c] = danger;
            }
        }

        threats(pos, col, us, them, f.threat1[c], f.threat2[c]);
    }
    return f;
}

Terms combine(const Features& f, Color stm) {
    const int u = static_cast<int>(stm), t = u ^ 1;
    Terms s{};
    s.mobility = (f.mobility[u] - f.mobility[t]) * params::init_mobility / 16;
    s.territory = (f.territory[u] - f.territory[t]) * params::init_territory / 4;
    s.coordination = (f.coordination[u] - f.coordination[t]) * params::init_coordination;
    s.push = (f.push_threats[u] - f.push_threats[t]) * params::init_push_threat;
    s.checks = (f.safe_checks[u] - f.safe_checks[t]) * params::init_safe_check;
    auto king = [&](int c) {
        const int d = f.king_danger[c];
        return params::init_king_pressure * d * d / 256 * f.phase / 24;
    };
    s.king = king(u) - king(t);
    s.latent = (f.threat1[u] + f.threat2[u]) * params::init_threat_latent / 256;
    s.pending = -f.threat1[t] * params::init_threat_pending / 256;
    s.doubled = -f.threat2[t] * params::init_threat_double / 256;
    s.tempo = params::init_tempo;
    s.raw = s.mobility + s.territory + s.coordination + s.push + s.checks + s.king + s.latent +
            s.pending + s.doubled + s.tempo;
    s.clamped = std::clamp(s.raw, -params::init_clamp, params::init_clamp);
    s.final_score = s.clamped * params::init_scale / 100;
    if (params::init_endgame_pct != 100)
        s.final_score = s.final_score *
                        (f.phase * 100 + (24 - f.phase) * params::init_endgame_pct) / 2400;
    return s;
}

}  // namespace

int evaluate(const Position& pos) {
    if (params::init_scale == 0) return 0;
    return combine(extract(pos), pos.side_to_move()).final_score;
}

void print_trace(const Position& pos, std::ostream& out) {
    const Features f = extract(pos);
    const Color stm = pos.side_to_move();
    const int u = static_cast<int>(stm), t = u ^ 1;
    const Terms s = combine(f, stm);
    auto row = [&](const std::string& name, int a, int b, int cp) {
        out << "info string   " << name << std::string(name.size() < 14 ? 14 - name.size() : 1, ' ')
            << "stm " << a << "  other " << b << "  -> " << cp << " cp\n";
    };
    out << "info string initiative (side to move: " << (stm == Color::White ? "white" : "black")
        << ", phase " << f.phase << ")\n";
    row("mobility", f.mobility[u], f.mobility[t], s.mobility);
    row("territory", f.territory[u], f.territory[t], s.territory);
    row("coordination", f.coordination[u], f.coordination[t], s.coordination);
    row("push threats", f.push_threats[u], f.push_threats[t], s.push);
    row("safe checks", f.safe_checks[u], f.safe_checks[t], s.checks);
    row("king danger", f.king_danger[u], f.king_danger[t], s.king);
    row("threat 1st", f.threat1[u], f.threat1[t], s.pending);
    row("threat 2nd", f.threat2[u], f.threat2[t], s.doubled);
    out << "info string   latent (the side to move's two threats) " << s.latent << " cp; tempo "
        << s.tempo << " cp\n";
    out << "info string   raw " << s.raw << "  clamped " << s.clamped << "  x" << params::init_scale
        << "%";
    if (params::init_endgame_pct != 100)
        out << ", tapered toward " << params::init_endgame_pct << "% (phase " << f.phase << ")";
    out << " = " << s.final_score << " cp" << std::endl;
}

}  // namespace initiative
