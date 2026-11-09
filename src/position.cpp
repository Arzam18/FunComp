#include "position.h"
#include "attacks.h"
#include "movegen.h"
#include <algorithm>
#include <bit>
#include <cassert>
#include <charconv>
#include <sstream>

namespace {
std::uint64_t zobrist_piece[12][64];
std::uint64_t zobrist_castle[16];
std::uint64_t zobrist_ep[65];
std::uint64_t zobrist_side;

bool zobrist_initialized = false;

// castling_ &= castle_mask[from] & castle_mask[to] clears exactly the rights
// lost when a king or rook leaves its home square or a rook is captured there.
constexpr std::array<int, 64> make_castle_mask() {
    std::array<int, 64> m{};
    for (int& x : m) x = 15;
    m[sq(4, 0)] = 15 & ~(Position::WK_CASTLE | Position::WQ_CASTLE);
    m[sq(0, 0)] = 15 & ~Position::WQ_CASTLE;
    m[sq(7, 0)] = 15 & ~Position::WK_CASTLE;
    m[sq(4, 7)] = 15 & ~(Position::BK_CASTLE | Position::BQ_CASTLE);
    m[sq(0, 7)] = 15 & ~Position::BQ_CASTLE;
    m[sq(7, 7)] = 15 & ~Position::BK_CASTLE;
    return m;
}
constexpr std::array<int, 64> castle_mask = make_castle_mask();

constexpr int color_index(Piece p) { return p < BP ? 0 : 1; }

std::uint64_t splitmix64(std::uint64_t& x) {
    x += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

char piece_char(Piece p) {
    static constexpr char chars[] = "PNBRQKpnbrqk";
    return p == NO_PIECE ? '.' : chars[p];
}

Piece char_piece(char c) {
    switch (c) {
        case 'P': return WP; case 'N': return WN; case 'B': return WB;
        case 'R': return WR; case 'Q': return WQ; case 'K': return WK;
        case 'p': return BP; case 'n': return BN; case 'b': return BB;
        case 'r': return BR; case 'q': return BQ; case 'k': return BK;
        default: return NO_PIECE;
    }
}
}

namespace {
// The reversible-move table behind upcoming-repetition detection: for every
// move a knight, bishop, rook, queen or king can make on an empty board, the
// change it makes to the position key (the piece leaving one square and
// arriving on the other, and the side to move flipping; the same whichever
// way the piece goes), with the two squares, whose path must be empty for
// the move to be possible. An open-addressing hash set: a key's home slot
// comes from Fibonacci hashing (a multiply by 2^64 / golden ratio, keeping
// the top bits), and a taken slot passes on to the next. 3,668 keys in 8,192
// slots keep probe runs short. A zero key marks an empty slot; no real
// change is zero.
struct ReversibleMove {
    std::uint64_t key = 0;
    std::uint8_t a = 0, b = 0;  // the two squares the move joins
};
constexpr int REVERSIBLE_BITS = 13;
std::array<ReversibleMove, std::size_t{1} << REVERSIBLE_BITS> reversible_moves{};

constexpr std::size_t reversible_slot(std::uint64_t key) {
    return static_cast<std::size_t>((key * 0x9E3779B97F4A7C15ULL) >> (64 - REVERSIBLE_BITS));
}

void init_reversible_moves() {
    init_attacks();
    constexpr PieceType types[] = {PieceType::Knight, PieceType::Bishop, PieceType::Rook,
                                   PieceType::Queen, PieceType::King};
    const std::size_t mask = reversible_moves.size() - 1;
    for (const Color c : {Color::White, Color::Black}) {
        for (const PieceType pt : types) {
            const int piece = piece_index(c, pt);
            for (int a = 0; a < 64; ++a) {
                Bitboard reach = pt == PieceType::Knight   ? knight_attacks(a)
                                 : pt == PieceType::Bishop ? bishop_attacks(a, 0)
                                 : pt == PieceType::Rook   ? rook_attacks(a, 0)
                                 : pt == PieceType::Queen  ? queen_attacks(a, 0)
                                                           : king_attacks(a);
                // Each pair once, the higher square as b. (bit(a) << 1 is 0 for
                // a = 63, where bit(a + 1) would be an undefined shift.)
                reach &= ~((bit(a) << 1) - 1);
                for (; reach; reach &= reach - 1) {
                    const int b = std::countr_zero(reach);
                    const std::uint64_t key =
                        zobrist_piece[piece][a] ^ zobrist_piece[piece][b] ^ zobrist_side;
                    std::size_t slot = reversible_slot(key);
                    while (reversible_moves[slot].key != 0) slot = (slot + 1) & mask;
                    reversible_moves[slot] = {key, static_cast<std::uint8_t>(a),
                                              static_cast<std::uint8_t>(b)};
                }
            }
        }
    }
}
}  // namespace

void init_zobrist() {
    if (zobrist_initialized) return;
    std::uint64_t seed = 0x123456789abcdef0ULL;
    for (auto& p : zobrist_piece)
        for (auto& x : p) x = splitmix64(seed);
    for (auto& x : zobrist_castle) x = splitmix64(seed);
    for (auto& x : zobrist_ep) x = splitmix64(seed);
    zobrist_side = splitmix64(seed);
    init_reversible_moves();
    zobrist_initialized = true;
}

bool Position::reversible_move_to(std::uint64_t other) const {
    const std::uint64_t change = key_ ^ other;
    const std::size_t mask = reversible_moves.size() - 1;
    for (std::size_t slot = reversible_slot(change); reversible_moves[slot].key != 0;
         slot = (slot + 1) & mask) {
        const ReversibleMove& r = reversible_moves[slot];
        if (r.key == change) return !(squares_between(r.a, r.b) & occupied_);
    }
    return false;
}

Position::Position() {
    init_zobrist();
    init_attacks();
    set_startpos();
}

void Position::clear() {
    bb_.fill(0);
    color_bb_.fill(0);
    board_.fill(NO_PIECE);
    occupied_ = 0;
    side_ = Color::White;
    castling_ = 0;
    ep_square_ = -1;
    halfmove_clock_ = 0;
    fullmove_ = 1;
    key_ = 0;
    pawn_key_ = 0;
    nonpawn_key_[0] = nonpawn_key_[1] = 0;
}

void Position::rebuild_occupancy() {
    color_bb_.fill(0);
    for (int p = WP; p <= WK; ++p) color_bb_[0] |= bb_[p];
    for (int p = BP; p <= BK; ++p) color_bb_[1] |= bb_[p];
    occupied_ = color_bb_[0] | color_bb_[1];
}

std::uint64_t Position::compute_key() const {
    std::uint64_t k = 0;
    for (int p = WP; p <= BK; ++p) {
        Bitboard b = bb_[p];
        while (b) {
            const int s = std::countr_zero(b);
            k ^= zobrist_piece[p][s];
            b &= b - 1;
        }
    }
    k ^= zobrist_castle[castling_];
    k ^= zobrist_ep[ep_square_ < 0 ? 64 : ep_square_];
    if (side_ == Color::Black) k ^= zobrist_side;
    return k;
}

std::uint64_t Position::compute_pawn_key() const {
    std::uint64_t k = 0;
    for (int p : {WP, BP}) {
        for (Bitboard b = bb_[p]; b; b &= b - 1)
            k ^= zobrist_piece[p][std::countr_zero(b)];
    }
    return k;
}

std::uint64_t Position::compute_nonpawn_key(int c) const {
    std::uint64_t k = 0;
    const int first = c == 0 ? WN : BN, last = c == 0 ? WK : BK;
    for (int p = first; p <= last; ++p) {
        for (Bitboard b = bb_[p]; b; b &= b - 1)
            k ^= zobrist_piece[p][std::countr_zero(b)];
    }
    return k;
}

namespace {
constexpr bool is_pawn(Piece p) { return p == WP || p == BP; }
}

template <bool UpdateKey>
void Position::put_piece(Piece p, int square) {
    const Bitboard b = bit(square);
    bb_[p] |= b;
    color_bb_[color_index(p)] |= b;
    occupied_ |= b;
    board_[square] = p;
    if constexpr (UpdateKey) {
        key_ ^= zobrist_piece[p][square];
        if (is_pawn(p)) pawn_key_ ^= zobrist_piece[p][square];
        else nonpawn_key_[color_index(p)] ^= zobrist_piece[p][square];
    }
}

template <bool UpdateKey>
void Position::remove_piece(int square) {
    const Piece p = board_[square];
    assert(p != NO_PIECE);
    const Bitboard b = bit(square);
    bb_[p] &= ~b;
    color_bb_[color_index(p)] &= ~b;
    occupied_ &= ~b;
    board_[square] = NO_PIECE;
    if constexpr (UpdateKey) {
        key_ ^= zobrist_piece[p][square];
        if (is_pawn(p)) pawn_key_ ^= zobrist_piece[p][square];
        else nonpawn_key_[color_index(p)] ^= zobrist_piece[p][square];
    }
}

template <bool UpdateKey>
void Position::move_piece(int from, int to) {
    const Piece p = board_[from];
    assert(p != NO_PIECE && board_[to] == NO_PIECE);
    const Bitboard from_to = bit(from) | bit(to);
    bb_[p] ^= from_to;
    color_bb_[color_index(p)] ^= from_to;
    occupied_ ^= from_to;
    board_[from] = NO_PIECE;
    board_[to] = p;
    if constexpr (UpdateKey) {
        const std::uint64_t k = zobrist_piece[p][from] ^ zobrist_piece[p][to];
        key_ ^= k;
        if (is_pawn(p)) pawn_key_ ^= k;
        else nonpawn_key_[color_index(p)] ^= k;
    }
}

bool Position::ep_capturable(int ep, Color capturer) const {
    // Pawns of `capturer` attack `ep` from exactly the squares a pawn of the
    // other color standing on `ep` would attack.
    return (pawn_attacks(opposite(capturer), ep) &
            bb_[piece_index(capturer, PieceType::Pawn)]) != 0;
}

void Position::set_startpos() {
    // Keep the call outside assert() so it still runs when NDEBUG is defined.
    const bool ok = set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    assert(ok && "start position FEN failed to parse");
    (void)ok;
}

bool Position::set_fen(const std::string& fen) {
    init_zobrist();

    // Parse everything into locals first. The position is only modified once
    // the whole FEN has been validated, so a bad FEN leaves *this untouched
    // instead of half-built.
    std::istringstream ss(fen);
    std::string board, stm, castle, ep;
    // The two clock fields may be left out (as in EPD): they default to 0 1.
    if (!(ss >> board >> stm >> castle >> ep)) return false;
    int halfmove = 0, fullmove = 1;
    std::string clock;
    if (ss >> clock) {
        std::istringstream cs(clock);
        if (!(cs >> halfmove) || !cs.eof()) return false;
        if (ss >> clock) {
            std::istringstream fs(clock);
            if (!(fs >> fullmove) || !fs.eof()) return false;
        }
    }

    std::array<Bitboard, 12> new_bb{};
    int rank = 7, file = 0;
    for (char c : board) {
        if (c == '/') {
            if (file != 8 || rank == 0) return false;
            --rank; file = 0;
        } else if (c >= '1' && c <= '8') {
            file += c - '0';
            if (file > 8) return false;
        } else {
            Piece p = char_piece(c);
            if (p == NO_PIECE || file >= 8 || rank < 0) return false;
            new_bb[p] |= bit(sq(file, rank));
            ++file;
        }
    }
    if (rank != 0 || file != 8) return false;

    Color new_side;
    if (stm == "w") new_side = Color::White;
    else if (stm == "b") new_side = Color::Black;
    else return false;

    int new_castling = 0;
    for (char c : castle) {
        if (c == 'K') new_castling |= WK_CASTLE;
        else if (c == 'Q') new_castling |= WQ_CASTLE;
        else if (c == 'k') new_castling |= BK_CASTLE;
        else if (c == 'q') new_castling |= BQ_CASTLE;
        else if (c != '-') return false;
    }

    int new_ep = -1;
    if (ep != "-") {
        if (ep.size() != 2 || ep[0] < 'a' || ep[0] > 'h' || ep[1] < '1' || ep[1] > '8')
            return false;
        new_ep = sq(ep[0] - 'a', ep[1] - '1');
    }

    if (halfmove < 0) return false;

    // Everything validated: commit.
    clear();
    bb_ = new_bb;
    side_ = new_side;
    castling_ = new_castling;
    ep_square_ = new_ep;
    halfmove_clock_ = halfmove;
    fullmove_ = std::max(fullmove, 1);
    for (int p = WP; p <= BK; ++p) {
        Bitboard b = bb_[p];
        while (b) {
            board_[std::countr_zero(b)] = static_cast<Piece>(p);
            b &= b - 1;
        }
    }
    rebuild_occupancy();
    if (ep_square_ >= 0 && !ep_capturable(ep_square_, side_)) ep_square_ = -1;
    key_ = compute_key();
    pawn_key_ = compute_pawn_key();
    for (int c = 0; c < 2; ++c) nonpawn_key_[c] = compute_nonpawn_key(c);
    checkers_ = compute_checkers();
    return true;
}

bool Position::square_attacked(int square, Color by) const {
    return (attackers_to(square, occupied_) & color_bb_[static_cast<int>(by)]) != 0;
}

bool Position::in_check_slow(Color c) const {
    Bitboard k = bb_[piece_index(c, PieceType::King)];
    if (!k) return true;
    return square_attacked(std::countr_zero(k), opposite(c));
}

Bitboard Position::compute_checkers() const {
    const Bitboard k = bb_[piece_index(side_, PieceType::King)];
    if (!k) return 0;
    return attackers_to(std::countr_zero(k), occupied_) &
           color_bb_[static_cast<int>(opposite(side_))];
}

void Position::apply_move(Move move, StateInfo& st) {
    st.key = key_;
    st.pawn_key = pawn_key_;
    st.nonpawn_key[0] = nonpawn_key_[0];
    st.nonpawn_key[1] = nonpawn_key_[1];
    st.changes.count = 0;
    st.castling = castling_;
    st.ep_square = ep_square_;
    st.halfmove_clock = halfmove_clock_;
    st.checkers = checkers_;

    const int from = move.from(), to = move.to();
    const Piece moving = board_[from];
    assert(moving != NO_PIECE);
    const Color us = side_;
    const Color them = opposite(us);

    auto change = [&st](Piece p, int f, int t) {
        PieceChange& d = st.changes.list[st.changes.count++];
        d.piece = p;
        d.from = static_cast<std::int8_t>(f);
        d.to = static_cast<std::int8_t>(t);
    };

    // Take the old castling rights and en-passant square out of the key;
    // the new ones are hashed back in at the end.
    key_ ^= zobrist_castle[castling_];
    key_ ^= zobrist_ep[ep_square_ < 0 ? 64 : ep_square_];

    // Captures (the en-passant victim is not on the destination square).
    if (move.is_en_passant()) {
        const int cap_sq = to + (us == Color::White ? -8 : 8);
        st.captured = board_[cap_sq];
        change(st.captured, cap_sq, -1);
        remove_piece(cap_sq);
    } else {
        st.captured = board_[to];
        if (st.captured != NO_PIECE) {
            change(st.captured, to, -1);
            remove_piece(to);
        }
    }

    const int moving_index = st.changes.count;
    change(moving, from, to);
    move_piece(from, to);

    // Castling rook movement.
    auto castle_rook = [&](int rfrom, int rto) {
        change(board_[rfrom], rfrom, rto);
        move_piece(rfrom, rto);
    };
    if (moving == WK && from == sq(4,0)) {
        if (to == sq(6,0)) castle_rook(sq(7,0), sq(5,0));
        else if (to == sq(2,0)) castle_rook(sq(0,0), sq(3,0));
    } else if (moving == BK && from == sq(4,7)) {
        if (to == sq(6,7)) castle_rook(sq(7,7), sq(5,7));
        else if (to == sq(2,7)) castle_rook(sq(0,7), sq(3,7));
    }

    // Promotion.
    const PieceType promo = move.promotion();
    if (promo != PieceType::None) {
        const Piece promoted = static_cast<Piece>(piece_index(us, promo));
        st.changes.list[moving_index].to = -1;  // the pawn leaves the board...
        change(promoted, -1, to);                // ...and the new piece appears
        remove_piece(to);
        put_piece(promoted, to);
    }

    castling_ &= castle_mask[from] & castle_mask[to];

    ep_square_ = -1;
    if (moving == WP && to - from == 16) {
        if (ep_capturable(from + 8, them)) ep_square_ = from + 8;
    } else if (moving == BP && from - to == 16) {
        if (ep_capturable(from - 8, them)) ep_square_ = from - 8;
    }

    if (moving == WP || moving == BP || st.captured != NO_PIECE) halfmove_clock_ = 0;
    else ++halfmove_clock_;

    key_ ^= zobrist_castle[castling_];
    key_ ^= zobrist_ep[ep_square_ < 0 ? 64 : ep_square_];
    key_ ^= zobrist_side;
    side_ = them;
    if (us == Color::Black) ++fullmove_;
}

void Position::unmake_move(Move move, const StateInfo& st) {
    const Color us = opposite(side_);
    const int from = move.from(), to = move.to();

    // Undo promotion: turn the promoted piece back into a pawn.
    if (move.promotion() != PieceType::None) {
        remove_piece<false>(to);
        put_piece<false>(static_cast<Piece>(piece_index(us, PieceType::Pawn)), to);
    }

    move_piece<false>(to, from);
    const Piece moved = board_[from];

    // Undo castling rook movement.
    if (moved == WK && from == sq(4,0)) {
        if (to == sq(6,0)) move_piece<false>(sq(5,0), sq(7,0));
        else if (to == sq(2,0)) move_piece<false>(sq(3,0), sq(0,0));
    } else if (moved == BK && from == sq(4,7)) {
        if (to == sq(6,7)) move_piece<false>(sq(5,7), sq(7,7));
        else if (to == sq(2,7)) move_piece<false>(sq(3,7), sq(0,7));
    }

    // Restore the captured piece.
    if (st.captured != NO_PIECE) {
        const int cap_sq = move.is_en_passant()
            ? to + (us == Color::White ? -8 : 8)
            : to;
        put_piece<false>(st.captured, cap_sq);
    }

    side_ = us;
    if (us == Color::Black) --fullmove_;
    castling_ = st.castling;
    ep_square_ = st.ep_square;
    halfmove_clock_ = st.halfmove_clock;
    key_ = st.key;  // the piece updates above skip hashing; restore the saved keys
    pawn_key_ = st.pawn_key;
    nonpawn_key_[0] = st.nonpawn_key[0];
    nonpawn_key_[1] = st.nonpawn_key[1];
    checkers_ = st.checkers;
}

bool Position::gives_check(Move m) const {
    const Color us = side_;
    const Bitboard their_king = bb_[piece_index(opposite(us), PieceType::King)];
    if (!their_king) return false;
    const int ksq = std::countr_zero(their_king);
    const int from = m.from(), to = m.to();
    const Piece moving = board_[from];
    const PieceType pt =
        m.promotion() != PieceType::None ? m.promotion() : static_cast<PieceType>(type_of(moving));

    // Occupancy after the move.
    Bitboard occ = (occupied_ ^ bit(from)) | bit(to);
    if (m.is_en_passant()) occ ^= bit(to + (us == Color::White ? -8 : 8));

    // Direct check by the piece on its new square.
    Bitboard direct = 0;
    switch (pt) {
        case PieceType::Pawn: direct = pawn_attacks(us, to); break;
        case PieceType::Knight: direct = knight_attacks(to); break;
        case PieceType::Bishop: direct = bishop_attacks(to, occ); break;
        case PieceType::Rook: direct = rook_attacks(to, occ); break;
        case PieceType::Queen: direct = queen_attacks(to, occ); break;
        default: break;
    }
    if (direct & their_king) return true;

    // Our sliders other than the moving piece, seeing the king through the
    // squares the move vacates.
    Bitboard diagonal = (bb_[piece_index(us, PieceType::Bishop)] |
                         bb_[piece_index(us, PieceType::Queen)]) & ~bit(from);
    Bitboard orthogonal = (bb_[piece_index(us, PieceType::Rook)] |
                           bb_[piece_index(us, PieceType::Queen)]) & ~bit(from);

    // Castling also moves the rook, which may give the check.
    if (pt == PieceType::King && (to - from == 2 || from - to == 2)) {
        const int rook_from = to > from ? from + 3 : from - 4;
        const int rook_to = to > from ? from + 1 : from - 1;
        occ = (occ ^ bit(rook_from)) | bit(rook_to);
        orthogonal = (orthogonal & ~bit(rook_from)) | bit(rook_to);
    }

    return (bishop_attacks(ksq, occ) & diagonal) || (rook_attacks(ksq, occ) & orthogonal);
}

bool Position::is_legal(Move m) const {
    const int from = m.from(), to = m.to();
    const int promo_bits = (m.data >> 12) & 7;  // 0: none (see Move)
    const PieceType promo = m.promotion();
    const Piece p = board_[from];
    const Color us = side_, them = opposite(us);
    if (p == NO_PIECE || (p < BP ? Color::White : Color::Black) != us) return false;
    if (from == to || (color_bb_[static_cast<int>(us)] & bit(to))) return false;
    if (promo_bits && (promo < PieceType::Knight || promo > PieceType::Queen)) return false;

    const auto pt = static_cast<PieceType>(type_of(p));
    const Bitboard theirs = color_bb_[static_cast<int>(them)];
    const Bitboard king = bb_[piece_index(us, PieceType::King)];
    if (!king) return false;
    const int ksq = std::countr_zero(king);

    // En passant and castling are rare: check them against the generator.
    const bool castling = pt == PieceType::King && (to - from == 2 || from - to == 2);
    if (m.is_en_passant() || castling) {
        if (m.is_en_passant() && (pt != PieceType::Pawn || to != ep_square_ || promo_bits)) return false;
        MoveList moves;
        int count = 0;
        if (castling) generate_quiets(*this, moves, count);
        else generate_captures(*this, moves, count);
        return std::find(moves.begin(), moves.begin() + count, m) != moves.begin() + count;
    }

    // Is the move possible for this piece at all?
    if (pt == PieceType::Pawn) {
        const int up = us == Color::White ? 8 : -8;
        const bool last_rank = rank_of(to) == (us == Color::White ? 7 : 0);
        if (last_rank != (promo_bits != 0)) return false;
        if (pawn_attacks(us, from) & bit(to)) {
            if (!(theirs & bit(to))) return false;
        } else if (to == from + up) {
            if (occupied_ & bit(to)) return false;
        } else if (to == from + 2 * up) {
            const bool home = rank_of(from) == (us == Color::White ? 1 : 6);
            if (!home || (occupied_ & (bit(from + up) | bit(to)))) return false;
        } else {
            return false;
        }
    } else {
        if (promo_bits) return false;
        Bitboard reach = 0;
        switch (pt) {
            case PieceType::Knight: reach = knight_attacks(from); break;
            case PieceType::Bishop: reach = bishop_attacks(from, occupied_); break;
            case PieceType::Rook: reach = rook_attacks(from, occupied_); break;
            case PieceType::Queen: reach = queen_attacks(from, occupied_); break;
            default: reach = king_attacks(from); break;
        }
        if (!(reach & bit(to))) return false;
    }

    // Does it leave our king safe? The king may not step onto an attacked
    // square (lifted off the board, so a checking slider covers the square
    // behind it).
    if (pt == PieceType::King)
        return !(attackers_to(to, occupied_ ^ king) & theirs & ~bit(to));

    // Any other piece: in double check it can't help; in single check it
    // must capture the checker or block; and a pinned piece must stay on
    // the line through our king.
    if (checkers_) {
        if (checkers_ & (checkers_ - 1)) return false;
        const int c = std::countr_zero(checkers_);
        if (!((squares_between(ksq, c) | checkers_) & bit(to))) return false;
    }
    const Bitboard snipers =
        (bishop_attacks(ksq, 0) & (bb_[piece_index(them, PieceType::Bishop)] |
                                   bb_[piece_index(them, PieceType::Queen)])) |
        (rook_attacks(ksq, 0) & (bb_[piece_index(them, PieceType::Rook)] |
                                 bb_[piece_index(them, PieceType::Queen)]));
    for (Bitboard s = snipers; s; s &= s - 1) {
        const int sq_ = std::countr_zero(s);
        if ((squares_between(ksq, sq_) & occupied_) == bit(from) && !(line_through(ksq, from) & bit(to)))
            return false;
    }
    return true;
}

Bitboard Position::threatened_by_lesser(Color c) const {
    const Color them = opposite(c);
    auto ours = [&](PieceType pt) { return bb_[piece_index(c, pt)]; };
    auto theirs = [&](PieceType pt) { return bb_[piece_index(them, pt)]; };

    // Pawn attacks for the whole set at once: a1 is square 0, so White's
    // captures go up by 7 (west) and 9 (east), Black's down by 9 and 7.
    constexpr Bitboard file_a = 0x0101010101010101ULL, file_h = 0x8080808080808080ULL;
    const Bitboard p = theirs(PieceType::Pawn);
    const Bitboard by_pawns = them == Color::White
                                  ? ((p & ~file_a) << 7) | ((p & ~file_h) << 9)
                                  : ((p & ~file_a) >> 9) | ((p & ~file_h) >> 7);
    Bitboard by_minors = 0, by_rooks = 0;
    for (Bitboard b = theirs(PieceType::Knight); b; b &= b - 1)
        by_minors |= knight_attacks(std::countr_zero(b));
    for (Bitboard b = theirs(PieceType::Bishop); b; b &= b - 1)
        by_minors |= bishop_attacks(std::countr_zero(b), occupied_);
    for (Bitboard b = theirs(PieceType::Rook); b; b &= b - 1)
        by_rooks |= rook_attacks(std::countr_zero(b), occupied_);

    const Bitboard minors = ours(PieceType::Knight) | ours(PieceType::Bishop);
    const Bitboard rooks = ours(PieceType::Rook), queens = ours(PieceType::Queen);
    return (by_pawns & (minors | rooks | queens)) | (by_minors & (rooks | queens)) |
           (by_rooks & queens);
}

void Position::make_null_move(StateInfo& st) {
    st.key = key_;
    st.pawn_key = pawn_key_;
    st.nonpawn_key[0] = nonpawn_key_[0];
    st.nonpawn_key[1] = nonpawn_key_[1];
    st.changes.count = 0;
    st.castling = castling_;
    st.ep_square = ep_square_;
    st.halfmove_clock = halfmove_clock_;
    st.checkers = checkers_;
    st.captured = NO_PIECE;

    key_ ^= zobrist_ep[ep_square_ < 0 ? 64 : ep_square_];
    ep_square_ = -1;
    key_ ^= zobrist_ep[64];
    key_ ^= zobrist_side;
    side_ = opposite(side_);
    ++halfmove_clock_;
    checkers_ = compute_checkers();
}

void Position::unmake_null_move(const StateInfo& st) {
    side_ = opposite(side_);
    ep_square_ = st.ep_square;
    halfmove_clock_ = st.halfmove_clock;
    key_ = st.key;
    checkers_ = st.checkers;
}

bool Position::is_consistent() const {
    // Mailbox must agree with the piece bitboards, square by square.
    for (int s = 0; s < 64; ++s) {
        Piece from_bb = NO_PIECE;
        for (int p = WP; p <= BK; ++p) {
            if (bb_[p] & bit(s)) {
                if (from_bb != NO_PIECE) return false;  // two pieces on one square
                from_bb = static_cast<Piece>(p);
            }
        }
        if (board_[s] != from_bb) return false;
    }

    Bitboard white = 0, black = 0;
    for (int p = WP; p <= WK; ++p) white |= bb_[p];
    for (int p = BP; p <= BK; ++p) black |= bb_[p];
    if (color_bb_[0] != white || color_bb_[1] != black) return false;
    if (occupied_ != (white | black)) return false;

    return key_ == compute_key() && pawn_key_ == compute_pawn_key() &&
           nonpawn_key_[0] == compute_nonpawn_key(0) && nonpawn_key_[1] == compute_nonpawn_key(1) &&
           checkers_ == compute_checkers();
}

std::string Position::move_to_uci(Move move) const {
    std::string s;
    s += char('a' + file_of(move.from()));
    s += char('1' + rank_of(move.from()));
    s += char('a' + file_of(move.to()));
    s += char('1' + rank_of(move.to()));
    if (move.promotion() != PieceType::None) {
        static constexpr char promo[] = "pnbrqk"; // indexed by PieceType
        s += promo[static_cast<int>(move.promotion())];
    }
    return s;
}

std::string Position::fen() const {
    std::ostringstream out;
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            Piece p = piece_at(sq(f,r));
            if (p == NO_PIECE) ++empty;
            else {
                if (empty) { out << empty; empty = 0; }
                out << piece_char(p);
            }
        }
        if (empty) out << empty;
        if (r) out << '/';
    }
    out << ' ' << (side_ == Color::White ? 'w' : 'b') << ' ';
    if (!castling_) out << '-';
    else {
        if (castling_ & WK_CASTLE) out << 'K';
        if (castling_ & WQ_CASTLE) out << 'Q';
        if (castling_ & BK_CASTLE) out << 'k';
        if (castling_ & BQ_CASTLE) out << 'q';
    }
    out << ' ';
    if (ep_square_ < 0) out << '-';
    else out << char('a' + file_of(ep_square_)) << char('1' + rank_of(ep_square_));
    out << ' ' << halfmove_clock_ << ' ' << fullmove_;
    return out.str();
}

Position::CheckInfo Position::check_info() const {
    CheckInfo ci{};
    const Color us = side_;
    const Bitboard their_king = bb_[piece_index(opposite(us), PieceType::King)];
    if (!their_king) {
        ci.king_square = -1;
        return ci;
    }
    const int ksq = std::countr_zero(their_king);
    ci.king_square = ksq;

    // A piece attacks the king from exactly the squares the king would
    // attack if it were that piece (pawns: of the opposite color).
    const Bitboard diag = bishop_attacks(ksq, occupied_);
    const Bitboard orth = rook_attacks(ksq, occupied_);
    ci.checking_from[static_cast<int>(PieceType::Pawn)] = pawn_attacks(opposite(us), ksq);
    ci.checking_from[static_cast<int>(PieceType::Knight)] = knight_attacks(ksq);
    ci.checking_from[static_cast<int>(PieceType::Bishop)] = diag;
    ci.checking_from[static_cast<int>(PieceType::Rook)] = orth;
    ci.checking_from[static_cast<int>(PieceType::Queen)] = diag | orth;
    ci.checking_from[static_cast<int>(PieceType::King)] = 0;

    // Our sliders aimed at the king through exactly one of our own pieces.
    const Bitboard ours = color_bb_[static_cast<int>(us)];
    Bitboard snipers =
        (bishop_attacks(ksq, 0) & (bb_[piece_index(us, PieceType::Bishop)] |
                                   bb_[piece_index(us, PieceType::Queen)])) |
        (rook_attacks(ksq, 0) & (bb_[piece_index(us, PieceType::Rook)] |
                                 bb_[piece_index(us, PieceType::Queen)]));
    while (snipers) {
        const int s = std::countr_zero(snipers);
        snipers &= snipers - 1;
        const Bitboard blockers = squares_between(ksq, s) & occupied_;
        if (blockers && !(blockers & (blockers - 1)) && (blockers & ours)) ci.discoverers |= blockers;
    }
    return ci;
}

bool Position::gives_check(Move m, const CheckInfo& ci) const {
    if (ci.king_square < 0) return false;
    const int from = m.from(), to = m.to();
    const Piece moving = board_[from];
    const auto pt = static_cast<PieceType>(type_of(moving));

    // Promotions, en passant (two squares vacated) and castling (the rook
    // moves too) are rare enough to take the full test.
    if (m.promotion() != PieceType::None || m.is_en_passant() ||
        (pt == PieceType::King && (to - from == 2 || from - to == 2)))
        return gives_check(m);

    // Direct check, using the occupancy before the move. Vacating `from`
    // could only matter if `from` lies between `to` and the king. Then the
    // move runs along that line, and either the piece already gave check
    // from `from` (impossible with us to move) or something else between
    // `from` and the king still blocks it. Either way the answer is the
    // same as with `from` still occupied.
    if (ci.checking_from[static_cast<int>(pt)] & bit(to)) return true;
    // Discovered check: a blocker leaving its line to the king.
    return (ci.discoverers & bit(from)) && !(line_through(ci.king_square, from) & bit(to));
}
