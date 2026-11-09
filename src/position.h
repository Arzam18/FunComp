#pragma once

#include "attacks.h"
#include "types.h"
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <string>

class Position {
public:
    static constexpr int WK_CASTLE = 1;
    static constexpr int WQ_CASTLE = 2;
    static constexpr int BK_CASTLE = 4;
    static constexpr int BQ_CASTLE = 8;

    Position();

    void set_startpos();
    bool set_fen(const std::string& fen);
    std::string fen() const;

    Color side_to_move() const { return side_; }
    Bitboard pieces(Piece p) const { return bb_[p]; }
    Bitboard occupancy(Color c) const { return color_bb_[static_cast<int>(c)]; }
    Bitboard occupied() const { return occupied_; }
    int castling_rights() const { return castling_; }
    int ep_square() const { return ep_square_; }
    int halfmove_clock() const { return halfmove_clock_; }
    std::uint64_t key() const { return key_; }
    // Zobrist key of the pawns alone, for pawn-structure caches and
    // correction history. Uses the same piece-square numbers as key().
    std::uint64_t pawn_key() const { return pawn_key_; }
    // Zobrist key of one colour's pieces other than pawns (king included),
    // for correction history.
    std::uint64_t nonpawn_key(Color c) const { return nonpawn_key_[static_cast<int>(c)]; }
    // Whether a single reversible move (a piece other than a pawn, over an
    // empty path) turns this position into the one with key `other`, or
    // that one into this: the test behind upcoming-repetition detection.
    bool reversible_move_to(std::uint64_t other) const;

    // O(1) lookup via the mailbox array.
    Piece piece_at(int square) const { return board_[square]; }
    // All pieces of either color attacking `square`, given `occupied` as the
    // blocker set (callers may pass a modified occupancy, e.g. with the king
    // removed, to test squares along a slider's line).
    // Inline: move generation, SEE and the checkers update call it at every
    // node.
    Bitboard attackers_to(int square, Bitboard occupied) const {
        const Bitboard diagonal = bb_[WB] | bb_[BB] | bb_[WQ] | bb_[BQ];
        const Bitboard orthogonal = bb_[WR] | bb_[BR] | bb_[WQ] | bb_[BQ];
        // A white pawn attacks `square` from exactly the squares a black pawn
        // on `square` would attack, and vice versa.
        return (pawn_attacks(Color::Black, square) & bb_[WP])
             | (pawn_attacks(Color::White, square) & bb_[BP])
             | (knight_attacks(square) & (bb_[WN] | bb_[BN]))
             | (king_attacks(square) & (bb_[WK] | bb_[BK]))
             | (bishop_attacks(square, occupied) & diagonal)
             | (rook_attacks(square, occupied) & orthogonal);
    }
    bool square_attacked(int square, Color by) const;
    // The enemy pieces giving check to the side to move. Kept up to date by
    // set_fen, make_move and the null move, so that asking costs a load.
    Bitboard checkers() const { return checkers_; }
    // True if `c` is in check (or, in a malformed position, has no king).
    bool in_check(Color c) const {
        if (c == side_ && bb_[piece_index(c, PieceType::King)]) return checkers_ != 0;
        return in_check_slow(c);
    }

    // Also records in st.changes every piece-square change the move makes.
    void make_move(Move move, StateInfo& st) {
        apply_move(move, st);
        checkers_ = compute_checkers();
    }
    // For a caller that already knows gives_check(move): a move that gives
    // no check leaves the opponent without checkers, so there's nothing to
    // compute.
    void make_move(Move move, StateInfo& st, bool gives_check) {
        apply_move(move, st);
        checkers_ = gives_check ? compute_checkers() : 0;
        assert(checkers_ == compute_checkers());
    }
    void unmake_move(Move move, const StateInfo& st);

    // True if legal move `m` would put the opponent in check, directly or
    // by discovery, without making it. For pruning decisions taken before
    // a move is made.
    bool gives_check(Move m) const;

    // True if `m` is a legal move in this position: exactly one of the moves
    // generate_legal would produce, flags included. `m` may be any 16-bit
    // value (a hash move can come from a colliding position), so nothing
    // about it is assumed. For trying hash moves and killers before
    // generating the move list.
    bool is_legal(Move m) const;

    // Pieces of `c` attacked by a cheaper enemy piece: minors, rooks and
    // queens by pawns; rooks and queens by knights and bishops; queens by
    // rooks. A static, every-node threat signal.
    Bitboard threatened_by_lesser(Color c) const;

    // Precomputed check information for the side to move, so that many
    // gives_check tests at one node cost a couple of bit operations each.
    // checking_from[pt]: squares from which a piece of type pt would attack
    // the enemy king. discoverers: our pieces that are the only blocker
    // between one of our sliders and the enemy king, so moving one off the
    // line uncovers a check.
    struct CheckInfo {
        Bitboard checking_from[6];
        Bitboard discoverers;
        int king_square;  // -1 if the enemy has no king (malformed position)
    };
    CheckInfo check_info() const;
    // Same answer as gives_check(m), for a CheckInfo made in this position.
    bool gives_check(Move m, const CheckInfo& ci) const;

    // Passes the turn without moving, for null-move pruning. Clears the en
    // passant square and advances the halfmove clock; st.changes is empty.
    void make_null_move(StateInfo& st);
    void unmake_null_move(const StateInfo& st);

    // True if `c` has a knight, bishop, rook or queen. Null-move pruning is
    // unsafe without one: king-and-pawn endings are full of zugzwang.
    bool has_non_pawn_material(Color c) const {
        return (occupancy(c) & ~bb_[piece_index(c, PieceType::Pawn)] &
                ~bb_[piece_index(c, PieceType::King)]) != 0;
    }

    std::string move_to_uci(Move move) const;

    // Debug aid: recomputes the mailbox, occupancy and Zobrist keys from the
    // piece bitboards and reports whether the incrementally maintained
    // versions agree. Slow; intended for tests only.
    bool is_consistent() const;

private:
    std::array<Bitboard, 12> bb_{};
    std::array<Bitboard, 2> color_bb_{};
    std::array<Piece, 64> board_{};   // mailbox: piece on each square or NO_PIECE
    Bitboard occupied_ = 0;
    Color side_ = Color::White;
    int castling_ = 0;
    int ep_square_ = -1;
    int halfmove_clock_ = 0;
    int fullmove_ = 1;  // the FEN's full-move number, counted up after Black moves
    std::uint64_t key_ = 0;
    std::uint64_t pawn_key_ = 0;
    std::uint64_t nonpawn_key_[2] = {0, 0};  // by colour index
    Bitboard checkers_ = 0;           // see checkers()

    void clear();
    void rebuild_occupancy();
    std::uint64_t compute_key() const;
    std::uint64_t compute_pawn_key() const;
    std::uint64_t compute_nonpawn_key(int c) const;
    Bitboard compute_checkers() const;
    // make_move without the checkers update.
    void apply_move(Move move, StateInfo& st);
    bool in_check_slow(Color c) const;

    // Incremental board updates. Each keeps bb_, color_bb_, occupied_ and
    // board_ in sync, and with UpdateKey also the piece part of key_ and
    // pawn_key_.
    // unmake_move passes false: it restores the saved key wholesale, so
    // hashing there would be wasted work.
    template <bool UpdateKey = true> void put_piece(Piece p, int square);
    template <bool UpdateKey = true> void remove_piece(int square);
    template <bool UpdateKey = true> void move_piece(int from, int to);

    // True if a pawn of the side to move could capture on `ep`. Only then is
    // an en passant square recorded, so positions that differ only by a
    // useless ep square share a Zobrist key (and a perft/TT entry).
    bool ep_capturable(int ep, Color capturer) const;
};

void init_zobrist();
