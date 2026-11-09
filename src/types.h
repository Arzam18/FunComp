#pragma once

#include <array>
#include <cstdint>
#include <string>

using Bitboard = std::uint64_t;

enum class Color : int { White = 0, Black = 1 };

constexpr Color opposite(Color c) {
    return c == Color::White ? Color::Black : Color::White;
}

enum class PieceType : int {
    Pawn = 0,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
    None
};

enum Piece : int {
    WP = 0, WN, WB, WR, WQ, WK,
    BP, BN, BB, BR, BQ, BK,
    NO_PIECE = -1
};

// A piece's type as an index, 0 (pawn) to 5 (king), and -1 for NO_PIECE: the
// same as p % 6 without a signed division.
constexpr int type_of(int p) { return p >= BP ? p - BP : p; }

constexpr int piece_index(Color c, PieceType pt) {
    return static_cast<int>(pt) + (c == Color::Black ? 6 : 0);
}

constexpr int sq(int file, int rank) {
    return rank * 8 + file;
}

constexpr Bitboard bit(int square) {
    return Bitboard{1} << square;
}

constexpr int file_of(int square) { return square & 7; }
constexpr int rank_of(int square) { return square >> 3; }

struct Move {
    // Deliberately left uninitialized so that Move, and therefore MoveList,
    // is trivially default-constructible: declaring a MoveList then costs
    // nothing instead of zeroing 512 bytes. Use Move{} for a null move.
    std::uint16_t data;

    // bits 0..5: from
    // bits 6..11: to
    // bits 12..14: promotion PieceType XOR 6, so that 0 is none (PieceType::None
    // is 6) and decoding needs no branch: queen 2, bishop 4, rook 5, knight 7
    // bit 15: en-passant flag
    Move() = default;
    static constexpr int PROMO_XOR = static_cast<int>(PieceType::None);

    constexpr Move(int from, int to, PieceType promotion = PieceType::None,
                   bool en_passant = false)
        : data(static_cast<std::uint16_t>(
              from | (to << 6) |
              ((static_cast<int>(promotion) ^ PROMO_XOR) << 12) |
              (en_passant ? 0x8000 : 0))) {}

    constexpr int from() const { return data & 63; }
    constexpr int to() const { return (data >> 6) & 63; }

    constexpr PieceType promotion() const {
        return static_cast<PieceType>(((data >> 12) & 7) ^ PROMO_XOR);
    }

    constexpr bool is_en_passant() const { return (data & 0x8000) != 0; }

    constexpr bool operator==(const Move&) const = default;
};

// One piece-square change made by a move, for incremental evaluation.
// from < 0: the piece appeared on `to` (promotion). to < 0: the piece left
// the board from `from` (capture, or the pawn that promoted).
struct PieceChange {
    Piece piece;
    std::int8_t from;
    std::int8_t to;
};

// Every piece-square change made by one move: at most three (a capturing
// promotion removes the victim and the pawn and adds the new piece).
struct MoveChanges {
    int count = 0;
    PieceChange list[3];
};

struct StateInfo {
    std::uint64_t key = 0;
    std::uint64_t pawn_key = 0;
    std::uint64_t nonpawn_key[2] = {0, 0};
    int castling = 0;
    int ep_square = -1;
    int halfmove_clock = 0;
    Piece captured = NO_PIECE;
    Bitboard checkers = 0;  // Position::checkers() before the move
    MoveChanges changes;  // filled in by make_move
};

using MoveList = std::array<Move, 256>;
