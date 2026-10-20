#include "pgn.h"
#include "movegen.h"
#include <cctype>

namespace pgn {

namespace {

constexpr const char* START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// The value of a tag line `[Name "value"]`, if it is tag `name`.
bool tag_value(const std::string& line, const std::string& name, std::string& value) {
    if (line.compare(1, name.size() + 1, name + " ") != 0) return false;
    const auto open = line.find('"');
    const auto close = line.rfind('"');
    if (open == std::string::npos || close <= open) return false;
    value = line.substr(open + 1, close - open - 1);
    return true;
}

bool is_result(const std::string& t) {
    return t == "1-0" || t == "0-1" || t == "1/2-1/2" || t == "*";
}

// Splits movetext into move tokens, dropping comments, variations, NAGs,
// move numbers and results. `depth` counts open variations across lines.
void add_moves(const std::string& text, int& depth, bool& in_comment,
               std::vector<std::string>& moves) {
    std::string token;
    auto flush = [&] {
        // Move numbers may be glued to the move ("8...Nc7").
        std::size_t i = 0;
        while (i < token.size() && (std::isdigit(static_cast<unsigned char>(token[i])) ||
                                    token[i] == '.'))
            ++i;
        std::string t = token.substr(i);
        if (!t.empty() && depth == 0 && t[0] != '$' && !is_result(token)) moves.push_back(t);
        token.clear();
    };
    for (const char c : text) {
        if (in_comment) {
            if (c == '}') in_comment = false;
            continue;
        }
        if (c == '{') { flush(); in_comment = true; }
        else if (c == ';') { flush(); return; }  // comment to the end of the line
        else if (c == '(') { flush(); ++depth; }
        else if (c == ')') { flush(); if (depth > 0) --depth; }
        else if (std::isspace(static_cast<unsigned char>(c))) flush();
        else token += c;
    }
    flush();
}

}  // namespace

const std::string& Game::tag(const std::string& name) const {
    static const std::string none;
    for (const auto& [n, v] : tags)
        if (n == name) return v;
    return none;
}

bool Reader::next(Game& game) {
    // Cleared, not replaced: the buffers are reused from game to game.
    game.fen.clear();
    game.moves.clear();
    game.result.clear();
    game.tags.clear();
    game.movetext.clear();
    bool has_movetext = false, has_tags = false;
    int depth = 0;
    bool in_comment = false;
    std::string& line = line_;
    for (;;) {
        if (has_pending_) {
            line.swap(pending_);
            has_pending_ = false;
        } else if (!std::getline(in_, line)) {
            break;
        }
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!in_comment && !line.empty() && line[0] == '[') {
            if (has_movetext) {  // tags after movetext start a new game
                pending_.swap(line);
                has_pending_ = true;
                break;
            }
            has_tags = true;
            // `[Name "value"]`
            const auto space = line.find(' ');
            if (space != std::string::npos && space > 1) {
                std::string value;
                std::string name = line.substr(1, space - 1);
                if (tag_value(line, name, value)) {
                    if (name == "FEN") game.fen = value;
                    else if (name == "Result") game.result = value;
                    game.tags.emplace_back(std::move(name), std::move(value));
                }
            }
            continue;
        }
        if (line.find_first_not_of(" \t") == std::string::npos) continue;
        has_movetext = true;
        if (split_) {
            add_moves(line, depth, in_comment, game.moves);
            continue;
        }
        // Unsplit: only the comments are followed, for the tag test above.
        for (const char c : line) {
            if (in_comment) {
                if (c == '}') in_comment = false;
            } else if (c == '{') {
                in_comment = true;
            } else if (c == ';') {
                break;
            }
        }
        game.movetext += line;
        game.movetext += '\n';
    }
    if (!has_tags && !has_movetext) return false;
    if (game.fen.empty()) game.fen = START_FEN;
    return true;
}

void split_moves(Game& game) {
    game.moves.clear();
    int depth = 0;
    bool in_comment = false;
    std::size_t start = 0;
    while (start < game.movetext.size()) {
        std::size_t end = game.movetext.find('\n', start);
        if (end == std::string::npos) end = game.movetext.size();
        add_moves(game.movetext.substr(start, end - start), depth, in_comment, game.moves);
        start = end + 1;
    }
}

std::vector<Game> read(std::istream& in) {
    std::vector<Game> games;
    Reader reader(in);
    Game game;
    while (reader.next(game)) games.push_back(game);
    return games;
}

Move parse_san(const Position& pos, const std::string& san) {
    std::string s = san;
    while (!s.empty() && (s.back() == '+' || s.back() == '#' || s.back() == '!' ||
                          s.back() == '?'))
        s.pop_back();
    MoveList moves;
    int count = 0;
    generate_legal(pos, moves, count);
    const auto legal = [&](int i) { return moves[static_cast<std::size_t>(i)]; };

    // Castling: the king's two-square move.
    if (s == "O-O" || s == "0-0" || s == "O-O-O" || s == "0-0-0") {
        const int step = s.size() == 3 ? 2 : -2;
        for (int i = 0; i < count; ++i) {
            const Move m = legal(i);
            if (type_of(pos.piece_at(m.from())) == static_cast<int>(PieceType::King) &&
                m.to() == m.from() + step)
                return m;
        }
        return Move{};
    }

    PieceType promo = PieceType::None;
    if (const auto eq = s.find('='); eq != std::string::npos && eq + 1 < s.size()) {
        const char p = s[eq + 1];
        promo = p == 'Q' ? PieceType::Queen : p == 'R' ? PieceType::Rook
              : p == 'B' ? PieceType::Bishop : p == 'N' ? PieceType::Knight : PieceType::None;
        s.erase(eq);
    }
    PieceType type = PieceType::Pawn;
    if (!s.empty() && std::string("KQRBN").find(s[0]) != std::string::npos) {
        type = s[0] == 'K' ? PieceType::King : s[0] == 'Q' ? PieceType::Queen
             : s[0] == 'R' ? PieceType::Rook : s[0] == 'B' ? PieceType::Bishop
                           : PieceType::Knight;
        s.erase(0, 1);
    }
    std::string rest;
    for (const char c : s)
        if (c != 'x' && c != '-') rest += c;
    if (rest.size() < 2) return Move{};
    const char file = rest[rest.size() - 2], rank = rest[rest.size() - 1];
    if (file < 'a' || file > 'h' || rank < '1' || rank > '8') return Move{};
    const int to = sq(file - 'a', rank - '1');
    const std::string from_hint = rest.substr(0, rest.size() - 2);  // file and/or rank

    Move found{};
    int matches = 0;
    for (int i = 0; i < count; ++i) {
        const Move m = legal(i);
        if (m.to() != to || m.promotion() != promo) continue;
        if (type_of(pos.piece_at(m.from())) != static_cast<int>(type)) continue;
        bool ok = true;
        for (const char h : from_hint) {
            if (h >= 'a' && h <= 'h') ok &= file_of(m.from()) == h - 'a';
            else if (h >= '1' && h <= '8') ok &= rank_of(m.from()) == h - '1';
            else ok = false;
        }
        if (ok) {
            found = m;
            ++matches;
        }
    }
    return matches == 1 ? found : Move{};
}

}  // namespace pgn
