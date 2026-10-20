// King state (src/nnue.cpp, king_state() and shelter(); the author,
// 2026-10-05): the first layer's block chosen by the king's state rather
// than its square, and output buckets split by the side to move's shelter.
// The same definitions, on bullet's boards (seen from the side to move).
// Castling rights (a fifth location while a right is left, ROADMAP K2) are
// not in bullet's board: parse() keeps them in its spare byte, and the
// data must be converted with it (convert(), --convert).

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::str::FromStr;

use bullet_lib::game::{formats::bulletformat::ChessBoard, inputs::SparseInputType, outputs::OutputBuckets};

pub const KING_STATES: usize = 15;
// The bare king state (src/nnue.cpp, king_location(); ROADMAP 5b): the
// location alone, castling rights one of them, no shelter.
pub const KING_LOCATIONS: usize = 5;

/// A side's own pawns and king square in its own coordinates (its pawns
/// moving up): the side to move's as on the board, the other's flipped.
fn own_view(pos: &ChessBoard, side: usize) -> (u64, usize) {
    let (mut pawns, mut king) = (0u64, 0usize);
    for (piece, square) in pos.into_iter() {
        if usize::from(piece & 8 != 0) != side {
            continue;
        }
        let sq = if side == 0 { square as usize } else { square as usize ^ 56 };
        match piece & 7 {
            0 => pawns |= 1u64 << sq,
            5 => king = sq,
            _ => {}
        }
    }
    (pawns, king)
}

/// Files of the king and its neighbours with no own pawn on the two ranks
/// in front of it, at most 2.
fn shelter(pawns: u64, king: usize) -> usize {
    let (f, r) = (king % 8, king / 8);
    let mut front = 0u64;
    for q in (r + 1)..=(r + 2).min(7) {
        front |= 0xFFu64 << (8 * q);
    }
    let mut missing = 0;
    for x in f.saturating_sub(1)..=(f + 1).min(7) {
        if pawns & front & (0x0101_0101_0101_0101u64 << x) == 0 {
            missing += 1;
        }
    }
    missing.min(2)
}

fn location(king: usize, castle: bool) -> usize {
    let (f, r) = (king % 8, king / 8);
    let edge = f.min(7 - f);
    if castle { 4 } else if r >= 2 { 3 } else if edge <= 1 { 0 } else if edge == 2 { 1 } else { 2 }
}

fn state(pawns: u64, king: usize, castle: bool) -> usize {
    location(king, castle) * 3 + shelter(pawns, king)
}

/// Piece inputs in KING_STATES blocks of 768, each view's block by its
/// king's state (or, `bare`, in KING_LOCATIONS blocks by its location
/// alone); mirrored left-right when that king is on files e-h.
/// With `merge`, the states merged: each state's block (src/nnue.h; the
/// engine reads the table from the header's layout bytes).
#[derive(Clone, Copy, Debug)]
pub struct KingStateBuckets {
    pub mirrored: bool,
    pub bare: bool,
    pub merge: Option<[usize; KING_STATES]>,
}

/// Merged king states (the author, 2026-10-18: on r123 three states hold
/// under 1% of the views each and a king-state net loses most to 7 king
/// buckets in its thinnest states). kingstate12: the full-shelter states of
/// the c/f, centre and advanced locations (states 3, 6, 9) folded into their
/// neighbour with one file open.
pub const MERGE12: [usize; KING_STATES] = [0, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 10, 11];
/// kingstate8 (the author's): the seven largest states on r123 kept (wing
/// covered, one open, two open; c/f, centre and advanced with two open;
/// castling rights with one open), every other state one block.
pub const MERGE8: [usize; KING_STATES] = [0, 1, 2, 7, 7, 3, 7, 7, 4, 7, 7, 5, 7, 6, 7];

impl KingStateBuckets {
    /// From --buckets: kinglocation, kingstate12, kingstate8, else every king state.
    pub fn named(name: &str) -> Self {
        let merge = match name {
            "kingstate12" => Some(MERGE12),
            "kingstate8" => Some(MERGE8),
            _ => None,
        };
        KingStateBuckets { mirrored: true, bare: name == "kinglocation", merge }
    }

    pub fn blocks(&self) -> usize {
        match (self.bare, self.merge) {
            (true, _) => KING_LOCATIONS,
            (false, Some(m)) => m.iter().max().unwrap() + 1,
            (false, None) => KING_STATES,
        }
    }

    fn block(&self, pawns: u64, king: usize, castle: bool) -> usize {
        if self.bare {
            return location(king, castle);
        }
        let s = state(pawns, king, castle);
        self.merge.map_or(s, |m| m[s])
    }
}

impl SparseInputType for KingStateBuckets {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        self.blocks() * 768
    }

    fn max_active(&self) -> usize {
        32
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let (stm_pawns, stm_king) = own_view(pos, 0);
        let (ntm_pawns, ntm_king) = own_view(pos, 1);
        let flip = |k: usize| if self.mirrored && k % 8 > 3 { 7 } else { 0 };
        let (stm_flip, ntm_flip) = (flip(stm_king), flip(ntm_king));
        let rights = pos.extra()[0];
        let stm_base = 768 * self.block(stm_pawns, stm_king, rights & 3 != 0);
        let ntm_base = 768 * self.block(ntm_pawns, ntm_king, rights & 12 != 0);
        for (piece, square) in pos.into_iter() {
            let (c, pt, sq) = (usize::from(piece & 8 != 0), (piece & 7) as usize, square as usize);
            f(
                stm_base + 384 * c + 64 * pt + (sq ^ stm_flip),
                ntm_base + 384 * (1 - c) + 64 * pt + ((sq ^ 56) ^ ntm_flip),
            );
        }
    }

    fn shorthand(&self) -> String {
        format!("{}x768{}", self.blocks(), if self.bare { "kinglocation" } else { "kingstate" })  // 12x768kingstate: merged
    }

    fn description(&self) -> String {
        "Piece-square inputs in blocks by king state".to_string()
    }
}

/// N material buckets (as bullet's MaterialCount), each split three ways
/// by the side to move's shelter.
#[derive(Clone, Copy, Default)]
pub struct KingStateOutput<const N: usize>;

impl<const N: usize> OutputBuckets<ChessBoard> for KingStateOutput<N> {
    const BUCKETS: usize = N * 3;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        (material_bucket(pos, N) * 3 + stm_shelter(pos)) as u8
    }
}

/// bullet's MaterialCount<N> bucket: pieces on the board, kings excluded.
pub fn material_bucket(pos: &ChessBoard, n: usize) -> usize {
    ((pos.occ().count_ones() as usize - 2) / 32usize.div_ceil(n)).min(n - 1)
}

/// The side to move's shelter (0 intact, 1 weakened, 2 broken).
pub fn stm_shelter(pos: &ChessBoard) -> usize {
    let (pawns, king) = own_view(pos, 0);
    shelter(pawns, king)
}

/// A text line ("<FEN> | score | result") as bullet's board, with the
/// FEN's castling rights in the spare byte extra[0], from the side to
/// move: bits 0 and 1 its short and long, bits 2 and 3 the other side's.
pub fn parse(line: &str) -> Result<ChessBoard, String> {
    let mut board = ChessBoard::from_str(line)?;
    let mut fields = line.split('|').next().unwrap_or("").split_whitespace().skip(1);
    let black = fields.next() == Some("b");
    let castling = fields.next().unwrap_or("-");
    let order = if black { "kqKQ" } else { "KQkq" };
    board.extra[0] = order.chars().enumerate().filter(|&(_, c)| castling.contains(c)).fold(0u8, |b, (i, _)| b | 1 << i);
    Ok(board)
}

/// bullet-utils' text conversion (convert --from text), keeping castling
/// rights: lines that don't parse are reported and skipped.
pub fn convert(input: &str, output: &str) {
    let file = BufReader::new(std::fs::File::open(input).expect("can't open the input"));
    let mut out = BufWriter::new(std::fs::File::create(output).expect("can't create the output"));
    let (mut results, mut rights) = ([0u64; 3], 0u64);
    for line in file.lines() {
        let line = line.expect("can't read the input");
        match parse(&line) {
            Ok(board) => {
                results[usize::from(board.result)] += 1;
                rights += u64::from(board.extra[0] != 0);
                // SAFETY: ChessBoard is bullet's 32-byte plain-data record.
                let bytes = unsafe {
                    std::slice::from_raw_parts((&board as *const ChessBoard).cast::<u8>(), std::mem::size_of::<ChessBoard>())
                };
                out.write_all(bytes).expect("can't write the output");
            }
            Err(message) => println!("error parsing: {message}"),
        }
    }
    out.flush().expect("can't write the output");
    println!(
        "{} positions (wins {}, draws {}, losses {}), {} with a castling right left",
        results.iter().sum::<u64>(),
        results[2],
        results[1],
        results[0],
        rights
    );
}
