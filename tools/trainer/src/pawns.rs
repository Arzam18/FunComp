// The pawn predicates (ROADMAP, Try 5, group A1) as extra network inputs,
// the trainer's side of src/pawn_features.h and src/nnue.h: the same
// definitions, on bullet's boards (seen from the side to move, whose pawns
// move up the board), and the same input layout: after the piece inputs,
// per view (predicate x 2 + the pawn is the other side's) x 64 + square, in
// the view's orientation and mirroring.

use crate::{pairs, squares};

use bullet_lib::game::{
    formats::bulletformat::ChessBoard,
    inputs::SparseInputType,
};

pub const PREDICATES: usize = 11;
pub const PAWN_FEATURES: usize = PREDICATES * 2 * 64;

const FILE_A: u64 = 0x0101_0101_0101_0101;

fn file_bb(f: usize) -> u64 {
    FILE_A << f
}
fn adjacent_files(f: usize) -> u64 {
    (if f > 0 { file_bb(f - 1) } else { 0 }) | (if f < 7 { file_bb(f + 1) } else { 0 })
}
fn ranks_above(sq: usize) -> u64 {
    let r = sq / 8;
    if r == 7 { 0 } else { !0u64 << (8 * (r + 1)) }
}
fn ranks_below(sq: usize) -> u64 {
    let r = sq / 8;
    if r == 0 { 0 } else { (1u64 << (8 * r)) - 1 }
}
fn rank_bb(sq: usize) -> u64 {
    0xFFu64 << (8 * (sq / 8))
}
// Squares attacked by pawns moving down the board.
fn down_attacks(pawns: u64) -> u64 {
    ((pawns & !FILE_A) >> 9) | ((pawns & !(FILE_A << 7)) >> 7)
}

/// Calls f(predicate, square) for every predicate of `own` pawns (moving up
/// the board) against `opp` pawns (moving down), in pawn_features.h's
/// order.
fn up_predicates(own: u64, opp: u64, mut f: impl FnMut(usize, usize)) {
    let opp_attacks = down_attacks(opp);
    let mut b = own;
    while b != 0 {
        let s = b.trailing_zeros() as usize;
        b &= b - 1;
        let file = s % 8;
        let here = 1u64 << s;
        let stop = here << 8;
        let ahead = ranks_above(s);
        let beside = rank_bb(s);
        let passed = opp & (file_bb(file) | adjacent_files(file)) & ahead == 0;
        // Own pawns defending s: from s - 7 (not from file h) and s - 9 (not from file a).
        let mut defend_from = 0u64;
        if s >= 9 && file > 0 {
            defend_from |= 1u64 << (s - 9);
        }
        if s >= 7 && file < 7 {
            defend_from |= 1u64 << (s - 7);
        }
        let defenders = (defend_from & own).count_ones();
        let helpers = own & adjacent_files(file) & (beside | ranks_below(s));
        // Squares s attacks: s + 7 (not from file a), s + 9 (not from file h).
        let mut attacks = 0u64;
        if file > 0 && s + 7 < 64 {
            attacks |= 1u64 << (s + 7);
        }
        if file < 7 && s + 9 < 64 {
            attacks |= 1u64 << (s + 9);
        }
        // Sentry: s alone stops an opposing pawn on an adjacent file (moving
        // down) from being passed.
        let mut sentry = false;
        let mut q_set = opp & adjacent_files(file);
        while q_set != 0 {
            let q = q_set.trailing_zeros() as usize;
            q_set &= q_set - 1;
            let path = (file_bb(q % 8) | adjacent_files(q % 8)) & ranks_below(q);
            if path & here != 0 && own & path & !here == 0 {
                sentry = true;
                break;
            }
        }
        let active = [
            own & adjacent_files(file) == 0,
            own & file_bb(file) & !here != 0,
            helpers == 0 && stop & opp_attacks != 0,
            passed,
            !passed
                && opp & file_bb(file) & ahead == 0
                && helpers.count_ones() >= (opp & adjacent_files(file) & ahead).count_ones(),
            defenders >= 1,
            defenders >= 2,
            own & adjacent_files(file) & beside != 0,
            stop & opp != 0,
            attacks & opp != 0,
            sentry,
        ];
        for (p, &on) in active.iter().enumerate() {
            if on {
                f(p, s);
            }
        }
    }
}

/// Calls f(predicate, colour, square) for both sides' pawns in a board seen
/// from the side to move (colour 0 the side to move, squares in its view).
pub fn predicates(pos: &ChessBoard, mut f: impl FnMut(usize, usize, usize)) {
    let (mut us, mut them) = (0u64, 0u64);
    for (piece, square) in pos.into_iter() {
        if piece & 7 == 0 {
            if piece & 8 == 0 {
                us |= 1u64 << square;
            } else {
                them |= 1u64 << square;
            }
        }
    }
    up_predicates(us, them, |p, s| f(p, 0, s));
    // The other side moves down: flip the board, its pawns move up.
    up_predicates(them.swap_bytes(), us.swap_bytes(), |p, s| f(p, 1, s ^ 56));
}

/// Piece inputs `B` followed by the extra feature sets the network declares,
/// in the engine's order (src/nnue.h): the pawn predicates, then knight
/// pairs, slider pairs and slider x pawn (pairs.rs), attacked
/// squares, least attackers and king zone (squares.rs), each in the view's
/// orientation and mirroring.
///
/// With `factorised`, a block of 768 factoriser inputs comes first (shared
/// by every king bucket's piece inputs: piece input % 768) and everything
/// else is shifted after it, as bullet's Factorised lays them out; done
/// here because Factorised counts every input twice towards the bound on
/// active inputs, extra ones too, and pawns with knight pairs then pass
/// what the GPU kernels take.
#[derive(Clone, Copy, Debug)]
pub struct WithExtras<B> {
    pub base: B,
    pub mirrored: bool,
    pub pawns: bool,
    pub knights: bool,
    pub sliders: bool,
    pub slider_pawns: bool,
    pub attacked: bool,
    pub least_attacker: bool,
    pub king_zone: bool,
    pub connected: bool,
    pub pawn_pieces: bool,
    pub pawn_spans: bool,
    pub control_counts: bool,
    pub line_control: bool,
    pub mobility: bool,
    pub passers: bool,
    pub factorised: bool,
    /// With the pawn predicates: one shared input per predicate and colour,
    /// active with each square's (training only: folded into the per-square
    /// weights at save; the author, 2026-10-18: "backward pawn" is otherwise
    /// learned 64 times over, each from a sliver of the data). After every
    /// other input.
    pub pawn_factorised: bool,
}

pub const FACTORISER: usize = 768;
pub const PAWN_FACTORISER: usize = PREDICATES * 2;

impl<B: SparseInputType<RequiredDataType = ChessBoard>> SparseInputType for WithExtras<B> {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        let factoriser = if self.factorised { FACTORISER } else { 0 };
        factoriser
            + self.base.num_inputs()
            + if self.pawns { PAWN_FEATURES } else { 0 }
            + if self.pawn_spans { SPAN_FEATURES } else { 0 }
            + if self.passers { PASSER_FEATURES } else { 0 }
            + if self.knights { pairs::KNIGHT_FEATURES } else { 0 }
            + if self.sliders { pairs::SLIDER_FEATURES } else { 0 }
            + if self.slider_pawns { pairs::SLIDER_PAWN_FEATURES } else { 0 }
            + if self.attacked { squares::ATTACKED_FEATURES } else { 0 }
            + if self.least_attacker { squares::LEAST_ATTACKER_FEATURES } else { 0 }
            + if self.king_zone { squares::KING_ZONE_FEATURES } else { 0 }
            + if self.connected { pairs::SLIDER_FEATURES } else { 0 }
            + if self.pawn_pieces { squares::PAWN_PIECE_FEATURES } else { 0 }
            + if self.control_counts { squares::CONTROL_COUNT_FEATURES } else { 0 }
            + if self.line_control { pairs::LINE_FEATURES } else { 0 }
            + if self.mobility { squares::MOBILITY_FEATURES } else { 0 }
            + if self.passers { squares::PASSER_PIECE_FEATURES } else { 0 }
            + if self.pawns && self.pawn_factorised { PAWN_FACTORISER } else { 0 }
    }

    fn max_active(&self) -> usize {
        self.base.max_active() * if self.factorised { 2 } else { 1 }
            + if self.pawns { PREDICATES * 16 } else { 0 }
            + if self.pawn_spans { SPAN_FEATURES } else { 0 }
            + if self.passers { PASSER_MAX } else { 0 }
            + if self.knights { pairs::MAX_ACTIVE } else { 0 }
            + if self.sliders { pairs::FAMILY_MAX } else { 0 }
            + if self.slider_pawns { pairs::FAMILY_MAX } else { 0 }
            + if self.attacked { squares::SQUARE_MAX } else { 0 }
            + if self.least_attacker { squares::SQUARE_MAX } else { 0 }
            + if self.king_zone { squares::SQUARE_MAX } else { 0 }
            + if self.connected { pairs::FAMILY_MAX } else { 0 }
            + if self.pawn_pieces { squares::SQUARE_MAX } else { 0 }
            + if self.control_counts { squares::CONTROL_COUNT_MAX } else { 0 }
            + if self.line_control { pairs::FAMILY_MAX } else { 0 }
            + if self.mobility { squares::MOBILITY_MAX } else { 0 }
            + if self.passers { squares::SQUARE_MAX } else { 0 }
            + if self.pawns && self.pawn_factorised { PREDICATES * 16 } else { 0 }
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let shift = if self.factorised { FACTORISER } else { 0 };
        self.base.map_features(pos, |stm, ntm| {
            f(shift + stm, shift + ntm);
            if self.factorised {
                f(stm % 768, ntm % 768);
            }
        });
        let mut off = shift + self.base.num_inputs();
        let flip = |ksq: u8| if self.mirrored && ksq % 8 > 3 { 7 } else { 0 };
        let (stm_flip, ntm_flip) = (flip(pos.our_ksq()), flip(pos.opp_ksq()));
        if self.pawns {
            let shared = self.num_inputs() - PAWN_FACTORISER;  // the predicate factoriser's block, when there is one
            predicates(pos, |p, c, s| {
                let stm = off + (p * 2 + c) * 64 + (s ^ stm_flip);
                let ntm = off + (p * 2 + (1 - c)) * 64 + ((s ^ 56) ^ ntm_flip);
                f(stm, ntm);
                if self.pawn_factorised {
                    f(shared + p * 2 + c, shared + p * 2 + (1 - c));
                }
            });
            off += PAWN_FEATURES;
        }
        if self.pawn_spans {
            for (stm, ntm) in pawn_spans(pos, [stm_flip, ntm_flip]) {
                f(off + stm, off + ntm);
            }
            off += SPAN_FEATURES;
        }
        if self.passers {
            passers(pos, |p, c, s| {
                let stm = off + (p * 2 + c) * 64 + (s ^ stm_flip);
                let ntm = off + (p * 2 + (1 - c)) * 64 + ((s ^ 56) ^ ntm_flip);
                f(stm, ntm);
            });
            off += PASSER_FEATURES;
        }
        if self.knights {
            let index = pairs::pair_index();
            pairs::knight_pairs(pos, |c, a, b| {
                let stm = index[a ^ stm_flip][b ^ stm_flip] as usize;
                let ntm = index[a ^ 56 ^ ntm_flip][b ^ 56 ^ ntm_flip] as usize;
                f(off + c * pairs::KNIGHT_PAIRS + stm, off + (1 - c) * pairs::KNIGHT_PAIRS + ntm);
            });
            off += pairs::KNIGHT_FEATURES;
        }
        if self.sliders {
            for (stm, ntm) in pairs::slider_pairs(pos, stm_flip, ntm_flip, false) {
                f(off + stm, off + ntm);
            }
            off += pairs::SLIDER_FEATURES;
        }
        if self.slider_pawns {
            for (stm, ntm) in pairs::slider_pawn(pos, stm_flip, ntm_flip) {
                f(off + stm, off + ntm);
            }
            off += pairs::SLIDER_PAWN_FEATURES;
        }
        if self.connected {
            for (stm, ntm) in pairs::slider_pairs(pos, stm_flip, ntm_flip, true) {
                f(off + stm, off + ntm);
            }
            off += pairs::SLIDER_FEATURES;
        }
        if self.line_control {
            for (stm, ntm) in pairs::line_control(pos, stm_flip, ntm_flip) {
                f(off + stm, off + ntm);
            }
            off += pairs::LINE_FEATURES;
        }
        let flips = [stm_flip, ntm_flip];
        let families: [(bool, fn(&ChessBoard, [usize; 2]) -> Vec<(usize, usize)>, usize); 7] = [
            (self.attacked, squares::attacked, squares::ATTACKED_FEATURES),
            (self.least_attacker, squares::least_attacker, squares::LEAST_ATTACKER_FEATURES),
            (self.king_zone, squares::king_zone, squares::KING_ZONE_FEATURES),
            (self.pawn_pieces, squares::pawn_pieces, squares::PAWN_PIECE_FEATURES),
            (self.control_counts, squares::control_counts, squares::CONTROL_COUNT_FEATURES),
            (self.mobility, squares::mobility, squares::MOBILITY_FEATURES),
            (self.passers, squares::passers, squares::PASSER_PIECE_FEATURES),
        ];
        for (on, family, size) in families {
            if on {
                for (stm, ntm) in family(pos, flips) {
                    f(off + stm, off + ntm);
                }
                off += size;
            }
        }
    }

    fn shorthand(&self) -> String {
        let p = if self.pawns { "+pawns" } else { "" };
        let k = if self.knights { "+knightpairs" } else { "" };
        let k = format!("{k}{}{}", if self.sliders { "+sliderpairs" } else { "" }, if self.slider_pawns { "+sliderpawn" } else { "" });
        let k = format!("{k}{}", if self.attacked || self.least_attacker || self.king_zone || self.connected || self.pawn_pieces || self.pawn_spans || self.control_counts || self.line_control || self.mobility || self.passers { "+more" } else { "" });
        let f = format!("{}{}", if self.factorised { "+factoriser" } else { "" }, if self.pawns && self.pawn_factorised { "+pawnfactoriser" } else { "" });
        format!("{}{p}{k}{f}", self.base.shorthand())
    }

    fn description(&self) -> String {
        let p = if self.pawns { ", with pawn predicates" } else { "" };
        let k = if self.knights { ", with knight pairs" } else { "" };
        let k = format!("{k}{}{}", if self.sliders { ", with slider pairs" } else { "" }, if self.slider_pawns { ", with slider x pawn" } else { "" });
        let k = format!("{k}{}", if self.attacked || self.least_attacker || self.king_zone || self.connected || self.pawn_pieces || self.pawn_spans || self.control_counts || self.line_control || self.mobility || self.passers { ", with more extra families" } else { "" });
        let f = format!("{}{}", if self.factorised { ", factorised" } else { "" }, if self.pawns && self.pawn_factorised { ", predicates factorised" } else { "" });
        format!("{}{p}{k}{f}", self.base.description())
    }
}

pub const SPAN_FEATURES: usize = 128;

/// The pawn-control map (src/pawn_features.h): for each side, the squares
/// its pawns attack now or can after advancing, stopping at the first pawn
/// ahead on the file. (stm, ntm) offsets into its block: colour x 64 +
/// square, in each view's orientation and mirroring.
pub fn pawn_spans(pos: &ChessBoard, flips: [usize; 2]) -> Vec<(usize, usize)> {
    let mut pawns = [0u64; 2];
    for (piece, square) in pos.into_iter() {
        if piece & 7 == 0 {
            pawns[usize::from(piece & 8 != 0)] |= 1u64 << square;
        }
    }
    let all = pawns[0] | pawns[1];
    let mut out = Vec::new();
    for c in 0..2 {
        // The side to move's pawns move up the board, the other's down.
        let mut span = 0u64;
        let mut b = pawns[c];
        while b != 0 {
            let s = b.trailing_zeros() as i32;
            b &= b - 1;
            let (f, r) = (s % 8, s / 8);
            let step = if c == 0 { 1 } else { -1 };
            let mut q = r + step;
            while (0..8).contains(&q) {
                if f > 0 {
                    span |= 1u64 << (q * 8 + f - 1);
                }
                if f < 7 {
                    span |= 1u64 << (q * 8 + f + 1);
                }
                if all & (1u64 << (q * 8 + f)) != 0 {
                    break;
                }
                q += step;
            }
        }
        while span != 0 {
            let s = span.trailing_zeros() as usize;
            span &= span - 1;
            let view = |v: usize| if v == 0 { s ^ flips[0] } else { s ^ 56 ^ flips[1] };
            out.push((c * 64 + view(0), (1 - c) * 64 + view(1)));
        }
    }
    out
}

pub const PASSER_LABELS: usize = 3;
pub const PASSER_FEATURES: usize = PASSER_LABELS * 2 * 64;
pub const PASSER_MAX: usize = PASSER_LABELS * 16;

/// Passed-pawn quality, the pawn-only part (src/pawn_features.h; the
/// author): for `own` pawns moving up against `opp` moving down, calls
/// f(label, square) for each passed pawn: 0 protected (an own pawn defends
/// it), 1 connected (an own passer on an adjacent file at most a rank
/// apart), 2 outside (no other pawn within two files, and `opp` has pawns).
fn up_passers(own: u64, opp: u64, mut f: impl FnMut(usize, usize)) {
    let mut passed = 0u64;
    let mut b = own;
    while b != 0 {
        let s = b.trailing_zeros() as usize;
        b &= b - 1;
        if opp & (file_bb(s % 8) | adjacent_files(s % 8)) & ranks_above(s) == 0 {
            passed |= 1u64 << s;
        }
    }
    let mut b = passed;
    while b != 0 {
        let s = b.trailing_zeros() as usize;
        b &= b - 1;
        let (file, r) = (s % 8, s / 8);
        let mut defend_from = 0u64;
        if s >= 9 && file > 0 {
            defend_from |= 1u64 << (s - 9);
        }
        if s >= 7 && file < 7 {
            defend_from |= 1u64 << (s - 7);
        }
        let mut near_ranks = 0xFFu64 << (8 * r);
        if r > 0 {
            near_ranks |= 0xFFu64 << (8 * (r - 1));
        }
        if r < 7 {
            near_ranks |= 0xFFu64 << (8 * (r + 1));
        }
        let mut near_files = file_bb(file) | adjacent_files(file);
        if file > 1 {
            near_files |= file_bb(file - 2);
        }
        if file < 6 {
            near_files |= file_bb(file + 2);
        }
        let active = [
            defend_from & own != 0,
            passed & adjacent_files(file) & near_ranks != 0,
            opp != 0 && (own | opp) & near_files & !(1u64 << s) == 0,
        ];
        for (p, &on) in active.iter().enumerate() {
            if on {
                f(p, s);
            }
        }
    }
}

/// The pawn-only passer labels of both sides: f(label, colour, square),
/// colour 0 the side to move.
pub fn passers(pos: &ChessBoard, mut f: impl FnMut(usize, usize, usize)) {
    let (mut us, mut them) = (0u64, 0u64);
    for (piece, square) in pos.into_iter() {
        if piece & 7 == 0 {
            if piece & 8 == 0 {
                us |= 1u64 << square;
            } else {
                them |= 1u64 << square;
            }
        }
    }
    up_passers(us, them, |p, s| f(p, 0, s));
    up_passers(them.swap_bytes(), us.swap_bytes(), |p, s| f(p, 1, s ^ 56));
}
