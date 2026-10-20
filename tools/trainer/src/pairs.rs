// Pair features (ROADMAP, Try 5, the shared framework), the trainer's side
// of src/pair_features.h: knight pairs (group E), two same-colour knights a
// knight's move apart, one feature per colour and unordered square pair,
// numbered as the engine numbers them (by lower square, then higher, from
// a1).

use bullet_lib::game::formats::bulletformat::ChessBoard;

pub const KNIGHT_PAIRS: usize = 168;
pub const KNIGHT_FEATURES: usize = 2 * KNIGHT_PAIRS;
// Active pairs a position at most, as the engine bounds them.
pub const MAX_ACTIVE: usize = 64;

fn knight_attacks(sq: usize) -> u64 {
    let (f, r) = ((sq % 8) as i32, (sq / 8) as i32);
    let mut b = 0u64;
    for (df, dr) in [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)] {
        let (nf, nr) = (f + df, r + dr);
        if (0..8).contains(&nf) && (0..8).contains(&nr) {
            b |= 1u64 << (nr * 8 + nf);
        }
    }
    b
}

/// pair_index()[a][b]: the index of the unordered pair {a, b}, or -1.
pub fn pair_index() -> &'static [[i16; 64]; 64] {
    static TABLE: std::sync::OnceLock<[[i16; 64]; 64]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [[-1i16; 64]; 64];
        let mut n = 0;
        for a in 0..64 {
            let mut m = knight_attacks(a);
            while m != 0 {
                let b = m.trailing_zeros() as usize;
                m &= m - 1;
                if b > a {
                    t[a][b] = n;
                    t[b][a] = n;
                    n += 1;
                }
            }
        }
        assert_eq!(n as usize, KNIGHT_PAIRS);
        t
    })
}

/// Calls f(colour, a, b) for every knight pair of a board seen from the
/// side to move (colour 0 the side to move, squares in its view).
pub fn knight_pairs(pos: &ChessBoard, mut f: impl FnMut(usize, usize, usize)) {
    let mut knights = [0u64; 2];
    for (piece, square) in pos.into_iter() {
        if piece & 7 == 1 {
            knights[usize::from(piece & 8 != 0)] |= 1u64 << square;
        }
    }
    for (c, &own) in knights.iter().enumerate() {
        let mut b = own;
        while b != 0 {
            let a = b.trailing_zeros() as usize;
            b &= b - 1;
            let mut m = knight_attacks(a) & own & b;  // partners above a
            while m != 0 {
                let s = m.trailing_zeros() as usize;
                m &= m - 1;
                f(c, a, s);
            }
        }
    }
}

// Slider pairs (group B) and slider x pawn (group C), as src/pair_features.h
// defines them. A family with more than FAMILY_MAX pairs in a position
// contributes none (the engine's rule).
pub const ORTHO_PAIRS: usize = 448;
pub const DIAG_PAIRS: usize = 280;
pub const SLIDER_FEATURES: usize = 16 * (ORTHO_PAIRS + DIAG_PAIRS);
pub const FILE_PAIRS: usize = 336;
pub const SLIDER_PAWN_FEATURES: usize = 8 * FILE_PAIRS;
pub const FAMILY_MAX: usize = 64;

struct LineTables {
    ortho: [[i16; 64]; 64],
    diag: [[i16; 64]; 64],
    file: [[i16; 64]; 64],
}

fn line_tables() -> &'static LineTables {
    static TABLES: std::sync::OnceLock<LineTables> = std::sync::OnceLock::new();
    TABLES.get_or_init(|| {
        let number = |test: &dyn Fn(usize, usize) -> bool| {
            let mut t = [[-1i16; 64]; 64];
            let mut n = 0;
            for a in 0..64 {
                for b in a + 1..64 {
                    if test(a, b) {
                        t[a][b] = n;
                        t[b][a] = n;
                        n += 1;
                    }
                }
            }
            t
        };
        let (f, r) = (|s: usize| (s % 8) as i32, |s: usize| (s / 8) as i32);
        let ortho = number(&|a, b| f(a) == f(b) || r(a) == r(b));
        let diag = number(&|a, b| (f(a) - f(b)).abs() == (r(a) - r(b)).abs());
        let mut file = [[-1i16; 64]; 64];
        let mut n = 0;
        for s in 0..64 {
            for p in 0..64 {
                if p != s && f(p) == f(s) && (1..=6).contains(&r(p)) {
                    file[s][p] = n;
                    n += 1;
                }
            }
        }
        assert_eq!(n as usize, FILE_PAIRS);
        LineTables { ortho, diag, file }
    })
}

/// The squares strictly between two squares on a line (empty if not on one).
pub fn between(a: usize, b: usize) -> u64 {
    let (fa, ra, fb, rb) = ((a % 8) as i32, (a / 8) as i32, (b % 8) as i32, (b / 8) as i32);
    let (df, dr) = (fb - fa, rb - ra);
    if !(df == 0 || dr == 0 || df.abs() == dr.abs()) || a == b {
        return 0;
    }
    let (sf, sr) = (df.signum(), dr.signum());
    let mut out = 0u64;
    let (mut f, mut r) = (fa + sf, ra + sr);
    while (f, r) != (fb, rb) {
        out |= 1u64 << (r * 8 + f);
        f += sf;
        r += sr;
    }
    out
}

/// Slider pairs of a board seen from the side to move: (stm, ntm) offsets
/// into the slider-pair block, with each view's squares mirrored by its flip;
/// with `connected`, only pairs with nothing between them (group D).
pub fn slider_pairs(pos: &ChessBoard, stm_flip: usize, ntm_flip: usize, connected: bool) -> Vec<(usize, usize)> {
    let mut occ = 0u64;
    for (_, square) in pos.into_iter() {
        occ |= 1u64 << square;
    }
    let t = line_tables();
    // (square, colour (0 the side to move), queen, moves orthogonally, diagonally)
    let mut s = Vec::new();
    for (piece, square) in pos.into_iter() {
        let pt = piece & 7;
        if (2..=4).contains(&pt) {
            s.push((square as usize, usize::from(piece & 8 != 0), usize::from(pt == 4), pt != 2, pt != 3));
        }
    }
    let feature = |a: (usize, usize, usize, bool, bool), b: (usize, usize, usize, bool, bool), view: usize, flip: usize| {
        let sq = |x: usize| if view == 0 { x ^ flip } else { x ^ 56 ^ flip };
        let (mut lo, mut hi) = ((sq(a.0), a.1 ^ view, a.2), (sq(b.0), b.1 ^ view, b.2));
        if lo.0 > hi.0 {
            std::mem::swap(&mut lo, &mut hi);
        }
        let kind = (lo.1 * 2 + hi.1) * 4 + lo.2 * 2 + hi.2;
        let (o, d) = (t.ortho[lo.0][hi.0], t.diag[lo.0][hi.0]);
        if o >= 0 && a.3 && b.3 {
            Some(kind * ORTHO_PAIRS + o as usize)
        } else if d >= 0 && a.4 && b.4 {
            Some(16 * ORTHO_PAIRS + kind * DIAG_PAIRS + d as usize)
        } else {
            None
        }
    };
    let mut out = Vec::new();
    for i in 0..s.len() {
        for j in i + 1..s.len() {
            if connected && between(s[i].0, s[j].0) & occ != 0 {
                continue;
            }
            if let (Some(x), Some(y)) = (feature(s[i], s[j], 0, stm_flip), feature(s[i], s[j], 1, ntm_flip)) {
                out.push((x, y));
            }
        }
    }
    if out.len() > FAMILY_MAX { Vec::new() } else { out }
}

/// Slider x pawn of a board seen from the side to move: (stm, ntm) offsets
/// into its block.
pub fn slider_pawn(pos: &ChessBoard, stm_flip: usize, ntm_flip: usize) -> Vec<(usize, usize)> {
    let t = line_tables();
    let (mut sliders, mut pawns) = (Vec::new(), Vec::new());
    for (piece, square) in pos.into_iter() {
        let (pt, c, sq) = (piece & 7, usize::from(piece & 8 != 0), square as usize);
        if pt == 0 {
            pawns.push((sq, c));
        } else if pt == 3 || pt == 4 {
            sliders.push((sq, c, usize::from(pt == 4)));
        }
    }
    let mut out = Vec::new();
    for &(s, c, queen) in &sliders {
        for &(p, pc) in &pawns {
            if p % 8 != s % 8 {
                continue;
            }
            let one = |view: usize, flip: usize| {
                let sq = |x: usize| if view == 0 { x ^ flip } else { x ^ 56 ^ flip };
                (((c ^ view) * 2 + queen) * 2 + (pc ^ view)) * FILE_PAIRS + t.file[sq(s)][sq(p)] as usize
            };
            out.push((one(0, stm_flip), one(1, ntm_flip)));
        }
    }
    if out.len() > FAMILY_MAX { Vec::new() } else { out }
}

/// Line control (src/pair_features.h, group L): files, ranks and the 22
/// diagonals of length 3 or more, presence encoded for each side.
pub const LINE_FEATURES: usize = 336;

fn mirror_files(b: u64) -> u64 {
    u64::from_le_bytes(b.to_le_bytes().map(|x| x.reverse_bits()))
}

fn diagonal_masks() -> &'static [u64; 22] {
    static MASKS: std::sync::OnceLock<[u64; 22]> = std::sync::OnceLock::new();
    MASKS.get_or_init(|| {
        let mut m = [0u64; 22];
        for s in 0..64usize {
            let (f, r) = ((s % 8) as i32, (s / 8) as i32);
            let (a, b) = (f - r + 7, f + r);
            if (2..=12).contains(&a) {
                m[(a - 2) as usize] |= 1u64 << s;
            }
            if (2..=12).contains(&b) {
                m[(11 + b - 2) as usize] |= 1u64 << s;
            }
        }
        m
    })
}

/// One view's line-control inputs, from bitboards in that view's
/// coordinates ([0] the view's side, [1] the other).
fn line_inputs(pawns: [u64; 2], majors: [u64; 2], diags: [u64; 2]) -> Vec<usize> {
    let mut out = Vec::new();
    let both = pawns[0] | pawns[1];
    for x in 0..8 {
        let file = 0x0101_0101_0101_0101u64 << x;
        let ps = (if pawns[0] & file != 0 { 2 } else { 0 }) + (if pawns[1] & file != 0 { 1 } else { 0 });
        let ms = (if majors[0] & file != 0 { 2 } else { 0 }) + (if majors[1] & file != 0 { 1 } else { 0 });
        if ms != 0 || ps == 0 {
            out.push(x * 16 + ps * 4 + ms);
        }
    }
    for y in 0..8 {
        let rank = 0xFFu64 << (8 * y);
        let ms = (if majors[0] & rank != 0 { 2 } else { 0 }) + (if majors[1] & rank != 0 { 1 } else { 0 });
        if ms != 0 {
            out.push(128 + y * 4 + ms);
        }
    }
    for (d, &line) in diagonal_masks().iter().enumerate() {
        let ds = (if diags[0] & line != 0 { 2 } else { 0 }) + (if diags[1] & line != 0 { 1 } else { 0 });
        if ds != 0 {
            out.push(160 + d * 8 + ds * 2 + usize::from(both & line == 0));
        }
    }
    out
}

/// (stm, ntm) pairs of line-control inputs (the two views have the same
/// number: a line counts in both or neither).
pub fn line_control(pos: &ChessBoard, stm_flip: usize, ntm_flip: usize) -> Vec<(usize, usize)> {
    let (mut pawns, mut majors, mut diags) = ([0u64; 2], [0u64; 2], [0u64; 2]);
    for (piece, square) in pos.into_iter() {
        let (pt, c, b) = (piece & 7, usize::from(piece & 8 != 0), 1u64 << square);
        match pt {
            0 => pawns[c] |= b,
            2 => diags[c] |= b,
            3 => majors[c] |= b,
            4 => {
                majors[c] |= b;
                diags[c] |= b;
            }
            _ => {}
        }
    }
    let stm_view = |b: u64| if stm_flip != 0 { mirror_files(b) } else { b };
    let ntm_view = |b: u64| {
        let v = b.swap_bytes();
        if ntm_flip != 0 { mirror_files(v) } else { v }
    };
    let stm = line_inputs(pawns.map(stm_view), majors.map(stm_view), diags.map(stm_view));
    let ntm = line_inputs(
        [ntm_view(pawns[1]), ntm_view(pawns[0])],
        [ntm_view(majors[1]), ntm_view(majors[0])],
        [ntm_view(diags[1]), ntm_view(diags[0])],
    );
    stm.into_iter().zip(ntm).collect()
}
