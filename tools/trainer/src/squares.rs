// Square control, the trainer's side of src/square_features.h: the same
// definitions and numbering, on bullet's boards (seen from the side to move:
// its pawns attack up the board). Each function returns (stm, ntm) offsets
// into its family's block, or nothing past the family's bound (the engine's
// all-or-nothing rule). The threat family T (bit 4) is retired (the
// originality audit, R2): its definition and indexing were Stockfish 19's
// threat inputs. Threat knowledge by chess logic is family W, on its own
// branch (family-w).

use bullet_lib::game::formats::bulletformat::ChessBoard;

pub const ATTACKED_FEATURES: usize = 128;
pub const LEAST_ATTACKER_FEATURES: usize = 768;
pub const KING_ZONE_FEATURES: usize = 128;
pub const SQUARE_MAX: usize = 128;

const DIAG: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
const ORTH: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const KNIGHT: [(i32, i32); 8] = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)];
const KING: [(i32, i32); 8] = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)];

fn steps(sq: usize, dirs: &[(i32, i32)]) -> u64 {
    let (f, r) = ((sq % 8) as i32, (sq / 8) as i32);
    let mut b = 0u64;
    for &(df, dr) in dirs {
        let (nf, nr) = (f + df, r + dr);
        if (0..8).contains(&nf) && (0..8).contains(&nr) {
            b |= 1u64 << (nr * 8 + nf);
        }
    }
    b
}

fn rays(sq: usize, dirs: &[(i32, i32)], occ: u64) -> u64 {
    let mut b = 0u64;
    for &(df, dr) in dirs {
        let (mut f, mut r) = ((sq % 8) as i32 + df, (sq / 8) as i32 + dr);
        while (0..8).contains(&f) && (0..8).contains(&r) {
            let s = 1u64 << (r * 8 + f);
            b |= s;
            if occ & s != 0 {
                break;
            }
            f += df;
            r += dr;
        }
    }
    b
}

/// A piece's attacks: colour 0 the side whose pawns attack up the board.
fn attacks(colour: usize, piece_type: usize, sq: usize, occ: u64) -> u64 {
    match piece_type {
        0 => steps(sq, if colour == 0 { &[(-1, 1), (1, 1)] } else { &[(-1, -1), (1, -1)] }),
        1 => steps(sq, &KNIGHT),
        2 => rays(sq, &DIAG, occ),
        3 => rays(sq, &ORTH, occ),
        4 => rays(sq, &DIAG, occ) | rays(sq, &ORTH, occ),
        _ => steps(sq, &KING),
    }
}

/// The board as (square, colour (0 the side to move), type) and occupancy.
fn pieces(pos: &ChessBoard) -> (Vec<(usize, usize, usize)>, [Option<(usize, usize)>; 64], u64) {
    let mut list = Vec::new();
    let mut at = [None; 64];
    let mut occ = 0u64;
    for (piece, square) in pos.into_iter() {
        let (sq, c, pt) = (square as usize, usize::from(piece & 8 != 0), (piece & 7) as usize);
        list.push((sq, c, pt));
        at[sq] = Some((c, pt));
        occ |= 1u64 << sq;
    }
    (list, at, occ)
}

// A square in a view: the side to move's as on the board, the other's
// flipped vertically; then mirrored by the view's flip.
fn view_sq(view: usize, flip: usize, sq: usize) -> usize {
    if view == 0 { sq ^ flip } else { sq ^ 56 ^ flip }
}

fn bounded(out: Vec<(usize, usize)>, bound: usize) -> Vec<(usize, usize)> {
    if out.len() > bound { Vec::new() } else { out }
}

/// Each side's attacks by piece type (colour 0 the side to move).
fn attacks_by_type(pos: &ChessBoard) -> [[u64; 6]; 2] {
    let (list, _, occ) = pieces(pos);
    let mut by = [[0u64; 6]; 2];
    for &(s, c, pt) in &list {
        by[c][pt] |= attacks(c, pt, s, occ);
    }
    by
}

pub fn attacked(pos: &ChessBoard, flips: [usize; 2]) -> Vec<(usize, usize)> {
    let by = attacks_by_type(pos);
    let mut out = Vec::new();
    for c in 0..2 {
        let mut b = by[c].iter().fold(0, |x, y| x | y);
        while b != 0 {
            let s = b.trailing_zeros() as usize;
            b &= b - 1;
            out.push((c * 64 + view_sq(0, flips[0], s), (1 - c) * 64 + view_sq(1, flips[1], s)));
        }
    }
    bounded(out, SQUARE_MAX)
}

pub fn least_attacker(pos: &ChessBoard, flips: [usize; 2]) -> Vec<(usize, usize)> {
    let by = attacks_by_type(pos);
    let mut out = Vec::new();
    for c in 0..2 {
        let mut done = 0u64;
        for pt in 0..6 {
            let mut b = by[c][pt] & !done;
            while b != 0 {
                let s = b.trailing_zeros() as usize;
                b &= b - 1;
                out.push((
                    (c * 64 + view_sq(0, flips[0], s)) * 6 + pt,
                    ((1 - c) * 64 + view_sq(1, flips[1], s)) * 6 + pt,
                ));
            }
            done |= by[c][pt];
        }
    }
    bounded(out, SQUARE_MAX)
}

pub fn king_zone(pos: &ChessBoard, flips: [usize; 2]) -> Vec<(usize, usize)> {
    let by = attacks_by_type(pos);
    let (list, _, _) = pieces(pos);
    let mut out = Vec::new();
    for c in 0..2 {
        let o = 1 - c;
        let Some(&(k, _, _)) = list.iter().find(|&&(_, pc, pt)| pc == o && pt == 5) else { continue };
        let ring = steps(k, &KING) | (1u64 << k);
        // The squares in front of the ring from the king's side: the side
        // to move's pawns push up the board, the other's down.
        let zone = ring | if o == 0 { ring << 8 } else { ring >> 8 };
        let mut b = by[c].iter().fold(0, |x, y| x | y) & zone;
        while b != 0 {
            let s = b.trailing_zeros() as usize;
            b &= b - 1;
            out.push((c * 64 + view_sq(0, flips[0], s), (1 - c) * 64 + view_sq(1, flips[1], s)));
        }
    }
    bounded(out, SQUARE_MAX)
}

pub const PAWN_PIECE_FEATURES: usize = 512;

/// The piece-dependent pawn predicates (group A2): (predicate x 2 + the
/// pawn's colour) x 64 + its square, as src/square_features.h defines them.
pub fn pawn_pieces(pos: &ChessBoard, flips: [usize; 2]) -> Vec<(usize, usize)> {
    let (list, at, occ) = pieces(pos);
    let mut pawns = [0u64; 2];
    let mut kings = [None; 2];
    for &(s, c, pt) in &list {
        if pt == 0 {
            pawns[c] |= 1u64 << s;
        } else if pt == 5 {
            kings[c] = Some(s);
        }
    }
    let piece_attackers = |c: usize, s: usize| {
        list.iter().any(|&(a, pc, pt)| pc == c && pt != 0 && attacks(c, pt, a, occ) & (1u64 << s) != 0)
    };
    // c's pieces pinned to their king by an enemy slider.
    let pinned = |c: usize| {
        let Some(k) = kings[c] else { return 0u64 };
        let mut out = 0u64;
        for &(a, pc, pt) in &list {
            let lines = match pt {
                2 => attacks(0, 2, k, 0),
                3 => attacks(0, 3, k, 0),
                4 => attacks(0, 4, k, 0),
                _ => 0,
            };
            if pc == c || lines & (1u64 << a) == 0 {
                continue;
            }
            let b = crate::pairs::between(k, a) & occ;
            if b.count_ones() == 1 && at[b.trailing_zeros() as usize].is_some_and(|(bc, _)| bc == c) {
                out |= b;
            }
        }
        out
    };
    let mut out = Vec::new();
    for c in 0..2 {
        let pins = pinned(c);
        let mut b = pawns[c];
        while b != 0 {
            let s = b.trailing_zeros() as usize;
            b &= b - 1;
            // Own pawns defending s: where a pawn of the other direction on s attacks.
            let defenders = attacks(1 - c, 0, s, 0) & pawns[c];
            let stop = if c == 0 { s + 8 } else { s.wrapping_sub(8) };
            let blocked = stop < 64 && occ & !(pawns[0] | pawns[1]) & (1u64 << stop) != 0;
            let on = [
                piece_attackers(c, s),
                defenders != 0 && defenders & !pins == 0,
                piece_attackers(1 - c, s),
                blocked,
            ];
            for (p, &yes) in on.iter().enumerate() {
                if yes {
                    out.push(((p * 2 + c) * 64 + view_sq(0, flips[0], s), (p * 2 + 1 - c) * 64 + view_sq(1, flips[1], s)));
                }
            }
        }
    }
    bounded(out, SQUARE_MAX)
}

pub const CONTROL_COUNT_FEATURES: usize = 384;
pub const CONTROL_COUNT_MAX: usize = 384;

/// Control counts (src/square_features.h, the author's "negative board"):
/// how many times each side attacks each square, as a thermometer: (colour
/// x 64 + square) x 3 + level, level k for at least k + 1 attackers.
pub fn control_counts(pos: &ChessBoard, flips: [usize; 2]) -> Vec<(usize, usize)> {
    let (list, _, occ) = pieces(pos);
    let mut out = Vec::new();
    for c in 0..2 {
        let mut level = [0u64; 3];
        for &(s, pc, pt) in &list {
            if pc != c {
                continue;
            }
            let b = attacks(c, pt, s, occ);
            level[2] |= level[1] & b;
            level[1] |= level[0] & b;
            level[0] |= b;
        }
        for (k, &l) in level.iter().enumerate() {
            let mut b = l;
            while b != 0 {
                let s = b.trailing_zeros() as usize;
                b &= b - 1;
                out.push(((c * 64 + view_sq(0, flips[0], s)) * 3 + k, ((1 - c) * 64 + view_sq(1, flips[1], s)) * 3 + k));
            }
        }
    }
    out
}

pub const MOBILITY_STEPS: usize = 8 + 13 + 14 + 24;
pub const MOBILITY_FEATURES: usize = 4 * MOBILITY_STEPS;
pub const MOBILITY_MAX: usize = MOBILITY_FEATURES;

/// Mobility (src/square_features.h, the author): each knight, bishop, rook
/// and queen's safe moves (attacked squares not holding its own pieces nor
/// attacked by an enemy pawn), capped by type, as a thermometer per side
/// and type: (colour x 2 + level) x 59 + type's first + k - 1 for "at least
/// level + 1 of those pieces have k or more". No squares, so no flips.
pub fn mobility(pos: &ChessBoard, _flips: [usize; 2]) -> Vec<(usize, usize)> {
    const FIRST: [usize; 4] = [0, 8, 21, 35];
    const CAP: [u32; 4] = [8, 13, 14, 24];
    let (list, _, occ) = pieces(pos);
    let (mut own, mut pawn_attacks) = ([0u64; 2], [0u64; 2]);
    for &(s, c, pt) in &list {
        own[c] |= 1u64 << s;
        if pt == 0 {
            pawn_attacks[c] |= attacks(c, 0, s, occ);
        }
    }
    let mut out = Vec::new();
    for c in 0..2 {
        let safe = !own[c] & !pawn_attacks[1 - c];
        for t in 1..=4 {
            let mut level = [0u32; 2];
            for &(s, pc, pt) in &list {
                if pc == c && pt == t {
                    let reach = (1u32 << (attacks(c, t, s, occ) & safe).count_ones().min(CAP[t - 1])) - 1;
                    level[1] |= level[0] & reach;
                    level[0] |= reach;
                }
            }
            for (l, &bits) in level.iter().enumerate() {
                let mut b = bits;
                while b != 0 {
                    let k = b.trailing_zeros() as usize;
                    b &= b - 1;
                    let at = FIRST[t - 1] + k;
                    out.push(((c * 2 + l) * MOBILITY_STEPS + at, ((1 - c) * 2 + l) * MOBILITY_STEPS + at));
                }
            }
        }
    }
    out
}

pub const PASSER_PIECE_LABELS: usize = 8;
pub const PASSER_PIECE_FEATURES: usize = PASSER_PIECE_LABELS * 2 * 64;

/// Passed-pawn quality, the piece-dependent part (src/square_features.h;
/// the author): for each passed pawn, (label x 2 + colour) x 64 + square:
/// blockaded by an enemy knight, by an enemy bishop, by any other piece;
/// path clear; path safe; an own rook or queen behind it, an enemy one;
/// unstoppable (a pawn ending, path clear, the enemy king outside its
/// square, the side to move counted).
pub fn passers(pos: &ChessBoard, flips: [usize; 2]) -> Vec<(usize, usize)> {
    let (list, at, occ) = pieces(pos);
    let by = attacks_by_type(pos);
    let mut pawns = [0u64; 2];
    let mut kings = [None; 2];
    let mut pawn_ending = true;
    for &(s, c, pt) in &list {
        match pt {
            0 => pawns[c] |= 1u64 << s,
            5 => kings[c] = Some(s),
            _ => pawn_ending = false,
        }
    }
    let mut out = Vec::new();
    for c in 0..2 {
        let o = 1 - c;
        let enemy_attacks = by[o].iter().fold(0, |x, y| x | y);
        let mut b = pawns[c];
        while b != 0 {
            let s = b.trailing_zeros() as usize;
            b &= b - 1;
            let (f, r) = (s % 8, s / 8);
            let file = 0x0101_0101_0101_0101u64 << f;
            // Ahead and behind on its file: colour 0 moves up, 1 down.
            let above = if r == 7 { 0 } else { !0u64 << (8 * (r + 1)) };
            let below = if r == 0 { 0 } else { (1u64 << (8 * r)) - 1 };
            let (ahead_ranks, behind_ranks) = if c == 0 { (above, below) } else { (below, above) };
            let side_files = (if f > 0 { file >> 1 } else { 0 }) | (if f < 7 { file << 1 } else { 0 });
            if pawns[o] & (file | side_files) & ahead_ranks != 0 {
                continue; // not passed
            }
            let ahead = file & ahead_ranks;
            let behind = file & behind_ranks;
            let stop = if c == 0 { s + 8 } else { s - 8 };
            let blocker = at[stop];
            let back = attacks(0, 3, s, occ) & behind & occ;
            let major_behind = |side: usize| {
                back != 0 && at[back.trailing_zeros() as usize].is_some_and(|(bc, bt)| bc == side && (bt == 3 || bt == 4))
            };
            let mut unstoppable = false;
            if let (true, true, Some(k)) = (pawn_ending, ahead & occ == 0, kings[o]) {
                let own_rank = if c == 0 { r } else { 7 - r };
                let steps = 7 - own_rank - usize::from(own_rank == 1);
                let promo_rank = if c == 0 { 7 } else { 0 };
                let distance = (k % 8).abs_diff(f).max((k / 8).abs_diff(promo_rank));
                // The side to move is colour 0: the enemy moves first when c is 1.
                unstoppable = distance as i64 - i64::from(c == 1) > steps as i64;
            }
            let on = [
                blocker.is_some_and(|(bc, bt)| bc == o && bt == 1),
                blocker.is_some_and(|(bc, bt)| bc == o && bt == 2),
                blocker.is_some_and(|(bc, bt)| !(bc == o && (bt == 1 || bt == 2))),
                ahead & occ == 0,
                ahead & enemy_attacks == 0,
                major_behind(c),
                major_behind(o),
                unstoppable,
            ];
            for (p, &yes) in on.iter().enumerate() {
                if yes {
                    out.push(((p * 2 + c) * 64 + view_sq(0, flips[0], s), (p * 2 + 1 - c) * 64 + view_sq(1, flips[1], s)));
                }
            }
        }
    }
    bounded(out, SQUARE_MAX)
}
