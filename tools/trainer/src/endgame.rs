// Output buckets by endgame type (src/nnue.cpp, endgame_type() and
// output_bucket(); ROADMAP, Try 5, G; the author, 2026-10-05): three buckets
// after the others take over where a type applies: 0 no pawns, 1 pawns only
// (besides the kings), 2 opposite-coloured bishops only (one each, with
// pawns). The others are the material buckets, split by the side to move's
// shelter with king-state outputs (KS).

use bullet_lib::game::{formats::bulletformat::ChessBoard, outputs::OutputBuckets};

use crate::kingstate;

pub const TYPES: usize = 3;

pub fn endgame_type(pos: &ChessBoard) -> Option<usize> {
    let (mut pawns, mut others) = (0u32, 0u32);
    let mut bishops = [(0u32, 0usize); 2]; // count and square colour, by side
    for (piece, square) in pos.into_iter() {
        let (c, pt, sq) = (usize::from(piece & 8 != 0), piece & 7, square as usize);
        match pt {
            0 => pawns += 1,
            5 => {}
            2 => {
                bishops[c].0 += 1;
                bishops[c].1 = (sq % 8 + sq / 8) % 2;
                others += 1;
            }
            _ => others += 1,
        }
    }
    if pawns == 0 {
        Some(0)
    } else if others == 0 {
        Some(1)
    } else if others == 2 && bishops[0].0 == 1 && bishops[1].0 == 1 && bishops[0].1 != bishops[1].1 {
        Some(2)
    } else {
        None
    }
}

/// N material buckets (split three ways by shelter when KS), then TYPES.
#[derive(Clone, Copy, Default)]
pub struct EndgameOutput<const N: usize, const KS: bool>;

impl<const N: usize, const KS: bool> OutputBuckets<ChessBoard> for EndgameOutput<N, KS> {
    const BUCKETS: usize = if KS { N * 3 + TYPES } else { N + TYPES };

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        let others = if KS { N * 3 } else { N };
        let b = match endgame_type(pos) {
            Some(t) => others + t,
            None if KS => kingstate::material_bucket(pos, N) * 3 + kingstate::stm_shelter(pos),
            None => kingstate::material_bucket(pos, N),
        };
        b as u8
    }
}
