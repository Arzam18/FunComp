// Coverage (ROADMAP, the plan to 3.0, 5a; the author, 2026-10-05: is there
// enough data for an architecture?). The first layer learns an input's
// weights only from positions where that input is active, so what an
// architecture needs is enough activations per input, not a total count.
// One pass over a data file with the same input definitions training
// uses: how often each input is active (both views), the inputs by band
// of activations and the share of all activations each band carries, the
// share of views in each king block (bucket or king state) and how many
// of its inputs are starved, the share of positions in each output
// bucket, and how many passes the schedule makes over the file. Counts
// from a sample are scaled to the whole file (the data is shuffled).
// A diagnostic, like held-out loss: games decide.

use std::fs::File;
use std::io::{BufReader, Read};

use bullet_lib::game::{formats::bulletformat::ChessBoard, inputs::SparseInputType};

use crate::kingstate;

pub struct Options {
    pub path: String,
    pub limit: u64,      // positions to read, 0 all
    pub blocks: usize,   // king blocks of 768 at the start of the inputs
    pub output_buckets: usize,
    pub output_kingstate: bool,
    pub output_endgame: bool,
    pub steps: u64,      // positions the schedule shows the trainer
    pub starved: u64,    // activations below which an input counts as starved
}

pub fn report<I: SparseInputType<RequiredDataType = ChessBoard>>(inputs: &I, o: &Options) {
    let total = std::fs::metadata(&o.path).expect("can't read the data").len() / 32;
    let n = if o.limit == 0 { total } else { o.limit.min(total) };
    let mut counts = vec![0u64; inputs.num_inputs()];
    let mut views = vec![0u64; o.blocks];
    let split = if o.output_kingstate { 3 } else { 1 };
    let types = if o.output_endgame { crate::endgame::TYPES } else { 0 };
    let mut outputs = vec![0u64; o.output_buckets * split + types];
    let mut reader = BufReader::with_capacity(1 << 22, File::open(&o.path).expect("can't open the data"));
    let mut buf = [0u8; 32];
    for _ in 0..n {
        reader.read_exact(&mut buf).expect("can't read the data");
        // SAFETY: the file is bullet's 32-byte plain-data boards.
        let board: ChessBoard = unsafe { std::ptr::read_unaligned(buf.as_ptr().cast()) };
        let mut first = true;
        inputs.map_features(&board, |stm, ntm| {
            counts[stm] += 1;
            counts[ntm] += 1;
            // Every piece input of a view lies in that view's king block.
            if first && stm < o.blocks * 768 && ntm < o.blocks * 768 {
                views[stm / 768] += 1;
                views[ntm / 768] += 1;
                first = false;
            }
        });
        let material = kingstate::material_bucket(&board, o.output_buckets);
        let shelter = if o.output_kingstate { kingstate::stm_shelter(&board) } else { 0 };
        match crate::endgame::endgame_type(&board).filter(|_| o.output_endgame) {
            Some(t) => outputs[o.output_buckets * split + t] += 1,
            None => outputs[material * split + shelter] += 1,
        }
    }

    let scale = total as f64 / n.max(1) as f64;
    let full = |c: u64| c as f64 * scale;
    let activations: f64 = counts.iter().map(|&c| full(c)).sum();
    println!(
        "{}: {} positions ({} read{}); the schedule's {} positions are {:.1} passes over it",
        o.path,
        total,
        n,
        if n < total { ", counts scaled to the whole file" } else { "" },
        o.steps,
        o.steps as f64 / total as f64
    );
    println!(
        "{} inputs, {:.1} active per position (both views), {:.0} activations per input on average",
        counts.len(),
        activations / total as f64,
        activations / counts.len() as f64
    );
    let bands = |part: &[u64]| {
        let total: f64 = part.iter().map(|&c| full(c)).sum();
        let bands: [(f64, f64, &str); 7] = [
            (0.0, 0.5, "never seen (includes impossible inputs: pawns on the end ranks, mirrored-away king files)"),
            (0.5, 1e3, "1 - 1k"),
            (1e3, 1e4, "1k - 10k"),
            (1e4, 1e5, "10k - 100k"),
            (1e5, 1e6, "100k - 1M"),
            (1e6, 1e7, "1M - 10M"),
            (1e7, f64::INFINITY, "10M and more"),
        ];
        for (lo, hi, name) in bands {
            let (k, a) = part
                .iter()
                .map(|&c| full(c))
                .filter(|&c| c >= lo && c < hi)
                .fold((0usize, 0f64), |(k, a), c| (k + 1, a + c));
            println!("  {name}: {k} ({:.1}%), {:.2}%", 100.0 * k as f64 / part.len() as f64, 100.0 * a / total.max(1.0));
        }
    };
    println!("activations per input over the file: inputs (share), share of all activations");
    bands(&counts);
    // The extra families' inputs follow the piece inputs: their own table
    // (one family per run makes it that family's).
    let base = o.blocks * 768;
    if counts.len() > base {
        let extra = &counts[base..];
        let a: f64 = extra.iter().map(|&c| full(c)).sum();
        println!(
            "extra inputs only: {}, {:.1} active per position (both views); inputs (share), share of their activations",
            extra.len(),
            a / total as f64
        );
        bands(extra);
    }
    if o.blocks > 1 {
        println!(
            "king blocks: share of views, inputs seen but under {} activations (starved), median activations of the block's seen inputs",
            o.starved
        );
        let all_views: u64 = views.iter().sum();
        for (b, &v) in views.iter().enumerate() {
            let mut seen: Vec<f64> = counts[b * 768..(b + 1) * 768].iter().map(|&c| full(c)).filter(|&c| c >= 0.5).collect();
            seen.sort_by(f64::total_cmp);
            let starved = seen.iter().filter(|&&c| c < o.starved as f64).count();
            let median = seen.get(seen.len() / 2).copied().unwrap_or(0.0);
            println!(
                "  block {b:2}: {:5.2}% of views, {starved} starved, median {:.0}",
                100.0 * v as f64 / all_views.max(1) as f64,
                median
            );
        }
    }
    if outputs.len() > 1 {
        let line: Vec<String> = outputs.iter().map(|&c| format!("{:.1}", 100.0 * c as f64 / n.max(1) as f64)).collect();
        println!(
            "output buckets (material{}{}), % of positions: {}",
            if o.output_kingstate { " x 3 by shelter" } else { "" },
            if o.output_endgame { ", then no pawns, pawns only, opposite bishops" } else { "" },
            line.join(" ")
        );
    }
}
