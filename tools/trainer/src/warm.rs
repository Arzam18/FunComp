// Warm starts (the author, 2026-10-07): a network begins from a trained
// base's checkpoint (`--init <checkpoint dir>`) instead of random weights,
// so a feature's test starts where the base ended and the feature learns
// what the base gets wrong. The base's weights are copied into the new
// layout so that the new network starts out computing the base's output:
//   - a weight of the same name and size: copied;
//   - the first layer of an extras network (factoriser block first, then
//     the piece inputs, then the extras: pawns.rs): the base's factoriser
//     and piece weights, the extras' rows zero;
//   - king blocks (buckets or king states, with the base's factoriser
//     copied): every block starts as the base's flat weights;
//   - output buckets: king-state outputs (material x 3) start as their
//     material bucket; endgame-type outputs as the material bucket that
//     type falls in most often in the data (the only one not exactly the
//     base's output: a type spans several material buckets).
// A fair test against it continues the base for the same superbatches.
// The optimiser's moments start afresh.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Write};

use bullet_lib::{
    game::formats::bulletformat::ChessBoard,
    nn::{
        ExecutionContext,
        optimiser::{AdamWOptimiser, Optimiser},
    },
};

use crate::{endgame, kingstate};

pub struct Plan {
    pub output_kingstate: bool,
    pub output_endgame: bool,
    pub data: String,
}

// bullet's labelled weights file: per weight its name, a newline, the
// count (usize, little-endian), the f32 values.
fn read_labelled(path: &str) -> HashMap<String, Vec<f32>> {
    let mut buf = Vec::new();
    File::open(path).unwrap_or_else(|e| panic!("can't open {path}: {e}")).read_to_end(&mut buf).unwrap();
    let mut map = HashMap::new();
    let mut at = 0;
    while at < buf.len() {
        let nl = at + buf[at..].iter().position(|&c| c == b'\n').expect("bad weights file");
        let id = String::from_utf8_lossy(&buf[at..nl]).to_string();
        at = nl + 1;
        let n = usize::from_le_bytes(buf[at..at + 8].try_into().unwrap());
        at += 8;
        let values = buf[at..at + 4 * n].chunks_exact(4).map(|w| f32::from_le_bytes(w.try_into().unwrap())).collect();
        at += 4 * n;
        map.insert(id, values);
    }
    map
}

fn write_labelled(path: &str, weights: &[(String, Vec<f32>)]) {
    let mut buf = Vec::new();
    for (id, values) in weights {
        buf.extend_from_slice(id.as_bytes());
        buf.push(b'\n');
        buf.extend_from_slice(&values.len().to_le_bytes());
        for v in values {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    File::create(path).and_then(|mut f| f.write_all(&buf)).unwrap_or_else(|e| panic!("can't write {path}: {e}"));
}

// For each endgame type, the material bucket (of `n`) it falls in most
// often, over the first million positions of the data.
fn endgame_material_modes(data: &str, n: usize) -> [usize; endgame::TYPES] {
    let total = std::fs::metadata(data).expect("can't read the data").len() / 32;
    let mut counts = vec![[0u64; 16]; endgame::TYPES];
    let mut reader = BufReader::with_capacity(1 << 22, File::open(data).expect("can't open the data"));
    let mut buf = [0u8; 32];
    for _ in 0..total.min(1_000_000) {
        reader.read_exact(&mut buf).expect("can't read the data");
        // SAFETY: the file is bullet's 32-byte plain-data boards.
        let board: ChessBoard = unsafe { std::ptr::read_unaligned(buf.as_ptr().cast()) };
        if let Some(t) = endgame::endgame_type(&board) {
            counts[t][kingstate::material_bucket(&board, n)] += 1;
        }
    }
    let mut modes = [0; endgame::TYPES];
    for (t, c) in counts.iter().enumerate() {
        modes[t] = (0..n).max_by_key(|&b| c[b]).unwrap();
    }
    modes
}

// The base bucket each of `new` output buckets starts from.
fn bucket_sources(plan: &Plan, base: usize, new: usize) -> Vec<usize> {
    if new == base {
        return (0..base).collect();
    }
    let split = if plan.output_kingstate { 3 } else { 1 };
    let mut src: Vec<usize> = (0..base * split).map(|b| b / split).collect();
    if plan.output_endgame {
        src.extend(endgame_material_modes(&plan.data, base));
    }
    assert_eq!(src.len(), new, "warm start: {new} output buckets can't be built from the base's {base}");
    src
}

// The base mapped into the target layout (weight name, rows, values): each
// weight's starting values and which of them a frozen-base run may train
// (1) or keeps as the base's (0). Trainable: the extras' rows, the king
// blocks' corrections (their values the base's, trained as deltas), new
// output buckets (endgame types; every king-state split), and with
// `train_output` the whole output layer.
fn map_base(
    base: &HashMap<String, Vec<f32>>,
    target: &[(String, usize, usize)],
    plan: &Plan,
    train_output: bool,
) -> Vec<(String, Vec<f32>, Vec<f32>)> {
    let size = |id: &str| target.iter().find(|t| t.0 == id).map(|t| t.2);
    let get = |id: &str| base.get(id).unwrap_or_else(|| panic!("warm start: the base has no {id}"));
    if size("l3b").is_some() {
        return map_layered(base, target, plan, train_output);
    }
    let (base_ob, new_ob) = (get("l1b").len(), size("l1b").expect("no l1b"));
    let src = bucket_sources(plan, base_ob, new_ob);
    let split = if plan.output_kingstate { 3 } else { 1 };
    let new_bucket = |b: usize| split > 1 || b >= base_ob;
    let output = |id: &str| id == "l1w" || id == "l1b";
    let mut out = Vec::new();
    for (id, rows, n) in target {
        let (values, mask) = match base.get(id) {
            Some(v) if v.len() == *n => {
                println!("warm start: {id} copied");
                (v.clone(), vec![if output(id) && train_output { 1.0 } else { 0.0 }; *n])
            }
            _ if id == "l0w" && size("l0f").is_some() => {
                // King blocks: each starts as the base's (flat) weights.
                let b = get("l0w");
                assert!(n % b.len() == 0, "warm start: l0w of {n} isn't blocks of the base's {}", b.len());
                println!("warm start: l0w, {} blocks of the base's", n / b.len());
                (b.repeat(n / b.len()), vec![1.0; *n])
            }
            _ if id == "l0w" => {
                // Extras layout: the factoriser block, the piece inputs, the extras.
                let (f, p) = (get("l0f"), get("l0w"));
                assert!(f.len() + p.len() <= *n, "warm start: l0w of {n} can't hold the base's");
                let mut v = vec![0.0; *n];
                v[..f.len()].copy_from_slice(f);
                v[f.len()..f.len() + p.len()].copy_from_slice(p);
                let mut m = vec![1.0; *n];
                m[..f.len() + p.len()].fill(0.0);
                println!("warm start: l0w, factoriser and pieces copied, {} extra inputs zero", (n - f.len() - p.len()) / rows);
                (v, m)
            }
            _ if output(id) => {
                // One value per (input, bucket), the bucket fastest.
                let b = get(id);
                let inputs = b.len() / base_ob;
                assert_eq!(inputs * new_ob, *n, "warm start: {id} doesn't match the base's inputs");
                println!("warm start: {id}, output buckets from the base's {src:?}");
                let v = (0..inputs).flat_map(|i| src.iter().map(move |&s| b[i * base_ob + s])).collect();
                let m = (0..inputs)
                    .flat_map(|_| (0..new_ob).map(|bk| if train_output || new_bucket(bk) { 1.0 } else { 0.0 }))
                    .collect();
                (v, m)
            }
            _ => panic!("warm start: no rule for {id} ({n} values)"),
        };
        out.push((id.clone(), values, mask));
    }
    out
}

// The same for the pairwise multi-layer networks (main.rs train_multi and
// train_multi_extras; the author, 2026-10-18: faster feature screens, and
// continued training on new data), from a layered base of the same hidden
// size and later layers:
//   - a weight of the same name and size: copied (the later layers frozen
//     unless `train_output`);
//   - the first layer of an extras network (king factoriser block, piece
//     inputs, extras, the predicates' factoriser last): the base's factoriser
//     and piece weights, every extra input zero and trainable;
//   - later layers with more output buckets (the endgame types): each new
//     bucket's rows from the base bucket it falls in most often, trainable.
// Every layer's rows are the output buckets times that layer's width, the
// bucket slowest; values one input's rows after another.
fn map_layered(
    base: &HashMap<String, Vec<f32>>,
    target: &[(String, usize, usize)],
    plan: &Plan,
    train_output: bool,
) -> Vec<(String, Vec<f32>, Vec<f32>)> {
    let size = |id: &str| target.iter().find(|t| t.0 == id).map(|t| t.2);
    let get = |id: &str| base.get(id).unwrap_or_else(|| panic!("warm start: the base has no {id} (a layered base?)"));
    let (base_ob, new_ob) = (get("l3b").len(), size("l3b").expect("no l3b"));
    let src = bucket_sources(plan, base_ob, new_ob);
    let later = |id: &str| matches!(id, "l1w" | "l1b" | "l2w" | "l2b" | "l3w" | "l3b");
    let mut out = Vec::new();
    for (id, rows, n) in target {
        let (values, mask) = match base.get(id) {
            Some(v) if v.len() == *n => {
                println!("warm start: {id} copied");
                (v.clone(), vec![if later(id) && train_output { 1.0 } else { 0.0 }; *n])
            }
            _ if id == "l0w" && size("l0f").is_none() => {
                // Extras layout: the king factoriser block, the piece inputs, the extras.
                let (f, p) = (get("l0f"), get("l0w"));
                assert!(f.len() + p.len() <= *n, "warm start: l0w of {n} can't hold the base's");
                let mut v = vec![0.0; *n];
                v[..f.len()].copy_from_slice(f);
                v[f.len()..f.len() + p.len()].copy_from_slice(p);
                let mut m = vec![1.0; *n];
                m[..f.len() + p.len()].fill(0.0);
                println!("warm start: l0w, factoriser and pieces copied, {} extra inputs zero", (n - f.len() - p.len()) / rows);
                (v, m)
            }
            _ if later(id) => {
                // Output buckets added: each new bucket's rows (width k) from its source bucket's.
                let b = get(id);
                let (k, inputs) = (rows / new_ob, n / rows);
                assert_eq!(b.len(), inputs * base_ob * k, "warm start: {id} doesn't match the base's layers");
                println!("warm start: {id}, output buckets from the base's {src:?}");
                let mut v = Vec::with_capacity(*n);
                let mut m = Vec::with_capacity(*n);
                for i in 0..inputs {
                    for (bk, &s) in src.iter().enumerate() {
                        for j in 0..k {
                            v.push(b[i * base_ob * k + s * k + j]);
                            m.push(if train_output || bk >= base_ob { 1.0 } else { 0.0 });
                        }
                    }
                }
                (v, m)
            }
            _ => panic!("warm start: no rule for {id} ({n} values) from a layered base"),
        };
        out.push((id.clone(), values, mask));
    }
    out
}

fn read_base(checkpoint: &str) -> HashMap<String, Vec<f32>> {
    read_labelled(&format!("{checkpoint}/optimiser_state/weights.bin"))
}

fn load_values(opt: &mut Optimiser<ExecutionContext, AdamWOptimiser>, values: &[(String, Vec<f32>)], path: &str) {
    if let Some(dir) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(dir).ok();
    }
    write_labelled(path, values);
    opt.load_weights_from_file(path).unwrap_or_else(|e| panic!("warm start: {e:?}"));
}

/// The first `n` values of weight `id` drawn again from a normal distribution
/// (mean 0, `stdev`; a fixed seed), the rest as they are: the factoriser block
/// of a layered net with extra inputs, whose affine bullet starts at a scale
/// set by all its inputs (tools/trainer/src/main.rs, train_multi_extras).
pub fn renormal_prefix(opt: &mut Optimiser<ExecutionContext, AdamWOptimiser>, id: &str, n: usize, stdev: f32, out_dir: &str) {
    let mut v = {
        let w = opt.cpu_weights().unwrap_or_else(|e| panic!("start: {e:?}"));
        w.get(id).values.f32().to_vec()
    };
    let mut s: u64 = 0x2545_F491_4F6C_DD1D;
    let mut uniform = move || {
        // splitmix64, then the top 53 bits as a number in (0, 1)
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64 + 1e-300
    };
    for x in v.iter_mut().take(n) {
        let (u1, u2) = (uniform(), uniform());
        *x = ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32 * stdev;
    }
    load_values(opt, &[(id.to_string(), v)], &format!("{out_dir}/start-{id}.bin"));
}

pub fn load(opt: &mut Optimiser<ExecutionContext, AdamWOptimiser>, checkpoint: &str, plan: &Plan, out_dir: &str) {
    let target: Vec<(String, usize, usize)> = {
        let w = opt.cpu_weights().unwrap_or_else(|e| panic!("warm start: {e:?}"));
        w.iter().map(|(id, v)| (id.clone(), v.shape.rows(), v.values.size())).collect()
    };
    let mapped = map_base(&read_base(checkpoint), &target, plan, false);
    let values: Vec<(String, Vec<f32>)> = mapped.into_iter().map(|(id, v, _)| (id, v)).collect();
    load_values(opt, &values, &format!("{out_dir}/warm-start.bin"));
    println!("warm start: from {checkpoint}");
}

// A frozen-base run (the author, 2026-10-08: train the base, add a feature
// at zero, train only the addition): every weight the network uses is
// C + M x W, C the base mapped into the layout (constant), M its trainable
// mask (constant), W the trained delta, starting at zero. The network
// starts as the base exactly; where M is 0 nothing can move it.
pub struct Frozen {
    pub weights: HashMap<String, (Vec<f32>, Vec<f32>)>,
}

impl Frozen {
    /// `target`: (weight name, rows, values) of the network as the trainer builds it.
    pub fn new(checkpoint: &str, target: &[(String, usize, usize)], plan: &Plan, train_output: bool) -> Self {
        let mapped = map_base(&read_base(checkpoint), target, plan, train_output);
        for (id, _, m) in &mapped {
            let free = m.iter().filter(|&&x| x > 0.0).count();
            println!("frozen base: {id}, {free} of {} values trainable", m.len());
        }
        Frozen { weights: mapped.into_iter().map(|(id, v, m)| (id, (v, m))).collect() }
    }

    /// The effective values of weight `id` given its delta.
    pub fn effective(&self, id: &str, delta: &[f32]) -> Vec<f32> {
        match self.weights.get(id) {
            Some((c, m)) => c.iter().zip(m).zip(delta).map(|((c, m), d)| c + m * d).collect(),
            None => delta.to_vec(),
        }
    }

    /// C and M into their weights (`<id>_c`, `<id>_m`), every delta at zero.
    pub fn zero_deltas(&self, opt: &mut Optimiser<ExecutionContext, AdamWOptimiser>, out_dir: &str) {
        let mut values = Vec::new();
        for (id, (c, m)) in &self.weights {
            values.push((id.clone(), vec![0.0; c.len()]));
            values.push((format!("{id}_c"), c.clone()));
            values.push((format!("{id}_m"), m.clone()));
        }
        load_values(opt, &values, &format!("{out_dir}/frozen-start.bin"));
    }

    /// The network's effective weights as a checkpoint another run can --init from.
    pub fn write_effective(&self, opt: &Optimiser<ExecutionContext, AdamWOptimiser>, dir: &str) {
        let w = opt.cpu_weights().unwrap_or_else(|e| panic!("frozen base: {e:?}"));
        let eff: Vec<(String, Vec<f32>)> = w
            .iter()
            .filter(|(id, _)| !id.ends_with("_c") && !id.ends_with("_m"))
            .map(|(id, v)| (id.clone(), self.effective(id, v.values.f32())))
            .collect();
        load_check_dir(dir);
        write_labelled(&format!("{dir}/optimiser_state/weights.bin"), &eff);
        println!("frozen base: effective weights written to {dir}");
    }
}

fn load_check_dir(dir: &str) {
    std::fs::create_dir_all(format!("{dir}/optimiser_state")).unwrap_or_else(|e| panic!("can't create {dir}: {e}"));
}
