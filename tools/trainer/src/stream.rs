// The streaming shuffle (ROADMAP, "After 3.0: the streaming shuffle"; the author, 2026-10-19 and 10-20): the trainer
// reads the exported files where they lie, a round's or a worker's each and unshuffled, instead of one joined and
// shuffled copy (round 4's mix would have needed about 200 GB at its peak).
//
// Each pass over the data visits every chunk (`--chunk` positions, consecutive in one file) once, in a fresh random
// order; a source's weight (`--data <file or folder>@<weight>`) repeats its chunks: 2 twice a pass, 0.5 each chunk
// in half the passes, chosen at random. The chunks fill a buffer (`--shuffle-mb`) that is shuffled whole before the
// trainer sees it, while a second buffer fills on another thread. A batch then draws from about buffer / chunk
// chunks spread over all the data, so the positions of one game rarely meet in a batch. A pass ends with its own
// (shorter) buffer, so passes never mix. Seeded (`--seed`): a run repeats exactly.
//
// `--stream-check N` (no GPU, no training): N positions through the loader against a plain sequential read of the
// same files: the count, an order-free checksum of the records (equal over whole passes when every weight is a whole
// number: the reference read repeats such a source as often), the
// lag-1 correlation of the piece count (high in game order, about 0 when shuffled) and the reading speed.

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    sync::{Arc, mpsc},
    thread,
    time::Instant,
};

use bullet_lib::game::formats::bulletformat::ChessBoard;
use bullet_trainer::reader::DataReader;

const RECORD: usize = std::mem::size_of::<ChessBoard>();
const _: () = assert!(RECORD == 32);

#[derive(Clone, Debug)]
pub struct Source {
    pub path: String,
    pub positions: u64,
    pub weight: f64,
}

// `<file or folder>[@weight]`: a folder stands for its `*.data` files, in name order, each with the weight.
pub fn sources(args: &[String]) -> Vec<Source> {
    let mut out = Vec::new();
    for arg in args {
        let (path, weight) = match arg.rsplit_once('@') {
            Some((p, w)) if w.parse::<f64>().is_ok() => (p, w.parse::<f64>().unwrap()),
            _ => (arg.as_str(), 1.0),
        };
        assert!(weight >= 0.0, "{arg}: a negative weight");
        for file in files_of(path) {
            let bytes = std::fs::metadata(&file).unwrap_or_else(|_| panic!("can't read {file}")).len();
            assert!(bytes % RECORD as u64 == 0, "{file}: {bytes} bytes, not whole {RECORD}-byte positions");
            if bytes > 0 {
                out.push(Source { path: file, positions: bytes / RECORD as u64, weight });
            }
        }
    }
    assert!(!out.is_empty(), "--data: no positions");
    out
}

// The file itself, or a folder's `*.data` files in name order.
pub fn files_of(path: &str) -> Vec<String> {
    let meta = std::fs::metadata(path).unwrap_or_else(|_| panic!("--data {path}: not found"));
    if !meta.is_dir() {
        return vec![path.to_string()];
    }
    let mut files: Vec<String> = std::fs::read_dir(path)
        .unwrap_or_else(|_| panic!("can't list {path}"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "data"))
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    files.sort();
    assert!(!files.is_empty(), "--data {path}: no .data files in the folder");
    files
}

// The first file of a `--data` argument (a folder's first, the weight dropped): for the modes that read one file.
pub fn first_file(arg: &str) -> String {
    let path = match arg.rsplit_once('@') {
        Some((p, w)) if w.parse::<f64>().is_ok() => p,
        _ => arg,
    };
    files_of(path).swap_remove(0)
}

// splitmix64: the trainer has no random crate, and the stream only needs a seeded, well-mixed sequence.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    // Uniform in 0..n (the multiply-shift map; its bias is below 2^-32 for any n the stream uses).
    fn below(&mut self, n: usize) -> usize {
        ((u128::from(self.next()) * n as u128) >> 64) as usize
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn shuffle<T>(v: &mut [T], rng: &mut Rng) {
    for i in (1..v.len()).rev() {
        v.swap(i, rng.below(i + 1));
    }
}

#[derive(Clone)]
pub struct StreamLoader {
    sources: Arc<Vec<Source>>,
    chunk: usize,
    capacity: usize,
    seed: u64,
}

impl StreamLoader {
    pub fn new(sources: Vec<Source>, chunk: usize, shuffle_mb: usize, seed: u64) -> Self {
        assert!(chunk > 0, "--chunk: at least one position");
        // Whole chunks, at least one.
        let capacity = ((shuffle_mb << 20) / RECORD / chunk).max(1) * chunk;
        let loader = Self { sources: Arc::new(sources), chunk, capacity, seed };
        let total: u64 = loader.sources.iter().map(|s| s.positions).sum();
        let per_pass: f64 = loader.sources.iter().map(|s| s.positions as f64 * s.weight).sum();
        println!(
            "stream: {} files, {total} positions ({per_pass:.0} a pass with the weights); chunks of {chunk}, buffers of \
             {capacity} positions ({} MB, two of them), seed {seed}",
            loader.sources.len(),
            capacity * RECORD >> 20
        );
        for s in loader.sources.iter().filter(|s| s.weight != 1.0) {
            println!("stream: {} weight {}", s.path, s.weight);
        }
        loader
    }

    // One pass: every chunk of every source, as often as its weight says, in random order.
    fn pass_order(&self, rng: &mut Rng) -> Vec<(u32, u32)> {
        let mut order = Vec::new();
        for (i, s) in self.sources.iter().enumerate() {
            let chunks = s.positions.div_ceil(self.chunk as u64);
            for c in 0..chunks {
                let copies = s.weight.floor() as usize + usize::from(rng.unit() < s.weight.fract());
                for _ in 0..copies {
                    order.push((i as u32, c as u32));
                }
            }
        }
        shuffle(&mut order, rng);
        order
    }

    // The producer: fills a buffer from the pass order, shuffles it, hands it over; ends when the consumer is gone.
    fn produce(&self, empty: mpsc::Receiver<Vec<ChessBoard>>, full: mpsc::SyncSender<(Vec<ChessBoard>, usize)>) {
        let mut rng = Rng(self.seed);
        let mut files: Vec<File> =
            self.sources.iter().map(|s| File::open(&s.path).unwrap_or_else(|_| panic!("can't open {}", s.path))).collect();
        let mut order: Vec<(u32, u32)> = Vec::new();
        let mut next = 0;
        let mut pass = 0u64;
        while let Ok(mut buf) = empty.recv() {
            let mut len = 0;
            loop {
                if next == order.len() {
                    if len > 0 {
                        break; // the pass ends with this buffer
                    }
                    order = self.pass_order(&mut rng);
                    assert!(!order.is_empty(), "stream: a pass with no chunks (every weight 0?)");
                    next = 0;
                    pass += 1;
                    println!("stream: pass {pass}, {} chunks", order.len());
                }
                let (s, c) = order[next];
                let source = &self.sources[s as usize];
                let start = u64::from(c) * self.chunk as u64;
                let n = (source.positions - start).min(self.chunk as u64) as usize;
                if len + n > buf.len() {
                    break; // the chunk opens the next buffer
                }
                next += 1;
                let file = &mut files[s as usize];
                file.seek(SeekFrom::Start(start * RECORD as u64)).expect("stream: seek");
                // SAFETY: ChessBoard is repr(C) plain data, valid for any bytes (bullet loads it the same way).
                let bytes =
                    unsafe { std::slice::from_raw_parts_mut(buf[len..len + n].as_mut_ptr().cast::<u8>(), n * RECORD) };
                file.read_exact(bytes).unwrap_or_else(|e| panic!("stream: reading {}: {e}", source.path));
                len += n;
            }
            shuffle(&mut buf[..len], &mut rng);
            if full.send((buf, len)).is_err() {
                return;
            }
        }
    }
}

impl DataReader<ChessBoard> for StreamLoader {
    fn read_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, skip: usize, mut f: F) {
        let (empty_tx, empty_rx) = mpsc::channel::<Vec<ChessBoard>>();
        let (full_tx, full_rx) = mpsc::sync_channel::<(Vec<ChessBoard>, usize)>(1);
        for _ in 0..2 {
            empty_tx.send(vec![ChessBoard::default(); self.capacity]).unwrap();
        }
        let me = self.clone();
        let producer = thread::spawn(move || me.produce(empty_rx, full_tx));
        let mut to_skip = skip;
        while let Ok((buf, len)) = full_rx.recv() {
            let skipped = to_skip.min(len);
            to_skip -= skipped;
            if skipped < len && f(&buf[skipped..len]) {
                break;
            }
            if empty_tx.send(buf).is_err() {
                break;
            }
        }
        drop(full_rx);
        drop(empty_tx);
        if producer.join().is_err() {
            panic!("stream: the reading thread failed");
        }
    }
}

// --stream-check: see the top of the file.
fn record_hash(b: &ChessBoard) -> u64 {
    // SAFETY: 32 bytes of plain data.
    let words: [u64; 4] = unsafe { std::mem::transmute_copy(b) };
    let mut h = 0u64;
    for w in words {
        let mut r = Rng(h ^ w);
        h = r.next();
    }
    h
}

#[derive(Default)]
struct Tally {
    count: u64,
    checksum: u64,
    // the lag-1 piece-count pairs within each slice
    n: f64,
    sx: f64,
    sy: f64,
    sxx: f64,
    syy: f64,
    sxy: f64,
}

impl Tally {
    fn add(&mut self, data: &[ChessBoard]) {
        for b in data {
            self.count += 1;
            self.checksum = self.checksum.wrapping_add(record_hash(b));
        }
        for w in data.windows(2) {
            let (x, y) = (f64::from(w[0].occ.count_ones()), f64::from(w[1].occ.count_ones()));
            self.n += 1.0;
            self.sx += x;
            self.sy += y;
            self.sxx += x * x;
            self.syy += y * y;
            self.sxy += x * y;
        }
    }

    fn correlation(&self) -> f64 {
        let cov = self.sxy / self.n - (self.sx / self.n) * (self.sy / self.n);
        let vx = self.sxx / self.n - (self.sx / self.n).powi(2);
        let vy = self.syy / self.n - (self.sy / self.n).powi(2);
        cov / (vx * vy).sqrt()
    }
}

pub fn check(data: &[String], chunk: usize, shuffle_mb: usize, seed: u64, positions: u64) {
    let srcs = sources(data);
    // Whole-number weights repeat a source in the reference read too, so a weighted pass checks exactly.
    let whole = srcs.iter().all(|s| s.weight.fract() == 0.0);
    let copies = |s: &Source| if whole { s.weight as u64 } else { 1 };
    let total: u64 = srcs.iter().map(|s| s.positions * copies(s)).sum();
    let limit = if positions == 0 { total } else { positions };

    // The files as they lie, front to back: the reference.
    let t = Instant::now();
    let mut plain = Tally::default();
    let mut left = limit;
    let mut buf = vec![ChessBoard::default(); 1 << 20];
    'files: loop {
        for s in srcs.iter().flat_map(|s| std::iter::repeat_n(s, copies(s) as usize)) {
            let mut file = File::open(&s.path).expect("can't open");
            let mut remaining = s.positions;
            while remaining > 0 {
                let n = remaining.min(buf.len() as u64).min(left) as usize;
                // SAFETY: as in produce().
                let bytes = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<u8>(), n * RECORD) };
                file.read_exact(bytes).expect("can't read");
                plain.add(&buf[..n]);
                remaining -= n as u64;
                left -= n as u64;
                if left == 0 {
                    break 'files;
                }
            }
        }
    }
    let plain_secs = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let loader = StreamLoader::new(srcs, chunk, shuffle_mb, seed);
    let mut stream = Tally::default();
    loader.read_chunks(0, |data| {
        let n = data.len().min((limit - stream.count) as usize);
        stream.add(&data[..n]);
        stream.count >= limit
    });
    let stream_secs = t.elapsed().as_secs_f64();

    for (name, tally, secs) in [("plain read", &plain, plain_secs), ("stream", &stream, stream_secs)] {
        println!(
            "{name:10}: {} positions, checksum {:016x}, lag-1 piece-count correlation {:+.4}, {:.1} s ({:.0} k positions/s)",
            tally.count,
            tally.checksum,
            tally.correlation(),
            secs,
            tally.count as f64 / secs / 1e3
        );
    }
    if whole && limit % total == 0 {
        println!(
            "checksums {} ({} whole passes: they must match)",
            if plain.checksum == stream.checksum { "match" } else { "DIFFER" },
            limit / total
        );
    }
}
