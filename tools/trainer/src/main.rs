// FunComp's network trainer, on bullet (github.com/jw1912/bullet, MIT).
// Its graphs follow bullet's examples (examples/simple.rs, progression/):
// bullet's licence and copyright notice are in tools/trainer/LICENSE-bullet.
//
// Networks: (768 x KB -> HIDDEN) x 2 -> OB, the piece-square inputs seen
// from each side (KB king buckets of them, chosen by that side's king
// square), a squared clipped ReLU, OB output buckets by piece count.
// Targets blend the game result with the search score:
//   target = wdl * result + (1 - wdl) * sigmoid(score / scale)
// scores from the engine's data (`datatext`, converted to bullet's format
// with bullet-utils), from White's side in the text, side to move's in the
// binary format.
//
// Usage (every argument optional except --data):
//   funcomp-trainer --data <file> [--data <file> ...] --id <name>
//     [--hidden 128] [--scale 400] [--wdl 0.3] [--superbatches 40]
//     [--batches 6104] [--batch-size 16384] [--lr 0.001] [--lr-step 18] [--lr-gamma 0.1]
//     [--save-rate 10] [--threads 4] [--out checkpoints]
//     [--buckets flat|seven|seven-unmirrored|ten] [--output-buckets 1|8]
//     [--pawns 1] [--knight-pairs 1] [--slider-pairs 1] [--slider-pawn 1]
//     [--attacked 1] [--least-attacker 1] [--king-zone 1] (--threats is retired)
//     [--connected-pairs 1] [--pawn-pieces 1] [--pawn-spans 1] [--control-counts 1]
//     [--line-control 1] [--mobility 1] [--passers 1] (passed-pawn quality)
//     [--output-endgame 1] (three output buckets by endgame type after the others)
//     [--buckets kingstate] (the block by king state) [--buckets kinglocation]
//     (by the bare king state: location and castling rights) [--output-kingstate 1]
//     (each material output bucket split three ways by shelter)
//     [--extras-path 1] (the extras input type with no families: a matched control)
//     [--queue 64] [--print-features "<FEN>"]
//     [--convert <text>,<data>] (bullet-utils' text conversion, castling rights kept)
//     [--coverage N] (with --data and the architecture's flags: how often each
//     input is active over N positions, 0 all; coverage.rs) [--starved 10000]
//     [--layers L1,L2] (the pairwise multi-layer network)
//     [--eval-fens <file>] (after training, the net's centipawns for each FEN)
//     [--decay 0.01] (AdamW weight decay, times the learning rate)
//     [--decay-bias 0] [--decay-output 0] (no decay on the biases, on the
//     layers after the first: those chosen by output bucket; ROADMAP 5a)
//     [--init <checkpoint dir>] (a warm start from a trained base's checkpoint;
//     warm.rs) [--init-only 1] (write the warm-started net, quantised, and stop)
//     [--freeze-base 1] (with --init: the base held fixed, only the additions train:
//     feature rows, king-block corrections, new output buckets; warm.rs)
//     [--train-output 1] (with --freeze-base: the output layer trains too)
//     [--resume <checkpoint dir>] [--start-superbatch N] (a run continued from its checkpoint: the weights and
//     the optimiser's moments as saved; with N the checkpoint's superbatch + 1 the lr schedule goes on where it
//     was; the data stream starts again from its seed; ROADMAP, "Dead layer1 neurons")
//     [--stream 1] (the streaming shuffle, stream.rs: --data names unshuffled files or folders of them, each
//     optionally <path>@<weight>, read in random chunks through a shuffled buffer; no joined or shuffled copy)
//     [--shuffle-mb 2048] [--chunk 65536] [--seed 1] (its buffer, chunk size in positions and seed)
//     [--stream-check N] (N positions through the stream against a plain read, 0 one pass; no training)
//
// With the defaults (flat, one output bucket) the network is bullet's bare
// format, as the first networks were. Otherwise king buckets train with a
// factoriser (weights shared by all buckets, folded into each at save) and
// the output weights are saved one bucket after another; give the result
// FunComp's header with tools/nnue/pack.py (same layout names, defined
// there and here alike).
//
// Quantisation (for the engine's inference): first layer weights and
// biases x QA, output weights x QB, output bias x QA * QB.

mod coverage;
mod endgame;
mod kingstate;
mod pairs;
mod pawns;
mod squares;
mod stream;
mod warm;

use bullet_lib::{
    game::{
        formats::bulletformat::ChessBoard,
        inputs::{Chess768, ChessBuckets, ChessBucketsMirrored, SparseInputType, get_num_buckets},
        outputs::MaterialCount,
    },
    nn::{
        InitSettings, Shape,
        optimiser::{AdamW, AdamWParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader},
};

const QA: i16 = 255;
const QB: i16 = 64;

struct Args {
    data: Vec<String>,
    id: String,
    hidden: usize,
    scale: f32,
    wdl: f32,
    superbatches: usize,
    batches: usize,
    batch_size: usize,
    lr: f32,
    lr_step: usize,
    lr_gamma: f32,
    save_rate: usize,
    threads: usize,
    out: String,
    buckets: String,
    output_buckets: usize,
    pawns: bool,
    knight_pairs: bool,
    slider_pairs: bool,
    slider_pawn: bool,
    attacked: bool,
    least_attacker: bool,
    king_zone: bool,
    connected_pairs: bool,
    pawn_pieces: bool,
    pawn_spans: bool,
    control_counts: bool,
    line_control: bool,
    mobility: bool,
    output_kingstate: bool,
    output_endgame: bool,
    passers: bool,
    // --pawn-factoriser: the predicates' shared weights (pawns.rs); layered nets only.
    pawn_factoriser: bool,
    extras_path: bool,
    print_features: Vec<String>,
    convert: Option<(String, String)>,
    coverage: Option<u64>,
    starved: u64,
    queue: usize,
    decay: f32,
    decay_bias: bool,
    decay_output: bool,
    layers: Option<(usize, usize)>,
    init: Option<String>,
    init_only: bool,
    freeze_base: bool,
    train_output: bool,
    resume: Option<String>,
    start_superbatch: usize,
    // After training: the trained net's raw output (before the sigmoid; times
    // the training scale, centipawns) for each FEN of this file, one a line:
    // the engine-against-trainer check (tools/nnue/check-nets.sh).
    eval_fens: Option<String>,
    // The streaming shuffle (stream.rs).
    stream: bool,
    shuffle_mb: usize,
    chunk: usize,
    seed: u64,
    stream_check: Option<u64>,
}

impl Args {
    fn any_extras(&self) -> bool {
        self.extras_path
            || self.pawns
            || self.knight_pairs
            || self.slider_pairs
            || self.slider_pawn
            || self.attacked
            || self.least_attacker
            || self.king_zone
            || self.connected_pairs
            || self.pawn_pieces
            || self.pawn_spans
            || self.control_counts
            || self.line_control
            || self.mobility
            || self.passers
    }
}

// The piece inputs `base` with the extra feature sets `args` asks for.
fn extras<B>(args: &Args, base: B, mirrored: bool, factorised: bool) -> pawns::WithExtras<B> {
    pawns::WithExtras {
        base,
        mirrored,
        pawns: args.pawns,
        knights: args.knight_pairs,
        sliders: args.slider_pairs,
        slider_pawns: args.slider_pawn,
        attacked: args.attacked,
        least_attacker: args.least_attacker,
        king_zone: args.king_zone,
        connected: args.connected_pairs,
        pawn_pieces: args.pawn_pieces,
        pawn_spans: args.pawn_spans,
        control_counts: args.control_counts,
        line_control: args.line_control,
        mobility: args.mobility,
        passers: args.passers,
        factorised,
        pawn_factorised: factorised && args.pawn_factoriser,  // a training layout only, like the king factoriser
    }
}

fn parse_args() -> Args {
    let mut a = Args {
        data: Vec::new(),
        id: "funcomp".to_string(),
        hidden: 128,
        scale: 400.0,
        wdl: 0.3,
        superbatches: 40,
        batches: 6104,
        batch_size: 16_384,
        lr: 0.001,
        lr_step: 18,
        lr_gamma: 0.1,
        save_rate: 10,
        threads: 4,
        out: "checkpoints".to_string(),
        buckets: "flat".to_string(),
        output_buckets: 1,
        pawns: false,
        knight_pairs: false,
        slider_pairs: false,
        slider_pawn: false,
        attacked: false,
        least_attacker: false,
        king_zone: false,
        connected_pairs: false,
        pawn_pieces: false,
        pawn_spans: false,
        control_counts: false,
        line_control: false,
        mobility: false,
        output_kingstate: false,
        output_endgame: false,
        passers: false,
        pawn_factoriser: false,
        extras_path: false,
        print_features: Vec::new(),
        convert: None,
        coverage: None,
        starved: 10_000,
        queue: 64,
        decay: 0.01,
        decay_bias: true,
        decay_output: true,
        layers: None,
        init: None,
        init_only: false,
        freeze_base: false,
        train_output: false,
        resume: None,
        start_superbatch: 1,
        eval_fens: None,
        stream: false,
        shuffle_mb: 2048,
        chunk: 65_536,
        seed: 1,
        stream_check: None,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i + 1 < argv.len() {
        let v = argv[i + 1].clone();
        let num = |s: &str| -> f64 { s.parse().unwrap_or_else(|_| panic!("bad number: {s}")) };
        match argv[i].as_str() {
            "--data" => a.data.push(v),
            "--id" => a.id = v,
            "--hidden" => a.hidden = num(&v) as usize,
            "--scale" => a.scale = num(&v) as f32,
            "--wdl" => a.wdl = num(&v) as f32,
            "--superbatches" => a.superbatches = num(&v) as usize,
            "--batches" => a.batches = num(&v) as usize,
            "--batch-size" => a.batch_size = num(&v) as usize,
            "--lr" => a.lr = num(&v) as f32,
            "--lr-step" => a.lr_step = num(&v) as usize,
            "--lr-gamma" => a.lr_gamma = num(&v) as f32,
            "--save-rate" => a.save_rate = num(&v) as usize,
            "--threads" => a.threads = num(&v) as usize,
            "--out" => a.out = v,
            "--buckets" => a.buckets = v,
            "--output-buckets" => a.output_buckets = num(&v) as usize,
            "--pawns" => a.pawns = v != "0",
            "--knight-pairs" => a.knight_pairs = v != "0",
            "--slider-pairs" => a.slider_pairs = v != "0",
            "--slider-pawn" => a.slider_pawn = v != "0",
            "--threats" => panic!("--threats: the threat family is retired (the originality audit, R2)"),
            "--attacked" => a.attacked = v != "0",
            "--least-attacker" => a.least_attacker = v != "0",
            "--king-zone" => a.king_zone = v != "0",
            "--connected-pairs" => a.connected_pairs = v != "0",
            "--pawn-pieces" => a.pawn_pieces = v != "0",
            "--pawn-spans" => a.pawn_spans = v != "0",
            "--control-counts" => a.control_counts = v != "0",
            "--line-control" => a.line_control = v != "0",
            "--mobility" => a.mobility = v != "0",
            "--output-kingstate" => a.output_kingstate = v != "0",
            "--output-endgame" => a.output_endgame = v != "0",
            "--passers" => a.passers = v != "0",
            "--pawn-factoriser" => a.pawn_factoriser = v != "0",
            "--extras-path" => a.extras_path = v != "0",
            "--print-features" => a.print_features.push(v),
            "--convert" => {
                let (input, output) = v.split_once(',').expect("--convert <text>,<data>");
                a.convert = Some((input.to_string(), output.to_string()));
            }
            "--coverage" => a.coverage = Some(num(&v) as u64),
            "--starved" => a.starved = num(&v) as u64,
            "--queue" => a.queue = num(&v) as usize,
            "--decay" => a.decay = num(&v) as f32,
            "--decay-bias" => a.decay_bias = v != "0",
            "--decay-output" => a.decay_output = v != "0",
            "--init" => a.init = Some(v),
            "--init-only" => a.init_only = v != "0",
            "--freeze-base" => a.freeze_base = v != "0",
            "--train-output" => a.train_output = v != "0",
            "--resume" => a.resume = Some(v),
            "--start-superbatch" => a.start_superbatch = num(&v) as usize,
            "--eval-fens" => a.eval_fens = Some(v),
            "--stream" => a.stream = v != "0",
            "--shuffle-mb" => a.shuffle_mb = num(&v) as usize,
            "--chunk" => a.chunk = num(&v) as usize,
            "--seed" => a.seed = num(&v) as u64,
            "--stream-check" => a.stream_check = Some(num(&v) as u64),
            "--layers" => {
                let (x, y) = v.split_once(',').expect("--layers L1,L2");
                a.layers = Some((num(x) as usize, num(y) as usize));
            }
            other => panic!("unknown argument {other}"),
        }
        i += 2;
    }
    assert!(!a.data.is_empty() || !a.print_features.is_empty() || a.convert.is_some(), "--data <file> is required");
    a
}

// King-bucket layouts, by king square in its own side's view (a1 = 0): the
// mirrored ones for files a-d, as tools/nnue/pack.py has them.
#[rustfmt::skip]
const SEVEN: [usize; 32] = [
    0, 0, 1, 2,
    3, 3, 4, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
];
// TEN is bullet's example layout (examples/progression/3_input_buckets.rs),
// never trained here.
#[rustfmt::skip]
const TEN: [usize; 32] = [
    0, 1, 2, 3,
    4, 4, 5, 5,
    6, 6, 6, 6,
    7, 7, 7, 7,
    8, 8, 8, 8, 8, 8, 8, 8,
    9, 9, 9, 9, 9, 9, 9, 9,
];
#[rustfmt::skip]
const SEVEN_UNMIRRORED: [usize; 64] = [
    0, 0, 0, 1, 1, 2, 2, 2,
    3, 3, 3, 4, 4, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
];

// The schedule, settings and data, the same for every architecture; then
// train. A macro: the trainer's type differs with the inputs and buckets.
macro_rules! run {
    ($trainer:expr, $args:expr) => {{
        run!($trainer, $args, $args.queue, None::<std::sync::Arc<warm::Frozen>>)
    }};
    ($trainer:expr, $args:expr, $queue:expr) => {{
        run!($trainer, $args, $queue, None::<std::sync::Arc<warm::Frozen>>)
    }};
    ($trainer:expr, $args:expr, $queue:expr, $frozen:expr) => {{
        let args: &Args = $args;
        let frozen: Option<std::sync::Arc<warm::Frozen>> = $frozen;
        assert!(frozen.is_some() || !args.freeze_base, "--freeze-base: not for this architecture");
        if let Some(checkpoint) = &args.init {
            let dir = format!("{}/{}-init", args.out, args.id);
            match &frozen {
                // The base is in the constants: the deltas start at zero.
                Some(f) => f.zero_deltas(&mut $trainer.optimiser, &dir),
                None => warm::load(&mut $trainer.optimiser, checkpoint, &warm_plan(args), &dir),
            }
            if args.init_only {
                $trainer.save_quantised(&format!("{dir}/quantised.bin")).expect("can't save the warm start");
                println!("warm start written to {dir}/quantised.bin");
                return;
            }
        }
        if let Some(checkpoint) = &args.resume {
            assert!(args.init.is_none() && frozen.is_none(), "--resume: not with --init or --freeze-base");
            $trainer.load_from_checkpoint(checkpoint);
        }
        assert!(
            (1..=args.superbatches).contains(&args.start_superbatch),
            "--start-superbatch: from 1 to --superbatches"
        );
        let schedule = TrainingSchedule {
            net_id: args.id.clone(),
            eval_scale: args.scale,
            steps: TrainingSteps {
                batch_size: args.batch_size,
                batches_per_superbatch: args.batches,
                start_superbatch: args.start_superbatch,
                end_superbatch: args.superbatches,
            },
            wdl_scheduler: wdl::ConstantWDL { value: args.wdl },
            lr_scheduler: lr::StepLR { start: args.lr, gamma: args.lr_gamma, step: args.lr_step },
            save_rate: args.save_rate,
        };
        let settings = LocalSettings {
            threads: args.threads,
            test_set: None,
            output_directory: &args.out,
            batch_queue_size: $queue,
        };
        if args.stream {
            let data_loader = stream::StreamLoader::new(stream::sources(&args.data), args.chunk, args.shuffle_mb, args.seed);
            $trainer.run(&schedule, &settings, &data_loader);
        } else {
            // Files read front to back: shuffled ones (tools/nnue/mix-data.sh); folders and weights need --stream 1.
            for d in &args.data {
                assert!(std::path::Path::new(d).is_file(), "--data {d}: not a file (a folder or a weight needs --stream 1)");
            }
            let files: Vec<&str> = args.data.iter().map(String::as_str).collect();
            let data_loader = loader::DirectSequentialDataLoader::new(&files);
            $trainer.run(&schedule, &settings, &data_loader);
        }
        // A frozen-base run's checkpoints hold the deltas: the effective
        // weights, for a next stage's --init.
        if let Some(f) = &frozen {
            f.write_effective(&$trainer.optimiser, &format!("{}/{}-effective", args.out, args.id));
        }
        if let Some(path) = &args.eval_fens {
            let fens = std::fs::read_to_string(path).expect("can't read --eval-fens");
            for fen in fens.lines().map(str::trim).filter(|l| !l.is_empty()) {
                println!("eval {:.3} {fen}", $trainer.eval(fen) * args.scale as f32);
            }
        }
    }};
}

// The warm start's view of the architecture flags.
fn warm_plan(args: &Args) -> warm::Plan {
    warm::Plan {
        output_kingstate: args.output_kingstate,
        output_endgame: args.output_endgame,
        data: stream::first_file(&args.data[0]),
    }
}

// A frozen-base run's constants (--freeze-base, after --init): `target`, the
// network's weights as the trainer builds them (name, rows, values).
fn frozen_base(args: &Args, target: &[(&str, usize, usize)]) -> Option<std::sync::Arc<warm::Frozen>> {
    if !args.freeze_base {
        return None;
    }
    let checkpoint = args.init.as_ref().expect("--freeze-base needs --init <checkpoint>");
    let t: Vec<(String, usize, usize)> = target.iter().map(|(id, r, n)| (id.to_string(), *r, *n)).collect();
    Some(std::sync::Arc::new(warm::Frozen::new(checkpoint, &t, &warm_plan(args), args.train_output)))
}

// A weight as a frozen-base network uses it: C + M x W. C and M are weights
// of their own (`<id>_c`, `<id>_m`, loaded at the start), detached so no
// gradient reaches them and without decay (frozen_params), so they never
// move: bullet's GPU graphs take no constant tensors.
fn freeze_node<'a>(
    builder: &'a bullet_lib::nn::ModelBuilder,
    frozen: &Option<std::sync::Arc<warm::Frozen>>,
    id: &str,
    node: bullet_lib::nn::ModelNode<'a>,
    shape: Shape,
) -> bullet_lib::nn::ModelNode<'a> {
    match frozen {
        Some(_) => {
            let c = builder.new_weights(format!("{id}_c"), shape, InitSettings::Zeroed).detach();
            let m = builder.new_weights(format!("{id}_m"), shape, InitSettings::Zeroed).detach();
            c + m * node
        }
        None => node,
    }
}

// The frozen parts' optimiser settings: no decay, no clipping (their
// gradient is zero, so with no decay AdamW leaves them exactly as loaded).
macro_rules! frozen_params {
    ($trainer:expr, $frozen:expr) => {{
        if let Some(f) = &$frozen {
            for id in f.weights.keys() {
                for part in ["_c", "_m"] {
                    let p = AdamWParams { decay: 0.0, max_weight: 1e6, min_weight: -1e6, ..Default::default() };
                    $trainer.optimiser.set_params_for_weight(&format!("{id}{part}"), p);
                }
            }
        }
    }};
}

// A saved weight's effective values (a frozen-base run: C + M x W).
fn effective(frozen: &Option<std::sync::Arc<warm::Frozen>>, id: &str, w: &[f32]) -> Vec<f32> {
    match frozen {
        Some(f) => f.effective(id, w),
        None => w.to_vec(),
    }
}

// King and output buckets: `$inputs` with `$kb` buckets, `$ob` outputs.
macro_rules! train_bucketed {
    ($args:expr, $inputs:expr, $kb:expr, $ob:literal) => {{
        train_bucketed!($args, $inputs, $kb, MaterialCount::<$ob>, $ob)
    }};
    ($args:expr, $inputs:expr, $kb:expr, $outputs:expr, $nob:expr) => {{
        let args: &Args = $args;
        let hidden = args.hidden;
        let kb: usize = $kb;
        let nob: usize = $nob;
        let frozen = frozen_base(args, &[
            ("l0f", hidden, 768 * hidden),
            ("l0w", hidden, 768 * kb * hidden),
            ("l0b", hidden, hidden),
            ("l1w", nob, 2 * hidden * nob),
            ("l1b", nob, nob),
        ]);
        let (f0, f1, f2, f3) = (frozen.clone(), frozen.clone(), frozen.clone(), frozen.clone());
        let mut trainer = ValueTrainerBuilder::default()
            .dual_perspective()
            .optimiser(AdamW)
            .inputs($inputs)
            .output_buckets($outputs)
            .save_format(&[
                // The factoriser folded into every bucket's weights.
                SavedFormat::id("l0w")
                    .transform(move |store, weights| {
                        let factoriser = effective(&f0, "l0f", store.get("l0f").values.f32()).repeat(kb);
                        effective(&f0, "l0w", &weights).into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
                    })
                    .round()
                    .quantise::<i16>(QA),
                SavedFormat::id("l0b").transform(move |_, w| effective(&f1, "l0b", &w)).round().quantise::<i16>(QA),
                // One output bucket's 2 x hidden weights after another.
                SavedFormat::id("l1w").transform(move |_, w| effective(&f2, "l1w", &w)).round().quantise::<i16>(QB).transpose(),
                SavedFormat::id("l1b").transform(move |_, w| effective(&f3, "l1b", &w)).round().quantise::<i16>(QA * QB),
            ])
            .loss_fn(|output, target| output.sigmoid().squared_error(target))
            .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
                let l0f = builder.new_weights("l0f", Shape::new(hidden, 768), InitSettings::Zeroed);
                let l0f = freeze_node(builder, &frozen, "l0f", l0f, Shape::new(hidden, 768));
                let mut l0 = builder.new_affine("l0", 768 * kb, hidden);
                l0.weights = freeze_node(builder, &frozen, "l0w", l0.weights, Shape::new(hidden, 768 * kb));
                l0.bias = freeze_node(builder, &frozen, "l0b", l0.bias, Shape::new(hidden, 1));
                l0.weights = l0.weights + l0f.repeat(kb);
                let mut l1 = builder.new_affine("l1", 2 * hidden, nob);
                l1.weights = freeze_node(builder, &frozen, "l1w", l1.weights, Shape::new(nob, 2 * hidden));
                l1.bias = freeze_node(builder, &frozen, "l1b", l1.bias, Shape::new(nob, 1));
                let stm = l0.forward(stm_inputs).screlu();
                let ntm = l0.forward(ntm_inputs).screlu();
                l1.forward(stm.concat(ntm)).select(output_buckets)
            });
        // The factoriser and the bucket weights add up: clip each tighter,
        // so the sum stays within the usual range.
        set_adamw!(trainer, args, "l0w": true, false; "l0f": true, false; "l0b": false, false;
            "l1w": false, true; "l1b": false, true);
        frozen_params!(trainer, frozen);
        run!(trainer, args, args.queue, frozen.clone());
    }};
}

// AdamW's settings for one weight (ROADMAP 5a, weight decay): bullet's
// defaults, the tighter clip where a factoriser and the weights it is
// folded into add up (`clipped`), the decay rate from --decay, none on the
// biases with --decay-bias 0 or on the layers chosen by output bucket
// (`output`) with --decay-output 0. Every weight is given its settings, so
// the defaults reproduce bullet's own.
fn adamw(args: &Args, name: &str, clipped: bool, output: bool) -> AdamWParams {
    let mut p = AdamWParams { decay: args.decay, ..Default::default() };
    if clipped {
        p.max_weight = 0.99;
        p.min_weight = -0.99;
    }
    if (name.ends_with('b') && !args.decay_bias) || (output && !args.decay_output) {
        p.decay = 0.0;
    }
    p
}

// Sets every weight's AdamW settings: (name, clipped, output) for each.
macro_rules! set_adamw {
    ($trainer:expr, $args:expr, $($name:literal: $clipped:expr, $output:expr);+ $(;)?) => {{
        $( $trainer.optimiser.set_params_for_weight($name, adamw($args, $name, $clipped, $output)); )+
    }};
}

// The pairwise multi-layer network (FunComp's header version 3): king
// buckets `$inputs` (`$kb` of them, with a factoriser), the first layer's
// two halves clipped to [0, 1] and multiplied pairwise, then
// HIDDEN -> $l1 -> $l2 -> 1 with SCReLU, every layer by output bucket ($ob).
// The later layers are saved in float, one output bucket after another.
macro_rules! train_multi {
    ($args:expr, $inputs:expr, $kb:expr, $ob:literal, $l1:expr, $l2:expr) => {{
        train_multi!($args, $inputs, $kb, MaterialCount::<$ob>, $ob, $l1, $l2)
    }};
    // Other output buckets ($outputs, $nob of them): the endgame types after the material buckets.
    ($args:expr, $inputs:expr, $kb:expr, $outputs:expr, $nob:expr, $l1:expr, $l2:expr) => {{
        let args: &Args = $args;
        let nob: usize = $nob;
        let hidden = args.hidden;
        let kb: usize = $kb;
        let (l1, l2): (usize, usize) = ($l1, $l2);
        // A frozen-base run (--init <a layered base> --freeze-base; the author, 2026-10-18): every weight
        // C + M x W, the base mapped by warm.rs map_layered (new output buckets trainable; the later layers
        // too with --train-output).
        let frozen = frozen_base(args, &[
            ("l0f", hidden, 768 * hidden),
            ("l0w", hidden, 768 * kb * hidden),
            ("l0b", hidden, hidden),
            ("l1w", nob * l1, nob * l1 * hidden),
            ("l1b", nob * l1, nob * l1),
            ("l2w", nob * l2, nob * l2 * l1),
            ("l2b", nob * l2, nob * l2),
            ("l3w", nob, nob * l2),
            ("l3b", nob, nob),
        ]);
        let (f0, f1, f2, f3, f4) = (frozen.clone(), frozen.clone(), frozen.clone(), frozen.clone(), frozen.clone());
        let (f5, f6, f7, f8) = (frozen.clone(), frozen.clone(), frozen.clone(), frozen.clone());
        let mut trainer = ValueTrainerBuilder::default()
            .dual_perspective()
            .optimiser(AdamW)
            .inputs($inputs)
            .output_buckets($outputs)
            .save_format(&[
                SavedFormat::id("l0w")
                    .transform(move |store, weights| {
                        let factoriser = effective(&f0, "l0f", store.get("l0f").values.f32()).repeat(kb);
                        effective(&f1, "l0w", &weights).into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
                    })
                    .round()
                    .quantise::<i16>(QA),
                SavedFormat::id("l0b").transform(move |_, w| effective(&f2, "l0b", &w)).round().quantise::<i16>(QA),
                SavedFormat::id("l1w").transform(move |_, w| effective(&f3, "l1w", &w)).transpose(),
                SavedFormat::id("l1b").transform(move |_, w| effective(&f4, "l1b", &w)),
                SavedFormat::id("l2w").transform(move |_, w| effective(&f5, "l2w", &w)).transpose(),
                SavedFormat::id("l2b").transform(move |_, w| effective(&f6, "l2b", &w)),
                SavedFormat::id("l3w").transform(move |_, w| effective(&f7, "l3w", &w)).transpose(),
                SavedFormat::id("l3b").transform(move |_, w| effective(&f8, "l3b", &w)),
            ])
            .loss_fn(|output, target| output.sigmoid().squared_error(target))
            .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
                // With king buckets the factoriser starts where a flat net's first layer would (normal, sd
                // sqrt(2 / 768)): bullet scales an affine's start by all its inputs, 768 x kb, and the
                // pairwise product of two such small halves left the later layers nothing to learn from
                // (1024 x 7 buckets with layers 16,16 stayed at its first loss for 6,104 batches, where the
                // flat nets learned at every width; 2026-10-15). Flat stays as it was.
                let l0f_init = if kb > 1 {
                    InitSettings::Normal { mean: 0.0, stdev: (2.0f32 / 768.0).sqrt() }
                } else {
                    InitSettings::Zeroed
                };
                let l0f = builder.new_weights("l0f", Shape::new(hidden, 768), l0f_init);
                let l0f = freeze_node(builder, &frozen, "l0f", l0f, Shape::new(hidden, 768));
                let mut l0 = builder.new_affine("l0", 768 * kb, hidden);
                l0.weights = freeze_node(builder, &frozen, "l0w", l0.weights, Shape::new(hidden, 768 * kb));
                l0.bias = freeze_node(builder, &frozen, "l0b", l0.bias, Shape::new(hidden, 1));
                l0.weights = l0.weights + l0f.repeat(kb);
                let mut w1 = builder.new_affine("l1", hidden, nob * l1);
                w1.weights = freeze_node(builder, &frozen, "l1w", w1.weights, Shape::new(nob * l1, hidden));
                w1.bias = freeze_node(builder, &frozen, "l1b", w1.bias, Shape::new(nob * l1, 1));
                let mut w2 = builder.new_affine("l2", l1, nob * l2);
                w2.weights = freeze_node(builder, &frozen, "l2w", w2.weights, Shape::new(nob * l2, l1));
                w2.bias = freeze_node(builder, &frozen, "l2b", w2.bias, Shape::new(nob * l2, 1));
                let mut w3 = builder.new_affine("l3", l2, nob);
                w3.weights = freeze_node(builder, &frozen, "l3w", w3.weights, Shape::new(nob, l2));
                w3.bias = freeze_node(builder, &frozen, "l3b", w3.bias, Shape::new(nob, 1));
                // The two halves of each side's first layer, clipped, multiplied.
                let ft = |input, start, end| l0.slice(start, end).forward(input).crelu();
                let stm = ft(stm_inputs, 0, hidden / 2) * ft(stm_inputs, hidden / 2, hidden);
                let ntm = ft(ntm_inputs, 0, hidden / 2) * ft(ntm_inputs, hidden / 2, hidden);
                let h1 = w1.forward(stm.concat(ntm)).select(output_buckets).screlu();
                let h2 = w2.forward(h1).select(output_buckets).screlu();
                w3.forward(h2).select(output_buckets)
            });
        set_adamw!(trainer, args, "l0w": true, false; "l0f": true, false; "l0b": false, false;
            "l1w": false, true; "l1b": false, true; "l2w": false, true; "l2b": false, true;
            "l3w": false, true; "l3b": false, true);
        frozen_params!(trainer, frozen);
        run!(trainer, args, args.queue, frozen.clone());
    }};
}

// The pairwise multi-layer network with extra feature sets (the author,
// 2026-10-18: the bare king state with the pawn predicates carrying the
// shelter, on the leader's layers): the inputs as train_extras lays them out
// (a factoriser block of 768 first, shared by the piece inputs, folded into
// them at save), the later layers as train_multi's. The factoriser block
// starts as train_multi's factoriser does (normal, sd sqrt(2 / 768)), not at
// bullet's scale for all the affine's inputs, which left the pairwise layers
// nothing to learn from (2026-10-15).
macro_rules! train_multi_extras {
    ($args:expr, $base:expr, $mirrored:expr, $ob:literal, $l1:expr, $l2:expr) => {{
        let args: &Args = $args;
        let hidden = args.hidden;
        let base = $base;
        let base_inputs = base.num_inputs();
        let inputs = extras(args, base, $mirrored, true);
        let total = inputs.num_inputs();
        // The predicates' factoriser (pawns.rs), the last block: folded into their per-square weights.
        let pf = if inputs.pawns && inputs.pawn_factorised { pawns::PAWN_FACTORISER } else { 0 };
        let (l1, l2): (usize, usize) = ($l1, $l2);
        let nob: usize = $ob;
        // A frozen-base run from a layered base (warm.rs map_layered): the extras' columns trainable, the
        // base's first layer as it was; the later layers too with --train-output.
        let frozen = frozen_base(args, &[
            ("l0w", hidden, total * hidden),
            ("l0b", hidden, hidden),
            ("l1w", nob * l1, nob * l1 * hidden),
            ("l1b", nob * l1, nob * l1),
            ("l2w", nob * l2, nob * l2 * l1),
            ("l2b", nob * l2, nob * l2),
            ("l3w", nob, nob * l2),
            ("l3b", nob, nob),
        ]);
        let (f0, f2, f3, f4) = (frozen.clone(), frozen.clone(), frozen.clone(), frozen.clone());
        let (f5, f6, f7, f8) = (frozen.clone(), frozen.clone(), frozen.clone(), frozen.clone());
        let mut trainer = ValueTrainerBuilder::default()
            .dual_perspective()
            .optimiser(AdamW)
            .inputs(inputs)
            .output_buckets(MaterialCount::<$ob>)
            .save_format(&[
                SavedFormat::id("l0w")
                    .transform(move |_store, weights| {
                        let weights = effective(&f0, "l0w", &weights);
                        let h = weights.len() / total;
                        let shared = total - pf;
                        (0..(total - 768 - pf) * h)
                            .map(|e| {
                                let (j, i) = (e / h, e % h);
                                let own = weights[(768 + j) * h + i];
                                if j < base_inputs {
                                    own + weights[(j % 768) * h + i]
                                } else if pf > 0 && j < base_inputs + pawns::PAWN_FEATURES {
                                    // (predicate x 2 + colour) x 64 + square: its predicate and colour's shared weight
                                    own + weights[(shared + (j - base_inputs) / 64) * h + i]
                                } else {
                                    own
                                }
                            })
                            .collect()
                    })
                    .round()
                    .quantise::<i16>(QA),
                SavedFormat::id("l0b").transform(move |_, w| effective(&f2, "l0b", &w)).round().quantise::<i16>(QA),
                SavedFormat::id("l1w").transform(move |_, w| effective(&f3, "l1w", &w)).transpose(),
                SavedFormat::id("l1b").transform(move |_, w| effective(&f4, "l1b", &w)),
                SavedFormat::id("l2w").transform(move |_, w| effective(&f5, "l2w", &w)).transpose(),
                SavedFormat::id("l2b").transform(move |_, w| effective(&f6, "l2b", &w)),
                SavedFormat::id("l3w").transform(move |_, w| effective(&f7, "l3w", &w)).transpose(),
                SavedFormat::id("l3b").transform(move |_, w| effective(&f8, "l3b", &w)),
            ])
            .loss_fn(|output, target| output.sigmoid().squared_error(target))
            .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
                let mut l0 = builder.new_affine("l0", total, hidden);
                l0.weights = freeze_node(builder, &frozen, "l0w", l0.weights, Shape::new(hidden, total));
                l0.bias = freeze_node(builder, &frozen, "l0b", l0.bias, Shape::new(hidden, 1));
                let mut w1 = builder.new_affine("l1", hidden, nob * l1);
                w1.weights = freeze_node(builder, &frozen, "l1w", w1.weights, Shape::new(nob * l1, hidden));
                w1.bias = freeze_node(builder, &frozen, "l1b", w1.bias, Shape::new(nob * l1, 1));
                let mut w2 = builder.new_affine("l2", l1, nob * l2);
                w2.weights = freeze_node(builder, &frozen, "l2w", w2.weights, Shape::new(nob * l2, l1));
                w2.bias = freeze_node(builder, &frozen, "l2b", w2.bias, Shape::new(nob * l2, 1));
                let mut w3 = builder.new_affine("l3", l2, nob);
                w3.weights = freeze_node(builder, &frozen, "l3w", w3.weights, Shape::new(nob, l2));
                w3.bias = freeze_node(builder, &frozen, "l3b", w3.bias, Shape::new(nob, 1));
                let ft = |input, start, end| l0.slice(start, end).forward(input).crelu();
                let stm = ft(stm_inputs, 0, hidden / 2) * ft(stm_inputs, hidden / 2, hidden);
                let ntm = ft(ntm_inputs, 0, hidden / 2) * ft(ntm_inputs, hidden / 2, hidden);
                let h1 = w1.forward(stm.concat(ntm)).select(output_buckets).screlu();
                let h2 = w2.forward(h1).select(output_buckets).screlu();
                w3.forward(h2).select(output_buckets)
            });
        if args.init.is_none() {
            std::fs::create_dir_all(&args.out).ok();
            warm::renormal_prefix(&mut trainer.optimiser, "l0w", 768 * hidden, (2.0f32 / 768.0).sqrt(), &args.out);
        }
        // Factoriser and piece weights add up at save: clip each tighter.
        set_adamw!(trainer, args, "l0w": true, false; "l0b": false, false;
            "l1w": false, true; "l1b": false, true; "l2w": false, true; "l2b": false, true;
            "l3w": false, true; "l3b": false, true);
        frozen_params!(trainer, frozen);
        // Up to 304 active inputs a position: a long queue of batches fills the GPU.
        run!(trainer, args, args.queue.min(8), frozen.clone());
    }};
}

fn train_flat(args: &Args) {
    let hidden = args.hidden;
    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(Chess768)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs| {
            let l0 = builder.new_affine("l0", 768, hidden);
            let l1 = builder.new_affine("l1", 2 * hidden, 1);
            let stm = l0.forward(stm_inputs).screlu();
            let ntm = l0.forward(ntm_inputs).screlu();
            l1.forward(stm.concat(ntm))
        });
    set_adamw!(trainer, args, "l0w": false, false; "l0b": false, false; "l1w": false, true; "l1b": false, true);
    run!(trainer, args);
}

// Piece inputs `$base` (mirrored or not) with the extra feature sets after
// them (pawn predicates, knight pairs: pawns.rs, pairs.rs), `$ob` outputs.
// The factoriser is an input block of its own (768 inputs first, the rest
// shifted after them; pawns.rs), shared by the piece inputs only, and folded
// into them at save.
macro_rules! train_extras {
    ($args:expr, $base:expr, $mirrored:expr, $ob:literal) => {{
        train_extras!($args, $base, $mirrored, MaterialCount::<$ob>, $ob)
    }};
    ($args:expr, $base:expr, $mirrored:expr, $outputs:expr, $nob:expr) => {{
        let args: &Args = $args;
        let hidden = args.hidden;
        let base = $base;
        let base_inputs = base.num_inputs();
        let inputs = extras(args, base, $mirrored, true);
        let total = inputs.num_inputs();
        let nob: usize = $nob;
        let frozen = frozen_base(args, &[
            ("l0w", hidden, total * hidden),
            ("l0b", hidden, hidden),
            ("l1w", nob, 2 * hidden * nob),
            ("l1b", nob, nob),
        ]);
        let (f0, f1, f2, f3) = (frozen.clone(), frozen.clone(), frozen.clone(), frozen.clone());
        let mut trainer = ValueTrainerBuilder::default()
            .dual_perspective()
            .optimiser(AdamW)
            .inputs(inputs)
            .output_buckets($outputs)
            .save_format(&[
                SavedFormat::id("l0w")
                    .transform(move |_store, weights| {
                        let weights = effective(&f0, "l0w", &weights);
                        let h = weights.len() / total;
                        (0..(total - 768) * h)
                            .map(|e| {
                                let (j, i) = (e / h, e % h);
                                let own = weights[(768 + j) * h + i];
                                if j < base_inputs { own + weights[(j % 768) * h + i] } else { own }
                            })
                            .collect()
                    })
                    .round()
                    .quantise::<i16>(QA),
                SavedFormat::id("l0b").transform(move |_, w| effective(&f1, "l0b", &w)).round().quantise::<i16>(QA),
                SavedFormat::id("l1w").transform(move |_, w| effective(&f2, "l1w", &w)).round().quantise::<i16>(QB).transpose(),
                SavedFormat::id("l1b").transform(move |_, w| effective(&f3, "l1b", &w)).round().quantise::<i16>(QA * QB),
            ])
            .loss_fn(|output, target| output.sigmoid().squared_error(target))
            .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
                let mut l0 = builder.new_affine("l0", total, hidden);
                l0.weights = freeze_node(builder, &frozen, "l0w", l0.weights, Shape::new(hidden, total));
                l0.bias = freeze_node(builder, &frozen, "l0b", l0.bias, Shape::new(hidden, 1));
                let mut l1 = builder.new_affine("l1", 2 * hidden, nob);
                l1.weights = freeze_node(builder, &frozen, "l1w", l1.weights, Shape::new(nob, 2 * hidden));
                l1.bias = freeze_node(builder, &frozen, "l1b", l1.bias, Shape::new(nob, 1));
                let stm = l0.forward(stm_inputs).screlu();
                let ntm = l0.forward(ntm_inputs).screlu();
                l1.forward(stm.concat(ntm)).select(output_buckets)
            });
        // Factoriser and piece weights add up at save: clip each tighter.
        set_adamw!(trainer, args, "l0w": true, false; "l0b": false, false; "l1w": false, true; "l1b": false, true);
        frozen_params!(trainer, frozen);
        // Up to 304 active inputs a position: a long queue of batches fills
        // the GPU.
        run!(trainer, args, args.queue.min(8), frozen.clone());
    }};
}

// Debugging: a position's active inputs, both views (side to move's first),
// sorted, to compare with the engine's (UCI command nnueinputs).
fn print_inputs<I: SparseInputType<RequiredDataType = ChessBoard>>(inputs: &I, fen: &str) {
    let board = kingstate::parse(&format!("{fen} | 0 | 0.5")).expect("bad FEN");
    let (mut stm, mut ntm) = (Vec::new(), Vec::new());
    inputs.map_features(&board, |s, n| {
        stm.push(s);
        ntm.push(n);
    });
    stm.sort_unstable();
    ntm.sort_unstable();
    let join = |v: &[usize]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ");
    println!("stm {}\nntm {}", join(&stm), join(&ntm));
}

fn print_all<I: SparseInputType<RequiredDataType = ChessBoard> + Copy>(args: &Args, base: I, mirrored: bool) {
    for fen in &args.print_features {
        if args.any_extras() {
            print_inputs(&extras(args, base, mirrored, false), fen);
        } else {
            print_inputs(&base, fen);
        }
        // The output bucket, as the engine's nnueinputs prints it.
        let board = kingstate::parse(&format!("{fen} | 0 | 0.5")).expect("bad FEN");
        let n = args.output_buckets;
        let bucket = match (args.output_kingstate, args.output_endgame) {
            (false, false) => kingstate::material_bucket(&board, n),
            (true, false) => kingstate::material_bucket(&board, n) * 3 + kingstate::stm_shelter(&board),
            (ks, true) => {
                let others = if ks { n * 3 } else { n };
                match endgame::endgame_type(&board) {
                    Some(t) => others + t,
                    None if ks => kingstate::material_bucket(&board, n) * 3 + kingstate::stm_shelter(&board),
                    None => kingstate::material_bucket(&board, n),
                }
            }
        };
        println!("bucket {bucket}");
    }
}

// The coverage report (coverage.rs) for the architecture the flags describe,
// the same input types training would use (the plain flat net unmirrored).
fn coverage_run<B: SparseInputType<RequiredDataType = ChessBoard>>(args: &Args, base: B, mirrored: bool, blocks: usize) {
    let o = coverage::Options {
        path: stream::first_file(&args.data[0]),
        limit: args.coverage.unwrap_or(0),
        blocks,
        output_buckets: args.output_buckets,
        output_kingstate: args.output_kingstate,
        output_endgame: args.output_endgame,
        steps: (args.superbatches * args.batches * args.batch_size) as u64,
        starved: args.starved,
    };
    if args.any_extras() {
        coverage::report(&extras(args, base, mirrored, false), &o);
    } else {
        coverage::report(&base, &o);
    }
}

fn main() {
    let args = parse_args();
    if let Some((input, output)) = &args.convert {
        kingstate::convert(input, output);
        return;
    }
    if let Some(n) = args.stream_check {
        stream::check(&args.data, args.chunk, args.shuffle_mb, args.seed, n);
        return;
    }
    let flat_mirrored = [0usize; 32];
    if args.coverage.is_some() {
        match args.buckets.as_str() {
            "flat" if !args.any_extras() && args.output_buckets == 1 && !args.output_kingstate && !args.output_endgame => {
                coverage_run(&args, Chess768, false, 1)
            }
            "flat" => coverage_run(&args, ChessBucketsMirrored::new(flat_mirrored), true, 1),
            "seven" => coverage_run(&args, ChessBucketsMirrored::new(SEVEN), true, get_num_buckets(&SEVEN)),
            "ten" => coverage_run(&args, ChessBucketsMirrored::new(TEN), true, get_num_buckets(&TEN)),
            "seven-unmirrored" => {
                coverage_run(&args, ChessBuckets::new(SEVEN_UNMIRRORED), false, get_num_buckets(&SEVEN_UNMIRRORED))
            }
            "kingstate" | "kinglocation" | "kingstate12" | "kingstate8" => {
                let k = kingstate::KingStateBuckets::named(&args.buckets);
                coverage_run(&args, k, true, k.blocks())
            }
            b => panic!("unsupported: --buckets {b}"),
        }
        return;
    }
    if !args.print_features.is_empty() {
        match args.buckets.as_str() {
            "flat" => print_all(&args, ChessBucketsMirrored::new(flat_mirrored), true),
            "seven" => print_all(&args, ChessBucketsMirrored::new(SEVEN), true),
            "ten" => print_all(&args, ChessBucketsMirrored::new(TEN), true),
            "seven-unmirrored" => print_all(&args, ChessBuckets::new(SEVEN_UNMIRRORED), false),
            "kingstate" | "kinglocation" | "kingstate12" | "kingstate8" => {
                print_all(&args, kingstate::KingStateBuckets::named(&args.buckets), true)
            }
            b => panic!("unsupported: --buckets {b}"),
        }
        return;
    }
    assert!(!args.pawn_factoriser || (args.layers.is_some() && args.pawns), "--pawn-factoriser: with --pawns and --layers only");
    if let Some((l1, l2)) = args.layers {
        if args.output_endgame || args.output_kingstate {
            // The endgame types' output buckets under later layers (the author, 2026-10-18); seven king buckets only.
            assert!(args.output_endgame && !args.output_kingstate && !args.any_extras(), "with --layers: --output-endgame alone");
            match (args.buckets.as_str(), args.output_buckets) {
                ("seven", 8) => train_multi!(
                    &args,
                    ChessBucketsMirrored::new(SEVEN),
                    get_num_buckets(&SEVEN),
                    endgame::EndgameOutput::<8, false>::default(),
                    8 + 3,
                    l1,
                    l2
                ),
                (b, o) => panic!("unsupported with --layers --output-endgame: --buckets {b} --output-buckets {o}"),
            }
            return;
        }
        if args.any_extras() {
            assert!(!args.output_kingstate && !args.output_endgame, "not with --layers and extra inputs");
            match (args.buckets.as_str(), args.output_buckets) {
                ("flat", 8) => train_multi_extras!(&args, ChessBucketsMirrored::new(flat_mirrored), true, 8, l1, l2),
                ("seven", 8) => train_multi_extras!(&args, ChessBucketsMirrored::new(SEVEN), true, 8, l1, l2),
                ("kingstate" | "kinglocation" | "kingstate12" | "kingstate8", 8) => {
                    train_multi_extras!(&args, kingstate::KingStateBuckets::named(&args.buckets), true, 8, l1, l2)
                }
                (b, o) => panic!("unsupported with --layers and extra inputs: --buckets {b} --output-buckets {o}"),
            }
            return;
        }
        match (args.buckets.as_str(), args.output_buckets) {
            ("flat", 1) => train_multi!(&args, ChessBucketsMirrored::new(flat_mirrored), 1, 1, l1, l2),
            ("flat", 8) => train_multi!(&args, ChessBucketsMirrored::new(flat_mirrored), 1, 8, l1, l2),
            ("seven", 1) => train_multi!(&args, ChessBucketsMirrored::new(SEVEN), get_num_buckets(&SEVEN), 1, l1, l2),
            ("seven", 8) => train_multi!(&args, ChessBucketsMirrored::new(SEVEN), get_num_buckets(&SEVEN), 8, l1, l2),
            ("ten", 8) => train_multi!(&args, ChessBucketsMirrored::new(TEN), get_num_buckets(&TEN), 8, l1, l2),
            // King labels and the unmirrored buckets under later layers (the author, 2026-10-17: the first layer's
            // block is chosen by king state or square; the later layers don't care which; engine and packer
            // already read the pair; tools/nnue/check-nets.sh holds each to the engine's evaluation).
            ("seven-unmirrored", 8) => train_multi!(
                &args,
                ChessBuckets::new(SEVEN_UNMIRRORED),
                get_num_buckets(&SEVEN_UNMIRRORED),
                8,
                l1,
                l2
            ),
            ("kingstate" | "kinglocation" | "kingstate12" | "kingstate8", 8) => {
                let kst = kingstate::KingStateBuckets::named(&args.buckets);
                let blocks = kst.blocks();
                train_multi!(&args, kst, blocks, 8, l1, l2)
            }
            (b, o) => panic!("unsupported with --layers: --buckets {b} --output-buckets {o}"),
        }
        return;
    }
    let flat = ChessBucketsMirrored::new(flat_mirrored);
    let seven = ChessBucketsMirrored::new(SEVEN);
    let kst = kingstate::KingStateBuckets::named(&args.buckets);
    let (ks, eg) = (args.output_kingstate, args.output_endgame);
    if args.any_extras() && (ks || eg) {
        type Ks = kingstate::KingStateOutput<8>;
        type Eg = endgame::EndgameOutput<8, false>;
        type EgKs = endgame::EndgameOutput<8, true>;
        match (args.buckets.as_str(), args.output_buckets, ks, eg) {
            ("flat", 8, true, false) => train_extras!(&args, flat, true, Ks::default(), 24),
            ("seven", 8, true, false) => train_extras!(&args, seven, true, Ks::default(), 24),
            ("kingstate" | "kinglocation" | "kingstate12" | "kingstate8", 8, true, false) => train_extras!(&args, kst, true, Ks::default(), 24),
            ("flat", 8, false, true) => train_extras!(&args, flat, true, Eg::default(), 11),
            ("seven", 8, false, true) => train_extras!(&args, seven, true, Eg::default(), 11),
            ("kingstate" | "kinglocation" | "kingstate12" | "kingstate8", 8, false, true) => train_extras!(&args, kst, true, Eg::default(), 11),
            ("flat", 8, true, true) => train_extras!(&args, flat, true, EgKs::default(), 27),
            ("seven", 8, true, true) => train_extras!(&args, seven, true, EgKs::default(), 27),
            ("kingstate" | "kinglocation" | "kingstate12" | "kingstate8", 8, true, true) => train_extras!(&args, kst, true, EgKs::default(), 27),
            (b, o, k, e) => panic!("unsupported: --buckets {b} --output-buckets {o} --output-kingstate {k} --output-endgame {e}"),
        }
        return;
    }
    if args.any_extras() {
        match (args.buckets.as_str(), args.output_buckets) {
            ("kingstate" | "kinglocation" | "kingstate12" | "kingstate8", 8) => train_extras!(&args, kst, true, 8),
            ("flat", 1) => train_extras!(&args, ChessBucketsMirrored::new(flat_mirrored), true, 1),
            ("flat", 8) => train_extras!(&args, ChessBucketsMirrored::new(flat_mirrored), true, 8),
            ("seven", 1) => train_extras!(&args, ChessBucketsMirrored::new(SEVEN), true, 1),
            ("seven", 8) => train_extras!(&args, ChessBucketsMirrored::new(SEVEN), true, 8),
            ("ten", 1) => train_extras!(&args, ChessBucketsMirrored::new(TEN), true, 1),
            ("ten", 8) => train_extras!(&args, ChessBucketsMirrored::new(TEN), true, 8),
            ("seven-unmirrored", 1) => train_extras!(&args, ChessBuckets::new(SEVEN_UNMIRRORED), false, 1),
            ("seven-unmirrored", 8) => train_extras!(&args, ChessBuckets::new(SEVEN_UNMIRRORED), false, 8),
            (b, o) => panic!("unsupported: --buckets {b} --output-buckets {o}"),
        }
        return;
    }
    if args.buckets.starts_with("king") || ks || eg {
        type Ks = kingstate::KingStateOutput<8>;
        type Eg = endgame::EndgameOutput<8, false>;
        type EgKs = endgame::EndgameOutput<8, true>;
        let (ksn, sevenn) = (kst.blocks(), get_num_buckets(&SEVEN));
        match (args.buckets.as_str(), args.output_buckets, ks, eg) {
            ("kingstate" | "kinglocation" | "kingstate12" | "kingstate8", 8, false, false) => train_bucketed!(&args, kst, ksn, 8),
            ("kingstate" | "kinglocation" | "kingstate12" | "kingstate8", 8, true, false) => train_bucketed!(&args, kst, ksn, Ks::default(), 24),
            ("flat", 8, true, false) => train_bucketed!(&args, flat, 1, Ks::default(), 24),
            ("seven", 8, true, false) => train_bucketed!(&args, seven, sevenn, Ks::default(), 24),
            ("kingstate" | "kinglocation" | "kingstate12" | "kingstate8", 8, false, true) => train_bucketed!(&args, kst, ksn, Eg::default(), 11),
            ("flat", 8, false, true) => train_bucketed!(&args, flat, 1, Eg::default(), 11),
            ("seven", 8, false, true) => train_bucketed!(&args, seven, sevenn, Eg::default(), 11),
            ("kingstate" | "kinglocation" | "kingstate12" | "kingstate8", 8, true, true) => train_bucketed!(&args, kst, ksn, EgKs::default(), 27),
            ("flat", 8, true, true) => train_bucketed!(&args, flat, 1, EgKs::default(), 27),
            ("seven", 8, true, true) => train_bucketed!(&args, seven, sevenn, EgKs::default(), 27),
            (b, o, k, e) => panic!("unsupported: --buckets {b} --output-buckets {o} --output-kingstate {k} --output-endgame {e}"),
        }
        return;
    }
    match (args.buckets.as_str(), args.output_buckets) {
        ("flat", 1) => train_flat(&args),
        ("flat", 8) => train_bucketed!(&args, ChessBucketsMirrored::new(flat_mirrored), 1, 8),
        ("seven", 1) => train_bucketed!(&args, ChessBucketsMirrored::new(SEVEN), get_num_buckets(&SEVEN), 1),
        ("seven", 8) => train_bucketed!(&args, ChessBucketsMirrored::new(SEVEN), get_num_buckets(&SEVEN), 8),
        ("ten", 1) => train_bucketed!(&args, ChessBucketsMirrored::new(TEN), get_num_buckets(&TEN), 1),
        ("ten", 8) => train_bucketed!(&args, ChessBucketsMirrored::new(TEN), get_num_buckets(&TEN), 8),
        ("seven-unmirrored", 1) => train_bucketed!(
            &args,
            ChessBuckets::new(SEVEN_UNMIRRORED),
            get_num_buckets(&SEVEN_UNMIRRORED),
            1
        ),
        ("seven-unmirrored", 8) => train_bucketed!(
            &args,
            ChessBuckets::new(SEVEN_UNMIRRORED),
            get_num_buckets(&SEVEN_UNMIRRORED),
            8
        ),
        (b, o) => panic!("unsupported: --buckets {b} --output-buckets {o}"),
    }
}
