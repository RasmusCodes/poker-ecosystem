//! Benchmark suites. Each writes one JSON document; `research/` turns them into
//! figures and reports. Timings are CPU seconds of the measuring thread or the
//! whole process (the development machine is shared, so wall time is noisy);
//! convergence is measured in iterations and nodes touched, which do not depend
//! on load.

use pe_cards::sys::{Stopwatch, peak_working_set_bytes};
use pe_cards::{Card, Game, Rng, cards_of, mask_of, parse_cards};
use pe_cfr::{Algorithm, Sampling, Tree};
use pe_solver::{BetMenu, Buckets, HuSolver, MultiwaySolver, Profile, Spot, SpotConfig, build_model, solve_hu};
use serde_json::{Value, json};

use crate::args::Args;

pub fn run(args: &Args) -> Result<Value, String> {
    let suite = args.positional.first().map(|s| s.as_str()).unwrap_or("all");
    let quick = args.flag("quick");
    let started = Stopwatch::start();
    let mut out = match suite {
        "eval" => eval(quick),
        "equity" => equity(quick),
        "cfr" => cfr(quick),
        "players" => players(quick),
        "river" => river(quick),
        "turn" => turn(quick),
        "abstraction" => abstraction(quick),
        "multiway" => multiway(quick),
        "exploit" => exploit(quick),
        "analytic" => analytic(quick),
        "ranges" => ranges(quick),
        "review" => review(quick),
        "all" => {
            let mut all = serde_json::Map::new();
            for s in ["eval", "equity", "cfr", "players", "river", "turn", "abstraction", "multiway", "exploit", "analytic"] {
                eprintln!("suite {s}...");
                let v = match s {
                    "eval" => eval(quick),
                    "equity" => equity(quick),
                    "cfr" => cfr(quick),
                    "players" => players(quick),
                    "river" => river(quick),
                    "turn" => turn(quick),
                    "abstraction" => abstraction(quick),
                    "multiway" => multiway(quick),
                    "exploit" => exploit(quick),
                    _ => analytic(quick),
                }?;
                all.insert(s.to_string(), v);
            }
            Ok(Value::Object(all))
        }
        other => Err(format!("unknown suite '{other}'")),
    }?;
    if let Value::Object(m) = &mut out {
        m.insert(
            "meta".into(),
            json!({
                "suite": suite,
                "quick": quick,
                "threads": rayon::current_num_threads(),
                "process_cpu_seconds": started.process_cpu(),
                "wall_seconds": started.wall(),
                "peak_working_set_bytes": peak_working_set_bytes(),
                "note": "CPU seconds on a shared laptop (AMD Ryzen AI 7 PRO 350, 8 cores / 16 threads) running other jobs; compare iterations and nodes touched for convergence",
            }),
        );
    }
    Ok(out)
}

// ---------------------------------------------------------------- evaluators

fn eval(quick: bool) -> Result<Value, String> {
    let n = if quick { 200_000 } else { 3_000_000 };
    let mut rng = Rng::new(1);
    let deck: Vec<Card> = (0..52).map(Card).collect();
    let short: Vec<Card> = cards_of(pe_cards::SHORT_DECK);
    let draw = |rng: &mut Rng, d: &[Card], k: usize| -> Vec<Card> {
        let mut v = d.to_vec();
        for i in 0..k {
            let j = i + rng.below((v.len() - i) as u64) as usize;
            v.swap(i, j);
        }
        v.truncate(k);
        v
    };
    let mut rows = Vec::new();
    // 7-card hold'em
    let masks: Vec<u64> = (0..n).map(|_| mask_of(&draw(&mut rng, &deck, 7))).collect();
    let sw = Stopwatch::start();
    let mut acc = 0u64;
    for &m in &masks {
        acc = acc.wrapping_add(pe_eval::eval7(m) as u64);
    }
    let cpu = sw.thread_cpu();
    rows.push(json!({ "evaluator": "hold'em 7-card", "evals": n, "cpu_seconds": cpu, "ns_per_eval": 1e9 * cpu / n as f64, "checksum": acc }));
    // short deck
    let masks: Vec<u64> = (0..n).map(|_| mask_of(&draw(&mut rng, &short, 7))).collect();
    let sw = Stopwatch::start();
    let mut acc = 0u64;
    for &m in &masks {
        acc = acc.wrapping_add(pe_eval::eval_high(m, pe_cards::HighRules::ShortDeckTriton) as u64);
    }
    let cpu = sw.thread_cpu();
    rows.push(json!({ "evaluator": "short deck 7-card", "evals": n, "cpu_seconds": cpu, "ns_per_eval": 1e9 * cpu / n as f64, "checksum": acc }));
    // Omaha high (4 + 5) and hi-lo low
    let m = n / 10;
    let deals: Vec<(Vec<Card>, Vec<Card>)> = (0..m)
        .map(|_| {
            let d = draw(&mut rng, &deck, 9);
            (d[..4].to_vec(), d[4..].to_vec())
        })
        .collect();
    let sw = Stopwatch::start();
    let mut acc = 0u64;
    for (h, b) in &deals {
        acc = acc.wrapping_add(pe_eval::eval_omaha_high(h, b, pe_cards::HighRules::Standard) as u64);
    }
    let cpu = sw.thread_cpu();
    rows.push(json!({ "evaluator": "Omaha high (2+3 of 4+5)", "evals": m, "cpu_seconds": cpu, "ns_per_eval": 1e9 * cpu / m as f64, "checksum": acc }));
    let sw = Stopwatch::start();
    let mut acc = 0u64;
    for (h, b) in &deals {
        acc = acc.wrapping_add(pe_eval::eval_omaha_low8(h, b) as u64);
    }
    let cpu = sw.thread_cpu();
    rows.push(json!({ "evaluator": "Omaha 8-or-better low", "evals": m, "cpu_seconds": cpu, "ns_per_eval": 1e9 * cpu / m as f64, "checksum": acc }));
    Ok(json!({ "rows": rows }))
}

// ---------------------------------------------------------------- equity

fn equity(quick: bool) -> Result<Value, String> {
    let h = |s: &str| parse_cards(s).unwrap();
    let mut exact = Vec::new();
    let cases: Vec<(&str, Game, Vec<Vec<Card>>, Vec<Vec<Card>>)> = vec![
        ("NLH preflop AsAh vs KsKh", Game::NLH, vec![h("AsAh"), h("KsKh")], vec![vec![]]),
        ("NLH preflop AhKh vs QsQd", Game::NLH, vec![h("AhKh"), h("QsQd")], vec![vec![]]),
        ("NLH flop 3-way", Game::NLH, vec![h("AhKh"), h("QsQd"), h("JcTc")], vec![h("Th7h2c")]),
        ("PLO flop", Game::PLO, vec![h("AhAdKhQd"), h("JsTs9c8c")], vec![h("7s6h2d")]),
        ("PLO preflop", Game::PLO, vec![h("AsAdKsKd"), h("JhTh9c8c")], vec![vec![]]),
        ("Short deck preflop AKs vs JJ", Game::SHORT_DECK, vec![h("AhKh"), h("JsJd")], vec![vec![]]),
        ("Double board flop", Game::NLH, vec![h("AhKh"), h("QsQd")], vec![h("Th7h2c"), h("9s8s3d")]),
        ("Omaha hi-lo flop", Game::O8, vec![h("As2sKdKc"), h("QhQdJhJd")], vec![h("3c4d9h")]),
    ];
    for (name, game, hands, boards) in &cases {
        if quick && name.contains("PLO preflop") {
            continue;
        }
        let sw = Stopwatch::start();
        let r = pe_equity::equity_exact(game, hands, boards, &[])?;
        exact.push(json!({ "case": name, "equity": r.equity, "runouts": r.runouts, "cpu_seconds": sw.process_cpu(), "wall_seconds": sw.wall() }));
    }
    // Monte Carlo accuracy and interval coverage against an exact answer
    let game = Game::NLH;
    let hands = vec![h("AhKh"), h("QsQd")];
    let truth = pe_equity::equity_exact(&game, &hands, &[vec![]], &[])?.equity[0];
    let mut mc = Vec::new();
    let reps = if quick { 20 } else { 100 };
    for &n in &[1_000u64, 10_000, 100_000, 1_000_000] {
        if quick && n > 100_000 {
            continue;
        }
        let mut abs = 0.0;
        let mut sq = 0.0;
        let mut covered = 0;
        let r_here = if n >= 1_000_000 { reps / 5 } else { reps };
        let sw = Stopwatch::start();
        for seed in 0..r_here {
            let r = pe_equity::equity_monte_carlo(&game, &hands, &[vec![]], &[], n, 1000 + seed as u64)?;
            let err = r.equity[0] - truth;
            abs += err.abs();
            sq += err * err;
            if err.abs() <= 1.96 * r.std_error[0] {
                covered += 1;
            }
        }
        let k = r_here as f64;
        mc.push(json!({
            "samples": n, "repetitions": r_here, "mean_abs_error": abs / k, "rmse": (sq / k).sqrt(),
            "ci95_coverage": covered as f64 / k, "cpu_seconds_per_run": sw.process_cpu() / k,
        }));
    }
    // run it twice
    let mut rit = Vec::new();
    let rit_cases: Vec<(&str, Vec<Vec<Card>>, Vec<Card>)> = vec![
        ("preflop AsAh vs KsKh", vec![h("AsAh"), h("KsKh")], vec![]),
        ("flop AhKh vs QsQd on Th7h2c", vec![h("AhKh"), h("QsQd")], h("Th7h2c")),
        ("turn AhKh vs QsQd on Th7h2c3d", vec![h("AhKh"), h("QsQd")], h("Th7h2c3d")),
        ("flop 8h9h vs AsAd on 7h6c2h (combo draw)", vec![h("8h9h"), h("AsAd")], h("7h6c2h")),
    ];
    for (name, hands, board) in rit_cases {
        if quick && name.starts_with("preflop") {
            continue;
        }
        let sw = Stopwatch::start();
        let r = pe_equity::rit::run_it_twice_exact(&game, &hands, &board, &[])?;
        rit.push(json!({ "case": name, "result": r, "cpu_seconds": sw.process_cpu() }));
    }
    Ok(json!({ "exact": exact, "monte_carlo": { "case": "NLH preflop AhKh vs QsQd", "exact_equity": truth, "rows": mc }, "run_it_twice": rit }))
}

// ---------------------------------------------------------------- CFR family

fn algos() -> Vec<Algorithm> {
    vec![Algorithm::Cfr, Algorithm::CfrPlus, Algorithm::Linear, Algorithm::DCFR, Algorithm::PredictiveCfrPlus]
}

fn cfr(quick: bool) -> Result<Value, String> {
    let mut runs = Vec::new();
    for (name, players, iters) in [("kuhn", 2, 10_000u64), ("leduc", 2, if quick { 500 } else { 5_000 })] {
        let tree = crate::cfr_tree(name, players)?;
        let marks = pe_cfr::checkpoints(iters);
        for algo in algos() {
            let (_, rep) = pe_cfr::run_cfr(&tree, algo, &marks);
            runs.push(serde_json::to_value(rep).unwrap());
        }
        let mc_iters = iters * if name == "kuhn" { 10 } else { 200 };
        let mc_marks = pe_cfr::checkpoints(mc_iters);
        let (_, es) = pe_cfr::run_mccfr(&tree, Sampling::External, &mc_marks, 11);
        runs.push(serde_json::to_value(es).unwrap());
        let (_, os) = pe_cfr::run_mccfr(&tree, Sampling::Outcome { epsilon: 0.6 }, &mc_marks, 12);
        runs.push(serde_json::to_value(os).unwrap());
    }
    // reference values
    let k = crate::cfr_tree("kuhn", 2)?;
    let l = crate::cfr_tree("leduc", 2)?;
    let refs = json!({
        "kuhn_value_p1": -1.0 / 18.0,
        "kuhn_uniform_exploitability": pe_cfr::nash_conv(&k, &k.uniform()).exploitability,
        "leduc_uniform_exploitability": pe_cfr::nash_conv(&l, &l.uniform()).exploitability,
        "leduc_infosets": l.infosets.len(),
    });
    Ok(json!({ "runs": runs, "references": refs }))
}

fn players(quick: bool) -> Result<Value, String> {
    let mut rows = Vec::new();
    let max_kuhn = if quick { 4 } else { 5 };
    for n in 2..=max_kuhn {
        let tree = Tree::compile(&pe_cfr::games::Kuhn::new(n));
        let iters = if n >= 5 { 300 } else { 2_000 };
        for algo in [Algorithm::Cfr, Algorithm::DCFR] {
            let (_, rep) = pe_cfr::run_cfr(&tree, algo, &pe_cfr::checkpoints(iters));
            rows.push(serde_json::to_value(rep).unwrap());
        }
        let (_, es) = pe_cfr::run_mccfr(&tree, Sampling::External, &pe_cfr::checkpoints(iters * 20), 3);
        rows.push(serde_json::to_value(es).unwrap());
    }
    for n in 2..=3 {
        let tree = Tree::compile(&pe_cfr::games::Leduc::new(n));
        let iters = if quick { 100 } else if n == 2 { 2_000 } else { 200 };
        let (_, rep) = pe_cfr::run_cfr(&tree, Algorithm::DCFR, &pe_cfr::checkpoints(iters));
        rows.push(serde_json::to_value(rep).unwrap());
        let (_, rep) = pe_cfr::run_cfr(&tree, Algorithm::Cfr, &pe_cfr::checkpoints(iters));
        rows.push(serde_json::to_value(rep).unwrap());
    }
    Ok(json!({ "runs": rows }))
}

// ---------------------------------------------------------------- river and turn spots

fn spot(game: &str, board: &str, r0: &str, r1: &str, pot: f64, stack: f64, menu: BetMenu) -> SpotConfig {
    SpotConfig {
        game: game.into(),
        board: board.into(),
        board2: None,
        ranges: vec![r0.into(), r1.into()],
        pot,
        stacks: vec![stack],
        menu,
        rake: None,
        side_rule: None,
        dead: None,
    }
}

pub fn standard_spots() -> Vec<(&'static str, SpotConfig)> {
    let two = BetMenu { bets: vec![0.33, 0.75], raises: vec![1.0], max_raises: 2, allin_threshold: 0.67, add_allin: false };
    let three = BetMenu { bets: vec![0.33, 0.75, 1.5], raises: vec![0.75], max_raises: 3, allin_threshold: 0.67, add_allin: true };
    vec![
        (
            "NLH SRP river BTN vs BB, Qs8h5d3c2s, 2 sizes",
            spot("NLH", "Qs8h5d3c2s", "22+,A2s+,K5s+,Q8s+,J8s+,T8s+,97s+,86s+,75s+,64s+,54s,A8o+,KTo+,QTo+,JTo", "TT+,AQs+,AKo,KQs,QJs,JTs,T9s,98s,A5s,A4s,Q8s,88,55,33,22,KQo", 6.0, 97.0, two.clone()),
        ),
        (
            "NLH SRP river, 3 sizes + all-in",
            spot("NLH", "Qs8h5d3c2s", "22+,A2s+,K5s+,Q8s+,J8s+,T8s+,97s+,86s+,75s+,64s+,54s,A8o+,KTo+,QTo+,JTo", "TT+,AQs+,AKo,KQs,QJs,JTs,T9s,98s,A5s,A4s,Q8s,88,55,33,22,KQo", 6.0, 97.0, three),
        ),
        (
            "NLH 3-bet pot river, Ah9d4c4s2h",
            spot("NLH", "Ah9d4c4s2h", "QQ+,AKs,AKo,AQs,A5s,A4s,KQs,JTs:0.5,T9s:0.5", "JJ-99,AQs,AJs,ATs,KQs,KJs,QJs,JTs,T9s,98s,87s,76s,AQo", 22.0, 78.0, two.clone()),
        ),
        (
            "PLO river, 4-card ranges",
            spot("PLO", "Kh9h6d3c2s", "AAxx:0.1,KKxx:0.3,99xx:0.3,KQJT,K9xx:0.05,QJT8,T875,JTxx:0.03", "AKxx:0.04,QQxx:0.2,JJxx:0.2,KQxx:0.03,T987,8754,QJT9,A5xx:0.02", 10.0, 45.0, two.clone()),
        ),
        (
            "Short deck river (Triton rules)",
            spot("SD", "Ah9d8c7s6h", "random", "TT+,AK,AQ,KQ,JT,T9,98s", 10.0, 50.0, two),
        ),
    ]
}

fn river(quick: bool) -> Result<Value, String> {
    let mut rows = Vec::new();
    let max_iter = if quick { 200 } else { 2_000 };
    for (name, cfg) in standard_spots() {
        for algo in algos() {
            if quick && !matches!(algo, Algorithm::Discounted { .. }) {
                continue;
            }
            let spot = Spot::from_config(cfg.clone())?;
            let mut s = HuSolver::new(spot, algo, None, 2_000_000)?;
            let sw = Stopwatch::start();
            let curve = solve_hu(&mut s, max_iter, 0.05, 10.max(max_iter / 40));
            let cpu = sw.process_cpu();
            let reach = |pct: f64| curve.iter().find(|c| c.pct_pot <= pct).map(|c| c.iteration);
            rows.push(json!({
                "spot": name, "game": cfg.game, "algorithm": algo.name(),
                "hands": [s.hands(0), s.hands(1)], "nodes": s.tree.nodes.len(), "decision_nodes": s.tree.decision_nodes(),
                "memory_bytes": s.memory_bytes(), "cpu_seconds": cpu, "iterations": s.iterations,
                "iterations_to": { "1.0": reach(1.0), "0.5": reach(0.5), "0.3": reach(0.3), "0.1": reach(0.1) },
                "final_pct_pot": curve.last().map(|c| c.pct_pot), "curve": curve,
            }));
        }
    }
    Ok(json!({ "rows": rows }))
}

fn turn(quick: bool) -> Result<Value, String> {
    let menu = BetMenu { bets: vec![0.5, 1.0], raises: vec![1.0], max_raises: 2, allin_threshold: 0.67, add_allin: false };
    let cfg = spot(
        "NLH",
        "Qs8h5d3c",
        "QQ,88,55,AQs,KQs,QJs,QTs,J9s,T9s,98s,87s,76s,65s,A5s,A4s,AhKh,AdKd,AQo,KQo",
        "TT+,AQs+,AKo,KQs,QJs,JTs,T9s,98s,Q8s,88,55,33,AhTh,KhJh",
        12.0,
        88.0,
        menu,
    );
    let max_iter = if quick { 40 } else { 400 };
    let mut rows = Vec::new();
    for algo in [Algorithm::DCFR, Algorithm::CfrPlus] {
        let spot = Spot::from_config(cfg.clone())?;
        let sw_build = Stopwatch::start();
        let mut s = HuSolver::new(spot, algo, None, 3_000_000)?;
        let build = sw_build.process_cpu();
        let sw = Stopwatch::start();
        let curve = solve_hu(&mut s, max_iter, 0.1, 10.max(max_iter / 20));
        rows.push(json!({
            "spot": "NLH turn Qs8h5d3c", "algorithm": algo.name(), "hands": [s.hands(0), s.hands(1)],
            "nodes": s.tree.nodes.len(), "decision_nodes": s.tree.decision_nodes(), "memory_bytes": s.memory_bytes(),
            "build_cpu_seconds": build, "solve_cpu_seconds": sw.process_cpu(), "solve_wall_seconds": sw.wall(),
            "iterations": s.iterations, "final_pct_pot": curve.last().map(|c| c.pct_pot), "curve": curve,
            "peak_working_set_bytes": peak_working_set_bytes(),
        }));
    }
    Ok(json!({ "rows": rows }))
}

// ---------------------------------------------------------------- abstraction

fn abstraction(quick: bool) -> Result<Value, String> {
    let (name, cfg) = standard_spots().into_iter().next().unwrap();
    let iters = if quick { 200 } else { 1_000 };
    // card abstraction: strength buckets on the river
    let mut cards = Vec::new();
    for k in [2usize, 3, 5, 8, 13, 20, 40, 0] {
        let spot = Spot::from_config(cfg.clone())?;
        let b = if k == 0 { None } else { Some(Buckets { count: k }) };
        let mut s = HuSolver::new(spot, Algorithm::DCFR, b, 2_000_000)?;
        solve_hu(&mut s, iters, 0.0, iters);
        let e = s.exploitability();
        cards.push(json!({ "buckets": if k == 0 { Value::Null } else { json!(k) }, "pct_pot": e.pct_pot, "ev_oop": e.ev[0], "memory_bytes": s.memory_bytes() }));
    }
    // action abstraction: the same spot under different bet menus
    let menus: Vec<(&str, BetMenu)> = vec![
        ("one size (75%)", BetMenu { bets: vec![0.75], raises: vec![1.0], max_raises: 2, allin_threshold: 0.67, add_allin: false }),
        ("two sizes (33%, 75%)", BetMenu { bets: vec![0.33, 0.75], raises: vec![1.0], max_raises: 2, allin_threshold: 0.67, add_allin: false }),
        ("three sizes (33%, 75%, 150%)", BetMenu { bets: vec![0.33, 0.75, 1.5], raises: vec![1.0], max_raises: 2, allin_threshold: 0.67, add_allin: false }),
        ("five sizes + all-in", BetMenu { bets: vec![0.25, 0.5, 0.75, 1.0, 1.5], raises: vec![0.5, 1.0], max_raises: 2, allin_threshold: 0.67, add_allin: true }),
    ];
    let mut actions = Vec::new();
    for (label, menu) in menus {
        let mut c = cfg.clone();
        c.menu = menu;
        let spot = Spot::from_config(c)?;
        let mut s = HuSolver::new(spot, Algorithm::DCFR, None, 2_000_000)?;
        let sw = Stopwatch::start();
        solve_hu(&mut s, iters, 0.05, 100);
        let e = s.exploitability();
        let root = pe_solver::node_report(&s, 0)?;
        actions.push(json!({
            "menu": label, "ev": e.ev, "pct_pot": e.pct_pot, "decision_nodes": s.tree.decision_nodes(),
            "memory_bytes": s.memory_bytes(), "cpu_seconds": sw.process_cpu(),
            "root_actions": root.actions, "root_frequencies": root.frequencies,
        }));
    }
    // Leduc: merging jacks and queens in information sets
    let leduc = pe_cfr::games::Leduc::new(2);
    let full = Tree::compile(&leduc);
    let abs = Tree::compile_abstracted(&leduc, |k: &str| {
        let mut p: Vec<String> = k.split('|').map(|s| s.to_string()).collect();
        for i in [1, 2] {
            if p[i] == "0" {
                p[i] = "1".into();
            }
        }
        p.join("|")
    });
    let lit = if quick { 200 } else { 2_000 };
    let (_, r_full) = pe_cfr::run_cfr(&full, Algorithm::DCFR, &pe_cfr::checkpoints(lit));
    let (_, r_abs) = pe_cfr::run_cfr(&abs, Algorithm::DCFR, &pe_cfr::checkpoints(lit));
    Ok(json!({ "spot": name, "card_buckets": cards, "bet_menus": actions, "leduc": { "full": r_full, "jack_queen_merged": r_abs } }))
}

// ---------------------------------------------------------------- multiway

fn multiway(quick: bool) -> Result<Value, String> {
    let menu = BetMenu { bets: vec![0.75], raises: vec![1.0], max_raises: 1, allin_threshold: 0.9, add_allin: false };
    // about 17 combos per player after board removal, so 2-4 players compare at equal range size
    let pools = ["AA,QJs,T8s,65s", "QQ,JTs,A5s,86s", "JJ,AQs,KJs,T9s", "TT,AJs,QTs,87s"];
    let iters = if quick { 50 } else { 300 };
    let mut rows = Vec::new();
    for n in 2..=4 {
        if quick && n == 4 {
            continue;
        }
        let cfg = SpotConfig {
            game: "NLH".into(),
            board: "Ks7d2c9h4s".into(),
            board2: None,
            ranges: pools[..n].iter().map(|s| s.to_string()).collect(),
            pot: 10.0 * n as f64,
            stacks: vec![60.0],
            menu: menu.clone(),
            rake: None,
            side_rule: None,
            dead: None,
        };
        let mut s = MultiwaySolver::new(Spot::from_config(cfg)?, Algorithm::DCFR, 50_000_000, 2_000_000)?;
        let mut curve = Vec::new();
        let sw = Stopwatch::start();
        let start = s.report();
        curve.push(json!({ "iteration": 0, "nash_conv_pct_pot": start.nash_conv_pct_pot }));
        for mark in pe_cfr::checkpoints(iters) {
            while s.iterations < mark {
                s.iterate();
            }
            let r = s.report();
            curve.push(json!({ "iteration": mark, "nash_conv_pct_pot": r.nash_conv_pct_pot, "cpu_seconds": sw.process_cpu() }));
        }
        let r = s.report();
        let train_cpu = sw.process_cpu();
        rows.push(json!({
            "players": n, "hands": (0..n).map(|p| s.spot.ranges[p].len()).collect::<Vec<_>>(), "tuples": s.tuple_count(),
            "nodes": s.tree.nodes.len(), "decision_nodes": s.tree.decision_nodes(), "memory_bytes": s.memory_bytes(),
            "iterations": s.iterations, "cpu_seconds": train_cpu, "cpu_seconds_per_iteration": train_cpu / s.iterations as f64,
            "final": r, "curve": curve,
        }));
    }
    Ok(json!({ "rows": rows, "note": "NashConv is exact; with three or more players CFR has no convergence guarantee" }))
}

// ---------------------------------------------------------------- exploitation

fn exploit(quick: bool) -> Result<Value, String> {
    let (name, cfg) = standard_spots().into_iter().next().unwrap();
    let iters = if quick { 150 } else { 1_000 };
    let base = Spot::from_config(cfg)?;
    let mut gto = HuSolver::new(base.clone(), Algorithm::DCFR, None, 2_000_000)?;
    solve_hu(&mut gto, iters, 0.05, 100);
    let gto_e = gto.exploitability();
    // player 0 = big blind (out of position, acts first), player 1 = button (in position)
    let profiles: Vec<(&str, usize, Profile)> = vec![
        ("Button folds 70% to river bets", 1, Profile { fold_to_bet: Some(0.70), raise_vs_bet: Some(0.03), ..Default::default() }),
        ("Button folds only 15% to river bets (calls too much)", 1, Profile { fold_to_bet: Some(0.15), raise_vs_bet: Some(0.02), ..Default::default() }),
        ("Button bets 70% when checked to, half of them bluffs", 1, Profile { bet_when_checked_to: Some(0.70), bluff_share: 0.5, size: Some(0.75), ..Default::default() }),
        ("Big blind folds 70% to a bet after checking", 0, Profile { fold_to_bet: Some(0.70), raise_vs_bet: Some(0.03), ..Default::default() }),
    ];
    let mut out = Vec::new();
    for (label, model_player, profile) in profiles {
        let exploiter = 1 - model_player;
        let model = build_model(&gto, model_player, &profile);
        let mut frontier = Vec::new();
        for &p in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
            let mut s = HuSolver::new(base.clone(), Algorithm::DCFR, None, 2_000_000)?;
            s.set_model(model_player, model.clone(), p);
            for _ in 0..iters {
                s.iterate();
            }
            let (ev_model, br_model) = s.values_vs_model(exploiter).unwrap();
            // exploitability of the exploiter's strategy alone: what a best-responding
            // opponent gains over the equilibrium value (taken from the equilibrium solve)
            s.clear_model();
            let e = s.exploitability();
            let exploiter_loss = (e.br[model_player] - gto_e.ev[model_player]).max(0.0);
            frontier.push(json!({
                "p": p, "ev_vs_model": ev_model, "best_response_vs_model": br_model,
                "exploitability_of_exploiter_pct_pot": 100.0 * exploiter_loss / base.pot,
            }));
        }
        out.push(json!({
            "profile": label, "model_player": model_player, "exploiter": exploiter, "frontier": frontier,
            "equilibrium_value_exploiter": gto_e.ev[exploiter], "equilibrium_pct_pot": gto_e.pct_pot,
        }));
    }
    Ok(json!({ "spot": name, "iterations": iters, "profiles": out }))
}

// ---------------------------------------------------------------- street-by-street ranges

/// The research corpus (`pe hh simulate --hands 20000 --seed 1 --straddle 0.05
/// --bomb-every 50 --double-board --run-twice 0.2 --rake 5`), regenerated in
/// process: the simulator is deterministic, so this is the same 20,000 hands.
fn research_corpus() -> pe_hh::SimOutput {
    pe_hh::simulate(&pe_hh::SimConfig {
        hands: 20_000,
        seed: 1,
        straddle_rate: 0.05,
        bomb_pot_every: 50,
        double_board_bombs: true,
        run_twice_rate: 0.2,
        rake_pct: 5.0,
        ..Default::default()
    })
}

#[derive(Clone, Copy)]
struct RangeRecord {
    half: usize,
    label: usize,
    style: usize,
    bits: f64,
    quantile: f64,
    share: f64,
}

const LABELS: [&str; 4] = ["flop", "turn", "river", "end"];

fn summarise_records(records: &[&RangeRecord]) -> Value {
    let n = records.len();
    if n == 0 {
        return json!({ "n": 0 });
    }
    let mean = |f: &dyn Fn(&RangeRecord) -> f64| records.iter().map(|r| f(r)).sum::<f64>() / n as f64;
    let mut bits: Vec<f64> = records.iter().map(|r| r.bits).collect();
    bits.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cover = |level: f64| records.iter().filter(|r| r.quantile <= level).count() as f64 / n as f64;
    json!({
        "n": n,
        "mean_bits": mean(&|r| r.bits),
        "median_bits": bits[n / 2],
        "p10_bits": bits[n / 10],
        "coverage_50": cover(0.5),
        "coverage_80": cover(0.8),
        "coverage_90": cover(0.9),
        "mean_effective_share": mean(&|r| r.share),
    })
}

/// Fits the noise level of the range model on simulated hands whose cards are
/// all known and measures what the ranges reveal (bits about the true hand
/// against a uniform range) and whether their credible regions are calibrated.
fn ranges(quick: bool) -> Result<Value, String> {
    use rayon::prelude::*;
    let sw = Stopwatch::start();
    let corpus = research_corpus();
    let styles: Vec<String> = pe_hh::sim::STYLES.iter().map(|s| s.to_string()).collect();
    let style_of: std::collections::BTreeMap<String, usize> =
        corpus.bots.iter().map(|b| (b.name.clone(), styles.iter().position(|s| *s == b.style).unwrap_or(0))).collect();
    let table = pe_hh::FrequencyTable::from_hands(&corpus.hands);
    let n_eval = if quick { 600 } else { 4000 };
    let eval: Vec<&pe_hh::Hand> = corpus.hands.iter().take(n_eval).filter(|h| pe_hh::ranges::is_holdem(h) && h.streets.len() > 1).collect();
    let strengths: Vec<Vec<Vec<f64>>> = eval.par_iter().map(|h| pe_hh::ranges::street_strengths(h)).collect();
    let strength_cpu = sw.process_cpu();
    let run = |opts: &pe_hh::RangeOptions| -> Vec<RangeRecord> {
        eval.par_iter()
            .zip(strengths.par_iter())
            .enumerate()
            .flat_map_iter(|(k, (h, s))| {
                let half = usize::from(k * 2 >= eval.len());
                let r = pe_hh::ranges::reconstruct_with(h, &table, opts, s).expect("hold'em hand");
                let mut v = Vec::new();
                for snap in &r.snapshots {
                    let Some(label) = LABELS.iter().position(|l| *l == snap.label) else { continue };
                    for seat in &snap.seats {
                        if let Some(a) = &seat.actual {
                            v.push(RangeRecord {
                                half,
                                label,
                                style: style_of.get(&seat.name).copied().unwrap_or(0),
                                bits: a.bits,
                                quantile: a.hpd_quantile,
                                share: seat.effective_combos / seat.possible_combos as f64,
                            });
                        }
                    }
                }
                v
            })
            .collect()
    };
    let base = pe_hh::RangeOptions::default();
    let noises = if quick { vec![0.0, 0.2, 0.4, 0.6] } else { vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] };
    let mut grid = Vec::new();
    let mut best = (f64::NEG_INFINITY, base.noise);
    for &noise in &noises {
        let opts = pe_hh::RangeOptions { noise, ..base.clone() };
        let recs = run(&opts);
        let tune: Vec<&RangeRecord> = recs.iter().filter(|r| r.half == 0).collect();
        let s = summarise_records(&tune);
        let mean_bits = s["mean_bits"].as_f64().unwrap_or(f64::NEG_INFINITY);
        if mean_bits > best.0 {
            best = (mean_bits, noise);
        }
        grid.push(json!({ "noise": noise, "tuning_half": s }));
    }
    // the chosen noise on the held-out half, by snapshot and by style
    let chosen = pe_hh::RangeOptions { noise: best.1, ..base.clone() };
    let recs = run(&chosen);
    let test: Vec<&RangeRecord> = recs.iter().filter(|r| r.half == 1).collect();
    let by_label: Vec<Value> = (0..LABELS.len())
        .map(|l| json!({ "snapshot": LABELS[l], "metrics": summarise_records(&test.iter().copied().filter(|r| r.label == l).collect::<Vec<_>>()) }))
        .collect();
    let by_style: Vec<Value> = (0..styles.len())
        .map(|s| json!({ "style": styles[s], "river": summarise_records(&test.iter().copied().filter(|r| r.style == s && r.label == 2).collect::<Vec<_>>()) }))
        .collect();
    // ablations on the held-out half: population frequencies only, and no noise
    let ablation = |opts: pe_hh::RangeOptions| -> Value {
        let recs = run(&opts);
        let t: Vec<&RangeRecord> = recs.iter().filter(|r| r.half == 1).collect();
        let river: Vec<&RangeRecord> = t.iter().copied().filter(|r| r.label == 2).collect();
        json!({ "all": summarise_records(&t), "river": summarise_records(&river) })
    };
    let population_only = ablation(pe_hh::RangeOptions { player_prior: 1e12, ..chosen.clone() });
    let no_noise = ablation(pe_hh::RangeOptions { noise: 0.0, ..chosen.clone() });
    Ok(json!({
        "corpus": { "hands": corpus.hands.len(), "players": corpus.bots.len(), "frequency_hands": table.hands, "evaluated_hands": eval.len() },
        "options": chosen,
        "chosen_noise": best.1,
        "grid": grid,
        "held_out": { "all": summarise_records(&test), "by_snapshot": by_label, "by_style": by_style },
        "ablations": { "population_frequencies_only": population_only, "no_noise": no_noise },
        "strength_cpu_seconds": strength_cpu,
        "note": "bits = log2(probability of the true combo x combos possible); 0 = no better than a uniform range. coverage_q = share of true combos inside the q credible region (calibrated when equal to q). Noise fitted on the first half of the evaluated hands, metrics on the second.",
    }))
}

/// Reviews the heads-up river decisions of simulated hands and summarises the
/// cost of each style's decisions against the equilibrium.
fn review(quick: bool) -> Result<Value, String> {
    let sw = Stopwatch::start();
    let corpus = research_corpus();
    let table = pe_hh::FrequencyTable::from_hands(&corpus.hands);
    let style_of: std::collections::BTreeMap<String, String> = corpus.bots.iter().map(|b| (b.name.clone(), b.style.clone())).collect();
    let n = if quick { 1500 } else { 6000 };
    let ropts = pe_hh::RangeOptions::default();
    let opts = pe_hh::ReviewOptions::default();
    let (reviews, failed) = pe_hh::review_hands(&corpus.hands[..n], &table, &ropts, &opts);
    // (cost, frequency, on the equilibrium path) per decision with known cards
    let mut by_style: std::collections::BTreeMap<String, Vec<(f64, f64, bool)>> = Default::default();
    let mut by_action: std::collections::BTreeMap<String, Vec<(f64, f64, bool)>> = Default::default();
    for r in &reviews {
        for d in &r.decisions {
            if let (Some(loss), Some(freq)) = (d.ev_loss_bb, d.frequency) {
                let style = style_of.get(&d.name).cloned().unwrap_or_default();
                let action = d.actions[d.taken].split(' ').next().unwrap_or("").to_string();
                let on = pe_hh::review::on_path(d);
                by_style.entry(style).or_default().push((loss, freq, on));
                by_action.entry(action).or_default().push((loss, freq, on));
            }
        }
    }
    let stats = |v: &[f64]| -> Value {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = s.len();
        if n == 0 {
            return json!({ "n": 0 });
        }
        json!({ "n": n, "mean_loss_bb": s.iter().sum::<f64>() / n as f64, "median_loss_bb": s[n / 2], "p90_loss_bb": s[(n * 9) / 10], "share_over_1bb": s.iter().filter(|x| **x > 1.0).count() as f64 / n as f64 })
    };
    let group = |v: &[(f64, f64, bool)]| -> Value {
        let all: Vec<f64> = v.iter().map(|x| x.0).collect();
        // strategies and values are reliable where both players get with some regularity
        let on: Vec<&(f64, f64, bool)> = v.iter().filter(|x| x.2).collect();
        let on_losses: Vec<f64> = on.iter().map(|x| x.0).collect();
        let rare = on.iter().filter(|x| x.1 < 0.05).count() as f64 / on.len().max(1) as f64;
        json!({ "decisions": stats(&all), "on_path": stats(&on_losses), "share_on_path": on.len() as f64 / v.len().max(1) as f64, "share_rarely_played": rare })
    };
    let styles: Vec<Value> = by_style.iter().map(|(style, v)| json!({ "style": style, "stats": group(v) })).collect();
    let actions: Vec<Value> = by_action.iter().map(|(a, v)| json!({ "action": a, "stats": group(v) })).collect();
    let all: Vec<(f64, f64, bool)> = by_style.values().flatten().copied().collect();
    let solves = reviews.len().max(1) as f64;
    let describe = |id: &str, d: &pe_hh::review::ReviewedDecision| {
        json!({ "hand": id, "player": d.name, "style": style_of.get(&d.name), "cards": d.hand, "action": d.actions[d.taken], "best": d.best.map(|b| d.actions[b].clone()), "loss_bb": d.ev_loss_bb, "frequency": d.frequency, "path_probability": d.path_probability, "opponent_path_probability": d.opponent_path_probability, "on_path": pe_hh::review::on_path(d) })
    };
    let costliest: Vec<Value> = pe_hh::review::costliest(&reviews, 12).iter().map(|(id, d)| describe(id, d)).collect();
    let on_path_reviews: Vec<pe_hh::RiverReview> = reviews
        .iter()
        .map(|r| pe_hh::RiverReview { decisions: r.decisions.iter().filter(|d| pe_hh::review::on_path(d)).cloned().collect(), ..r.clone() })
        .collect();
    let costliest_on_path: Vec<Value> = pe_hh::review::costliest(&on_path_reviews, 12).iter().map(|(id, d)| describe(id, d)).collect();
    Ok(json!({
        "hands": n,
        "reviewed_hands": reviews.len(),
        "failed": failed,
        "decisions": by_style.values().map(|v| v.len()).sum::<usize>(),
        "mean_exploitability_pct": reviews.iter().map(|r| r.exploitability_pct).sum::<f64>() / solves,
        "mean_iterations": reviews.iter().map(|r| r.iterations as f64).sum::<f64>() / solves,
        "mean_combos": [reviews.iter().map(|r| r.combos[0] as f64).sum::<f64>() / solves, reviews.iter().map(|r| r.combos[1] as f64).sum::<f64>() / solves],
        "overall": group(&all),
        "by_style": styles,
        "by_action": actions,
        "costliest": costliest,
        "costliest_on_path": costliest_on_path,
        "range_options": ropts,
        "options": opts,
        "cpu_seconds": sw.process_cpu(),
        "note": "Costs are big blinds against an opponent playing the equilibrium of the reconstructed river game (model-based ranges, the hand's own bet sizes); they illustrate the review, they are not the bots' true losses. On path: the equilibrium reaches the decision with the hand's own line at least 5% of the time and with the opponent's range at least 2%; off the path the solver explores little and its values are rough. share_rarely_played: on-path decisions the equilibrium takes less than 5% of the time.",
    }))
}

// ---------------------------------------------------------------- analytic benchmarks

fn analytic(quick: bool) -> Result<Value, String> {
    // clairvoyance game as a card game: OOP holds nut quads or air that plays the
    // board, IP holds bluff-catchers; no card removal between the ranges.
    let mut rows = Vec::new();
    for &b in &[0.5, 1.0, 2.0] {
        let menu = BetMenu { bets: vec![b], raises: vec![], max_raises: 1, allin_threshold: 1.0, add_allin: false };
        let cfg = spot("NLH", "2s2c2d7h7c", "2h3c,2h4c,2h5c,2h6c,3d4d,3h4h,5d6d,5h6h", "8s8d,8h8c,9s9d,9h9c", 10.0, 100.0, menu);
        let game = pe_cfr::games::Clairvoyance { v: 0.5, b };
        let (bluff, call, value) = game.solution();
        let mut s = HuSolver::new(Spot::from_config(cfg)?, Algorithm::DCFR, None, 10_000)?;
        let mut curve = Vec::new();
        for mark in pe_cfr::checkpoints(if quick { 1_000 } else { 10_000 }) {
            while s.iterations < mark {
                s.iterate();
            }
            let sigma = s.average_strategy(0);
            let n = s.hands(0);
            // hands 4.. are the air combos in parse order? find them by strength instead
            let vals = s.river_values(0).unwrap();
            let max_v = *vals.iter().max().unwrap();
            let air: Vec<usize> = (0..n).filter(|&h| vals[h] < max_v).collect();
            let bet_air = air.iter().map(|&h| sigma[n + h]).sum::<f64>() / air.len() as f64;
            let code = format!("b{}", (b * 100.0).round());
            let bet_node = s.tree.find(&[code.as_str()]).unwrap();
            let sig_ip = s.average_strategy(bet_node);
            let m = s.hands(1);
            let call_freq = (0..m).map(|h| sig_ip[m + h]).sum::<f64>() / m as f64;
            let e = s.exploitability();
            let ev_game = (e.ev[0] - 5.0) / 10.0; // convert to the pot-1, antes-0.5 convention
            curve.push(json!({
                "iteration": mark, "bluff_freq": bet_air, "call_freq": call_freq, "value": ev_game,
                "bluff_error": (bet_air - bluff).abs(), "call_error": (call_freq - call).abs(), "value_error": (ev_game - value).abs(),
                "pct_pot": e.pct_pot,
            }));
        }
        rows.push(json!({ "bet_size": b, "closed_form": { "bluff_freq": bluff, "call_freq": call, "value": value }, "curve": curve }));
    }
    // known game values
    let k = crate::cfr_tree("kuhn", 2)?;
    let (ks, _) = pe_cfr::run_cfr(&k, Algorithm::DCFR, &[if quick { 2_000 } else { 100_000 }]);
    let l = crate::cfr_tree("leduc", 2)?;
    let (ls, lr) = pe_cfr::run_cfr(&l, Algorithm::DCFR, &[if quick { 500 } else { 20_000 }]);
    let kv = pe_cfr::expected_values(&k, &ks)[0];
    let lv = pe_cfr::expected_values(&l, &ls)[0];
    Ok(json!({
        "clairvoyance": rows,
        "kuhn": { "computed": kv, "known": -1.0 / 18.0, "error": (kv + 1.0 / 18.0).abs() },
        "leduc": { "computed": lv, "published": -0.0856, "exploitability": lr.curve.last().and_then(|c| c.exploitability) },
    }))
}
