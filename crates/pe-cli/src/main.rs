//! `pe`: the poker-ecosystem command-line tool. Every command prints JSON (or
//! writes it with `--out`), so the Python research layer and the benchmark
//! scripts can drive it.

mod args;
mod bench;
mod hh;
mod vectors;

use args::Args;
use pe_cards::{Game, parse_cards};
use pe_cfr::{Algorithm, Sampling, Tree};
use pe_solver::{HuSolver, MultiwaySolver, Spot, SpotConfig};
use serde_json::{Value, json};

const HELP: &str = "pe - poker research toolkit

USAGE:
  pe eval CARDS [--game NLH]                       evaluate a 5-7 card hand
  pe equity --hands \"AhKh QsQd\" [--board ..] [--board2 ..] [--dead ..] [--game NLH] [--mc N] [--seed S]
  pe range-equity --ranges \"AA,KK;QQ+,AK\" [--board ..] [--game NLH] [--samples N]
  pe rit --hands \"AhKh QsQd\" [--board ..] [--game NLH]   exact run-it-twice statistics
  pe rank-preflop [--samples N]                    equity of all 169 classes vs a random hand
  pe cfr --game kuhn|leduc|clairvoyance [--players N] [--algo dcfr] [--iters N] [--sampling es|os]
  pe solve SPOT.json [--algo dcfr] [--iters N] [--target PCT] [--check N] [--buckets K]
  pe solve-mw SPOT.json [--algo dcfr] [--iters N] [--check N]
  pe bench SUITE [--out FILE] [--quick]            suites: eval equity cfr players river turn abstraction multiway exploit analytic all
  pe vectors [--dir DIR]                           test vectors for the JavaScript engine
  pe hh parse FILE.txt [--jsonl OUT]               parse PokerStars text, reconstruct, all-in EV
  pe hh simulate [--hands N] [--seed S] [--dir DIR] [--straddle P] [--bomb-every N] [--double-board] [--run-twice P]
  pe hh summary FILE                               counts for a .txt/.json/.jsonl file

Options common to all commands: --out FILE writes the JSON result to FILE.";

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.is_empty() || raw[0] == "help" || raw[0] == "--help" || raw[0] == "-h" {
        println!("{HELP}");
        return;
    }
    let cmd = raw[0].clone();
    let args = Args::parse(&raw[1..], &["quick", "exact"]);
    let result = match cmd.as_str() {
        "eval" => cmd_eval(&args),
        "equity" => cmd_equity(&args),
        "range-equity" => cmd_range_equity(&args),
        "rit" => cmd_rit(&args),
        "rank-preflop" => cmd_rank(&args),
        "cfr" => cmd_cfr(&args),
        "solve" => cmd_solve(&args),
        "solve-mw" => cmd_solve_mw(&args),
        "bench" => bench::run(&args),
        "vectors" => vectors::run(&args),
        "hh" => hh::run(&args),
        other => Err(format!("unknown command '{other}' (try `pe help`)")),
    };
    match result {
        Ok(v) => {
            let text = serde_json::to_string_pretty(&v).unwrap();
            if let Some(path) = args.get("out") {
                if let Some(dir) = std::path::Path::new(path).parent() {
                    let _ = std::fs::create_dir_all(dir);
                }
                if let Err(e) = std::fs::write(path, &text) {
                    eprintln!("error: cannot write {path}: {e}");
                    std::process::exit(1);
                }
                eprintln!("wrote {path}");
            } else {
                println!("{text}");
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

fn game_arg(args: &Args) -> Result<Game, String> {
    Game::parse(&args.str_or("game", "NLH"))
}

fn hands_arg(args: &Args, game: &Game) -> Result<Vec<Vec<pe_cards::Card>>, String> {
    let s = args.get("hands").ok_or("--hands is required")?;
    let parts: Vec<&str> = s.split(|c| c == ' ' || c == ';' || c == '|').filter(|p| !p.is_empty()).collect();
    let mut out = Vec::new();
    for p in parts {
        let cards = parse_cards(p)?;
        if cards.len() != game.hole_cards {
            return Err(format!("'{p}' should have {} cards", game.hole_cards));
        }
        out.push(cards);
    }
    Ok(out)
}

fn cards_opt(args: &Args, k: &str) -> Result<Vec<pe_cards::Card>, String> {
    match args.get(k) {
        Some(s) if !s.is_empty() => parse_cards(s),
        _ => Ok(vec![]),
    }
}

fn cmd_eval(args: &Args) -> Result<Value, String> {
    let game = game_arg(args)?;
    let cards = parse_cards(args.positional.first().ok_or("give the cards")?)?;
    if !(5..=7).contains(&cards.len()) {
        return Err("give 5 to 7 cards".into());
    }
    let v = pe_eval::eval_high(pe_cards::mask_of(&cards), game.high);
    Ok(json!({
        "cards": pe_cards::cards_to_string(&cards),
        "value": v,
        "category": pe_eval::category(v).name(),
        "description": pe_eval::describe(v, game.high),
        "dense_rank": if game.high == pe_cards::HighRules::Standard { Some(pe_eval::dense_rank(v)) } else { None },
    }))
}

fn cmd_equity(args: &Args) -> Result<Value, String> {
    let game = game_arg(args)?;
    let hands = hands_arg(args, &game)?;
    let mut boards = vec![cards_opt(args, "board")?];
    if args.get("board2").is_some() {
        boards.push(cards_opt(args, "board2")?);
    }
    let dead = cards_opt(args, "dead")?;
    let sw = pe_cards::sys::Stopwatch::start();
    let r = match args.get("mc") {
        Some(n) => pe_equity::equity_monte_carlo(&game, &hands, &boards, &dead, n.parse().map_err(|_| "bad --mc")?, args.num("seed", 1)?)?,
        None => pe_equity::equity_exact(&game, &hands, &boards, &dead)?,
    };
    Ok(json!({ "game": game.code(), "result": r, "wall_seconds": sw.wall(), "cpu_seconds": sw.process_cpu() }))
}

fn cmd_range_equity(args: &Args) -> Result<Value, String> {
    let game = game_arg(args)?;
    let ranking = pe_solver::preflop_ranking();
    let board = cards_opt(args, "board")?;
    let dead = pe_cards::mask_of(&board);
    let lists: Vec<pe_cards::HandList> = args
        .get("ranges")
        .ok_or("--ranges is required")?
        .split(';')
        .map(|r| pe_cards::parse_hand_list(&game, r, dead, Some(&ranking)))
        .collect::<Result<_, _>>()?;
    if args.flag("exact") && lists.len() == 2 {
        let r = pe_equity::range_vs_range_exact(&game, &lists[0], &lists[1], &board)?;
        return Ok(json!({ "exact": true, "equity": r.equity, "runouts": r.runouts }));
    }
    let r = pe_equity::range_equity_monte_carlo(&game, &lists, &[board], &[], args.num("samples", 200_000)?, args.num("seed", 1)?)?;
    Ok(json!({ "exact": false, "result": r }))
}

fn cmd_rit(args: &Args) -> Result<Value, String> {
    let game = game_arg(args)?;
    let hands = hands_arg(args, &game)?;
    let board = cards_opt(args, "board")?;
    let r = pe_equity::rit::run_it_twice_exact(&game, &hands, &board, &cards_opt(args, "dead")?)?;
    Ok(serde_json::to_value(r).unwrap())
}

fn cmd_rank(args: &Args) -> Result<Value, String> {
    let n = args.num("samples", 2_000_000u64)?;
    let ranking = pe_equity::preflop_ranking(n, args.num("seed", 7)?);
    let rows: Vec<Value> = ranking.iter().map(|(c, e)| json!({ "class": pe_cards::class_name(*c), "equity": e })).collect();
    Ok(json!({ "samples_per_class": n, "order": ranking.iter().map(|(c, _)| pe_cards::class_name(*c)).collect::<Vec<_>>(), "rows": rows }))
}

pub fn cfr_tree(name: &str, players: usize) -> Result<Tree, String> {
    Ok(match name {
        "kuhn" => Tree::compile(&pe_cfr::games::Kuhn::new(players)),
        "leduc" => Tree::compile(&pe_cfr::games::Leduc::new(players)),
        "clairvoyance" => Tree::compile(&pe_cfr::games::Clairvoyance { v: 0.5, b: 1.0 }),
        other => return Err(format!("unknown game '{other}'")),
    })
}

fn cmd_cfr(args: &Args) -> Result<Value, String> {
    let tree = cfr_tree(&args.str_or("game", "kuhn"), args.num("players", 2)?)?;
    let iters: u64 = args.num("iters", 1000)?;
    let marks = pe_cfr::checkpoints(iters);
    let report = match args.get("sampling") {
        Some("es") => pe_cfr::run_mccfr(&tree, Sampling::External, &marks, args.num("seed", 1)?).1,
        Some("os") => pe_cfr::run_mccfr(&tree, Sampling::Outcome { epsilon: 0.6 }, &marks, args.num("seed", 1)?).1,
        _ => pe_cfr::run_cfr(&tree, Algorithm::parse(&args.str_or("algo", "dcfr"))?, &marks).1,
    };
    Ok(serde_json::to_value(report).unwrap())
}

fn read_spot(args: &Args) -> Result<SpotConfig, String> {
    let path = args.positional.first().ok_or("give a spot JSON file")?;
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))
}

fn cmd_solve(args: &Args) -> Result<Value, String> {
    let cfg = read_spot(args)?;
    let algo = Algorithm::parse(&args.str_or("algo", "dcfr"))?;
    let buckets: usize = args.num("buckets", 0)?;
    if buckets > 0 {
        let spot = Spot::from_config(cfg)?;
        let mut s = HuSolver::new(spot, algo, Some(pe_solver::Buckets { count: buckets }), 2_000_000)?;
        let curve = pe_solver::solve_hu(&mut s, args.num("iters", 1000)?, args.num("target", 0.0)?, args.num("check", 100)?);
        return Ok(json!({ "buckets": buckets, "curve": curve, "result": s.exploitability() }));
    }
    let report = pe_solver::run_hu(cfg, algo, args.num("iters", 1000)?, args.num("target", 0.3)?, args.num("check", 50)?)?;
    Ok(serde_json::to_value(report).unwrap())
}

fn cmd_solve_mw(args: &Args) -> Result<Value, String> {
    let cfg = read_spot(args)?;
    let algo = Algorithm::parse(&args.str_or("algo", "dcfr"))?;
    let mut s = MultiwaySolver::new(Spot::from_config(cfg)?, algo, 20_000_000, 2_000_000)?;
    let iters: u64 = args.num("iters", 500)?;
    let check: u64 = args.num("check", 50)?;
    let sw = pe_cards::sys::Stopwatch::start();
    let mut curve = Vec::new();
    while s.iterations < iters {
        let next = (s.iterations + check).min(iters);
        while s.iterations < next {
            s.iterate();
        }
        let r = s.report();
        curve.push(json!({ "iteration": s.iterations, "nash_conv_pct_pot": r.nash_conv_pct_pot, "cpu_seconds": sw.process_cpu() }));
    }
    let r = s.report();
    Ok(json!({ "tuples": s.tuple_count(), "memory_bytes": s.memory_bytes(), "curve": curve, "result": r }))
}
