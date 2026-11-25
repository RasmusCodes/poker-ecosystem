//! `pe hh ...`: hand-history commands.

use pe_hh::{Hand, analyse, parse_pokerstars, simulate, summarise, to_jsonl, write_pokerstars};
use rayon::prelude::*;
use serde_json::{Value, json};

use crate::args::Args;

pub fn run(args: &Args) -> Result<Value, String> {
    let sub = args.positional.first().map(|s| s.as_str()).unwrap_or("");
    match sub {
        "parse" => parse(args),
        "simulate" => sim(args),
        "summary" => summary(args),
        "ranges" => ranges(args),
        "review" => review(args),
        _ => Err("usage: pe hh parse FILE.txt [--jsonl OUT] | pe hh simulate [--hands N] [--seed S] [--mississippi P] [--dir DIR] | pe hh summary FILE | pe hh ranges FILE [--id ID] [--limit N] [--tremble E] | pe hh review FILE [--limit N] [--top N] [--out FILE]".into()),
    }
}

fn range_options(args: &Args) -> Result<pe_hh::RangeOptions, String> {
    let d = pe_hh::RangeOptions::default();
    Ok(pe_hh::RangeOptions {
        noise: args.num("noise", d.noise)?,
        tremble: args.num("tremble", d.tremble)?,
        player_prior: args.num("player-prior", d.player_prior)?,
        population_prior: args.num("population-prior", d.population_prior)?,
    })
}

/// Street-by-street ranges of every hand in a file (or of one hand with `--id`);
/// the frequencies come from all hands in the file.
fn ranges(args: &Args) -> Result<Value, String> {
    let path = args.positional.get(1).ok_or("give a hand-history file")?;
    let hands = load(path)?;
    let table = pe_hh::FrequencyTable::from_hands(&hands);
    let opts = range_options(args)?;
    let limit: usize = args.num("limit", 50)?;
    let chosen: Vec<&Hand> = match args.get("id") {
        Some(id) => hands.iter().filter(|h| h.id == id).collect(),
        None => hands.iter().filter(|h| pe_hh::ranges::is_holdem(h)).take(limit).collect(),
    };
    if chosen.is_empty() {
        return Err("no hold'em hand to reconstruct".into());
    }
    let out: Vec<Value> = chosen.par_iter().map(|h| pe_hh::reconstruct_ranges(h, &table, &opts).map_or_else(|e| json!({ "id": h.id, "error": e }), |r| json!(r))).collect();
    Ok(json!({ "options": opts, "frequency_hands": table.hands, "hands": out }))
}

/// Reviews every heads-up river decision in a file against an equilibrium.
fn review(args: &Args) -> Result<Value, String> {
    let path = args.positional.get(1).ok_or("give a hand-history file")?;
    let mut hands = load(path)?;
    let table = pe_hh::FrequencyTable::from_hands(&hands);
    let ropts = range_options(args)?;
    let d = pe_hh::ReviewOptions::default();
    let opts = pe_hh::ReviewOptions {
        max_iterations: args.num("iterations", d.max_iterations)?,
        target_pct: args.num("target", d.target_pct)?,
        check_every: d.check_every,
        prune: args.num("prune", d.prune)?,
    };
    let limit: usize = args.num("limit", usize::MAX)?;
    hands.truncate(limit);
    let sw = pe_cards::sys::Stopwatch::start();
    let (reviews, failed) = pe_hh::review_hands(&hands, &table, &ropts, &opts);
    let top: usize = args.num("top", 20)?;
    let costliest: Vec<Value> = pe_hh::review::costliest(&reviews, top).into_iter().map(|(id, d)| json!({ "hand": id, "decision": d })).collect();
    // `--out FILE` is handled by main (it writes the returned JSON)
    Ok(json!({
        "hands": hands.len(), "reviewed": reviews.len(), "failed": failed, "range_options": ropts, "options": opts,
        "cpu_seconds": sw.process_cpu(), "costliest": costliest, "reviews": reviews,
    }))
}

fn load(path: &str) -> Result<Vec<Hand>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    if path.ends_with(".json") || path.ends_with(".jsonl") {
        pe_hh::read_hands(&text)
    } else {
        let (hands, warnings) = parse_pokerstars(&text);
        for w in warnings.iter().take(20) {
            eprintln!("warning: {w}");
        }
        Ok(hands)
    }
}

fn parse(args: &Args) -> Result<Value, String> {
    let path = args.positional.get(1).ok_or("give a hand-history file")?;
    let mut hands = load(path)?;
    hands.par_iter_mut().for_each(analyse);
    if let Some(out) = args.get("jsonl") {
        std::fs::write(out, to_jsonl(&hands)).map_err(|e| format!("{out}: {e}"))?;
        eprintln!("wrote {} hands to {out}", hands.len());
    }
    Ok(json!({ "summary": summarise(&hands), "issues": hands.iter().filter_map(|h| h.derived.as_ref().filter(|d| !d.ok).map(|d| json!({ "id": h.id, "issues": d.issues }))).take(50).collect::<Vec<_>>() }))
}

fn sim(args: &Args) -> Result<Value, String> {
    let cfg = pe_hh::SimConfig {
        hands: args.num("hands", 10_000)?,
        seats: args.num("seats", 6)?,
        bots: args.num("bots", 24)?,
        seed: args.num("seed", 1)?,
        straddle_rate: args.num("straddle", 0.0)?,
        mississippi_rate: args.num("mississippi", 0.0)?,
        bomb_pot_every: args.num("bomb-every", 0)?,
        double_board_bombs: args.flag("double-board"),
        run_twice_rate: args.num("run-twice", 0.0)?,
        rake_pct: args.num("rake", 5.0)?,
        ..Default::default()
    };
    let sw = pe_cards::sys::Stopwatch::start();
    let mut out = simulate(&cfg);
    let sim_cpu = sw.thread_cpu();
    out.hands.par_iter_mut().for_each(analyse);
    let dir = args.str_or("dir", "data/synthetic");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    std::fs::write(format!("{dir}/hands.jsonl"), to_jsonl(&out.hands)).map_err(|e| e.to_string())?;
    let text: String = out.hands.iter().map(|h| write_pokerstars(h) + "\n").collect();
    std::fs::write(format!("{dir}/hands.txt"), text).map_err(|e| e.to_string())?;
    std::fs::write(format!("{dir}/players.json"), serde_json::to_string_pretty(&out.bots).unwrap()).map_err(|e| e.to_string())?;
    std::fs::write(format!("{dir}/config.json"), serde_json::to_string_pretty(&cfg).unwrap()).map_err(|e| e.to_string())?;
    Ok(json!({ "dir": dir, "config": cfg, "summary": summarise(&out.hands), "simulate_cpu_seconds": sim_cpu, "total_cpu_seconds": sw.process_cpu() }))
}

fn summary(args: &Args) -> Result<Value, String> {
    let path = args.positional.get(1).ok_or("give a file")?;
    let hands = load(path)?;
    Ok(serde_json::to_value(summarise(&hands)).unwrap())
}
