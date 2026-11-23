//! Test vectors for the JavaScript engine of the web demo. The browser engine is
//! a port of the Rust code; its test suite (`web/tests/run.mjs`) checks it
//! against these files, so both implementations must agree on evaluations,
//! equities, trees, exploitability and strategies.

use pe_cards::{Card, Game, Rng, cards_of, cards_to_string, mask_of, parse_cards};
use pe_cfr::Algorithm;
use pe_solver::{BetMenu, HuSolver, Spot, SpotConfig};
use serde_json::{Value, json};

use crate::args::Args;

fn write(dir: &str, name: &str, v: &Value) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let path = format!("{dir}/{name}");
    std::fs::write(&path, serde_json::to_string(v).unwrap()).map_err(|e| format!("{path}: {e}"))
}

fn draw(rng: &mut Rng, deck: &[Card], k: usize) -> Vec<Card> {
    let mut v = deck.to_vec();
    for i in 0..k {
        let j = i + rng.below((v.len() - i) as u64) as usize;
        v.swap(i, j);
    }
    v.truncate(k);
    v
}

pub fn solver_cases() -> Vec<(&'static str, SpotConfig, u64)> {
    let menu = BetMenu { bets: vec![0.33, 0.75], raises: vec![1.0], max_raises: 2, allin_threshold: 0.67, add_allin: false };
    let base = |game: &str, board: &str, r0: &str, r1: &str| SpotConfig {
        game: game.into(),
        board: board.into(),
        board2: None,
        ranges: vec![r0.into(), r1.into()],
        pot: 10.0,
        stacks: vec![40.0],
        menu: menu.clone(),
        rake: None,
        side_rule: None,
        dead: None,
    };
    let mut seven_deuce = base("NLH", "Ts9c4h3d3s", "AA,KK,AQs,72o,64s,53s", "QQ-TT,AJs,KQs,98s");
    seven_deuce.side_rule = Some(pe_solver::SideRule::SevenDeuce { amount: 5.0 });
    let mut raked = base("NLH", "Qs8h5d3c2s", "QQ+,AQs+,AKo,QJs,JTs,T9s,98s,A5s,76s", "JJ-88,AQ,KQ,QJ,QT,J9s,T9s");
    raked.rake = Some(pe_solver::Rake { percent: 5.0, cap: 3.0 });
    let mut double = base("NLH", "Ks7d2c9h4s", "AA,KQ,JTs,99", "QQ,K9s,AK,T8s");
    double.board2 = Some("As8s3d5c6h".into());
    vec![
        ("nlh-river", base("NLH", "Qs8h5d3c2s", "QQ+,AQs+,AKo,QJs,JTs,T9s,98s,A5s,76s,65s", "JJ-88,AQ,KQ,QJ,QT,J9s,T9s"), 200),
        ("nlh-turn", base("NLH", "Ks7d2c9h", "AA,KQs,98s,T8s", "QQ,K9s,T9s,JTs"), 30),
        ("plo-river", base("PLO", "Ah9h4d2cTs", "AAKx:0.2,JJT9,KQJ9", "QJ87,KKxx:0.05,9876"), 100),
        ("sd-river", base("SD", "Ah9d8c7s6h", "TT+,AK,AQ,KQ,JT", "99+,AJ,KJ,QJ,T9s,87s"), 100),
        ("o8-river", base("O8", "Ah4d7c9sKs", "A2xxds:0.5,A3xxds:0.5,KKQJ", "AKxxds:0.3,23xxds,QJT9"), 100),
        ("double-board", double, 100),
        ("seven-deuce", seven_deuce, 200),
        ("raked", raked, 200),
    ]
}

pub fn run(args: &Args) -> Result<Value, String> {
    let dir = args.str_or("dir", "research/vectors");
    let mut rng = Rng::new(2026);
    let deck: Vec<Card> = (0..52).map(Card).collect();
    let short = cards_of(pe_cards::SHORT_DECK);
    // evaluations
    let mut evals = Vec::new();
    for _ in 0..2000 {
        let c = draw(&mut rng, &deck, 7);
        evals.push(json!({ "rules": "standard", "cards": cards_to_string(&c), "value": pe_eval::eval7(mask_of(&c)) }));
    }
    for _ in 0..500 {
        let c = draw(&mut rng, &deck, 5);
        evals.push(json!({ "rules": "standard", "cards": cards_to_string(&c), "value": pe_eval::eval7(mask_of(&c)) }));
    }
    for rules in [pe_cards::HighRules::ShortDeckTriton, pe_cards::HighRules::ShortDeckClassic] {
        for _ in 0..500 {
            let c = draw(&mut rng, &short, 7);
            let name = if rules == pe_cards::HighRules::ShortDeckTriton { "triton" } else { "classic" };
            evals.push(json!({ "rules": name, "cards": cards_to_string(&c), "value": pe_eval::eval_high(mask_of(&c), rules) }));
        }
    }
    let mut omaha = Vec::new();
    for _ in 0..500 {
        let c = draw(&mut rng, &deck, 9);
        omaha.push(json!({
            "hole": cards_to_string(&c[..4]), "board": cards_to_string(&c[4..]),
            "high": pe_eval::eval_omaha_high(&c[..4], &c[4..], pe_cards::HighRules::Standard),
            "low": pe_eval::eval_omaha_low8(&c[..4], &c[4..]),
        }));
    }
    write(&dir, "eval.json", &json!({ "hands": evals, "omaha": omaha }))?;
    // equities
    let h = |s: &str| parse_cards(s).unwrap();
    let cases: Vec<(&str, Game, Vec<Vec<Card>>, Vec<Vec<Card>>)> = vec![
        ("NLH", Game::NLH, vec![h("AhKh"), h("QsQd")], vec![h("Th7h2c")]),
        ("NLH", Game::NLH, vec![h("AhKh"), h("QsQd"), h("JcTc")], vec![h("Th7h2c")]),
        ("NLH", Game::NLH, vec![h("AhKh"), h("QsQd")], vec![h("Th7h2c3d")]),
        ("PLO", Game::PLO, vec![h("AhAdKhQd"), h("JsTs9c8c")], vec![h("7s6h2d")]),
        ("SD", Game::SHORT_DECK, vec![h("AhKh"), h("9s8s")], vec![h("Th7c6d")]),
        ("O8", Game::O8, vec![h("As2sKdKc"), h("QhQdJhJd")], vec![h("3c4d9h")]),
        ("NLH", Game::NLH, vec![h("AhKh"), h("QsQd")], vec![h("Th7h2c"), h("9s8s3d")]),
    ];
    let mut eq = Vec::new();
    for (code, game, hands, boards) in &cases {
        let r = pe_equity::equity_exact(game, hands, boards, &[])?;
        eq.push(json!({
            "game": code, "hands": hands.iter().map(|x| cards_to_string(x)).collect::<Vec<_>>(),
            "boards": boards.iter().map(|x| cards_to_string(x)).collect::<Vec<_>>(), "equity": r.equity, "runouts": r.runouts,
        }));
    }
    let rit = pe_equity::rit::run_it_twice_exact(&Game::NLH, &[h("AhKh"), h("QsQd")], &h("Th7h2c"), &[])?;
    write(&dir, "equity.json", &json!({ "cases": eq, "rit": { "hands": ["AhKh", "QsQd"], "board": "Th7h2c", "result": rit } }))?;
    // preflop order
    let order: Vec<String> = pe_cards::ranking::preflop_order().into_iter().map(pe_cards::class_name).collect();
    write(&dir, "preflop_ranking.json", &json!(order))?;
    // solver
    let mut solves = Vec::new();
    for (name, cfg, iters) in solver_cases() {
        let spot = Spot::from_config(cfg.clone())?;
        let mut s = HuSolver::new(spot, Algorithm::DCFR, None, 1_000_000)?;
        for _ in 0..iters {
            s.iterate();
        }
        let e = s.exploitability();
        let root = pe_solver::node_report(&s, 0)?;
        let sample: Vec<Value> = root
            .hands
            .iter()
            .step_by((root.hands.len() / 12).max(1))
            .map(|r| json!({ "hand": r.hand, "probs": r.probs, "evs": r.evs }))
            .collect();
        solves.push(json!({
            "name": name, "config": cfg, "algorithm": "DCFR", "iterations": iters,
            "nodes": s.tree.nodes.len(), "decision_nodes": s.tree.decision_nodes(), "hands": [s.hands(0), s.hands(1)],
            "ev": e.ev, "br": e.br, "pct_pot": e.pct_pot,
            "root_actions": root.actions, "root_frequencies": root.frequencies, "root_sample": sample,
        }));
    }
    write(&dir, "solver.json", &json!(solves))?;
    Ok(json!({ "wrote": dir, "files": ["eval.json", "equity.json", "preflop_ranking.json", "solver.json"] }))
}
