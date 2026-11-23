use super::*;
use games::{Clairvoyance, Kuhn, Leduc};

#[test]
fn tree_sizes() {
    let k2 = Tree::compile(&Kuhn::new(2));
    assert_eq!(k2.infosets.len(), 12);
    assert!(k2.is_zero_sum());
    // 288 with suit isomorphism (keys use ranks); OpenSpiel's 936 distinguishes suits
    let l2 = Tree::compile(&Leduc::new(2));
    assert_eq!(l2.infosets.len(), 288);
    assert!(l2.is_zero_sum());
}

/// An independent, deliberately plain CFR straight over the game definition
/// (states cloned, information sets in a HashMap), used to cross-check the
/// compiled-tree solver.
fn reference_cfr<G: GameDef>(def: &G, iters: usize, plus: bool) -> std::collections::HashMap<String, Vec<f64>> {
    use std::collections::HashMap;
    struct R {
        regret: HashMap<String, Vec<f64>>,
        sum: HashMap<String, Vec<f64>>,
    }
    fn sigma(r: &R, key: &str, k: usize) -> Vec<f64> {
        match r.regret.get(key) {
            None => vec![1.0 / k as f64; k],
            Some(v) => {
                let pos: Vec<f64> = v.iter().map(|x| x.max(0.0)).collect();
                let t: f64 = pos.iter().sum();
                if t > 0.0 { pos.iter().map(|x| x / t).collect() } else { vec![1.0 / k as f64; k] }
            }
        }
    }
    fn walk<G: GameDef>(def: &G, s: &G::State, i: usize, pi_i: f64, pi_o: f64, r: &mut R, t: f64, plus: bool, upd: &mut Vec<(String, Vec<f64>, Vec<f64>)>) -> f64 {
        match def.spec(s) {
            NodeSpec::Terminal(u) => u[i],
            NodeSpec::Chance(p) => p.iter().enumerate().map(|(a, q)| if *q > 0.0 { q * walk(def, &def.next(s, a), i, pi_i, pi_o * q, r, t, plus, upd) } else { 0.0 }).sum(),
            NodeSpec::Decision { player, infoset, actions } => {
                let k = actions.len();
                let sg = sigma(r, &infoset, k);
                if player == i {
                    let vals: Vec<f64> = (0..k).map(|a| walk(def, &def.next(s, a), i, pi_i * sg[a], pi_o, r, t, plus, upd)).collect();
                    let v: f64 = (0..k).map(|a| sg[a] * vals[a]).sum();
                    upd.push((infoset, (0..k).map(|a| pi_o * (vals[a] - v)).collect(), (0..k).map(|a| pi_i * sg[a]).collect()));
                    v
                } else {
                    // every branch is walked (no pruning), so strategy sums are exact
                    (0..k).map(|a| sg[a] * walk(def, &def.next(s, a), i, pi_i, pi_o * sg[a], r, t, plus, upd)).sum()
                }
            }
        }
    }
    let mut r = R { regret: HashMap::new(), sum: HashMap::new() };
    for it in 1..=iters {
        for i in 0..def.num_players() {
            let mut upd = Vec::new();
            walk(def, &def.initial(), i, 1.0, 1.0, &mut r, it as f64, plus, &mut upd);
            for (key, reg, st) in upd {
                let k = reg.len();
                let e = r.regret.entry(key.clone()).or_insert(vec![0.0; k]);
                for a in 0..k {
                    e[a] += reg[a];
                    if plus {
                        e[a] = e[a].max(0.0);
                    }
                }
                let w = if plus { it as f64 } else { 1.0 };
                let e = r.sum.entry(key).or_insert(vec![0.0; k]);
                for a in 0..k {
                    e[a] += w * st[a];
                }
            }
        }
    }
    r.sum
}

#[test]
fn compiled_cfr_matches_independent_reference() {
    // vanilla CFR on Leduc: the compiled-tree solver and the plain reference must agree
    let g = Leduc::new(2);
    let t = Tree::compile(&g);
    for iters in [10usize, 40] {
        let sums = reference_cfr(&g, iters, false);
        let mut probs = vec![0.0; t.abs_actions];
        for is in &t.infosets {
            let a = &t.abs_infosets[is.abs as usize];
            let s = &sums[&is.key];
            let tot: f64 = s.iter().sum();
            for k in 0..a.num_actions {
                probs[a.offset + k] = if tot > 0.0 { s[k] / tot } else { 1.0 / a.num_actions as f64 };
            }
        }
        let reference = nash_conv(&t, &Strategy { probs }).exploitability.unwrap();
        let (_, mine) = run_cfr(&t, Algorithm::Cfr, &[iters as u64]);
        let compiled = mine.curve[0].exploitability.unwrap();
        assert!((reference - compiled).abs() < 1e-12, "{iters}: reference {reference} compiled {compiled}");
    }
}

#[test]
#[ignore]
fn debug_leduc_current_vs_average() {
    let t = Tree::compile(&Leduc::new(2));
    for algo in [Algorithm::CfrPlus, Algorithm::Linear, Algorithm::DCFR, Algorithm::PredictiveCfrPlus] {
        let mut s = Cfr::new(&t, algo);
        let mut line = String::new();
        for mark in [100u64, 1000, 5000] {
            while s.iterations < mark {
                s.iterate();
            }
            let avg = nash_conv(&t, &s.average_strategy()).exploitability.unwrap();
            let cur = nash_conv(&t, &s.current_strategy()).exploitability.unwrap();
            line += &format!(" {mark}: avg {avg:.5} cur {cur:.5} |");
        }
        println!("{}{line}", algo.name());
    }
}

#[test]
#[ignore]
fn debug_leduc_curves() {
    let t = Tree::compile(&Leduc::new(2));
    for algo in [Algorithm::Cfr, Algorithm::CfrPlus, Algorithm::Linear, Algorithm::DCFR, Algorithm::PredictiveCfrPlus] {
        let (_, rep) = run_cfr(&t, algo, &[10, 100, 1000]);
        let c: Vec<String> = rep.curve.iter().map(|p| format!("{}:{:.5}", p.iteration, p.exploitability.unwrap())).collect();
        println!("{} {}", algo.name(), c.join(" "));
    }
    let (_, rep) = run_mccfr(&t, Sampling::External, &[1000, 10000, 100000], 1);
    let c: Vec<String> = rep.curve.iter().map(|p| format!("{}:{:.5}", p.iteration, p.exploitability.unwrap())).collect();
    println!("ES {}", c.join(" "));
}

#[test]
fn uniform_exploitability_matches_known_values() {
    // OpenSpiel's reference values for the uniform random policy
    let k2 = Tree::compile(&Kuhn::new(2));
    let r = nash_conv(&k2, &k2.uniform());
    assert!((r.exploitability.unwrap() - 11.0 / 24.0).abs() < 1e-12, "{r:?}");
    // OpenSpiel reports NashConv 4.747222... for uniform Leduc
    let l2 = Tree::compile(&Leduc::new(2));
    let r = nash_conv(&l2, &l2.uniform());
    assert!((r.nash_conv - 4.747_222_222_222_222).abs() < 1e-9, "{r:?}");
}

#[test]
fn kuhn_converges_to_known_value() {
    let t = Tree::compile(&Kuhn::new(2));
    for algo in [Algorithm::Cfr, Algorithm::CfrPlus, Algorithm::Linear, Algorithm::DCFR, Algorithm::PredictiveCfrPlus] {
        let (s, rep) = run_cfr(&t, algo, &[2000]);
        let last = rep.curve.last().unwrap();
        assert!(last.exploitability.unwrap() < 2e-3, "{} {:?}", algo.name(), last);
        let v = expected_values(&t, &s)[0];
        assert!((v + 1.0 / 18.0).abs() < 3e-3, "{} value {v}", algo.name());
    }
}

#[test]
fn leduc_converges_to_known_value() {
    let t = Tree::compile(&Leduc::new(2));
    let (s, rep) = run_cfr(&t, Algorithm::DCFR, &[1000]);
    let e = rep.curve.last().unwrap().exploitability.unwrap();
    assert!(e < 5e-3, "DCFR exploitability {e}");
    let v = expected_values(&t, &s)[0];
    assert!((v + 0.0856).abs() < 0.01, "value {v}");
}

#[test]
fn clairvoyance_matches_closed_form() {
    let g = Clairvoyance { v: 0.5, b: 1.0 };
    let (bluff, call, value) = g.solution();
    assert!((bluff - 0.5).abs() < 1e-12 && (call - 0.5).abs() < 1e-12 && (value - 0.25).abs() < 1e-12);
    let t = Tree::compile(&g);
    let (s, _) = run_cfr(&t, Algorithm::CfrPlus, &[5000]);
    let air = t.infosets.iter().position(|i| i.key == "0:air").unwrap() as u32;
    let caller = t.infosets.iter().position(|i| i.key == "1:bet").unwrap() as u32;
    assert!((t.probs_at(&s, air)[1] - bluff).abs() < 0.01);
    assert!((t.probs_at(&s, caller)[1] - call).abs() < 0.01);
    assert!((expected_values(&t, &s)[0] - value).abs() < 1e-3);
}

#[test]
fn monte_carlo_cfr_converges_on_kuhn() {
    let t = Tree::compile(&Kuhn::new(2));
    let (_, es) = run_mccfr(&t, Sampling::External, &[20_000], 1);
    assert!(es.curve.last().unwrap().exploitability.unwrap() < 0.01, "{:?}", es.curve);
    let (_, os) = run_mccfr(&t, Sampling::Outcome { epsilon: 0.6 }, &[200_000], 2);
    assert!(os.curve.last().unwrap().exploitability.unwrap() < 0.03, "{:?}", os.curve);
}

#[test]
fn three_player_kuhn_nash_conv_falls() {
    let t = Tree::compile(&Kuhn::new(3));
    let start = nash_conv(&t, &t.uniform()).nash_conv;
    let (_, rep) = run_cfr(&t, Algorithm::Cfr, &[2000]);
    let end = rep.curve.last().unwrap();
    assert!(end.exploitability.is_none(), "no two-player exploitability for three players");
    assert!(end.nash_conv < start / 20.0, "{start} -> {}", end.nash_conv);
}

#[test]
fn abstraction_is_evaluated_in_the_real_game() {
    // merge jacks and queens (ranks 0 and 1) in every information set
    let g = Leduc::new(2);
    let abs = Tree::compile_abstracted(&g, |k: &str| {
        let mut parts: Vec<String> = k.split('|').map(|s| s.to_string()).collect();
        if parts[1] == "0" {
            parts[1] = "1".into();
        }
        if parts[2] == "0" {
            parts[2] = "1".into();
        }
        parts.join("|")
    });
    assert!(abs.abs_infosets.len() < abs.infosets.len());
    let (s, rep) = run_cfr(&abs, Algorithm::DCFR, &[500]);
    let e = rep.curve.last().unwrap().exploitability.unwrap();
    let full = Tree::compile(&g);
    let (_, rep_full) = run_cfr(&full, Algorithm::DCFR, &[500]);
    // the abstract strategy is clearly more exploitable in the real game
    assert!(e > 5.0 * rep_full.curve.last().unwrap().exploitability.unwrap(), "abstract {e}");
    let _ = s;
}
