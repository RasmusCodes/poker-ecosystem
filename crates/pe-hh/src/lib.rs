//! Hand histories: the `pe-hh/1` model, a PokerStars parser and writer, pot
//! reconstruction, all-in expected value, river showdown strength, a simulator
//! for synthetic corpora with ground truth, street-by-street ranges
//! ([`ranges`]) and an automatic review of heads-up river decisions against an
//! equilibrium ([`review`]).
//!
//! The Bayesian opponent model with credible intervals, clustering and the
//! hand database live in the Python package (`python/poker_ecosystem`), which
//! reads the JSON this crate writes.

pub mod allin;
pub mod model;
pub mod pokerstars;
pub mod ranges;
pub mod reconstruct;
pub mod review;
pub mod sim;
pub mod writer;

pub use allin::{annotate_allin, annotate_river_strength, river_strength};
pub use model::{Hand, read_file, read_hands, to_jsonl};
pub use pokerstars::parse_pokerstars;
pub use ranges::{FrequencyTable, HandRanges, RangeOptions, reconstruct_ranges};
pub use reconstruct::reconstruct;
pub use review::{ReviewOptions, RiverReview, review_hands, review_river};
pub use sim::{BotInfo, SimConfig, SimOutput, simulate};
pub use writer::write_pokerstars;

/// Reconstructs a hand and adds all-in equity and river strength.
pub fn analyse(hand: &mut Hand) {
    reconstruct(hand);
    annotate_allin(hand);
    annotate_river_strength(hand);
}

/// Counts for a short command-line summary.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Summary {
    pub hands: usize,
    pub reconstructed_ok: usize,
    pub with_issues: usize,
    pub players: usize,
    pub showdowns: usize,
    pub allins: usize,
    pub run_twice: usize,
    pub double_board: usize,
    pub bomb_pots: usize,
    pub games: std::collections::BTreeMap<String, usize>,
}

pub fn summarise(hands: &[Hand]) -> Summary {
    let mut s = Summary { hands: hands.len(), ..Default::default() };
    let mut names = std::collections::BTreeSet::new();
    for h in hands {
        for p in &h.players {
            names.insert(p.name.clone());
        }
        *s.games.entry(h.game.clone()).or_insert(0) += 1;
        if let Some(d) = &h.derived {
            if d.ok {
                s.reconstructed_ok += 1;
            } else {
                s.with_issues += 1;
            }
            if !d.showdown_seats.is_empty() {
                s.showdowns += 1;
            }
            if d.allin.is_some() {
                s.allins += 1;
            }
        }
        match h.board_kind.as_str() {
            "run_twice" => s.run_twice += 1,
            "double" => s.double_board += 1,
            _ => {}
        }
        if h.bomb_pot {
            s.bomb_pots += 1;
        }
    }
    s.players = names.len();
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(hands: usize) -> SimConfig {
        SimConfig {
            hands,
            seed: 7,
            straddle_rate: 0.1,
            mississippi_rate: 0.1,
            bomb_pot_every: 25,
            double_board_bombs: true,
            run_twice_rate: 0.3,
            rake_pct: 5.0,
            ..Default::default()
        }
    }

    /// The first `k` preflop actions follow `model::preflop_order`.
    fn follows_preflop_order(h: &Hand) -> bool {
        let order = model::preflop_order(h);
        match h.street(model::StreetName::Preflop) {
            Some(pre) => {
                let k = pre.actions.len().min(order.len());
                pre.actions.iter().take(k).map(|a| a.seat).eq(order[..k].iter().copied())
            }
            None => true,
        }
    }

    #[test]
    fn mississippi_straddles_act_last_before_the_flop() {
        let out = simulate(&SimConfig { hands: 400, seed: 11, mississippi_rate: 1.0, ..Default::default() });
        let mut seen = 0;
        for h in &out.hands {
            let m: Vec<&model::Post> = h.posts.iter().filter(|p| p.kind == model::PostKind::Mississippi).collect();
            assert_eq!(m.len(), 1, "hand {}", h.id);
            let button = model::clockwise_from_button(&h.dealt_seats(), h.button)[0];
            assert_eq!(m[0].seat, button, "hand {}: the button straddles", h.id);
            // under the gun acts first, the cutoff passes to the small blind, the button acts last
            let order = model::preflop_order(h);
            assert_eq!(*order.last().unwrap(), button);
            assert!(follows_preflop_order(h), "hand {}: {:?} vs {:?}", h.id, h.streets[0].actions.iter().map(|a| a.seat).collect::<Vec<_>>(), order);
            // the written text carries only "posts straddle": the parser recovers the kind from the seat
            let (parsed, warnings) = parse_pokerstars(&write_pokerstars(h));
            assert!(warnings.is_empty(), "{warnings:?}");
            assert_eq!(parsed[0].posts, h.posts);
            let mut a = parsed[0].clone();
            analyse(&mut a);
            assert!(a.derived.as_ref().unwrap().ok, "hand {}: {:?}", h.id, a.derived.as_ref().unwrap().issues);
            seen += 1;
        }
        assert_eq!(seen, 400);
    }

    #[test]
    fn parser_tells_straddle_kinds_apart() {
        let text = "PokerStars Hand #9100:  Hold'em No Limit ($1/$2 USD) - 2026/11/05 21:00:00 ET
Table 'T' 6-max Seat #1 is the button
Seat 1: A ($200 in chips)
Seat 2: B ($200 in chips)
Seat 3: C ($200 in chips)
Seat 4: D ($200 in chips)
Seat 5: E ($200 in chips)
B: posts small blind $1
C: posts big blind $2
A: posts straddle $4
*** HOLE CARDS ***
D: folds
E: calls $4
B: folds
C: folds
A: checks
*** FLOP *** [2c 7d 9h]
E: checks
A: checks
*** TURN *** [2c 7d 9h] [3s]
E: checks
A: checks
*** RIVER *** [2c 7d 9h 3s] [4d]
E: checks
A: checks
*** SHOW DOWN ***
E: shows [Ah As]
A: shows [Kd Kc]
E collected $11 from pot
*** SUMMARY ***
Total pot $11 | Rake $0
Board [2c 7d 9h 3s 4d]
";
        let (hands, warnings) = parse_pokerstars(text);
        assert!(warnings.is_empty(), "{warnings:?}");
        let h = &hands[0];
        assert_eq!(h.posts[2].kind, model::PostKind::Mississippi, "the button's straddle");
        assert_eq!(model::preflop_order(h), vec![4, 5, 2, 3, 1]);
        assert!(follows_preflop_order(h));
        // the same post from under the gun is an ordinary straddle (only the post matters here)
        let (hands, _) = parse_pokerstars(&text.replace("A: posts straddle $4", "D: posts straddle $4"));
        assert_eq!(hands[0].posts[2].kind, model::PostKind::Straddle);
    }

    #[test]
    fn simulated_hands_reconstruct_and_round_trip() {
        let out = simulate(&config(2500));
        assert_eq!(out.hands.len(), 2500);
        let mut special = (0, 0, 0, 0);
        for h in &out.hands {
            let mut a = h.clone();
            reconstruct(&mut a);
            let d = a.derived.as_ref().unwrap();
            assert!(d.ok, "hand {} does not reconstruct: {:?}", h.id, d.issues);
            // text round trip: everything the text format carries survives
            let text = write_pokerstars(h);
            let (parsed, warnings) = parse_pokerstars(&text);
            assert!(warnings.is_empty(), "hand {}: {warnings:?}\n{text}", h.id);
            assert_eq!(parsed.len(), 1);
            let p = &parsed[0];
            assert_eq!(p.posts, h.posts, "{text}");
            let strip = |h: &Hand| -> Vec<Vec<(u32, model::ActionType, i64, i64, bool)>> {
                h.streets.iter().map(|s| s.actions.iter().map(|a| (a.seat, a.kind, a.amount, a.to, a.all_in)).collect()).collect()
            };
            assert_eq!(strip(p), strip(h), "hand {}\n{text}", h.id);
            assert_eq!(p.boards, h.boards, "{text}");
            assert_eq!(p.total_pot, h.total_pot);
            assert_eq!(p.rake, h.rake);
            assert_eq!(p.uncalled, h.uncalled);
            let mut c1: Vec<_> = p.collected.iter().map(|c| (c.seat, c.amount, c.pot, c.board)).collect();
            let mut c2: Vec<_> = h.collected.iter().map(|c| (c.seat, c.amount, c.pot, c.board)).collect();
            c1.sort();
            c2.sort();
            assert_eq!(c1, c2, "{text}");
            let mut pp = p.clone();
            reconstruct(&mut pp);
            assert!(pp.derived.as_ref().unwrap().ok, "parsed hand {} does not reconstruct", h.id);
            match h.board_kind.as_str() {
                "run_twice" => special.0 += 1,
                "double" => special.1 += 1,
                _ => {}
            }
            if h.posts.iter().any(|p| p.kind == model::PostKind::Straddle) {
                special.2 += 1;
            }
            if h.posts.iter().any(|p| p.kind == model::PostKind::Mississippi) {
                special.3 += 1;
            }
            assert!(follows_preflop_order(h), "hand {}", h.id);
        }
        assert!(special.0 > 0 && special.1 > 0 && special.2 > 0 && special.3 > 0, "{special:?}");
    }

    #[test]
    fn simulator_is_deterministic_and_truth_is_valid() {
        let a = simulate(&config(300));
        let b = simulate(&config(300));
        assert_eq!(to_jsonl(&a.hands), to_jsonl(&b.hands));
        for h in &a.hands {
            for s in &h.streets {
                for act in &s.actions {
                    let t = act.truth.as_ref().expect("synthetic actions carry truth");
                    let sum: f64 = t.values().sum();
                    assert!((sum - 1.0).abs() < 1e-9, "{t:?}");
                    assert!(t.get(act.kind.as_str()).copied().unwrap_or(0.0) > 0.0, "sampled an action with zero probability: {t:?}");
                }
            }
        }
    }

    #[test]
    fn allin_ev_on_a_known_matchup() {
        let text = "PokerStars Hand #9001:  Hold'em No Limit ($1/$2 USD) - 2026/11/05 21:00:00 ET
Table 'T' 2-max Seat #1 is the button
Seat 1: A ($100 in chips)
Seat 2: B ($100 in chips)
A: posts small blind $1
B: posts big blind $2
*** HOLE CARDS ***
A: raises $98 to $100 and is all-in
B: calls $98 and is all-in
*** FLOP *** [2c 7d 9h]
*** TURN *** [2c 7d 9h] [3s]
*** RIVER *** [2c 7d 9h 3s] [4d]
*** SHOW DOWN ***
A: shows [Ah As]
B: shows [Kd Kc]
A collected $200 from pot
*** SUMMARY ***
Total pot $200 | Rake $0
Board [2c 7d 9h 3s 4d]
";
        let (mut hands, warnings) = parse_pokerstars(text);
        assert!(warnings.is_empty(), "{warnings:?}");
        let h = &mut hands[0];
        analyse(h);
        let d = h.derived.as_ref().unwrap();
        assert!(d.ok, "{:?}", d.issues);
        let ai = d.allin.as_ref().expect("an all-in");
        let eq = ai.equity[&1];
        assert!((eq - 0.82).abs() < 0.01, "{eq}");
        // A's expected net is equity * 200 - 100
        assert!((ai.ev_net[&1] as f64 - (eq * 200_000.0 - 100_000.0)).abs() < 2.0);
        assert!(d.river_strength.contains_key(&1));
    }

    #[test]
    fn sample_file_reconstructs() {
        let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/samples/pokerstars-sample.txt")).unwrap();
        let (mut hands, warnings) = parse_pokerstars(&text);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert!(hands.len() >= 10);
        for h in hands.iter_mut() {
            analyse(h);
            let d = h.derived.as_ref().unwrap();
            assert!(d.ok, "hand {}: {:?}", h.id, d.issues);
        }
    }
}
