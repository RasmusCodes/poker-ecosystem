//! Automatic river review: each heads-up river decision of a hand compared with
//! an equilibrium of the river as the hand reached it.
//!
//! For a hold'em hand where exactly two players see the river and at least one
//! acts on it, the review:
//!
//! 1. takes both players' ranges at the start of the river from
//!    [`crate::ranges`] (combos below `prune` times the most likely combo are
//!    left out; the players' actual hands always stay in);
//! 2. builds the river spot from the hand (board, pot and stacks in big blinds)
//!    with a bet menu of 33% and 75% of the pot plus every size actually used,
//!    pot-sized raises plus the raises used, and all-in;
//! 3. solves it with Discounted CFR to `target_pct` of the pot (exploitability
//!    computed exactly) or `max_iterations`;
//! 4. walks the actions taken through the tree. For each decision of a player
//!    whose cards are known it reports that hand's equilibrium strategy and the
//!    value of every action when both players play the equilibrium afterwards.
//!    The gap between the best action and the one taken is the decision's cost.
//!
//! The cost is measured inside a model: the ranges are estimates from observed
//! frequencies, the menu is an abstraction, and the opponent is assumed to play
//! the equilibrium rather than their actual strategy. A large cost means the
//! action is clearly worse than an alternative against a balanced opponent with
//! these ranges, not that it lost money against this opponent.

use std::collections::BTreeSet;

use pe_cards::{HandList, NUM_COMBOS, cards_to_string, combo_cards, combo_index, combo_mask, mask_of};
use pe_cfr::Algorithm;
use pe_solver::{Action, BetMenu, HuSolver, NodeKind, Spot, SpotConfig, solve_hu};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::model::{ActionType, Hand, StreetName};
use crate::ranges::{FrequencyTable, RangeOptions, decisions, is_holdem, reconstruct_ranges};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReviewOptions {
    pub max_iterations: u64,
    /// Stop once exploitability is at most this percentage of the pot.
    pub target_pct: f64,
    pub check_every: u64,
    /// Combos below this share of the most likely combo are left out of the solve.
    pub prune: f64,
}

impl Default for ReviewOptions {
    fn default() -> Self {
        ReviewOptions { max_iterations: 600, target_pct: 0.5, check_every: 50, prune: 1e-4 }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ReviewedDecision {
    /// Position of the action in the hand (streets concatenated).
    pub index: usize,
    pub seat: u32,
    pub name: String,
    /// Labels of the tree's actions at this decision.
    pub actions: Vec<String>,
    pub taken: usize,
    /// Whether the tree has the exact size used (always true for fold, check, call).
    pub size_matched: bool,
    /// The whole range's action frequencies at the equilibrium.
    pub range_frequencies: Vec<f64>,
    /// The actual hand, when known, with its equilibrium strategy, the value of
    /// each action in big blinds, and the cost of the action taken.
    pub hand: Option<String>,
    pub probs: Vec<f64>,
    pub evs: Vec<f64>,
    pub best: Option<usize>,
    pub frequency: Option<f64>,
    pub ev_loss_bb: Option<f64>,
    /// Probability that the equilibrium, holding this hand, takes the player's
    /// earlier river actions of this hand and so reaches this decision. Below a
    /// few percent the decision is off the equilibrium path for this hand: its
    /// frequency there means little, while the values and the cost still compare
    /// the actions from here.
    pub path_probability: Option<f64>,
    /// Probability that the opponent's equilibrium range takes the opponent's
    /// river actions so far. Near zero, the decision answers a line the
    /// equilibrium hardly ever takes: the solver explores it little, so its
    /// strategy and values there are rough.
    pub opponent_path_probability: f64,
}

/// A decision both players reach with some regularity at the equilibrium
/// (the hand's own line at least 5%, the opponent's range at least 2%).
pub fn on_path(d: &ReviewedDecision) -> bool {
    d.path_probability.is_some_and(|p| p >= 0.05) && d.opponent_path_probability >= 0.02
}

#[derive(Clone, Debug, Serialize)]
pub struct RiverReview {
    pub id: String,
    pub board: String,
    pub pot_bb: f64,
    /// Out of position first, in position second.
    pub stacks_bb: [f64; 2],
    pub seats: [u32; 2],
    pub names: [String; 2],
    /// Combos in each solved range and their effective number (exp(entropy)).
    pub combos: [usize; 2],
    pub effective_combos: [f64; 2],
    pub bets: Vec<f64>,
    pub raises: Vec<f64>,
    pub iterations: u64,
    pub exploitability_pct: f64,
    pub decisions: Vec<ReviewedDecision>,
    pub label: String,
}

fn combo_of(cards: &[pe_cards::Card]) -> Option<usize> {
    if cards.len() == 2 && cards[0] != cards[1] { Some(combo_index(cards[0], cards[1])) } else { None }
}

/// Reviews the river of one hand; `None` when the hand has no heads-up river
/// decision (or is not single-board hold'em).
pub fn review_river(hand: &Hand, table: &FrequencyTable, ropts: &RangeOptions, opts: &ReviewOptions) -> Result<Option<RiverReview>, String> {
    if !is_holdem(hand) || hand.board_kind != "single" || hand.bb <= 0 {
        return Ok(None);
    }
    let Some(river) = hand.street(StreetName::River) else { return Ok(None) };
    let board = hand.board_cards(0);
    if river.actions.is_empty() || board.len() != 5 {
        return Ok(None);
    }
    let ds = decisions(hand);
    let Some(first) = ds.iter().find(|d| d.street == StreetName::River) else { return Ok(None) };
    let folded: BTreeSet<u32> = ds.iter().filter(|d| d.street < StreetName::River && d.action == ActionType::Fold).map(|d| d.seat).collect();
    let alive: Vec<u32> = hand.dealt_seats().into_iter().filter(|s| !folded.contains(s)).collect();
    if alive.len() != 2 {
        return Ok(None);
    }
    let oop = first.seat;
    let ip = if alive[0] == oop { alive[1] } else { alive[0] };
    let seats = [oop, ip];
    let bb = hand.bb as f64;
    // stacks at the start of the river
    let stack_at = |seat: u32| -> i64 {
        let mut s = hand.player(seat).map_or(0, |p| p.stack);
        for p in hand.posts.iter().filter(|p| p.seat == seat) {
            s -= p.amount;
        }
        for st in hand.streets.iter().filter(|st| st.name < StreetName::River) {
            for a in st.actions.iter().filter(|a| a.seat == seat) {
                s -= a.amount;
            }
        }
        s
    };
    let stacks = [stack_at(oop), stack_at(ip)];
    let pot_start = first.pot_before;
    // menu: the default sizes plus the sizes used
    let mut bets = vec![0.33, 0.75];
    let mut raises = vec![1.0];
    let mut street_in = [0i64; 2];
    let mut pot_now = pot_start;
    let mut aggressions = 0usize;
    let side = |seat: u32| usize::from(seat != oop);
    for a in &river.actions {
        let p = side(a.seat);
        let max_in = street_in[0].max(street_in[1]);
        match a.kind {
            ActionType::Bet => {
                bets.push((street_in[p] + a.amount) as f64 / pot_now as f64);
                aggressions += 1;
            }
            ActionType::Raise => {
                let call = max_in - street_in[p];
                raises.push((street_in[p] + a.amount - max_in) as f64 / (pot_now + call) as f64);
                aggressions += 1;
            }
            _ => {}
        }
        street_in[p] += a.amount;
        pot_now += a.amount;
    }
    let tidy = |v: &mut Vec<f64>| {
        v.retain(|x| x.is_finite() && *x > 0.0);
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    };
    tidy(&mut bets);
    tidy(&mut raises);
    let menu = BetMenu { bets: bets.clone(), raises: raises.clone(), max_raises: aggressions.max(2), allin_threshold: 1.0, add_allin: true };
    // ranges at the start of the river
    let ranges = reconstruct_ranges(hand, table, ropts)?;
    let snap = ranges.snapshot("river").ok_or_else(|| format!("hand {}: no river snapshot", hand.id))?;
    let known: [Option<usize>; 2] = seats.map(|s| hand.hole_cards(s).and_then(|c| combo_of(&c)));
    let mut lists = Vec::new();
    let mut effective = [0.0; 2];
    for k in 0..2 {
        let sr = snap.seats.iter().find(|s| s.seat == seats[k]).ok_or_else(|| format!("hand {}: seat {} has no river range", hand.id, seats[k]))?;
        effective[k] = sr.effective_combos;
        let probs = &sr.probs;
        let mut max = 0.0f64;
        for &p in probs {
            max = max.max(p);
        }
        let mut l = HandList::default();
        for i in 0..NUM_COMBOS {
            let p = probs[i];
            if p <= 0.0 || (p < opts.prune * max && known[k] != Some(i)) {
                continue;
            }
            let (a, b) = combo_cards(i);
            l.push(vec![b, a], p);
        }
        lists.push(l);
    }
    let combos = [lists[0].len(), lists[1].len()];
    let config = SpotConfig {
        game: "NLH".into(),
        board: cards_to_string(&board),
        board2: None,
        ranges: vec!["reconstructed".into(), "reconstructed".into()],
        pot: pot_start as f64 / bb,
        stacks: vec![stacks[0] as f64 / bb, stacks[1] as f64 / bb],
        menu,
        rake: None,
        side_rule: None,
        dead: None,
    };
    let spot = Spot::from_lists(config, lists)?;
    let mut solver = HuSolver::new(spot, Algorithm::DCFR, None, 2_000_000)?;
    solve_hu(&mut solver, opts.max_iterations, opts.target_pct, opts.check_every);
    let e = solver.exploitability();
    // walk the actions taken
    let mut node = 0usize;
    let mut street_in = [0i64; 2];
    let mut path = [1.0f64; 2];
    let mut range_path = [1.0f64; 2];
    let mut out = Vec::new();
    for (j, a) in river.actions.iter().enumerate() {
        let p = side(a.seat);
        let NodeKind::Decision { player, actions, children } = &solver.tree.nodes[node].kind else {
            return Err(format!("hand {}: the river tree ends before the actions do", hand.id));
        };
        if *player != p {
            return Err(format!("hand {}: the acting order differs from the tree", hand.id));
        }
        let (taken, matched) = match a.kind {
            ActionType::Fold => (actions.iter().position(|x| matches!(x, Action::Fold)), true),
            ActionType::Check => (actions.iter().position(|x| matches!(x, Action::Check)), true),
            ActionType::Call => (actions.iter().position(|x| matches!(x, Action::Call)), true),
            ActionType::Bet | ActionType::Raise => {
                let target = (street_in[p] + a.amount) as f64 / bb;
                let mut best: Option<(usize, f64)> = None;
                for (i, x) in actions.iter().enumerate() {
                    let to = match x {
                        Action::Bet { to, .. } | Action::Raise { to, .. } | Action::AllIn { to } => *to,
                        _ => continue,
                    };
                    let d = if a.all_in && matches!(x, Action::AllIn { .. }) { -1.0 } else { (to - target).abs() };
                    if best.is_none_or(|b| d < b.1) {
                        best = Some((i, d));
                    }
                }
                match best {
                    Some((i, d)) => (Some(i), d <= 1e-6 * target.max(1.0)),
                    None => (None, false),
                }
            }
        };
        let taken = taken.ok_or_else(|| format!("hand {}: action {} is not in the river tree", hand.id, a.kind.as_str()))?;
        let k = actions.len();
        let n = solver.hands(p);
        let sigma = solver.average_strategy(node);
        let (evs, reach) = solver.action_values(node)?;
        let range_frequencies = pe_solver::profile::action_frequencies(&sigma, &reach, k);
        let mut d = ReviewedDecision {
            index: first.index + j,
            seat: a.seat,
            name: hand.name_of(a.seat).to_string(),
            actions: actions.iter().map(|x| x.label()).collect(),
            taken,
            size_matched: matched,
            range_frequencies: range_frequencies.clone(),
            hand: None,
            probs: Vec::new(),
            evs: Vec::new(),
            best: None,
            frequency: None,
            ev_loss_bb: None,
            path_probability: None,
            opponent_path_probability: range_path[1 - p],
        };
        range_path[p] *= range_frequencies[taken];
        if let Some(ci) = known[p] {
            let m = combo_mask(ci);
            if let Some(h) = solver.spot.ranges[p].hands.iter().position(|hh| mask_of(hh) == m) {
                let probs: Vec<f64> = (0..k).map(|x| sigma[x * n + h]).collect();
                let ev: Vec<f64> = (0..k).map(|x| evs[x][h]).collect();
                let mut best = 0;
                for x in 1..k {
                    if ev[x] > ev[best] {
                        best = x;
                    }
                }
                let (c1, c2) = combo_cards(ci);
                d.hand = Some(cards_to_string(&[c1, c2]));
                d.frequency = Some(probs[taken]);
                d.ev_loss_bb = Some(ev[best] - ev[taken]);
                d.best = Some(best);
                d.path_probability = Some(path[p]);
                path[p] *= probs[taken];
                d.probs = probs;
                d.evs = ev;
            }
        }
        out.push(d);
        node = children[taken];
        street_in[p] += a.amount;
    }
    Ok(Some(RiverReview {
        id: hand.id.clone(),
        board: cards_to_string(&board),
        pot_bb: pot_start as f64 / bb,
        stacks_bb: [stacks[0] as f64 / bb, stacks[1] as f64 / bb],
        seats,
        names: [hand.name_of(oop).to_string(), hand.name_of(ip).to_string()],
        combos,
        effective_combos: effective,
        bets,
        raises,
        iterations: solver.iterations,
        exploitability_pct: e.pct_pot,
        decisions: out,
        label: "Equilibrium of the river with model-based ranges and the hand's own bet sizes; costs assume an opponent playing that equilibrium".into(),
    }))
}

/// Reviews many hands in parallel (order kept); hands without a heads-up river
/// decision are skipped, failures are returned with the hand id.
pub fn review_hands(hands: &[Hand], table: &FrequencyTable, ropts: &RangeOptions, opts: &ReviewOptions) -> (Vec<RiverReview>, Vec<(String, String)>) {
    let results: Vec<(String, Result<Option<RiverReview>, String>)> = hands.par_iter().map(|h| (h.id.clone(), review_river(h, table, ropts, opts))).collect();
    let mut ok = Vec::new();
    let mut failed = Vec::new();
    for (id, r) in results {
        match r {
            Ok(Some(v)) => ok.push(v),
            Ok(None) => {}
            Err(e) => failed.push((id, e)),
        }
    }
    (ok, failed)
}

/// The reviewed decisions with a known hand, costliest first.
pub fn costliest(reviews: &[RiverReview], limit: usize) -> Vec<(String, ReviewedDecision)> {
    let mut v: Vec<(String, ReviewedDecision)> =
        reviews.iter().flat_map(|r| r.decisions.iter().filter(|d| d.ev_loss_bb.is_some()).map(move |d| (r.id.clone(), d.clone()))).collect();
    v.sort_by(|a, b| b.1.ev_loss_bb.unwrap().partial_cmp(&a.1.ev_loss_bb.unwrap()).unwrap().then(a.0.cmp(&b.0)).then(a.1.index.cmp(&b.1.index)));
    v.truncate(limit);
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_pokerstars;

    const HAND: &str = "PokerStars Hand #9200:  Hold'em No Limit ($1/$2 USD) - 2026/11/05 21:00:00 ET
Table 'T' 6-max Seat #1 is the button
Seat 1: Hero ($200 in chips)
Seat 2: C ($200 in chips)
Seat 3: Villain ($200 in chips)
C: posts small blind $1
Villain: posts big blind $2
*** HOLE CARDS ***
Dealt to Hero [Ah Kd]
Hero: raises $4 to $6
C: folds
Villain: calls $4
*** FLOP *** [Ac 8d 3s]
Villain: checks
Hero: bets $4
Villain: calls $4
*** TURN *** [Ac 8d 3s] [6h]
Villain: checks
Hero: bets $10
Villain: calls $10
*** RIVER *** [Ac 8d 3s 6h] [2c]
Villain: checks
Hero: checks
*** SHOW DOWN ***
Villain: shows [8c 8h]
Hero: shows [Ah Kd]
Villain collected $41 from pot
*** SUMMARY ***
Total pot $41 | Rake $0
Board [Ac 8d 3s 6h 2c]
";

    #[test]
    fn reviews_a_checked_river() {
        let (mut hands, warnings) = parse_pokerstars(HAND);
        assert!(warnings.is_empty(), "{warnings:?}");
        crate::analyse(&mut hands[0]);
        assert!(hands[0].derived.as_ref().unwrap().ok);
        let table = FrequencyTable::from_hands(&hands);
        let r = review_river(&hands[0], &table, &RangeOptions::default(), &ReviewOptions::default()).unwrap().expect("a heads-up river");
        assert_eq!(r.seats, [3, 1], "the big blind is out of position");
        assert!((r.pot_bb - 20.5).abs() < 1e-9, "{}", r.pot_bb);
        assert!((r.stacks_bb[0] - 90.0).abs() < 1e-9 && (r.stacks_bb[1] - 90.0).abs() < 1e-9);
        assert_eq!(r.decisions.len(), 2);
        // both hands are known: the villain showed, the hero was dealt in
        for d in &r.decisions {
            assert!(d.hand.is_some());
            let p: f64 = d.probs.iter().sum();
            assert!((p - 1.0).abs() < 1e-9);
            assert!(d.ev_loss_bb.unwrap() >= 0.0);
            assert_eq!(d.actions[d.taken], "Check");
        }
        assert!(r.exploitability_pct < 1.0, "{}", r.exploitability_pct);
    }

    #[test]
    fn bet_sizes_used_are_in_the_tree() {
        let text = HAND
            .replace("Villain: checks\nHero: checks\n*** SHOW DOWN", "Villain: checks\nHero: bets $20.50\nVillain: calls $20.50\n*** SHOW DOWN")
            .replace("$41", "$82");
        let (mut hands, warnings) = parse_pokerstars(&text);
        assert!(warnings.is_empty(), "{warnings:?}");
        crate::analyse(&mut hands[0]);
        assert!(hands[0].derived.as_ref().unwrap().ok, "{:?}", hands[0].derived.as_ref().unwrap().issues);
        let table = FrequencyTable::from_hands(&hands);
        let r = review_river(&hands[0], &table, &RangeOptions::default(), &ReviewOptions::default()).unwrap().unwrap();
        assert!(r.bets.iter().any(|b| (b - 0.5).abs() < 1e-9), "{:?}", r.bets);
        let hero = &r.decisions[1];
        assert!(hero.size_matched);
        assert_eq!(hero.actions[hero.taken], "Bet 50%");
        assert!(r.decisions[2].size_matched);
    }
}
