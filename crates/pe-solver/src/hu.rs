//! Heads-up range-vs-range solving with vector-form CFR.
//!
//! Every traversal carries a vector of reach probabilities for each hand of the
//! opponent and returns counterfactual values for each hand of the traverser.
//! Showdowns use the O(n) card-removal sweep of `pe_equity::sweep`, so a solve
//! is exact for the given ranges, board and bet menu: the only abstraction is the
//! menu of bet sizes (and, optionally, card buckets on the river, see
//! [`Buckets`]).
//!
//! Exploitability is computed exactly with a vector best response in the same
//! tree and reported as a percentage of the starting pot. With rake the game is
//! not constant-sum; the same NashConv / 2 figure is reported and labelled.
//!
//! Exploitative tools:
//! * **Locks** fix a player's strategy at chosen nodes while both players adapt
//!   everywhere else (classic node locking).
//! * **Opponent models** are complete strategies for one player (equilibrium
//!   play with profile-driven overrides, see `profile`). Against a model the
//!   solver computes the maximally exploitative best response, or a restricted
//!   Nash response (Johanson, Zinkevich & Bowling 2007): the opponent plays the
//!   model with probability `p` and freely with `1 - p`, which trades
//!   exploitation against exploitability.

use std::cell::UnsafeCell;
use std::collections::HashMap;

use pe_cards::CardMask;
use pe_cfr::Algorithm;
use pe_equity::settle::{PreparedBoard, PreparedHand, high_value, low_value};
use pe_equity::sweep::{ShowdownPlan, Sums};
use rayon::prelude::*;
use serde::Serialize;

use crate::spot::{Rake, SideRule, Spot};
use crate::tree::{BetTree, NodeKind};

/// Interior mutability for per-node solver state. Safe because a traversal
/// visits each decision node from exactly one thread: subtrees below different
/// cards of a chance node are disjoint.
struct Slot<T>(UnsafeCell<T>);
unsafe impl<T: Send> Sync for Slot<T> {}
impl<T> Slot<T> {
    fn new(v: T) -> Self {
        Slot(UnsafeCell::new(v))
    }
    #[allow(clippy::mut_from_ref)]
    unsafe fn get(&self) -> &mut T {
        unsafe { &mut *self.0.get() }
    }
}

pub struct NodeData {
    pub player: usize,
    pub n_actions: usize,
    /// Hands, or buckets when the node is abstracted.
    pub n_slots: usize,
    pub regrets: Vec<f64>,
    pub strat_sum: Vec<f64>,
    pub last: Vec<f64>,
    pub lock: Option<Vec<f64>>,
    /// Bucket of each hand (abstracted nodes only).
    pub bucket_of: Option<Vec<u32>>,
}

struct BoardData {
    hi: [ShowdownPlan; 2],
    hi2: Option<[ShowdownPlan; 2]>,
    lo: Option<[ShowdownPlan; 2]>,
    has_low: [Vec<bool>; 2],
    vals: [Vec<u32>; 2],
}

/// River card buckets: hands of the acting player at every river decision node
/// are grouped into `count` equal-weight buckets by hand strength on that board,
/// and each bucket shares one strategy.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Buckets {
    pub count: usize,
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Train { t: f64, dpos: f64, dneg: f64, dsum: f64, wsum: f64 },
    Eval,
    BestResponse,
}

pub struct HuSolver {
    pub spot: Spot,
    pub tree: BetTree,
    pub algo: Algorithm,
    data: Vec<Option<Slot<NodeData>>>,
    showdown_board: Vec<usize>,
    boards: Vec<BoardData>,
    compat: [ShowdownPlan; 2],
    pub weights: [Vec<f64>; 2],
    is72: [Vec<bool>; 2],
    bounty: f64,
    rake: Option<Rake>,
    deal_count: f64,
    pub iterations: u64,
    /// Opponent model for best responses and restricted Nash responses: the
    /// modelled player and a strategy for each of their decision nodes.
    model: Option<(usize, Vec<Option<Vec<f64>>>)>,
    /// Restricted Nash response weight `p` on the model (0 = no model in training).
    rnr_p: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Exploitability {
    pub ev: [f64; 2],
    pub br: [f64; 2],
    pub nash_conv: f64,
    /// NashConv / 2 as a percentage of the starting pot.
    pub pct_pot: f64,
    pub constant_sum: bool,
}

impl HuSolver {
    pub fn new(spot: Spot, algo: Algorithm, buckets: Option<Buckets>, max_nodes: usize) -> Result<HuSolver, String> {
        if spot.players() != 2 {
            return Err("the heads-up solver needs exactly two ranges".into());
        }
        let menu = spot.config.menu.clone();
        let tree = BetTree::build(&spot, &menu, max_nodes)?;
        let n = [spot.ranges[0].len(), spot.ranges[1].len()];
        let weights = [spot.ranges[0].weights.clone(), spot.ranges[1].weights.clone()];
        let masks = [spot.masks[0].clone(), spot.masks[1].clone()];
        let prepared: [Vec<PreparedHand>; 2] = [
            spot.ranges[0].hands.iter().map(|h| PreparedHand::new(h)).collect(),
            spot.ranges[1].hands.iter().map(|h| PreparedHand::new(h)).collect(),
        ];
        // showdown boards
        let mut board_index: HashMap<CardMask, usize> = HashMap::new();
        let mut showdown_board = vec![usize::MAX; tree.nodes.len()];
        let mut needed: Vec<CardMask> = Vec::new();
        for (i, node) in tree.nodes.iter().enumerate() {
            if let NodeKind::Terminal { showdown: true } = node.kind {
                let next = needed.len();
                let idx = *board_index.entry(node.board).or_insert_with(|| {
                    needed.push(node.board);
                    next
                });
                showdown_board[i] = idx;
            }
        }
        let game = spot.game;
        let board2 = spot.board2.as_ref().map(|b| PreparedBoard::new(b));
        let boards: Vec<BoardData> = needed
            .par_iter()
            .map(|&bm| {
                let pb = PreparedBoard::from_mask(bm);
                let value_of = |p: usize, b: &PreparedBoard, low: bool| -> Vec<u32> {
                    prepared[p]
                        .iter()
                        .map(|h| {
                            if h.mask & b.mask != 0 {
                                0
                            } else if low {
                                low_value(&game, h, b)
                            } else {
                                high_value(&game, h, b)
                            }
                        })
                        .collect()
                };
                let vals = [value_of(0, &pb, false), value_of(1, &pb, false)];
                let plan = |v: &[Vec<u32>; 2], i: usize| ShowdownPlan::new(&v[i], &masks[i], &v[1 - i], &masks[1 - i]);
                let hi = [plan(&vals, 0), plan(&vals, 1)];
                let hi2 = board2.as_ref().map(|b2| {
                    let v2 = [value_of(0, b2, false), value_of(1, b2, false)];
                    [plan(&v2, 0), plan(&v2, 1)]
                });
                let (lo, has_low) = if game.hilo {
                    let lv = [value_of(0, &pb, true), value_of(1, &pb, true)];
                    let has = [lv[0].iter().map(|&x| x > 0).collect(), lv[1].iter().map(|&x| x > 0).collect()];
                    (Some([plan(&lv, 0), plan(&lv, 1)]), has)
                } else {
                    (None, [vec![false; n[0]], vec![false; n[1]]])
                };
                BoardData { hi, hi2, lo, has_low, vals }
            })
            .collect();
        let zero = [vec![0u32; n[0]], vec![0u32; n[1]]];
        let compat = [
            ShowdownPlan::new(&zero[0], &masks[0], &zero[1], &masks[1]),
            ShowdownPlan::new(&zero[1], &masks[1], &zero[0], &masks[0]),
        ];
        // per-node state
        let mut data = Vec::with_capacity(tree.nodes.len());
        for node in &tree.nodes {
            match &node.kind {
                NodeKind::Decision { player, actions, .. } => {
                    let p = *player;
                    let k = actions.len();
                    let river = node.board_len == 5 && spot.board.len() < 5;
                    let bucket_of = match buckets {
                        Some(b) if river || (spot.board.len() == 5 && b.count > 0) => {
                            Some(bucket_hands(&game, &prepared[p], &weights[p], node.board, b.count))
                        }
                        _ => None,
                    };
                    let slots = bucket_of.as_ref().map_or(n[p], |_| buckets.unwrap().count);
                    let pred = matches!(algo, Algorithm::PredictiveCfrPlus);
                    data.push(Some(Slot::new(NodeData {
                        player: p,
                        n_actions: k,
                        n_slots: slots,
                        regrets: vec![0.0; k * slots],
                        strat_sum: vec![0.0; k * slots],
                        last: if pred { vec![0.0; k * slots] } else { Vec::new() },
                        lock: None,
                        bucket_of,
                    })));
                }
                _ => data.push(None),
            }
        }
        let hole = game.hole_cards as u32;
        let deck_left = (game.deck() & !(spot.board_mask() | spot.dead)).count_ones();
        let deal_count = (deck_left - 2 * hole) as f64;
        let bounty = match &spot.config.side_rule {
            Some(SideRule::SevenDeuce { amount }) => *amount,
            None => 0.0,
        };
        Ok(HuSolver {
            is72: [spot.is_72[0].clone(), spot.is_72[1].clone()],
            rake: spot.config.rake.clone(),
            spot,
            tree,
            algo,
            data,
            showdown_board,
            boards,
            compat,
            weights,
            bounty,
            deal_count,
            iterations: 0,
            model: None,
            rnr_p: 0.0,
        })
    }

    pub fn hands(&self, p: usize) -> usize {
        self.weights[p].len()
    }

    /// Bytes of solver state (regrets, sums, predictions).
    pub fn memory_bytes(&self) -> usize {
        self.data
            .iter()
            .flatten()
            .map(|d| {
                let d = unsafe { d.get() };
                8 * (d.regrets.len() + d.strat_sum.len() + d.last.len())
            })
            .sum()
    }

    fn rake_of(&self, pot: f64) -> f64 {
        match &self.rake {
            Some(r) => (pot * r.percent / 100.0).min(r.cap),
            None => 0.0,
        }
    }

    /// Fixes `player`'s strategy at `node` (`probs[a * hands + h]`).
    pub fn lock(&mut self, node: usize, probs: Vec<f64>) -> Result<(), String> {
        let d = self.data[node].as_ref().ok_or("not a decision node")?;
        let d = unsafe { d.get() };
        let n = self.weights[d.player].len();
        if probs.len() != d.n_actions * n {
            return Err(format!("lock needs {} probabilities", d.n_actions * n));
        }
        d.lock = Some(probs);
        Ok(())
    }

    /// Sets an opponent model (a full strategy for `player`) and the restricted
    /// Nash response weight used in training (0 trains a plain equilibrium).
    pub fn set_model(&mut self, player: usize, strategy: Vec<Option<Vec<f64>>>, p: f64) {
        self.model = Some((player, strategy));
        self.rnr_p = p.clamp(0.0, 1.0);
    }

    pub fn clear_model(&mut self) {
        self.model = None;
        self.rnr_p = 0.0;
    }

    /// Strategy at a node for each hand, `[a * hands + h]`.
    fn strategy(&self, node: usize, average: bool) -> Vec<f64> {
        let d = unsafe { self.data[node].as_ref().unwrap().get() };
        let n = self.weights[d.player].len();
        if let Some(l) = &d.lock {
            return l.clone();
        }
        let k = d.n_actions;
        let src = if average { &d.strat_sum } else { &d.regrets };
        let pred = !average && !d.last.is_empty();
        let mut slot_probs = vec![0.0; k * d.n_slots];
        for s in 0..d.n_slots {
            let mut total = 0.0;
            for a in 0..k {
                let mut x = src[a * d.n_slots + s];
                if pred {
                    x += d.last[a * d.n_slots + s];
                }
                let x = if average { x } else { x.max(0.0) };
                slot_probs[a * d.n_slots + s] = x;
                total += x;
            }
            for a in 0..k {
                slot_probs[a * d.n_slots + s] = if total > 0.0 { slot_probs[a * d.n_slots + s] / total } else { 1.0 / k as f64 };
            }
        }
        match &d.bucket_of {
            None => slot_probs,
            Some(b) => {
                let mut out = vec![0.0; k * n];
                for h in 0..n {
                    for a in 0..k {
                        out[a * n + h] = slot_probs[a * d.n_slots + b[h] as usize];
                    }
                }
                out
            }
        }
    }

    /// Average strategy at a decision node, `[a * hands + h]`.
    pub fn average_strategy(&self, node: usize) -> Vec<f64> {
        self.strategy(node, true)
    }

    fn model_at(&self, node: usize) -> Option<&Vec<f64>> {
        self.model.as_ref().and_then(|(_, m)| m[node].as_ref())
    }

    fn terminal(&self, node: usize, trav: usize, opp: &[f64]) -> Vec<f64> {
        let nd = &self.tree.nodes[node];
        let n = self.weights[trav].len();
        let o = 1 - trav;
        let p0 = self.tree.start_pot;
        let mut out = vec![0.0; n];
        let NodeKind::Terminal { showdown } = nd.kind else { unreachable!() };
        let mut totals = vec![0.0; n];
        if !showdown {
            let folder = if nd.folded[0] { 0 } else { 1 };
            let c_f = nd.contrib[folder];
            let rake = self.rake_of(p0 + 2.0 * c_f);
            self.compat[trav].compatible_totals(opp, &mut totals);
            if trav != folder {
                for h in 0..n {
                    out[h] = (p0 + c_f - rake) * totals[h];
                    if self.bounty > 0.0 && self.is72[trav][h] {
                        out[h] += self.bounty * totals[h];
                    }
                }
            } else {
                for h in 0..n {
                    out[h] = -c_f * totals[h];
                }
                if self.bounty > 0.0 {
                    let w72: Vec<f64> = opp.iter().zip(&self.is72[o]).map(|(w, &b)| if b { *w } else { 0.0 }).collect();
                    let mut t72 = vec![0.0; n];
                    self.compat[trav].compatible_totals(&w72, &mut t72);
                    for h in 0..n {
                        out[h] -= self.bounty * t72[h];
                    }
                }
            }
            return out;
        }
        let m = nd.contrib[0].min(nd.contrib[1]);
        let q = p0 + 2.0 * m;
        let q2 = q - self.rake_of(q);
        let bd = &self.boards[self.showdown_board[node]];
        let mut hi = vec![Sums::default(); n];
        bd.hi[trav].sums(opp, &mut hi);
        let mut share: Vec<f64> = hi.iter().map(|s| s.win + 0.5 * s.tie).collect();
        if let Some(lo_plans) = &bd.lo {
            // half the pot to the best qualifying low; the high hand scoops when no low qualifies
            let mut lo = vec![Sums::default(); n];
            lo_plans[trav].sums(opp, &mut lo);
            let opp_nolow: Vec<f64> = opp.iter().zip(&bd.has_low[o]).map(|(w, &l)| if l { 0.0 } else { *w }).collect();
            let mut hi_nolow = vec![Sums::default(); n];
            bd.hi[trav].sums(&opp_nolow, &mut hi_nolow);
            for h in 0..n {
                let x = if bd.has_low[trav][h] { lo[h].win + 0.5 * lo[h].tie } else { hi_nolow[h].win + 0.5 * hi_nolow[h].tie };
                share[h] = 0.5 * share[h] + 0.5 * x;
            }
        }
        if let Some(h2) = &bd.hi2 {
            let mut s2 = vec![Sums::default(); n];
            h2[trav].sums(opp, &mut s2);
            for h in 0..n {
                share[h] = 0.5 * share[h] + 0.5 * (s2[h].win + 0.5 * s2[h].tie);
            }
        }
        for h in 0..n {
            let total = hi[h].total();
            out[h] = q2 * share[h] - m * total;
        }
        if self.bounty > 0.0 {
            let w72: Vec<f64> = opp.iter().zip(&self.is72[o]).map(|(w, &b)| if b { *w } else { 0.0 }).collect();
            let mut s72 = vec![Sums::default(); n];
            bd.hi[trav].sums(&w72, &mut s72);
            for h in 0..n {
                if self.is72[trav][h] {
                    out[h] += self.bounty * hi[h].win;
                }
                out[h] -= self.bounty * s72[h].lose;
            }
        }
        out
    }

    /// The recursive traversal. `fixed` is the opponent's reach in the model branch.
    fn walk(&self, node: usize, trav: usize, reach_self: &[f64], reach_opp: &[f64], fixed: Option<&[f64]>, mode: Mode) -> Vec<f64> {
        let nd = &self.tree.nodes[node];
        match &nd.kind {
            NodeKind::Terminal { .. } => match fixed {
                Some(f) => {
                    let opp: Vec<f64> = reach_opp.iter().zip(f).map(|(a, b)| a + b).collect();
                    self.terminal(node, trav, &opp)
                }
                None => self.terminal(node, trav, reach_opp),
            },
            NodeKind::Chance { cards, children } => {
                let n = self.weights[trav].len();
                let o = 1 - trav;
                let ms = &self.spot.masks[trav];
                let mo = &self.spot.masks[o];
                let parts: Vec<Vec<f64>> = cards
                    .par_iter()
                    .zip(children.par_iter())
                    .map(|(c, &child)| {
                        let cm = c.mask();
                        let rs: Vec<f64> = reach_self.iter().zip(ms).map(|(r, m)| if m & cm != 0 { 0.0 } else { *r }).collect();
                        let ro: Vec<f64> = reach_opp.iter().zip(mo).map(|(r, m)| if m & cm != 0 { 0.0 } else { *r }).collect();
                        let rf: Option<Vec<f64>> = fixed.map(|f| f.iter().zip(mo).map(|(r, m)| if m & cm != 0 { 0.0 } else { *r }).collect());
                        let mut v = self.walk(child, trav, &rs, &ro, rf.as_deref(), mode);
                        for h in 0..n {
                            if ms[h] & cm != 0 {
                                v[h] = 0.0;
                            }
                        }
                        v
                    })
                    .collect();
                let mut out = vec![0.0; n];
                for v in parts {
                    for h in 0..n {
                        out[h] += v[h];
                    }
                }
                let d = self.deal_count;
                out.iter_mut().for_each(|x| *x /= d);
                out
            }
            NodeKind::Decision { player, children, .. } => {
                let p = *player;
                let k = children.len();
                if p == trav {
                    let n = self.weights[trav].len();
                    let average = !matches!(mode, Mode::Train { .. });
                    let sigma = self.strategy(node, average);
                    let mut vals: Vec<Vec<f64>> = Vec::with_capacity(k);
                    for a in 0..k {
                        let rs: Vec<f64> = (0..n).map(|h| reach_self[h] * sigma[a * n + h]).collect();
                        vals.push(self.walk(children[a], trav, &rs, reach_opp, fixed, mode));
                    }
                    let mut v = vec![0.0; n];
                    match mode {
                        Mode::BestResponse => {
                            for h in 0..n {
                                v[h] = (0..k).map(|a| vals[a][h]).fold(f64::NEG_INFINITY, f64::max);
                            }
                        }
                        _ => {
                            for a in 0..k {
                                for h in 0..n {
                                    v[h] += sigma[a * n + h] * vals[a][h];
                                }
                            }
                        }
                    }
                    if let Mode::Train { t, dpos, dneg, dsum, wsum } = mode {
                        self.update(node, trav, &sigma, &vals, &v, reach_self, t, dpos, dneg, dsum, wsum);
                    }
                    v
                } else {
                    let n_o = self.weights[p].len();
                    let average = !matches!(mode, Mode::Train { .. });
                    let sigma = self.strategy(node, average);
                    let model = if fixed.is_some() { self.model_at(node) } else { None };
                    let mut v = vec![0.0; self.weights[trav].len()];
                    for a in 0..k {
                        let ro: Vec<f64> = (0..n_o).map(|h| reach_opp[h] * sigma[a * n_o + h]).collect();
                        let rf: Option<Vec<f64>> = fixed.map(|f| {
                            let src = model.unwrap_or(&sigma);
                            (0..n_o).map(|h| f[h] * src[a * n_o + h]).collect()
                        });
                        // Unreached subtrees contribute no value, but training still walks
                        // them: the traverser's average strategy needs its own realization
                        // weight there whatever the opponent's current strategy does.
                        let unreached = ro.iter().all(|x| *x == 0.0) && rf.as_ref().is_none_or(|r| r.iter().all(|x| *x == 0.0));
                        if unreached && !matches!(mode, Mode::Train { .. }) {
                            continue;
                        }
                        let child = self.walk(children[a], trav, reach_self, &ro, rf.as_deref(), mode);
                        for (x, c) in v.iter_mut().zip(child) {
                            *x += c;
                        }
                    }
                    v
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn update(&self, node: usize, trav: usize, sigma: &[f64], vals: &[Vec<f64>], v: &[f64], reach_self: &[f64], t: f64, dpos: f64, dneg: f64, dsum: f64, wsum: f64) {
        let d = unsafe { self.data[node].as_ref().unwrap().get() };
        if d.lock.is_some() {
            return;
        }
        let n = self.weights[trav].len();
        let k = d.n_actions;
        let s = d.n_slots;
        // instantaneous regrets and strategy contributions per slot
        let mut inst = vec![0.0; k * s];
        let mut contrib = vec![0.0; k * s];
        for h in 0..n {
            let slot = d.bucket_of.as_ref().map_or(h, |b| b[h] as usize);
            for a in 0..k {
                inst[a * s + slot] += vals[a][h] - v[h];
                contrib[a * s + slot] += reach_self[h] * sigma[a * n + h];
            }
        }
        for i in 0..k * s {
            let r = inst[i];
            match self.algo {
                Algorithm::Cfr => {
                    d.regrets[i] += r;
                    d.strat_sum[i] += contrib[i];
                }
                Algorithm::CfrPlus => {
                    d.regrets[i] = (d.regrets[i] + r).max(0.0);
                    d.strat_sum[i] += t * contrib[i];
                }
                Algorithm::Linear => {
                    d.regrets[i] += t * r;
                    d.strat_sum[i] += t * contrib[i];
                }
                Algorithm::Discounted { .. } => {
                    let old = d.regrets[i];
                    d.regrets[i] = old * if old > 0.0 { dpos } else { dneg } + r;
                    d.strat_sum[i] = d.strat_sum[i] * dsum + contrib[i] * wsum;
                }
                Algorithm::PredictiveCfrPlus => {
                    d.regrets[i] = (d.regrets[i] + r).max(0.0);
                    d.last[i] = r;
                    d.strat_sum[i] += t * t * contrib[i];
                }
            }
        }
    }

    fn train_mode(&self, t: f64) -> Mode {
        match self.algo {
            Algorithm::Discounted { alpha, beta, gamma } => {
                let tp = t - 1.0;
                let a = pe_cfr::solver::exact_pow(tp, alpha);
                let b = pe_cfr::solver::exact_pow(tp, beta);
                Mode::Train { t, dpos: a / (a + 1.0), dneg: b / (b + 1.0), dsum: pe_cfr::solver::exact_pow(tp / t, gamma), wsum: 1.0 }
            }
            _ => Mode::Train { t, dpos: 1.0, dneg: 1.0, dsum: 1.0, wsum: 1.0 },
        }
    }

    /// One iteration (alternating updates for both players).
    pub fn iterate(&mut self) {
        self.iterations += 1;
        let mode = self.train_mode(self.iterations as f64);
        for trav in 0..2 {
            let o = 1 - trav;
            match &self.model {
                Some((mp, _)) if *mp == o && self.rnr_p > 0.0 => {
                    let p = self.rnr_p;
                    let free: Vec<f64> = self.weights[o].iter().map(|w| w * (1.0 - p)).collect();
                    let fixed: Vec<f64> = self.weights[o].iter().map(|w| w * p).collect();
                    self.walk(0, trav, &self.weights[trav].clone(), &free, Some(&fixed), mode);
                }
                _ => {
                    self.walk(0, trav, &self.weights[trav].clone(), &self.weights[o].clone(), None, mode);
                }
            }
        }
    }

    /// Joint weight of card-compatible hand pairs (the normaliser for values).
    pub fn joint_weight(&self) -> f64 {
        let mut t = vec![0.0; self.weights[0].len()];
        self.compat[0].compatible_totals(&self.weights[1], &mut t);
        t.iter().zip(&self.weights[0]).map(|(a, b)| a * b).sum()
    }

    /// Counterfactual values at the root for `trav` under the average strategies.
    pub fn root_values(&self, trav: usize) -> Vec<f64> {
        let o = 1 - trav;
        self.walk(0, trav, &self.weights[trav], &self.weights[o], None, Mode::Eval)
    }

    /// Exact exploitability of the average strategies.
    pub fn exploitability(&self) -> Exploitability {
        let z = self.joint_weight();
        let mut ev = [0.0; 2];
        let mut br = [0.0; 2];
        for p in 0..2 {
            let o = 1 - p;
            let v = self.walk(0, p, &self.weights[p], &self.weights[o], None, Mode::Eval);
            ev[p] = v.iter().zip(&self.weights[p]).map(|(a, b)| a * b).sum::<f64>() / z;
            let b = self.walk(0, p, &self.weights[p], &self.weights[o], None, Mode::BestResponse);
            br[p] = b.iter().zip(&self.weights[p]).map(|(a, b)| a * b).sum::<f64>() / z;
        }
        let nash_conv = (br[0] - ev[0]) + (br[1] - ev[1]);
        Exploitability { ev, br, nash_conv, pct_pot: 100.0 * nash_conv / 2.0 / self.tree.start_pot, constant_sum: self.rake.is_none() }
    }

    /// Value for `p` (per starting pot unit) of its average strategy against the
    /// opponent model, and of a best response to the model.
    pub fn values_vs_model(&self, p: usize) -> Option<(f64, f64)> {
        let (mp, _) = self.model.as_ref()?;
        if *mp != 1 - p {
            return None;
        }
        let z = self.joint_weight();
        let o = 1 - p;
        let zero = vec![0.0; self.weights[o].len()];
        let ev = self.walk(0, p, &self.weights[p], &zero, Some(&self.weights[o]), Mode::Eval);
        let br = self.walk(0, p, &self.weights[p], &zero, Some(&self.weights[o]), Mode::BestResponse);
        let f = |v: Vec<f64>| v.iter().zip(&self.weights[p]).map(|(a, b)| a * b).sum::<f64>() / z;
        Some((f(ev), f(br)))
    }

    /// Per-hand values at the root (chips per hand, i.e. counterfactual value
    /// divided by the compatible opponent weight) under the average strategies.
    pub fn hand_values(&self, p: usize) -> Vec<f64> {
        let v = self.root_values(p);
        let mut t = vec![0.0; v.len()];
        self.compat[p].compatible_totals(&self.weights[1 - p], &mut t);
        v.iter().zip(&t).map(|(a, b)| if *b > 0.0 { a / b } else { 0.0 }).collect()
    }

    /// Value of each action for each hand at a decision node of `p`, under the
    /// average strategies, as chips per hand. Also returns the opponent weight
    /// reaching the node for each of `p`'s hands (0 where the hand never gets there).
    pub fn action_values(&self, node: usize) -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
        let NodeKind::Decision { player, children, .. } = &self.tree.nodes[node].kind else {
            return Err("not a decision node".into());
        };
        let p = *player;
        let o = 1 - p;
        // reach of the opponent into this node along the path from the root
        let mut path = vec![node];
        let mut cur = node;
        while let Some(par) = self.tree.nodes[cur].parent {
            path.push(par);
            cur = par;
        }
        path.reverse();
        let mut ro = self.weights[o].clone();
        let mut rs = self.weights[p].clone();
        for w in path.windows(2) {
            let (a_node, b_node) = (w[0], w[1]);
            match &self.tree.nodes[a_node].kind {
                NodeKind::Decision { player: q, children: ch, .. } => {
                    let a = ch.iter().position(|&c| c == b_node).unwrap();
                    let sigma = self.average_strategy(a_node);
                    let nq = self.weights[*q].len();
                    let target = if *q == o { &mut ro } else { &mut rs };
                    for h in 0..nq {
                        target[h] *= sigma[a * nq + h];
                    }
                }
                NodeKind::Chance { cards, children: ch } => {
                    let k = ch.iter().position(|&c| c == b_node).unwrap();
                    let cm = cards[k].mask();
                    for (r, m) in ro.iter_mut().zip(&self.spot.masks[o]) {
                        if m & cm != 0 {
                            *r = 0.0;
                        }
                    }
                    for (r, m) in rs.iter_mut().zip(&self.spot.masks[p]) {
                        if m & cm != 0 {
                            *r = 0.0;
                        }
                    }
                }
                _ => {}
            }
        }
        let mut totals = vec![0.0; self.weights[p].len()];
        self.compat[p].compatible_totals(&ro, &mut totals);
        let mut out = Vec::new();
        for &c in children {
            let v = self.walk(c, p, &rs, &ro, None, Mode::Eval);
            out.push(v.iter().zip(&totals).map(|(a, b)| if *b > 1e-12 { a / b } else { 0.0 }).collect());
        }
        let reach: Vec<f64> = rs.iter().zip(&totals).map(|(r, t)| if *t > 1e-12 { *r } else { 0.0 }).collect();
        Ok((out, reach))
    }

    /// Hand values on the final board of a river spot (for strength orderings).
    pub fn river_values(&self, p: usize) -> Option<Vec<u32>> {
        self.boards.first().map(|b| b.vals[p].clone())
    }
}

/// Equal-weight buckets of a player's hands by high-hand strength on `board`.
fn bucket_hands(game: &pe_cards::Game, hands: &[PreparedHand], weights: &[f64], board: CardMask, count: usize) -> Vec<u32> {
    let pb = PreparedBoard::from_mask(board);
    let vals: Vec<u32> = hands.iter().map(|h| if h.mask & board != 0 { 0 } else { high_value(game, h, &pb) }).collect();
    let mut order: Vec<usize> = (0..hands.len()).collect();
    order.sort_by_key(|&i| vals[i]);
    let total: f64 = order.iter().filter(|&&i| hands[i].mask & board == 0).map(|&i| weights[i]).sum();
    let mut out = vec![0u32; hands.len()];
    let mut acc = 0.0;
    let mut i = 0;
    while i < order.len() {
        // hands of equal strength share a bucket
        let mut j = i;
        let mut w = 0.0;
        while j < order.len() && vals[order[j]] == vals[order[i]] {
            if hands[order[j]].mask & board == 0 {
                w += weights[order[j]];
            }
            j += 1;
        }
        let mid = if total > 0.0 { (acc + w / 2.0) / total } else { 0.0 };
        let b = ((mid * count as f64) as usize).min(count - 1) as u32;
        for &h in &order[i..j] {
            out[h] = b;
        }
        acc += w;
        i = j;
    }
    out
}
