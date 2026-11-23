//! Multiway river solving (3 to 6 players) by exact enumeration of joint hands.
//!
//! Every card-compatible combination of one hand per player is listed once; a
//! terminal's value for a hand sums, over the combinations containing it, the
//! other players' reach times the pot share it wins (side pots included). That
//! makes card removal exact between all players, folded ones included, at a
//! cost that grows with the product of the range sizes: this solver is meant for
//! modest ranges.
//!
//! What is and is not claimed: CFR here is a heuristic. With three or more
//! players there is no guarantee that the average strategy approaches a Nash
//! equilibrium, and equilibria need not be unique or interchangeable. NashConv
//! (the total gain available to players switching to best responses) is computed
//! exactly and reported; a small NashConv means the profile is an approximate
//! equilibrium of this abstracted game, nothing more.

use pe_cfr::Algorithm;
use pe_equity::settle::{PreparedBoard, PreparedHand, high_value};
use rayon::prelude::*;
use serde::Serialize;

use crate::spot::Spot;
use crate::tree::{BetTree, NodeKind, side_pots};

struct Store {
    player: usize,
    k: usize,
    regrets: Vec<f64>,
    strat_sum: Vec<f64>,
}

struct Term {
    contrib: Vec<f64>,
    pots: Vec<(f64, Vec<usize>)>,
}

pub struct MultiwaySolver {
    pub spot: Spot,
    pub tree: BetTree,
    pub algo: Algorithm,
    data: Vec<Option<Store>>,
    terms: Vec<Option<Term>>,
    tuples: Vec<u16>,
    vals: Vec<Vec<u32>>,
    pub weights: Vec<Vec<f64>>,
    pub iterations: u64,
    players: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct MultiwayReport {
    pub players: usize,
    pub tuples: usize,
    pub ev: Vec<f64>,
    pub br: Vec<f64>,
    pub gains: Vec<f64>,
    pub nash_conv: f64,
    /// NashConv as a percentage of the starting pot.
    pub nash_conv_pct_pot: f64,
    pub label: &'static str,
}

#[derive(Clone, Copy)]
enum Mode {
    Train { t: f64, dpos: f64, dneg: f64, dsum: f64 },
    Eval,
    Br,
}

impl MultiwaySolver {
    pub fn new(spot: Spot, algo: Algorithm, max_tuples: usize, max_nodes: usize) -> Result<MultiwaySolver, String> {
        let players = spot.players();
        if players < 2 {
            return Err("need at least two players".into());
        }
        if spot.board.len() != 5 || spot.board2.is_some() || spot.game.hilo {
            return Err("the multiway solver handles single-board high-only rivers".into());
        }
        let menu = spot.config.menu.clone();
        let tree = BetTree::build(&spot, &menu, max_nodes)?;
        let board = PreparedBoard::new(&spot.board);
        let vals: Vec<Vec<u32>> = spot
            .ranges
            .iter()
            .map(|r| r.hands.iter().map(|h| high_value(&spot.game, &PreparedHand::new(h), &board)).collect())
            .collect();
        // enumerate compatible tuples
        let mut tuples: Vec<u16> = Vec::new();
        let mut cur = vec![0u16; players];
        fn rec(p: usize, used: u64, spot: &Spot, cur: &mut Vec<u16>, out: &mut Vec<u16>, max: usize) -> Result<(), String> {
            if p == spot.players() {
                if out.len() / spot.players() >= max {
                    return Err(format!("more than {max} joint hand combinations: narrow the ranges"));
                }
                out.extend_from_slice(cur);
                return Ok(());
            }
            for (h, &m) in spot.masks[p].iter().enumerate() {
                if m & used == 0 {
                    cur[p] = h as u16;
                    rec(p + 1, used | m, spot, cur, out, max)?;
                }
            }
            Ok(())
        }
        rec(0, 0, &spot, &mut cur, &mut tuples, max_tuples)?;
        let mut data = Vec::with_capacity(tree.nodes.len());
        let mut terms = Vec::with_capacity(tree.nodes.len());
        for node in &tree.nodes {
            match &node.kind {
                NodeKind::Decision { player, actions, .. } => {
                    let n = spot.ranges[*player].len();
                    let k = actions.len();
                    data.push(Some(Store { player: *player, k, regrets: vec![0.0; k * n], strat_sum: vec![0.0; k * n] }));
                    terms.push(None);
                }
                NodeKind::Terminal { .. } => {
                    data.push(None);
                    terms.push(Some(Term { contrib: node.contrib.clone(), pots: side_pots(tree.start_pot, &node.contrib, &node.folded) }));
                }
                NodeKind::Chance { .. } => return Err("multiway spots must start on the river".into()),
            }
        }
        let weights = spot.ranges.iter().map(|r| r.weights.clone()).collect();
        Ok(MultiwaySolver { spot, tree, algo, data, terms, tuples, vals, weights, iterations: 0, players })
    }

    pub fn tuple_count(&self) -> usize {
        self.tuples.len() / self.players
    }

    pub fn memory_bytes(&self) -> usize {
        let stores: usize = self.data.iter().flatten().map(|s| 16 * s.regrets.len()).sum();
        stores + self.tuples.len() * 2
    }

    fn strategy(&self, node: usize, average: bool) -> Vec<f64> {
        let s = self.data[node].as_ref().unwrap();
        let n = self.weights[s.player].len();
        let k = s.k;
        let src = if average { &s.strat_sum } else { &s.regrets };
        let mut out = vec![0.0; k * n];
        for h in 0..n {
            let mut tot = 0.0;
            for a in 0..k {
                let x = if average { src[a * n + h] } else { src[a * n + h].max(0.0) };
                out[a * n + h] = x;
                tot += x;
            }
            for a in 0..k {
                out[a * n + h] = if tot > 0.0 { out[a * n + h] / tot } else { 1.0 / k as f64 };
            }
        }
        out
    }

    pub fn average_strategy(&self, node: usize) -> Vec<f64> {
        self.strategy(node, true)
    }

    fn terminal(&self, node: usize, i: usize, reach: &[Vec<f64>]) -> Vec<f64> {
        let term = self.terms[node].as_ref().unwrap();
        let np = self.players;
        let n = self.weights[i].len();
        let chunk = 4096 * np;
        self.tuples
            .par_chunks(chunk)
            .map(|ts| {
                let mut out = vec![0.0; n];
                for t in ts.chunks_exact(np) {
                    let mut w = 1.0;
                    for j in 0..np {
                        if j != i {
                            w *= reach[j][t[j] as usize];
                            if w == 0.0 {
                                break;
                            }
                        }
                    }
                    if w == 0.0 {
                        continue;
                    }
                    let mut u = -term.contrib[i];
                    for (amount, elig) in &term.pots {
                        if !elig.contains(&i) {
                            continue;
                        }
                        let best = elig.iter().map(|&j| self.vals[j][t[j] as usize]).max().unwrap();
                        if self.vals[i][t[i] as usize] == best {
                            let k = elig.iter().filter(|&&j| self.vals[j][t[j] as usize] == best).count();
                            u += amount / k as f64;
                        }
                    }
                    out[t[i] as usize] += w * u;
                }
                out
            })
            .reduce(|| vec![0.0; n], |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                a
            })
    }

    fn walk(&mut self, node: usize, i: usize, reach: &mut Vec<Vec<f64>>, mode: Mode) -> Vec<f64> {
        let kind = self.tree.nodes[node].kind.clone();
        match kind {
            NodeKind::Terminal { .. } => self.terminal(node, i, reach),
            NodeKind::Chance { .. } => unreachable!(),
            NodeKind::Decision { player, children, .. } => {
                let n_p = self.weights[player].len();
                let average = !matches!(mode, Mode::Train { .. });
                let sigma = self.strategy(node, average);
                let k = children.len();
                let saved = reach[player].clone();
                let mut vals = Vec::with_capacity(k);
                for a in 0..k {
                    for h in 0..n_p {
                        reach[player][h] = saved[h] * sigma[a * n_p + h];
                    }
                    // skip unreached subtrees only outside training (see hu.rs)
                    let training = matches!(mode, Mode::Train { .. });
                    if !training && player != i && reach[player].iter().all(|x| *x == 0.0) {
                        vals.push(vec![0.0; self.weights[i].len()]);
                        continue;
                    }
                    vals.push(self.walk(children[a], i, reach, mode));
                }
                reach[player] = saved;
                let n_i = self.weights[i].len();
                let mut v = vec![0.0; n_i];
                if player == i {
                    match mode {
                        Mode::Br => {
                            for h in 0..n_i {
                                v[h] = (0..k).map(|a| vals[a][h]).fold(f64::NEG_INFINITY, f64::max);
                            }
                        }
                        _ => {
                            for a in 0..k {
                                for h in 0..n_i {
                                    v[h] += sigma[a * n_i + h] * vals[a][h];
                                }
                            }
                        }
                    }
                    if let Mode::Train { t, dpos, dneg, dsum } = mode {
                        let algo = self.algo;
                        let own = reach[i].clone();
                        let s = self.data[node].as_mut().unwrap();
                        for a in 0..k {
                            for h in 0..n_i {
                                let idx = a * n_i + h;
                                let r = vals[a][h] - v[h];
                                let c = own[h] * sigma[idx];
                                match algo {
                                    Algorithm::Cfr => {
                                        s.regrets[idx] += r;
                                        s.strat_sum[idx] += c;
                                    }
                                    Algorithm::CfrPlus | Algorithm::PredictiveCfrPlus => {
                                        s.regrets[idx] = (s.regrets[idx] + r).max(0.0);
                                        s.strat_sum[idx] += t * c;
                                    }
                                    Algorithm::Linear => {
                                        s.regrets[idx] += t * r;
                                        s.strat_sum[idx] += t * c;
                                    }
                                    Algorithm::Discounted { .. } => {
                                        let old = s.regrets[idx];
                                        s.regrets[idx] = old * if old > 0.0 { dpos } else { dneg } + r;
                                        s.strat_sum[idx] = s.strat_sum[idx] * dsum + c;
                                    }
                                }
                            }
                        }
                    }
                } else {
                    for a in 0..k {
                        for h in 0..n_i {
                            v[h] += vals[a][h];
                        }
                    }
                }
                v
            }
        }
    }

    pub fn iterate(&mut self) {
        self.iterations += 1;
        let t = self.iterations as f64;
        let mode = match self.algo {
            Algorithm::Discounted { alpha, beta, gamma } => {
                let tp = t - 1.0;
                let a = pe_cfr::solver::exact_pow(tp, alpha);
                let b = pe_cfr::solver::exact_pow(tp, beta);
                Mode::Train { t, dpos: a / (a + 1.0), dneg: b / (b + 1.0), dsum: pe_cfr::solver::exact_pow(tp / t, gamma) }
            }
            _ => Mode::Train { t, dpos: 1.0, dneg: 1.0, dsum: 1.0 },
        };
        for i in 0..self.players {
            let mut reach = self.weights.clone();
            self.walk(0, i, &mut reach, mode);
        }
    }

    fn joint_weight(&self) -> f64 {
        let np = self.players;
        self.tuples.chunks_exact(np).map(|t| (0..np).map(|j| self.weights[j][t[j] as usize]).product::<f64>()).sum()
    }

    /// Exact NashConv of the average strategies.
    pub fn report(&mut self) -> MultiwayReport {
        let z = self.joint_weight();
        let np = self.players;
        let mut ev = vec![0.0; np];
        let mut br = vec![0.0; np];
        for i in 0..np {
            let mut reach = self.weights.clone();
            let v = self.walk(0, i, &mut reach, Mode::Eval);
            ev[i] = v.iter().zip(&self.weights[i]).map(|(a, b)| a * b).sum::<f64>() / z;
            let mut reach = self.weights.clone();
            let b = self.walk(0, i, &mut reach, Mode::Br);
            br[i] = b.iter().zip(&self.weights[i]).map(|(a, b)| a * b).sum::<f64>() / z;
        }
        let gains: Vec<f64> = (0..np).map(|i| (br[i] - ev[i]).max(0.0)).collect();
        let nc: f64 = gains.iter().sum();
        MultiwayReport {
            players: np,
            tuples: self.tuple_count(),
            ev,
            br,
            gains,
            nash_conv: nc,
            nash_conv_pct_pot: 100.0 * nc / self.tree.start_pot,
            label: if np == 2 { "two-player: NashConv/2 is the exploitability" } else { "multiway CFR approximation: no Nash guarantee" },
        }
    }
}
