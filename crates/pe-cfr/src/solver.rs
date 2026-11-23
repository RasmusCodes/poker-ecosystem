//! Full-width counterfactual regret minimisation on compiled trees.
//!
//! All variants use alternating updates (one traversal per player per iteration)
//! and differ in how regrets and the average strategy are accumulated:
//!
//! | algorithm | regrets | average strategy weight |
//! |---|---|---|
//! | CFR (Zinkevich et al. 2007) | sum | 1 |
//! | CFR+ (Tammelin 2014) | floored at 0 after each update | t |
//! | Linear CFR (Brown & Sandholm 2019) | weighted by t | t |
//! | Discounted CFR (α, β, γ) | positive × t^α/(t^α+1), negative × t^β/(t^β+1) | × (t/(t+1))^γ |
//! | Predictive CFR+ (Farina et al. 2021) | floored, strategy from regrets + last instantaneous regret | t² |
//!
//! In two-player zero-sum games the average strategy of each converges to a Nash
//! equilibrium. With three or more players, or with an imperfect-recall
//! abstraction, no such guarantee holds; callers must report NashConv instead.

use serde::{Deserialize, Serialize};

use crate::tree::{Kind, Strategy, Tree};

/// `x^e` computed with correctly rounded operations for the exponents DCFR uses
/// (0, 1, 1.5, 2, 3), so the Rust engine and the JavaScript port agree bit for bit;
/// other exponents fall back to `powf`.
pub fn exact_pow(x: f64, e: f64) -> f64 {
    if e == 0.0 {
        1.0
    } else if e == 1.0 {
        x
    } else if e == 1.5 {
        x * x.sqrt()
    } else if e == 2.0 {
        x * x
    } else if e == 3.0 {
        x * x * x
    } else {
        x.powf(e)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Algorithm {
    Cfr,
    CfrPlus,
    Linear,
    Discounted { alpha: f64, beta: f64, gamma: f64 },
    PredictiveCfrPlus,
}

impl Algorithm {
    pub const DCFR: Algorithm = Algorithm::Discounted { alpha: 1.5, beta: 0.0, gamma: 2.0 };
    pub fn name(&self) -> String {
        match self {
            Algorithm::Cfr => "CFR".into(),
            Algorithm::CfrPlus => "CFR+".into(),
            Algorithm::Linear => "LCFR".into(),
            Algorithm::Discounted { alpha, beta, gamma } => {
                if (*alpha, *beta, *gamma) == (1.5, 0.0, 2.0) {
                    "DCFR".into()
                } else {
                    format!("DCFR({alpha},{beta},{gamma})")
                }
            }
            Algorithm::PredictiveCfrPlus => "PCFR+".into(),
        }
    }
    pub fn parse(s: &str) -> Result<Algorithm, String> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "cfr" | "vanilla" => Algorithm::Cfr,
            "cfr+" | "cfrplus" => Algorithm::CfrPlus,
            "lcfr" | "linear" => Algorithm::Linear,
            "dcfr" | "discounted" => Algorithm::DCFR,
            "pcfr+" | "pcfr" | "predictive" => Algorithm::PredictiveCfrPlus,
            other => return Err(format!("unknown algorithm '{other}'")),
        })
    }
    fn floors(&self) -> bool {
        matches!(self, Algorithm::CfrPlus | Algorithm::PredictiveCfrPlus)
    }
}

pub struct Cfr<'t> {
    pub tree: &'t Tree,
    pub algo: Algorithm,
    regrets: Vec<f64>,
    strat_sum: Vec<f64>,
    /// Last instantaneous regret (Predictive CFR+ prediction).
    last: Vec<f64>,
    inst_regret: Vec<f64>,
    inst_strat: Vec<f64>,
    current: Vec<f64>,
    pub iterations: u64,
    /// Node visits so far (a machine-independent cost measure).
    pub nodes_touched: u64,
}

impl<'t> Cfr<'t> {
    pub fn new(tree: &'t Tree, algo: Algorithm) -> Cfr<'t> {
        let n = tree.abs_actions;
        let mut s = Cfr {
            tree,
            algo,
            regrets: vec![0.0; n],
            strat_sum: vec![0.0; n],
            last: vec![0.0; n],
            inst_regret: vec![0.0; n],
            inst_strat: vec![0.0; n],
            current: vec![0.0; n],
            iterations: 0,
            nodes_touched: 0,
        };
        s.refresh_current();
        s
    }

    /// Recomputes the current strategy at every abstract set from the regrets.
    fn refresh_current(&mut self) {
        let predictive = matches!(self.algo, Algorithm::PredictiveCfrPlus);
        for a in &self.tree.abs_infosets {
            let (o, k) = (a.offset, a.num_actions);
            let mut total = 0.0;
            for i in o..o + k {
                let r = if predictive { self.regrets[i] + self.last[i] } else { self.regrets[i] };
                let p = r.max(0.0);
                self.current[i] = p;
                total += p;
            }
            if total > 0.0 {
                for i in o..o + k {
                    self.current[i] /= total;
                }
            } else {
                for i in o..o + k {
                    self.current[i] = 1.0 / k as f64;
                }
            }
        }
    }

    fn walk(&mut self, node: u32, player: usize, pi_i: f64, pi_o: f64) -> f64 {
        self.nodes_touched += 1;
        let tree = self.tree;
        let nd = tree.nodes[node as usize];
        match nd.kind {
            Kind::Terminal => tree.util(&nd, player),
            Kind::Chance => {
                let mut v = 0.0;
                for a in 0..nd.num_children as usize {
                    let p = tree.chance_prob(&nd, a);
                    if p > 0.0 {
                        v += p * self.walk(tree.child(&nd, a), player, pi_i, pi_o * p);
                    }
                }
                v
            }
            Kind::Decision => {
                let abs = &tree.abs_infosets[tree.infosets[nd.infoset as usize].abs as usize];
                let (o, k) = (abs.offset, abs.num_actions);
                if nd.player as usize == player {
                    let mut vals = [0.0f64; 16];
                    let mut v = 0.0;
                    for a in 0..k {
                        let s = self.current[o + a];
                        vals[a] = self.walk(tree.child(&nd, a), player, pi_i * s, pi_o);
                        v += s * vals[a];
                    }
                    for a in 0..k {
                        self.inst_regret[o + a] += pi_o * (vals[a] - v);
                    }
                    v
                } else {
                    let mut v = 0.0;
                    for a in 0..k {
                        let s = self.current[o + a];
                        if s > 0.0 {
                            v += s * self.walk(tree.child(&nd, a), player, pi_i, pi_o * s);
                        }
                    }
                    v
                }
            }
        }
    }

    /// Adds `player`'s realization weight times the current strategy at each of the
    /// player's information sets to `inst_strat`. Done in its own pass, from the
    /// player's own reach alone: the traversal skips subtrees the opponents never
    /// reach, and accumulating there would under-weight later iterations at sets a
    /// best responder can still steer into (which stalls the average strategy of
    /// DCFR and PCFR+).
    fn accumulate_realization(&mut self, player: usize) {
        let tree = self.tree;
        let mut own = vec![0.0f64; tree.nodes.len()];
        own[0] = 1.0;
        let mut seen = vec![false; tree.infosets.len()];
        for idx in 0..tree.nodes.len() {
            let nd = tree.nodes[idx];
            let r = own[idx];
            match nd.kind {
                Kind::Terminal => {}
                Kind::Chance => {
                    for a in 0..nd.num_children as usize {
                        own[tree.child(&nd, a) as usize] = r;
                    }
                }
                Kind::Decision => {
                    let abs = &tree.abs_infosets[tree.infosets[nd.infoset as usize].abs as usize];
                    let o = abs.offset;
                    let mine = nd.player as usize == player;
                    for a in 0..nd.num_children as usize {
                        own[tree.child(&nd, a) as usize] = if mine { r * self.current[o + a] } else { r };
                    }
                    if mine && !seen[nd.infoset as usize] {
                        seen[nd.infoset as usize] = true;
                        for a in 0..abs.num_actions {
                            self.inst_strat[o + a] += r * self.current[o + a];
                        }
                    }
                }
            }
        }
    }

    /// One iteration: a traversal and update for each player in turn.
    pub fn iterate(&mut self) {
        self.iterations += 1;
        let t = self.iterations as f64;
        for player in 0..self.tree.num_players {
            self.inst_regret.iter_mut().for_each(|x| *x = 0.0);
            self.inst_strat.iter_mut().for_each(|x| *x = 0.0);
            self.walk(0, player, 1.0, 1.0);
            self.accumulate_realization(player);
            let tree = self.tree;
            for a in &tree.abs_infosets {
                if a.player != player {
                    continue;
                }
                for i in a.offset..a.offset + a.num_actions {
                    let r = self.inst_regret[i];
                    let s = self.inst_strat[i];
                    match self.algo {
                        Algorithm::Cfr => {
                            self.regrets[i] += r;
                            self.strat_sum[i] += s;
                        }
                        Algorithm::CfrPlus => {
                            self.regrets[i] = (self.regrets[i] + r).max(0.0);
                            self.strat_sum[i] += t * s;
                        }
                        Algorithm::Linear => {
                            self.regrets[i] += t * r;
                            self.strat_sum[i] += t * s;
                        }
                        Algorithm::Discounted { alpha, beta, gamma } => {
                            let tp = t - 1.0;
                            let old = self.regrets[i];
                            let disc = if old > 0.0 {
                                let x = exact_pow(tp, alpha);
                                x / (x + 1.0)
                            } else {
                                let x = exact_pow(tp, beta);
                                x / (x + 1.0)
                            };
                            self.regrets[i] = old * disc + r;
                            self.strat_sum[i] = self.strat_sum[i] * exact_pow(tp / t, gamma) + s;
                        }
                        Algorithm::PredictiveCfrPlus => {
                            self.regrets[i] = (self.regrets[i] + r).max(0.0);
                            self.last[i] = r;
                            self.strat_sum[i] += t * t * s;
                        }
                    }
                    debug_assert!(!self.algo.floors() || self.regrets[i] >= 0.0);
                }
            }
            self.refresh_current();
        }
    }

    pub fn current_strategy(&self) -> Strategy {
        Strategy { probs: self.current.clone() }
    }

    pub fn average_strategy(&self) -> Strategy {
        let mut probs = vec![0.0; self.tree.abs_actions];
        for a in &self.tree.abs_infosets {
            let (o, k) = (a.offset, a.num_actions);
            let total: f64 = self.strat_sum[o..o + k].iter().sum();
            for i in o..o + k {
                probs[i] = if total > 0.0 { self.strat_sum[i] / total } else { 1.0 / k as f64 };
            }
        }
        Strategy { probs }
    }

    /// Bytes of solver state (regrets, sums and buffers).
    pub fn memory_bytes(&self) -> usize {
        6 * self.tree.abs_actions * std::mem::size_of::<f64>()
    }
}
