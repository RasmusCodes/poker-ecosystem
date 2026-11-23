//! Monte Carlo CFR (Lanctot et al. 2009) on compiled trees.
//!
//! * External sampling: the updating player explores every action; chance and the
//!   other players are sampled once. The average strategy is updated at sampled
//!   opponent nodes ("simple averaging").
//! * Outcome sampling: one trajectory per update, with ε-exploration at the
//!   updating player's nodes and importance weights; the average strategy uses
//!   stochastically weighted averaging (Lanctot 2013, Algorithm 3).
//!
//! Both are unbiased estimators of the full-width counterfactual regrets, so in
//! two-player zero-sum games they converge in expectation; each iteration is far
//! cheaper but noisier. Compare them with full-width CFR by nodes touched.

use pe_cards::Rng;

use crate::tree::{Kind, Strategy, Tree};

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Sampling {
    External,
    Outcome { epsilon: f64 },
}

pub struct Mccfr<'t> {
    pub tree: &'t Tree,
    pub sampling: Sampling,
    regrets: Vec<f64>,
    strat_sum: Vec<f64>,
    rng: Rng,
    pub iterations: u64,
    pub nodes_touched: u64,
}

impl<'t> Mccfr<'t> {
    pub fn new(tree: &'t Tree, sampling: Sampling, seed: u64) -> Mccfr<'t> {
        Mccfr { tree, sampling, regrets: vec![0.0; tree.abs_actions], strat_sum: vec![0.0; tree.abs_actions], rng: Rng::new(seed), iterations: 0, nodes_touched: 0 }
    }

    #[inline]
    fn regret_matching(&self, o: usize, k: usize, out: &mut [f64; 16]) {
        let mut total = 0.0;
        for a in 0..k {
            let p = self.regrets[o + a].max(0.0);
            out[a] = p;
            total += p;
        }
        if total > 0.0 {
            for a in 0..k {
                out[a] /= total;
            }
        } else {
            for a in 0..k {
                out[a] = 1.0 / k as f64;
            }
        }
    }

    fn sample(&mut self, probs: &[f64]) -> usize {
        let x = self.rng.next_f64();
        let mut acc = 0.0;
        for (i, p) in probs.iter().enumerate() {
            acc += p;
            if x < acc {
                return i;
            }
        }
        probs.iter().rposition(|p| *p > 0.0).unwrap_or(0)
    }

    fn external(&mut self, node: u32, player: usize) -> f64 {
        self.nodes_touched += 1;
        let tree = self.tree;
        let nd = tree.nodes[node as usize];
        match nd.kind {
            Kind::Terminal => tree.util(&nd, player),
            Kind::Chance => {
                let probs: Vec<f64> = (0..nd.num_children as usize).map(|a| tree.chance_prob(&nd, a)).collect();
                let a = self.sample(&probs);
                self.external(tree.child(&nd, a), player)
            }
            Kind::Decision => {
                let abs = &tree.abs_infosets[tree.infosets[nd.infoset as usize].abs as usize];
                let (o, k) = (abs.offset, abs.num_actions);
                let mut sigma = [0.0; 16];
                self.regret_matching(o, k, &mut sigma);
                if nd.player as usize == player {
                    let mut vals = [0.0; 16];
                    let mut v = 0.0;
                    for a in 0..k {
                        vals[a] = self.external(tree.child(&nd, a), player);
                        v += sigma[a] * vals[a];
                    }
                    for a in 0..k {
                        self.regrets[o + a] += vals[a] - v;
                    }
                    v
                } else {
                    for a in 0..k {
                        self.strat_sum[o + a] += sigma[a];
                    }
                    let a = self.sample(&sigma[..k]);
                    self.external(tree.child(&nd, a), player)
                }
            }
        }
    }

    /// Returns (sampled utility / sample probability, tail reach).
    fn outcome(&mut self, node: u32, player: usize, pi_o: f64, s: f64, eps: f64) -> (f64, f64) {
        self.nodes_touched += 1;
        let tree = self.tree;
        let nd = tree.nodes[node as usize];
        match nd.kind {
            Kind::Terminal => (tree.util(&nd, player) / s, 1.0),
            Kind::Chance => {
                let probs: Vec<f64> = (0..nd.num_children as usize).map(|a| tree.chance_prob(&nd, a)).collect();
                let a = self.sample(&probs);
                self.outcome(tree.child(&nd, a), player, pi_o, s, eps)
            }
            Kind::Decision => {
                let abs = &tree.abs_infosets[tree.infosets[nd.infoset as usize].abs as usize];
                let (o, k) = (abs.offset, abs.num_actions);
                let mut sigma = [0.0; 16];
                self.regret_matching(o, k, &mut sigma);
                let mine = nd.player as usize == player;
                let mut q = sigma;
                if mine {
                    for a in 0..k {
                        q[a] = eps / k as f64 + (1.0 - eps) * sigma[a];
                    }
                }
                let a = self.sample(&q[..k]);
                let (u, tail) = if mine {
                    self.outcome(tree.child(&nd, a), player, pi_o, s * q[a], eps)
                } else {
                    self.outcome(tree.child(&nd, a), player, pi_o * sigma[a], s * q[a], eps)
                };
                if mine {
                    let w = u * pi_o;
                    for b in 0..k {
                        let r = if b == a { w * tail * (1.0 - sigma[a]) } else { -w * tail * sigma[a] };
                        self.regrets[o + b] += r;
                    }
                } else {
                    for b in 0..k {
                        self.strat_sum[o + b] += pi_o * sigma[b] / s;
                    }
                }
                (u, tail * sigma[a])
            }
        }
    }

    pub fn iterate(&mut self) {
        self.iterations += 1;
        for player in 0..self.tree.num_players {
            match self.sampling {
                Sampling::External => {
                    self.external(0, player);
                }
                Sampling::Outcome { epsilon } => {
                    self.outcome(0, player, 1.0, 1.0, epsilon);
                }
            }
        }
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

    pub fn memory_bytes(&self) -> usize {
        2 * self.tree.abs_actions * std::mem::size_of::<f64>()
    }
}
