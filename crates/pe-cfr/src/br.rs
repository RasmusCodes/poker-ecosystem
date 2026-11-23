//! Exact best responses, expected values, NashConv and exploitability.
//!
//! A best response for player `i` picks, at each of `i`'s real information sets,
//! the action that maximises `sum over h in I of reach_-i(h) * v_i(h·a)`, where
//! `reach_-i` multiplies chance and the other players' strategy along the path
//! and `v_i` is `i`'s value below `h` under the best response. Decisions are made
//! deepest-first through memoised recursion (perfect recall guarantees no cycles).
//!
//! * NashConv = sum over players of (best-response value - current value). It is
//!   zero exactly at a Nash equilibrium, for any number of players.
//! * Exploitability, reported for two-player zero-sum games, is NashConv / 2: the
//!   average amount a best-responding opponent wins per game.

use crate::tree::{Kind, Strategy, Tree};

/// Expected utility of every player when all follow `s`.
pub fn expected_values(tree: &Tree, s: &Strategy) -> Vec<f64> {
    let n = tree.num_players;
    let mut reach = vec![0.0; tree.nodes.len()];
    reach[0] = 1.0;
    let mut out = vec![0.0; n];
    for idx in 0..tree.nodes.len() {
        let r = reach[idx];
        if r == 0.0 {
            continue;
        }
        let nd = tree.nodes[idx];
        match nd.kind {
            Kind::Terminal => {
                for p in 0..n {
                    out[p] += r * tree.util(&nd, p);
                }
            }
            Kind::Chance => {
                for a in 0..nd.num_children as usize {
                    reach[tree.child(&nd, a) as usize] += r * tree.chance_prob(&nd, a);
                }
            }
            Kind::Decision => {
                let probs = tree.probs_at(s, nd.infoset);
                for a in 0..nd.num_children as usize {
                    reach[tree.child(&nd, a) as usize] += r * probs[a];
                }
            }
        }
    }
    out
}

struct BrCtx<'a> {
    tree: &'a Tree,
    s: &'a Strategy,
    player: usize,
    reach_o: Vec<f64>,
    value: Vec<f64>,
    known: Vec<bool>,
    choice: Vec<i32>,
    members: Vec<Vec<u32>>,
}

impl BrCtx<'_> {
    fn value(&mut self, node: u32) -> f64 {
        let i = node as usize;
        if self.known[i] {
            return self.value[i];
        }
        let tree = self.tree;
        let nd = tree.nodes[i];
        let v = match nd.kind {
            Kind::Terminal => tree.util(&nd, self.player),
            Kind::Chance => (0..nd.num_children as usize).map(|a| tree.chance_prob(&nd, a) * self.value(tree.child(&nd, a))).sum(),
            Kind::Decision => {
                if nd.player as usize == self.player {
                    let a = self.decide(nd.infoset);
                    self.value(tree.child(&nd, a))
                } else {
                    let probs = tree.probs_at(self.s, nd.infoset).to_vec();
                    let mut v = 0.0;
                    for (a, p) in probs.iter().enumerate() {
                        if *p > 0.0 {
                            v += p * self.value(tree.child(&nd, a));
                        }
                    }
                    v
                }
            }
        };
        self.known[i] = true;
        self.value[i] = v;
        v
    }

    fn decide(&mut self, infoset: u32) -> usize {
        if self.choice[infoset as usize] >= 0 {
            return self.choice[infoset as usize] as usize;
        }
        let members = self.members[infoset as usize].clone();
        let k = self.tree.infosets[infoset as usize].actions.len();
        let mut best = (f64::NEG_INFINITY, 0usize);
        for a in 0..k {
            let mut score = 0.0;
            for &h in &members {
                let w = self.reach_o[h as usize];
                if w > 0.0 {
                    let nd = self.tree.nodes[h as usize];
                    score += w * self.value(self.tree.child(&nd, a));
                }
            }
            if score > best.0 + 1e-15 {
                best = (score, a);
            }
        }
        self.choice[infoset as usize] = best.1 as i32;
        best.1
    }
}

/// Value of a best response for `player` against `s`, and the chosen action at each
/// of the player's real information sets (`-1` for sets never decided).
pub fn best_response(tree: &Tree, s: &Strategy, player: usize) -> (f64, Vec<i32>) {
    let n_nodes = tree.nodes.len();
    let mut reach_o = vec![0.0; n_nodes];
    reach_o[0] = 1.0;
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); tree.infosets.len()];
    for idx in 0..n_nodes {
        let nd = tree.nodes[idx];
        let r = reach_o[idx];
        match nd.kind {
            Kind::Terminal => {}
            Kind::Chance => {
                for a in 0..nd.num_children as usize {
                    reach_o[tree.child(&nd, a) as usize] = r * tree.chance_prob(&nd, a);
                }
            }
            Kind::Decision => {
                if nd.player as usize == player {
                    members[nd.infoset as usize].push(idx as u32);
                    for a in 0..nd.num_children as usize {
                        reach_o[tree.child(&nd, a) as usize] = r;
                    }
                } else {
                    let probs = tree.probs_at(s, nd.infoset);
                    for a in 0..nd.num_children as usize {
                        reach_o[tree.child(&nd, a) as usize] = r * probs[a];
                    }
                }
            }
        }
    }
    let mut ctx = BrCtx {
        tree,
        s,
        player,
        reach_o,
        value: vec![0.0; n_nodes],
        known: vec![false; n_nodes],
        choice: vec![-1; tree.infosets.len()],
        members,
    };
    let v = ctx.value(0);
    (v, ctx.choice)
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct NashConvReport {
    /// Sum of every player's best-response gain.
    pub nash_conv: f64,
    /// Each player's gain from deviating to a best response.
    pub gains: Vec<f64>,
    /// Each player's value under the profile.
    pub values: Vec<f64>,
    /// NashConv / 2 for two-player zero-sum games, `None` otherwise.
    pub exploitability: Option<f64>,
}

pub fn nash_conv(tree: &Tree, s: &Strategy) -> NashConvReport {
    let values = expected_values(tree, s);
    let gains: Vec<f64> = (0..tree.num_players).map(|p| (best_response(tree, s, p).0 - values[p]).max(0.0)).collect();
    let nc: f64 = gains.iter().sum();
    let exploitability = if tree.num_players == 2 && tree.is_zero_sum() { Some(nc / 2.0) } else { None };
    NashConvReport { nash_conv: nc, gains, values, exploitability }
}
