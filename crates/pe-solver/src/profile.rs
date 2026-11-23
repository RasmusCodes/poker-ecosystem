//! Opponent profiles: observed frequencies turned into a complete strategy.
//!
//! A profile says how often a player folds to a bet, raises, bets when checked
//! to, leads, and what share of bets are bluffs. [`build_model`] starts from the
//! player's equilibrium strategy and overrides every node a profile parameter
//! covers with a *threshold strategy* on the range that reaches the node:
//!
//! * facing a bet: raise with the strongest `raise` share, fold the weakest `fold`
//!   share, call with the rest;
//! * betting: bet the strongest `bet * (1 - bluff)` share for value and the weakest
//!   `bet * bluff` share as bluffs, check the middle.
//!
//! Strength is the hand's high-hand value on the current board (exact on the
//! river; on earlier streets it ignores draws, which is a stated simplification).
//! Threshold strategies are how frequency statistics are usually turned into
//! ranges; they are a model of the opponent, not an observation of it.

use pe_equity::settle::{PreparedBoard, PreparedHand, high_value};
use serde::{Deserialize, Serialize};

use crate::hu::HuSolver;
use crate::tree::{Action, NodeKind};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Profile {
    /// Fold to a bet (first bet of the street).
    pub fold_to_bet: Option<f64>,
    /// Raise when facing a bet.
    pub raise_vs_bet: Option<f64>,
    /// Fold when facing a raise.
    pub fold_to_raise: Option<f64>,
    /// Bet when checked to.
    pub bet_when_checked_to: Option<f64>,
    /// Bet when first to act on the street (lead / donk).
    pub lead: Option<f64>,
    /// Share of bets and raises that are bluffs (the weakest hands).
    #[serde(default = "default_bluff")]
    pub bluff_share: f64,
    /// Preferred bet size as a fraction of the pot (the closest menu size is used).
    pub size: Option<f64>,
}

fn default_bluff() -> f64 {
    0.33
}

fn street_actions(path: &str) -> Vec<String> {
    let seg = path.rsplit('|').next().unwrap_or("");
    seg.split('-').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect()
}

/// Picks the bet/raise action whose size is closest to `size` (or the smallest).
fn choose_aggressive(actions: &[Action], size: Option<f64>) -> Option<usize> {
    let cands: Vec<(usize, f64)> = actions
        .iter()
        .enumerate()
        .filter_map(|(i, a)| match a {
            Action::Bet { size: s, .. } | Action::Raise { size: s, .. } => Some((i, *s)),
            Action::AllIn { .. } => Some((i, 100.0)),
            _ => None,
        })
        .collect();
    if cands.is_empty() {
        return None;
    }
    let target = size.unwrap_or(0.0);
    cands.iter().min_by(|a, b| (a.1 - target).abs().partial_cmp(&(b.1 - target).abs()).unwrap()).map(|c| c.0)
}

/// Assigns `share`s of the reaching weight to actions by strength: `top` share
/// from the strongest hands, `bottom` share from the weakest, rest in the middle.
/// Returns per hand (top, middle, bottom) fractions (hands at a boundary split).
fn thresholds(strength: &[u32], reach: &[f64], top: f64, bottom: f64) -> Vec<(f64, f64, f64)> {
    let n = strength.len();
    let total: f64 = reach.iter().sum();
    let mut out = vec![(0.0, 1.0, 0.0); n];
    if total <= 0.0 {
        return out;
    }
    let mut order: Vec<usize> = (0..n).filter(|&h| reach[h] > 0.0).collect();
    order.sort_by(|&a, &b| strength[b].cmp(&strength[a]));
    let top_w = top.clamp(0.0, 1.0) * total;
    let bottom_w = bottom.clamp(0.0, 1.0 - top.clamp(0.0, 1.0)) * total;
    // walk from the top
    let mut acc = 0.0;
    for &h in &order {
        let w = reach[h];
        let take = ((top_w - acc).max(0.0)).min(w);
        out[h].0 = take / w;
        acc += w;
    }
    let mut acc = 0.0;
    for &h in order.iter().rev() {
        let w = reach[h];
        let room = w * (1.0 - out[h].0);
        let take = ((bottom_w - acc).max(0.0)).min(room);
        out[h].2 = if w > 0.0 { take / w } else { 0.0 };
        acc += take;
        if acc >= bottom_w {
            // keep walking only to fill remaining
        }
    }
    for o in out.iter_mut() {
        o.1 = (1.0 - o.0 - o.2).max(0.0);
    }
    out
}

/// Builds a complete model strategy for `player`: the solver's average strategy,
/// overridden wherever the profile has a parameter. Returns one entry per tree
/// node (`None` for nodes that are not `player`'s decisions).
pub fn build_model(solver: &HuSolver, player: usize, profile: &Profile) -> Vec<Option<Vec<f64>>> {
    let tree = &solver.tree;
    let spot = &solver.spot;
    let n = solver.hands(player);
    let prepared: Vec<PreparedHand> = spot.ranges[player].hands.iter().map(|h| PreparedHand::new(h)).collect();
    let mut model: Vec<Option<Vec<f64>>> = vec![None; tree.nodes.len()];
    let mut stack: Vec<(usize, Vec<f64>)> = vec![(0, solver.weights[player].clone())];
    while let Some((node, reach)) = stack.pop() {
        let nd = &tree.nodes[node];
        match &nd.kind {
            NodeKind::Terminal { .. } => {}
            NodeKind::Chance { cards, children } => {
                for (c, &ch) in cards.iter().zip(children) {
                    let cm = c.mask();
                    let r: Vec<f64> = reach.iter().zip(&spot.masks[player]).map(|(w, m)| if m & cm != 0 { 0.0 } else { *w }).collect();
                    stack.push((ch, r));
                }
            }
            NodeKind::Decision { player: p, actions, children } => {
                if *p != player {
                    for &ch in children {
                        stack.push((ch, reach.clone()));
                    }
                    continue;
                }
                let k = actions.len();
                let mut sigma = solver.average_strategy(node);
                let acts = street_actions(&nd.path);
                let facing = actions.iter().any(|a| matches!(a, Action::Fold));
                let aggressions = acts.iter().filter(|c| c.starts_with('b') || c.starts_with('r') || c == &"a").count();
                let board = PreparedBoard::from_mask(nd.board);
                let strength: Vec<u32> =
                    prepared.iter().map(|h| if h.mask & nd.board != 0 { 0 } else { high_value(&spot.game, h, &board) }).collect();
                let mut apply = |top: f64, bottom: f64, top_a: usize, mid_a: usize, bottom_a: usize| {
                    let th = thresholds(&strength, &reach, top, bottom);
                    for h in 0..n {
                        for a in 0..k {
                            sigma[a * n + h] = 0.0;
                        }
                        sigma[top_a * n + h] += th[h].0;
                        sigma[mid_a * n + h] += th[h].1;
                        sigma[bottom_a * n + h] += th[h].2;
                    }
                };
                if facing {
                    let fold_a = actions.iter().position(|a| matches!(a, Action::Fold)).unwrap();
                    let call_a = actions.iter().position(|a| matches!(a, Action::Call)).unwrap();
                    let raise_a = choose_aggressive(actions, profile.size);
                    let fold = if aggressions >= 2 { profile.fold_to_raise } else { profile.fold_to_bet };
                    if let Some(f) = fold {
                        let r = if aggressions >= 2 { 0.0 } else { profile.raise_vs_bet.unwrap_or(0.0) };
                        match raise_a {
                            Some(ra) if r > 0.0 => apply(r, f, ra, call_a, fold_a),
                            _ => apply(0.0, f, call_a, call_a, fold_a),
                        }
                    }
                } else {
                    let check_a = actions.iter().position(|a| matches!(a, Action::Check)).unwrap();
                    let freq = if acts.is_empty() { profile.lead } else { profile.bet_when_checked_to };
                    if let (Some(b), Some(bet_a)) = (freq, choose_aggressive(actions, profile.size)) {
                        let bluff = profile.bluff_share.clamp(0.0, 1.0);
                        // value bets from the top, bluffs from the bottom, both with the same action
                        let th = thresholds(&strength, &reach, b * (1.0 - bluff), b * bluff);
                        for h in 0..n {
                            for a in 0..k {
                                sigma[a * n + h] = 0.0;
                            }
                            sigma[bet_a * n + h] = th[h].0 + th[h].2;
                            sigma[check_a * n + h] = th[h].1;
                        }
                    }
                }
                for (a, &ch) in children.iter().enumerate() {
                    let r: Vec<f64> = (0..n).map(|h| reach[h] * sigma[a * n + h]).collect();
                    stack.push((ch, r));
                }
                model[node] = Some(sigma);
            }
        }
    }
    model
}

/// Frequency of each action at a node for a strategy, weighted by the reaching range.
pub fn action_frequencies(sigma: &[f64], reach: &[f64], k: usize) -> Vec<f64> {
    let n = reach.len();
    let total: f64 = reach.iter().sum();
    (0..k).map(|a| if total > 0.0 { (0..n).map(|h| reach[h] * sigma[a * n + h]).sum::<f64>() / total } else { 0.0 }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thresholds_split_weight() {
        let strength = vec![10, 20, 30, 40];
        let reach = vec![1.0, 1.0, 1.0, 1.0];
        let th = thresholds(&strength, &reach, 0.25, 0.5);
        // strongest (40) is top; two weakest (10, 20) are bottom
        assert_eq!(th[3], (1.0, 0.0, 0.0));
        assert_eq!(th[0], (0.0, 0.0, 1.0));
        assert_eq!(th[1], (0.0, 0.0, 1.0));
        assert_eq!(th[2], (0.0, 1.0, 0.0));
        let th = thresholds(&strength, &reach, 0.1, 0.0);
        assert!((th[3].0 - 0.4).abs() < 1e-12);
    }
}
