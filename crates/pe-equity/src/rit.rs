//! Exact run-it-twice statistics.
//!
//! Running it twice deals two runouts from the same deck (the second from the
//! cards left after the first) and gives each half the pot. Each run is marginally
//! a uniform runout, so the expected share is exactly the run-once equity. The
//! variance changes: `Var(twice) = (E[s^2] + E[s1 s2]) / 2 - mu^2`, where the cross
//! moment `E[s1 s2]` over ordered pairs of disjoint runouts is computed exactly
//! with inclusion-exclusion:
//!
//! `sum over r2 disjoint from r1 of s(r2) = sum over S ⊆ r1 of (-1)^|S| A(S)`,
//! `A(S) = sum of s(r) over runouts r containing S` (with `A(∅)` the total).
//!
//! That costs `2^k` lookups per runout instead of a quadratic pair loop, so even a
//! preflop all-in (k = 5, 1.7 million runouts) is exact.

use pe_cards::{Card, Game, cards_of, mask_of};
use rayon::prelude::*;
use serde::Serialize;

use crate::settle::{PreparedBoard, PreparedHand, add_shares};

#[derive(Clone, Debug, Serialize)]
pub struct RitResult {
    /// Expected pot share (identical for one run and two runs).
    pub equity: Vec<f64>,
    /// Standard deviation of the pot share when the board is run once.
    pub sd_once: Vec<f64>,
    /// Standard deviation of the pot share when it is run twice.
    pub sd_twice: Vec<f64>,
    /// Correlation between the two runs' shares (negative: card removal links them).
    pub corr_runs: Vec<f64>,
    pub runouts: u64,
}

fn binom_table() -> Vec<[u64; 6]> {
    let mut t = vec![[0u64; 6]; 64];
    for (x, row) in t.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = pe_cards::choose(x as u64, j as u64);
        }
    }
    t
}

#[inline]
fn colex(pos: &[usize], b: &[[u64; 6]]) -> usize {
    pos.iter().enumerate().map(|(j, &p)| b[p][j + 1] as usize).sum()
}

pub fn run_it_twice_exact(game: &Game, hands: &[Vec<Card>], board: &[Card], dead: &[Card]) -> Result<RitResult, String> {
    let mut used = mask_of(board) | mask_of(dead);
    for h in hands {
        if h.len() != game.hole_cards {
            return Err(format!("{} needs {} hole cards", game.code(), game.hole_cards));
        }
        let m = mask_of(h);
        if m & used != 0 {
            return Err("a card is used twice".into());
        }
        used |= m;
    }
    let remaining: Vec<Card> = cards_of(game.deck() & !used);
    let n = remaining.len();
    let k = 5 - board.len();
    if k == 0 {
        return Err("the board is complete: nothing to run".into());
    }
    if 2 * k > n {
        return Err("not enough cards to run twice".into());
    }
    let p = hands.len();
    let binom = binom_table();
    // all k-subsets of positions, in colex order
    let mut runouts: Vec<Vec<usize>> = Vec::new();
    let positions: Vec<usize> = (0..n).collect();
    pe_cards::combos::for_each_subset(&positions, k, &mut |s| runouts.push(s.to_vec()));
    let owned: Vec<PreparedHand> = hands.iter().map(|h| PreparedHand::new(h)).collect();
    let prepared: Vec<&PreparedHand> = owned.iter().collect();
    let bmask = mask_of(board);
    let shares: Vec<Vec<f64>> = runouts
        .par_iter()
        .map(|r| {
            let m = r.iter().fold(bmask, |m, &i| m | remaining[i].mask());
            let mut out = vec![0.0; p];
            add_shares(game, &prepared, &[PreparedBoard::from_mask(m)], 1.0, &mut out, &mut Vec::new());
            out
        })
        .collect();
    let nr = runouts.len();
    // A(S) for |S| = 1..=k, indexed by colex rank within each size
    let mut acc: Vec<Vec<f64>> = (1..=k).map(|size| vec![0.0; binom[n][size] as usize * p]).collect();
    let mut total = vec![0.0; p];
    let mut sub = Vec::with_capacity(k);
    for (r, s) in runouts.iter().zip(&shares) {
        for i in 0..p {
            total[i] += s[i];
        }
        for bits in 1u32..(1 << k) {
            sub.clear();
            for (j, &pos) in r.iter().enumerate() {
                if bits & (1 << j) != 0 {
                    sub.push(pos);
                }
            }
            let size = sub.len();
            let base = colex(&sub, &binom) * p;
            let row = &mut acc[size - 1];
            for i in 0..p {
                row[base + i] += s[i];
            }
        }
    }
    let mut cross = vec![0.0; p];
    let mut sq = vec![0.0; p];
    let mut inner = vec![0.0; p];
    for (r, s) in runouts.iter().zip(&shares) {
        inner.copy_from_slice(&total);
        for bits in 1u32..(1 << k) {
            sub.clear();
            for (j, &pos) in r.iter().enumerate() {
                if bits & (1 << j) != 0 {
                    sub.push(pos);
                }
            }
            let size = sub.len();
            let base = colex(&sub, &binom) * p;
            let sign = if size % 2 == 1 { -1.0 } else { 1.0 };
            let row = &acc[size - 1];
            for i in 0..p {
                inner[i] += sign * row[base + i];
            }
        }
        for i in 0..p {
            cross[i] += s[i] * inner[i];
            sq[i] += s[i] * s[i];
        }
    }
    let n2 = binom[n - k][k] as f64;
    let mut out = RitResult { equity: vec![0.0; p], sd_once: vec![0.0; p], sd_twice: vec![0.0; p], corr_runs: vec![0.0; p], runouts: nr as u64 };
    for i in 0..p {
        let mu = total[i] / nr as f64;
        let e2 = sq[i] / nr as f64;
        let e12 = cross[i] / (nr as f64 * n2);
        let var1 = (e2 - mu * mu).max(0.0);
        let var2 = ((e2 + e12) / 2.0 - mu * mu).max(0.0);
        out.equity[i] = mu;
        out.sd_once[i] = var1.sqrt();
        out.sd_twice[i] = var2.sqrt();
        out.corr_runs[i] = if var1 > 0.0 { (e12 - mu * mu) / var1 } else { 0.0 };
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pe_cards::parse_cards;

    /// Brute force over ordered pairs of disjoint runouts (small cases only).
    fn brute(game: &Game, hands: &[Vec<Card>], board: &[Card]) -> (f64, f64, f64) {
        let mut used = mask_of(board);
        for h in hands {
            used |= mask_of(h);
        }
        let rem = cards_of(game.deck() & !used);
        let k = 5 - board.len();
        let mut runs: Vec<(u64, f64)> = Vec::new();
        let owned: Vec<PreparedHand> = hands.iter().map(|h| PreparedHand::new(h)).collect();
        let prepared: Vec<&PreparedHand> = owned.iter().collect();
        pe_cards::combos::for_each_subset(&rem, k, &mut |s| {
            let m = mask_of(s) | mask_of(board);
            let mut out = vec![0.0; hands.len()];
            add_shares(game, &prepared, &[PreparedBoard::from_mask(m)], 1.0, &mut out, &mut Vec::new());
            runs.push((mask_of(s), out[0]));
        });
        let mu = runs.iter().map(|r| r.1).sum::<f64>() / runs.len() as f64;
        let mut acc = 0.0;
        let mut cnt = 0.0;
        for a in &runs {
            for b in &runs {
                if a.0 & b.0 == 0 {
                    let x = (a.1 + b.1) / 2.0;
                    acc += x * x;
                    cnt += 1.0;
                }
            }
        }
        let var2 = acc / cnt - mu * mu;
        let var1 = runs.iter().map(|r| r.1 * r.1).sum::<f64>() / runs.len() as f64 - mu * mu;
        (mu, var1.sqrt(), var2.max(0.0).sqrt())
    }

    #[test]
    fn matches_brute_force_on_turn_and_flop() {
        let g = Game::NLH;
        let hands = vec![parse_cards("AhKh").unwrap(), parse_cards("QsQd").unwrap()];
        for board in ["Th7h2c", "Th7h2c3d"] {
            let b = parse_cards(board).unwrap();
            let r = run_it_twice_exact(&g, &hands, &b, &[]).unwrap();
            let (mu, sd1, sd2) = brute(&g, &hands, &b);
            assert!((r.equity[0] - mu).abs() < 1e-12);
            assert!((r.sd_once[0] - sd1).abs() < 1e-12);
            assert!((r.sd_twice[0] - sd2).abs() < 1e-9, "{} vs {}", r.sd_twice[0], sd2);
            assert!(r.sd_twice[0] < r.sd_once[0]);
            assert!(r.corr_runs[0] < 0.0);
        }
    }
}
