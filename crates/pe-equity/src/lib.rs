//! Equity: each player's expected share of the pot.
//!
//! * [`equity_exact`] enumerates every remaining runout (in parallel), for any
//!   number of players, one or two boards, high or hi-lo games.
//! * [`equity_monte_carlo`] samples runouts and reports standard errors.
//! * [`range_equity_monte_carlo`] samples hands from weighted ranges with exact
//!   card removal (rejection of colliding deals).
//! * [`sweep`] is the exact card-removal showdown sweep shared with the solver,
//!   and [`range_vs_range_exact`] uses it for heads-up range equities.
//! * [`rit`] computes exact run-it-twice expectations and variances.

pub mod rit;
pub mod settle;
pub mod sweep;

use pe_cards::{Card, CardMask, Game, HandList, Rng, cards_of, mask_of};
use rayon::prelude::*;
use serde::Serialize;
use settle::{PreparedBoard, PreparedHand, add_shares};

#[derive(Clone, Debug, Serialize)]
pub struct EquityResult {
    /// Expected pot share of each player (sums to 1).
    pub equity: Vec<f64>,
    /// Standard error of each share (0 when exact).
    pub std_error: Vec<f64>,
    /// Runouts enumerated or sampled.
    pub runouts: u64,
    pub exact: bool,
}

fn validate(game: &Game, hands: &[Vec<Card>], boards: &[Vec<Card>], dead: &[Card]) -> Result<CardMask, String> {
    if hands.len() < 2 || hands.len() > settle::MAX_PLAYERS {
        return Err(format!("need 2 to {} hands", settle::MAX_PLAYERS));
    }
    if boards.is_empty() || boards.len() > 2 {
        return Err("one or two boards".into());
    }
    let mut used: CardMask = 0;
    let deck = game.deck();
    let add = |c: &Card, used: &mut CardMask| -> Result<(), String> {
        if c.mask() & deck == 0 {
            return Err(format!("{c} is not in the {} deck", game.code()));
        }
        if *used & c.mask() != 0 {
            return Err(format!("{c} is used twice"));
        }
        *used |= c.mask();
        Ok(())
    };
    for h in hands {
        if h.len() != game.hole_cards {
            return Err(format!("{} needs {} hole cards per hand", game.code(), game.hole_cards));
        }
        for c in h {
            add(c, &mut used)?;
        }
    }
    for b in boards {
        if b.len() > 5 {
            return Err("a board has at most five cards".into());
        }
        for c in b {
            add(c, &mut used)?;
        }
    }
    for c in dead {
        add(c, &mut used)?;
    }
    let need: usize = boards.iter().map(|b| 5 - b.len()).sum();
    let left = (deck & !used).count_ones() as usize;
    if need > left {
        return Err(format!("{need} cards needed but only {left} left"));
    }
    Ok(used)
}

/// Enumerates every completion of the boards. Cost grows as C(remaining, missing);
/// heads-up hold'em preflop is 1,712,304 runouts.
pub fn equity_exact(game: &Game, hands: &[Vec<Card>], boards: &[Vec<Card>], dead: &[Card]) -> Result<EquityResult, String> {
    let used = validate(game, hands, boards, dead)?;
    let remaining: Vec<Card> = cards_of(game.deck() & !used);
    let owned: Vec<PreparedHand> = hands.iter().map(|h| PreparedHand::new(h)).collect();
    let prepared: Vec<&PreparedHand> = owned.iter().collect();
    let n = hands.len();
    let need: Vec<usize> = boards.iter().map(|b| 5 - b.len()).collect();
    let base: Vec<CardMask> = boards.iter().map(|b| mask_of(b)).collect();
    let rem_n = remaining.len();

    // Parallelise over the first missing card of the first board (or over nothing).
    let total_need: usize = need.iter().sum();
    let (sums, count) = if total_need == 0 {
        let pb: Vec<PreparedBoard> = base.iter().map(|&m| PreparedBoard::from_mask(m)).collect();
        let mut out = vec![0.0; n];
        add_shares(game, &prepared, &pb, 1.0, &mut out, &mut Vec::new());
        (out, 1u64)
    } else {
        (0..rem_n)
            .into_par_iter()
            .map(|first| {
                let mut out = vec![0.0; n];
                let mut count = 0u64;
                let mut scratch = Vec::with_capacity(n);
                let mut pb = [PreparedBoard::from_mask(0); 2];
                enumerate_from(&remaining, first, &need, &base, &mut |boards_masks| {
                    for (i, &m) in boards_masks.iter().enumerate() {
                        pb[i] = PreparedBoard::from_mask(m);
                    }
                    add_shares(game, &prepared, &pb[..boards_masks.len()], 1.0, &mut out, &mut scratch);
                    count += 1;
                });
                (out, count)
            })
            .reduce(|| (vec![0.0; n], 0), |(mut a, ca), (b, cb)| {
                for i in 0..n {
                    a[i] += b[i];
                }
                (a, ca + cb)
            })
    };
    let equity = sums.iter().map(|s| s / count as f64).collect();
    Ok(EquityResult { equity, std_error: vec![0.0; n], runouts: count, exact: true })
}

/// Calls `f` with the completed board masks for every completion whose first dealt
/// card (lowest index into `remaining` for board 0) is `first`.
fn enumerate_from(remaining: &[Card], first: usize, need: &[usize], base: &[CardMask], f: &mut impl FnMut(&[CardMask])) {
    // board 0 first: choose need[0] cards with the smallest index = first
    let k0 = need[0];
    let mut boards = base.to_vec();
    if k0 == 0 {
        // no card for board 0: parallelise over board 1's first card instead
        if first == 0 {
            if need.len() > 1 {
                for_each_k_subset(remaining, need[1], 0, 0, &mut |m| {
                    boards[1] = base[1] | m;
                    f(&boards);
                });
            } else {
                f(&boards);
            }
        }
        return;
    }
    let first_mask = remaining[first].mask();
    for_each_k_subset(remaining, k0 - 1, first + 1, first_mask, &mut |m0| {
        boards[0] = base[0] | m0;
        if need.len() > 1 && need[1] > 0 {
            let rest: Vec<Card> = remaining.iter().copied().filter(|c| c.mask() & m0 == 0).collect();
            let b0 = boards[0];
            for_each_k_subset(&rest, need[1], 0, 0, &mut |m1| {
                let bs = [b0, base[1] | m1];
                f(&bs);
            });
        } else {
            f(&boards);
        }
    });
}

/// Calls `f(prefix | subset)` for every k-subset of `cards[start..]`.
fn for_each_k_subset(cards: &[Card], k: usize, start: usize, prefix: CardMask, f: &mut impl FnMut(CardMask)) {
    if k == 0 {
        f(prefix);
        return;
    }
    let n = cards.len();
    if n < start + k {
        return;
    }
    for i in start..=n - k {
        for_each_k_subset(cards, k - 1, i + 1, prefix | cards[i].mask(), f);
    }
}

/// Completes the boards with uniformly random cards from `cards` that are not in
/// `taken`, by rejection (at most ten cards are needed, so collisions are rare).
#[inline]
fn deal_boards(rng: &mut Rng, cards: &[Card], taken: CardMask, need: &[usize], base: &[CardMask], out: &mut [PreparedBoard; 2]) {
    let mut taken = taken | base.iter().fold(0, |a, b| a | b);
    for (b, (&k, &m0)) in need.iter().zip(base).enumerate() {
        let mut m = m0;
        for _ in 0..k {
            loop {
                let c = cards[rng.below(cards.len() as u64) as usize].mask();
                if c & taken == 0 {
                    taken |= c;
                    m |= c;
                    break;
                }
            }
        }
        out[b] = PreparedBoard::from_mask(m);
    }
}

/// Samples `samples` runouts uniformly (with a deterministic seed).
pub fn equity_monte_carlo(
    game: &Game,
    hands: &[Vec<Card>],
    boards: &[Vec<Card>],
    dead: &[Card],
    samples: u64,
    seed: u64,
) -> Result<EquityResult, String> {
    let used = validate(game, hands, boards, dead)?;
    let remaining: Vec<Card> = cards_of(game.deck() & !used);
    let owned: Vec<PreparedHand> = hands.iter().map(|h| PreparedHand::new(h)).collect();
    let prepared: Vec<&PreparedHand> = owned.iter().collect();
    let n = hands.len();
    let need: Vec<usize> = boards.iter().map(|b| 5 - b.len()).collect();
    let base: Vec<CardMask> = boards.iter().map(|b| mask_of(b)).collect();
    let chunks = 64u64;
    let per = samples.div_ceil(chunks);
    let (sum, sumsq, count) = (0..chunks)
        .into_par_iter()
        .map(|chunk| {
            let mut rng = Rng::stream(seed, chunk);
            let mut sum = vec![0.0; n];
            let mut sumsq = vec![0.0; n];
            let mut one = vec![0.0; n];
            let mut scratch = Vec::with_capacity(n);
            let mut pb = [PreparedBoard::from_mask(0); 2];
            let todo = per.min(samples.saturating_sub(chunk * per));
            for _ in 0..todo {
                deal_boards(&mut rng, &remaining, used, &need, &base, &mut pb);
                one.iter_mut().for_each(|x| *x = 0.0);
                add_shares(game, &prepared, &pb[..need.len()], 1.0, &mut one, &mut scratch);
                for i in 0..n {
                    sum[i] += one[i];
                    sumsq[i] += one[i] * one[i];
                }
            }
            (sum, sumsq, todo)
        })
        .reduce(|| (vec![0.0; n], vec![0.0; n], 0), |(mut a, mut b, ca), (c, d, cc)| {
            for i in 0..n {
                a[i] += c[i];
                b[i] += d[i];
            }
            (a, b, ca + cc)
        });
    let cnt = count.max(1) as f64;
    let equity: Vec<f64> = sum.iter().map(|s| s / cnt).collect();
    let std_error = (0..n).map(|i| ((sumsq[i] / cnt - equity[i] * equity[i]).max(0.0) / cnt).sqrt()).collect();
    Ok(EquityResult { equity, std_error, runouts: count, exact: false })
}

/// Monte Carlo equity of weighted ranges: each sample draws one hand per player by
/// weight (redrawing the whole deal on a card collision, which is exact card
/// removal) and one runout.
pub fn range_equity_monte_carlo(
    game: &Game,
    ranges: &[HandList],
    boards: &[Vec<Card>],
    dead: &[Card],
    samples: u64,
    seed: u64,
) -> Result<EquityResult, String> {
    if ranges.len() < 2 {
        return Err("need at least two ranges".into());
    }
    let board_mask: CardMask = boards.iter().map(|b| mask_of(b)).fold(0, |a, b| a | b) | mask_of(dead);
    let lists: Vec<HandList> = ranges.iter().map(|r| r.without_dead(board_mask)).collect();
    for (i, l) in lists.iter().enumerate() {
        if l.is_empty() {
            return Err(format!("range {} is empty after removing board and dead cards", i + 1));
        }
    }
    let n = lists.len();
    let prepared: Vec<Vec<PreparedHand>> = lists.iter().map(|l| l.hands.iter().map(|h| PreparedHand::new(h)).collect()).collect();
    let cumulative: Vec<Vec<f64>> = lists
        .iter()
        .map(|l| {
            let mut acc = 0.0;
            l.weights.iter().map(|w| {
                acc += w;
                acc
            }).collect()
        })
        .collect();
    let need: Vec<usize> = boards.iter().map(|b| 5 - b.len()).collect();
    let base: Vec<CardMask> = boards.iter().map(|b| mask_of(b)).collect();
    let deck_cards: Vec<Card> = cards_of(game.deck() & !board_mask);
    let chunks = 64u64;
    let per = samples.div_ceil(chunks);
    let (sum, sumsq, count, rejected) = (0..chunks)
        .into_par_iter()
        .map(|chunk| {
            let mut rng = Rng::stream(seed, chunk);
            let mut sum = vec![0.0; n];
            let mut sumsq = vec![0.0; n];
            let mut one = vec![0.0; n];
            let mut scratch = Vec::with_capacity(n);
            let mut pb = [PreparedBoard::from_mask(0); 2];
            let mut rejected = 0u64;
            let todo = per.min(samples.saturating_sub(chunk * per));
            let mut picks = vec![0usize; n];
            let mut done = 0;
            while done < todo {
                let mut used: CardMask = 0;
                let mut ok = true;
                for p in 0..n {
                    let c = &cumulative[p];
                    let x = rng.next_f64() * c[c.len() - 1];
                    let idx = c.partition_point(|&v| v <= x).min(c.len() - 1);
                    let m = prepared[p][idx].mask;
                    if used & m != 0 {
                        ok = false;
                        break;
                    }
                    used |= m;
                    picks[p] = idx;
                }
                if !ok {
                    rejected += 1;
                    if rejected > 1000 + 1000 * (done + 1) {
                        break; // ranges almost never compatible
                    }
                    continue;
                }
                deal_boards(&mut rng, &deck_cards, used | board_mask, &need, &base, &mut pb);
                let hands: Vec<&PreparedHand> = (0..n).map(|p| &prepared[p][picks[p]]).collect();
                one.iter_mut().for_each(|x| *x = 0.0);
                add_shares(game, &hands, &pb[..need.len()], 1.0, &mut one, &mut scratch);
                for i in 0..n {
                    sum[i] += one[i];
                    sumsq[i] += one[i] * one[i];
                }
                done += 1;
            }
            (sum, sumsq, done, rejected)
        })
        .reduce(|| (vec![0.0; n], vec![0.0; n], 0, 0), |(mut a, mut b, ca, ra), (c, d, cc, rc)| {
            for i in 0..n {
                a[i] += c[i];
                b[i] += d[i];
            }
            (a, b, ca + cc, ra + rc)
        });
    if count == 0 {
        return Err(format!("the ranges are incompatible ({rejected} colliding deals)"));
    }
    let cnt = count as f64;
    let equity: Vec<f64> = sum.iter().map(|s| s / cnt).collect();
    let std_error = (0..n).map(|i| ((sumsq[i] / cnt - equity[i] * equity[i]).max(0.0) / cnt).sqrt()).collect();
    Ok(EquityResult { equity, std_error, runouts: count, exact: false })
}

/// Heads-up range-vs-range equity by exact enumeration of the runouts, using the
/// card-removal sweep on each runout. Practical from the flop on (990 runouts in
/// hold'em); per-hand equities are returned for both ranges.
#[derive(Clone, Debug, Serialize)]
pub struct RangeEquity {
    /// Overall equity of each range (weighted by the joint, card-compatible deal).
    pub equity: [f64; 2],
    /// Per-hand equity of range 0's hands and range 1's hands against the other range.
    pub hand_equity: [Vec<f64>; 2],
    /// Weight of compatible opposing hands for each hand (for weighting).
    pub hand_weight: [Vec<f64>; 2],
    pub runouts: u64,
}

pub fn range_vs_range_exact(game: &Game, r0: &HandList, r1: &HandList, board: &[Card]) -> Result<RangeEquity, String> {
    if game.hilo {
        return Err("hi-lo range equity uses the Monte Carlo path".into());
    }
    let bmask = mask_of(board);
    let a = r0.without_dead(bmask);
    let b = r1.without_dead(bmask);
    let remaining: Vec<Card> = cards_of(game.deck() & !bmask);
    let k = 5 - board.len();
    let mut runouts: Vec<CardMask> = Vec::new();
    for_each_k_subset(&remaining, k, 0, 0, &mut |m| runouts.push(m));
    let pa: Vec<PreparedHand> = a.hands.iter().map(|h| PreparedHand::new(h)).collect();
    let pb: Vec<PreparedHand> = b.hands.iter().map(|h| PreparedHand::new(h)).collect();
    let hand_size = game.hole_cards;
    let (eq0, w0, eq1, w1) = runouts
        .par_iter()
        .map(|&run| {
            let full = PreparedBoard::from_mask(bmask | run);
            // hands that use a runout card are dead on this runout
            let wa: Vec<f64> = a.weights.iter().zip(&pa).map(|(w, h)| if h.mask & run != 0 { 0.0 } else { *w }).collect();
            let wb: Vec<f64> = b.weights.iter().zip(&pb).map(|(w, h)| if h.mask & run != 0 { 0.0 } else { *w }).collect();
            let va: Vec<u32> = pa.iter().map(|h| settle::high_value(game, h, &full)).collect();
            let vb: Vec<u32> = pb.iter().map(|h| settle::high_value(game, h, &full)).collect();
            let ma: Vec<CardMask> = pa.iter().map(|h| h.mask).collect();
            let mb: Vec<CardMask> = pb.iter().map(|h| h.mask).collect();
            let sa = sweep::showdown_sums(&va, &ma, &vb, &mb, &wb, hand_size);
            let sb = sweep::showdown_sums(&vb, &mb, &va, &ma, &wa, hand_size);
            let mut e0 = vec![0.0; pa.len()];
            let mut t0 = vec![0.0; pa.len()];
            for i in 0..pa.len() {
                if wa[i] > 0.0 {
                    e0[i] = sa[i].win + 0.5 * sa[i].tie;
                    t0[i] = sa[i].total();
                }
            }
            let mut e1 = vec![0.0; pb.len()];
            let mut t1 = vec![0.0; pb.len()];
            for i in 0..pb.len() {
                if wb[i] > 0.0 {
                    e1[i] = sb[i].win + 0.5 * sb[i].tie;
                    t1[i] = sb[i].total();
                }
            }
            (e0, t0, e1, t1)
        })
        .reduce(
            || (vec![0.0; pa.len()], vec![0.0; pa.len()], vec![0.0; pb.len()], vec![0.0; pb.len()]),
            |(mut a0, mut a1, mut a2, mut a3), (b0, b1, b2, b3)| {
                for i in 0..a0.len() {
                    a0[i] += b0[i];
                    a1[i] += b1[i];
                }
                for i in 0..a2.len() {
                    a2[i] += b2[i];
                    a3[i] += b3[i];
                }
                (a0, a1, a2, a3)
            },
        );
    let hand_eq0: Vec<f64> = eq0.iter().zip(&w0).map(|(e, t)| if *t > 0.0 { e / t } else { 0.0 }).collect();
    let hand_eq1: Vec<f64> = eq1.iter().zip(&w1).map(|(e, t)| if *t > 0.0 { e / t } else { 0.0 }).collect();
    let tot0: f64 = a.weights.iter().zip(&w0).map(|(w, t)| w * t).sum();
    let num0: f64 = a.weights.iter().zip(&eq0).map(|(w, e)| w * e).sum();
    let overall0 = if tot0 > 0.0 { num0 / tot0 } else { 0.0 };
    // per-hand weights are summed over runouts; normalise to a per-runout mean
    let nrun = runouts.len() as f64;
    Ok(RangeEquity {
        equity: [overall0, 1.0 - overall0],
        hand_equity: [hand_eq0, hand_eq1],
        hand_weight: [w0.iter().map(|t| t / nrun).collect(), w1.iter().map(|t| t / nrun).collect()],
        runouts: runouts.len() as u64,
    })
}

/// Equity of every preflop class against a uniformly random hand (Monte Carlo),
/// returned strongest first as `(class index, equity)`. Used for `x%` ranges.
pub fn preflop_ranking(samples_per_class: u64, seed: u64) -> Vec<(usize, f64)> {
    let game = Game::NLH;
    let mut out: Vec<(usize, f64)> = (0..pe_cards::NUM_CLASSES)
        .into_par_iter()
        .map(|class| {
            let combos = pe_cards::combos::class_combos(class);
            let (a, b) = pe_cards::combo_cards(combos[0]);
            let mut opp = HandList::default();
            let used = a.mask() | b.mask();
            for i in 0..pe_cards::NUM_COMBOS {
                let (c, d) = pe_cards::combo_cards(i);
                if (c.mask() | d.mask()) & used == 0 {
                    opp.push(vec![c, d], 1.0);
                }
            }
            let mut me = HandList::default();
            me.push(vec![a, b], 1.0);
            let r = range_equity_monte_carlo(&game, &[me, opp], &[vec![]], &[], samples_per_class, seed ^ class as u64).unwrap();
            (class, r.equity[0])
        })
        .collect();
    out.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
    out
}

#[cfg(test)]
mod tests;
