//! Exact heads-up showdown sums with card removal, in O(n) per call.
//!
//! For each hero hand `h` and opponent weights `w`, [`ShowdownPlan::sums`] returns
//! the total opponent weight that `h` beats, ties and loses to, counting only
//! opponent hands that share no card with `h`. Hands are sorted by strength once;
//! each call then walks both lists, and the weight of opponent hands that collide
//! with `h` is removed by inclusion-exclusion over the non-empty subsets of `h`'s
//! cards:
//!
//! `blocked(h) = sum over S ⊆ h, S ≠ ∅ of (-1)^(|S|+1) * W(S)`,
//!
//! where `W(S)` is the weight of opponent hands containing every card of `S`.
//! Two-card hands need 3 subset terms, four-card Omaha hands 15. Every subset that
//! occurs in both a hero hand and an opponent hand gets a dense slot, so a call is
//! pure array arithmetic. The plan is reused for every iteration of a solve,
//! because hand strengths on a fixed board never change.

use std::collections::HashMap;

use pe_cards::CardMask;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sums {
    pub win: f64,
    pub tie: f64,
    pub lose: f64,
}

impl Sums {
    pub fn total(&self) -> f64 {
        self.win + self.tie + self.lose
    }
}

#[derive(Clone, Debug)]
pub struct ShowdownPlan {
    hero_vals: Vec<u32>,
    opp_vals: Vec<u32>,
    hero_order: Vec<u32>,
    opp_order: Vec<u32>,
    /// For each opponent hand, the slots of its card subsets that matter.
    opp_slots: Vec<Vec<u32>>,
    /// For each hero hand, (slot, sign) of its non-empty card subsets that occur in some opponent hand.
    hero_slots: Vec<Vec<(u32, f64)>>,
    n_slots: usize,
}

/// Every non-empty subset of a mask.
fn subsets(m: CardMask) -> impl Iterator<Item = CardMask> {
    let mut sub = m;
    let mut done = m == 0;
    std::iter::from_fn(move || {
        if done {
            return None;
        }
        let cur = sub;
        sub = (sub.wrapping_sub(1)) & m;
        if sub == 0 {
            done = true;
        }
        Some(cur)
    })
}

impl ShowdownPlan {
    pub fn new(hero_vals: &[u32], hero_masks: &[CardMask], opp_vals: &[u32], opp_masks: &[CardMask]) -> ShowdownPlan {
        let mut opp_subsets: HashMap<CardMask, ()> = HashMap::new();
        for &m in opp_masks {
            for s in subsets(m) {
                opp_subsets.insert(s, ());
            }
        }
        let mut slot_of: HashMap<CardMask, u32> = HashMap::new();
        let mut hero_slots = Vec::with_capacity(hero_masks.len());
        for &m in hero_masks {
            let mut v = Vec::new();
            for s in subsets(m) {
                if opp_subsets.contains_key(&s) {
                    let next = slot_of.len() as u32;
                    let slot = *slot_of.entry(s).or_insert(next);
                    let sign = if s.count_ones() % 2 == 1 { 1.0 } else { -1.0 };
                    v.push((slot, sign));
                }
            }
            hero_slots.push(v);
        }
        let opp_slots = opp_masks
            .iter()
            .map(|&m| subsets(m).filter_map(|s| slot_of.get(&s).copied()).collect())
            .collect();
        let mut hero_order: Vec<u32> = (0..hero_vals.len() as u32).collect();
        hero_order.sort_by_key(|&i| hero_vals[i as usize]);
        let mut opp_order: Vec<u32> = (0..opp_vals.len() as u32).collect();
        opp_order.sort_by_key(|&i| opp_vals[i as usize]);
        ShowdownPlan {
            hero_vals: hero_vals.to_vec(),
            opp_vals: opp_vals.to_vec(),
            hero_order,
            opp_order,
            opp_slots,
            hero_slots,
            n_slots: slot_of.len(),
        }
    }

    pub fn hero_len(&self) -> usize {
        self.hero_vals.len()
    }

    #[inline]
    fn blocked(&self, h: usize, acc: &[f64]) -> f64 {
        self.hero_slots[h].iter().map(|&(s, sign)| sign * acc[s as usize]).sum()
    }

    /// Opponent weight each hero hand beats, ties and loses to (card-compatible only).
    pub fn sums(&self, opp_w: &[f64], out: &mut [Sums]) {
        let n_opp = self.opp_vals.len();
        let mut acc_all = vec![0.0; self.n_slots];
        let mut acc_lt = vec![0.0; self.n_slots];
        let mut acc_le = vec![0.0; self.n_slots];
        let mut t_all = 0.0;
        for v in 0..n_opp {
            let w = opp_w[v];
            if w != 0.0 {
                t_all += w;
                for &s in &self.opp_slots[v] {
                    acc_all[s as usize] += w;
                }
            }
        }
        let (mut t_lt, mut t_le) = (0.0, 0.0);
        let (mut p_lt, mut p_le) = (0usize, 0usize);
        for &h in &self.hero_order {
            let h = h as usize;
            let val = self.hero_vals[h];
            while p_lt < n_opp && self.opp_vals[self.opp_order[p_lt] as usize] < val {
                let v = self.opp_order[p_lt] as usize;
                let w = opp_w[v];
                if w != 0.0 {
                    t_lt += w;
                    for &s in &self.opp_slots[v] {
                        acc_lt[s as usize] += w;
                    }
                }
                p_lt += 1;
            }
            while p_le < n_opp && self.opp_vals[self.opp_order[p_le] as usize] <= val {
                let v = self.opp_order[p_le] as usize;
                let w = opp_w[v];
                if w != 0.0 {
                    t_le += w;
                    for &s in &self.opp_slots[v] {
                        acc_le[s as usize] += w;
                    }
                }
                p_le += 1;
            }
            let total = t_all - self.blocked(h, &acc_all);
            let lt = t_lt - self.blocked(h, &acc_lt);
            let le = t_le - self.blocked(h, &acc_le);
            out[h] = Sums { win: lt.max(0.0), tie: (le - lt).max(0.0), lose: (total - le).max(0.0) };
        }
    }

    /// Card-compatible opponent weight for each hero hand (for fold terminals).
    pub fn compatible_totals(&self, opp_w: &[f64], out: &mut [f64]) {
        let mut acc = vec![0.0; self.n_slots];
        let mut t = 0.0;
        for (v, &w) in opp_w.iter().enumerate() {
            if w != 0.0 {
                t += w;
                for &s in &self.opp_slots[v] {
                    acc[s as usize] += w;
                }
            }
        }
        for h in 0..self.hero_vals.len() {
            out[h] = (t - self.blocked(h, &acc)).max(0.0);
        }
    }
}

/// One-shot convenience wrapper around [`ShowdownPlan`].
pub fn showdown_sums(
    hero_vals: &[u32],
    hero_masks: &[CardMask],
    opp_vals: &[u32],
    opp_masks: &[CardMask],
    opp_w: &[f64],
    _hand_size: usize,
) -> Vec<Sums> {
    let plan = ShowdownPlan::new(hero_vals, hero_masks, opp_vals, opp_masks);
    let mut out = vec![Sums::default(); hero_vals.len()];
    plan.sums(opp_w, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use pe_cards::{Game, HandList, Rng, mask_of, parse_cards, parse_hand_list};

    fn brute(hv: &[u32], hm: &[CardMask], ov: &[u32], om: &[CardMask], w: &[f64]) -> Vec<Sums> {
        hv.iter()
            .zip(hm)
            .map(|(&v, &m)| {
                let mut s = Sums::default();
                for i in 0..ov.len() {
                    if om[i] & m != 0 {
                        continue;
                    }
                    if ov[i] < v {
                        s.win += w[i];
                    } else if ov[i] == v {
                        s.tie += w[i];
                    } else {
                        s.lose += w[i];
                    }
                }
                s
            })
            .collect()
    }

    fn check(game: &Game, a: &HandList, b: &HandList, board: &str, seed: u64) {
        let board = parse_cards(board).unwrap();
        let bm = mask_of(&board);
        let a = a.without_dead(bm);
        let b = b.without_dead(bm);
        let pb = crate::settle::PreparedBoard::new(&board);
        let val = |l: &HandList| -> Vec<u32> {
            l.hands.iter().map(|h| crate::settle::high_value(game, &crate::settle::PreparedHand::new(h), &pb)).collect()
        };
        let (va, vb) = (val(&a), val(&b));
        let (ma, mb) = (a.masks(), b.masks());
        let mut rng = Rng::new(seed);
        let w: Vec<f64> = (0..b.len()).map(|_| if rng.next_f64() < 0.2 { 0.0 } else { rng.next_f64() }).collect();
        let fast = showdown_sums(&va, &ma, &vb, &mb, &w, game.hole_cards);
        let slow = brute(&va, &ma, &vb, &mb, &w);
        for (f, s) in fast.iter().zip(&slow) {
            assert!((f.win - s.win).abs() < 1e-9 && (f.tie - s.tie).abs() < 1e-9 && (f.lose - s.lose).abs() < 1e-9, "{f:?} vs {s:?}");
        }
    }

    #[test]
    fn matches_brute_force_holdem() {
        let g = Game::NLH;
        let a = parse_hand_list(&g, "random", 0, None).unwrap();
        let b = parse_hand_list(&g, "22+,A2s+,K9o+,QTs+,T8s+,75s:0.5", 0, None).unwrap();
        check(&g, &a, &b, "Kh8d3c2s2h", 1);
        check(&g, &b, &a, "AsKsQs7d2c", 2);
    }

    #[test]
    fn matches_brute_force_omaha() {
        let g = Game::PLO;
        let a = parse_hand_list(&g, "AAKx,KKQx,QJT9ds,9876:0.5", 0, None).unwrap();
        let b = parse_hand_list(&g, "JJT9,T98x,AKQJ:0.3", 0, None).unwrap();
        check(&g, &a, &b, "Ah9h4d2cTs", 3);
    }

    #[test]
    fn matches_brute_force_short_deck() {
        let g = Game::SHORT_DECK;
        let a = parse_hand_list(&g, "random", 0, None).unwrap();
        let b = parse_hand_list(&g, "66+,AT+,KQ", 0, None).unwrap();
        check(&g, &a, &b, "Ah9h8d7c6s", 4);
    }
}
