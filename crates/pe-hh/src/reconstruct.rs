//! Replays a hand: contributions, uncalled bets, main and side pots, and a check
//! against the summary (total pot, rake, collected amounts). Fills `derived`.

use std::collections::{BTreeMap, BTreeSet};

use crate::model::{ActionType, Derived, Hand, Pot, StreetName};

/// Main and side pots from per-seat contributions. Money from folded players
/// stays in the pots; a level nobody still in can win joins the next pot.
pub fn side_pots(contrib: &BTreeMap<u32, i64>, folded: &BTreeSet<u32>) -> Vec<Pot> {
    let mut levels: Vec<i64> = contrib.values().copied().filter(|&c| c > 0).collect();
    levels.sort_unstable();
    levels.dedup();
    let mut raw: Vec<Pot> = Vec::new();
    let mut prev = 0;
    for lv in levels {
        let amount: i64 = contrib.values().map(|&c| c.min(lv) - c.min(prev)).sum();
        let eligible: Vec<u32> = contrib.iter().filter(|(s, c)| **c >= lv && !folded.contains(s)).map(|(s, _)| *s).collect();
        raw.push(Pot { amount, eligible });
        prev = lv;
    }
    // carry dead levels forward, then merge neighbours with the same eligibility
    let mut out: Vec<Pot> = Vec::new();
    let mut carry = 0;
    for p in raw {
        if p.eligible.is_empty() {
            carry += p.amount;
            continue;
        }
        let amount = p.amount + carry;
        carry = 0;
        match out.last_mut() {
            Some(last) if last.eligible == p.eligible => last.amount += amount,
            _ => out.push(Pot { amount, eligible: p.eligible }),
        }
    }
    if carry > 0 {
        match out.last_mut() {
            Some(last) => last.amount += carry,
            None => out.push(Pot { amount: carry, eligible: Vec::new() }),
        }
    }
    out
}

/// Seats that folded at some point.
pub fn folded_seats(hand: &Hand) -> BTreeSet<u32> {
    hand.streets.iter().flat_map(|s| s.actions.iter()).filter(|a| a.kind == ActionType::Fold).map(|a| a.seat).collect()
}

/// Total chips each seat put in (posts and actions, uncalled bets returned).
pub fn contributions(hand: &Hand) -> BTreeMap<u32, i64> {
    let mut c: BTreeMap<u32, i64> = hand.dealt_seats().into_iter().map(|s| (s, 0)).collect();
    for p in &hand.posts {
        *c.entry(p.seat).or_insert(0) += p.amount;
    }
    for st in &hand.streets {
        for a in &st.actions {
            *c.entry(a.seat).or_insert(0) += a.amount;
        }
    }
    for u in &hand.uncalled {
        *c.entry(u.seat).or_insert(0) -= u.amount;
    }
    c
}

pub fn reconstruct(hand: &mut Hand) {
    let mut issues = Vec::new();
    let contrib = contributions(hand);
    let folded = folded_seats(hand);
    let pots = side_pots(&contrib, &folded);
    let total: i64 = contrib.values().sum();
    if hand.total_pot > 0 && total != hand.total_pot {
        issues.push(format!("contributions add to {total}, the summary says {}", hand.total_pot));
    }
    let collected: i64 = hand.collected.iter().map(|c| c.amount).sum();
    let stated = if hand.total_pot > 0 { hand.total_pot } else { total };
    if collected + hand.rake != stated {
        issues.push(format!("collected {collected} plus rake {} differs from the pot {stated}", hand.rake));
    }
    for c in &hand.collected {
        if folded.contains(&c.seat) {
            issues.push(format!("seat {} collected after folding", c.seat));
        }
    }
    for p in &hand.players {
        let c = *contrib.get(&p.seat).unwrap_or(&0);
        if c > p.stack {
            issues.push(format!("seat {} put in {c}, more than its stack {}", p.seat, p.stack));
        }
        if c < 0 {
            issues.push(format!("seat {} has a negative contribution", p.seat));
        }
    }
    let mut net: BTreeMap<u32, i64> = contrib.iter().map(|(s, c)| (*s, -c)).collect();
    for c in &hand.collected {
        *net.entry(c.seat).or_insert(0) += c.amount;
    }
    let preflop_aggressor = hand
        .street(StreetName::Preflop)
        .and_then(|s| s.actions.iter().filter(|a| a.kind == ActionType::Raise || a.kind == ActionType::Bet).next_back())
        .map(|a| a.seat);
    let still_in: Vec<u32> = hand.dealt_seats().into_iter().filter(|s| !folded.contains(s)).collect();
    let showdown_seats = if still_in.len() >= 2 { still_in } else { Vec::new() };
    let old = hand.derived.take().unwrap_or_default();
    hand.derived = Some(Derived {
        ok: issues.is_empty(),
        issues,
        contrib,
        net,
        pots,
        players_dealt: hand.dealt_seats().len() as u32,
        preflop_aggressor,
        showdown_seats,
        allin: old.allin,
        river_strength: old.river_strength,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_way_all_in_side_pots() {
        // A all-in 20, B and C 50 each, D folded after 5
        let contrib: BTreeMap<u32, i64> = [(1, 20), (2, 50), (3, 50), (4, 5)].into_iter().collect();
        let folded: BTreeSet<u32> = [4].into_iter().collect();
        let pots = side_pots(&contrib, &folded);
        assert_eq!(pots.len(), 2);
        assert_eq!(pots[0], Pot { amount: 65, eligible: vec![1, 2, 3] });
        assert_eq!(pots[1], Pot { amount: 60, eligible: vec![2, 3] });
    }
}
