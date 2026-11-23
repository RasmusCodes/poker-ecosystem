//! All-in expected value and river showdown strength.
//!
//! * `derived.allin`: when two or more players are all-in before the river with
//!   their cards known, each pot is shared by equity among its eligible players.
//!   Equity is exact when at most 50,000 runouts remain (any flop or turn all-in,
//!   and multiway pots on later streets), otherwise Monte Carlo with 40,000 deals
//!   seeded by the hand id (standard error about 0.25 percentage points).
//!   Running it twice does not change equity, so it is computed on one board;
//!   a double board splits each pot in half by board.
//! * `derived.river_strength`: for each hand known at a river showdown, the share
//!   of all opponent holdings (from the cards not on the board or in that hand)
//!   it beats, ties counting half. Hold'em-style games enumerate every holding;
//!   Omaha games sample 3,000 holdings with a seeded generator.

use std::collections::BTreeMap;

use pe_cards::{Card, Game, Rng, cards_of, mask_of};

use crate::model::{AllIn, Hand, StreetName};

fn seed_of(id: &str) -> u64 {
    id.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

fn equity_of(game: &Game, hands: &[Vec<Card>], boards: &[Vec<Card>], seed: u64) -> Option<Vec<f64>> {
    let used: u64 = hands.iter().map(|h| mask_of(h)).fold(0, |a, b| a | b) | boards.iter().map(|b| mask_of(b)).fold(0, |a, b| a | b);
    let left = (game.deck() & !used).count_ones() as u64;
    let need: u64 = boards.iter().map(|b| 5 - b.len() as u64).sum();
    let runouts = pe_cards::choose(left, need);
    let r = if runouts <= 50_000 {
        pe_equity::equity_exact(game, hands, boards, &[]).ok()?
    } else {
        pe_equity::equity_monte_carlo(game, hands, boards, &[], 40_000, seed).ok()?
    };
    Some(r.equity)
}

/// Fills `derived.allin` (requires `reconstruct` first).
pub fn annotate_allin(hand: &mut Hand) {
    let Some(game) = hand.game_def() else { return };
    let Some(derived) = hand.derived.as_ref() else { return };
    let folded = crate::reconstruct::folded_seats(hand);
    let still_in: Vec<u32> = hand.dealt_seats().into_iter().filter(|s| !folded.contains(s)).collect();
    if still_in.len() < 2 {
        return;
    }
    // the betting closed on the last street with actions; earlier than the river means a run-out
    let last = hand.streets.last().map_or(if hand.bomb_pot { StreetName::Flop } else { StreetName::Preflop }, |s| s.name);
    if last == StreetName::River {
        return;
    }
    let behind = still_in
        .iter()
        .filter(|s| {
            let stack = hand.player(**s).map_or(0, |p| p.stack);
            stack - derived.contrib.get(s).copied().unwrap_or(0) > 0
        })
        .count();
    if behind > 1 {
        return; // not an all-in: the hand ended otherwise
    }
    let mut cards: BTreeMap<u32, Vec<Card>> = BTreeMap::new();
    for &s in &still_in {
        match hand.hole_cards(s) {
            Some(c) if c.len() == game.hole_cards => {
                cards.insert(s, c);
            }
            _ => return,
        }
    }
    let known = last.board_len();
    let double = hand.board_kind == "double";
    let boards: Vec<Vec<Card>> = if double {
        (0..2).map(|i| hand.board_cards(i).into_iter().take(known).collect()).collect()
    } else {
        vec![hand.board_cards(0).into_iter().take(known).collect()]
    };
    let seed = seed_of(&hand.id);
    let total: i64 = derived.pots.iter().map(|p| p.amount).sum();
    let factor = if total > 0 { (total - hand.rake).max(0) as f64 / total as f64 } else { 1.0 };
    let mut expected: BTreeMap<u32, f64> = still_in.iter().map(|s| (*s, 0.0)).collect();
    let mut main_equity: BTreeMap<u32, f64> = BTreeMap::new();
    for (i, pot) in derived.pots.iter().enumerate() {
        let elig: Vec<u32> = pot.eligible.iter().copied().filter(|s| cards.contains_key(s)).collect();
        if elig.is_empty() {
            continue;
        }
        let shares = if elig.len() == 1 {
            vec![1.0]
        } else {
            let hands: Vec<Vec<Card>> = elig.iter().map(|s| cards[s].clone()).collect();
            match equity_of(&game, &hands, &boards, seed ^ i as u64) {
                Some(e) => e,
                None => return,
            }
        };
        for (s, e) in elig.iter().zip(&shares) {
            *expected.get_mut(s).unwrap() += e * pot.amount as f64 * factor;
            if i == 0 {
                main_equity.insert(*s, *e);
            }
        }
    }
    let ev_net: BTreeMap<u32, i64> = expected.iter().map(|(s, e)| (*s, e.round() as i64 - derived.contrib.get(s).copied().unwrap_or(0))).collect();
    let d = hand.derived.as_mut().unwrap();
    d.allin = Some(AllIn { street: last, seats: still_in, equity: main_equity, ev_net });
}

/// Share of opponent holdings a hand beats on a complete board (ties half).
pub fn river_strength(game: &Game, hole: &[Card], board: &[Card], seed: u64) -> f64 {
    let me = pe_eval::eval_hand(game, hole, board);
    let dead = mask_of(hole) | mask_of(board);
    let rest: Vec<Card> = cards_of(game.deck() & !dead);
    let mut score = 0.0;
    let mut n = 0.0;
    let mut tally = |opp: &[Card]| {
        let v = pe_eval::eval_hand(game, opp, board);
        score += if me > v {
            1.0
        } else if me == v {
            0.5
        } else {
            0.0
        };
        n += 1.0;
    };
    if game.hole_cards == 2 {
        pe_cards::combos::for_each_subset(&rest, 2, &mut |opp| tally(opp));
    } else {
        let mut rng = Rng::new(seed);
        let mut deck = rest.clone();
        for _ in 0..3000 {
            for i in 0..game.hole_cards {
                let j = i + rng.below((deck.len() - i) as u64) as usize;
                deck.swap(i, j);
            }
            let opp: Vec<Card> = deck[..game.hole_cards].to_vec();
            tally(&opp);
        }
    }
    if n > 0.0 { score / n } else { 0.0 }
}

/// Fills `derived.river_strength` for every hand known at a river showdown.
pub fn annotate_river_strength(hand: &mut Hand) {
    let Some(game) = hand.game_def() else { return };
    let board = hand.board_cards(0);
    if board.len() != 5 || hand.board_kind == "run_twice" {
        return;
    }
    let Some(derived) = hand.derived.as_ref() else { return };
    let seats = derived.showdown_seats.clone();
    let seed = seed_of(&hand.id);
    let mut out = BTreeMap::new();
    for s in seats {
        if let Some(c) = hand.hole_cards(s) {
            if c.len() == game.hole_cards {
                out.insert(s, river_strength(&game, &c, &board, seed ^ s as u64));
            }
        }
    }
    hand.derived.as_mut().unwrap().river_strength = out;
}

#[cfg(test)]
mod tests {
    use super::*;
    use pe_cards::parse_cards;

    #[test]
    fn strength_extremes() {
        let g = Game::NLH;
        let board = parse_cards("AsKsQsJs2d").unwrap();
        assert!((river_strength(&g, &parse_cards("Ts3c").unwrap(), &board, 1) - 1.0).abs() < 1e-12);
        let weak = river_strength(&g, &parse_cards("3c4d").unwrap(), &board, 1);
        assert!(weak < 0.2, "{weak}");
    }
}
