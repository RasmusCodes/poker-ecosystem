//! Settling one complete deal: who gets which share of the pot.

use pe_cards::{Card, CardMask, Game, HoleRule};
use pe_eval::{HandValue, LowValue, eval_high, eval_omaha_pre, hole_pairs};

/// Most players at one table.
pub const MAX_PLAYERS: usize = 10;

/// Per-player data precomputed once per hand.
#[derive(Clone, Debug)]
pub struct PreparedHand {
    pub cards: Vec<Card>,
    pub mask: CardMask,
    pub pairs: Vec<CardMask>,
    /// Distinct low ranks in the hand (ace = bit 0 ... eight = bit 7).
    pub low_ranks: u32,
}

impl PreparedHand {
    pub fn new(cards: &[Card]) -> PreparedHand {
        let mask = pe_cards::mask_of(cards);
        PreparedHand { cards: cards.to_vec(), mask, pairs: hole_pairs(cards), low_ranks: low_ranks_of_mask(mask) }
    }
}

/// Distinct ace-to-five low ranks in a card mask.
#[inline]
pub fn low_ranks_of_mask(m: CardMask) -> u32 {
    let rm = ((m | (m >> 16) | (m >> 32) | (m >> 48)) & 0x1FFF) as u32;
    ((rm >> 12) & 1) | ((rm & 0x7F) << 1)
}

#[inline]
fn lowest_bits(m: u32, k: u32) -> Option<u32> {
    if m.count_ones() < k {
        return None;
    }
    let mut out = 0;
    let mut rest = m;
    for _ in 0..k {
        let b = rest & rest.wrapping_neg();
        out |= b;
        rest &= rest - 1;
    }
    Some(out)
}

/// A complete board prepared for evaluation (no heap allocation).
#[derive(Clone, Copy, Debug)]
pub struct PreparedBoard {
    pub mask: CardMask,
    pub triples: [CardMask; 10],
    pub n_triples: u8,
    pub low_ranks: u32,
}

impl PreparedBoard {
    pub fn from_mask(mask: CardMask) -> PreparedBoard {
        let mut bits = [0u64; 5];
        let mut n = 0;
        let mut m = mask;
        while m != 0 && n < 5 {
            bits[n] = m & m.wrapping_neg();
            m &= m - 1;
            n += 1;
        }
        let mut triples = [0u64; 10];
        let mut t = 0;
        for i in 0..n {
            for j in i + 1..n {
                for k in j + 1..n {
                    triples[t] = bits[i] | bits[j] | bits[k];
                    t += 1;
                }
            }
        }
        PreparedBoard { mask, triples, n_triples: t as u8, low_ranks: low_ranks_of_mask(mask) }
    }
    pub fn new(cards: &[Card]) -> PreparedBoard {
        PreparedBoard::from_mask(pe_cards::mask_of(cards))
    }
    #[inline]
    pub fn triples(&self) -> &[CardMask] {
        &self.triples[..self.n_triples as usize]
    }
}

#[inline]
pub fn high_value(game: &Game, hand: &PreparedHand, board: &PreparedBoard) -> HandValue {
    match game.hole_rule {
        HoleRule::AnyFive => eval_high(hand.mask | board.mask, game.high),
        HoleRule::TwoPlusThree => eval_omaha_pre(&hand.pairs, board.triples(), game.high),
    }
}

/// Omaha-style low (two hole + three board) or any-five low; 0 when none qualifies.
#[inline]
pub fn low_value(game: &Game, hand: &PreparedHand, board: &PreparedBoard) -> LowValue {
    if !game.hilo || board.low_ranks.count_ones() < 3 {
        return 0;
    }
    match game.hole_rule {
        HoleRule::AnyFive => lowest_bits(hand.low_ranks | board.low_ranks, 5).map_or(0, |l| 256 - l),
        HoleRule::TwoPlusThree => {
            let h = hand.low_ranks;
            let mut best: u32 = u32::MAX;
            let mut a = h;
            while a != 0 {
                let ba = a & a.wrapping_neg();
                a &= a - 1;
                let mut b = a;
                while b != 0 {
                    let bb = b & b.wrapping_neg();
                    b &= b - 1;
                    let pair = ba | bb;
                    if let Some(t) = lowest_bits(board.low_ranks & !pair, 3) {
                        best = best.min(pair | t);
                    }
                }
            }
            if best == u32::MAX { 0 } else { 256 - best }
        }
    }
}

/// Adds each player's pot share for one complete deal into `out`, scaled by `weight`.
/// Ties split equally; in hi-lo games half the pot goes to the best qualifying low
/// (the high hand scoops when no low qualifies). Multiple boards split the pot evenly.
pub fn add_shares(game: &Game, hands: &[&PreparedHand], boards: &[PreparedBoard], weight: f64, out: &mut [f64], scratch: &mut Vec<u32>) {
    let per_board = weight / boards.len() as f64;
    let n = hands.len();
    for board in boards {
        scratch.clear();
        scratch.extend(hands.iter().map(|h| high_value(game, h, board)));
        let best_hi = *scratch.iter().max().unwrap();
        let n_hi = scratch.iter().filter(|&&v| v == best_hi).count() as f64;
        let mut hi_part = per_board;
        if game.hilo {
            let mut lows = [0u32; MAX_PLAYERS];
            for i in 0..n {
                lows[i] = low_value(game, hands[i], board);
            }
            let best_lo = *lows[..n].iter().max().unwrap();
            if best_lo > 0 {
                hi_part = per_board / 2.0;
                let n_lo = lows[..n].iter().filter(|&&v| v == best_lo).count() as f64;
                for i in 0..n {
                    if lows[i] == best_lo {
                        out[i] += hi_part / n_lo;
                    }
                }
            }
        }
        for (i, &v) in scratch.iter().enumerate() {
            if v == best_hi {
                out[i] += hi_part / n_hi;
            }
        }
    }
}
