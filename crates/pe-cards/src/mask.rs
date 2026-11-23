//! Card sets as `u64` bit masks in suit-major layout: bit `suit * 16 + rank`.

use crate::card::Card;

/// A set of cards. Bits 13..=15 of each 16-bit suit lane are always zero.
pub type CardMask = u64;

pub const FULL_DECK: CardMask = 0x1FFF_1FFF_1FFF_1FFF;
/// The 36-card short deck (sixes through aces).
pub const SHORT_DECK: CardMask = 0x1FF0_1FF0_1FF0_1FF0;

#[inline]
pub fn mask_of(cards: &[Card]) -> CardMask {
    cards.iter().fold(0, |m, c| m | c.mask())
}

#[inline]
pub fn count(m: CardMask) -> u32 {
    m.count_ones()
}

/// Cards in a mask, in increasing bit order (clubs first).
pub fn cards_of(mut m: CardMask) -> Vec<Card> {
    let mut v = Vec::with_capacity(m.count_ones() as usize);
    while m != 0 {
        let b = m.trailing_zeros();
        v.push(Card::from_bit(b));
        m &= m - 1;
    }
    v
}

/// The four 13-bit suit masks (clubs, diamonds, hearts, spades).
#[inline]
pub fn suit_masks(m: CardMask) -> [u32; 4] {
    [
        (m & 0x1FFF) as u32,
        ((m >> 16) & 0x1FFF) as u32,
        ((m >> 32) & 0x1FFF) as u32,
        ((m >> 48) & 0x1FFF) as u32,
    ]
}

/// Iterator over the cards of a mask.
pub struct MaskIter(pub CardMask);

impl Iterator for MaskIter {
    type Item = Card;
    #[inline]
    fn next(&mut self) -> Option<Card> {
        if self.0 == 0 {
            return None;
        }
        let b = self.0.trailing_zeros();
        self.0 &= self.0 - 1;
        Some(Card::from_bit(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deck_masks() {
        assert_eq!(count(FULL_DECK), 52);
        assert_eq!(count(SHORT_DECK), 36);
        let all: Vec<Card> = MaskIter(FULL_DECK).collect();
        assert_eq!(all.len(), 52);
        assert_eq!(mask_of(&all), FULL_DECK);
        for c in MaskIter(SHORT_DECK) {
            assert!(c.rank() >= 4);
        }
    }
}
