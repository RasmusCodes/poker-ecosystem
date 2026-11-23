//! Omaha hands (four or five hole cards) and a small pattern language for ranges.
//!
//! Pattern items (comma or space separated, each optionally `:weight`):
//!
//! | item | meaning |
//! |---|---|
//! | `AhKhQdJd` | one specific hand |
//! | `AAKK`, `KQJT` | these ranks (a multiset that must be present) |
//! | `AAxx`, `AA**` | `x`/`*` match any rank, so `AAxx` includes hands with three aces |
//! | suffix `ds`, `ss`, `r`, `3s`, `m` | suit shape: double-suited (2-2), single-suited (2-1-1), rainbow (1-1-1-1), three of a suit, monotone |
//! | `random` | every hand |
//!
//! Five-card Omaha uses five-symbol patterns; `ds` then means 2-2-1 and `ss` 2-1-1-1.

use crate::card::{Card, parse_rank};
use crate::mask::{CardMask, MaskIter, mask_of};

/// A weighted list of hands of any size (the generic range representation).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HandList {
    pub hands: Vec<Vec<Card>>,
    pub weights: Vec<f64>,
}

impl HandList {
    pub fn push(&mut self, hand: Vec<Card>, w: f64) {
        self.hands.push(hand);
        self.weights.push(w);
    }
    pub fn len(&self) -> usize {
        self.hands.len()
    }
    pub fn is_empty(&self) -> bool {
        self.hands.is_empty()
    }
    pub fn masks(&self) -> Vec<CardMask> {
        self.hands.iter().map(|h| mask_of(h)).collect()
    }
    /// Drops hands that use a card in `dead`.
    pub fn without_dead(&self, dead: CardMask) -> HandList {
        let mut out = HandList::default();
        for (h, &w) in self.hands.iter().zip(&self.weights) {
            if mask_of(h) & dead == 0 && w > 0.0 {
                out.push(h.clone(), w);
            }
        }
        out
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    Any,
    DoubleSuited,
    SingleSuited,
    Rainbow,
    ThreeSuited,
    Monotone,
}

struct Pattern {
    ranks: Vec<u8>, // required ranks (multiset)
    wild: usize,
    shape: Shape,
}

fn parse_pattern(item: &str, n: usize) -> Result<Pattern, String> {
    let lower = item.to_ascii_lowercase();
    let (body, shape) = if let Some(b) = lower.strip_suffix("ds") {
        (b.to_string(), Shape::DoubleSuited)
    } else if let Some(b) = lower.strip_suffix("ss") {
        (b.to_string(), Shape::SingleSuited)
    } else if let Some(b) = lower.strip_suffix("3s") {
        (b.to_string(), Shape::ThreeSuited)
    } else if lower.len() == n + 1 && lower.ends_with('r') {
        (lower[..n].to_string(), Shape::Rainbow)
    } else if lower.len() == n + 1 && lower.ends_with('m') {
        (lower[..n].to_string(), Shape::Monotone)
    } else {
        (lower.clone(), Shape::Any)
    };
    let chars: Vec<char> = body.chars().collect();
    if chars.len() != n {
        return Err(format!("pattern '{item}' must have {n} rank symbols"));
    }
    let mut ranks = Vec::new();
    let mut wild = 0;
    for c in chars {
        if c == 'x' || c == '*' {
            wild += 1;
        } else {
            ranks.push(parse_rank(c).ok_or_else(|| format!("bad rank '{c}' in '{item}'"))?);
        }
    }
    Ok(Pattern { ranks, wild, shape })
}

fn suit_shape(hand: &[Card]) -> Vec<usize> {
    let mut counts = [0usize; 4];
    for c in hand {
        counts[c.suit() as usize] += 1;
    }
    let mut v: Vec<usize> = counts.into_iter().filter(|&c| c > 0).collect();
    v.sort_unstable_by(|a, b| b.cmp(a));
    v
}

fn matches(p: &Pattern, hand: &[Card]) -> bool {
    let mut counts = [0u8; 13];
    for c in hand {
        counts[c.rank() as usize] += 1;
    }
    for &r in &p.ranks {
        if counts[r as usize] == 0 {
            return false;
        }
        counts[r as usize] -= 1;
    }
    debug_assert_eq!(p.ranks.len() + p.wild, hand.len());
    let shape = suit_shape(hand);
    let n = hand.len();
    match p.shape {
        Shape::Any => true,
        Shape::DoubleSuited => {
            if n == 4 {
                shape == [2, 2]
            } else {
                shape == [2, 2, 1]
            }
        }
        Shape::SingleSuited => {
            if n == 4 {
                shape == [2, 1, 1]
            } else {
                shape == [2, 1, 1, 1]
            }
        }
        Shape::Rainbow => shape.iter().all(|&c| c == 1),
        Shape::ThreeSuited => shape[0] == 3,
        Shape::Monotone => shape[0] == n,
    }
}

/// Calls `f` for every `n`-card hand drawn from `deck` (ascending card order).
pub fn for_each_hand(deck: CardMask, n: usize, f: &mut impl FnMut(&[Card])) {
    let cards: Vec<Card> = MaskIter(deck).collect();
    crate::combos::for_each_subset(&cards, n, f);
}

/// Parses an Omaha range for `n`-card hands drawn from `deck` (minus `dead`).
pub fn parse_omaha_range(s: &str, n: usize, deck: CardMask, dead: CardMask) -> Result<HandList, String> {
    let mut chosen: std::collections::HashMap<Vec<Card>, f64> = std::collections::HashMap::new();
    let mut order: Vec<Vec<Card>> = Vec::new();
    let avail = deck & !dead;
    for raw in s.split(|c: char| c == ',' || c.is_whitespace()) {
        let tok = raw.trim();
        if tok.is_empty() {
            continue;
        }
        let (item, w) = match tok.split_once(':') {
            Some((i, w)) => {
                let w = w.trim();
                let v = if let Some(p) = w.strip_suffix('%') {
                    p.parse::<f64>().map_err(|_| format!("bad weight '{w}'"))? / 100.0
                } else {
                    w.parse::<f64>().map_err(|_| format!("bad weight '{w}'"))?
                };
                (i.trim(), v)
            }
            None => (tok, 1.0),
        };
        let mut add = |h: Vec<Card>| {
            if !chosen.contains_key(&h) {
                order.push(h.clone());
            }
            chosen.insert(h, w);
        };
        if item.eq_ignore_ascii_case("random") || item.eq_ignore_ascii_case("any") {
            for_each_hand(avail, n, &mut |h| add(h.to_vec()));
            continue;
        }
        // a specific hand: n cards written out
        if let Ok(cards) = crate::card::parse_cards(item) {
            if cards.len() == n && item.len() == 2 * n {
                if mask_of(&cards) & !avail != 0 {
                    continue; // dead or not in this deck
                }
                let mut h = cards;
                h.sort();
                add(h);
                continue;
            }
        }
        let p = parse_pattern(item, n)?;
        for_each_hand(avail, n, &mut |h| {
            if matches(&p, h) {
                add(h.to_vec());
            }
        });
    }
    let mut out = HandList::default();
    for h in order {
        let w = chosen[&h];
        if w > 0.0 {
            out.push(h, w);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::combos::choose;
    use crate::mask::FULL_DECK;

    #[test]
    fn pattern_counts() {
        let all = parse_omaha_range("random", 4, FULL_DECK, 0).unwrap();
        assert_eq!(all.len() as u64, choose(52, 4));
        // exactly AAKK: 6 * 6 = 36
        let aakk = parse_omaha_range("AAKK", 4, FULL_DECK, 0).unwrap();
        assert_eq!(aakk.len(), 36);
        // AAKK double-suited: the two aces' suits must equal the kings' suits: 6 ways
        let ds = parse_omaha_range("AAKKds", 4, FULL_DECK, 0).unwrap();
        assert_eq!(ds.len(), 6);
        // rainbow share of all hands: 13^4 * 4! / C(52,4)
        let r = parse_omaha_range("xxxxr", 4, FULL_DECK, 0).unwrap();
        assert_eq!(r.len() as u64, 13u64.pow(4) * 24 / 24 * 1); // 4 distinct suits, any ranks: 13^4 hands
        let specific = parse_omaha_range("AhKhQdJd:0.5", 4, FULL_DECK, 0).unwrap();
        assert_eq!(specific.len(), 1);
        assert_eq!(specific.weights[0], 0.5);
    }

    #[test]
    fn dead_cards_filter() {
        let dead = crate::card::Card::parse("As").unwrap().mask();
        let aa = parse_omaha_range("AAxx", 4, FULL_DECK, dead).unwrap();
        // aces left: 3; hands with at least two of them and two other cards from 48 non-aces... plus three-ace hands
        let expected = choose(3, 2) * choose(48, 2) + choose(3, 3) * 48;
        assert_eq!(aa.len() as u64, expected);
    }
}
