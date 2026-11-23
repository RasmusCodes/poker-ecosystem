//! Exact hand evaluation.
//!
//! * [`eval_high`]: best five-card high hand from 5-7 cards, for standard and
//!   short-deck rules, straight from the four 13-bit suit masks (no lookup of
//!   card subsets, no approximation).
//! * [`eval_omaha_high`]: Omaha's exactly-two-plus-three rule.
//! * [`eval_omaha_low8`]: ace-to-five eight-or-better low for Omaha Hi-Lo and Big O.
//! * [`dense_rank`]: the familiar 1..=7462 class index of a standard high hand.
//!
//! A [`HandValue`] compares correctly with `>` within one rule set. Its layout is
//! `order << 26 | category << 22 | payload`, where `order` is the category's rank
//! under the rules (short deck moves flushes above full houses and, under Triton
//! rules, three of a kind above straights).
//!
//! Correctness is checked exhaustively in the tests: every 5-card and 7-card
//! standard hand (the published category counts), every 5-card short-deck hand,
//! and agreement with an independent naive evaluator.

use std::sync::OnceLock;

use pe_cards::{Card, CardMask, Game, HighRules, HoleRule};

pub type HandValue = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Category {
    HighCard = 0,
    Pair = 1,
    TwoPair = 2,
    Trips = 3,
    Straight = 4,
    Flush = 5,
    FullHouse = 6,
    Quads = 7,
    StraightFlush = 8,
}

impl Category {
    pub const ALL: [Category; 9] = [
        Category::HighCard,
        Category::Pair,
        Category::TwoPair,
        Category::Trips,
        Category::Straight,
        Category::Flush,
        Category::FullHouse,
        Category::Quads,
        Category::StraightFlush,
    ];
    pub fn from_u32(v: u32) -> Category {
        Category::ALL[v as usize]
    }
    pub fn name(self) -> &'static str {
        match self {
            Category::HighCard => "high card",
            Category::Pair => "pair",
            Category::TwoPair => "two pair",
            Category::Trips => "three of a kind",
            Category::Straight => "straight",
            Category::Flush => "flush",
            Category::FullHouse => "full house",
            Category::Quads => "four of a kind",
            Category::StraightFlush => "straight flush",
        }
    }
}

const CAT_SHIFT: u32 = 22;
const ORDER_SHIFT: u32 = 26;

/// Rank of a category under the rules (higher beats lower).
#[inline]
pub fn category_order(cat: Category, rules: HighRules) -> u32 {
    use Category::*;
    match rules {
        HighRules::Standard => cat as u32,
        HighRules::ShortDeckTriton => match cat {
            HighCard => 0,
            Pair => 1,
            TwoPair => 2,
            Straight => 3,
            Trips => 4,
            FullHouse => 5,
            Flush => 6,
            Quads => 7,
            StraightFlush => 8,
        },
        HighRules::ShortDeckClassic => match cat {
            HighCard => 0,
            Pair => 1,
            TwoPair => 2,
            Trips => 3,
            Straight => 4,
            FullHouse => 5,
            Flush => 6,
            Quads => 7,
            StraightFlush => 8,
        },
    }
}

#[inline]
fn make(cat: Category, payload: u32, rules: HighRules) -> HandValue {
    (category_order(cat, rules) << ORDER_SHIFT) | ((cat as u32) << CAT_SHIFT) | payload
}

#[inline]
pub fn category(v: HandValue) -> Category {
    Category::from_u32((v >> CAT_SHIFT) & 0xF)
}

#[inline]
fn top_bit(m: u32) -> u32 {
    debug_assert!(m != 0);
    31 - m.leading_zeros()
}

/// Keeps the `k` highest set bits of `m`.
#[inline]
fn keep_top(mut m: u32, k: u32) -> u32 {
    while m.count_ones() > k {
        m &= m - 1;
    }
    m
}

/// Top rank of the best straight in a 13-bit rank mask, if any.
#[inline]
fn straight_top(m: u32, short: bool) -> Option<u32> {
    let t = m & (m >> 1) & (m >> 2) & (m >> 3) & (m >> 4);
    if t != 0 {
        return Some(top_bit(t) + 4);
    }
    if short {
        // A-6-7-8-9: the ace plays below the six
        if m & 0x10F0 == 0x10F0 {
            return Some(7);
        }
    } else if m & 0x100F == 0x100F {
        return Some(3);
    }
    None
}

/// Best high hand among 5 to 7 cards.
#[inline]
pub fn eval_high(m: CardMask, rules: HighRules) -> HandValue {
    debug_assert!((5..=7).contains(&m.count_ones()), "eval_high needs 5-7 cards");
    let short = rules.is_short_deck();
    let s = [(m & 0x1FFF) as u32, ((m >> 16) & 0x1FFF) as u32, ((m >> 32) & 0x1FFF) as u32, ((m >> 48) & 0x1FFF) as u32];
    // With at most seven cards a flush excludes quads and full houses (they would
    // need eight cards), so a flush suit settles the hand.
    for &sm in &s {
        if sm.count_ones() >= 5 {
            if let Some(t) = straight_top(sm, short) {
                return make(Category::StraightFlush, t, rules);
            }
            return make(Category::Flush, keep_top(sm, 5), rules);
        }
    }
    let any = s[0] | s[1] | s[2] | s[3];
    let two = (s[0] & s[1]) | (s[2] & s[3]) | ((s[0] | s[1]) & (s[2] | s[3]));
    let three = (s[0] & s[1] & (s[2] | s[3])) | (s[2] & s[3] & (s[0] | s[1]));
    let four = s[0] & s[1] & s[2] & s[3];
    if four != 0 {
        let q = top_bit(four);
        return make(Category::Quads, (q << 4) | top_bit(any & !(1 << q)), rules);
    }
    let trips = if three != 0 {
        let t = top_bit(three);
        let rest = two & !(1 << t);
        if rest != 0 {
            return make(Category::FullHouse, (t << 4) | top_bit(rest), rules);
        }
        Some(make(Category::Trips, (t << 13) | keep_top(any & !(1 << t), 2), rules))
    } else {
        None
    };
    if let Some(st) = straight_top(any, short) {
        let sv = make(Category::Straight, st, rules);
        return match trips {
            Some(tv) if tv > sv => tv,
            _ => sv,
        };
    }
    if let Some(tv) = trips {
        return tv;
    }
    if two != 0 {
        let p1 = top_bit(two);
        let rest = two & !(1 << p1);
        if rest != 0 {
            let p2 = top_bit(rest);
            let kick = top_bit(any & !(1 << p1) & !(1 << p2));
            return make(Category::TwoPair, (p1 << 8) | (p2 << 4) | kick, rules);
        }
        return make(Category::Pair, (p1 << 13) | keep_top(any & !(1 << p1), 3), rules);
    }
    make(Category::HighCard, keep_top(any, 5), rules)
}

/// Standard-rules shortcut.
#[inline]
pub fn eval7(m: CardMask) -> HandValue {
    eval_high(m, HighRules::Standard)
}

/// Masks of every 2-card subset of the hole cards.
pub fn hole_pairs(hole: &[Card]) -> Vec<CardMask> {
    let mut v = Vec::with_capacity(hole.len() * (hole.len() - 1) / 2);
    for i in 0..hole.len() {
        for j in i + 1..hole.len() {
            v.push(hole[i].mask() | hole[j].mask());
        }
    }
    v
}

/// Masks of every 3-card subset of the board.
pub fn board_triples(board: &[Card]) -> Vec<CardMask> {
    let mut v = Vec::with_capacity(10);
    for i in 0..board.len() {
        for j in i + 1..board.len() {
            for k in j + 1..board.len() {
                v.push(board[i].mask() | board[j].mask() | board[k].mask());
            }
        }
    }
    v
}

/// Omaha high value from precomputed hole pairs and board triples.
#[inline]
pub fn eval_omaha_pre(pairs: &[CardMask], triples: &[CardMask], rules: HighRules) -> HandValue {
    let mut best = 0;
    for &t in triples {
        for &p in pairs {
            let v = eval_high(p | t, rules);
            if v > best {
                best = v;
            }
        }
    }
    best
}

/// Omaha high: exactly two hole cards and three board cards.
pub fn eval_omaha_high(hole: &[Card], board: &[Card], rules: HighRules) -> HandValue {
    eval_omaha_pre(&hole_pairs(hole), &board_triples(board), rules)
}

/// Low-rank bit of a card for ace-to-five lows (ace = bit 0 ... eight = bit 7), if it is eight or lower.
#[inline]
pub fn low_bit(c: Card) -> Option<u32> {
    let r = c.rank();
    if r == 12 {
        Some(0)
    } else if r <= 6 {
        Some(r as u32 + 1)
    } else {
        None
    }
}

/// Low value: 0 = no qualifying low, otherwise `256 - mask` of the five low ranks
/// (larger is better; the wheel A-2-3-4-5 scores 225).
pub type LowValue = u32;

fn low_ranks(cards: &[Card]) -> u32 {
    cards.iter().filter_map(|&c| low_bit(c)).fold(0u32, |m, b| m | (1 << b))
}

/// The lowest `k` set bits of `m`, or `None` if there are fewer.
#[inline]
fn keep_low(m: u32, k: u32) -> Option<u32> {
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

/// Omaha ace-to-five eight-or-better low: two hole cards with different low ranks
/// plus the three lowest board low ranks that differ from them.
pub fn eval_omaha_low8(hole: &[Card], board: &[Card]) -> LowValue {
    let board_low = low_ranks(board);
    if board_low.count_ones() < 3 {
        return 0;
    }
    let hole_low: Vec<u32> = {
        let m = low_ranks(hole);
        (0..8).filter(|b| m & (1 << b) != 0).collect()
    };
    let mut best: Option<u32> = None;
    for i in 0..hole_low.len() {
        for j in i + 1..hole_low.len() {
            let pair = (1 << hole_low[i]) | (1 << hole_low[j]);
            if let Some(t) = keep_low(board_low & !pair, 3) {
                let m = pair | t;
                if best.is_none_or(|b| m < b) {
                    best = Some(m);
                }
            }
        }
    }
    best.map_or(0, |m| 256 - m)
}

/// High value of a hand for a game (hold'em: any five; Omaha: two plus three).
pub fn eval_hand(game: &Game, hole: &[Card], board: &[Card]) -> HandValue {
    match game.hole_rule {
        HoleRule::AnyFive => {
            let m = pe_cards::mask_of(hole) | pe_cards::mask_of(board);
            if m.count_ones() <= 7 {
                eval_high(m, game.high)
            } else {
                best_of_subsets(&[hole, board].concat(), game.high)
            }
        }
        HoleRule::TwoPlusThree => eval_omaha_high(hole, board, game.high),
    }
}

/// Low value of a hand for a hi-lo game (0 if the game has no low or none qualifies).
pub fn eval_low(game: &Game, hole: &[Card], board: &[Card]) -> LowValue {
    if !game.hilo {
        return 0;
    }
    match game.hole_rule {
        HoleRule::TwoPlusThree => eval_omaha_low8(hole, board),
        HoleRule::AnyFive => {
            let m = low_ranks(hole) | low_ranks(board);
            keep_low(m, 5).map_or(0, |l| 256 - l)
        }
    }
}

fn best_of_subsets(cards: &[Card], rules: HighRules) -> HandValue {
    let mut best = 0;
    pe_cards::combos::for_each_subset(cards, 5, &mut |s| {
        let v = eval_high(pe_cards::mask_of(s), rules);
        if v > best {
            best = v;
        }
    });
    best
}

/// All distinct standard 5-card values in increasing order (7,462 of them).
fn standard_classes() -> &'static Vec<HandValue> {
    static CLASSES: OnceLock<Vec<HandValue>> = OnceLock::new();
    CLASSES.get_or_init(|| {
        let mut set = std::collections::BTreeSet::new();
        let cards: Vec<Card> = (0..52).map(Card).collect();
        pe_cards::combos::for_each_subset(&cards, 5, &mut |s| {
            set.insert(eval7(pe_cards::mask_of(s)));
        });
        set.into_iter().collect()
    })
}

/// The standard 1..=7462 equivalence-class rank of a standard-rules value
/// (7462 = royal flush, 1 = 7-5-4-3-2 offsuit).
pub fn dense_rank(v: HandValue) -> u16 {
    let classes = standard_classes();
    match classes.binary_search(&v) {
        Ok(i) => (i + 1) as u16,
        Err(_) => 0,
    }
}

fn rank_plural(r: u32) -> &'static str {
    pe_cards::card::RANK_PLURALS[r as usize]
}
fn rank_name(r: u32) -> &'static str {
    pe_cards::card::RANK_NAMES[r as usize]
}

/// PokerStars-style description, e.g. "a pair of Queens", "a straight, Nine to King".
pub fn describe(v: HandValue, rules: HighRules) -> String {
    let p = v & ((1 << CAT_SHIFT) - 1);
    let straight_low = |top: u32| -> u32 {
        if rules.is_short_deck() && top == 7 {
            12
        } else if !rules.is_short_deck() && top == 3 {
            12
        } else {
            top - 4
        }
    };
    match category(v) {
        Category::HighCard => format!("high card {}", rank_name(top_bit(p))),
        Category::Pair => format!("a pair of {}", rank_plural(p >> 13)),
        Category::TwoPair => format!("two pair, {} and {}", rank_plural(p >> 8), rank_plural((p >> 4) & 15)),
        Category::Trips => format!("three of a kind, {}", rank_plural(p >> 13)),
        Category::Straight => format!("a straight, {} to {}", rank_name(straight_low(p)), rank_name(p)),
        Category::Flush => format!("a flush, {} high", rank_name(top_bit(p))),
        Category::FullHouse => format!("a full house, {} full of {}", rank_plural(p >> 4), rank_plural(p & 15)),
        Category::Quads => format!("four of a kind, {}", rank_plural(p >> 4)),
        Category::StraightFlush => {
            if p == 12 {
                "a Royal Flush".to_string()
            } else {
                format!("a straight flush, {} to {}", rank_name(straight_low(p)), rank_name(p))
            }
        }
    }
}

/// Describes a low value, e.g. "8,6,4,3,A low".
pub fn describe_low(v: LowValue) -> String {
    if v == 0 {
        return "no low".into();
    }
    let m = 256 - v;
    let names = ["A", "2", "3", "4", "5", "6", "7", "8"];
    let parts: Vec<&str> = (0..8).rev().filter(|b| m & (1 << b) != 0).map(|b| names[b as usize]).collect();
    format!("{} low", parts.join(","))
}

#[cfg(test)]
mod tests;
