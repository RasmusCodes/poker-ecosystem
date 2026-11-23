//! Hold'em ranges: a weight in `[0, 1]` for each of the 1,326 hole-card combos.
//!
//! Notation (comma or space separated, each item optionally `:weight`, where the
//! weight is a fraction like `0.5` or a percentage like `50%`):
//!
//! | item | meaning |
//! |---|---|
//! | `AA`, `AKs`, `AKo`, `AK` | a class (`AK` = suited and offsuit) |
//! | `TT+` | pairs from TT up to AA |
//! | `A2s+`, `KTo+`, `K9+` | the kicker climbs to one below the top card |
//! | `22-55`, `A2s-A5s`, `K9o-KQo` | inclusive spans with a fixed top card |
//! | `AhKd` | one specific combo |
//! | `random`, `any` | every combo |
//! | `15%` | the strongest 15% of combos by a supplied preflop ranking |
//!
//! Later items override earlier ones, so `AK:0.5, AKs` gives AKs full weight.

use crate::card::{Card, parse_rank};
use crate::combos::{NUM_CLASSES, NUM_COMBOS, class_combos, class_name, class_of, class_size, combo_cards, combo_index};
use crate::mask::CardMask;

#[derive(Clone, Debug, PartialEq)]
pub struct HoldemRange {
    pub weights: Vec<f64>,
}

impl Default for HoldemRange {
    fn default() -> Self {
        HoldemRange::empty()
    }
}

/// Grid index of the class `(hi, lo, suited)` (ranks 0..=12, `hi >= lo`).
fn class_index(hi: u8, lo: u8, suited: bool) -> usize {
    let r_hi = 12 - hi as usize;
    let r_lo = 12 - lo as usize;
    if hi == lo {
        r_hi * 13 + r_hi
    } else if suited {
        r_hi * 13 + r_lo
    } else {
        r_lo * 13 + r_hi
    }
}

impl HoldemRange {
    pub fn empty() -> Self {
        HoldemRange { weights: vec![0.0; NUM_COMBOS] }
    }

    pub fn full() -> Self {
        HoldemRange { weights: vec![1.0; NUM_COMBOS] }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        Self::parse_ranked(s, None)
    }

    /// Parses range notation. `ranking` (169 class indices, strongest first) is
    /// needed only for `x%` items.
    pub fn parse_ranked(s: &str, ranking: Option<&[usize]>) -> Result<Self, String> {
        let mut r = HoldemRange::empty();
        for raw in s.split(|c: char| c == ',' || c.is_whitespace()) {
            let tok = raw.trim();
            if tok.is_empty() {
                continue;
            }
            let (item, weight) = match tok.split_once(':') {
                Some((i, w)) => (i.trim(), parse_weight(w)?),
                None => (tok, 1.0),
            };
            let combos = item_combos(item, ranking)?;
            for c in combos {
                r.weights[c] = weight;
            }
        }
        Ok(r)
    }

    /// Strongest `pct` percent of all combos (by combo count) under `ranking`. The
    /// boundary class gets a fractional weight so the total is exact.
    pub fn top_percent(pct: f64, ranking: &[usize]) -> Self {
        let mut r = HoldemRange::empty();
        let mut remaining = (pct.clamp(0.0, 100.0) / 100.0) * NUM_COMBOS as f64;
        for &class in ranking {
            if remaining <= 1e-9 {
                break;
            }
            let size = class_size(class) as f64;
            let w = (remaining / size).min(1.0);
            for c in class_combos(class) {
                r.weights[c] = w;
            }
            remaining -= w * size;
        }
        r
    }

    pub fn set_combo(&mut self, a: Card, b: Card, w: f64) {
        self.weights[combo_index(a, b)] = w;
    }

    /// Zeroes every combo that uses a card in `dead`.
    pub fn remove_dead(&mut self, dead: CardMask) {
        for i in 0..NUM_COMBOS {
            let (a, b) = combo_cards(i);
            if (a.mask() | b.mask()) & dead != 0 {
                self.weights[i] = 0.0;
            }
        }
    }

    /// Combos with positive weight, as `(combo index, weight)`.
    pub fn combos(&self) -> Vec<(usize, f64)> {
        self.weights.iter().enumerate().filter(|(_, w)| **w > 0.0).map(|(i, w)| (i, *w)).collect()
    }

    /// Weighted number of combos.
    pub fn size(&self) -> f64 {
        self.weights.iter().sum()
    }

    /// Mean weight of each class (169 grid cells).
    pub fn class_weights(&self) -> Vec<f64> {
        let mut sums = vec![0.0; NUM_CLASSES];
        for i in 0..NUM_COMBOS {
            let (a, b) = combo_cards(i);
            sums[class_of(a, b)] += self.weights[i];
        }
        (0..NUM_CLASSES).map(|c| sums[c] / class_size(c) as f64).collect()
    }

    /// Compact class-level notation (classes with uniform weight), e.g. `"AA,AKs:0.5"`.
    /// Classes with mixed per-combo weights are written combo by combo.
    pub fn to_notation(&self) -> String {
        let mut parts = Vec::new();
        for class in 0..NUM_CLASSES {
            let combos = class_combos(class);
            let ws: Vec<f64> = combos.iter().map(|&c| self.weights[c]).collect();
            if ws.iter().all(|w| *w == 0.0) {
                continue;
            }
            if ws.iter().all(|w| (*w - ws[0]).abs() < 1e-12) {
                parts.push(fmt_item(&class_name(class), ws[0]));
            } else {
                for (&c, &w) in combos.iter().zip(&ws) {
                    if w > 0.0 {
                        let (a, b) = combo_cards(c);
                        parts.push(fmt_item(&format!("{a}{b}"), w));
                    }
                }
            }
        }
        parts.join(",")
    }
}

fn fmt_item(name: &str, w: f64) -> String {
    if (w - 1.0).abs() < 1e-12 { name.to_string() } else { format!("{name}:{}", trim_float(w)) }
}

fn trim_float(w: f64) -> String {
    let s = format!("{w:.4}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

fn parse_weight(w: &str) -> Result<f64, String> {
    let w = w.trim();
    let v = if let Some(p) = w.strip_suffix('%') {
        p.trim().parse::<f64>().map_err(|_| format!("bad weight '{w}'"))? / 100.0
    } else {
        w.parse::<f64>().map_err(|_| format!("bad weight '{w}'"))?
    };
    if !(0.0..=1.0).contains(&v) {
        return Err(format!("weight '{w}' is outside 0..1"));
    }
    Ok(v)
}

/// A class spec: (hi rank, lo rank, suitedness) where suitedness is None for "both".
type ClassSpec = (u8, u8, Option<bool>);

fn parse_class(s: &str) -> Result<ClassSpec, String> {
    let ch: Vec<char> = s.chars().collect();
    if ch.len() < 2 || ch.len() > 3 {
        return Err(format!("bad hand class '{s}'"));
    }
    let a = parse_rank(ch[0]).ok_or_else(|| format!("bad rank in '{s}'"))?;
    let b = parse_rank(ch[1]).ok_or_else(|| format!("bad rank in '{s}'"))?;
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    let suited = match ch.get(2).map(|c| c.to_ascii_lowercase()) {
        None => None,
        Some('s') => Some(true),
        Some('o') => Some(false),
        Some(c) => return Err(format!("bad suffix '{c}' in '{s}'")),
    };
    if hi == lo && suited == Some(true) {
        return Err(format!("a pair cannot be suited: '{s}'"));
    }
    Ok((hi, lo, suited))
}

fn spec_combos(spec: ClassSpec) -> Vec<usize> {
    let (hi, lo, suited) = spec;
    let mut v = Vec::new();
    if hi == lo {
        v.extend(class_combos(class_index(hi, lo, false)));
    } else {
        if suited != Some(false) {
            v.extend(class_combos(class_index(hi, lo, true)));
        }
        if suited != Some(true) {
            v.extend(class_combos(class_index(hi, lo, false)));
        }
    }
    v
}

fn item_combos(item: &str, ranking: Option<&[usize]>) -> Result<Vec<usize>, String> {
    let lower = item.to_ascii_lowercase();
    if lower == "random" || lower == "any" || lower == "all" {
        return Ok((0..NUM_COMBOS).collect());
    }
    if let Some(p) = item.strip_suffix('%') {
        let pct: f64 = p.trim().parse().map_err(|_| format!("bad percentage '{item}'"))?;
        let ranking = ranking.ok_or_else(|| format!("'{item}' needs a preflop ranking"))?;
        let r = HoldemRange::top_percent(pct, ranking);
        // percentage items select whole combos at weight 1 except the boundary class
        return Ok(r.combos().into_iter().filter(|(_, w)| *w >= 0.5).map(|(c, _)| c).collect());
    }
    // a specific combo: rank suit rank suit
    let ch: Vec<char> = item.chars().collect();
    if ch.len() == 4 && crate::card::parse_suit(ch[1]).is_some() && crate::card::parse_suit(ch[3]).is_some() {
        let a = Card::parse(&item[0..2])?;
        let b = Card::parse(&item[2..4])?;
        if a == b {
            return Err(format!("duplicate card in '{item}'"));
        }
        return Ok(vec![combo_index(a, b)]);
    }
    if let Some((from, to)) = item.split_once('-') {
        let f = parse_class(from.trim())?;
        let t = parse_class(to.trim())?;
        let mut out = Vec::new();
        if f.0 == f.1 && t.0 == t.1 {
            let (lo, hi) = if f.0 <= t.0 { (f.0, t.0) } else { (t.0, f.0) };
            for r in lo..=hi {
                out.extend(spec_combos((r, r, None)));
            }
            return Ok(out);
        }
        if f.0 != t.0 || f.2 != t.2 || f.0 == f.1 || t.0 == t.1 {
            return Err(format!("span '{item}' must keep the top card and suitedness"));
        }
        let (lo, hi) = if f.1 <= t.1 { (f.1, t.1) } else { (t.1, f.1) };
        for k in lo..=hi {
            out.extend(spec_combos((f.0, k, f.2)));
        }
        return Ok(out);
    }
    if let Some(base) = item.strip_suffix('+') {
        let (hi, lo, suited) = parse_class(base)?;
        let mut out = Vec::new();
        if hi == lo {
            for r in hi..=12 {
                out.extend(spec_combos((r, r, None)));
            }
        } else {
            for k in lo..hi {
                out.extend(spec_combos((hi, k, suited)));
            }
        }
        return Ok(out);
    }
    Ok(spec_combos(parse_class(item)?))
}

/// Class name of a combo index (convenience for reports).
pub fn combo_class_name(idx: usize) -> String {
    let (a, b) = combo_cards(idx);
    class_name(class_of(a, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic_items() {
        let r = HoldemRange::parse("AA").unwrap();
        assert_eq!(r.size(), 6.0);
        let r = HoldemRange::parse("AK").unwrap();
        assert_eq!(r.size(), 16.0);
        let r = HoldemRange::parse("AKs,AKo:0.5").unwrap();
        assert_eq!(r.size(), 4.0 + 6.0);
        let r = HoldemRange::parse("TT+").unwrap();
        assert_eq!(r.size(), 30.0);
        let r = HoldemRange::parse("A2s+").unwrap();
        assert_eq!(r.size(), 12.0 * 4.0);
        let r = HoldemRange::parse("22-55").unwrap();
        assert_eq!(r.size(), 24.0);
        let r = HoldemRange::parse("A5s-A2s").unwrap();
        assert_eq!(r.size(), 16.0);
        let r = HoldemRange::parse("AhKd").unwrap();
        assert_eq!(r.size(), 1.0);
        let r = HoldemRange::parse("random").unwrap();
        assert_eq!(r.size(), 1326.0);
        let r = HoldemRange::parse("K9+ 50%").err();
        assert!(r.is_some());
    }

    #[test]
    fn notation_round_trip() {
        let src = "AA,KK:0.5,AKs,T9s:0.25";
        let r = HoldemRange::parse(src).unwrap();
        let back = HoldemRange::parse(&r.to_notation()).unwrap();
        assert_eq!(r, back);
    }

    #[test]
    fn dead_cards_and_top_percent() {
        let mut r = HoldemRange::parse("AA").unwrap();
        r.remove_dead(Card::parse("As").unwrap().mask());
        assert_eq!(r.size(), 3.0);
        let ranking: Vec<usize> = (0..NUM_CLASSES).collect();
        let t = HoldemRange::top_percent(10.0, &ranking);
        assert!((t.size() - 132.6).abs() < 1e-9);
    }
}
