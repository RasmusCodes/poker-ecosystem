//! Combinatorics: binomial coefficients, the 1,326 hold'em hole-card combos and
//! the 169 starting-hand classes on the familiar 13 x 13 grid.

use crate::card::{Card, RANK_CHARS};
use crate::mask::CardMask;

pub const NUM_COMBOS: usize = 1326;
pub const NUM_CLASSES: usize = 169;

pub fn choose(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut r: u64 = 1;
    for i in 0..k {
        r = r * (n - i) / (i + 1);
    }
    r
}

/// Index of the hole-card combo `{a, b}` (card ids, any order) in `0..1326`.
#[inline]
pub fn combo_index(a: Card, b: Card) -> usize {
    let (hi, lo) = if a.0 > b.0 { (a.0 as usize, b.0 as usize) } else { (b.0 as usize, a.0 as usize) };
    hi * (hi - 1) / 2 + lo
}

/// The two cards of combo `idx`, higher card id first.
pub fn combo_cards(idx: usize) -> (Card, Card) {
    debug_assert!(idx < NUM_COMBOS);
    // hi is the largest h with h*(h-1)/2 <= idx
    let mut hi = (((8 * idx + 1) as f64).sqrt() as usize + 1) / 2;
    while hi * (hi - 1) / 2 > idx {
        hi -= 1;
    }
    while (hi + 1) * hi / 2 <= idx {
        hi += 1;
    }
    let lo = idx - hi * (hi - 1) / 2;
    (Card(hi as u8), Card(lo as u8))
}

#[inline]
pub fn combo_mask(idx: usize) -> CardMask {
    let (a, b) = combo_cards(idx);
    a.mask() | b.mask()
}

/// Grid index (row * 13 + col) of the class of a two-card hand. Rows and columns run
/// from ace (0) down to deuce (12); pairs are on the diagonal, suited hands above it
/// (row = higher rank) and offsuit hands below it.
pub fn class_of(a: Card, b: Card) -> usize {
    let (hi, lo) = if a.rank() >= b.rank() { (a, b) } else { (b, a) };
    let r_hi = 12 - hi.rank() as usize;
    let r_lo = 12 - lo.rank() as usize;
    if hi.rank() == lo.rank() {
        r_hi * 13 + r_hi
    } else if hi.suit() == lo.suit() {
        r_hi * 13 + r_lo
    } else {
        r_lo * 13 + r_hi
    }
}

/// Name of a class grid index, e.g. `"AKs"`, `"T9o"`, `"77"`.
pub fn class_name(class: usize) -> String {
    let row = class / 13;
    let col = class % 13;
    let rc = |i: usize| RANK_CHARS[12 - i];
    if row == col {
        format!("{}{}", rc(row), rc(col))
    } else if row < col {
        format!("{}{}s", rc(row), rc(col))
    } else {
        format!("{}{}o", rc(col), rc(row))
    }
}

/// Number of combos in a class (6 pairs, 4 suited, 12 offsuit).
pub fn class_size(class: usize) -> usize {
    let row = class / 13;
    let col = class % 13;
    if row == col {
        6
    } else if row < col {
        4
    } else {
        12
    }
}

/// All combos of a class.
pub fn class_combos(class: usize) -> Vec<usize> {
    (0..NUM_COMBOS)
        .filter(|&i| {
            let (a, b) = combo_cards(i);
            class_of(a, b) == class
        })
        .collect()
}

/// Calls `f` with every k-subset of `items` (lexicographic by position).
pub fn for_each_subset<T: Copy>(items: &[T], k: usize, f: &mut impl FnMut(&[T])) {
    let n = items.len();
    if k > n {
        return;
    }
    if k == 0 {
        f(&[]);
        return;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    let mut buf: Vec<T> = Vec::with_capacity(k);
    loop {
        buf.clear();
        buf.extend(idx.iter().map(|&i| items[i]));
        f(&buf);
        // the rightmost position that can still move right
        let mut i = k;
        while i > 0 && idx[i - 1] == i - 1 + n - k {
            i -= 1;
        }
        if i == 0 {
            return;
        }
        idx[i - 1] += 1;
        for j in i..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combo_index_bijection() {
        let mut seen = vec![false; NUM_COMBOS];
        for a in 0..52u8 {
            for b in 0..a {
                let i = combo_index(Card(a), Card(b));
                assert!(!seen[i]);
                seen[i] = true;
                assert_eq!(combo_cards(i), (Card(a), Card(b)));
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn classes_cover_combos() {
        let mut counts = vec![0usize; NUM_CLASSES];
        for i in 0..NUM_COMBOS {
            let (a, b) = combo_cards(i);
            counts[class_of(a, b)] += 1;
        }
        for c in 0..NUM_CLASSES {
            assert_eq!(counts[c], class_size(c), "class {}", class_name(c));
        }
        assert_eq!(class_name(0), "AA");
        assert_eq!(class_name(1), "AKs");
        assert_eq!(class_name(13), "AKo");
        assert_eq!(class_name(168), "22");
    }

    #[test]
    fn subsets_and_binomials() {
        assert_eq!(choose(52, 5), 2_598_960);
        assert_eq!(choose(52, 7), 133_784_560);
        let mut n = 0;
        for_each_subset(&[1, 2, 3, 4, 5], 3, &mut |_s| n += 1);
        assert_eq!(n, 10);
        let mut all = Vec::new();
        for_each_subset(&[1, 2, 3, 4], 2, &mut |s| all.push((s[0], s[1])));
        assert_eq!(all, vec![(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]);
    }
}
