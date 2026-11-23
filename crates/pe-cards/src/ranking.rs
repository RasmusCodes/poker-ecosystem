//! Preflop hand-class orderings for `x%` ranges.
//!
//! [`EQUITY_ORDER`] ranks the 169 classes by all-in equity against one uniformly
//! random hand, computed with this repository's equity engine
//! (`pe rank-preflop`, Monte Carlo with 400,000 deals per class; neighbouring
//! classes closer than the Monte Carlo error may swap). It is the standard
//! "top x% of hands" ordering; it is a convention, not a statement about how
//! hands should be played.

use crate::combos::{NUM_CLASSES, class_name};

/// Class names, strongest first (equity vs one random hand; generated with
/// `pe rank-preflop --samples 400000`, so neighbours closer than about 0.1% may swap).
pub const EQUITY_ORDER: [&str; 169] = [
    "AA", "KK", "QQ", "JJ", "TT", "99", "88", "AKs", "77", "AQs", "AJs", "AKo", "ATs",
    "AQo", "AJo", "KQs", "66", "ATo", "A9s", "KJs", "A8s", "KTs", "KQo", "A7s", "A9o", "KJo",
    "55", "QJs", "K9s", "A5s", "A6s", "A8o", "KTo", "QTs", "A4s", "A7o", "K8s", "A3s", "QJo",
    "A6o", "K9o", "Q9s", "JTs", "A5o", "K7s", "A2s", "QTo", "44", "A4o", "K6s", "K8o", "Q8s",
    "A3o", "K5s", "J9s", "Q9o", "JTo", "K7o", "K4s", "A2o", "Q7s", "J8s", "K6o", "K3s", "T9s",
    "Q6s", "Q8o", "33", "K5o", "J9o", "K2s", "Q5s", "T8s", "K4o", "J7s", "Q4s", "T9o", "Q7o",
    "K3o", "J8o", "Q6o", "Q3s", "98s", "T7s", "K2o", "J6s", "22", "Q5o", "Q2s", "J5s", "T8o",
    "J7o", "Q4o", "J4s", "97s", "T6s", "J3s", "Q3o", "98o", "87s", "T7o", "J6o", "J2s", "96s",
    "Q2o", "T5s", "J5o", "T4s", "86s", "97o", "J4o", "T6o", "T3s", "95s", "76s", "J3o", "87o",
    "T2s", "85s", "96o", "T5o", "J2o", "75s", "94s", "T4o", "93s", "86o", "65s", "84s", "T3o",
    "95o", "92s", "76o", "74s", "T2o", "85o", "54s", "64s", "83s", "94o", "75o", "82s", "65o",
    "73s", "93o", "63s", "53s", "84o", "92o", "74o", "43s", "72s", "54o", "64o", "52s", "62s",
    "83o", "42s", "82o", "73o", "53o", "63o", "32s", "43o", "72o", "52o", "62o", "42o", "32o",
];

/// Bill Chen's quick hand-strength formula (a documented fallback ordering).
pub fn chen_score(class: usize) -> f64 {
    let name = class_name(class);
    let ch: Vec<char> = name.chars().collect();
    let r = |c: char| crate::card::parse_rank(c).unwrap() as i32;
    let (hi, lo) = (r(ch[0]), r(ch[1]));
    let pts = |rank: i32| -> f64 {
        match rank {
            12 => 10.0,
            11 => 8.0,
            10 => 7.0,
            9 => 6.0,
            x => (x + 2) as f64 / 2.0,
        }
    };
    if hi == lo {
        return (pts(hi) * 2.0).max(5.0);
    }
    let mut s = pts(hi);
    if ch.get(2) == Some(&'s') {
        s += 2.0;
    }
    let gap = hi - lo - 1;
    s -= match gap {
        0 => 0.0,
        1 => 1.0,
        2 => 2.0,
        3 => 4.0,
        _ => 5.0,
    };
    if gap <= 1 && hi < 10 {
        s += 1.0;
    }
    s.ceil()
}

/// The 169 class indices strongest first: the equity order when available,
/// otherwise the Chen formula (ties broken by class index).
pub fn preflop_order() -> Vec<usize> {
    if EQUITY_ORDER.len() == NUM_CLASSES {
        return EQUITY_ORDER.iter().map(|n| class_index_by_name(n).expect("valid class name")).collect();
    }
    let mut v: Vec<usize> = (0..NUM_CLASSES).collect();
    v.sort_by(|&a, &b| chen_score(b).partial_cmp(&chen_score(a)).unwrap().then(a.cmp(&b)));
    v
}

/// Grid index of a class name such as `"AKs"`.
pub fn class_index_by_name(name: &str) -> Option<usize> {
    (0..NUM_CLASSES).find(|&c| class_name(c) == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_is_a_permutation_with_aces_first() {
        let v = preflop_order();
        assert_eq!(v.len(), NUM_CLASSES);
        let mut s = v.clone();
        s.sort();
        assert_eq!(s, (0..NUM_CLASSES).collect::<Vec<_>>());
        assert_eq!(class_name(v[0]), "AA");
    }
}
