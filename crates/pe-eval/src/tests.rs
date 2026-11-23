use super::*;
use pe_cards::{Rng, SHORT_DECK, cards_of, mask_of, parse_cards};

/// An independent, deliberately simple evaluator for five cards: counts ranks,
/// sorts groups, and returns (category order, tie-break ranks).
fn naive5(cards: &[Card], rules: HighRules) -> (u32, Vec<u32>) {
    assert_eq!(cards.len(), 5);
    let short = rules.is_short_deck();
    let mut counts = [0u32; 13];
    for c in cards {
        counts[c.rank() as usize] += 1;
    }
    let flush = cards.iter().all(|c| c.suit() == cards[0].suit());
    let mut ranks: Vec<u32> = cards.iter().map(|c| c.rank() as u32).collect();
    ranks.sort_unstable_by(|a, b| b.cmp(a));
    let distinct = {
        let mut d = ranks.clone();
        d.dedup();
        d.len() == 5
    };
    let straight_top = if distinct && ranks[0] - ranks[4] == 4 {
        Some(ranks[0])
    } else if distinct && !short && ranks == vec![12, 3, 2, 1, 0] {
        Some(3)
    } else if distinct && short && ranks == vec![12, 7, 6, 5, 4] {
        Some(7)
    } else {
        None
    };
    let mut groups: Vec<(u32, u32)> = (0..13u32).filter(|&r| counts[r as usize] > 0).map(|r| (counts[r as usize], r)).collect();
    groups.sort_unstable_by(|a, b| b.cmp(a));
    let shape: Vec<u32> = groups.iter().map(|g| g.0).collect();
    let tiebreak: Vec<u32> = groups.iter().map(|g| g.1).collect();
    // independent statement of the category orders
    let (sf, quads, fh, fl, st, trips, tp, pair, high) = match rules {
        HighRules::Standard => (8, 7, 6, 5, 4, 3, 2, 1, 0),
        HighRules::ShortDeckTriton => (8, 7, 5, 6, 3, 4, 2, 1, 0),
        HighRules::ShortDeckClassic => (8, 7, 5, 6, 4, 3, 2, 1, 0),
    };
    if let Some(t) = straight_top {
        if flush {
            return (sf, vec![t]);
        }
    }
    if shape == [4, 1] {
        return (quads, tiebreak);
    }
    if shape == [3, 2] {
        return (fh, tiebreak);
    }
    if flush {
        return (fl, ranks);
    }
    if let Some(t) = straight_top {
        return (st, vec![t]);
    }
    if shape == [3, 1, 1] {
        return (trips, tiebreak);
    }
    if shape == [2, 2, 1] {
        return (tp, tiebreak);
    }
    if shape == [2, 1, 1, 1] {
        return (pair, tiebreak);
    }
    (high, ranks)
}

fn all_cards() -> Vec<Card> {
    (0..52).map(Card).collect()
}

#[test]
fn five_card_counts_and_classes() {
    let mut counts = [0u64; 9];
    let mut classes = std::collections::BTreeSet::new();
    let cards = all_cards();
    pe_cards::combos::for_each_subset(&cards, 5, &mut |s| {
        let v = eval7(mask_of(s));
        counts[category(v) as usize] += 1;
        classes.insert(v);
    });
    assert_eq!(counts, [1_302_540, 1_098_240, 123_552, 54_912, 10_200, 5_108, 3_744, 624, 40]);
    assert_eq!(classes.len(), 7462);
    assert_eq!(dense_rank(eval7(mask_of(&parse_cards("AsKsQsJsTs").unwrap()))), 7462);
    assert_eq!(dense_rank(eval7(mask_of(&parse_cards("7c5d4h3s2c").unwrap()))), 1);
}

#[test]
fn seven_card_counts() {
    // the published distribution of all 133,784,560 seven-card hands
    let mut counts = [0u64; 9];
    let bits: Vec<u64> = (0..52).map(|i| Card(i).mask()).collect();
    for a in 0..52 {
        let ma = bits[a];
        for b in a + 1..52 {
            let mb = ma | bits[b];
            for c in b + 1..52 {
                let mc = mb | bits[c];
                for d in c + 1..52 {
                    let md = mc | bits[d];
                    for e in d + 1..52 {
                        let me = md | bits[e];
                        for f in e + 1..52 {
                            let mf = me | bits[f];
                            for g in f + 1..52 {
                                counts[category(eval7(mf | bits[g])) as usize] += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(
        counts,
        [23_294_460, 58_627_800, 31_433_400, 6_461_620, 6_180_020, 4_047_644, 3_473_184, 224_848, 41_584]
    );
}

#[test]
fn agrees_with_naive_evaluator() {
    let mut rng = Rng::new(11);
    for rules in [HighRules::Standard, HighRules::ShortDeckTriton, HighRules::ShortDeckClassic] {
        let deck: Vec<Card> = if rules.is_short_deck() { cards_of(SHORT_DECK) } else { all_cards() };
        for _ in 0..200_000 {
            let mut d = deck.clone();
            rng.shuffle(&mut d);
            let (h1, h2) = (&d[0..5], &d[5..10]);
            let (v1, v2) = (eval_high(mask_of(h1), rules), eval_high(mask_of(h2), rules));
            let (n1, n2) = (naive5(h1, rules), naive5(h2, rules));
            assert_eq!(v1.cmp(&v2), n1.cmp(&n2), "{h1:?} vs {h2:?} under {rules:?}");
        }
    }
}

#[test]
fn seven_cards_equal_best_five_subset() {
    let mut rng = Rng::new(5);
    for rules in [HighRules::Standard, HighRules::ShortDeckTriton, HighRules::ShortDeckClassic] {
        let deck: Vec<Card> = if rules.is_short_deck() { cards_of(SHORT_DECK) } else { all_cards() };
        for n in [6usize, 7] {
            for _ in 0..50_000 {
                let mut d = deck.clone();
                rng.shuffle(&mut d);
                let hand = &d[..n];
                let mut best = 0;
                pe_cards::combos::for_each_subset(hand, 5, &mut |s| best = best.max(eval_high(mask_of(s), rules)));
                assert_eq!(eval_high(mask_of(hand), rules), best, "{hand:?} {rules:?}");
            }
        }
    }
}

#[test]
fn short_deck_five_card_counts() {
    // derived combinatorially for the 36-card deck (see docs/VALIDATION.md)
    let cards = cards_of(SHORT_DECK);
    let mut counts = [0u64; 9];
    pe_cards::combos::for_each_subset(&cards, 5, &mut |s| {
        counts[category(eval_high(mask_of(s), HighRules::ShortDeckTriton)) as usize] += 1;
    });
    assert_eq!(counts, [122_400, 193_536, 36_288, 16_128, 6_120, 480, 1_728, 288, 24]);
    assert_eq!(counts.iter().sum::<u64>(), 376_992);
}

#[test]
fn short_deck_orderings() {
    let v = |s: &str, r| eval_high(mask_of(&parse_cards(s).unwrap()), r);
    let t = HighRules::ShortDeckTriton;
    let c = HighRules::ShortDeckClassic;
    // flush beats full house in both short-deck rule sets
    assert!(v("AhKh9h8h6h", t) > v("AsAdAcKdKc", t));
    assert!(v("AhKh9h8h6h", c) > v("AsAdAcKdKc", c));
    // trips beat straights under Triton rules, lose to them under classic rules
    assert!(v("7s7d7cKdQc", t) > v("TsJdQcKdAc", t));
    assert!(v("7s7d7cKdQc", c) < v("TsJdQcKdAc", c));
    // A-6-7-8-9 is a straight, and the lowest one
    assert_eq!(category(v("As6d7c8d9c", t)), Category::Straight);
    assert!(v("As6d7c8d9c", t) < v("6s7d8c9dTc", t));
    // and A-2-3-4-5 is a straight in the standard game
    assert_eq!(category(v("As2d3c4d5c", HighRules::Standard)), Category::Straight);
    // seven cards holding trips and a straight: Triton takes the trips
    let both = mask_of(&parse_cards("TsTdTcJdQcKhAh").unwrap());
    assert_eq!(category(eval_high(both, t)), Category::Trips);
    assert_eq!(category(eval_high(both, c)), Category::Straight);
}

#[test]
fn omaha_two_plus_three() {
    let h = |s: &str| parse_cards(s).unwrap();
    // four aces in hand: only two may play
    let v = eval_omaha_high(&h("AhAdAcAs"), &h("KhKdKc2s3s"), HighRules::Standard);
    assert_eq!(category(v), Category::FullHouse);
    assert_eq!(describe(v, HighRules::Standard), "a full house, Kings full of Aces");
    // four hearts on board, one in hand: no flush
    let v = eval_omaha_high(&h("AhKdQcJs"), &h("2h3h4h5h9c"), HighRules::Standard);
    assert_eq!(category(v), Category::HighCard);
    // two hearts in hand: flush
    let v = eval_omaha_high(&h("AhKhQcJs"), &h("2h3h4h5d9c"), HighRules::Standard);
    assert_eq!(category(v), Category::Flush);
    // five-card Omaha uses the same rule: J-T from hand with 9-8-7 from the board
    let v = eval_omaha_high(&h("AhKhQcJsTs"), &h("9d8c7h3d4s"), HighRules::Standard);
    assert_eq!(category(v), Category::Straight);
    assert_eq!(describe(v, HighRules::Standard), "a straight, Seven to Jack");
    // and no straight when only two board cards connect
    let v = eval_omaha_high(&h("AhKhQcJsTs"), &h("9d8c2h3d4s"), HighRules::Standard);
    assert_eq!(category(v), Category::HighCard);
}

#[test]
fn omaha_lows() {
    let h = |s: &str| parse_cards(s).unwrap();
    assert_eq!(describe_low(eval_omaha_low8(&h("Ad2cKhKd"), &h("3s4h5dKsQc"))), "5,4,3,2,A low");
    assert_eq!(eval_omaha_low8(&h("AsAd2s2d"), &h("3c4cKdQhJh")), 0);
    // counterfeited deuce: A2 plays with 3-4-7 from a 2-3-4-7-8 board
    assert_eq!(describe_low(eval_omaha_low8(&h("As2dKhQh"), &h("2c3c4d7h8s"))), "7,4,3,2,A low");
    // no low with fewer than three low board ranks
    assert_eq!(eval_omaha_low8(&h("As2d3h4h"), &h("5c5d9sTsJh")), 0);
    // better low wins
    let a = eval_omaha_low8(&h("As2dKhQh"), &h("3c4c8dTsJh"));
    let b = eval_omaha_low8(&h("As3dKhQh"), &h("2c4c8dTsJh"));
    assert!(a > 0 && b > 0);
    assert_eq!(a, b); // both make 8-4-3-2-A
}

#[test]
fn descriptions() {
    let v = |s: &str| eval7(mask_of(&parse_cards(s).unwrap()));
    let r = HighRules::Standard;
    assert_eq!(describe(v("QsQd7c4h2s"), r), "a pair of Queens");
    assert_eq!(describe(v("KsKd4c4h2s"), r), "two pair, Kings and Fours");
    assert_eq!(describe(v("9sTdJcQhKs"), r), "a straight, Nine to King");
    assert_eq!(describe(v("As2d3c4h5s"), r), "a straight, Ace to Five");
    assert_eq!(describe(v("AsKsQsJsTs"), r), "a Royal Flush");
    assert_eq!(describe(v("As9s7s4s2s"), r), "a flush, Ace high");
    assert_eq!(describe(v("Ac9d7s4h2s"), r), "high card Ace");
    assert_eq!(
        describe(eval_high(mask_of(&parse_cards("As6d7c8d9c").unwrap()), HighRules::ShortDeckTriton), HighRules::ShortDeckTriton),
        "a straight, Ace to Nine"
    );
}
