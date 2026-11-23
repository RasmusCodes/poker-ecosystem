use super::*;
use pe_cards::{parse_cards, parse_hand_list};

fn h(s: &str) -> Vec<Card> {
    parse_cards(s).unwrap()
}

#[test]
fn exact_turn_matches_independent_loop() {
    let game = Game::NLH;
    let hands = vec![h("AhKh"), h("QsQd"), h("7c7d")];
    let board = h("Th7h2c3d");
    let r = equity_exact(&game, &hands, &[board.clone()], &[]).unwrap();
    // independent: loop over rivers with the plain evaluator
    let used = mask_of(&board) | hands.iter().fold(0, |m, x| m | mask_of(x));
    let mut eq = [0.0; 3];
    let mut n = 0.0;
    for river in cards_of(pe_cards::FULL_DECK & !used) {
        let mut b = board.clone();
        b.push(river);
        let vals: Vec<u32> = hands.iter().map(|x| pe_eval::eval_hand(&game, x, &b)).collect();
        let best = *vals.iter().max().unwrap();
        let k = vals.iter().filter(|&&v| v == best).count() as f64;
        for i in 0..3 {
            if vals[i] == best {
                eq[i] += 1.0 / k;
            }
        }
        n += 1.0;
    }
    for i in 0..3 {
        assert!((r.equity[i] - eq[i] / n).abs() < 1e-12);
    }
    assert_eq!(r.runouts, 42);
}

#[test]
fn preflop_exact_suit_symmetry_and_sum() {
    let game = Game::NLH;
    let r = equity_exact(&game, &[h("AsKs"), h("AhKh")], &[vec![]], &[]).unwrap();
    assert_eq!(r.runouts, 1_712_304);
    assert!((r.equity[0] - 0.5).abs() < 1e-12);
    let r = equity_exact(&game, &[h("AsAh"), h("KsKh")], &[vec![]], &[]).unwrap();
    assert!((r.equity[0] + r.equity[1] - 1.0).abs() < 1e-12);
    assert!(r.equity[0] > 0.80 && r.equity[0] < 0.84, "{:?}", r.equity);
}

#[test]
fn monte_carlo_agrees_with_exact() {
    let game = Game::NLH;
    let hands = vec![h("AhKh"), h("QsQd")];
    let board = vec![h("Th7h2c")];
    let ex = equity_exact(&game, &hands, &board, &[]).unwrap();
    let mc = equity_monte_carlo(&game, &hands, &board, &[], 200_000, 9).unwrap();
    assert!((ex.equity[0] - mc.equity[0]).abs() < 4.0 * mc.std_error[0] + 1e-9, "{ex:?} {mc:?}");
    assert_eq!(ex.runouts, 990);
}

#[test]
fn omaha_and_short_deck_exact() {
    let r = equity_exact(&Game::PLO, &[h("AhAdKhQd"), h("JsTs9c8c")], &[h("7s6h2d")], &[]).unwrap();
    assert!((r.equity.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    let r = equity_exact(&Game::SHORT_DECK, &[h("AhKh"), h("9s8s")], &[h("Th7c6d")], &[]).unwrap();
    assert!((r.equity.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    // a deuce is not in the short deck
    assert!(equity_exact(&Game::SHORT_DECK, &[h("AhKh"), h("2s8s")], &[vec![]], &[]).is_err());
}

#[test]
fn double_board_and_hilo() {
    // double board: identical-strength hands by suit symmetry split exactly
    let r = equity_exact(&Game::NLH, &[h("AsKs"), h("AhKh")], &[h("2c3d4c"), h("9d8c7d")], &[]).unwrap();
    assert!((r.equity[0] - 0.5).abs() < 1e-12);
    // hi-lo: a hand with no low cards can at best scoop when no low qualifies
    let g = Game::O8;
    let r = equity_exact(&g, &[h("As2sKdKc"), h("QhQdJhJd")], &[h("3c4d9h")], &[]).unwrap();
    assert!((r.equity.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!(r.equity[0] > 0.5);
    let mc = equity_monte_carlo(&g, &[h("As2sKdKc"), h("QhQdJhJd")], &[h("3c4d9h")], &[], 100_000, 3).unwrap();
    assert!((r.equity[0] - mc.equity[0]).abs() < 4.0 * mc.std_error[0]);
}

#[test]
fn range_exact_matches_hand_exact_and_mc() {
    let game = Game::NLH;
    let a = parse_hand_list(&game, "AhKh", 0, None).unwrap();
    let b = parse_hand_list(&game, "QsQd", 0, None).unwrap();
    let board = h("Th7h2c");
    let rr = range_vs_range_exact(&game, &a, &b, &board).unwrap();
    let ex = equity_exact(&game, &[h("AhKh"), h("QsQd")], &[board.clone()], &[]).unwrap();
    assert!((rr.equity[0] - ex.equity[0]).abs() < 1e-12);
    let a = parse_hand_list(&game, "TT+,AK,AQs,KQs,JTs:0.5", 0, None).unwrap();
    let b = parse_hand_list(&game, "22-99,A2s+,KTs+,QTs+,J9s+,T8s+,AJo+,KQo", 0, None).unwrap();
    let rr = range_vs_range_exact(&game, &a, &b, &board).unwrap();
    let mc = range_equity_monte_carlo(&game, &[a, b], &[board.clone()], &[], 300_000, 5).unwrap();
    assert!((rr.equity[0] - mc.equity[0]).abs() < 4.0 * mc.std_error[0] + 1e-9, "{} vs {}", rr.equity[0], mc.equity[0]);
}

#[test]
fn range_mc_aa_vs_kk() {
    let game = Game::NLH;
    let a = parse_hand_list(&game, "AA", 0, None).unwrap();
    let b = parse_hand_list(&game, "KK", 0, None).unwrap();
    let r = range_equity_monte_carlo(&game, &[a, b], &[vec![]], &[], 200_000, 1).unwrap();
    assert!((r.equity[0] - 0.82).abs() < 0.01, "{:?}", r.equity);
}
