//! Writes a hand as PokerStars-style text (the inverse of the parser), so that
//! simulated hands can be exported and parsed back.

use std::fmt::Write as _;

use crate::model::{ActionType, Hand, PostKind, StreetName};

/// Formats milli-units the way the hand's currency is written.
pub fn fmt_amount(hand: &Hand, milli: i64) -> String {
    let sym = match hand.currency.as_str() {
        "USD" => "$",
        "EUR" => "€",
        "GBP" => "£",
        _ => "",
    };
    let neg = if milli < 0 { "-" } else { "" };
    let m = milli.abs();
    let (i, f) = (m / 1000, m % 1000);
    if sym.is_empty() {
        if f == 0 {
            format!("{neg}{i}")
        } else {
            format!("{neg}{i}.{}", format!("{f:03}").trim_end_matches('0'))
        }
    } else if f % 10 == 0 {
        format!("{neg}{sym}{i}.{:02}", f / 10)
    } else {
        format!("{neg}{sym}{i}.{f:03}")
    }
}

fn game_text(h: &Hand) -> String {
    let g = match h.game.as_str() {
        "PLO" => "Omaha",
        "PLO5" => "5 Card Omaha",
        "O8" => "Omaha Hi/Lo",
        "SD" => "6+ Hold'em",
        _ => "Hold'em",
    };
    let l = match h.limit.as_str() {
        "PL" => "Pot Limit",
        "FL" => "Limit",
        _ => "No Limit",
    };
    format!("{g} {l}")
}

fn cards(c: &[String]) -> String {
    format!("[{}]", c.join(" "))
}

pub fn write_pokerstars(h: &Hand) -> String {
    let mut s = String::new();
    let cur = match h.currency.as_str() {
        "USD" | "EUR" | "GBP" => format!(" {}", h.currency),
        _ => String::new(),
    };
    let mut tags = String::new();
    if h.bomb_pot {
        tags.push_str(" [Bomb Pot]");
    }
    if h.board_kind == "double" {
        tags.push_str(" [Double Board]");
    }
    let time = if h.time.is_empty() { "2026/01/01 00:00:00 ET" } else { &h.time };
    let _ = writeln!(s, "PokerStars Hand #{}:  {} ({}/{}{}){} - {}", h.id, game_text(h), fmt_amount(h, h.sb), fmt_amount(h, h.bb), cur, tags, time);
    let _ = writeln!(s, "Table '{}' {}-max Seat #{} is the button", h.table, h.max_seats.max(h.players.len() as u32), h.button);
    for p in &h.players {
        let _ = writeln!(s, "Seat {}: {} ({} in chips){}", p.seat, p.name, fmt_amount(h, p.stack), if p.sitting_out { " is sitting out" } else { "" });
    }
    for p in &h.posts {
        let what = match p.kind {
            PostKind::Sb => "posts small blind",
            PostKind::Bb => "posts big blind",
            PostKind::Ante => "posts the ante",
            PostKind::Straddle | PostKind::Mississippi => "posts straddle",
            PostKind::Dead => "posts dead blind",
            PostKind::Bomb => "posts bomb pot ante",
        };
        let _ = writeln!(s, "{}: {} {}", h.name_of(p.seat), what, fmt_amount(h, p.amount));
    }
    let _ = writeln!(s, "*** HOLE CARDS ***");
    for p in &h.players {
        if p.hero {
            if let Some(c) = &p.hole {
                let _ = writeln!(s, "Dealt to {} {}", p.name, cards(c));
            }
        }
    }
    let run_twice = h.board_kind == "run_twice";
    let double = h.board_kind == "double";
    let b0: Vec<String> = h.boards.first().cloned().unwrap_or_default();
    let b1: Vec<String> = h.boards.get(1).cloned().unwrap_or_default();
    // the board dealt before the all-in when the hand is run twice: the shared prefix
    let shared = if run_twice { b0.iter().zip(&b1).take_while(|(a, b)| a == b).count() } else { b0.len() };
    let street_line = |s: &mut String, name: StreetName, board: &[String], suffix: &str, prefix: &str| {
        let n = name.board_len();
        if board.len() < n {
            return;
        }
        let label = match name {
            StreetName::Flop => "FLOP",
            StreetName::Turn => "TURN",
            StreetName::River => "RIVER",
            StreetName::Preflop => return,
        };
        let line = if name == StreetName::Flop { cards(&board[..3]) } else { format!("{} {}", cards(&board[..n - 1]), cards(&board[n - 1..n])) };
        let _ = writeln!(s, "*** {prefix}{label}{suffix} *** {line}");
    };
    let mut dealt = 0usize;
    let preflop_level: i64 = h.posts.iter().filter(|p| p.kind.is_live()).map(|p| p.amount).max().unwrap_or(0);
    for st in &h.streets {
        // "raises X to Y": X is the increment over the bet being raised
        let mut level = if st.name == StreetName::Preflop { preflop_level } else { 0 };
        if st.name != StreetName::Preflop {
            // deal any street skipped without betting (cannot happen with betting streets in order)
            for name in StreetName::ALL.iter().copied().filter(|n| n.board_len() > dealt && n.board_len() <= st.name.board_len()) {
                street_line(&mut s, name, &b0, "", "");
                if double {
                    street_line(&mut s, name, &b1, " 2", "");
                }
            }
            dealt = st.name.board_len();
        }
        for a in &st.actions {
            let name = h.name_of(a.seat);
            let allin = if a.all_in { " and is all-in" } else { "" };
            let _ = match a.kind {
                ActionType::Fold => writeln!(s, "{name}: folds"),
                ActionType::Check => writeln!(s, "{name}: checks"),
                ActionType::Call => writeln!(s, "{name}: calls {}{allin}", fmt_amount(h, a.amount)),
                ActionType::Bet => writeln!(s, "{name}: bets {}{allin}", fmt_amount(h, a.amount)),
                ActionType::Raise => writeln!(s, "{name}: raises {} to {}{allin}", fmt_amount(h, (a.to - level).max(0)), fmt_amount(h, a.to)),
            };
            if a.kind.is_aggressive() {
                level = level.max(a.to);
            }
        }
    }
    for u in &h.uncalled {
        let _ = writeln!(s, "Uncalled bet ({}) returned to {}", fmt_amount(h, u.amount), h.name_of(u.seat));
    }
    // remaining board cards dealt without betting
    if run_twice {
        for (i, (b, prefix)) in [(&b0, "FIRST "), (&b1, "SECOND ")].into_iter().enumerate() {
            let _ = i;
            for name in [StreetName::Flop, StreetName::Turn, StreetName::River] {
                if name.board_len() > shared {
                    street_line(&mut s, name, b, "", prefix);
                }
            }
        }
    } else {
        for name in [StreetName::Flop, StreetName::Turn, StreetName::River] {
            if name.board_len() > dealt {
                street_line(&mut s, name, &b0, "", "");
                if double {
                    street_line(&mut s, name, &b1, " 2", "");
                }
            }
        }
    }
    let side_pots_exist = h.collected.iter().any(|c| c.pot > 0);
    let pot_label = |pot: u32| -> String {
        if pot == 0 {
            if side_pots_exist { "main pot".into() } else { "pot".into() }
        } else {
            format!("side pot-{pot}")
        }
    };
    // pots split by board get FIRST / SECOND SHOW DOWN sections; anything else
    // (an uncontested pot, a plain showdown) is written the usual way
    let split = (run_twice || double) && h.collected.iter().any(|c| c.board.is_some());
    let showdown = !h.shown.is_empty() || h.derived.as_ref().is_some_and(|d| !d.showdown_seats.is_empty());
    let write_collect = |s: &mut String, board: Option<u32>| {
        for c in h.collected.iter().filter(|c| c.board == board || (!split && board.is_none())) {
            let _ = writeln!(s, "{} collected {} from {}", h.name_of(c.seat), fmt_amount(h, c.amount), pot_label(c.pot));
        }
    };
    if split {
        write_collect(&mut s, None);
        for (k, label) in [(1u32, "FIRST SHOW DOWN"), (2, "SECOND SHOW DOWN")] {
            let _ = writeln!(s, "*** {label} ***");
            if k == 1 {
                for sh in &h.shown {
                    let _ = writeln!(s, "{}: shows {}", h.name_of(sh.seat), cards(&sh.cards));
                }
            }
            write_collect(&mut s, Some(k));
        }
    } else {
        if showdown {
            let _ = writeln!(s, "*** SHOW DOWN ***");
            for sh in &h.shown {
                if sh.description.is_empty() {
                    let _ = writeln!(s, "{}: shows {}", h.name_of(sh.seat), cards(&sh.cards));
                } else {
                    let _ = writeln!(s, "{}: shows {} ({})", h.name_of(sh.seat), cards(&sh.cards), sh.description);
                }
            }
        }
        write_collect(&mut s, None);
    }
    let _ = writeln!(s, "*** SUMMARY ***");
    let _ = writeln!(s, "Total pot {} | Rake {}", fmt_amount(h, h.total_pot), fmt_amount(h, h.rake));
    if run_twice {
        let _ = writeln!(s, "Hand was run twice");
        let _ = writeln!(s, "FIRST Board {}", cards(&b0));
        let _ = writeln!(s, "SECOND Board {}", cards(&b1));
    } else if !b0.is_empty() {
        let _ = writeln!(s, "Board {}", cards(&b0));
        if double {
            let _ = writeln!(s, "Board 2 {}", cards(&b1));
        }
    }
    for p in &h.players {
        let won: i64 = h.collected.iter().filter(|c| c.seat == p.seat).map(|c| c.amount).sum();
        let shown = h.shown.iter().find(|x| x.seat == p.seat);
        let line = match (shown, won > 0) {
            (Some(sh), true) => format!("showed {} and won ({})", cards(&sh.cards), fmt_amount(h, won)),
            (Some(sh), false) => format!("showed {} and lost", cards(&sh.cards)),
            (None, true) => format!("collected ({})", fmt_amount(h, won)),
            (None, false) => "folded".to_string(),
        };
        let _ = writeln!(s, "Seat {}: {} {}", p.seat, p.name, line);
    }
    s
}
