//! PokerStars hand-history parser.
//!
//! Supported: cash games and tournaments; Hold'em (No Limit, Pot Limit, Limit),
//! Omaha, Omaha Hi/Lo, 5 Card Omaha and 6+ Hold'em; antes, dead blinds,
//! straddles, all-ins, uncalled bets, main and side pots, run-it-twice sections,
//! hands shown at showdown or revealed in the summary.
//!
//! Extensions for formats PokerStars does not deal (documented in
//! docs/HAND-HISTORIES.md; real sites write these differently and need adapters):
//! * a header tag `[Bomb Pot]` with `posts bomb pot ante $X` lines and no
//!   preflop betting;
//! * a header tag `[Double Board]` with `*** FLOP 2 ***`, `*** TURN 2 ***` and
//!   `*** RIVER 2 ***` lines for the second board, and the pot split by
//!   `*** FIRST SHOW DOWN ***` / `*** SECOND SHOW DOWN ***` sections as when a
//!   hand is run twice;
//! * `posts straddle $X` lines (as several sites write them).
//!
//! Unknown lines are skipped and reported as warnings; nothing panics.

use crate::model::{Action, ActionType, Collected, Hand, Player, Post, PostKind, Shown, Street, StreetName, Uncalled, assign_positions};

/// Parses a decimal amount with optional currency symbol and thousands separators
/// into thousandths: `$1,234.56` -> 1234560.
pub fn parse_amount(s: &str) -> Option<i64> {
    let t: String = s.trim().chars().filter(|c| !matches!(c, '$' | '€' | '£' | ',' | ' ')).collect();
    let t = t.trim_end_matches(|c: char| c.is_ascii_alphabetic()).trim();
    if t.is_empty() {
        return None;
    }
    let (neg, t) = match t.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, t),
    };
    let (int, frac) = t.split_once('.').unwrap_or((t, ""));
    if !int.chars().all(|c| c.is_ascii_digit()) || !frac.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let i: i64 = if int.is_empty() { 0 } else { int.parse().ok()? };
    let mut f: String = frac.chars().take(3).collect();
    while f.len() < 3 {
        f.push('0');
    }
    let v = i * 1000 + f.parse::<i64>().ok()?;
    Some(if neg { -v } else { v })
}

/// The bracketed card groups of a line: `[7c 8d 2h] [Ks]` -> [["7c","8d","2h"], ["Ks"]].
fn bracket_groups(line: &str) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    let mut rest = line;
    while let Some(a) = rest.find('[') {
        let Some(b) = rest[a..].find(']') else { break };
        let inner = &rest[a + 1..a + b];
        let cards: Vec<String> = inner.split_whitespace().map(|s| s.to_string()).collect();
        if !cards.is_empty() && cards.iter().all(|c| pe_cards::Card::parse(c).is_ok()) {
            out.push(cards.iter().map(|c| pe_cards::Card::parse(c).unwrap().to_string()).collect());
        }
        rest = &rest[a + b + 1..];
    }
    out
}

fn header(line: &str, hand: &mut Hand, warnings: &mut Vec<String>) {
    hand.site = "PokerStars".into();
    if let Some(i) = line.find("Hand #") {
        let rest = &line[i + 6..];
        hand.id = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    }
    let lower = line.to_ascii_lowercase();
    let tournament = lower.contains("tournament #");
    hand.game = if lower.contains("5 card omaha") {
        "PLO5"
    } else if lower.contains("omaha hi/lo") {
        "O8"
    } else if lower.contains("omaha") {
        "PLO"
    } else if lower.contains("6+ hold'em") {
        "SD"
    } else {
        "NLH"
    }
    .to_string();
    hand.limit = if lower.contains("no limit") {
        "NL"
    } else if lower.contains("pot limit") {
        "PL"
    } else {
        "FL"
    }
    .to_string();
    hand.bomb_pot = lower.contains("[bomb pot]");
    if lower.contains("[double board]") {
        hand.board_kind = "double".into();
    }
    // stakes: the last parenthesised a/b group
    let mut stakes: Option<&str> = None;
    let mut search = line;
    while let Some(a) = search.find('(') {
        let Some(b) = search[a..].find(')') else { break };
        let inner = &search[a + 1..a + b];
        if inner.contains('/') {
            stakes = Some(inner);
        }
        search = &search[a + b + 1..];
    }
    match stakes {
        Some(s) => {
            let mut parts = s.split('/');
            let sb = parts.next().and_then(parse_amount);
            let bb = parts.next().and_then(|p| parse_amount(p.split_whitespace().next().unwrap_or("")));
            hand.sb = sb.unwrap_or(0);
            hand.bb = bb.unwrap_or(0);
            hand.currency = if tournament {
                "chips".into()
            } else if s.contains('€') || s.contains("EUR") {
                "EUR".into()
            } else if s.contains('£') || s.contains("GBP") {
                "GBP".into()
            } else if s.contains('$') || s.contains("USD") {
                "USD".into()
            } else {
                "chips".into()
            };
        }
        None => warnings.push(format!("hand {}: no stakes in the header", hand.id)),
    }
    if let Some(i) = line.rfind(" - ") {
        hand.time = line[i + 3..].trim().to_string();
    }
}

fn table_line(line: &str, hand: &mut Hand) {
    // Table 'Name' 6-max Seat #3 is the button
    if let (Some(a), Some(b)) = (line.find('\''), line.rfind('\'')) {
        if b > a {
            hand.table = line[a + 1..b].to_string();
        }
    }
    if let Some(i) = line.find("-max") {
        let digits: String = line[..i].chars().rev().take_while(|c| c.is_ascii_digit()).collect::<Vec<_>>().into_iter().rev().collect();
        hand.max_seats = digits.parse().unwrap_or(0);
    }
    if let Some(i) = line.find("Seat #") {
        hand.button = line[i + 6..].chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap_or(0);
    }
}

fn seat_line(line: &str) -> Option<Player> {
    // Seat 1: Player One ($10.00 in chips) [is sitting out]
    let rest = line.strip_prefix("Seat ")?;
    let colon = rest.find(':')?;
    let seat: u32 = rest[..colon].trim().parse().ok()?;
    let body = &rest[colon + 1..];
    let open = body.rfind(" (")?;
    let close = body[open..].find(')')? + open;
    let inner = &body[open + 2..close];
    if !inner.contains("in chips") {
        return None;
    }
    let stack = parse_amount(inner.split(" in chips").next()?)?;
    let name = body[..open].trim().to_string();
    let tail = &body[close + 1..];
    Some(Player { seat, name, stack, position: None, hole: None, hero: false, sitting_out: tail.contains("sitting out") || tail.contains("out of hand") })
}

/// Finds the player whose name starts the line and is followed by `": "`.
fn split_name<'a>(line: &'a str, names: &[(String, u32)]) -> Option<(u32, &'a str)> {
    let mut best: Option<(usize, u32)> = None;
    for (n, seat) in names {
        if line.len() > n.len() + 1 && line.starts_with(n.as_str()) && line[n.len()..].starts_with(": ") && best.is_none_or(|b| n.len() > b.0) {
            best = Some((n.len(), *seat));
        }
    }
    best.map(|(len, seat)| (seat, &line[len + 2..]))
}

/// The player named at the start of a line that is not `"name: ..."` (e.g. collections).
fn leading_name(line: &str, names: &[(String, u32)]) -> Option<(u32, usize)> {
    let mut best: Option<(usize, u32)> = None;
    for (n, seat) in names {
        if line.starts_with(n.as_str()) && line[n.len()..].starts_with(' ') && best.is_none_or(|b| n.len() > b.0) {
            best = Some((n.len(), *seat));
        }
    }
    best.map(|(len, seat)| (seat, len))
}

struct Ctx {
    street: Option<StreetName>,
    contrib: std::collections::BTreeMap<u32, i64>,
    boards: Vec<Vec<String>>,
    run_boards: [Vec<String>; 2],
    in_summary: bool,
    showdown_board: Option<u32>,
    run_twice: bool,
}

fn add_shown(hand: &mut Hand, seat: u32, cards: Vec<String>) {
    if cards.is_empty() || hand.shown.iter().any(|s| s.seat == seat) {
        return;
    }
    hand.shown.push(Shown { seat, cards, description: String::new() });
}

fn action_line(rest: &str, seat: u32, hand: &mut Hand, ctx: &mut Ctx, warnings: &mut Vec<String>) {
    let all_in = rest.contains("and is all-in");
    let r = rest.replace(" and is all-in", "");
    let r = r.trim();
    let words: Vec<&str> = r.split_whitespace().collect();
    let first = words.first().copied().unwrap_or("");
    let push = |hand: &mut Hand, ctx: &mut Ctx, kind: ActionType, amount: i64| {
        let street = ctx.street.unwrap_or(StreetName::Preflop);
        let c = ctx.contrib.entry(seat).or_insert(0);
        *c += amount;
        let to = *c;
        if hand.streets.last().is_none_or(|s| s.name != street) {
            hand.streets.push(Street { name: street, actions: Vec::new() });
        }
        hand.streets.last_mut().unwrap().actions.push(Action { seat, kind, amount, to, all_in, truth: None });
    };
    match first {
        "folds" => {
            push(hand, ctx, ActionType::Fold, 0);
            if let Some(g) = bracket_groups(r).into_iter().next() {
                add_shown(hand, seat, g);
            }
        }
        "checks" => push(hand, ctx, ActionType::Check, 0),
        "calls" => {
            let a = words.get(1).and_then(|w| parse_amount(w)).unwrap_or(0);
            push(hand, ctx, ActionType::Call, a);
        }
        "bets" => {
            let a = words.get(1).and_then(|w| parse_amount(w)).unwrap_or(0);
            push(hand, ctx, ActionType::Bet, a);
        }
        "raises" => {
            // raises $X to $Y
            let to = words.iter().position(|w| *w == "to").and_then(|i| words.get(i + 1)).and_then(|w| parse_amount(w));
            match to {
                Some(t) => {
                    let have = *ctx.contrib.get(&seat).unwrap_or(&0);
                    push(hand, ctx, ActionType::Raise, t - have);
                }
                None => warnings.push(format!("hand {}: unreadable raise '{r}'", hand.id)),
            }
        }
        "posts" => {
            let amount = words.last().and_then(|w| parse_amount(w)).unwrap_or(0);
            let lower = r.to_ascii_lowercase();
            let (kind, live) = if lower.contains("small & big blinds") {
                // a dead small blind plus a live big blind
                let dead = (amount - hand.bb).max(0);
                if dead > 0 {
                    hand.posts.push(Post { seat, kind: PostKind::Dead, amount: dead });
                }
                (PostKind::Bb, amount - dead)
            } else if lower.contains("small blind") {
                (PostKind::Sb, amount)
            } else if lower.contains("big blind") {
                (PostKind::Bb, amount)
            } else if lower.contains("bomb pot") {
                (PostKind::Bomb, amount)
            } else if lower.contains("ante") {
                (PostKind::Ante, amount)
            } else if lower.contains("straddle") {
                (PostKind::Straddle, amount)
            } else if lower.contains("dead") {
                (PostKind::Dead, amount)
            } else {
                warnings.push(format!("hand {}: unknown post '{r}'", hand.id));
                return;
            };
            hand.posts.push(Post { seat, kind, amount: live });
            if kind.is_live() {
                *ctx.contrib.entry(seat).or_insert(0) += live;
            }
            if kind == PostKind::Ante && hand.ante == 0 {
                hand.ante = amount;
            }
        }
        "shows" => {
            if let Some(g) = bracket_groups(r).into_iter().next() {
                add_shown(hand, seat, g);
            }
        }
        "mucks" | "doesn't" | "is" | "sits" | "has" | "said," | "leaves" | "joins" | "will" | "was" | "re-buys" | "adds" | "finished" | "wins" => {}
        _ => warnings.push(format!("hand {}: unknown action '{r}'", hand.id)),
    }
}

/// Parses every hand in a PokerStars hand-history text.
pub fn parse_pokerstars(text: &str) -> (Vec<Hand>, Vec<String>) {
    let mut hands = Vec::new();
    let mut warnings = Vec::new();
    let mut block: Vec<&str> = Vec::new();
    let flush = |block: &mut Vec<&str>, hands: &mut Vec<Hand>, warnings: &mut Vec<String>| {
        if !block.is_empty() {
            if let Some(h) = parse_block(block, warnings) {
                hands.push(h);
            }
            block.clear();
        }
    };
    for raw in text.trim_start_matches('\u{feff}').lines() {
        let line = raw.trim_end();
        if line.contains("Hand #") && (line.starts_with("PokerStars") || line.contains("PokerStars")) {
            flush(&mut block, &mut hands, &mut warnings);
        }
        if line.trim().is_empty() {
            continue;
        }
        block.push(line);
    }
    flush(&mut block, &mut hands, &mut warnings);
    (hands, warnings)
}

fn parse_block(lines: &[&str], warnings: &mut Vec<String>) -> Option<Hand> {
    let first = lines.first()?;
    if !first.contains("Hand #") {
        warnings.push(format!("skipped text that is not a hand: '{}'", first.chars().take(60).collect::<String>()));
        return None;
    }
    let mut hand = Hand::default();
    header(first, &mut hand, warnings);
    let mut ctx = Ctx {
        street: if hand.bomb_pot { None } else { Some(StreetName::Preflop) },
        contrib: Default::default(),
        boards: vec![Vec::new(), Vec::new()],
        run_boards: [Vec::new(), Vec::new()],
        in_summary: false,
        showdown_board: None,
        run_twice: false,
    };
    let mut names: Vec<(String, u32)> = Vec::new();
    for &line in &lines[1..] {
        let l = line.trim();
        if l.starts_with("Table '") {
            table_line(l, &mut hand);
            continue;
        }
        if !ctx.in_summary {
            if let Some(p) = seat_line(l) {
                names.push((p.name.clone(), p.seat));
                hand.players.push(p);
                continue;
            }
        }
        if let Some(marker) = l.strip_prefix("*** ") {
            let name = marker.split(" ***").next().unwrap_or("").trim();
            let groups = bracket_groups(l);
            let cards: Vec<String> = groups.concat();
            match name {
                "HOLE CARDS" => ctx.street = Some(StreetName::Preflop),
                "FLOP" | "TURN" | "RIVER" => {
                    let s = match name {
                        "FLOP" => StreetName::Flop,
                        "TURN" => StreetName::Turn,
                        _ => StreetName::River,
                    };
                    ctx.street = Some(s);
                    ctx.contrib.clear();
                    ctx.boards[0] = cards;
                    if hand.streets.last().is_none_or(|x| x.name != s) {
                        hand.streets.push(Street { name: s, actions: Vec::new() });
                    }
                }
                "FLOP 2" | "TURN 2" | "RIVER 2" => ctx.boards[1] = cards,
                "FIRST FLOP" | "FIRST TURN" | "FIRST RIVER" => {
                    ctx.run_twice = true;
                    ctx.run_boards[0] = cards;
                }
                "SECOND FLOP" | "SECOND TURN" | "SECOND RIVER" => {
                    ctx.run_twice = true;
                    ctx.run_boards[1] = cards;
                }
                "SHOW DOWN" => ctx.showdown_board = None,
                "FIRST SHOW DOWN" => ctx.showdown_board = Some(1),
                "SECOND SHOW DOWN" => ctx.showdown_board = Some(2),
                "SUMMARY" => ctx.in_summary = true,
                _ => warnings.push(format!("hand {}: unknown section '{name}'", hand.id)),
            }
            continue;
        }
        if ctx.in_summary {
            summary_line(l, &mut hand, &names, &mut ctx);
            continue;
        }
        if let Some(rest) = l.strip_prefix("Dealt to ") {
            if let Some((seat, len)) = leading_name(rest, &names).or_else(|| names.iter().find(|(n, _)| rest == n).map(|(n, s)| (*s, n.len()))) {
                let g = bracket_groups(&rest[len..]);
                if let Some(p) = hand.players.iter_mut().find(|p| p.seat == seat) {
                    if let Some(cards) = g.into_iter().last() {
                        p.hole = Some(cards);
                        p.hero = true;
                    }
                }
            }
            continue;
        }
        if let Some(rest) = l.strip_prefix("Uncalled bet (") {
            if let Some(close) = rest.find(')') {
                let amount = parse_amount(&rest[..close]).unwrap_or(0);
                if let Some(name) = rest[close..].split(" returned to ").nth(1) {
                    if let Some((_, seat)) = names.iter().find(|(n, _)| n == name.trim()) {
                        hand.uncalled.push(Uncalled { seat: *seat, amount });
                        if let Some(c) = ctx.contrib.get_mut(seat) {
                            *c -= amount;
                        }
                    }
                }
            }
            continue;
        }
        if let Some((seat, rest)) = split_name(l, &names) {
            action_line(rest, seat, &mut hand, &mut ctx, warnings);
            continue;
        }
        if let Some((seat, len)) = leading_name(l, &names) {
            let rest = l[len..].trim();
            if let Some(after) = rest.strip_prefix("collected ") {
                let amount = parse_amount(after.split_whitespace().next().unwrap_or("")).unwrap_or(0);
                let pot = if let Some(i) = after.find("side pot-") {
                    after[i + 9..].chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap_or(1)
                } else if after.contains("side pot") {
                    1
                } else {
                    0
                };
                hand.collected.push(Collected { seat, amount, pot, board: ctx.showdown_board });
                continue;
            }
            if rest.starts_with("shows") || rest.starts_with("mucks") {
                if let Some(g) = bracket_groups(rest).into_iter().next() {
                    add_shown(&mut hand, seat, g);
                }
                continue;
            }
            // chat, timeouts, connection notices
            continue;
        }
        warnings.push(format!("hand {}: skipped '{}'", hand.id, l.chars().take(80).collect::<String>()));
    }
    // streets dealt without betting (all-in run-outs) are not betting rounds
    hand.streets.retain(|s| !s.actions.is_empty());
    // boards
    if ctx.run_twice {
        hand.board_kind = "run_twice".into();
        hand.boards = vec![ctx.run_boards[0].clone(), ctx.run_boards[1].clone()];
    } else if hand.board_kind == "double" {
        hand.boards = vec![ctx.boards[0].clone(), ctx.boards[1].clone()];
    } else if !ctx.boards[0].is_empty() {
        hand.boards = vec![ctx.boards[0].clone()];
    } else if hand.boards.is_empty() {
        hand.boards = vec![Vec::new()];
    }
    // descriptions of shown hands (from the evaluator, for a consistent vocabulary)
    if let Some(game) = hand.game_def() {
        let board = hand.board_cards(0);
        if board.len() == 5 {
            for s in hand.shown.iter_mut() {
                if let Some(cards) = crate::model::strings_to_cards(&s.cards) {
                    if cards.len() == game.hole_cards {
                        let v = pe_eval::eval_hand(&game, &cards, &board);
                        s.description = pe_eval::describe(v, game.high);
                    }
                }
            }
        }
    }
    for p in hand.players.iter_mut() {
        if p.hole.is_none() {
            if let Some(s) = hand.shown.iter().find(|s| s.seat == p.seat) {
                p.hole = Some(s.cards.clone());
            }
        }
    }
    assign_positions(&mut hand);
    Some(hand)
}

fn summary_line(l: &str, hand: &mut Hand, names: &[(String, u32)], ctx: &mut Ctx) {
    if let Some(rest) = l.strip_prefix("Total pot ") {
        let total = parse_amount(rest.split_whitespace().next().unwrap_or("")).unwrap_or(0);
        hand.total_pot = total;
        if let Some(i) = rest.find("Rake ") {
            hand.rake = parse_amount(rest[i + 5..].split_whitespace().next().unwrap_or("")).unwrap_or(0);
        }
        return;
    }
    if l.starts_with("Hand was run twice") {
        ctx.run_twice = true;
        return;
    }
    if l.starts_with("FIRST Board") || l.starts_with("SECOND Board") || l.starts_with("Board") {
        let cards: Vec<String> = bracket_groups(l).concat();
        if l.starts_with("FIRST") {
            if ctx.run_boards[0].len() < cards.len() {
                ctx.run_boards[0] = cards;
            }
        } else if l.starts_with("SECOND") {
            if ctx.run_boards[1].len() < cards.len() {
                ctx.run_boards[1] = cards;
            }
        } else if l.starts_with("Board 2") {
            if ctx.boards[1].len() < cards.len() {
                ctx.boards[1] = cards;
            }
        } else if ctx.boards[0].len() < cards.len() {
            ctx.boards[0] = cards;
        }
        return;
    }
    if let Some(rest) = l.strip_prefix("Seat ") {
        if let Some(colon) = rest.find(": ") {
            let body = &rest[colon + 2..];
            if let Some((seat, _)) = leading_name(body, names).or_else(|| names.iter().find(|(n, _)| body.starts_with(n.as_str())).map(|(n, s)| (*s, n.len()))) {
                if body.contains("showed [") || body.contains("mucked [") {
                    if let Some(g) = bracket_groups(body).into_iter().next() {
                        add_shown(hand, seat, g);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amounts() {
        assert_eq!(parse_amount("$0.05"), Some(50));
        assert_eq!(parse_amount("$1,234.56"), Some(1_234_560));
        assert_eq!(parse_amount("1500"), Some(1_500_000));
        assert_eq!(parse_amount("€2"), Some(2000));
        assert_eq!(parse_amount("0.005"), Some(5));
        assert_eq!(parse_amount("abc"), None);
    }

    #[test]
    fn parses_a_basic_hand() {
        let text = "PokerStars Hand #1001:  Hold'em No Limit ($0.05/$0.10 USD) - 2026/11/05 21:00:00 ET
Table 'Test' 6-max Seat #4 is the button
Seat 1: Alice ($10.00 in chips)
Seat 2: Bob Smith ($12.34 in chips)
Seat 3: Hero ($10.00 in chips)
Seat 4: Dan ($9.50 in chips)
Alice: posts small blind $0.05
Bob Smith: posts big blind $0.10
*** HOLE CARDS ***
Dealt to Hero [Ah Kd]
Hero: raises $0.20 to $0.30
Dan: folds
Alice: folds
Bob Smith: calls $0.20
*** FLOP *** [7c 8d 2h]
Bob Smith: checks
Hero: bets $0.40
Bob Smith: folds
Uncalled bet ($0.40) returned to Hero
Hero collected $0.63 from pot
*** SUMMARY ***
Total pot $0.65 | Rake $0.02
Board [7c 8d 2h]
Seat 1: Alice (small blind) folded before Flop
Seat 2: Bob Smith (big blind) folded on the Flop
Seat 3: Hero (button) collected ($0.63)
Seat 4: Dan folded before Flop (didn't bet)
";
        let (hands, warnings) = parse_pokerstars(text);
        assert!(warnings.is_empty(), "{warnings:?}");
        let h = &hands[0];
        assert_eq!(h.id, "1001");
        assert_eq!((h.sb, h.bb), (50, 100));
        assert_eq!(h.players.len(), 4);
        assert_eq!(h.players[1].name, "Bob Smith");
        assert_eq!(h.players[3].position.as_deref(), Some("BTN"));
        assert_eq!(h.players[0].position.as_deref(), Some("SB"));
        assert_eq!(h.players[2].position.as_deref(), Some("CO"));
        assert!(h.players[2].hero);
        assert_eq!(h.streets.len(), 2);
        let raise = &h.streets[0].actions[0];
        assert_eq!((raise.kind, raise.amount, raise.to), (ActionType::Raise, 300, 300));
        let call = &h.streets[0].actions[3];
        assert_eq!((call.kind, call.amount, call.to), (ActionType::Call, 200, 300));
        assert_eq!(h.uncalled[0].amount, 400);
        assert_eq!(h.total_pot, 650);
        assert_eq!(h.rake, 20);
        assert_eq!(h.collected[0].amount, 630);
        assert_eq!(h.boards, vec![vec!["7c".to_string(), "8d".into(), "2h".into()]]);
    }
}
