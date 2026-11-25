//! A No-Limit Hold'em cash-game simulator for synthetic corpora with ground truth.
//!
//! A pool of bots with known styles (nit, TAG, LAG, calling station, maniac,
//! weak-tight; parameters jittered per bot) sits down at a table, a subset per
//! hand. Cards are dealt with the deterministic `Rng`, so a seed fixes the corpus.
//!
//! Preflop play uses position-dependent ranges (top-x% of the 169 classes),
//! smoothed by a logistic so that boundary hands mix. Postflop play uses hand
//! strength (the share of random holdings beaten on the current board, raised to
//! the number of opponents, plus a crude draw bonus) against pot odds, with the
//! style's c-bet, stab, lead, bluff and calling parameters. These bots are a
//! model of player types, not strong players.
//!
//! Every decision records `truth`: the probability the bot gave each action type
//! before sampling, so the true frequency of any statistic is known exactly.

use std::collections::BTreeMap;

use pe_cards::{Card, Rng, cards_of, combos::class_of, mask_of};
use serde::{Deserialize, Serialize};

use crate::model::{Action, ActionType, Collected, Hand, Player, Post, PostKind, Shown, Street, StreetName, Uncalled, assign_positions};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimConfig {
    pub hands: usize,
    /// Seats at the table (2-9).
    pub seats: usize,
    /// Bots in the pool (at least `seats`); a random subset plays each hand.
    pub bots: usize,
    pub seed: u64,
    /// Blinds in milli-units.
    pub sb: i64,
    pub bb: i64,
    pub stack_bb: f64,
    /// Chance that the player after the big blind straddles (2 bb, live).
    pub straddle_rate: f64,
    /// Chance that the button posts a Mississippi straddle instead (2 bb, live;
    /// action starts under the gun, skips the button, and the button acts last
    /// before the flop). Tables of four or more.
    #[serde(default)]
    pub mississippi_rate: f64,
    /// Every n-th hand is a bomb pot (0 = never): everyone antes 2 bb, no preflop betting.
    pub bomb_pot_every: usize,
    /// Bomb pots use two boards, each worth half the pot.
    pub double_board_bombs: bool,
    /// Chance that an all-in before the river is run twice.
    pub run_twice_rate: f64,
    /// Rake percentage and cap in big blinds (taken only when a flop is dealt).
    pub rake_pct: f64,
    pub rake_cap_bb: f64,
    pub first_id: u64,
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig {
            hands: 1000,
            seats: 6,
            bots: 24,
            seed: 1,
            sb: 500,
            bb: 1000,
            stack_bb: 100.0,
            straddle_rate: 0.0,
            mississippi_rate: 0.0,
            bomb_pot_every: 0,
            double_board_bombs: false,
            run_twice_rate: 0.0,
            rake_pct: 0.0,
            rake_cap_bb: 3.0,
            first_id: 1,
        }
    }
}

/// Style parameters (see the module documentation for their role).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Params {
    pub rfi_early: f64,
    pub rfi_middle: f64,
    pub rfi_late: f64,
    pub rfi_sb: f64,
    pub limp: f64,
    pub three_bet: f64,
    pub call_open: f64,
    pub fold_to_three_bet: f64,
    pub four_bet: f64,
    pub cbet: [f64; 3],
    pub stab: f64,
    pub lead: f64,
    pub bluff: f64,
    pub fold_bias: f64,
    pub raise_value: f64,
    pub raise_bluff: f64,
    pub sizes: Vec<(f64, f64)>,
    pub open_bb: f64,
}

pub const STYLES: [&str; 6] = ["nit", "tag", "lag", "station", "maniac", "weak_tight"];

pub fn style_params(style: &str) -> Params {
    let p = |rfi: [f64; 4], limp, tb, co, f3, fb, cb: [f64; 3], stab, lead, bluff, fbias, rv, rb, sizes: Vec<(f64, f64)>, open| Params {
        rfi_early: rfi[0],
        rfi_middle: rfi[1],
        rfi_late: rfi[2],
        rfi_sb: rfi[3],
        limp,
        three_bet: tb,
        call_open: co,
        fold_to_three_bet: f3,
        four_bet: fb,
        cbet: cb,
        stab,
        lead,
        bluff,
        fold_bias: fbias,
        raise_value: rv,
        raise_bluff: rb,
        sizes,
        open_bb: open,
    };
    match style {
        "nit" => p([0.08, 0.11, 0.18, 0.15], 0.0, 0.03, 0.06, 0.70, 0.015, [0.55, 0.40, 0.35], 0.20, 0.02, 0.25, 0.10, 0.92, 0.01, vec![(0.5, 0.5), (0.75, 0.5)], 3.0),
        "tag" => p([0.14, 0.19, 0.38, 0.30], 0.0, 0.07, 0.10, 0.55, 0.03, [0.65, 0.50, 0.45], 0.35, 0.04, 0.60, 0.03, 0.88, 0.04, vec![(0.33, 0.4), (0.66, 0.4), (1.0, 0.2)], 2.5),
        "lag" => p([0.22, 0.28, 0.50, 0.45], 0.0, 0.12, 0.16, 0.42, 0.05, [0.78, 0.62, 0.55], 0.50, 0.08, 1.00, -0.03, 0.84, 0.08, vec![(0.33, 0.3), (0.75, 0.4), (1.25, 0.3)], 2.5),
        "station" => p([0.12, 0.15, 0.25, 0.20], 0.6, 0.025, 0.35, 0.30, 0.01, [0.45, 0.30, 0.25], 0.15, 0.10, 0.20, -0.22, 0.95, 0.01, vec![(0.5, 0.6), (1.0, 0.4)], 3.0),
        "maniac" => p([0.40, 0.45, 0.65, 0.60], 0.2, 0.20, 0.25, 0.25, 0.10, [0.90, 0.80, 0.70], 0.70, 0.20, 1.60, -0.10, 0.75, 0.15, vec![(0.75, 0.4), (1.5, 0.6)], 3.5),
        _ => p([0.07, 0.09, 0.14, 0.10], 0.4, 0.02, 0.12, 0.80, 0.01, [0.40, 0.25, 0.20], 0.10, 0.03, 0.12, 0.16, 0.95, 0.005, vec![(0.5, 1.0)], 3.0),
    }
}

fn jitter(p: &Params, rng: &mut Rng) -> Params {
    let (bluff_r, bias_r) = (rng.next_f64(), rng.next_f64());
    let mut j = |x: f64| (x * (0.85 + 0.30 * rng.next_f64())).clamp(0.0, 1.0);
    Params {
        rfi_early: j(p.rfi_early),
        rfi_middle: j(p.rfi_middle),
        rfi_late: j(p.rfi_late),
        rfi_sb: j(p.rfi_sb),
        limp: j(p.limp),
        three_bet: j(p.three_bet),
        call_open: j(p.call_open),
        fold_to_three_bet: j(p.fold_to_three_bet),
        four_bet: j(p.four_bet),
        cbet: [j(p.cbet[0]), j(p.cbet[1]), j(p.cbet[2])],
        stab: j(p.stab),
        lead: j(p.lead),
        bluff: p.bluff * (0.85 + 0.30 * bluff_r),
        fold_bias: p.fold_bias + 0.02 * (bias_r - 0.5),
        raise_value: p.raise_value,
        raise_bluff: j(p.raise_bluff),
        sizes: p.sizes.clone(),
        open_bb: p.open_bb,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BotInfo {
    pub name: String,
    pub style: String,
    pub profile: serde_json::Value,
}

pub struct SimOutput {
    pub hands: Vec<Hand>,
    pub bots: Vec<BotInfo>,
}

struct Bot {
    name: String,
    params: Params,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Top fraction of hands (by combos) at or above each class in the preflop order.
fn class_percentiles() -> Vec<f64> {
    let order = pe_cards::ranking::preflop_order();
    let mut pct = vec![0.0; pe_cards::NUM_CLASSES];
    let mut acc = 0.0;
    for c in order {
        acc += pe_cards::combos::class_size(c) as f64 / 1326.0;
        pct[c] = acc;
    }
    pct
}

/// Share of random holdings this hand beats on the current board (ties half).
fn hand_strength(hole: &[Card], board: &[Card], dead: u64) -> f64 {
    let me_mask = mask_of(hole) | mask_of(board);
    let me = pe_eval::eval_high(me_mask, pe_cards::HighRules::Standard);
    let bm = mask_of(board);
    let rest = cards_of(pe_cards::FULL_DECK & !me_mask & !dead);
    let (mut s, mut n) = (0.0, 0.0);
    for i in 0..rest.len() {
        for j in i + 1..rest.len() {
            let v = pe_eval::eval_high(bm | rest[i].mask() | rest[j].mask(), pe_cards::HighRules::Standard);
            s += if me > v { 1.0 } else if me == v { 0.5 } else { 0.0 };
            n += 1.0;
        }
    }
    s / n
}

fn draw_bonus(hole: &[Card], board: &[Card]) -> f64 {
    if board.len() >= 5 {
        return 0.0;
    }
    let all: Vec<Card> = hole.iter().chain(board).copied().collect();
    let mut bonus: f64 = 0.0;
    for suit in 0..4 {
        let n = all.iter().filter(|c| c.suit() == suit).count();
        let mine = hole.iter().any(|c| c.suit() == suit);
        if n == 4 && mine {
            bonus += if board.len() == 3 { 0.12 } else { 0.07 };
        }
    }
    // four consecutive ranks (bit 0 = ace low, bit r + 1 = rank r) using a hole card
    let ranks: u32 = all.iter().fold(0, |m, c| m | (1 << c.rank()));
    let low = (ranks << 1) | ((ranks >> 12) & 1);
    for lo in 0..=10u32 {
        let window = 0b1111u32 << lo;
        let uses_hole = hole.iter().any(|c| {
            let b = c.rank() as u32 + 1;
            (lo..lo + 4).contains(&b) || (c.rank() == 12 && lo == 0)
        });
        if low & window == window && uses_hole {
            bonus += if board.len() == 3 { 0.07 } else { 0.04 };
            break;
        }
    }
    bonus.min(0.18)
}

struct Seat {
    bot: usize,
    seat: u32,
    stack: i64,
    hole: Vec<Card>,
    folded: bool,
    total: i64,
    street: i64,
}

fn sample(rng: &mut Rng, probs: &[(ActionType, f64)]) -> ActionType {
    let w: Vec<f64> = probs.iter().map(|p| p.1).collect();
    probs[rng.weighted(&w)].0
}

fn normalise(v: &mut Vec<(ActionType, f64)>) {
    v.retain(|p| p.1 > 1e-9);
    let t: f64 = v.iter().map(|p| p.1).sum();
    for p in v.iter_mut() {
        p.1 /= t;
    }
}

pub fn simulate(cfg: &SimConfig) -> SimOutput {
    let mut rng = Rng::new(cfg.seed);
    let n_seats = cfg.seats.clamp(2, 9);
    let n_bots = cfg.bots.max(n_seats);
    let mut bots = Vec::new();
    let mut infos = Vec::new();
    for i in 0..n_bots {
        let style = STYLES[i % STYLES.len()];
        let params = jitter(&style_params(style), &mut rng);
        let name = format!("Player{:02}", i + 1);
        infos.push(BotInfo { name: name.clone(), style: style.to_string(), profile: serde_json::to_value(&params).unwrap() });
        bots.push(Bot { name, params });
    }
    let pct = class_percentiles();
    let mut hands = Vec::with_capacity(cfg.hands);
    for k in 0..cfg.hands {
        let mut order: Vec<usize> = (0..n_bots).collect();
        rng.shuffle(&mut order);
        let button_idx = k % n_seats;
        let hand = play_hand(cfg, k, &order[..n_seats], button_idx, &bots, &pct, &mut rng);
        hands.push(hand);
    }
    SimOutput { hands, bots: infos }
}

#[allow(clippy::too_many_arguments)]
fn play_hand(cfg: &SimConfig, k: usize, sitting: &[usize], button_idx: usize, bots: &[Bot], pct: &[f64], rng: &mut Rng) -> Hand {
    let n = sitting.len();
    let bb = cfg.bb;
    let stack0 = (cfg.stack_bb * bb as f64) as i64;
    let bomb = cfg.bomb_pot_every > 0 && (k + 1) % cfg.bomb_pot_every == 0;
    let double = bomb && cfg.double_board_bombs;
    let mut deck: Vec<Card> = (0..52).map(Card).collect();
    rng.shuffle(&mut deck);
    let mut seats: Vec<Seat> = (0..n)
        .map(|i| Seat { bot: sitting[i], seat: i as u32 + 1, stack: stack0, hole: vec![deck[2 * i], deck[2 * i + 1]], folded: false, total: 0, street: 0 })
        .collect();
    let mut next_card = 2 * n;
    let board_full: Vec<Card> = deck[next_card..next_card + 5].to_vec();
    next_card += 5;
    let board2_full: Vec<Card> = deck[next_card..next_card + 5].to_vec();
    next_card += 5;
    let mut hand = Hand {
        id: (cfg.first_id + k as u64).to_string(),
        site: "Synthetic".into(),
        game: "NLH".into(),
        limit: "NL".into(),
        currency: "chips".into(),
        time: format!("2026/01/01 {:02}:{:02}:{:02} ET", (k / 3600) % 24, (k / 60) % 60, k % 60),
        table: "Sim".into(),
        max_seats: n as u32,
        button: button_idx as u32 + 1,
        sb: cfg.sb,
        bb,
        bomb_pot: bomb,
        board_kind: if double { "double".into() } else { "single".into() },
        synthetic: true,
        ..Default::default()
    };
    for s in &seats {
        hand.players.push(Player {
            seat: s.seat,
            name: bots[s.bot].name.clone(),
            stack: s.stack,
            position: None,
            hole: Some(s.hole.iter().map(|c| c.to_string()).collect()),
            hero: s.seat == 1,
            sitting_out: false,
        });
    }
    assign_positions(&mut hand);
    let pos_of: BTreeMap<u32, String> = hand.players.iter().map(|p| (p.seat, p.position.clone().unwrap_or_default())).collect();
    // clockwise order from the button
    let after = |i: usize| (button_idx + i) % n;
    let post = |seats: &mut Vec<Seat>, hand: &mut Hand, idx: usize, kind: PostKind, amount: i64| {
        let a = amount.min(seats[idx].stack);
        seats[idx].stack -= a;
        seats[idx].total += a;
        if kind.is_live() {
            seats[idx].street += a;
        }
        hand.posts.push(Post { seat: seats[idx].seat, kind, amount: a });
    };
    let mut first_to_act;
    let mut current_bet;
    let mut mississippi = false;
    if bomb {
        for i in 1..=n {
            post(&mut seats, &mut hand, after(i), PostKind::Bomb, 2 * bb);
        }
        current_bet = 0;
        first_to_act = after(1);
    } else {
        let (sb_i, bb_i) = if n == 2 { (after(0), after(1)) } else { (after(1), after(2)) };
        post(&mut seats, &mut hand, sb_i, PostKind::Sb, cfg.sb);
        post(&mut seats, &mut hand, bb_i, PostKind::Bb, bb);
        current_bet = bb;
        first_to_act = if n == 2 { sb_i } else { after(3) };
        if n > 3 && cfg.straddle_rate > 0.0 && rng.next_f64() < cfg.straddle_rate {
            let st_i = after(3);
            post(&mut seats, &mut hand, st_i, PostKind::Straddle, 2 * bb);
            current_bet = 2 * bb;
            first_to_act = after(4);
        } else if n > 3 && cfg.mississippi_rate > 0.0 && rng.next_f64() < cfg.mississippi_rate {
            // the button straddles: under the gun starts, the button acts last
            post(&mut seats, &mut hand, after(0), PostKind::Mississippi, 2 * bb);
            current_bet = 2 * bb;
            first_to_act = after(3);
            mississippi = true;
        }
    }
    // the next seat to act: clockwise, except that before the flop a Mississippi
    // straddler is skipped by the cutoff and acts after the big blind
    let next_seat = |i: usize, street: StreetName| -> usize {
        if mississippi && street == StreetName::Preflop {
            if i == after(n - 1) {
                return after(1);
            }
            if i == after(2) {
                return after(0);
            }
            if i == after(0) {
                return after(3);
            }
        }
        (i + 1) % n
    };
    let mut board_len = if bomb { 3 } else { 0 };
    let mut aggressor: Option<usize> = None;
    let mut preflop_opener: Option<usize> = None;
    let streets = if bomb { vec![StreetName::Flop, StreetName::Turn, StreetName::River] } else { StreetName::ALL.to_vec() };
    for (si, &sname) in streets.iter().enumerate() {
        board_len = sname.board_len();
        if si > 0 || bomb {
            for s in seats.iter_mut() {
                s.street = 0;
            }
            current_bet = 0;
            first_to_act = after(1);
            if n == 2 {
                first_to_act = after(1); // the big blind acts first after the flop heads-up
            }
        }
        let able = seats.iter().filter(|s| !s.folded && s.stack > 0).count();
        let alive = seats.iter().filter(|s| !s.folded).count();
        if alive < 2 {
            break;
        }
        if able < 2 && seats.iter().filter(|s| !s.folded).all(|s| s.street >= current_bet || s.stack == 0) {
            continue; // nobody left to bet: deal on
        }
        let board: Vec<Card> = board_full[..board_len].to_vec();
        let mut actions: Vec<Action> = Vec::new();
        let mut need: Vec<bool> = seats.iter().map(|s| !s.folded && s.stack > 0).collect();
        let mut raises: usize = if sname == StreetName::Preflop && !bomb { 1 } else { 0 };
        let mut last_raise = if sname == StreetName::Preflop && !bomb { bb } else { 0 };
        let mut street_aggressor: Option<usize> = None;
        let mut i = first_to_act;
        let mut guard = 0;
        loop {
            guard += 1;
            if guard > 200 {
                break;
            }
            let alive = seats.iter().filter(|s| !s.folded).count();
            if alive < 2 {
                break;
            }
            if !need.iter().enumerate().any(|(j, &x)| x && !seats[j].folded && seats[j].stack > 0) {
                break;
            }
            if !need[i] || seats[i].folded || seats[i].stack == 0 {
                i = next_seat(i, sname);
                continue;
            }
            let p = &bots[seats[i].bot].params;
            let to_call = (current_bet - seats[i].street).min(seats[i].stack);
            let pot: i64 = seats.iter().map(|s| s.total).sum();
            let opponents = seats.iter().enumerate().filter(|(j, s)| *j != i && !s.folded).count();
            let can_raise = raises < 4 && seats[i].stack > to_call && seats.iter().enumerate().any(|(j, s)| j != i && !s.folded && s.stack > 0);
            let mut probs: Vec<(ActionType, f64)>;
            if sname == StreetName::Preflop {
                let class = class_of(seats[i].hole[0], seats[i].hole[1]);
                let x = pct[class];
                let w = 0.015;
                let pos = pos_of[&seats[i].seat].as_str();
                let rfi = match pos {
                    "BTN" | "CO" => p.rfi_late,
                    "SB" => p.rfi_sb,
                    "HJ" | "LJ" => p.rfi_middle,
                    _ => p.rfi_early,
                };
                let is_bb_option = to_call == 0;
                probs = if raises <= 1 && preflop_opener.is_none() {
                    // unopened (blinds and limps only)
                    let open = sigmoid((rfi - x) / w);
                    if is_bb_option {
                        let iso = sigmoid((p.three_bet * 1.5 - x) / w);
                        vec![(ActionType::Check, 1.0 - iso), (ActionType::Raise, iso)]
                    } else {
                        let limp = p.limp * sigmoid((rfi + 0.15 - x) / w) * (1.0 - open);
                        vec![(ActionType::Fold, (1.0 - open - limp).max(0.0)), (ActionType::Call, limp), (ActionType::Raise, open)]
                    }
                } else if raises == 2 {
                    // facing an open
                    let tb = sigmoid((p.three_bet - x) / w);
                    let bb_bonus = if pos == "BB" { 0.12 } else { 0.0 };
                    let cont = sigmoid((p.three_bet + p.call_open + bb_bonus - x) / w).max(tb);
                    vec![(ActionType::Fold, 1.0 - cont), (ActionType::Call, cont - tb), (ActionType::Raise, tb)]
                } else if raises == 3 && preflop_opener == Some(i) {
                    let cont = sigmoid((rfi * (1.0 - p.fold_to_three_bet) - x) / w);
                    let fb = sigmoid((p.four_bet - x) / w).min(cont);
                    vec![(ActionType::Fold, 1.0 - cont), (ActionType::Call, cont - fb), (ActionType::Raise, fb)]
                } else {
                    let jam = sigmoid((0.02 - x) / w);
                    let call = sigmoid((0.035 - x) / w).max(jam);
                    vec![(ActionType::Fold, 1.0 - call), (ActionType::Call, call - jam), (ActionType::Raise, jam)]
                };
            } else {
                let dead = 0u64;
                let hs = hand_strength(&seats[i].hole, &board, dead);
                let s_eff = (hs.powi(opponents.max(1) as i32) + draw_bonus(&seats[i].hole, &board)).min(1.0);
                let street_idx = sname.index() - 1;
                if to_call == 0 {
                    let base = if aggressor == Some(i) {
                        p.cbet[street_idx]
                    } else if aggressor.is_some_and(|a| !seats[a].folded) && actions.is_empty() {
                        p.lead
                    } else {
                        p.stab
                    };
                    let bet = if s_eff >= 0.75 {
                        (base + 0.35).min(0.95)
                    } else if s_eff >= 0.5 {
                        base * 0.7
                    } else if s_eff >= 0.3 {
                        base * 0.4
                    } else {
                        (base * p.bluff).min(0.9)
                    }
                    .clamp(0.01, 0.97);
                    probs = vec![(ActionType::Check, 1.0 - bet), (ActionType::Bet, bet)];
                } else {
                    let po = to_call as f64 / (pot + to_call) as f64;
                    let threshold = po + p.fold_bias + 0.05 * (raises.saturating_sub(1)) as f64;
                    let cont = sigmoid((s_eff - threshold) / 0.07);
                    let raise = if s_eff >= p.raise_value {
                        0.6 * cont
                    } else if s_eff < 0.25 {
                        p.raise_bluff
                    } else {
                        0.02 * cont
                    };
                    let call = (cont - if s_eff >= p.raise_value { raise } else { 0.0 }).max(0.0);
                    probs = vec![(ActionType::Fold, (1.0 - call - raise).max(0.0)), (ActionType::Call, call), (ActionType::Raise, raise)];
                }
            }
            if !can_raise {
                let agg: f64 = probs.iter().filter(|p| p.0.is_aggressive()).map(|p| p.1).sum();
                probs.retain(|p| !p.0.is_aggressive());
                if let Some(c) = probs.iter_mut().find(|p| p.0 == ActionType::Call || p.0 == ActionType::Check) {
                    c.1 += agg;
                } else {
                    probs.push((ActionType::Call, agg));
                }
            }
            if to_call == 0 {
                // folding is never offered when checking is free
                let f: f64 = probs.iter().filter(|p| p.0 == ActionType::Fold).map(|p| p.1).sum();
                probs.retain(|p| p.0 != ActionType::Fold);
                if let Some(c) = probs.iter_mut().find(|p| p.0 == ActionType::Check) {
                    c.1 += f;
                } else {
                    probs.push((ActionType::Check, f));
                }
                for p in probs.iter_mut() {
                    if p.0 == ActionType::Call {
                        p.0 = ActionType::Check;
                    }
                }
            }
            let mut merged: Vec<(ActionType, f64)> = Vec::new();
            for (a, w) in probs {
                match merged.iter_mut().find(|m| m.0 == a) {
                    Some(m) => m.1 += w,
                    None => merged.push((a, w)),
                }
            }
            let mut probs = merged;
            normalise(&mut probs);
            let choice = sample(rng, &probs);
            let truth: BTreeMap<String, f64> = probs.iter().map(|(a, w)| (a.as_str().to_string(), *w)).collect();
            let s = &mut seats[i];
            let (kind, amount) = match choice {
                ActionType::Fold => {
                    s.folded = true;
                    (ActionType::Fold, 0)
                }
                ActionType::Check => (ActionType::Check, 0),
                ActionType::Call => {
                    let a = to_call.min(s.stack);
                    (ActionType::Call, a)
                }
                ActionType::Bet | ActionType::Raise => {
                    let target = if sname == StreetName::Preflop {
                        if raises <= 1 {
                            let limpers = 0;
                            ((p.open_bb + limpers as f64) * bb as f64) as i64
                        } else if raises == 2 {
                            current_bet * 3
                        } else if raises == 3 {
                            (current_bet as f64 * 2.3) as i64
                        } else {
                            s.street + s.stack
                        }
                    } else if current_bet == 0 {
                        let w: Vec<f64> = p.sizes.iter().map(|x| x.1).collect();
                        let size = p.sizes[rng.weighted(&w)].0;
                        (size * pot as f64) as i64
                    } else {
                        current_bet * 3
                    };
                    let min_to = current_bet + last_raise.max(bb);
                    let mut to = target.max(min_to);
                    let max_to = s.street + s.stack;
                    if to as f64 >= 0.8 * max_to as f64 {
                        to = max_to;
                    }
                    let to = to.min(max_to);
                    let kind = if current_bet == 0 && !(sname == StreetName::Preflop && !bomb) { ActionType::Bet } else { ActionType::Raise };
                    (kind, to - s.street)
                }
            };
            let s = &mut seats[i];
            s.stack -= amount;
            s.street += amount;
            s.total += amount;
            let all_in = s.stack == 0 && amount > 0;
            if kind.is_aggressive() {
                let inc = s.street - current_bet;
                if inc > 0 {
                    last_raise = last_raise.max(inc);
                }
                current_bet = current_bet.max(s.street);
                raises += 1;
                street_aggressor = Some(i);
                if sname == StreetName::Preflop && preflop_opener.is_none() {
                    preflop_opener = Some(i);
                }
                for (j, x) in need.iter_mut().enumerate() {
                    *x = j != i;
                }
            }
            need[i] = false;
            actions.push(Action { seat: seats[i].seat, kind, amount, to: seats[i].street, all_in, truth: Some(truth) });
            i = next_seat(i, sname);
        }
        if !actions.is_empty() {
            hand.streets.push(Street { name: sname, actions });
        }
        // an uncalled bet: the top street contribution beyond the second highest goes back
        let mut by_street: Vec<(usize, i64)> = seats.iter().enumerate().map(|(i, s)| (i, s.street)).collect();
        by_street.sort_by_key(|x| -x.1);
        if by_street.len() >= 2 && by_street[0].1 > by_street[1].1 {
            let (i, top) = by_street[0];
            let excess = top - by_street[1].1;
            seats[i].stack += excess;
            seats[i].total -= excess;
            seats[i].street -= excess;
            hand.uncalled.push(Uncalled { seat: seats[i].seat, amount: excess });
        }
        if street_aggressor.is_some() {
            aggressor = street_aggressor;
        } else if sname != StreetName::Preflop {
            // a street checked through hands the initiative to nobody
            aggressor = None;
        }
    }
    // boards dealt: all five when two or more players remain, else what was reached
    let alive: Vec<usize> = (0..n).filter(|&i| !seats[i].folded).collect();
    let dealt_len = if alive.len() >= 2 { 5 } else { board_len };
    let run_twice = !double && alive.len() >= 2 && hand.streets.last().is_some_and(|s| s.name != StreetName::River) && rng.next_f64() < cfg.run_twice_rate;
    let second_run: Vec<Card> = if run_twice {
        let keep = hand.streets.last().map_or(0, |s| s.name.board_len());
        let mut b = board_full[..keep].to_vec();
        b.extend_from_slice(&deck[next_card..next_card + 5 - keep]);
        b
    } else {
        Vec::new()
    };
    let to_strings = |b: &[Card]| -> Vec<String> { b.iter().map(|c| c.to_string()).collect() };
    hand.boards = if double {
        vec![to_strings(&board_full[..dealt_len]), to_strings(&board2_full[..dealt_len])]
    } else if run_twice {
        hand.board_kind = "run_twice".into();
        vec![to_strings(&board_full), to_strings(&second_run)]
    } else {
        vec![to_strings(&board_full[..dealt_len])]
    };
    // pots and payouts
    let contrib: BTreeMap<u32, i64> = seats.iter().map(|s| (s.seat, s.total)).collect();
    let folded: std::collections::BTreeSet<u32> = seats.iter().filter(|s| s.folded).map(|s| s.seat).collect();
    let mut pots = crate::reconstruct::side_pots(&contrib, &folded);
    let total: i64 = contrib.values().sum();
    let flop_seen = bomb || hand.streets.iter().any(|s| s.name != StreetName::Preflop) || (alive.len() >= 2 && dealt_len >= 3);
    let rake = if cfg.rake_pct > 0.0 && flop_seen {
        (((total as f64) * cfg.rake_pct / 100.0) as i64).min((cfg.rake_cap_bb * bb as f64) as i64)
    } else {
        0
    };
    if rake > 0 {
        pots[0].amount -= rake;
    }
    hand.total_pot = total;
    hand.rake = rake;
    let seat_idx: BTreeMap<u32, usize> = seats.iter().enumerate().map(|(i, s)| (s.seat, i)).collect();
    let boards_for_payout: Vec<Vec<Card>> = if double {
        vec![board_full.clone(), board2_full.clone()]
    } else if run_twice {
        vec![board_full.clone(), second_run.clone()]
    } else {
        vec![board_full.clone()]
    };
    let split = boards_for_payout.len();
    // clockwise order from the button, for odd chips
    let clockwise: Vec<u32> = (1..=n).map(|i| seats[after(i)].seat).collect();
    for (pi, pot) in pots.iter().enumerate() {
        let elig: Vec<u32> = pot.eligible.clone();
        if elig.len() == 1 || alive.len() == 1 {
            let w = if alive.len() == 1 { seats[alive[0]].seat } else { elig[0] };
            hand.collected.push(Collected { seat: w, amount: pot.amount, pot: pi as u32, board: None });
            continue;
        }
        let parts: Vec<i64> = if split == 2 { vec![pot.amount / 2 + pot.amount % 2, pot.amount / 2] } else { vec![pot.amount] };
        for (bi, part) in parts.iter().enumerate() {
            let b = &boards_for_payout[bi];
            let vals: Vec<(u32, u32)> = elig.iter().map(|s| (*s, pe_eval::eval_high(mask_of(&seats[seat_idx[s]].hole) | mask_of(b), pe_cards::HighRules::Standard))).collect();
            let best = vals.iter().map(|v| v.1).max().unwrap();
            let mut winners: Vec<u32> = vals.iter().filter(|v| v.1 == best).map(|v| v.0).collect();
            winners.sort_by_key(|w| clockwise.iter().position(|c| c == w).unwrap());
            let share = part / winners.len() as i64;
            let mut odd = part - share * winners.len() as i64;
            for w in winners {
                let extra = if odd > 0 { 1 } else { 0 };
                odd -= extra;
                let board = if split == 2 { Some(bi as u32 + 1) } else { None };
                match hand.collected.iter_mut().find(|c| c.seat == w && c.pot == pi as u32 && c.board == board) {
                    Some(c) => c.amount += share + extra,
                    None => hand.collected.push(Collected { seat: w, amount: share + extra, pot: pi as u32, board }),
                }
            }
        }
    }
    if alive.len() >= 2 {
        let board = &boards_for_payout[0];
        for &i in &alive {
            let v = pe_eval::eval_high(mask_of(&seats[i].hole) | mask_of(board), pe_cards::HighRules::Standard);
            hand.shown.push(Shown {
                seat: seats[i].seat,
                cards: seats[i].hole.iter().map(|c| c.to_string()).collect(),
                description: if split == 1 { pe_eval::describe(v, pe_cards::HighRules::Standard) } else { String::new() },
            });
        }
    }
    hand
}
