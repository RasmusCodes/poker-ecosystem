//! The `pe-hh/1` JSON model (see `docs/HAND-HISTORY-SCHEMA.md`).
//!
//! Amounts are integer thousandths of the hand's unit ("milli"), cards are
//! two-character strings (`"Ah"`), seats are the site's 1-based seat numbers.
//! Maps keyed by seat serialise with string keys (`{"3": 1500}`), as JSON
//! requires. Readers ignore unknown fields; optional fields default.

use std::collections::BTreeMap;

use pe_cards::{Card, Game};
use serde::{Deserialize, Serialize};

pub const SCHEMA: &str = "pe-hh/1";

/// A betting round.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StreetName {
    #[default]
    Preflop,
    Flop,
    Turn,
    River,
}

impl StreetName {
    pub const ALL: [StreetName; 4] = [StreetName::Preflop, StreetName::Flop, StreetName::Turn, StreetName::River];

    pub fn as_str(self) -> &'static str {
        match self {
            StreetName::Preflop => "preflop",
            StreetName::Flop => "flop",
            StreetName::Turn => "turn",
            StreetName::River => "river",
        }
    }

    /// Board cards visible while this street is played (0, 3, 4, 5).
    pub fn board_len(self) -> usize {
        match self {
            StreetName::Preflop => 0,
            StreetName::Flop => 3,
            StreetName::Turn => 4,
            StreetName::River => 5,
        }
    }

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn from_index(i: usize) -> Option<StreetName> {
        StreetName::ALL.get(i).copied()
    }

    /// The street on which a board of `len` cards was completed (0 → preflop).
    pub fn from_board_len(len: usize) -> StreetName {
        match len {
            0..=2 => StreetName::Preflop,
            3 => StreetName::Flop,
            4 => StreetName::Turn,
            _ => StreetName::River,
        }
    }
}

/// Action types of the schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ActionType {
    Fold,
    Check,
    Call,
    Bet,
    Raise,
}

impl ActionType {
    pub const ALL: [ActionType; 5] = [ActionType::Fold, ActionType::Check, ActionType::Call, ActionType::Bet, ActionType::Raise];

    pub fn as_str(self) -> &'static str {
        match self {
            ActionType::Fold => "fold",
            ActionType::Check => "check",
            ActionType::Call => "call",
            ActionType::Bet => "bet",
            ActionType::Raise => "raise",
        }
    }

    pub fn parse(s: &str) -> Option<ActionType> {
        ActionType::ALL.into_iter().find(|a| a.as_str() == s)
    }

    pub fn is_aggressive(self) -> bool {
        matches!(self, ActionType::Bet | ActionType::Raise)
    }
}

/// Forced bets. Blinds and straddles are live; antes, dead blinds and bomb antes are dead money.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PostKind {
    Sb,
    Bb,
    Ante,
    Straddle,
    Mississippi,
    Dead,
    Bomb,
}

impl PostKind {
    pub fn is_live(self) -> bool {
        matches!(self, PostKind::Sb | PostKind::Bb | PostKind::Straddle | PostKind::Mississippi)
    }
    pub fn as_str(self) -> &'static str {
        match self {
            PostKind::Sb => "sb",
            PostKind::Bb => "bb",
            PostKind::Ante => "ante",
            PostKind::Straddle => "straddle",
            PostKind::Mississippi => "mississippi",
            PostKind::Dead => "dead",
            PostKind::Bomb => "bomb",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Hand {
    pub schema: String,
    pub id: String,
    pub site: String,
    pub game: String,
    pub limit: String,
    pub currency: String,
    pub time: String,
    pub table: String,
    pub max_seats: u32,
    pub button: u32,
    pub sb: i64,
    pub bb: i64,
    pub ante: i64,
    pub bomb_pot: bool,
    pub board_kind: String,
    pub players: Vec<Player>,
    pub posts: Vec<Post>,
    pub streets: Vec<Street>,
    pub boards: Vec<Vec<String>>,
    pub shown: Vec<Shown>,
    pub collected: Vec<Collected>,
    pub uncalled: Vec<Uncalled>,
    pub total_pot: i64,
    pub rake: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub derived: Option<Derived>,
    pub synthetic: bool,
}

impl Default for Hand {
    fn default() -> Self {
        Hand {
            schema: SCHEMA.to_string(),
            id: String::new(),
            site: String::new(),
            game: "NLH".to_string(),
            limit: "NL".to_string(),
            currency: "chips".to_string(),
            time: String::new(),
            table: String::new(),
            max_seats: 0,
            button: 0,
            sb: 0,
            bb: 0,
            ante: 0,
            bomb_pot: false,
            board_kind: "single".to_string(),
            players: Vec::new(),
            posts: Vec::new(),
            streets: Vec::new(),
            boards: Vec::new(),
            shown: Vec::new(),
            collected: Vec::new(),
            uncalled: Vec::new(),
            total_pot: 0,
            rake: 0,
            derived: None,
            synthetic: false,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Player {
    pub seat: u32,
    pub name: String,
    pub stack: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub position: Option<String>,
    pub hole: Option<Vec<String>>,
    pub hero: bool,
    pub sitting_out: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Post {
    pub seat: u32,
    pub kind: PostKind,
    #[serde(default)]
    pub amount: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Street {
    pub name: StreetName,
    #[serde(default)]
    pub actions: Vec<Action>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Action {
    pub seat: u32,
    #[serde(rename = "type")]
    pub kind: ActionType,
    #[serde(default)]
    pub amount: i64,
    #[serde(default)]
    pub to: i64,
    #[serde(default)]
    pub all_in: bool,
    /// Synthetic hands: the bot's probability of each action type it considered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truth: Option<BTreeMap<String, f64>>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Shown {
    pub seat: u32,
    pub cards: Vec<String>,
    pub description: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Collected {
    pub seat: u32,
    pub amount: i64,
    /// 0 = main pot, 1.. = side pots.
    pub pot: u32,
    /// 1-based board when the pot was split by board (run it twice, double board).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub board: Option<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Uncalled {
    pub seat: u32,
    pub amount: i64,
}

/// Reconstruction output (see [`crate::reconstruct()`] and [`crate::annotate_allin()`]).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Derived {
    pub ok: bool,
    pub issues: Vec<String>,
    pub contrib: BTreeMap<u32, i64>,
    pub net: BTreeMap<u32, i64>,
    pub pots: Vec<Pot>,
    pub players_dealt: u32,
    pub preflop_aggressor: Option<u32>,
    pub showdown_seats: Vec<u32>,
    pub allin: Option<AllIn>,
    pub river_strength: BTreeMap<u32, f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Pot {
    pub amount: i64,
    pub eligible: Vec<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AllIn {
    /// The street whose betting closed the action (board cards known: 0, 3 or 4).
    pub street: StreetName,
    pub seats: Vec<u32>,
    /// Each seat's share of the main pot (all seats are eligible for it).
    pub equity: BTreeMap<u32, f64>,
    /// Expected collection over all contested pots (after rake) minus contribution.
    pub ev_net: BTreeMap<u32, i64>,
}

// ------------------------------------------------------------------ helpers

pub fn cards_to_strings(cards: &[Card]) -> Vec<String> {
    cards.iter().map(|c| c.to_string()).collect()
}

/// Parses card strings; `None` if any is malformed.
pub fn strings_to_cards(cards: &[String]) -> Option<Vec<Card>> {
    cards.iter().map(|s| Card::parse(s).ok()).collect()
}

impl Hand {
    /// One compact JSON line (the `.jsonl` format).
    pub fn to_json_line(&self) -> String {
        serde_json::to_string(self).expect("a hand always serialises")
    }

    pub fn from_json(text: &str) -> Result<Hand, String> {
        serde_json::from_str(text).map_err(|e| format!("invalid hand JSON: {e}"))
    }

    /// The evaluator's game definition (`None` for unknown codes).
    pub fn game_def(&self) -> Option<Game> {
        Game::parse(&self.game).ok()
    }

    pub fn player(&self, seat: u32) -> Option<&Player> {
        self.players.iter().find(|p| p.seat == seat)
    }

    pub fn name_of(&self, seat: u32) -> &str {
        self.player(seat).map_or("", |p| p.name.as_str())
    }

    /// Seats dealt in (not sitting out), ascending.
    pub fn dealt_seats(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.players.iter().filter(|p| !p.sitting_out).map(|p| p.seat).collect();
        v.sort_unstable();
        v
    }

    /// Hole cards of a seat from `players[].hole`, else from `shown`.
    pub fn hole_cards(&self, seat: u32) -> Option<Vec<Card>> {
        if let Some(h) = self.player(seat).and_then(|p| p.hole.as_ref()) {
            if let Some(c) = strings_to_cards(h) {
                if !c.is_empty() {
                    return Some(c);
                }
            }
        }
        self.shown.iter().find(|s| s.seat == seat).and_then(|s| strings_to_cards(&s.cards)).filter(|c| !c.is_empty())
    }

    /// Board `i` as cards (empty when absent or malformed).
    pub fn board_cards(&self, i: usize) -> Vec<Card> {
        self.boards.get(i).and_then(|b| strings_to_cards(b)).unwrap_or_default()
    }

    pub fn street(&self, name: StreetName) -> Option<&Street> {
        self.streets.iter().find(|s| s.name == name)
    }
}

// ------------------------------------------------------------------ readers and writers

/// Reads a JSON array of hands, a single hand object, or JSON Lines (auto-detected).
pub fn read_hands(text: &str) -> Result<Vec<Hand>, String> {
    let t = text.trim_start_matches('\u{feff}').trim();
    if t.is_empty() {
        return Ok(Vec::new());
    }
    if t.starts_with('[') {
        return read_json_array(t);
    }
    match read_jsonl(t) {
        Ok(v) => Ok(v),
        Err(e) => serde_json::from_str::<Hand>(t).map(|h| vec![h]).map_err(|_| e),
    }
}

/// Reads a `.json` file's content: an array of hands (or one hand object).
pub fn read_json_array(text: &str) -> Result<Vec<Hand>, String> {
    let t = text.trim_start_matches('\u{feff}').trim();
    if t.starts_with('{') {
        return Hand::from_json(t).map(|h| vec![h]);
    }
    serde_json::from_str(t).map_err(|e| format!("invalid JSON array of hands: {e}"))
}

/// Reads JSON Lines: one hand per non-empty line.
pub fn read_jsonl(text: &str) -> Result<Vec<Hand>, String> {
    let mut out = Vec::new();
    for (i, line) in text.trim_start_matches('\u{feff}').lines().enumerate() {
        let l = line.trim();
        if l.is_empty() {
            continue;
        }
        out.push(serde_json::from_str(l).map_err(|e| format!("line {}: invalid hand JSON: {e}", i + 1))?);
    }
    Ok(out)
}

/// Reads a `.json` or `.jsonl` file.
pub fn read_file(path: &std::path::Path) -> Result<Vec<Hand>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    read_hands(&text)
}

/// JSON Lines text, one hand per line (trailing newline included).
pub fn to_jsonl(hands: &[Hand]) -> String {
    let mut s = String::new();
    for h in hands {
        s.push_str(&h.to_json_line());
        s.push('\n');
    }
    s
}

/// A JSON array of hands.
pub fn to_json_array(hands: &[Hand]) -> String {
    serde_json::to_string(hands).expect("hands always serialise")
}

// ------------------------------------------------------------------ positions

/// Labels of the `k = n - 3` seats before the button, in preflop order (schema table).
const PRE_BUTTON: [&[&str]; 8] = [
    &[],
    &["CO"],
    &["HJ", "CO"],
    &["UTG", "HJ", "CO"],
    &["UTG", "LJ", "HJ", "CO"],
    &["UTG", "UTG1", "LJ", "HJ", "CO"],
    &["UTG", "UTG1", "UTG2", "LJ", "HJ", "CO"],
    &["UTG", "UTG1", "UTG2", "UTG3", "LJ", "HJ", "CO"],
];

/// Position labels for `n` players dealt in, clockwise from the button:
/// `BTN, SB, BB, <pre-button labels in preflop order>`. Heads-up: `BTN, BB`.
/// Defined for 1-10 players; extra seats beyond ten are labelled `MP`.
pub fn position_labels(n: usize) -> Vec<&'static str> {
    match n {
        0 => vec![],
        1 => vec!["BTN"],
        2 => vec!["BTN", "BB"],
        _ => {
            let k = n - 3;
            let mut v = vec!["BTN", "SB", "BB"];
            if k < PRE_BUTTON.len() {
                v.extend_from_slice(PRE_BUTTON[k]);
            } else {
                v.push("UTG");
                v.extend(std::iter::repeat_n("MP", k - 7));
                v.extend_from_slice(&PRE_BUTTON[7][1..]);
            }
            v
        }
    }
}

/// Dealt seats in clockwise order starting at the button. With a dead button
/// (button seat not dealt) the last dealt seat at or before it acts as button.
pub fn clockwise_from_button(seats: &[u32], button: u32) -> Vec<u32> {
    let mut order: Vec<u32> = seats.to_vec();
    order.sort_unstable();
    order.dedup();
    if order.is_empty() {
        return order;
    }
    let btn = order.iter().copied().filter(|&s| s <= button).max().unwrap_or(*order.last().unwrap());
    let i = order.iter().position(|&s| s == btn).unwrap();
    order.rotate_left(i);
    order
}

/// Seat → position label for the given dealt seats and button.
pub fn seat_positions(seats: &[u32], button: u32) -> BTreeMap<u32, &'static str> {
    let order = clockwise_from_button(seats, button);
    let labels = position_labels(order.len());
    order.into_iter().zip(labels).collect()
}

/// Sets `players[].position` (dealt players only; sitting-out players get none).
pub fn assign_positions(hand: &mut Hand) {
    let pos = seat_positions(&hand.dealt_seats(), hand.button);
    for p in &mut hand.players {
        p.position = if p.sitting_out { None } else { pos.get(&p.seat).map(|s| s.to_string()) };
    }
}

/// Tells ordinary straddles from Mississippi straddles. Hand-history text writes
/// both as `posts straddle`; the seat decides. A straddle posted by the seat after
/// the previous live blind (big blind → under the gun → the next seat for a
/// re-straddle) is ordinary; one from any other seat, usually the button, is a
/// Mississippi straddle, whose poster acts last before the flop.
pub fn classify_straddles(hand: &mut Hand) {
    let order = clockwise_from_button(&hand.dealt_seats(), hand.button);
    let n = order.len();
    let Some(bb) = hand.posts.iter().find(|p| p.kind == PostKind::Bb).map(|p| p.seat) else { return };
    let Some(mut prev) = order.iter().position(|&s| s == bb) else { return };
    for p in hand.posts.iter_mut() {
        if matches!(p.kind, PostKind::Straddle | PostKind::Mississippi) {
            let next = (prev + 1) % n;
            if order[next] == p.seat {
                p.kind = PostKind::Straddle;
                prev = next;
            } else {
                p.kind = PostKind::Mississippi;
            }
        }
    }
}

/// Seats in preflop acting order for the first round. Normally clockwise from the
/// seat after the last live blind or straddle; a Mississippi straddler is moved to
/// act last, after the blinds. Bomb pots have no preflop round (empty).
pub fn preflop_order(hand: &Hand) -> Vec<u32> {
    if hand.bomb_pot {
        return Vec::new();
    }
    let order = clockwise_from_button(&hand.dealt_seats(), hand.button);
    let n = order.len();
    if n < 2 {
        return order;
    }
    // heads-up the button posts the small blind and acts first
    let mut last = if n == 2 { 0 } else { 2 % n };
    if let Some(bb) = hand.posts.iter().find(|p| p.kind == PostKind::Bb) {
        if let Some(i) = order.iter().position(|&s| s == bb.seat) {
            last = i;
        }
    }
    for p in &hand.posts {
        if p.kind == PostKind::Straddle {
            if let Some(i) = order.iter().position(|&s| s == p.seat) {
                last = i;
            }
        }
    }
    let mut out: Vec<u32> = (1..=n).map(|k| order[(last + k) % n]).collect();
    if n == 2 {
        out = vec![order[0], order[1]];
    }
    if let Some(m) = hand.posts.iter().find(|p| p.kind == PostKind::Mississippi) {
        if let Some(i) = out.iter().position(|&s| s == m.seat) {
            let s = out.remove(i);
            out.push(s);
        }
    }
    out
}
