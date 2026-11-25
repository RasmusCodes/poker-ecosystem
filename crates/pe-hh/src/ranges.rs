//! Street-by-street ranges: what the public actions of a hold'em hand say about
//! each player's hole cards.
//!
//! The reconstruction is Bayes' rule with an explicit model of play:
//!
//! 1. **Prior.** Every player starts with the 1,326 combos equally likely; board
//!    cards remove the combos they block as they are dealt.
//! 2. **Frequencies.** Each decision has a context: before the flop, opening
//!    (by position group), the big blind's or straddler's option, facing an
//!    open, facing a 3-bet; after the flop, a c-bet spot (the player made the
//!    last bet or raise of the previous street), a lead into that player, any
//!    other first bet, facing a bet, facing a raise. For every player and
//!    context the table counts how often they folded, checked or called, and bet
//!    or raised, across all the hands given. The estimate shrinks those counts
//!    towards the population's (Dirichlet-multinomial posterior mean with
//!    `player_prior` pseudo-decisions), and the population towards a fixed
//!    default (`population_prior`).
//! 3. **Likelihood.** A *threshold strategy* turns the frequencies into a
//!    strategy for every combo: the combos in the current range are ordered by
//!    strength and cut into consecutive bands by weight. Before the flop the
//!    weakest fold, the middle calls and the strongest raise. After the flop,
//!    bets and raises are polarised: value from the top and bluffs from the
//!    bottom, with bluff shares of 28% for bets and 33% for raises (the shares
//!    that make a call break even against a two-thirds-pot bet and a pot-sized
//!    raise); facing a bet, the weakest of the rest fold and the middle calls.
//!    Combos of equal strength are treated alike.
//! 4. **Noise.** Players do not sort their ranges perfectly: they slow-play,
//!    float and give up. With probability `noise` an action is drawn from the
//!    player's frequencies regardless of the cards, and with a tiny
//!    probability `tremble` uniformly, so an action shifts the range rather than
//!    cutting it. `noise` is fitted on simulated hands whose cards are known
//!    (`pe bench ranges`).
//!
//! Strength before the flop is the preflop equity order of the 169 classes. On
//! the flop and turn it is the combo's exact equity against one random hand over
//! every runout, which counts draws; on the river, the share of random hands it
//! beats. Both use exact card removal.
//!
//! Every quantity here is deterministic and computed in a fixed order, so the
//! browser port (`web/js/engine/ranges.js`) reproduces it to the last bit.

use std::collections::{BTreeMap, BTreeSet};

use pe_cards::{Card, CardMask, NUM_CLASSES, NUM_COMBOS, cards_to_string, class_of, combo_cards, combo_mask, mask_of};
use serde::{Deserialize, Serialize};

use crate::model::{ActionType, Hand, StreetName, seat_positions};

/// Action categories used by the model: fold, check or call, bet or raise.
pub const CATEGORIES: [&str; 3] = ["fold", "check/call", "bet/raise"];

/// Bluff share of a polarised bet range: s / (1 + 2s) for a two-thirds-pot bet.
pub fn bet_bluff_share() -> f64 {
    0.66 / 2.32
}

/// Bluff share of a polarised raising range: s / (1 + 2s) for a pot-sized raise.
pub fn raise_bluff_share() -> f64 {
    1.0 / 3.0
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RangeOptions {
    /// Probability that an action follows the player's frequencies regardless of the cards.
    pub noise: f64,
    /// Probability that an action is a uniformly random choice among the available ones.
    pub tremble: f64,
    /// Pseudo-decisions of the population prior in each player's estimate.
    pub player_prior: f64,
    /// Pseudo-decisions of the default frequencies in the population estimate.
    pub population_prior: f64,
}

impl Default for RangeOptions {
    fn default() -> Self {
        // noise fitted by `pe bench ranges` (log score on held-out simulated hands)
        RangeOptions { noise: 0.5, tremble: 0.001, player_prior: 10.0, population_prior: 20.0 }
    }
}

// ------------------------------------------------------------------ decisions

/// One voluntary action with its public context.
#[derive(Clone, Debug, Serialize)]
pub struct Decision {
    /// Position of the action in the hand (streets concatenated).
    pub index: usize,
    pub street: StreetName,
    pub seat: u32,
    pub action: ActionType,
    /// `<street>_<spot>`, e.g. `preflop_vs_open` or `river_vs_bet`.
    pub context: String,
    /// 0 fold, 1 check or call, 2 bet or raise.
    pub category: usize,
    /// Which categories the player could choose.
    pub available: [bool; 3],
    /// Chips in the pot before the action and the amount to call (milli-units).
    pub pot_before: i64,
    pub to_call: i64,
    pub stack_before: i64,
}

pub fn category_of(a: ActionType) -> usize {
    match a {
        ActionType::Fold => 0,
        ActionType::Check | ActionType::Call => 1,
        ActionType::Bet | ActionType::Raise => 2,
    }
}

/// Replays a hand and labels every action with its context.
pub fn decisions(hand: &Hand) -> Vec<Decision> {
    let dealt = hand.dealt_seats();
    let pos = seat_positions(&dealt, hand.button);
    let mut stack: BTreeMap<u32, i64> = dealt.iter().map(|&s| (s, hand.player(s).map_or(0, |p| p.stack))).collect();
    let mut street_in: BTreeMap<u32, i64> = dealt.iter().map(|&s| (s, 0)).collect();
    let mut folded: BTreeSet<u32> = BTreeSet::new();
    let mut pot = 0i64;
    for p in &hand.posts {
        *stack.entry(p.seat).or_insert(0) -= p.amount;
        pot += p.amount;
        if p.kind.is_live() {
            *street_in.entry(p.seat).or_insert(0) += p.amount;
        }
    }
    let mut out = Vec::new();
    let mut initiative: Option<u32> = None;
    let mut index = 0;
    for st in &hand.streets {
        if st.name != StreetName::Preflop {
            for v in street_in.values_mut() {
                *v = 0;
            }
        }
        let mut aggressions = 0usize;
        let mut last_aggressor: Option<u32> = None;
        for (acted_this_street, a) in st.actions.iter().enumerate() {
            let max_in = street_in.values().copied().max().unwrap_or(0);
            let mine = street_in.get(&a.seat).copied().unwrap_or(0);
            let to_call = (max_in - mine).max(0);
            let my_stack = stack.get(&a.seat).copied().unwrap_or(0);
            let others = dealt.iter().any(|&s| s != a.seat && !folded.contains(&s) && stack.get(&s).copied().unwrap_or(0) > 0);
            let mut available = [to_call > 0, true, my_stack > to_call && others];
            let category = category_of(a.kind);
            available[category] = true;
            let spot = if st.name == StreetName::Preflop {
                match aggressions {
                    0 if to_call > 0 => match pos.get(&a.seat).copied() {
                        Some("SB") | Some("BB") => "open_blind",
                        Some("BTN") | Some("CO") => "open_late",
                        _ => "open_early",
                    },
                    0 => "option",
                    1 => "vs_open",
                    _ => "vs_3bet",
                }
            } else {
                let leading = initiative.is_some_and(|i| i != a.seat && !folded.contains(&i)) && acted_this_street == 0;
                match aggressions {
                    0 if initiative == Some(a.seat) => "cbet",
                    0 if leading => "lead",
                    0 => "stab",
                    1 => "vs_bet",
                    _ => "vs_raise",
                }
            };
            out.push(Decision {
                index,
                street: st.name,
                seat: a.seat,
                action: a.kind,
                context: format!("{}_{spot}", st.name.as_str()),
                category,
                available,
                pot_before: pot,
                to_call,
                stack_before: my_stack,
            });
            *stack.entry(a.seat).or_insert(0) -= a.amount;
            *street_in.entry(a.seat).or_insert(0) += a.amount;
            pot += a.amount;
            if a.kind == ActionType::Fold {
                folded.insert(a.seat);
            }
            if a.kind.is_aggressive() {
                aggressions += 1;
                last_aggressor = Some(a.seat);
            }
            index += 1;
        }
        initiative = last_aggressor;
    }
    out
}

// ------------------------------------------------------------------ frequencies

/// Default frequencies (fold, check/call, bet/raise) of a context: a rough
/// six-max population, used only until data outweighs them.
pub fn default_frequencies(context: &str) -> [f64; 3] {
    let (street, spot) = context.split_once('_').unwrap_or(("", context));
    match spot {
        "open_early" => [0.80, 0.03, 0.17],
        "open_late" => [0.60, 0.05, 0.35],
        "open_blind" => [0.60, 0.15, 0.25],
        "option" => [0.0, 0.85, 0.15],
        "vs_open" => [0.70, 0.20, 0.10],
        "vs_3bet" => [0.50, 0.35, 0.15],
        "cbet" => match street {
            "flop" => [0.0, 0.40, 0.60],
            "turn" => [0.0, 0.50, 0.50],
            _ => [0.0, 0.55, 0.45],
        },
        "lead" => [0.0, 0.92, 0.08],
        "stab" => [0.0, 0.60, 0.40],
        "vs_bet" => [0.45, 0.45, 0.10],
        _ => [0.40, 0.45, 0.15],
    }
}

/// Decision counts by player and context (fold, check/call, bet/raise).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FrequencyTable {
    pub population: BTreeMap<String, [f64; 3]>,
    pub players: BTreeMap<String, BTreeMap<String, [f64; 3]>>,
    pub hands: usize,
}

impl FrequencyTable {
    /// Counts every hold'em decision in which a bet or raise was possible.
    pub fn from_hands(hands: &[Hand]) -> FrequencyTable {
        let mut t = FrequencyTable::default();
        for h in hands {
            if !is_holdem(h) {
                continue;
            }
            t.hands += 1;
            for d in decisions(h) {
                if !d.available[2] {
                    continue;
                }
                let name = h.name_of(d.seat).to_string();
                t.population.entry(d.context.clone()).or_insert([0.0; 3])[d.category] += 1.0;
                t.players.entry(name).or_default().entry(d.context).or_insert([0.0; 3])[d.category] += 1.0;
            }
        }
        t
    }

    /// Estimated frequencies of `player` in `context`, restricted to the available
    /// categories and renormalised.
    pub fn estimate(&self, player: &str, context: &str, available: [bool; 3], opts: &RangeOptions) -> [f64; 3] {
        let d = default_frequencies(context);
        let pc = self.population.get(context).copied().unwrap_or([0.0; 3]);
        let pn = pc[0] + pc[1] + pc[2];
        let m0 = opts.population_prior;
        let pop = [(pc[0] + m0 * d[0]) / (pn + m0), (pc[1] + m0 * d[1]) / (pn + m0), (pc[2] + m0 * d[2]) / (pn + m0)];
        let mc = self.players.get(player).and_then(|m| m.get(context)).copied().unwrap_or([0.0; 3]);
        let mn = mc[0] + mc[1] + mc[2];
        let m = opts.player_prior;
        let mut p = [(mc[0] + m * pop[0]) / (mn + m), (mc[1] + m * pop[1]) / (mn + m), (mc[2] + m * pop[2]) / (mn + m)];
        for a in 0..3 {
            if !available[a] {
                p[a] = 0.0;
            }
        }
        let t = p[0] + p[1] + p[2];
        if t > 0.0 {
            for x in p.iter_mut() {
                *x /= t;
            }
        } else {
            let k = available.iter().filter(|x| **x).count() as f64;
            for a in 0..3 {
                p[a] = if available[a] { 1.0 / k } else { 0.0 };
            }
        }
        p
    }
}

/// Hold'em on a full deck (the only game the range model covers).
pub fn is_holdem(h: &Hand) -> bool {
    h.game == "NLH" || h.game == "FLH" || h.game == "LHE"
}

// ------------------------------------------------------------------ strength

/// Preflop strength of every combo: 168 for aces down to 0 for 32o.
pub fn preflop_strengths() -> Vec<f64> {
    let order = pe_cards::ranking::preflop_order();
    let mut rank = vec![0usize; NUM_CLASSES];
    for (i, &c) in order.iter().enumerate() {
        rank[c] = i;
    }
    (0..NUM_COMBOS)
        .map(|i| {
            let (a, b) = combo_cards(i);
            (NUM_CLASSES - 1 - rank[class_of(a, b)]) as f64
        })
        .collect()
}

/// On a complete board: for every combo, the share of random opponent holdings
/// it beats, ties counting half, with exact card removal. Combos that touch the
/// board get -1.
pub fn showdown_shares(board: CardMask) -> Vec<f64> {
    let mut idx: Vec<usize> = Vec::with_capacity(1081);
    let mut vals: Vec<u32> = Vec::with_capacity(1081);
    for i in 0..NUM_COMBOS {
        let m = combo_mask(i);
        if m & board == 0 {
            idx.push(i);
            vals.push(pe_eval::eval7(board | m));
        }
    }
    let mut order: Vec<usize> = (0..idx.len()).collect();
    order.sort_by(|&a, &b| vals[a].cmp(&vals[b]).then(a.cmp(&b)));
    let left = 52 - board.count_ones() as usize - 2;
    let compatible = (left * (left - 1) / 2) as f64;
    let mut out = vec![-1.0; NUM_COMBOS];
    let mut below = 0.0f64;
    let mut below_card = [0.0f64; 52];
    let mut g = 0;
    while g < order.len() {
        let v = vals[order[g]];
        let mut e = g;
        while e < order.len() && vals[order[e]] == v {
            e += 1;
        }
        let mut eq_card = [0.0f64; 52];
        for &k in &order[g..e] {
            let (a, b) = combo_cards(idx[k]);
            eq_card[a.id()] += 1.0;
            eq_card[b.id()] += 1.0;
        }
        let size = (e - g) as f64;
        for &k in &order[g..e] {
            let (a, b) = combo_cards(idx[k]);
            let lt = below - below_card[a.id()] - below_card[b.id()];
            let eq = size - eq_card[a.id()] - eq_card[b.id()] + 1.0;
            out[idx[k]] = (lt + 0.5 * eq) / compatible;
        }
        below += size;
        for c in 0..52 {
            below_card[c] += eq_card[c];
        }
        g = e;
    }
    out
}

/// Strength of every combo on a board of 0, 3, 4 or 5 cards (see the module
/// documentation); -1 for combos that touch the board.
pub fn board_strengths(board: &[Card]) -> Vec<f64> {
    let bm = mask_of(board);
    match board.len() {
        0 => preflop_strengths(),
        5 => showdown_shares(bm),
        _ => {
            let rest: Vec<Card> = (0..52u8).map(Card).filter(|c| c.mask() & bm == 0).collect();
            let mut acc = vec![0.0f64; NUM_COMBOS];
            let mut cnt = vec![0.0f64; NUM_COMBOS];
            let mut add = |extra: CardMask| {
                let s = showdown_shares(bm | extra);
                for i in 0..NUM_COMBOS {
                    if s[i] >= 0.0 {
                        acc[i] += s[i];
                        cnt[i] += 1.0;
                    }
                }
            };
            if board.len() == 4 {
                for c in &rest {
                    add(c.mask());
                }
            } else {
                for i in 0..rest.len() {
                    for j in i + 1..rest.len() {
                        add(rest[i].mask() | rest[j].mask());
                    }
                }
            }
            (0..NUM_COMBOS).map(|i| if cnt[i] > 0.0 { acc[i] / cnt[i] } else { -1.0 }).collect()
        }
    }
}

// ------------------------------------------------------------------ threshold strategy

/// Bands of a context, weakest first: (category, share of the range).
pub fn bands(context: &str, f: [f64; 3]) -> Vec<(usize, f64)> {
    if context.starts_with("preflop") {
        return vec![(0, f[0]), (1, f[1]), (2, f[2])];
    }
    let spot = context.split_once('_').map_or("", |x| x.1);
    if spot == "cbet" || spot == "lead" || spot == "stab" {
        let b = bet_bluff_share();
        vec![(0, f[0]), (2, f[2] * b), (1, f[1]), (2, f[2] * (1.0 - b))]
    } else {
        let r = raise_bluff_share();
        vec![(2, f[2] * r), (0, f[0]), (1, f[1]), (2, f[2] * (1.0 - r))]
    }
}

/// Splits the combos with positive weight into consecutive bands by strength
/// (weakest first), each taking its share of the total weight. Returns each
/// combo's probability of every category; combos of equal strength get the same.
pub fn band_strategy(weights: &[f64], strength: &[f64], bands: &[(usize, f64)]) -> Vec<[f64; 3]> {
    let mut out = vec![[0.0; 3]; NUM_COMBOS];
    let mut order: Vec<usize> = (0..NUM_COMBOS).filter(|&i| weights[i] > 0.0).collect();
    order.sort_by(|&a, &b| strength[a].partial_cmp(&strength[b]).unwrap().then(a.cmp(&b)));
    let mut total = 0.0;
    for &i in &order {
        total += weights[i];
    }
    if total <= 0.0 {
        return out;
    }
    let mut bounds = Vec::with_capacity(bands.len());
    let mut acc = 0.0;
    for (k, &(_, s)) in bands.iter().enumerate() {
        acc += s * total;
        bounds.push(if k + 1 == bands.len() { total } else { acc });
    }
    let mut lo = 0.0;
    let mut g = 0;
    while g < order.len() {
        let s = strength[order[g]];
        let mut e = g;
        let mut w = 0.0;
        while e < order.len() && strength[order[e]] == s {
            w += weights[order[e]];
            e += 1;
        }
        let hi = lo + w;
        let mut sigma = [0.0; 3];
        let mut b_lo = 0.0;
        for (k, &(cat, _)) in bands.iter().enumerate() {
            let b_hi = bounds[k];
            let overlap = (hi.min(b_hi) - lo.max(b_lo)).max(0.0);
            sigma[cat] += overlap / w;
            b_lo = b_hi;
        }
        for &i in &order[g..e] {
            out[i] = sigma;
        }
        lo = hi;
        g = e;
    }
    out
}

// ------------------------------------------------------------------ reconstruction

#[derive(Clone, Debug, Serialize)]
pub struct ActualHand {
    pub cards: String,
    pub prob: f64,
    /// log2(prob × possible combos): what the actions revealed about this hand
    /// compared with knowing nothing, in bits.
    pub bits: f64,
    /// Probability of the combos more likely than the actual one, plus half of
    /// the equally likely ones. Uniform on [0, 1] when the model is calibrated.
    pub hpd_quantile: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct SeatRange {
    pub seat: u32,
    pub name: String,
    pub position: String,
    /// Probability of each of the 1,326 combos (0 for combos on the board).
    #[serde(skip)]
    pub probs: Vec<f64>,
    /// Probability of each starting-hand class (13 x 13 grid order).
    pub grid: Vec<f64>,
    /// Combos of each class not blocked by the board.
    pub grid_combos: Vec<u32>,
    /// exp(entropy): as uncertain as a uniform range of this many combos.
    pub effective_combos: f64,
    pub possible_combos: usize,
    /// The ten most likely combos.
    pub top: Vec<(String, f64)>,
    pub actual: Option<ActualHand>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Snapshot {
    /// `preflop`, `flop`, `turn`, `river` (start of the street) or `end`.
    pub label: String,
    pub board: String,
    pub seats: Vec<SeatRange>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ActionNote {
    pub index: usize,
    pub street: StreetName,
    pub seat: u32,
    pub action: ActionType,
    pub context: String,
    /// Estimated frequencies of the player in this context (fold, check/call, bet/raise).
    pub frequencies: [f64; 3],
    /// Probability of the action under the model with the player's actual cards.
    pub likelihood_actual: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct HandRanges {
    pub id: String,
    pub snapshots: Vec<Snapshot>,
    pub actions: Vec<ActionNote>,
}

impl HandRanges {
    pub fn snapshot(&self, label: &str) -> Option<&Snapshot> {
        self.snapshots.iter().find(|s| s.label == label)
    }
}

fn combo_of(cards: &[Card]) -> Option<usize> {
    if cards.len() == 2 && cards[0] != cards[1] { Some(pe_cards::combo_index(cards[0], cards[1])) } else { None }
}

fn summarise(hand: &Hand, seat: u32, probs: &[f64], blocked: CardMask) -> SeatRange {
    let mut grid = vec![0.0; NUM_CLASSES];
    let mut grid_combos = vec![0u32; NUM_CLASSES];
    let mut entropy = 0.0;
    let mut possible = 0;
    for i in 0..NUM_COMBOS {
        let (a, b) = combo_cards(i);
        if combo_mask(i) & blocked != 0 {
            continue;
        }
        possible += 1;
        let c = class_of(a, b);
        grid_combos[c] += 1;
        grid[c] += probs[i];
        if probs[i] > 0.0 {
            entropy -= probs[i] * probs[i].ln();
        }
    }
    let mut ranked: Vec<usize> = (0..NUM_COMBOS).filter(|&i| probs[i] > 0.0).collect();
    ranked.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap().then(a.cmp(&b)));
    let top = ranked
        .iter()
        .take(10)
        .map(|&i| {
            let (a, b) = combo_cards(i);
            (cards_to_string(&[a, b]), probs[i])
        })
        .collect();
    let actual = hand.hole_cards(seat).and_then(|c| combo_of(&c)).filter(|&i| combo_mask(i) & blocked == 0).map(|i| {
        let p = probs[i];
        let (mut above, mut equal) = (0.0, 0.0);
        for &q in probs {
            if q > p {
                above += q;
            } else if q == p {
                equal += q;
            }
        }
        let (a, b) = combo_cards(i);
        ActualHand {
            cards: cards_to_string(&[a, b]),
            prob: p,
            bits: (p.max(1e-300) * possible as f64).log2(),
            hpd_quantile: above + 0.5 * equal,
        }
    });
    let player = hand.player(seat);
    SeatRange {
        seat,
        name: player.map_or(String::new(), |p| p.name.clone()),
        position: player.and_then(|p| p.position.clone()).unwrap_or_default(),
        probs: probs.to_vec(),
        grid,
        grid_combos,
        effective_combos: entropy.exp(),
        possible_combos: possible,
        top,
        actual,
    }
}

fn normalise(w: &mut [f64]) {
    let mut t = 0.0;
    for x in w.iter() {
        t += *x;
    }
    if t > 0.0 {
        for x in w.iter_mut() {
            *x /= t;
        }
    }
}

/// Strength of every combo for each street of the hand that has actions, in
/// the order of `hand.streets` (see [`board_strengths`]).
pub fn street_strengths(hand: &Hand) -> Vec<Vec<f64>> {
    let board0 = hand.board_cards(0);
    hand.streets.iter().map(|st| board_strengths(&board0[..st.name.board_len().min(board0.len())])).collect()
}

/// Reconstructs every player's range street by street (hold'em only).
pub fn reconstruct_ranges(hand: &Hand, table: &FrequencyTable, opts: &RangeOptions) -> Result<HandRanges, String> {
    if !is_holdem(hand) {
        return Err(format!("hand {}: ranges are reconstructed for hold'em only (game {})", hand.id, hand.game));
    }
    reconstruct_with(hand, table, opts, &street_strengths(hand))
}

/// [`reconstruct_ranges`] with the street strengths precomputed (they depend
/// only on the board, so experiments reuse them across options).
pub fn reconstruct_with(hand: &Hand, table: &FrequencyTable, opts: &RangeOptions, strengths: &[Vec<f64>]) -> Result<HandRanges, String> {
    if strengths.len() != hand.streets.len() {
        return Err(format!("hand {}: one strength vector per street is needed", hand.id));
    }
    let dealt = hand.dealt_seats();
    let mut w: BTreeMap<u32, Vec<f64>> = dealt.iter().map(|&s| (s, vec![1.0 / NUM_COMBOS as f64; NUM_COMBOS])).collect();
    let mut folded: BTreeSet<u32> = BTreeSet::new();
    let decisions = decisions(hand);
    let board0 = hand.board_cards(0);
    let board1 = if hand.board_kind == "double" { hand.board_cards(1) } else { Vec::new() };
    let visible = |len: usize| -> (Vec<Card>, CardMask) {
        let b: Vec<Card> = board0.iter().take(len).copied().collect();
        let m = mask_of(&b) | mask_of(&board1.iter().take(len).copied().collect::<Vec<_>>());
        (b, m)
    };
    let mut snapshots = Vec::new();
    let mut notes = Vec::new();
    let mut di = 0;
    let mut last_len = 0;
    for (si, st) in hand.streets.iter().enumerate() {
        let len = st.name.board_len().min(board0.len());
        let (board, blocked) = visible(len);
        last_len = len;
        for (_, v) in w.iter_mut() {
            for i in 0..NUM_COMBOS {
                if combo_mask(i) & blocked != 0 {
                    v[i] = 0.0;
                }
            }
            normalise(v);
        }
        let strength = &strengths[si];
        snapshots.push(Snapshot {
            label: st.name.as_str().to_string(),
            board: cards_to_string(&board),
            seats: dealt.iter().filter(|s| !folded.contains(s)).map(|&s| summarise(hand, s, &w[&s], blocked)).collect(),
        });
        while di < decisions.len() && decisions[di].street == st.name {
            let d = &decisions[di];
            di += 1;
            let name = hand.name_of(d.seat);
            let f = table.estimate(name, &d.context, d.available, opts);
            let sigma = band_strategy(&w[&d.seat], strength, &bands(&d.context, f));
            let k = d.available.iter().filter(|x| **x).count() as f64;
            let (informed, noise, tremble) = (1.0 - opts.noise - opts.tremble, opts.noise * f[d.category], opts.tremble / k);
            let likelihood = |i: usize| informed * sigma[i][d.category] + noise + tremble;
            let actual = hand.hole_cards(d.seat).and_then(|c| combo_of(&c)).map(likelihood);
            let v = w.get_mut(&d.seat).unwrap();
            for i in 0..NUM_COMBOS {
                if v[i] > 0.0 {
                    v[i] *= likelihood(i);
                }
            }
            normalise(v);
            notes.push(ActionNote { index: d.index, street: d.street, seat: d.seat, action: d.action, context: d.context.clone(), frequencies: f, likelihood_actual: actual });
            if d.action == ActionType::Fold {
                folded.insert(d.seat);
            }
        }
    }
    // the end: every board card dealt, for the players still in
    let end_len = if dealt.len() - folded.len() >= 2 { board0.len() } else { last_len };
    let (board, blocked) = visible(end_len);
    for (_, v) in w.iter_mut() {
        for i in 0..NUM_COMBOS {
            if combo_mask(i) & blocked != 0 {
                v[i] = 0.0;
            }
        }
        normalise(v);
    }
    snapshots.push(Snapshot {
        label: "end".into(),
        board: cards_to_string(&board),
        seats: dealt.iter().filter(|s| !folded.contains(s)).map(|&s| summarise(hand, s, &w[&s], blocked)).collect(),
    });
    Ok(HandRanges { id: hand.id.clone(), snapshots, actions: notes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pe_cards::parse_cards;

    #[test]
    fn showdown_shares_match_brute_force() {
        let board = parse_cards("Kh8d3c2s2h").unwrap();
        let bm = mask_of(&board);
        let fast = showdown_shares(bm);
        for i in (0..NUM_COMBOS).step_by(37) {
            let m = combo_mask(i);
            if m & bm != 0 {
                assert_eq!(fast[i], -1.0);
                continue;
            }
            let me = pe_eval::eval7(bm | m);
            let (mut s, mut n) = (0.0, 0.0);
            for j in 0..NUM_COMBOS {
                let o = combo_mask(j);
                if o & (bm | m) != 0 {
                    continue;
                }
                let v = pe_eval::eval7(bm | o);
                s += if me > v { 1.0 } else if me == v { 0.5 } else { 0.0 };
                n += 1.0;
            }
            assert!((fast[i] - s / n).abs() < 1e-12, "{i}: {} vs {}", fast[i], s / n);
        }
    }

    #[test]
    fn bands_split_ties_evenly() {
        let mut w = vec![0.0; NUM_COMBOS];
        let mut s = vec![0.0; NUM_COMBOS];
        for i in 0..4 {
            w[i] = 1.0;
            s[i] = if i < 2 { 1.0 } else { 2.0 };
        }
        // a quarter folds: half of the weaker tie group
        let sigma = band_strategy(&w, &s, &[(0, 0.25), (1, 0.75)]);
        assert_eq!(sigma[0], [0.5, 0.5, 0.0]);
        assert_eq!(sigma[1], [0.5, 0.5, 0.0]);
        assert_eq!(sigma[3], [0.0, 1.0, 0.0]);
    }

    #[test]
    fn draws_count_on_the_flop() {
        // on Ah7h2c the same ranks are worth more with a flush draw once runouts count
        let board = parse_cards("Ah7h2c").unwrap();
        let s = board_strengths(&board);
        let idx = |c: &str| {
            let v = parse_cards(c).unwrap();
            pe_cards::combo_index(v[0], v[1])
        };
        assert!(s[idx("6h5h")] > s[idx("6d5c")] + 0.05, "{} vs {}", s[idx("6h5h")], s[idx("6d5c")]);
        assert!(s[idx("AsAd")] > s[idx("KhQh")]);
        assert_eq!(s[idx("Ah5d")], -1.0, "blocked by the board");
    }
}
