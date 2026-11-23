//! Public betting trees for postflop spots (2 to 6 players).
//!
//! Players act in seat order on every street (player 0 first). A bet of `x` adds
//! `x` times the pot; a raise of `x` first calls, then adds `x` times the pot
//! after the call. Bets that commit at least `allin_threshold` of the remaining
//! stack become all-ins. When a street closes, a chance node deals the next card
//! (one child per possible card); when fewer than two players can still bet, the
//! remaining cards are dealt without betting.

use pe_cards::{Card, CardMask, MaskIter};
use serde::{Deserialize, Serialize};

use crate::spot::{BetMenu, Spot};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Action {
    Fold,
    Check,
    Call,
    /// Opening bet; `to` is the player's contribution on this street afterwards.
    Bet { to: f64, size: f64 },
    Raise { to: f64, size: f64 },
    AllIn { to: f64 },
}

impl Action {
    pub fn label(&self) -> String {
        match self {
            Action::Fold => "Fold".into(),
            Action::Check => "Check".into(),
            Action::Call => "Call".into(),
            Action::Bet { size, .. } => format!("Bet {}%", (size * 100.0).round()),
            Action::Raise { size, .. } => format!("Raise {}%", (size * 100.0).round()),
            Action::AllIn { .. } => "All-in".into(),
        }
    }
    pub fn code(&self) -> String {
        match self {
            Action::Fold => "f".into(),
            Action::Check => "x".into(),
            Action::Call => "c".into(),
            Action::Bet { size, .. } => format!("b{}", (size * 100.0).round()),
            Action::Raise { size, .. } => format!("r{}", (size * 100.0).round()),
            Action::AllIn { .. } => "a".into(),
        }
    }
    pub fn is_aggressive(&self) -> bool {
        matches!(self, Action::Bet { .. } | Action::Raise { .. } | Action::AllIn { .. })
    }
}

#[derive(Clone, Debug, Serialize)]
pub enum NodeKind {
    Decision { player: usize, actions: Vec<Action>, children: Vec<usize> },
    /// Deals one card to the (first) board.
    Chance { cards: Vec<Card>, children: Vec<usize> },
    Terminal { showdown: bool },
}

#[derive(Clone, Debug, Serialize)]
pub struct Node {
    pub kind: NodeKind,
    /// Cards dealt on the first board so far (including the starting board).
    pub board: CardMask,
    pub board_len: usize,
    /// Contributions of each player since the start of the spot.
    pub contrib: Vec<f64>,
    pub folded: Vec<bool>,
    /// Starting pot plus all contributions.
    pub pot: f64,
    pub parent: Option<usize>,
    /// Action codes from the root, e.g. `x-b75-c|Th|x`.
    pub path: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct BetTree {
    pub nodes: Vec<Node>,
    pub players: usize,
    pub start_pot: f64,
}

#[derive(Clone)]
struct State {
    board: CardMask,
    board_len: usize,
    contrib: Vec<f64>,
    street: Vec<f64>,
    stacks: Vec<f64>,
    folded: Vec<bool>,
    need: Vec<bool>,
    raises: usize,
    last_inc: f64,
    path: String,
}

const EPS: f64 = 1e-9;

impl BetTree {
    pub fn build(spot: &Spot, menu: &BetMenu, max_nodes: usize) -> Result<BetTree, String> {
        let n = spot.players();
        let mut t = BetTree { nodes: Vec::new(), players: n, start_pot: spot.pot };
        let st = State {
            board: pe_cards::mask_of(&spot.board),
            board_len: spot.board.len(),
            contrib: vec![0.0; n],
            street: vec![0.0; n],
            stacks: spot.stacks.clone(),
            folded: vec![false; n],
            need: spot.stacks.iter().map(|s| *s > EPS).collect(),
            raises: 0,
            last_inc: 0.0,
            path: String::new(),
        };
        let able = st.need.iter().filter(|&&x| x).count();
        if able >= 2 {
            let first = (0..n).find(|&p| st.need[p]).unwrap();
            t.decision(spot, menu, st, first, None, max_nodes)?;
        } else {
            t.street_over(spot, menu, st, None, max_nodes)?;
        }
        Ok(t)
    }

    fn push(&mut self, st: &State, start_pot: f64, parent: Option<usize>, kind: NodeKind, max_nodes: usize) -> Result<usize, String> {
        if self.nodes.len() >= max_nodes {
            return Err(format!("the tree exceeds {max_nodes} nodes: use fewer bet sizes or a later street"));
        }
        let pot = start_pot + st.contrib.iter().sum::<f64>();
        self.nodes.push(Node {
            kind,
            board: st.board,
            board_len: st.board_len,
            contrib: st.contrib.clone(),
            folded: st.folded.clone(),
            pot,
            parent,
            path: st.path.clone(),
        });
        Ok(self.nodes.len() - 1)
    }

    fn actions(&self, menu: &BetMenu, st: &State, p: usize) -> Vec<Action> {
        let n = st.street.len();
        let max_c = st.street.iter().cloned().fold(0.0, f64::max);
        let facing = st.street[p] < max_c - EPS;
        let call = (max_c - st.street[p]).min(st.stacks[p]);
        let pot_now = self.start_pot + st.contrib.iter().sum::<f64>();
        let mut acts = Vec::new();
        if facing {
            acts.push(Action::Fold);
            acts.push(Action::Call);
        } else {
            acts.push(Action::Check);
        }
        let others_can_act = (0..n).any(|q| q != p && !st.folded[q] && st.stacks[q] > EPS);
        if st.raises < menu.max_raises && st.stacks[p] > call + EPS && others_can_act {
            let max_to = st.street[p] + st.stacks[p];
            let sizes = if facing || st.raises > 0 { &menu.raises } else { &menu.bets };
            let mut tos: Vec<(f64, Action)> = Vec::new();
            for &x in sizes {
                let to = if facing || st.raises > 0 { max_c + x * (pot_now + call) } else { st.street[p] + x * pot_now };
                let min_to = max_c + st.last_inc.max(EPS);
                let to = to.max(min_to);
                let added = to - st.street[p];
                if to >= max_to - EPS || added >= menu.allin_threshold * st.stacks[p] {
                    tos.push((max_to, Action::AllIn { to: max_to }));
                } else if facing || st.raises > 0 {
                    tos.push((to, Action::Raise { to, size: x }));
                } else {
                    tos.push((to, Action::Bet { to, size: x }));
                }
            }
            if menu.add_allin {
                tos.push((max_to, Action::AllIn { to: max_to }));
            }
            tos.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let mut last = f64::NEG_INFINITY;
            for (to, a) in tos {
                if to > last + 1e-6 {
                    acts.push(a);
                    last = to;
                }
            }
        }
        acts
    }

    fn decision(&mut self, spot: &Spot, menu: &BetMenu, st: State, p: usize, parent: Option<usize>, max_nodes: usize) -> Result<usize, String> {
        let actions = self.actions(menu, &st, p);
        let idx = self.push(&st, spot.pot, parent, NodeKind::Decision { player: p, actions: actions.clone(), children: Vec::new() }, max_nodes)?;
        let mut children = Vec::with_capacity(actions.len());
        for a in &actions {
            let child = self.apply(spot, menu, &st, p, a, idx, max_nodes)?;
            children.push(child);
        }
        if let NodeKind::Decision { children: c, .. } = &mut self.nodes[idx].kind {
            *c = children;
        }
        Ok(idx)
    }

    #[allow(clippy::too_many_arguments)]
    fn apply(&mut self, spot: &Spot, menu: &BetMenu, st: &State, p: usize, a: &Action, parent: usize, max_nodes: usize) -> Result<usize, String> {
        let n = st.street.len();
        let mut s = st.clone();
        let max_c = st.street.iter().cloned().fold(0.0, f64::max);
        s.need[p] = false;
        match a {
            Action::Fold => s.folded[p] = true,
            Action::Check => {}
            Action::Call => {
                let add = (max_c - s.street[p]).min(s.stacks[p]);
                s.street[p] += add;
                s.contrib[p] += add;
                s.stacks[p] -= add;
            }
            Action::Bet { to, .. } | Action::Raise { to, .. } | Action::AllIn { to } => {
                let add = to - s.street[p];
                let inc = to - max_c;
                s.street[p] = *to;
                s.contrib[p] += add;
                s.stacks[p] = (s.stacks[p] - add).max(0.0);
                s.raises += 1;
                if inc > s.last_inc {
                    s.last_inc = inc;
                }
                for q in 0..n {
                    if q != p && !s.folded[q] && s.stacks[q] > EPS {
                        s.need[q] = true;
                    }
                }
            }
        }
        if !s.path.is_empty() && !s.path.ends_with('|') {
            s.path.push('-');
        }
        s.path.push_str(&a.code());
        let active = (0..n).filter(|&q| !s.folded[q]).count();
        if active == 1 {
            return self.push(&s, spot.pot, Some(parent), NodeKind::Terminal { showdown: false }, max_nodes);
        }
        if let Some(q) = (1..=n).map(|k| (p + k) % n).find(|&q| !s.folded[q] && s.need[q] && s.stacks[q] > EPS) {
            return self.decision(spot, menu, s, q, Some(parent), max_nodes);
        }
        self.street_over(spot, menu, s, Some(parent), max_nodes)
    }

    fn street_over(&mut self, spot: &Spot, menu: &BetMenu, s: State, parent: Option<usize>, max_nodes: usize) -> Result<usize, String> {
        if s.board_len >= 5 {
            return self.push(&s, spot.pot, parent, NodeKind::Terminal { showdown: true }, max_nodes);
        }
        if spot.board2.is_some() {
            return Err("double-board spots are solved on the river (both boards complete)".into());
        }
        let deck = spot.game.deck() & !(s.board | spot.board_mask() | spot.dead);
        let cards: Vec<Card> = MaskIter(deck).collect();
        let idx = self.push(&s, spot.pot, parent, NodeKind::Chance { cards: cards.clone(), children: Vec::new() }, max_nodes)?;
        let n = s.street.len();
        let mut children = Vec::with_capacity(cards.len());
        for c in &cards {
            let mut t = s.clone();
            t.board |= c.mask();
            t.board_len += 1;
            t.street = vec![0.0; n];
            t.raises = 0;
            t.last_inc = 0.0;
            t.need = (0..n).map(|q| !t.folded[q] && t.stacks[q] > EPS).collect();
            t.path.push('|');
            t.path.push_str(&c.to_string());
            t.path.push('|');
            let able = t.need.iter().filter(|&&x| x).count();
            let child = if able >= 2 {
                let first = (0..n).find(|&q| t.need[q]).unwrap();
                self.decision(spot, menu, t, first, Some(idx), max_nodes)?
            } else {
                for x in t.need.iter_mut() {
                    *x = false;
                }
                self.street_over(spot, menu, t, Some(idx), max_nodes)?
            };
            children.push(child);
        }
        if let NodeKind::Chance { children: c, .. } = &mut self.nodes[idx].kind {
            *c = children;
        }
        Ok(idx)
    }

    pub fn decision_nodes(&self) -> usize {
        self.nodes.iter().filter(|n| matches!(n.kind, NodeKind::Decision { .. })).count()
    }

    /// Node reached from the root by following action codes (e.g. `["x", "b75"]`) on the first street.
    pub fn find(&self, codes: &[&str]) -> Option<usize> {
        let mut idx = 0;
        for code in codes {
            match &self.nodes[idx].kind {
                NodeKind::Decision { actions, children, .. } => {
                    let k = actions.iter().position(|a| a.code() == *code)?;
                    idx = children[k];
                }
                _ => return None,
            }
        }
        Some(idx)
    }
}

/// Side-pot settlement for a terminal: for each player, the amount they put in
/// that is still contested (uncalled excess returned). Returns (pots, eligibility)
/// where each pot lists the players who can win it; the starting pot joins the
/// first pot.
pub fn side_pots(start_pot: f64, contrib: &[f64], folded: &[bool]) -> Vec<(f64, Vec<usize>)> {
    let n = contrib.len();
    let mut levels: Vec<f64> = contrib.to_vec();
    levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    levels.dedup_by(|a, b| (*a - *b).abs() < EPS);
    let mut pots = Vec::new();
    let mut prev = 0.0;
    for &lv in &levels {
        if lv <= prev + EPS {
            continue;
        }
        let mut amount = 0.0;
        for q in 0..n {
            amount += (contrib[q].min(lv) - prev).max(0.0);
        }
        let eligible: Vec<usize> = (0..n).filter(|&q| !folded[q] && contrib[q] >= lv - EPS).collect();
        pots.push((amount, eligible));
        prev = lv;
    }
    if pots.is_empty() {
        pots.push((0.0, (0..n).filter(|&q| !folded[q]).collect()));
    }
    pots[0].0 += start_pot;
    if pots[0].1.is_empty() {
        pots[0].1 = (0..n).filter(|&q| !folded[q]).collect();
    }
    // a pot nobody else contests is returned to its only contributor
    pots
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spot::SpotConfig;

    fn river(menu: BetMenu, stacks: f64) -> (Spot, BetTree) {
        let cfg = SpotConfig {
            game: "NLH".into(),
            board: "Ks7d2c9h4s".into(),
            board2: None,
            ranges: vec!["AA,KQ".into(), "QQ,K9s".into()],
            pot: 10.0,
            stacks: vec![stacks],
            menu: menu.clone(),
            rake: None,
            side_rule: None,
            dead: None,
        };
        let s = Spot::from_config(cfg).unwrap();
        let t = BetTree::build(&s, &menu, 100_000).unwrap();
        (s, t)
    }

    #[test]
    fn river_tree_shape() {
        let menu = BetMenu { bets: vec![0.5], raises: vec![1.0], max_raises: 1, allin_threshold: 0.99, add_allin: false };
        let (_, t) = river(menu, 100.0);
        // root: check / bet 50; after check: check / bet 50; after a bet: fold / call
        let root = &t.nodes[0];
        let NodeKind::Decision { actions, .. } = &root.kind else { panic!() };
        assert_eq!(actions.len(), 2);
        let xb = t.find(&["x", "b50"]).unwrap();
        let NodeKind::Decision { actions, player, .. } = &t.nodes[xb].kind else { panic!() };
        assert_eq!(*player, 0);
        assert_eq!(actions.iter().map(|a| a.code()).collect::<Vec<_>>(), vec!["f", "c"]);
        let xbc = t.find(&["x", "b50", "c"]).unwrap();
        assert!(matches!(t.nodes[xbc].kind, NodeKind::Terminal { showdown: true }));
        assert!((t.nodes[xbc].pot - 20.0).abs() < 1e-9);
    }

    #[test]
    fn all_in_threshold_and_raises() {
        let menu = BetMenu { bets: vec![0.75], raises: vec![1.0], max_raises: 3, allin_threshold: 0.67, add_allin: false };
        let (_, t) = river(menu, 20.0);
        // bet 7.5 of 20, raise to 7.5 + 7.5 + (25 + 7.5)... capped -> all-in for 20
        let b = t.find(&["b75"]).unwrap();
        let NodeKind::Decision { actions, .. } = &t.nodes[b].kind else { panic!() };
        assert!(actions.iter().any(|a| matches!(a, Action::AllIn { .. })), "{actions:?}");
    }

    #[test]
    fn side_pot_settlement() {
        let pots = side_pots(10.0, &[50.0, 20.0, 50.0], &[false, false, false]);
        assert_eq!(pots.len(), 2);
        assert!((pots[0].0 - 70.0).abs() < 1e-9);
        assert_eq!(pots[0].1, vec![0, 1, 2]);
        assert!((pots[1].0 - 60.0).abs() < 1e-9);
        assert_eq!(pots[1].1, vec![0, 2]);
    }
}
