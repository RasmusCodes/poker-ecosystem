//! Benchmark games with known properties.
//!
//! * [`Kuhn`]: N-player Kuhn poker (N + 1 cards, ante 1, one bet of 1). For two
//!   players the game value is -1/18 for the first player.
//! * [`Leduc`]: N-player Leduc hold'em (N + 1 ranks in two suits, ante 1, bets of
//!   2 then 4, at most two raises a round, one public card). The two-player value
//!   is about -0.0856 for the first player.
//! * [`Clairvoyance`]: a half-street river game with a closed-form equilibrium: the
//!   bettor holds the nuts with probability `v` (otherwise air) against a pure
//!   bluff-catcher, pot 1, bet `b`. At equilibrium bluffs make up `b/(1+2b)` of
//!   bets and the caller calls `1/(1+b)` of the time.

use crate::tree::{GameDef, NodeSpec};

fn permutations(n_items: usize, k: usize) -> Vec<Vec<u8>> {
    fn rec(n: usize, k: usize, cur: &mut Vec<u8>, used: &mut [bool], out: &mut Vec<Vec<u8>>) {
        if cur.len() == k {
            out.push(cur.clone());
            return;
        }
        for i in 0..n {
            if !used[i] {
                used[i] = true;
                cur.push(i as u8);
                rec(n, k, cur, used, out);
                cur.pop();
                used[i] = false;
            }
        }
    }
    let mut out = Vec::new();
    rec(n_items, k, &mut Vec::new(), &mut vec![false; n_items], &mut out);
    out
}

// ---------------------------------------------------------------- Kuhn

pub struct Kuhn {
    pub players: usize,
    deals: Vec<Vec<u8>>,
}

impl Kuhn {
    pub fn new(players: usize) -> Kuhn {
        assert!((2..=6).contains(&players));
        Kuhn { players, deals: permutations(players + 1, players) }
    }
}

#[derive(Clone)]
pub struct KuhnState {
    cards: Option<Vec<u8>>,
    history: Vec<u8>, // 0 = pass/fold, 1 = bet/call
}

impl GameDef for Kuhn {
    type State = KuhnState;
    fn name(&self) -> String {
        format!("kuhn-{}p", self.players)
    }
    fn num_players(&self) -> usize {
        self.players
    }
    fn initial(&self) -> KuhnState {
        KuhnState { cards: None, history: Vec::new() }
    }
    fn spec(&self, s: &KuhnState) -> NodeSpec {
        let n = self.players;
        let Some(cards) = &s.cards else {
            return NodeSpec::Chance(vec![1.0 / self.deals.len() as f64; self.deals.len()]);
        };
        let first_bet = s.history.iter().position(|&a| a == 1);
        let done = match first_bet {
            None => s.history.len() == n,
            Some(b) => s.history.len() == b + n,
        };
        if done {
            let mut contrib = vec![1.0; n];
            let mut contenders: Vec<usize> = Vec::new();
            match first_bet {
                None => contenders.extend(0..n),
                Some(b) => {
                    for (k, &a) in s.history.iter().enumerate().skip(b) {
                        let p = k % n;
                        if a == 1 {
                            contrib[p] += 1.0;
                            contenders.push(p);
                        }
                    }
                }
            }
            let pot: f64 = contrib.iter().sum();
            let winner = *contenders.iter().max_by_key(|&&p| cards[p]).unwrap();
            let u = (0..n).map(|p| if p == winner { pot - contrib[p] } else { -contrib[p] }).collect();
            return NodeSpec::Terminal(u);
        }
        let player = s.history.len() % n;
        let hist: String = s.history.iter().map(|&a| if a == 1 { 'b' } else { 'p' }).collect();
        let actions = if first_bet.is_some() { vec!["fold".into(), "call".into()] } else { vec!["check".into(), "bet".into()] };
        NodeSpec::Decision { player, infoset: format!("{player}:{}:{hist}", cards[player]), actions }
    }
    fn next(&self, s: &KuhnState, a: usize) -> KuhnState {
        let mut t = s.clone();
        if t.cards.is_none() {
            t.cards = Some(self.deals[a].clone());
        } else {
            t.history.push(a as u8);
        }
        t
    }
}

// ---------------------------------------------------------------- Leduc

pub struct Leduc {
    pub players: usize,
    pub ranks: u8,
    pub raise_sizes: [f64; 2],
    pub max_raises: usize,
}

impl Leduc {
    pub fn new(players: usize) -> Leduc {
        assert!((2..=4).contains(&players));
        Leduc { players, ranks: players as u8 + 1, raise_sizes: [2.0, 4.0], max_raises: 2 }
    }
    fn deck_size(&self) -> usize {
        self.ranks as usize * 2
    }
}

#[derive(Clone)]
pub struct LeducState {
    /// Private cards dealt so far (card = rank * 2 + suit).
    private: Vec<u8>,
    public: Option<u8>,
    round: usize,
    /// Action kinds per round: 0 fold, 1 check/call, 2 bet/raise (for information-set keys).
    hist: [Vec<u8>; 2],
    folded: Vec<bool>,
    contrib: Vec<f64>,
    need: Vec<bool>,
    to_act: usize,
    raises: usize,
    /// The betting round is over and the public card has not been dealt yet.
    awaiting_public: bool,
    showdown: bool,
}

impl LeducState {
    fn active(&self) -> usize {
        self.folded.iter().filter(|&&f| !f).count()
    }
    fn max_contrib(&self) -> f64 {
        self.contrib.iter().cloned().fold(0.0, f64::max)
    }
    fn start_round(&mut self) {
        let n = self.folded.len();
        self.need = (0..n).map(|p| !self.folded[p]).collect();
        self.raises = 0;
        self.to_act = (0..n).find(|&p| !self.folded[p]).unwrap();
    }
}

impl GameDef for Leduc {
    type State = LeducState;
    fn name(&self) -> String {
        format!("leduc-{}p", self.players)
    }
    fn num_players(&self) -> usize {
        self.players
    }
    fn initial(&self) -> LeducState {
        LeducState {
            private: Vec::new(),
            public: None,
            round: 0,
            hist: [Vec::new(), Vec::new()],
            folded: vec![false; self.players],
            contrib: vec![1.0; self.players],
            need: vec![true; self.players],
            to_act: 0,
            raises: 0,
            awaiting_public: false,
            showdown: false,
        }
    }
    fn spec(&self, s: &LeducState) -> NodeSpec {
        let n = self.players;
        if s.private.len() < n {
            let left = self.deck_size() - s.private.len();
            return NodeSpec::Chance(vec![1.0 / left as f64; left]);
        }
        if s.awaiting_public {
            let left = self.deck_size() - n;
            return NodeSpec::Chance(vec![1.0 / left as f64; left]);
        }
        let pot: f64 = s.contrib.iter().sum();
        if s.active() == 1 {
            let w = (0..n).find(|&p| !s.folded[p]).unwrap();
            return NodeSpec::Terminal((0..n).map(|p| if p == w { pot - s.contrib[p] } else { -s.contrib[p] }).collect());
        }
        if s.showdown {
            let public_rank = s.public.unwrap() / 2;
            let strength = |p: usize| -> u32 {
                let r = s.private[p] / 2;
                if r == public_rank { 100 + r as u32 } else { r as u32 }
            };
            let best = (0..n).filter(|&p| !s.folded[p]).map(strength).max().unwrap();
            let winners: Vec<usize> = (0..n).filter(|&p| !s.folded[p] && strength(p) == best).collect();
            let share = pot / winners.len() as f64;
            return NodeSpec::Terminal((0..n).map(|p| if winners.contains(&p) { share - s.contrib[p] } else { -s.contrib[p] }).collect());
        }
        let p = s.to_act;
        let facing = s.contrib[p] < s.max_contrib();
        let mut actions = Vec::new();
        if facing {
            actions.push("fold".to_string());
            actions.push("call".to_string());
        } else {
            actions.push("check".to_string());
        }
        if s.raises < self.max_raises {
            actions.push(if s.raises == 0 { "bet".into() } else { "raise".into() });
        }
        let code = |h: &Vec<u8>| -> String { h.iter().map(|&a| ['f', 'c', 'r'][a as usize]).collect() };
        let pubr = s.public.map_or("-".to_string(), |c| (c / 2).to_string());
        NodeSpec::Decision {
            player: p,
            infoset: format!("{p}|{}|{pubr}|{}/{}", s.private[p] / 2, code(&s.hist[0]), code(&s.hist[1])),
            actions,
        }
    }
    fn next(&self, s: &LeducState, a: usize) -> LeducState {
        let n = self.players;
        let mut t = s.clone();
        if t.private.len() < n {
            let avail: Vec<u8> = (0..self.deck_size() as u8).filter(|c| !t.private.contains(c)).collect();
            t.private.push(avail[a]);
            if t.private.len() == n {
                t.start_round();
            }
            return t;
        }
        if t.awaiting_public {
            let avail: Vec<u8> = (0..self.deck_size() as u8).filter(|c| !t.private.contains(c)).collect();
            t.public = Some(avail[a]);
            t.awaiting_public = false;
            t.round = 1;
            t.start_round();
            return t;
        }
        let p = t.to_act;
        let max_c = t.max_contrib();
        let facing = t.contrib[p] < max_c;
        let kind: u8 = if facing { a as u8 } else { a as u8 + 1 };
        t.need[p] = false;
        match kind {
            0 => t.folded[p] = true,
            1 => t.contrib[p] = max_c,
            _ => {
                t.contrib[p] = max_c + self.raise_sizes[t.round];
                t.raises += 1;
                for q in 0..n {
                    if q != p && !t.folded[q] {
                        t.need[q] = true;
                    }
                }
            }
        }
        let round = t.round;
        t.hist[round].push(kind);
        if t.active() == 1 {
            return t; // terminal by folds
        }
        match (1..=n).map(|k| (p + k) % n).find(|&q| !t.folded[q] && t.need[q]) {
            Some(q) => t.to_act = q,
            None => {
                if t.round == 0 {
                    t.awaiting_public = true;
                } else {
                    t.showdown = true;
                }
            }
        }
        t
    }
}

// ---------------------------------------------------------------- clairvoyance

pub struct Clairvoyance {
    /// Probability the bettor holds the nuts.
    pub v: f64,
    /// Bet size in units of the pot (pot = 1).
    pub b: f64,
}

#[derive(Clone)]
pub struct ClairState {
    nuts: Option<bool>,
    hist: Vec<u8>,
}

impl Clairvoyance {
    /// Closed-form equilibrium: (bettor's bluff frequency with air, caller's call
    /// frequency, game value for the bettor). Valid when `v <= (1+b)/(1+2b)`.
    pub fn solution(&self) -> (f64, f64, f64) {
        let (v, b) = (self.v, self.b);
        let bluff_freq = (v * b / (1.0 + b)) / (1.0 - v);
        let call = 1.0 / (1.0 + b);
        let value = v * (0.5 + b / (1.0 + b)) - (1.0 - v) * 0.5;
        (bluff_freq, call, value)
    }
}

impl GameDef for Clairvoyance {
    type State = ClairState;
    fn name(&self) -> String {
        format!("clairvoyance-v{}-b{}", self.v, self.b)
    }
    fn num_players(&self) -> usize {
        2
    }
    fn initial(&self) -> ClairState {
        ClairState { nuts: None, hist: Vec::new() }
    }
    fn spec(&self, s: &ClairState) -> NodeSpec {
        let Some(nuts) = s.nuts else {
            return NodeSpec::Chance(vec![self.v, 1.0 - self.v]);
        };
        let sign = if nuts { 1.0 } else { -1.0 };
        match s.hist.as_slice() {
            [] => NodeSpec::Decision {
                player: 0,
                infoset: format!("0:{}", if nuts { "nuts" } else { "air" }),
                actions: vec!["check".into(), "bet".into()],
            },
            [0] => NodeSpec::Terminal(vec![0.5 * sign, -0.5 * sign]),
            [1] => NodeSpec::Decision { player: 1, infoset: "1:bet".into(), actions: vec!["fold".into(), "call".into()] },
            [1, 0] => NodeSpec::Terminal(vec![0.5, -0.5]),
            [1, 1] => NodeSpec::Terminal(vec![(0.5 + self.b) * sign, -(0.5 + self.b) * sign]),
            _ => unreachable!(),
        }
    }
    fn next(&self, s: &ClairState, a: usize) -> ClairState {
        let mut t = s.clone();
        if t.nuts.is_none() {
            t.nuts = Some(a == 0);
        } else {
            t.hist.push(a as u8);
        }
        t
    }
}
