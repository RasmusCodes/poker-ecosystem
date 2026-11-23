//! Compiled extensive-form game trees.
//!
//! A game is described once through [`GameDef`] (a state machine), then compiled
//! into flat arrays: nodes, children, chance probabilities, terminal utilities and
//! information sets. Every algorithm in this crate runs on the compiled form,
//! which keeps traversals cache-friendly and makes memory use easy to account for.
//!
//! An optional *abstraction* maps real information sets onto fewer abstract ones
//! (for example, merging card ranks). Algorithms learn one strategy per abstract
//! set; best responses and exploitability are always computed in the real game,
//! with every real set playing its abstract set's strategy.

use std::collections::HashMap;

/// What a state is, as reported by the game definition.
pub enum NodeSpec {
    Terminal(Vec<f64>),
    Chance(Vec<f64>),
    Decision { player: usize, infoset: String, actions: Vec<String> },
}

/// A game as a state machine. Information-set keys must encode exactly what the
/// acting player knows (perfect recall is the definer's responsibility).
pub trait GameDef {
    type State: Clone;
    fn num_players(&self) -> usize;
    fn initial(&self) -> Self::State;
    fn spec(&self, s: &Self::State) -> NodeSpec;
    fn next(&self, s: &Self::State, action: usize) -> Self::State;
    fn name(&self) -> String;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Terminal,
    Chance,
    Decision,
}

#[derive(Clone, Copy, Debug)]
pub struct Node {
    pub kind: Kind,
    pub player: u8,
    /// Real information set (decision nodes).
    pub infoset: u32,
    pub first_child: u32,
    pub num_children: u32,
    /// Index into `probs` (chance) or `utils` (terminal, `num_players` entries).
    pub data: u32,
    pub depth: u16,
}

#[derive(Clone, Debug)]
pub struct Infoset {
    pub key: String,
    pub player: usize,
    pub actions: Vec<String>,
    /// Abstract information set used by the learning algorithms.
    pub abs: u32,
}

#[derive(Clone, Debug)]
pub struct AbsInfoset {
    pub key: String,
    pub player: usize,
    pub num_actions: usize,
    /// Offset of this set's actions in flat strategy/regret arrays.
    pub offset: usize,
}

#[derive(Clone, Debug)]
pub struct Tree {
    pub name: String,
    pub num_players: usize,
    pub nodes: Vec<Node>,
    pub children: Vec<u32>,
    pub probs: Vec<f64>,
    pub utils: Vec<f64>,
    pub infosets: Vec<Infoset>,
    pub abs_infosets: Vec<AbsInfoset>,
    /// Total actions over abstract sets (length of flat strategy arrays).
    pub abs_actions: usize,
}

/// A behaviour strategy over abstract information sets (flat, by `AbsInfoset::offset`).
#[derive(Clone, Debug)]
pub struct Strategy {
    pub probs: Vec<f64>,
}

impl Tree {
    /// Compiles a game with no abstraction.
    pub fn compile<G: GameDef>(def: &G) -> Tree {
        Self::compile_abstracted(def, |k: &str| k.to_string())
    }

    /// Compiles a game, mapping each information-set key to an abstract key.
    /// Abstract sets must agree on the acting player and the number of actions.
    pub fn compile_abstracted<G: GameDef>(def: &G, abstraction: impl Fn(&str) -> String) -> Tree {
        let mut t = Tree {
            name: def.name(),
            num_players: def.num_players(),
            nodes: Vec::new(),
            children: Vec::new(),
            probs: Vec::new(),
            utils: Vec::new(),
            infosets: Vec::new(),
            abs_infosets: Vec::new(),
            abs_actions: 0,
        };
        let mut iset_ids: HashMap<String, u32> = HashMap::new();
        let mut abs_ids: HashMap<String, u32> = HashMap::new();
        // breadth-first so that children of a node are contiguous
        let mut queue: std::collections::VecDeque<(G::State, u32, u16)> = std::collections::VecDeque::new();
        t.nodes.push(Node { kind: Kind::Terminal, player: 0, infoset: 0, first_child: 0, num_children: 0, data: 0, depth: 0 });
        queue.push_back((def.initial(), 0, 0));
        while let Some((state, idx, depth)) = queue.pop_front() {
            let spec = def.spec(&state);
            let mut node = Node { kind: Kind::Terminal, player: 0, infoset: 0, first_child: 0, num_children: 0, data: 0, depth };
            let n_children = match &spec {
                NodeSpec::Terminal(u) => {
                    assert_eq!(u.len(), t.num_players, "terminal utility length");
                    node.kind = Kind::Terminal;
                    node.data = t.utils.len() as u32;
                    t.utils.extend_from_slice(u);
                    0
                }
                NodeSpec::Chance(p) => {
                    node.kind = Kind::Chance;
                    node.data = t.probs.len() as u32;
                    let s: f64 = p.iter().sum();
                    assert!((s - 1.0).abs() < 1e-9, "chance probabilities sum to {s}");
                    t.probs.extend_from_slice(p);
                    p.len()
                }
                NodeSpec::Decision { player, infoset, actions } => {
                    node.kind = Kind::Decision;
                    node.player = *player as u8;
                    let next_id = t.infosets.len() as u32;
                    let id = *iset_ids.entry(infoset.clone()).or_insert(next_id);
                    if id == next_id {
                        let akey = abstraction(infoset);
                        let next_abs = t.abs_infosets.len() as u32;
                        let abs = *abs_ids.entry(akey.clone()).or_insert(next_abs);
                        if abs == next_abs {
                            t.abs_infosets.push(AbsInfoset { key: akey, player: *player, num_actions: actions.len(), offset: t.abs_actions });
                            t.abs_actions += actions.len();
                        } else {
                            let a = &t.abs_infosets[abs as usize];
                            assert!(a.player == *player && a.num_actions == actions.len(), "abstraction merges incompatible sets: {infoset}");
                        }
                        t.infosets.push(Infoset { key: infoset.clone(), player: *player, actions: actions.clone(), abs });
                    } else {
                        let is = &t.infosets[id as usize];
                        assert!(is.player == *player && is.actions.len() == actions.len(), "inconsistent information set {infoset}");
                    }
                    node.infoset = id;
                    actions.len()
                }
            };
            node.num_children = n_children as u32;
            node.first_child = t.children.len() as u32;
            for a in 0..n_children {
                let child_idx = t.nodes.len() as u32;
                t.nodes.push(Node { kind: Kind::Terminal, player: 0, infoset: 0, first_child: 0, num_children: 0, data: 0, depth: depth + 1 });
                t.children.push(child_idx);
                queue.push_back((def.next(&state, a), child_idx, depth + 1));
            }
            t.nodes[idx as usize] = node;
        }
        t
    }

    #[inline]
    pub fn child(&self, node: &Node, a: usize) -> u32 {
        self.children[node.first_child as usize + a]
    }

    #[inline]
    pub fn util(&self, node: &Node, player: usize) -> f64 {
        self.utils[node.data as usize + player]
    }

    #[inline]
    pub fn chance_prob(&self, node: &Node, a: usize) -> f64 {
        self.probs[node.data as usize + a]
    }

    /// Uniform strategy over abstract sets.
    pub fn uniform(&self) -> Strategy {
        let mut probs = vec![0.0; self.abs_actions];
        for a in &self.abs_infosets {
            for k in 0..a.num_actions {
                probs[a.offset + k] = 1.0 / a.num_actions as f64;
            }
        }
        Strategy { probs }
    }

    /// Bytes used by the compiled tree (excluding strings).
    pub fn memory_bytes(&self) -> usize {
        self.nodes.len() * std::mem::size_of::<Node>()
            + self.children.len() * 4
            + self.probs.len() * 8
            + self.utils.len() * 8
            + self.infosets.len() * std::mem::size_of::<Infoset>()
            + self.abs_infosets.len() * std::mem::size_of::<AbsInfoset>()
    }

    /// Whether this tree is constant-sum (every terminal's utilities add to the same value).
    pub fn is_zero_sum(&self) -> bool {
        let n = self.num_players;
        self.utils.chunks(n).all(|u| u.iter().sum::<f64>().abs() < 1e-9)
    }

    /// Strategy at a real information set's node under an abstract strategy.
    #[inline]
    pub fn probs_at<'a>(&self, s: &'a Strategy, infoset: u32) -> &'a [f64] {
        let a = &self.abs_infosets[self.infosets[infoset as usize].abs as usize];
        &s.probs[a.offset..a.offset + a.num_actions]
    }
}
