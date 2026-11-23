//! Range-vs-range postflop solving.
//!
//! * [`spot`]: what is being solved (game, board, ranges, pot, stacks, bet menu,
//!   rake, side rules such as the 7-2 bounty, double boards).
//! * [`tree`]: the public betting tree for 2-6 players.
//! * [`hu`]: heads-up vector-form CFR with exact best responses, node locking,
//!   opponent models and restricted Nash responses.
//! * [`multiway`]: exact-enumeration multiway rivers with exact NashConv.
//! * [`profile`]: turning observed frequencies into a model strategy.
//!
//! Labels used in every report: heads-up results are *equilibrium approximations
//! of the abstracted game* (the bet menu is the abstraction) with exact
//! exploitability; multiway results are *multiway CFR approximations* with exact
//! NashConv and no equilibrium guarantee.

pub mod hu;
pub mod multiway;
pub mod profile;
pub mod spot;
pub mod tree;

use pe_cards::sys::Stopwatch;
use pe_cfr::Algorithm;
use serde::Serialize;

pub use hu::{Buckets, Exploitability, HuSolver};
pub use multiway::{MultiwayReport, MultiwaySolver};
pub use profile::{Profile, build_model};
pub use spot::{BetMenu, Rake, SideRule, Spot, SpotConfig};
pub use tree::{Action, BetTree, NodeKind};

/// The preflop class order used for `x%` ranges.
pub fn preflop_ranking() -> Vec<usize> {
    pe_cards::ranking::preflop_order()
}

#[derive(Clone, Debug, Serialize)]
pub struct CurvePoint {
    pub iteration: u64,
    pub pct_pot: f64,
    pub cpu_seconds: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct HandRow {
    pub hand: String,
    pub weight: f64,
    pub probs: Vec<f64>,
    pub evs: Vec<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct NodeReport {
    pub node: usize,
    pub player: usize,
    pub path: String,
    pub actions: Vec<String>,
    /// Action frequencies over the range reaching the node.
    pub frequencies: Vec<f64>,
    pub hands: Vec<HandRow>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SolveReport {
    pub game: String,
    pub board: String,
    pub hands: Vec<usize>,
    pub nodes: usize,
    pub decision_nodes: usize,
    pub algorithm: String,
    pub iterations: u64,
    pub curve: Vec<CurvePoint>,
    pub result: Exploitability,
    pub memory_bytes: usize,
    pub cpu_seconds: f64,
    pub wall_seconds: f64,
    pub label: String,
    pub root: NodeReport,
}

/// Solves until `target_pct` (% of pot) or `max_iterations`, checking every
/// `check_every` iterations.
pub fn solve_hu(solver: &mut HuSolver, max_iterations: u64, target_pct: f64, check_every: u64) -> Vec<CurvePoint> {
    let mut curve = Vec::new();
    let sw = Stopwatch::start();
    while solver.iterations < max_iterations {
        let next = (solver.iterations + check_every).min(max_iterations);
        while solver.iterations < next {
            solver.iterate();
        }
        let e = solver.exploitability();
        curve.push(CurvePoint { iteration: solver.iterations, pct_pot: e.pct_pot, cpu_seconds: sw.process_cpu() });
        if e.pct_pot <= target_pct {
            break;
        }
    }
    curve
}

/// Report for one decision node: per-hand strategy and action values.
pub fn node_report(solver: &HuSolver, node: usize) -> Result<NodeReport, String> {
    let NodeKind::Decision { player, actions, .. } = &solver.tree.nodes[node].kind else {
        return Err("not a decision node".into());
    };
    let p = *player;
    let k = actions.len();
    let sigma = solver.average_strategy(node);
    let (evs, reach) = solver.action_values(node)?;
    let n = solver.hands(p);
    let frequencies = profile::action_frequencies(&sigma, &reach, k);
    let hands = (0..n)
        .filter(|&h| reach[h] > 0.0)
        .map(|h| HandRow {
            hand: pe_cards::cards_to_string(&solver.spot.ranges[p].hands[h]),
            weight: reach[h],
            probs: (0..k).map(|a| sigma[a * n + h]).collect(),
            evs: (0..k).map(|a| evs[a][h]).collect(),
        })
        .collect();
    Ok(NodeReport { node, player: p, path: solver.tree.nodes[node].path.clone(), actions: actions.iter().map(|a| a.label()).collect(), frequencies, hands })
}

/// Builds, solves and reports a heads-up spot.
pub fn run_hu(config: SpotConfig, algo: Algorithm, max_iterations: u64, target_pct: f64, check_every: u64) -> Result<SolveReport, String> {
    let sw = Stopwatch::start();
    let spot = Spot::from_config(config)?;
    let mut solver = HuSolver::new(spot, algo, None, 2_000_000)?;
    let curve = solve_hu(&mut solver, max_iterations, target_pct, check_every);
    let result = solver.exploitability();
    let root = node_report(&solver, 0)?;
    Ok(SolveReport {
        game: solver.spot.game.code().to_string(),
        board: solver.spot.config.board.clone(),
        hands: vec![solver.hands(0), solver.hands(1)],
        nodes: solver.tree.nodes.len(),
        decision_nodes: solver.tree.decision_nodes(),
        algorithm: algo.name(),
        iterations: solver.iterations,
        curve,
        memory_bytes: solver.memory_bytes(),
        cpu_seconds: sw.process_cpu(),
        wall_seconds: sw.wall(),
        label: if result.constant_sum {
            "Equilibrium approximation of the abstracted game (bet menu); exploitability exact in that game".into()
        } else {
            "Raked game (not constant-sum): NashConv/2 reported; equilibrium approximation of the abstracted game".into()
        },
        result,
        root,
    })
}

/// Aggregates a per-hand row list into the 13x13 class grid (hold'em-style games):
/// for each class, the reach-weighted mean action probabilities.
pub fn class_grid(report: &NodeReport) -> Vec<Option<Vec<f64>>> {
    let k = report.actions.len();
    let mut sums = vec![vec![0.0; k]; pe_cards::NUM_CLASSES];
    let mut w = vec![0.0; pe_cards::NUM_CLASSES];
    for row in &report.hands {
        let cards = pe_cards::parse_cards(&row.hand).unwrap();
        if cards.len() != 2 {
            return vec![None; pe_cards::NUM_CLASSES];
        }
        let c = pe_cards::class_of(cards[0], cards[1]);
        for a in 0..k {
            sums[c][a] += row.weight * row.probs[a];
        }
        w[c] += row.weight;
    }
    (0..pe_cards::NUM_CLASSES).map(|c| if w[c] > 0.0 { Some(sums[c].iter().map(|x| x / w[c]).collect()) } else { None }).collect()
}

#[cfg(test)]
mod tests;
