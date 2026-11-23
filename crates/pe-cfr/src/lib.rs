//! Game-theoretic solving on compiled extensive-form trees.
//!
//! * [`tree`]: game definitions compile into flat trees, optionally abstracted.
//! * [`solver`]: full-width CFR, CFR+, Linear CFR, Discounted CFR, Predictive CFR+.
//! * [`mccfr`]: external- and outcome-sampling Monte Carlo CFR.
//! * [`br`]: exact best responses, NashConv and two-player exploitability.
//! * [`games`]: Kuhn and Leduc poker for any number of players, and a
//!   clairvoyance river game with a closed-form solution.
//!
//! What the numbers mean: in two-player zero-sum games, exploitability is the
//! exact distance from equilibrium of the computed strategy in the game that was
//! solved. With three or more players the algorithms still run and NashConv is
//! still exact, but no theorem says it goes to zero, and a profile with small
//! NashConv is only an approximate equilibrium of that game, one of possibly many.

pub mod br;
pub mod games;
pub mod mccfr;
pub mod solver;
pub mod tree;

use pe_cards::sys::Stopwatch;
use serde::Serialize;

pub use br::{NashConvReport, best_response, expected_values, nash_conv};
pub use mccfr::{Mccfr, Sampling};
pub use solver::{Algorithm, Cfr};
pub use tree::{GameDef, NodeSpec, Strategy, Tree};

#[derive(Clone, Debug, Serialize)]
pub struct ConvergencePoint {
    pub iteration: u64,
    pub nodes_touched: u64,
    pub cpu_seconds: f64,
    pub nash_conv: f64,
    pub exploitability: Option<f64>,
    pub values: Vec<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RunReport {
    pub game: String,
    pub method: String,
    pub players: usize,
    pub infosets: usize,
    pub abstract_infosets: usize,
    pub nodes: usize,
    pub tree_bytes: usize,
    pub solver_bytes: usize,
    pub curve: Vec<ConvergencePoint>,
}

/// Geometric checkpoints 1, 2, 5, 10, 20, 50, ... up to `max`, plus `max`.
pub fn checkpoints(max: u64) -> Vec<u64> {
    let mut v = Vec::new();
    let mut base = 1u64;
    'outer: loop {
        for m in [1, 2, 5] {
            let x = base * m;
            if x > max {
                break 'outer;
            }
            v.push(x);
        }
        base *= 10;
    }
    if v.last() != Some(&max) {
        v.push(max);
    }
    v
}

/// Runs a full-width algorithm, measuring NashConv at each checkpoint.
pub fn run_cfr(tree: &Tree, algo: Algorithm, marks: &[u64]) -> (Strategy, RunReport) {
    let mut s = Cfr::new(tree, algo);
    let mut curve = Vec::new();
    let sw = Stopwatch::start();
    let mut cpu = 0.0;
    for &m in marks {
        let lap = Stopwatch::start();
        while s.iterations < m {
            s.iterate();
        }
        cpu += lap.thread_cpu();
        let avg = s.average_strategy();
        let r = nash_conv(tree, &avg);
        curve.push(ConvergencePoint { iteration: m, nodes_touched: s.nodes_touched, cpu_seconds: cpu, nash_conv: r.nash_conv, exploitability: r.exploitability, values: r.values });
    }
    let _ = sw;
    let report = RunReport {
        game: tree.name.clone(),
        method: algo.name(),
        players: tree.num_players,
        infosets: tree.infosets.len(),
        abstract_infosets: tree.abs_infosets.len(),
        nodes: tree.nodes.len(),
        tree_bytes: tree.memory_bytes(),
        solver_bytes: s.memory_bytes(),
        curve,
    };
    (s.average_strategy(), report)
}

/// Runs Monte Carlo CFR, measuring NashConv at each checkpoint.
pub fn run_mccfr(tree: &Tree, sampling: Sampling, marks: &[u64], seed: u64) -> (Strategy, RunReport) {
    let mut s = Mccfr::new(tree, sampling, seed);
    let mut curve = Vec::new();
    let mut cpu = 0.0;
    for &m in marks {
        let lap = Stopwatch::start();
        while s.iterations < m {
            s.iterate();
        }
        cpu += lap.thread_cpu();
        let avg = s.average_strategy();
        let r = nash_conv(tree, &avg);
        curve.push(ConvergencePoint { iteration: m, nodes_touched: s.nodes_touched, cpu_seconds: cpu, nash_conv: r.nash_conv, exploitability: r.exploitability, values: r.values });
    }
    let method = match sampling {
        Sampling::External => "ES-MCCFR".to_string(),
        Sampling::Outcome { epsilon } => format!("OS-MCCFR(eps={epsilon})"),
    };
    let report = RunReport {
        game: tree.name.clone(),
        method,
        players: tree.num_players,
        infosets: tree.infosets.len(),
        abstract_infosets: tree.abs_infosets.len(),
        nodes: tree.nodes.len(),
        tree_bytes: tree.memory_bytes(),
        solver_bytes: s.memory_bytes(),
        curve,
    };
    (s.average_strategy(), report)
}

#[cfg(test)]
mod tests;
