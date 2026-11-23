# Research reports

| Report | Question |
|---|---|
| [validation.md](validation.md) | Do the solvers reproduce known answers and each other? |
| [cfr-algorithms.md](cfr-algorithms.md) | How do CFR, CFR+, Linear CFR, Discounted CFR, Predictive CFR+ and Monte Carlo CFR compare? |
| [solver-benchmarks.md](solver-benchmarks.md) | Runtime, memory and convergence of range-vs-range solving (NLH, PLO, Short Deck; river and turn) |
| [abstraction.md](abstraction.md) | How much do card buckets and bet menus cost? |
| [multiplayer.md](multiplayer.md) | What happens with three or more players, and how does cost grow? |
| [exploitation.md](exploitation.md) | Equilibrium vs exploitative play against an opponent model (restricted Nash responses) |
| [equity.md](equity.md) | Exact equity, Monte Carlo error and interval coverage, running it twice |
| [opponent-model.md](opponent-model.md) | How well does the Bayesian opponent model recover simulated players' tendencies? |

Raw results are in `../results/`, produced by `scripts/run-benchmarks.ps1` (Rust) and `python -m poker_ecosystem.evaluate` (Python); `scripts/make_reports.py` regenerates these reports and the figures in `../figures/`.
