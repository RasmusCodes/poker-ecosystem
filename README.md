# Poker Ecosystem

A poker research platform: exact hand evaluation and equity, a family of game-theoretic solvers with exact error measurement, range-vs-range postflop solving, Bayesian opponent modelling, exploitative play against a model, hand-history analysis with street-by-street ranges and an automatic river review, and an interactive lab that runs in the browser.

**[Open the interactive lab](https://rasmuscodes.github.io/apps/poker-ecosystem/)** · [Project write-up](https://rasmuscodes.github.io/projects/poker-ecosystem/) · [Research reports](research/reports/README.md)

It is built to say plainly what each number is. Every solver result carries its error: exploitability for two players, NashConv for more. Every approximation is labelled as one.

## What is solved, and how exactly

| Result | Status | How it is measured |
|---|---|---|
| Hand values (hold'em, short deck, Omaha, Omaha 8-or-better) | **Exact** | All 133,784,560 seven-card hands reproduce the published category counts; short-deck counts derived combinatorially; agreement with an independent naive evaluator |
| Equity by enumeration | **Exact** | Every runout enumerated (any number of players, one or two boards, hi-lo splits) |
| Equity by Monte Carlo | **Approximate**, with standard errors | 95% intervals cover the exact value at the nominal rate ([equity.md](research/reports/equity.md)) |
| Run-it-twice mean and variance | **Exact** | Inclusion-exclusion over runout subsets, checked against brute force |
| Kuhn, Leduc, clairvoyance river | **Converged to the known solution** | Kuhn value −1/18 to 5e-12; Leduc −0.0856; closed-form river frequencies to 1e-4 |
| Heads-up river and turn solutions | **ε-equilibrium of an abstracted game** | The bet menu is the abstraction; exploitability inside that game is computed exactly by a vector best response and reported in % of the pot (typically 0.03-0.1%) |
| Card-bucketed solutions | **Abstraction**, error measured | Strategies evaluated in the real game: 16% of the pot with 2 buckets, 0.16% with 40 ([abstraction.md](research/reports/abstraction.md)) |
| Multiway (3-4 player) river solutions | **CFR approximation, no Nash guarantee** | NashConv computed exactly with exact card removal; a small value certifies an approximate equilibrium of this game only, one of possibly many ([multiplayer.md](research/reports/multiplayer.md)) |
| Exploitative strategies | **Best response / restricted Nash response to a model** | Value against the model and own exploitability both reported ([exploitation.md](research/reports/exploitation.md)) |
| Opponent models | **Bayesian estimates** with 90% credible intervals | Calibrated on simulated players with known true frequencies ([opponent-model.md](research/reports/opponent-model.md)) |
| Profile → strategy mapping | **Heuristic model** | Threshold play on the reaching range; a modelling assumption, stated as such |
| Street-by-street ranges | **Bayesian estimate under a model of play** | Scored on simulated hands with known cards: the true hand is twice as likely as under a uniform range (+1.0 bit), and 90% regions contain it 94% of the time ([hand-histories.md](research/reports/hand-histories.md)) |
| River review | **Equilibrium of an abstracted river with estimated ranges** | Exploitability of each solve computed exactly; a decision's cost assumes an opponent playing that equilibrium |

## Headline results

All numbers below are measured, from [`research/results`](research/results); see the reports for method and context.

* **Evaluator:** 15.6 ns per seven-card hold'em hand on one thread; Omaha (60 two-plus-three combinations) 677 ns.
* **CFR family on Leduc:** after 1,000 iterations, vanilla CFR is at 1.2e-2 chips per hand, CFR+ 2.5e-4, Discounted CFR 1.9e-4; DCFR reaches 9.6e-6 at 5,000.
* **River solving (Rust):** Discounted CFR reaches 0.1% of the pot in about 150 iterations on hold'em, Short Deck and Omaha rivers, a fraction of a second for hold'em ranges and 35 CPU-seconds for a PLO river with 32,352 × 41,844 hands.
* **Abstraction:** card buckets cost 16% of the pot at 2 buckets and 0.16% at 40, against 0.003% for the exact solve; merging jacks and queens in Leduc leaves the "solved" strategy 0.75 chips per hand exploitable.
* **More players:** exact multiway rivers cost about twelve times more per added player (280 → 3,490 → 38,885 joint hands); NashConv still falls to about 0.03% of the pot in 300 iterations, an observation rather than a guarantee.
* **Exploitation:** against a button that folds 70% to river bets, a restricted Nash response with p = 0.5 wins 37% of the pot more than equilibrium play and remains exploitable for 20% of the pot; the pure best response wins 42% more but is exploitable for 45%.
* **Opponent model:** on 24 simulated players, the posterior error falls from 0.077 (50 hands each) to 0.019 (2,500 hands), half the error of raw frequencies; 90% intervals contain the truth 88-93% of the time; clustering recovers the six true styles (adjusted Rand index 0.95).
* **Hand histories:** 20,000 simulated hands (straddles, bomb pots, double boards, run-it-twice, rake) all reconstruct to the cent and survive a PokerStars text round trip; Mississippi straddles are told from ordinary ones by seat and act last before the flop; the browser parser matches the Rust parser field by field.
* **Street-by-street ranges:** on held-out simulated hands the reconstructed ranges make the true hand 2.0 times as likely as a uniform range (+1.0 bit), against 1.5 times with population frequencies only; with the fitted noise level the 50/80/90% regions contain the true hand 57/87/94% of the time, where pure threshold play manages 29/45/51%.

## Components

| Path | What it is |
|---|---|
| `crates/pe-cards` | Cards, bit masks, game variants, hold'em range notation, Omaha hand patterns, preflop ranking, seeded RNG, CPU-time and memory probes |
| `crates/pe-eval` | Exact evaluators: hold'em 5-7 cards, short deck (Triton and classic rules), Omaha two-plus-three, ace-to-five 8-or-better low |
| `crates/pe-equity` | Exact and Monte Carlo equity, ranges, double boards, hi-lo, the O(n) card-removal showdown sweep, exact run-it-twice statistics |
| `crates/pe-cfr` | Compiled game trees; CFR, CFR+, Linear CFR, Discounted CFR, Predictive CFR+; external- and outcome-sampling MCCFR; exact best response, NashConv, exploitability; Kuhn and Leduc for N players, a clairvoyance river; infoset abstraction |
| `crates/pe-solver` | Range-vs-range postflop solving: betting trees for 2-6 players, heads-up vector CFR with exact card removal (hold'em, Omaha, Short Deck, Omaha hi-lo, double boards, rake, the 7-2 bounty), node locking, opponent models, best and restricted Nash responses, river card buckets; exact-enumeration multiway rivers |
| `crates/pe-hh` | The `pe-hh/1` hand model, PokerStars parser and writer (ordinary and Mississippi straddles), pot reconstruction with side pots and rake, all-in equity, river showdown strength, a simulator of bots with known styles, street-by-street Bayesian ranges, and an automatic review of heads-up river decisions against an equilibrium |
| `crates/pe-cli` | The `pe` command: evaluation, equity, run it twice, solving, hand histories, benchmarks, test vectors |
| `python/poker_ecosystem` | Research tooling (standard library only): statistics, hierarchical Beta-Binomial opponent model, clustering, SQLite hand database, leak finder, SVG charts, model evaluation |
| `web/` | The interactive lab: a JavaScript port of the engine (checked against the Rust engine to the last digit), Web Worker solving, five tabs |
| `research/` | Benchmark results, figures and reports; test vectors for the JavaScript engine |
| `docs/` | [Architecture](docs/ARCHITECTURE.md), [algorithms](docs/ALGORITHMS.md), [games](docs/GAMES.md), [hand histories](docs/HAND-HISTORIES.md), [schema](docs/HAND-HISTORY-SCHEMA.md), [limitations](docs/LIMITATIONS.md) |

## Quick start

Requirements: Rust 1.85+ (edition 2024), Python 3.10+, and Node 18+ for the JavaScript tests. The Rust engine depends only on serde, serde_json and rayon; the Python package and the web lab have no dependencies.

```sh
cargo test --workspace                 # 74 Rust tests, including exhaustive evaluator checks
cargo build --release                  # builds target/release/pe

pe eval AsKsQsJsTs                     # "a Royal Flush"
pe equity --hands "AhKh QsQd" --board Th7h2c
pe rit --hands "AhKh QsQd" --board Th7h2c          # exact run-it-twice statistics
pe cfr --game leduc --algo dcfr --iters 2000       # convergence curve with exact exploitability
pe solve research/spots/srp-river.json --iters 1000 --target 0.1
pe hh simulate --hands 20000 --seed 1 --dir data/synthetic
pe hh parse data/samples/pokerstars-sample.txt --jsonl hands.jsonl
pe hh ranges data/samples/pokerstars-sample.txt --id 300000000012   # street-by-street ranges
pe hh review data/synthetic/hands.jsonl --limit 2000 --out reviews.json   # river decisions vs equilibrium
pe bench river --out research/results/bench-river.json

python -m unittest discover -s python/tests -t python      # 109 tests
python -m poker_ecosystem.evaluate --hands data/synthetic/hands.jsonl --players data/synthetic/players.json
python -m poker_ecosystem.hhdb reviews hands.sqlite reviews.json && python -m poker_ecosystem.hhdb costly hands.sqlite
node web/tests/run.mjs                 # 26,612 parity checks against the Rust engine
python -m http.server 8766 --directory web                # the lab at http://localhost:8766
```

Regenerating everything: `scripts/run-benchmarks.ps1`, then `python scripts/make_reports.py`, `python scripts/export_research.py`, `python scripts/export_web_data.py`, `pe vectors` and `pe hh review web/data/hands-sample.json --out web/data/hands-sample-reviews.json` (the lab's precomputed river reviews of its simulated sample).

## Design notes

* **One engine, two languages.** The browser lab is a port of the Rust engine rather than WebAssembly (the development machine had no WebAssembly toolchain). `pe vectors` writes evaluator values, exact equities, trees, solver results, street-by-street ranges and river reviews; `web/tests/run.mjs` checks the port against them: 26,612 checks, range probabilities equal to the last bit and everything else to 1e-6 or better.
* **Deterministic.** Every random choice flows from an explicit seed; DCFR's discount powers use correctly rounded operations so Rust and JavaScript agree bit for bit.
* **Measured, not assumed.** Benchmarks report CPU seconds (the development laptop is shared) and deterministic quantities: iterations, nodes touched, exploitability.

## Limitations

The short version (full list in [docs/LIMITATIONS.md](docs/LIMITATIONS.md)): postflop solving starts on the turn or river with given ranges; there is no preflop or flop solver here (see the NLH and PLO solver projects below); bet menus are an abstraction whose cost against off-menu sizes is not bounded; multiway solving is exact only for small ranges and carries no equilibrium guarantee; the opponent model and the street-by-street ranges are validated on simulated players, who are more stationary than people; the river review measures costs against an equilibrium opponent with estimated ranges, and covers heads-up hold'em rivers only; hand-history support covers PokerStars and documented extensions for bomb pots and double boards; stud and draw games are not covered.

## Related projects

* [No-Limit Hold'em solver](https://rasmuscodes.github.io/projects/nlh-solver/): full-game heads-up flops, preflop push/fold and ICM.
* [Pot-Limit Omaha solver](https://rasmuscodes.github.io/projects/plo-solver/): exact flops and a pre-solved library.
* [Figgie bot](https://github.com/RasmusCodes/figgie-bot): Bayesian trading in Jane Street's card game.
* [Game Tree Studio](https://github.com/RasmusCodes/game-tree-studio): draw extensive-form games.

## License

MIT. See [LICENSE](LICENSE).
