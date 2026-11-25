# Architecture

```
                 ┌──────────────┐
                 │   pe-cards   │  cards, masks, games, ranges, RNG, CPU/memory probes
                 └──────┬───────┘
          ┌─────────────┼──────────────┐
   ┌──────▼─────┐ ┌─────▼──────┐ ┌─────▼──────┐
   │  pe-eval   │ │   pe-cfr   │ │            │
   │ evaluators │ │ tabular CFR│ │            │
   └──────┬─────┘ │ MCCFR, BR  │ │            │
   ┌──────▼─────┐ └─────┬──────┘ │            │
   │ pe-equity  │       │        │            │
   │ equity,    ├───────┼────────►  pe-hh     │  parser, writer, reconstruction,
   │ sweeps, RIT│       │        │            │  all-in EV, simulator,
   └──────┬─────┘       │        └──▲──┬──────┘  street-by-street ranges,
   ┌──────▼─────────────▼──┐        │  │         river review
   │       pe-solver       ├────────┘  │
   │ trees, vector CFR,    │           │
   │ BR, models, multiway  │           │
   └──────────┬────────────┘           │
         ┌────▼───────────────────────▼────┐
         │            pe-cli (pe)          │  JSON in, JSON out
         └────┬──────────────┬─────────────┘
              │              │ pe vectors
     results, │              ▼
     corpora  │      web/js/engine  (JavaScript port, checked by web/tests/run.mjs)
              ▼              │
   python/poker_ecosystem    ▼
   (stats, Bayes, DB,     web/ (the lab: Web Worker + five tabs)
    clustering, reports)
```

## Layers

* **Rust workspace (`crates/`)**: everything performance-critical. Seven crates with one direction of dependency. The only outside dependencies are serde, serde_json and rayon; evaluator tables, the RNG, the hand-history parser and the benchmark harness are our own, so results are reproducible across dependency upgrades.
* **Command line (`pe`)**: the boundary between the engine and everything else. Every command prints JSON or writes it with `--out`, so Python, scripts and the web build consume the same data.
* **Python (`python/poker_ecosystem`)**: research tooling with no dependencies beyond the standard library: hand statistics, the hierarchical Beta-Binomial opponent model, clustering, a SQLite hand database, leak detection, SVG charts and the model evaluation. It reads the `pe-hh/1` JSON that the Rust parser and simulator write ([schema](HAND-HISTORY-SCHEMA.md)).
* **Web (`web/`)**: the interactive lab, static files with no build step. `web/js/engine` is a line-by-line JavaScript port of the evaluator, equity, tree, vector-CFR solver, opponent-profile mapping, PokerStars parser, street-by-street ranges and river review. Solving and range work run in module Web Workers. The port is tied to the Rust engine by test vectors (`pe vectors`) and 26,612 parity checks (`node web/tests/run.mjs`): identical evaluator values, exact equities to 1e-12, identical trees, solver values, exploitability and per-hand strategies after the same iterations to 1e-7, the parser on 412 hands, range probabilities equal to the last bit, and river reviews to 1e-6.

## Data flow for the research results

1. `pe hh simulate` deals a synthetic corpus with bots of known styles (`data/synthetic/`, regenerated from a seed; not committed).
2. `python -m poker_ecosystem.evaluate` fits the opponent model at several sample sizes and scores it against the bots' true frequencies.
3. `scripts/run-benchmarks.ps1` runs the twelve `pe bench` suites into `research/results/` (`ranges` and `review` regenerate the 20,000-hand corpus in process).
4. `scripts/make_reports.py` turns results into `research/reports/*.md` and `research/figures/*.svg`; `scripts/export_research.py` and `scripts/export_web_data.py` write the lab's `web/data/*.json`.

## Key representations

| Thing | Representation |
|---|---|
| Card | `u8` id `rank * 4 + suit` |
| Card set | `u64` in suit-major layout (bit `suit * 16 + rank`), so the evaluator reads four 13-bit suit masks with shifts |
| Hand value | `u32`: `order << 26 \| category << 22 \| payload`; compares with `>` within one rule set (short deck reorders categories) |
| Range | weighted list of hands, in a fixed order shared by Rust and JavaScript |
| Game tree (tabular) | flat arrays of nodes, children, chance probabilities, utilities and information sets, optionally mapped onto abstract information sets |
| Postflop tree | public betting tree (2-6 players) with chance nodes per dealt card; per decision node, regrets and strategy sums per hand (or per bucket) |
| Amounts in hand histories | integer thousandths of the currency unit |

## Why a JavaScript port instead of WebAssembly

The development machine had no WebAssembly toolchain and nothing was downloaded. A port with parity tests keeps the lab dependency-free and loadable as plain files from the website, and the tests make divergence visible immediately. The cost is two implementations to maintain; the vectors are regenerated with one command.
