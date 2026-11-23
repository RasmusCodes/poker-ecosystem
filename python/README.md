# poker_ecosystem: Python research layer

Opponent statistics, a hierarchical Bayesian opponent model, clustering, a SQLite hand
database, a leak finder, SVG charts and a ground-truth evaluation. It reads the `pe-hh/1`
hand-history format (`docs/HAND-HISTORY-SCHEMA.md`). Python 3.12, **standard library only**.

```powershell
# tests, from the repository root
python -m unittest discover -s python/tests -t python

# command-line tools (put python/ on the path first)
$env:PYTHONPATH = "python"
python -m poker_ecosystem                                    # overview
python -m poker_ecosystem.hhdb import hands.sqlite data\*.jsonl
python -m poker_ecosystem.hhdb query hands.sqlite --player Alice --street river --action river:raise
python -m poker_ecosystem.hhdb stats hands.sqlite --player Alice --by-position --json
python -m poker_ecosystem.hhdb players hands.sqlite --min-hands 500
python -m poker_ecosystem.evaluate --hands sim.jsonl --players players.json
```

| module | purpose |
|---|---|
| `hh` | load `.json`/`.jsonl`, positions and groups (EP/MP/LP/BLINDS), big-blind conversion, replay: pot before each action, amount to call, raises so far, preflop aggressor |
| `stats` | per-player counters (opportunities, successes) per position and pooled, sizing bins, river showdown strengths, multi-street transitions |
| `bayes` | incomplete beta, Beta quantiles, Beta-binomial, empirical-Bayes prior fits, Dirichlet-multinomial posteriors |
| `opponent` | hierarchical model, credible intervals, preflop ranges, `exploit_profile()` for river solver locks |
| `ranges` | the 169 hand classes; `research/vectors/preflop_ranking.json` if present, else the Chen formula |
| `cluster` | k-means++, diagonal Gaussian mixture with BIC, adjusted Rand index, heuristic style names |
| `hhdb` | SQLite database (hands, players, hand_players, actions, boards, stats_cache), idempotent import, filtered queries, CLI |
| `leaks` | P(true frequency outside a reference band), heuristic EV-impact ranking, all-in luck |
| `svg` | line (log axes), grouped bars with error bars, scatter, histogram; `currentColor` ink, light/dark palette |
| `evaluate` | error vs hands observed, interval calibration, shrinkage benefit, style recovery; JSON, SVG and markdown outputs |

## Statistic definitions

Counts are per player, per position (the player's position in that hand) and pooled. A
*decision* is one entry of a street's `actions`. Preflop the blinds and straddles are not
raises: the open is the first raise, the 3-bet the second. At most one opportunity per
player, statistic and hand, except `af_*` (one per decision). Preflop statistics need a
preflop betting round, so bomb pots have none.

| key | opportunity | success |
|---|---|---|
| `vpip` | had at least one preflop decision (walks have none) | any call, bet or raise |
| `pfr` | as `vpip` | any bet or raise |
| `rfi` | first preflop decision, every earlier preflop action a fold | raise |
| `limp` | first preflop decision, no raise yet, positive amount to call | call |
| `cold_call` | first preflop decision facing exactly one raise | call |
| `three_bet` | first preflop decision facing exactly one raise, made by another player | raise |
| `fold_to_three_bet` | made the open; next own decision faces exactly two raises | fold |
| `four_bet` | first preflop decision facing exactly two raises, the last by another player | raise |
| `steal` | `rfi` opportunity in CO, BTN or SB | raise |
| `fold_to_steal` | SB/BB first decision facing one RFI raise from CO/BTN/SB, only folds since | fold |
| `cbet_flop`, `cbet_turn`, `cbet_river` | preflop aggressor's first own decision on the street with no bet yet; the turn needs a flop c-bet, the river a turn c-bet | bet |
| `fold_vs_cbet_flop`, `fold_vs_cbet_turn`, `fold_vs_cbet_river`; `call_vs_cbet_flop`, `call_vs_cbet_turn`, `call_vs_cbet_river`; `raise_vs_cbet_flop`, `raise_vs_cbet_turn`, `raise_vs_cbet_river` | first own decision on the street facing a bet while the c-bet is the only bet | fold / call / raise |
| `donk_flop`, `donk_turn`, `donk_river` | first own decision, no bet yet, before the previous street's last aggressor (still in, not all-in) acts | bet |
| `check_raise_flop`, `check_raise_turn`, `check_raise_river` | checked earlier on the street, then at the first decision facing a bet could raise | raise |
| `bet_unopened_flop`, `bet_unopened_turn`, `bet_unopened_river` | first own decision on the street with no bet yet | bet |
| `fold_vs_bet_flop`, `fold_vs_bet_turn`, `fold_vs_bet_river`; `call_vs_bet_flop`, `call_vs_bet_turn`, `call_vs_bet_river`; `raise_vs_bet_flop`, `raise_vs_bet_turn`, `raise_vs_bet_river` | first own decision on the street facing a bet, exactly one bet and no raise so far | fold / call / raise |
| `af_flop`, `af_turn`, `af_river` | every own bet, raise, call or fold on the street (checks excluded) | bet or raise |
| `wtsd` | saw the flop (all-in run-outs and bomb pots included) | reached showdown |
| `wsd` | reached showdown | collected a positive amount |
| `river_bluff` | own river bet shown at showdown with `derived.river_strength` | strength < 0.5 (configurable) |
| `xr_flop_bet_turn` | check-raised the flop; first turn decision with no bet yet | bet |

Also kept: bet and raise sizes in pot-fraction bins `<40`, `40-60`, `60-85`, `85-115`, `>115`
(half-open, exact integer comparisons) and `allin`; the value-bet threshold (lower 10% quantile
of shown value-bet strength); and line transitions per street (`flop:xr -> turn:b`, c-bet chains).
Shown river bets describe *called* bets only. Full definitions, including how each statistic is
compared with synthetic `truth` ("decision", "compensator" or not comparable), are in the
`stats` module docstring.

## Model and approximations

Population Beta priors per statistic and position are fitted by maximum marginal likelihood
(guarded Newton on logit mean and log concentration, golden-section fallback) plus a weak
boundary-avoiding Gamma(2) prior on the between-player SD (Chung et al. 2013), which stops the
fit collapsing to "no spread" when a small sample cannot detect it (`boundary_avoiding=False`
gives pure ML). A player's position-level prior is centred on the population-position mean
shifted by the player's leave-one-position-out logit deviation; its logit variance is the
deviation's own uncertainty (delta method) plus a residual variance fitted across players.
These are empirical-Bayes point estimates: hyperparameter uncertainty is ignored, and a single
Beta prior over-shrinks players in a pool of sharply distinct styles. On the test-only toy
corpus (24 bots, 4 styles) pooled 90% intervals covered 79-87% of true frequencies for 50-500
hands. Ranges assume a top-x% structure. The EV-impact rank in `leaks` is a heuristic for
ordering, not an EV estimate. Reports repeat these caveats.

## Assumptions about producers of `pe-hh/1`

* `sb`, `bb`, `straddle` and `mississippi` posts are live (they count towards the preflop
  street contribution); `ante`, `bomb` and `dead` are dead money.
* `amount` is authoritative for chip accounting; `streets` hold the betting rounds in order and
  bomb pots have no `preflop` entry; boards are used to detect all-in run-outs.
* Keys of seat-indexed objects in `derived` are seat numbers as strings.
* Synthetic hands carry `truth` on every action; probabilities are normalised by the reader.
* `players.json` is a list of `{name, style, profile}`.
