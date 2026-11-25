# Validation against known answers

The solvers are checked against games whose answers are known, and against independent implementations of themselves.

## Known values

| Benchmark | Known | Computed | Difference |
|---|---:|---:|---:|
| Kuhn poker: value for the first player | -0.055556 | -0.055556 | 5.5e-12 |
| Leduc hold'em: value for the first player | -0.0856 | -0.08561 | exploitability 4.6e-06 |
| Kuhn poker: exploitability of the uniform policy | 0.458333 (OpenSpiel) | 0.458333 | exact |
| Leduc hold'em: NashConv of the uniform policy | 4.747222 (OpenSpiel) | 4.747222 | exact |

## A river with a closed-form equilibrium

The clairvoyance game: the bettor holds the nuts or air (half each), the caller holds bluff-catchers that beat only the air; pot 1, bet *b*. At equilibrium the bettor bets every nut hand and bluffs so that bluffs are *b*/(1+2*b*) of its bets, and the caller calls 1/(1+*b*) of the time. It is solved here as a real card game by the range solver: out of position holds quads (2h with a kicker) or air that plays the board on 2s 2c 2d 7h 7c, in position holds 88 or 99. No card is shared between the two ranges, so the card game is exactly the abstract one.

| Bet | Bluff (closed form) | Bluff (solver) | Call (closed form) | Call (solver) | Value (closed form) | Value (solver) | Iterations |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5x pot | 0.3333 | 0.3334 | 0.6667 | 0.6667 | 0.1667 | 0.1667 | 10,000 |
| 1.0x pot | 0.5000 | 0.4999 | 0.5000 | 0.4999 | 0.2500 | 0.2500 | 10,000 |
| 2.0x pot | 0.6667 | 0.6667 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 10,000 |

## Cross-checks inside the test suite

| Check | What it shows |
|---|---|
| All 2,598,960 five-card and 133,784,560 seven-card hands | The evaluator reproduces the published category counts (7,462 distinct values) |
| All 376,992 five-card short-deck hands | Category counts derived combinatorially (flush 480 < full house 1,728) |
| 200,000 random hands per rule set | Agreement with an independent, deliberately naive evaluator |
| Vector solver vs. generic tabular CFR | For NLH river, NLH turn, PLO river and a 7-2 bounty spot, the range solver's strategy evaluated in an explicitly built game tree gives the same values and exploitability to 1e-9 |
| Compiled CFR vs. a plain reference CFR | Identical exploitability on Leduc after 10 and 40 iterations (difference below 1e-12) |
| Two-player multiway solver vs. heads-up solver | The joint-enumeration solver and the sweep solver agree to 1e-7 |
| Run-it-twice statistics | Exact inclusion-exclusion result equals brute force over ordered pairs of runouts |
| JavaScript engine vs. Rust engine | 26,612 checks: every evaluator value, exact equities, run-it-twice moments, tree sizes, solver values, exploitability and per-hand strategies after the same iterations, the hand-history parser on 412 hands, street-by-street range probabilities (equal to the last bit) and river reviews |

## A bug the validation caught

The first version accumulated each player's average strategy during the regret traversal, which skips subtrees the opponent's current strategy never reaches. That under-weights later iterations at information sets a best responder can still steer into. CFR+ hid it, but the average strategies of Discounted CFR and Predictive CFR+ stalled at an exploitability near 0.12 on Leduc while their current strategies converged. The fix accumulates realization weights from each player's own reach in a separate pass (tabular solver) or walks unreached subtrees during training (vector solvers). After it, DCFR reaches 1.9e-04 at 1,000 iterations on Leduc instead of 0.117.
