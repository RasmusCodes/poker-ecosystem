# Game support and priorities

## Support matrix

| Game or format | Evaluation | Equity | Postflop solving | Hand histories | Notes |
|---|---|---|---|---|---|
| No-Limit Hold'em, heads-up | exact | exact / MC | river and turn, exact card removal | PokerStars | the core case |
| Pot-Limit Omaha (4 cards) | exact (2+3) | exact / MC | river and turn | PokerStars | 15 inclusion-exclusion terms per hand |
| Five-card Omaha | exact | exact / MC | supported by the engine (large ranges are slow) | PokerStars ("5 Card Omaha") | |
| Short Deck 6+ | exact, Triton and classic rules | exact / MC | river and turn | PokerStars ("6+ Hold'em") | A-6-7-8-9 straight; flush > full house |
| Omaha Hi-Lo, Big O | exact low and high | exact / MC | river and turn (three sweeps per showdown) | PokerStars ("Omaha Hi/Lo") | mixed-game components |
| Multiway pots (3-6 players) | – | exact / MC for any number of hands | rivers by exact joint enumeration, NashConv reported | yes | no equilibrium guarantee |
| Run it twice | – | exact mean and variance (any street) | equity-neutral by construction | PokerStars sections | variance reduction quantified |
| Double board | – | exact / MC | rivers (both boards complete) | documented extension | pot split by board |
| Bomb pots | – | multiway flop equity | solving starts postflop anyway | documented extension; simulator deals them | everyone antes, no preflop betting |
| Double-board bomb pots | – | exact / MC | rivers | documented extension; simulator deals them | |
| The 7-2 game | – | – | bounty in every payoff (folds and showdowns) | – | zero-sum side payments |
| Mississippi / button straddles | – | – | preflop structure only | straddle posts parsed; positions from the button | no preflop solver here |
| Mixed games (rotation) | components above | | | | rotation is a scheduling concern; each component is supported |

## Priorities: strategic complexity against feasibility

The order of work followed the value of each format to a research platform and what can be done exactly:

1. **Heads-up NLH postflop**: the reference case. Exact solving and exact exploitability are feasible on the river and turn, which makes it the place to validate every algorithm.
2. **PLO**: the same machinery with four-card blockers; strategically richer (more equity realisation, more nut-blocking) and only slightly harder to solve exactly with the subset-slot sweep.
3. **Multiway pots**: the most strategically different case (no equilibrium guarantee, collusion-free best responses only) and the one where exactness is most expensive; done exactly for small ranges to measure what approximations would cost.
4. **Run it twice**: cheap to do exactly and frequently misunderstood (it changes variance, not equity).
5. **Double boards and bomb pots**: payoff and dealing changes on top of existing solvers; the strategic novelty is ranges that must play on two boards at once.
6. **The 7-2 game**: a single payoff rule with a surprising equilibrium effect (the defender calls wider, which can make 72o bluff *less*).
7. **Short Deck**: an evaluator change; the 36-card deck increases equity running closer, which the solver handles unchanged.
8. **Straddles and mixed games**: mostly preflop structure and scheduling, outside the postflop solver's scope; supported in parsing, positions and evaluation.
