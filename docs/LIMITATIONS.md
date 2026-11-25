# Limitations

## Solving

* **No preflop or flop solver.** Postflop solving starts from given turn or river ranges. Flop trees are supported by the tree builder but are expensive without isomorphism and turn/river abstraction; the [NLH solver](https://rasmuscodes.github.io/projects/nlh-solver/) and [PLO solver](https://rasmuscodes.github.io/projects/plo-solver/) projects cover full flops and preflop.
* **Bet menus are an abstraction.** Solutions are equilibria of the game with the listed sizes. Their exploitability against an opponent using other sizes is not bounded here; off-tree translation is not implemented.
* **Card buckets** (optional) are equal-weight strength buckets on the river only; their cost is measured, but they are not a state-of-the-art abstraction (no potential-aware or imperfect-recall flop abstraction).
* **Multiway solving** is exact only for modest ranges (cost is the product of the range sizes) and carries no equilibrium guarantee. NashConv is computed exactly, but equilibrium selection is not addressed: a different algorithm or seed could converge elsewhere.
* **Hi-lo and double-board** solving needs complete boards (rivers) for double boards; the 7-2 bounty is modelled as paid whenever a pot is won with 7-2 offsuit (shown), the most common house rule.
* **Rake** makes the game non-zero-sum; NashConv/2 is still reported, labelled as such.

## Opponent modelling

* The model is **validated on simulated bots** with stationary, parametric strategies. Real opponents adapt, tilt and mix; expect wider errors.
* A single Beta prior per statistic over-shrinks players when the population is a few sharply distinct types; the evaluation report quantifies this.
* Turning frequencies into a strategy uses **threshold play on hand strength** (strongest hands raise, weakest fold, value from the top, bluffs from the bottom). Real players do not sort their range perfectly; the exploit is only as good as this mapping.
* Strength on the turn ignores draws when building the model's strategy.

## Hand histories

* PokerStars is the supported site format. Bomb pots and double boards use documented extensions; other sites need adapters.
* Straddles are told apart by seat: the seat after the big blind (or after the previous straddler) posts an ordinary straddle, any other seat a Mississippi straddle, which acts last before the flop. House rules that let other seats straddle with different action orders need an adapter.
* There is no real hand-history data in the repository: samples are hand-written and the research corpus is simulated.
* **Street-by-street ranges are a model.** They follow from Bayes' rule with a threshold model of play (strongest hands raise, polarised bets after the flop) driven by each player's estimated frequencies, plus noise for slow-plays and floats. The noise level is fitted on simulated bots; with it, the true hand lies inside the 50%, 80% and 90% credible regions 57%, 87% and 94% of the time on held-out simulated hands: a little conservative, most of all at the start of the flop, and close to nominal by the end of a hand. Real players may differ from the bots in ways the fit does not capture. Ranges are reconstructed for hold'em only, from an observer's point of view (a hero's own cards are not removed from the opponents' ranges), and double-board hands rank hands on the first board.
* **The river review is a review against a model.** It covers hold'em rivers played heads-up on one board. A decision's cost is measured against an opponent who plays the equilibrium of the river with the reconstructed ranges and a bet menu containing the sizes used; it is not money lost against the actual opponent, and earlier streets are not reviewed (turn solves take about a minute in the browser). Decisions the equilibrium would rarely reach (the player's own line under 5% or the opponent's under 2%) are flagged as off the path: CFR explores them little, so their strategies and values are rough even when the solve's exploitability is small.

## Engineering

* The browser engine is a JavaScript port, not WebAssembly: about 3-10× slower than the Rust engine. Turn spots take about a minute in a browser.
* Benchmarks ran on a shared laptop at below-normal priority; CPU seconds are reported, but absolute timings are noisy. Convergence is compared in iterations and nodes touched.
