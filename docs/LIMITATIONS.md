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
* There is no real hand-history data in the repository: samples are hand-written and the research corpus is simulated.
* The street-by-street analysis covers reconstruction, all-in EV, river strength and statistics; decision review against solver output is done in the lab by loading a spot into the solver, not automatically for every hand of a database.

## Engineering

* The browser engine is a JavaScript port, not WebAssembly: about 3-10× slower than the Rust engine. Turn spots take about a minute in a browser.
* Benchmarks ran on a shared laptop at below-normal priority; CPU seconds are reported, but absolute timings are noisy. Convergence is compared in iterations and nodes touched.
