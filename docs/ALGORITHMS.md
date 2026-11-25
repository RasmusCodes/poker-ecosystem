# Algorithms

## Evaluation

The seven-card evaluator works directly on the four 13-bit suit masks. With at most seven cards a flush excludes quads and full houses (they would need eight cards), so a suit with five or more cards settles the hand (straight flush or flush). Otherwise, masks of ranks held at least two, three and four times come from bit operations (`two = s0&s1 | s2&s3 | (s0|s1)&(s2|s3)` and similar), and the best category and kickers follow. Short deck changes two things: A-6-7-8-9 is the lowest straight, and the category order (flush above full house; under Triton rules trips above straight), applied through an order table so a hand holding both trips and a straight takes whichever the rules rank higher. Omaha evaluates the 6 × 10 two-plus-three combinations; the ace-to-five low takes, for each pair of distinct low hole ranks, the three lowest remaining board low ranks.

## Equity

Exact equity enumerates every completion of the board(s) in parallel; Monte Carlo samples runouts with a seeded generator and reports standard errors. Ranges are sampled with exact card removal by rejecting colliding deals. Hi-lo splits half the pot to the best qualifying low (the high hand scoops when none qualifies); double boards split the pot by board.

**Run it twice, exactly.** The second run is dealt from the cards the first left, so each run is marginally a uniform runout and the expected share equals the one-run equity. The variance needs the cross moment over ordered pairs of disjoint runouts. With `A(S)` the sum of shares over runouts containing the card set `S`, the sum over runouts disjoint from `r` is `Σ_{S ⊆ r} (-1)^|S| A(S)`, so the cross moment costs `2^k` lookups per runout (k cards to come) instead of a quadratic loop: a preflop all-in is exact.

## The CFR family

All tabular variants use alternating updates on a compiled tree.

| Algorithm | Regret update | Average strategy weight |
|---|---|---|
| CFR | `R += r` | 1 |
| CFR+ | `R = max(R + r, 0)` | t |
| Linear CFR | `R += t·r` | t |
| Discounted CFR (1.5, 0, 2) | positive `R` × t^1.5/(t^1.5+1), negative × 1/2, then `+ r` | sums × (t/(t+1))² |
| Predictive CFR+ | `R = max(R + r, 0)`; strategy from `max(R + r_prev, 0)` | t² |

The average strategy accumulates each player's **realization weight** (product of their own action probabilities to the information set) times the current strategy. It must be computed from the player's own reach in a separate pass: accumulating it inside the regret traversal, which skips subtrees the opponent's current strategy never reaches, under-weights later iterations at information sets a best responder can still steer into. That bug stalled DCFR and PCFR+ near 0.12 chips per hand on Leduc before it was found ([validation.md](../research/reports/validation.md)).

**Monte Carlo CFR.** External sampling explores all of the updating player's actions and samples chance and opponents, updating the average strategy at sampled opponent nodes. Outcome sampling follows one trajectory with ε-exploration (ε = 0.6) and importance weights, with stochastically weighted averaging (Lanctot 2013, Algorithm 3).

**Best response, NashConv, exploitability.** A best response for player *i* picks, at each of *i*'s information sets, the action maximising `Σ_{h∈I} π_{-i}(h)·v_i(h·a)`, deepest sets first (memoised recursion; perfect recall prevents cycles). NashConv = Σ_i (BR_i − u_i). For two-player zero-sum games exploitability = NashConv / 2. With three or more players NashConv is reported and no convergence claim is made.

**Abstraction.** Information-set keys can be mapped onto abstract keys (for example merging card ranks); CFR learns one strategy per abstract set, and best responses are always computed in the real game with each real set playing its abstract set's strategy.

## Range-vs-range solving (vector CFR)

A traversal carries one reach probability per opponent hand and returns one counterfactual value per traverser hand. At a showdown with pot `Q` (rake removed), contribution `m` and opponent reach `w`, the value of hand `h` is `Q·share(h) − m·total(h)`, where `share` and `total` come from the **card-removal sweep**: hands are sorted by strength once per board; each call walks both lists accumulating opponent weight below and at the hero's strength, and removes the weight of opponent hands that collide with `h` by inclusion-exclusion over the non-empty subsets of `h`'s cards (`W(S)` = weight of opponent hands containing every card of `S`):

`blocked(h) = Σ_{S ⊆ h, S ≠ ∅} (−1)^(|S|+1) · W(S)`

Two-card hands need three subset terms, Omaha hands fifteen; every subset that occurs in both ranges gets a dense slot, so a sweep is linear array arithmetic. Hi-lo needs three sweeps (high, low, and high among hands without a low); double boards one per board; the 7-2 bounty two more (the winner's 72o and the opponent's 72o).

At a chance node each child card zeroes the hands containing it and the children's values are averaged over the cards possible for a given pair of hands. Exploitability is computed exactly by a vector best response (maximum over actions per hand at the responder's nodes).

**Multiway rivers** enumerate every card-compatible combination of one hand per player once; a terminal's value for a hand sums, over combinations containing it, the other players' reach times the pot share it wins (side pots included). This is exact, including card removal by folded players, and costs the product of the range sizes.

## Exploitation

* **Opponent model.** A profile (fold to a bet, raise a bet, fold to a raise, bet when checked to, lead, bluff share, preferred size) overrides the modelled player's equilibrium strategy at the nodes it covers with threshold play on the range reaching each node: raise the strongest share, fold the weakest, call with the rest; bet value from the top and bluffs from the bottom. Strength is the high-hand value on the current board.
* **Best response to the model**: the maximally exploitative strategy (value against the model, and unbounded exploitability in principle).
* **Restricted Nash response** (Johanson, Zinkevich and Bowling 2007): a chance event the exploiter cannot see makes the opponent play the model with probability *p* and freely otherwise. Implemented by carrying two opponent reach vectors (model branch and free branch); terminal values are linear in opponent reach, so the exploiter's values use their sum. Sweeping *p* from 0 (equilibrium) to 1 (best response) traces the frontier between exploitation and safety.

## Bayesian opponent model

Each statistic (VPIP, 3-bet, fold to a c-bet, check-raise, fold to a river bet, …) is a count of successes in opportunities with a precise definition ([python/README.md](../python/README.md)). A population Beta prior is fitted by maximum marginal likelihood with a weak penalty on the between-player spread; position-level priors are centred on the population at that position shifted by the player's overall deviation on the logit scale; posteriors are Beta with equal-tailed 90% intervals. Bet sizes use Dirichlet-multinomial posteriors. Clustering runs k-means and a diagonal Gaussian mixture (BIC selects *k*) on standardised logit posterior means.

## Street-by-street ranges

A player's range is a posterior over the 1,326 hold'em combos given the public actions. With prior weights *w(c)* (uniform, then zero for combos the board blocks), an action *a* in context *x* updates

*w(c) ← w(c) · [(1 − λ − ε) σ(c, a) + λ f(a) + ε / k]*,

where *f* are the player's frequencies of fold, check/call and bet/raise in context *x* (counts shrunk towards the population with a Dirichlet-multinomial posterior mean), *σ* is a threshold strategy that cuts the current range, ordered by strength, into consecutive bands with those frequencies (polarised after the flop: bluffs at the bottom, then folds, calls, value at the top), *k* is the number of available actions, *λ* = 0.5 the share of actions that carry no information about the cards and *ε* = 0.001 a uniform tremble. Strength is the preflop equity order before the flop, the exact equity against a random hand over every remaining runout on the flop and turn (computed with a counting sweep: for each runout, combos are sorted by hand value and each combo's wins and ties against all other combos come from running totals per card, with exact card removal), and the share of hands beaten on the river. *λ* maximises the mean log probability of the true hands on simulated hands (`pe bench ranges`), and the posterior credible regions are checked for calibration on held-out hands.

## River review

The two reconstructed river ranges, the hand's pot and stacks and a menu containing the sizes used define a heads-up river game, solved by Discounted CFR to 0.5% of the pot. Walking the actions taken through its tree gives, for each known hand, the equilibrium strategy and the counterfactual value of every action under the equilibrium continuation; the cost of a decision is the best action's value minus the value of the action taken.
