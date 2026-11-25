# Hand histories

## Formats

The PokerStars parser (`pe hh parse`, and the same parser in the browser) reads cash games and tournaments: Hold'em (No Limit, Pot Limit, Limit), Omaha, Omaha Hi/Lo, 5 Card Omaha and 6+ Hold'em; antes, dead blinds ("posts small & big blinds"), straddles, all-ins, uncalled bets, main and side pots, run-it-twice sections, and hands shown at showdown or revealed in the summary. Amounts become integer thousandths, so pot arithmetic is exact. Unknown lines are skipped with a warning; nothing aborts a file.

Straddles are all written `posts straddle`; the seat decides the kind. The seat after the big blind (or after the previous straddler, for a re-straddle) posts an ordinary straddle; any other seat, usually the button, posts a **Mississippi straddle**. Its poster acts last before the flop: action starts under the gun, the cutoff passes to the small blind, and the button closes the round after the big blind (`model::preflop_order`). The simulator deals both kinds (`--straddle`, `--mississippi`), and a test checks that the first round of every simulated hand follows that order and that the kind survives a text round trip.

PokerStars does not deal bomb pots or double boards, so these use documented **extensions** (real sites write them differently and need small adapters):

* header tags `[Bomb Pot]` and `[Double Board]`;
* `Name: posts bomb pot ante $X`; a bomb pot has no preflop betting round;
* `*** FLOP 2 ***`, `*** TURN 2 ***`, `*** RIVER 2 ***` for the second board, and `Board 2 [...]` in the summary;
* pots split by board are collected under `*** FIRST SHOW DOWN ***` and `*** SECOND SHOW DOWN ***`, as when a hand is run twice.

The output is the `pe-hh/1` JSON model ([schema](HAND-HISTORY-SCHEMA.md)). The writer turns a hand back into PokerStars text; a test simulates 2,500 hands (with straddles, double-board bomb pots, run-it-twice and rake), writes them out, parses them back and requires every action, post, board, uncalled bet and collection to match.

## Reconstruction

`reconstruct` replays posts and actions, returns uncalled bets, builds main and side pots with their eligible players (money from folded players stays in, a level nobody can win joins the next pot), and checks the result against the summary: contributions add to the stated pot, collections plus rake equal it, nobody collects after folding, nobody puts in more than their stack. All 12 hand-written sample hands and all 20,000 simulated hands reconstruct.

## Street-by-street ranges

`pe hh ranges FILE` (and the lab's *Reconstruct ranges* button) works out what the public actions say about every player's hole cards, at the start of each street and at the end of the hand ([`ranges.rs`](../crates/pe-hh/src/ranges.rs)):

1. Every player starts with the 1,326 combos equally likely; board cards remove the combos they block.
2. Each decision gets a context (opening by position, the big blind's or straddler's option, facing an open or a 3-bet; after the flop a c-bet spot, a lead, any other first bet, facing a bet or a raise). The player's frequencies of fold, check/call and bet/raise in that context are counted across every hand in the file and shrunk towards the population's (a Dirichlet-multinomial posterior mean with 10 pseudo-decisions), which is itself shrunk towards a fixed default.
3. A threshold strategy turns the frequencies into a strategy per combo. Combos are ordered by strength (the preflop equity order; on the flop and turn the exact equity against a random hand over every runout, so draws count; on the river the share of hands beaten) and cut into bands by weight: preflop the weakest fold, the middle calls and the strongest raise; after the flop bets and raises are polarised, value from the top and bluffs from the bottom (bluff shares of 28% for bets and 33% for raises).
4. With probability 0.5 an action follows the player's frequencies regardless of the cards (and with 0.001 it is uniformly random): players slow-play, float and give up. That noise level is fitted by `pe bench ranges` on simulated hands whose cards are all known, maximising the log score of the true hands on half of them.
5. Each action multiplies the actor's combo weights by their likelihood (Bayes' rule).

Each range is reported as a 13 × 13 grid, its effective size (exp of the entropy: "as uncertain as N equally likely combos"), the most likely combos and, when a player's cards are known, how likely the model found them. On 1,903 held-out simulated hands the true hand is on average 2.0 times as likely under the reconstructed range as under a uniform one (+1.0 bit), against 1.5 times (+0.6 bits) with population frequencies only and a disastrous −4.9 bits without the noise. See [hand-histories.md](../research/reports/hand-histories.md).

## River review

`pe hh review FILE [--out reviews.json]` (and the lab's *Review the river*) compares every heads-up river decision with an equilibrium of the river as the hand reached it ([`review.rs`](../crates/pe-hh/src/review.rs)): both players' reconstructed ranges at the start of the river, the hand's board, pot and stacks, and a bet menu of 33% and 75% of the pot, pot-sized raises and all-in plus every size actually used. Discounted CFR solves it to 0.5% of the pot (exploitability computed exactly). For each decision of a player whose cards are known, the review reports that hand's equilibrium strategy, the value of each action when both play the equilibrium afterwards, and the cost of the action taken: the gap to the best action. `python -m poker_ecosystem.hhdb reviews DB reviews.json` loads the results into the database and `hhdb costly DB` lists the most expensive decisions.

The cost is measured in a model (estimated ranges, an abstract bet menu, an opponent who plays the equilibrium); it says an action is clearly worse than an alternative against a balanced opponent, not that it lost money against this one.

## Analysis

* **All-in expected value**: when two or more players are all-in before the river with their cards known, each pot is shared by equity among its eligible players (exact when at most 50,000 runouts remain, otherwise 40,000 seeded Monte Carlo deals). The difference between the actual and the expected result is the luck in the hand.
* **River strength**: for hands known at a river showdown, the share of all opponent holdings they beat (all holdings for hold'em; 3,000 sampled for Omaha). The opponent model uses it to separate value bets from bluffs among river bets that were called.
* **Statistics and leaks** (Python): VPIP, PFR, RFI by position, 3-bet, fold to 3-bet, steals, c-bets and responses by street, donk bets, check-raises, showdown frequencies, bet-size distributions, river bluff share and value threshold, multi-street lines; posteriors with credible intervals; leaks as the probability that a player's frequency lies outside a reference band.
* **Database** (Python, SQLite): idempotent import of JSON/JSONL; queries by player, position, game, pot size, street reached, action, showdown, all-in, board texture; per-player statistics; reviewed river decisions ranked by cost.

## The simulator

`pe hh simulate` deals No-Limit Hold'em cash games between bots of six styles (nit, TAG, LAG, calling station, maniac, weak-tight; parameters jittered per bot). Preflop play uses position-dependent ranges smoothed by a logistic so boundary hands mix; postflop play uses hand strength (share of random holdings beaten, raised to the number of opponents, plus a crude draw bonus) against pot odds with each style's c-bet, stab, lead, bluff and calling parameters. **Every decision records the probabilities the bot gave each action before sampling**, so the true frequency of any statistic is known exactly and the opponent model can be scored against it. The bots model player types; they are not strong players.

## What the corpus is

There is no real hand-history data in this repository: the samples are hand-written and the corpus is simulated. Real players are less stationary and less parametric than the bots, so the opponent-model results are best read as an upper bound on how well the model can do with a given number of hands.
