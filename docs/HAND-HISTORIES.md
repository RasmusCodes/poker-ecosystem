# Hand histories

## Formats

The PokerStars parser (`pe hh parse`, and the same parser in the browser) reads cash games and tournaments: Hold'em (No Limit, Pot Limit, Limit), Omaha, Omaha Hi/Lo, 5 Card Omaha and 6+ Hold'em; antes, dead blinds ("posts small & big blinds"), straddles, all-ins, uncalled bets, main and side pots, run-it-twice sections, and hands shown at showdown or revealed in the summary. Amounts become integer thousandths, so pot arithmetic is exact. Unknown lines are skipped with a warning; nothing aborts a file.

PokerStars does not deal bomb pots or double boards, so these use documented **extensions** (real sites write them differently and need small adapters):

* header tags `[Bomb Pot]` and `[Double Board]`;
* `Name: posts bomb pot ante $X`; a bomb pot has no preflop betting round;
* `*** FLOP 2 ***`, `*** TURN 2 ***`, `*** RIVER 2 ***` for the second board, and `Board 2 [...]` in the summary;
* pots split by board are collected under `*** FIRST SHOW DOWN ***` and `*** SECOND SHOW DOWN ***`, as when a hand is run twice.

The output is the `pe-hh/1` JSON model ([schema](HAND-HISTORY-SCHEMA.md)). The writer turns a hand back into PokerStars text; a test simulates 2,500 hands (with straddles, double-board bomb pots, run-it-twice and rake), writes them out, parses them back and requires every action, post, board, uncalled bet and collection to match.

## Reconstruction

`reconstruct` replays posts and actions, returns uncalled bets, builds main and side pots with their eligible players (money from folded players stays in, a level nobody can win joins the next pot), and checks the result against the summary: contributions add to the stated pot, collections plus rake equal it, nobody collects after folding, nobody puts in more than their stack. All 11 hand-written sample hands and all 20,000 simulated hands reconstruct.

## Analysis

* **All-in expected value**: when two or more players are all-in before the river with their cards known, each pot is shared by equity among its eligible players (exact when at most 50,000 runouts remain, otherwise 40,000 seeded Monte Carlo deals). The difference between the actual and the expected result is the luck in the hand.
* **River strength**: for hands known at a river showdown, the share of all opponent holdings they beat (all holdings for hold'em; 3,000 sampled for Omaha). The opponent model uses it to separate value bets from bluffs among river bets that were called.
* **Statistics and leaks** (Python): VPIP, PFR, RFI by position, 3-bet, fold to 3-bet, steals, c-bets and responses by street, donk bets, check-raises, showdown frequencies, bet-size distributions, river bluff share and value threshold, multi-street lines; posteriors with credible intervals; leaks as the probability that a player's frequency lies outside a reference band.
* **Database** (Python, SQLite): idempotent import of JSON/JSONL; queries by player, position, game, pot size, street reached, action, showdown, all-in, board texture; per-player statistics.

## The simulator

`pe hh simulate` deals No-Limit Hold'em cash games between bots of six styles (nit, TAG, LAG, calling station, maniac, weak-tight; parameters jittered per bot). Preflop play uses position-dependent ranges smoothed by a logistic so boundary hands mix; postflop play uses hand strength (share of random holdings beaten, raised to the number of opponents, plus a crude draw bonus) against pot odds with each style's c-bet, stab, lead, bluff and calling parameters. **Every decision records the probabilities the bot gave each action before sampling**, so the true frequency of any statistic is known exactly and the opponent model can be scored against it. The bots model player types; they are not strong players.

## What the corpus is

There is no real hand-history data in this repository: the samples are hand-written and the corpus is simulated. Real players are less stationary and less parametric than the bots, so the opponent-model results are best read as an upper bound on how well the model can do with a given number of hands.
