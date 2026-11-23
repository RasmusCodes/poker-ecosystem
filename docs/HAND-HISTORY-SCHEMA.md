# Hand-history JSON schema (`pe-hh/1`)

Every hand-history tool in this repository speaks one JSON format. The Rust parser
(`pe hh parse`) and the hand simulator (`pe hh simulate`) write it; the Python research
package (`python/poker_ecosystem`) and the web demo read it. A file is either a JSON
array of hands or JSON Lines (one hand per line, extension `.jsonl`).

## Conventions

- **Amounts are integers in thousandths of the hand's unit** ("milli"): `$0.25` is `250`,
  a 1,500-chip tournament stack is `1500000`. This keeps sums exact in every language.
- **Cards** are two-character strings: rank `23456789TJQKA`, suit `cdhs` (`"Ah"`, `"Tc"`).
- **Seats** are the site's seat numbers (1-based). Players are identified by `seat` within
  a hand and by `name` across hands.
- Field order in files is not significant. Unknown fields must be ignored by readers.

## Top level

| Field | Type | Meaning |
|---|---|---|
| `schema` | string | Always `"pe-hh/1"` |
| `id` | string | Site hand id |
| `site` | string | `"PokerStars"`, `"Synthetic"`, ... |
| `game` | string | `"NLH"`, `"PLO"`, `"PLO5"`, `"SD"` (short deck 6+), `"O8"` (Omaha hi-lo), `"BIGO"` |
| `limit` | string | `"NL"`, `"PL"` or `"FL"` |
| `currency` | string | `"USD"`, `"EUR"`, `"chips"` |
| `time` | string | As written by the site (`"2026/11/05 21:00:00 ET"`) |
| `table` | string | Table name |
| `max_seats` | int | Table size (2-10) |
| `button` | int | Button seat |
| `sb`, `bb` | int | Blind sizes (milli) |
| `ante` | int | Per-player ante (milli), 0 if none |
| `bomb_pot` | bool | Every player posted `bomb` antes and preflop betting was skipped |
| `board_kind` | string | `"single"`, `"double"` (double-board), `"run_twice"` |
| `players` | array | See below, in seat order |
| `posts` | array | Forced bets, in the order posted |
| `streets` | array | Betting rounds that happened, in order |
| `boards` | array of arrays | Final board(s). One board normally; two for double-board or run-it-twice |
| `shown` | array | `{ "seat", "cards": [...], "description" }` for every hand shown or revealed |
| `collected` | array | `{ "seat", "amount", "pot" }`: `pot` 0 = main pot, 1.. = side pots; `board` (1-based) when a pot is split by board |
| `uncalled` | array | `{ "seat", "amount" }`: uncalled bets returned |
| `total_pot` | int | As stated in the summary (includes rake) |
| `rake` | int | Rake taken |
| `derived` | object | Computed by the reconstruction, see below (absent if not reconstructed) |
| `synthetic` | bool | `true` for simulated hands (then `truth` fields may be present) |

## `players[]`

| Field | Type | Meaning |
|---|---|---|
| `seat` | int | Seat number |
| `name` | string | Screen name |
| `stack` | int | Starting stack (milli) |
| `position` | string | `BTN`, `SB`, `BB`, `UTG`, `UTG1`, `UTG2`, `UTG3`, `LJ`, `HJ`, `CO` (see below) |
| `hole` | array or null | Hole cards when known (hero's dealt cards or cards shown); always present in synthetic hands |
| `hero` | bool | The `Dealt to` player |
| `sitting_out` | bool | Sitting out (not dealt in); such players have no position |

Position labels depend on the number of players dealt in. Heads-up the button posts the
small blind and is labelled `BTN`. With `k = n - 3` seats before the button:

| k | labels in preflop order (then `BTN`, `SB`, `BB`) |
|---|---|
| 0 | (none) |
| 1 | `CO` |
| 2 | `HJ`, `CO` |
| 3 | `UTG`, `HJ`, `CO` |
| 4 | `UTG`, `LJ`, `HJ`, `CO` |
| 5 | `UTG`, `UTG1`, `LJ`, `HJ`, `CO` |
| 6 | `UTG`, `UTG1`, `UTG2`, `LJ`, `HJ`, `CO` |
| 7 | `UTG`, `UTG1`, `UTG2`, `UTG3`, `LJ`, `HJ`, `CO` |

## `posts[]`

`{ "seat", "kind", "amount" }` with `kind` one of `sb`, `bb`, `ante`, `straddle`,
`mississippi` (a straddle by the button or a seat other than UTG), `dead` (dead blind),
`bomb` (bomb-pot ante). Antes and bomb antes are dead money; blinds and straddles are live.

## `streets[]`

`{ "name": "preflop" | "flop" | "turn" | "river", "actions": [...] }`. Board cards are not
repeated here: the flop is the first three cards of each board, the turn the fourth, the
river the fifth. A bomb pot has no `preflop` entry.

Each action:

| Field | Type | Meaning |
|---|---|---|
| `seat` | int | Who acted |
| `type` | string | `fold`, `check`, `call`, `bet`, `raise` |
| `amount` | int | Chips this action put in (0 for fold/check) |
| `to` | int | The player's total contribution on this street after the action |
| `all_in` | bool | The action put the player all-in |
| `truth` | object | Synthetic hands only: the acting bot's probabilities over the legal action types it considered, e.g. `{ "fold": 0.2, "call": 0.5, "raise": 0.3 }` |

A raise's `to` is the "raises X to Y" total; `amount` is what was added by this action.

## `derived` (reconstruction output)

| Field | Type | Meaning |
|---|---|---|
| `ok` | bool | Replay matched the summary (pot, rake, collected amounts) |
| `issues` | array of strings | What did not match |
| `contrib` | object | seat → total chips put in (posts included, uncalled bets excluded) |
| `net` | object | seat → collected minus contributed |
| `pots` | array | `{ "amount", "eligible": [seats] }`, main pot first |
| `players_dealt` | int | Players dealt in |
| `preflop_aggressor` | int or null | Seat of the last preflop raiser |
| `showdown_seats` | array | Seats that reached showdown |
| `allin` | object or null | `{ "street", "seats", "equity": {seat: share}, "ev_net": {seat: milli} }` when two or more players were all-in before the river with known cards (exact enumeration) |
| `river_strength` | object | seat → fraction of all hands this hand beats on the final board (ties count half), for players whose cards are known at a river showdown |

## Ground truth for simulated corpora

`pe hh simulate` also writes `players.json`: one entry per bot with its `name`, `style`
(e.g. `"TAG"`), and the `profile` parameters that drive its decisions. Combined with the
per-action `truth` probabilities this gives the exact frequency each bot *intends* in every
situation, which the opponent-model evaluation compares its estimates against.
