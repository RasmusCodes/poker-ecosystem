"""Reading and replaying ``pe-hh/1`` hand histories.

The format is defined in ``docs/HAND-HISTORY-SCHEMA.md``. Hands stay the plain
``dict`` objects produced by :mod:`json`; this module adds loaders, small
lookup helpers and a deterministic *replay* that reconstructs, for every
action, the pot, the amount to call, how many bets/raises were already made on
the street and whether the actor could still raise. Every statistic in
:mod:`poker_ecosystem.stats` is computed from that replay.

Readers ignore unknown fields and tolerate the optional ``derived``, ``truth``
and ``hole`` fields being absent. Amounts stay integer thousandths ("milli")
until they are explicitly converted to big blinds with :func:`to_bb`.

Replay conventions (assumptions the producers of ``pe-hh/1`` must honour)
-------------------------------------------------------------------------
* ``posts`` of kind ``sb``, ``bb``, ``straddle`` and ``mississippi`` are live:
  they count towards the poster's preflop street contribution. ``ante``,
  ``bomb`` and ``dead`` posts are dead money and only enter the pot.
* An action's ``amount`` is authoritative for chip accounting; ``to`` is
  recomputed as street contribution before the action plus ``amount``.
* ``streets`` are replayed in the canonical order preflop, flop, turn, river,
  and actions within a street in array order.
* At the end of each betting round the part of the largest street
  contribution that nobody matched is returned to its owner (uncalled bet).
* Raises are counted per street. Preflop the blinds and straddles are not
  raises, so the first voluntary raise is the open (2-bet), the second the
  3-bet and so on. Postflop the first bet is the first aggression.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

Hand = Mapping[str, Any]

SCHEMA = "pe-hh/1"
STREETS: tuple[str, ...] = ("preflop", "flop", "turn", "river")
POSTFLOP_STREETS: tuple[str, ...] = ("flop", "turn", "river")
ACTION_TYPES: tuple[str, ...] = ("fold", "check", "call", "bet", "raise")
AGGRESSIVE = frozenset({"bet", "raise"})
VOLUNTARY = frozenset({"call", "bet", "raise"})
LIVE_POST_KINDS = frozenset({"sb", "bb", "straddle", "mississippi"})
DEAD_POST_KINDS = frozenset({"ante", "bomb", "dead"})
POSITIONS: tuple[str, ...] = (
    "UTG", "UTG1", "UTG2", "UTG3", "LJ", "HJ", "CO", "BTN", "SB", "BB",
)
#: Position groups. UTG3 (ten-handed only) is counted as middle position.
POSITION_GROUPS: dict[str, str] = {
    "UTG": "EP", "UTG1": "EP", "UTG2": "EP",
    "UTG3": "MP", "LJ": "MP", "HJ": "MP",
    "CO": "LP", "BTN": "LP",
    "SB": "BLINDS", "BB": "BLINDS",
}
GROUPS: tuple[str, ...] = ("EP", "MP", "LP", "BLINDS")
BOARD_SIZE: dict[str, int] = {"flop": 3, "turn": 4, "river": 5}

# Labels of the k = n - 3 seats before the button, in preflop order (schema table).
_PRE_BUTTON: dict[int, tuple[str, ...]] = {
    0: (),
    1: ("CO",),
    2: ("HJ", "CO"),
    3: ("UTG", "HJ", "CO"),
    4: ("UTG", "LJ", "HJ", "CO"),
    5: ("UTG", "UTG1", "LJ", "HJ", "CO"),
    6: ("UTG", "UTG1", "UTG2", "LJ", "HJ", "CO"),
    7: ("UTG", "UTG1", "UTG2", "UTG3", "LJ", "HJ", "CO"),
}


class HandError(ValueError):
    """A hand does not follow the ``pe-hh/1`` contract closely enough to be used."""


# --------------------------------------------------------------------------- loading


def check_hand(hand: Any) -> dict[str, Any]:
    """Validate the minimum structure of one hand and return it unchanged.

    Only the schema tag (when present) and the ``players`` array are checked;
    everything else is optional for readers.
    """
    if not isinstance(hand, dict):
        raise HandError(f"a hand must be a JSON object, got {type(hand).__name__}")
    schema = hand.get("schema", SCHEMA)
    if schema != SCHEMA and not (isinstance(schema, str) and schema.startswith(SCHEMA + ".")):
        raise HandError(f"hand {hand.get('id')!r}: unsupported schema {schema!r}")
    if not isinstance(hand.get("players"), list):
        raise HandError(f"hand {hand.get('id')!r}: 'players' must be an array")
    return hand


def iter_hands(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield the hands of a ``.json`` (array) or ``.jsonl`` (one hand per line) file.

    JSON Lines files are streamed. A ``.json`` file holding a single hand object
    instead of an array is accepted as a one-hand corpus.
    """
    path = Path(path)
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        with path.open("r", encoding="utf-8-sig") as fh:
            for line_no, line in enumerate(fh, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    hand = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise HandError(f"{path}:{line_no}: invalid JSON ({exc.msg})") from exc
                yield check_hand(hand)
        return
    with path.open("r", encoding="utf-8-sig") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise HandError(f"{path}: expected a JSON array of hands")
    for hand in data:
        yield check_hand(hand)


def load_hands(*paths: str | Path) -> list[dict[str, Any]]:
    """Load every hand of one or more files into a list."""
    hands: list[dict[str, Any]] = []
    for path in paths:
        hands.extend(iter_hands(path))
    return hands


def iter_many(paths: Iterable[str | Path]) -> Iterator[dict[str, Any]]:
    """Chain :func:`iter_hands` over several files."""
    for path in paths:
        yield from iter_hands(path)


# --------------------------------------------------------------------------- lookups


def dealt_players(hand: Hand) -> list[Mapping[str, Any]]:
    """Players dealt into the hand (``sitting_out`` players excluded), seat order."""
    players = [p for p in hand.get("players") or [] if not p.get("sitting_out")]
    return sorted(players, key=lambda p: int(p["seat"]))


def players_by_seat(hand: Hand) -> dict[int, Mapping[str, Any]]:
    """All listed players keyed by seat number."""
    return {int(p["seat"]): p for p in hand.get("players") or []}


def seat_names(hand: Hand) -> dict[int, str]:
    """Seat -> screen name for the players dealt in."""
    return {int(p["seat"]): str(p.get("name") or f"seat{p['seat']}") for p in dealt_players(hand)}


def derive_positions(seats: Sequence[int], button: int) -> dict[int, str]:
    """Position labels from seat numbers and the button, following the schema table.

    Used only as a fallback when a producer omitted ``position``. With a dead
    button the last dealt seat at or before ``button`` is labelled ``BTN``.
    """
    order = sorted(int(s) for s in seats)
    n = len(order)
    if n == 0:
        return {}
    if n > 10:
        raise HandError(f"{n} players dealt in; position labels exist for 2-10")
    if n == 1:
        return {order[0]: "BTN"}
    btn = max((s for s in order if s <= button), default=order[-1])
    i = order.index(btn)
    clockwise = order[i:] + order[:i]
    if n == 2:
        return {clockwise[0]: "BTN", clockwise[1]: "BB"}
    labels = {clockwise[0]: "BTN", clockwise[1]: "SB", clockwise[2]: "BB"}
    labels.update(zip(clockwise[3:], _PRE_BUTTON[n - 3]))
    return labels


def player_positions(hand: Hand) -> dict[int, str]:
    """Seat -> position label, taken from ``players[].position`` (derived if missing)."""
    given = {int(p["seat"]): p.get("position") for p in dealt_players(hand)}
    if all(isinstance(v, str) and v for v in given.values()):
        return {s: str(v) for s, v in given.items()}
    fallback = derive_positions(list(given), int(hand.get("button") or 0))
    return {s: str(v) if isinstance(v, str) and v else fallback[s] for s, v in given.items()}


def position_group(position: str | None) -> str | None:
    """EP / MP / LP / BLINDS for a position label (``None`` if unknown)."""
    return POSITION_GROUPS.get(position or "")


def seat_map(obj: Mapping[Any, Any] | None) -> dict[int, Any]:
    """Convert a JSON object keyed by seat (keys are strings in JSON) to int keys."""
    if not obj:
        return {}
    return {int(k): v for k, v in obj.items()}


def derived(hand: Hand) -> Mapping[str, Any]:
    """The ``derived`` object, or an empty mapping when the hand was not reconstructed."""
    value = hand.get("derived")
    return value if isinstance(value, Mapping) else {}


def big_blind(hand: Hand) -> int:
    """The big blind in milli.

    Falls back to the largest ``bb``/``straddle`` post and then to the largest
    post of any kind when ``bb`` is missing or zero (e.g. ante-only bomb pots).
    """
    bb = int(hand.get("bb") or 0)
    if bb > 0:
        return bb
    posts = hand.get("posts") or []
    live = [int(p.get("amount") or 0) for p in posts if p.get("kind") in ("bb", "straddle", "mississippi")]
    any_post = [int(p.get("amount") or 0) for p in posts]
    for candidates in (live, any_post):
        if candidates and max(candidates) > 0:
            return max(candidates)
    raise HandError(f"hand {hand.get('id')!r}: no big blind to normalise amounts")


def to_bb(amount: float, hand: Hand) -> float:
    """Convert a milli amount to big blinds of this hand."""
    return float(amount) / big_blind(hand)


def boards(hand: Hand) -> list[list[str]]:
    """All final boards (one normally, two for double-board / run-it-twice)."""
    return [list(b) for b in hand.get("boards") or [] if isinstance(b, list)]


def board(hand: Hand, index: int = 0) -> list[str]:
    """One board (the first by default); empty when the hand ended preflop."""
    all_boards = boards(hand)
    return all_boards[index] if index < len(all_boards) else []


def flop_texture(cards: Sequence[str]) -> dict[str, bool] | None:
    """Texture of the first three cards: paired, monotone, two-tone, rainbow."""
    if len(cards) < 3:
        return None
    flop = [str(c) for c in cards[:3]]
    ranks = {c[0] for c in flop}
    suits = len({c[-1] for c in flop})
    return {
        "paired": len(ranks) < 3,
        "monotone": suits == 1,
        "two_tone": suits == 2,
        "rainbow": suits == 3,
    }


def collected_by_seat(hand: Hand) -> dict[int, int]:
    """Total collected per seat over all pots and boards."""
    out: dict[int, int] = {}
    for entry in hand.get("collected") or []:
        seat = int(entry["seat"])
        out[seat] = out.get(seat, 0) + int(entry.get("amount") or 0)
    return out


def river_strength(hand: Hand) -> dict[int, float]:
    """``derived.river_strength`` with integer seat keys (empty when absent)."""
    return {s: float(v) for s, v in seat_map(derived(hand).get("river_strength")).items()}


def allin_info(hand: Hand) -> Mapping[str, Any] | None:
    """``derived.allin`` when present."""
    value = derived(hand).get("allin")
    return value if isinstance(value, Mapping) else None


def streets_dealt(hand: Hand) -> tuple[str, ...]:
    """Streets whose cards were dealt: betting rounds plus all-in run-outs.

    A board street counts as dealt when the hand has a betting round of that
    name or any final board holds enough cards for it.
    """
    named = {s.get("name") for s in hand.get("streets") or []}
    longest = max((len(b) for b in boards(hand)), default=0)
    dealt = []
    if "preflop" in named or not hand.get("bomb_pot"):
        dealt.append("preflop")
    for name in POSTFLOP_STREETS:
        if name in named or longest >= BOARD_SIZE[name]:
            dealt.append(name)
    return tuple(dealt)


# --------------------------------------------------------------------------- replay


@dataclass(frozen=True)
class Decision:
    """One action together with the situation the actor faced.

    ``to_call`` is the chips needed to call, capped at the actor's remaining
    stack (0 means checking was possible). ``aggressions_before`` counts the
    bets and raises already made on this street (blinds and straddles are not
    counted). ``can_raise`` is true when the actor had chips beyond the call
    and at least one opponent still able to act; incomplete-raise rules are
    not modelled.
    """

    street: str
    index: int
    seat: int
    type: str
    amount: int
    to: int
    all_in: bool
    truth: Mapping[str, Any] | None
    pot_before: int
    to_call: int
    bet_level: int
    aggressions_before: int
    first_decision: bool
    checked_before: bool
    can_raise: bool
    last_aggressor: int | None

    @property
    def facing_bet(self) -> bool:
        """True when the actor had to put chips in to continue."""
        return self.to_call > 0

    @property
    def aggressive(self) -> bool:
        """True for bets and raises."""
        return self.type in AGGRESSIVE

    def size_fraction(self) -> tuple[int, int] | None:
        """Bet or raise size as (numerator, denominator) of a pot fraction.

        A bet of ``b`` into pot ``P`` is ``b / P``. A raise to ``T`` facing a
        bet level ``L`` is the raise increment over the pot after calling:
        ``(T - L) / (P + to_call)``. ``None`` for passive actions.
        """
        if self.type not in AGGRESSIVE:
            return None
        return self.to - self.bet_level, self.pot_before + self.to_call


@dataclass(frozen=True)
class Replay:
    """Result of :func:`replay`: per-action situations and per-street bookkeeping."""

    hand: Hand
    seats: tuple[int, ...]
    names: dict[int, str]
    positions: dict[int, str]
    decisions: tuple[Decision, ...]
    by_street: dict[str, tuple[Decision, ...]]
    folded_before: dict[str, frozenset[int]]
    allin_before: dict[str, frozenset[int]]
    street_aggressor: dict[str, int | None]
    streets_dealt: tuple[str, ...]
    folded: frozenset[int]
    contributions: dict[int, int]
    returned: dict[int, int]
    final_pot: int

    @property
    def preflop_aggressor(self) -> int | None:
        """Seat of the last preflop raiser (``None`` for limped pots and bomb pots)."""
        return self.street_aggressor.get("preflop")

    def alive_at(self, street: str) -> frozenset[int]:
        """Seats not folded when ``street`` began (empty if it was never dealt)."""
        if street not in self.folded_before:
            return frozenset()
        return frozenset(s for s in self.seats if s not in self.folded_before[street])

    def saw(self, seat: int, street: str) -> bool:
        """Did ``seat`` see ``street`` (dealt and not folded before it)?"""
        return seat in self.alive_at(street)

    def decisions_of(self, seat: int, street: str) -> tuple[Decision, ...]:
        """The actions of one seat on one street, in order."""
        return tuple(d for d in self.by_street.get(street, ()) if d.seat == seat)


class _Ledger:
    """Mutable chip bookkeeping used while replaying one hand."""

    def __init__(self, hand: Hand) -> None:
        players = dealt_players(hand)
        self.seats = tuple(int(p["seat"]) for p in players)
        self.stacks = {int(p["seat"]): p.get("stack") for p in players}
        self.contrib = {s: 0 for s in self.seats}
        self.street = {s: 0 for s in self.seats}
        self.returned = {s: 0 for s in self.seats}
        self.pot = 0
        self.folded: set[int] = set()
        self.allin: set[int] = set()

    def remaining(self, seat: int) -> int | None:
        stack = self.stacks.get(seat)
        return None if stack is None else int(stack) - self.contrib.get(seat, 0)

    def put(self, seat: int, amount: int, live: bool) -> None:
        self.pot += amount
        if seat not in self.contrib:
            return
        self.contrib[seat] += amount
        if live:
            self.street[seat] += amount
        left = self.remaining(seat)
        if left is not None and left <= 0:
            self.allin.add(seat)

    def can_raise(self, seat: int, to_call: int) -> bool:
        left = self.remaining(seat)
        others = any(s != seat and s not in self.folded and s not in self.allin for s in self.seats)
        return others and (left is None or left > to_call)

    def return_uncalled(self) -> None:
        ranked = sorted(self.street.items(), key=lambda kv: kv[1], reverse=True)
        if len(ranked) < 2 or ranked[0][1] <= ranked[1][1]:
            return
        seat, top = ranked[0]
        excess = top - ranked[1][1]
        self.pot -= excess
        self.contrib[seat] -= excess
        self.street[seat] -= excess
        self.returned[seat] += excess


def replay(hand: Hand) -> Replay:
    """Replay a hand and describe the situation behind every action."""
    ledger = _Ledger(hand)
    for post in hand.get("posts") or []:
        ledger.put(int(post["seat"]), int(post.get("amount") or 0), post.get("kind") in LIVE_POST_KINDS)

    by_name = {s.get("name"): s for s in hand.get("streets") or []}
    dealt = streets_dealt(hand)
    decisions: list[Decision] = []
    by_street: dict[str, tuple[Decision, ...]] = {}
    folded_before: dict[str, frozenset[int]] = {}
    allin_before: dict[str, frozenset[int]] = {}
    aggressor: dict[str, int | None] = {}
    for name in STREETS:
        if name not in dealt and name not in by_name:
            continue
        folded_before[name] = frozenset(ledger.folded)
        allin_before[name] = frozenset(ledger.allin)
        street = by_name.get(name)
        if street is None:
            continue
        if name != "preflop":
            ledger.street = {s: 0 for s in ledger.seats}
        played, aggressor[name] = _replay_street(name, street.get("actions") or [], ledger)
        by_street[name] = played
        decisions.extend(played)
        ledger.return_uncalled()

    return Replay(
        hand=hand,
        seats=ledger.seats,
        names=seat_names(hand),
        positions=player_positions(hand),
        decisions=tuple(decisions),
        by_street=by_street,
        folded_before=folded_before,
        allin_before=allin_before,
        street_aggressor=aggressor,
        streets_dealt=dealt,
        folded=frozenset(ledger.folded),
        contributions=dict(ledger.contrib),
        returned={s: v for s, v in ledger.returned.items() if v},
        final_pot=ledger.pot,
    )


def _replay_street(
    name: str, actions: Sequence[Mapping[str, Any]], ledger: _Ledger
) -> tuple[tuple[Decision, ...], int | None]:
    level = max(ledger.street.values(), default=0)
    aggressions = 0
    last_aggressor: int | None = None
    acted: set[int] = set()
    checked: set[int] = set()
    played: list[Decision] = []
    for index, act in enumerate(actions):
        seat = int(act["seat"])
        kind = str(act.get("type"))
        amount = int(act.get("amount") or 0)
        before = ledger.street.get(seat, 0)
        to_call = max(0, level - before)
        left = ledger.remaining(seat)
        if left is not None:
            to_call = min(to_call, max(left, 0))
        truth = act.get("truth")
        played.append(
            Decision(
                street=name,
                index=index,
                seat=seat,
                type=kind,
                amount=amount,
                to=before + amount,
                all_in=bool(act.get("all_in")),
                truth=truth if isinstance(truth, Mapping) else None,
                pot_before=ledger.pot,
                to_call=to_call,
                bet_level=level,
                aggressions_before=aggressions,
                first_decision=seat not in acted,
                checked_before=seat in checked,
                can_raise=ledger.can_raise(seat, to_call),
                last_aggressor=last_aggressor,
            )
        )
        ledger.put(seat, amount, live=True)
        acted.add(seat)
        if kind == "check":
            checked.add(seat)
        elif kind == "fold":
            ledger.folded.add(seat)
        elif kind in AGGRESSIVE:
            aggressions += 1
            last_aggressor = seat
        if act.get("all_in"):
            ledger.allin.add(seat)
        level = max(level, ledger.street.get(seat, 0))
    return tuple(played), last_aggressor


# --------------------------------------------------------------------------- derived helpers


def preflop_aggressor(hand: Hand) -> int | None:
    """Seat of the last preflop raiser, or ``None``."""
    return replay(hand).preflop_aggressor


def pot_before_actions(hand: Hand) -> list[tuple[str, int, int, int]]:
    """(street, action index, seat, pot before the action in milli) for every action."""
    return [(d.street, d.index, d.seat, d.pot_before) for d in replay(hand).decisions]


def showdown_seats(hand: Hand, rp: Replay | None = None) -> frozenset[int]:
    """Seats that reached showdown.

    Uses ``derived.showdown_seats`` when present; otherwise every unfolded
    player when two or more remained at the end of the hand.
    """
    given = derived(hand).get("showdown_seats")
    if isinstance(given, list):
        return frozenset(int(s) for s in given)
    rp = rp or replay(hand)
    alive = [s for s in rp.seats if s not in rp.folded]
    return frozenset(alive) if len(alive) >= 2 else frozenset()


def net_by_seat(hand: Hand, rp: Replay | None = None) -> dict[int, int]:
    """Collected minus contributed per dealt seat (``derived.net`` when present)."""
    given = seat_map(derived(hand).get("net"))
    rp = rp or replay(hand)
    collected = collected_by_seat(hand)
    return {
        s: int(given[s]) if s in given else collected.get(s, 0) - rp.contributions.get(s, 0)
        for s in rp.seats
    }
