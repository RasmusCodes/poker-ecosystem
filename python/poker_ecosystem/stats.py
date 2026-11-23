"""Per-player frequency counters computed from ``pe-hh/1`` hands.

Every frequency statistic is a pair of counters per player, both per table
position and pooled over positions: *opportunities* ``n`` and *successes*
``k``. The position is the player's position in that hand. A "decision" is one
entry of a street's ``actions`` array, replayed with the conventions of
:mod:`poker_ecosystem.hh` (blinds and straddles are not raises, so the open is
the first raise, the 3-bet the second, and so on). A statistic counts at most
one opportunity per player and hand, except ``af_*`` (one per decision).

Statistic definitions
---------------------

Preflop statistics need a preflop betting round (bomb pots have none).

| key | opportunity | success | truth |
|---|---|---|---|
| `vpip` | had at least one preflop decision (walks and players all-in from a blind have none) | any call, bet or raise preflop | compensator |
| `pfr` | as `vpip` | any bet or raise preflop | compensator |
| `rfi` | first preflop decision with every earlier preflop action a fold | raise | decision |
| `limp` | first preflop decision with no raise yet and a positive amount to call | call | decision |
| `cold_call` | first preflop decision facing exactly one raise | call | decision |
| `three_bet` | first preflop decision facing exactly one raise, made by another player | raise | decision |
| `fold_to_three_bet` | made the open (first raise) and the next own decision faces exactly two raises | fold | decision |
| `four_bet` | first preflop decision facing exactly two raises, the last by another player | raise | decision |
| `steal` | `rfi` opportunity in CO, BTN or SB | raise | decision |
| `fold_to_steal` | SB or BB, first decision, facing one raise that was an RFI from CO/BTN/SB with only folds after it | fold | decision |

For each postflop street S in flop, turn, river:

| key | opportunity | success | truth |
|---|---|---|---|
| `cbet_S` | preflop aggressor (last preflop raiser) at the first own decision on S with no bet yet; `cbet_turn` also needs a flop c-bet and `cbet_river` a turn c-bet | bet | decision |
| `fold_vs_cbet_S`, `call_vs_cbet_S`, `raise_vs_cbet_S` | first own decision on S facing a bet while the street's c-bet is the only bet so far | fold / call / raise | decision |
| `donk_S` | first own decision on S with no bet yet, before the previous street's last aggressor (still in, not all-in) has acted on S | bet | decision |
| `check_raise_S` | checked earlier on S, then at the first own decision facing a bet had chips to raise and an opponent able to act | raise | decision |
| `bet_unopened_S` | first own decision on S with no bet yet | bet | decision |
| `fold_vs_bet_S`, `call_vs_bet_S`, `raise_vs_bet_S` | first own decision on S facing a bet, with exactly one bet and no raise on S so far | fold / call / raise | decision |
| `af_S` | every own bet, raise, call or fold on S (checks excluded; several per hand) | bet or raise | none |

Showdown and multi-street statistics:

| key | opportunity | success | truth |
|---|---|---|---|
| `wtsd` | saw the flop: not folded when it was dealt (includes all-in run-outs and bomb pots) | reached showdown (`derived.showdown_seats`, else two or more unfolded players at the end) | none |
| `wsd` | reached showdown | collected a positive amount from any pot | none |
| `river_bluff` | own river bet (type `bet`) at a showdown where the bettor's `derived.river_strength` is known | strength < bluff threshold (default 0.5) | none |
| `xr_flop_bet_turn` | check-raised the flop and has a first turn decision with no bet yet | bet | decision |

Truth column (synthetic corpora only): "decision" means the opportunity is a
single decision and its true probability is the bot's ``truth`` probability
of the success action types there (``truth`` is normalised to sum to one; a
missing action type has probability 0). "compensator" means the opportunity
spans several decisions; its truth value is the sum, over the decisions up to
and including the first success, of the success probability at each. Its
expectation equals the probability of success (it is the compensator of the
success indicator), so averaging it over opportunities is unbiased. "none"
means the success is not a single action choice (it depends on opponents or on
cards) and cannot be checked against ``truth``.

Other outputs
-------------

* Bet sizing: every bet or raise is binned by pot fraction (see
  :meth:`hh.Decision.size_fraction`) into the half-open intervals
  ``<40`` [0, 0.40), ``40-60`` [0.40, 0.60), ``60-85`` [0.60, 0.85),
  ``85-115`` [0.85, 1.15), ``>115`` [1.15, inf), and ``allin`` when the action
  put the player all-in (overrides the fraction). Kept per street and kind
  (``bet`` or ``raise``); preflop raises use the same definition.
* River showdown strengths: ``derived.river_strength`` of every shown river
  bet. ``river_bluff`` is the bluff share among those bets. Because only bets
  that reached showdown are visible, this estimates the bluff share of
  *called* bets; it is close to the bluff share of all bets only when calling
  decisions do not depend on the bettor's cards (card removal breaks this
  slightly).
* Value-bet threshold: the lower ``q`` quantile (default 0.10, inverse-CDF
  definition) of river strength among shown river bets with strength at or
  above the bluff threshold.
* Multi-street transitions: per player, counts of (street, line) -> (next
  street, line) where a line is the player's action letters on the street
  (``x`` check, ``b`` bet, ``c`` call, ``r`` raise, ``f`` fold, e.g. ``xr`` is
  a check-raise), and for the preflop aggressor the c-bet chain states
  ``bet`` / ``check`` (first decision unopened) or ``faced_bet``.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from . import hh
from .hh import AGGRESSIVE, POSTFLOP_STREETS, VOLUNTARY, Decision, Replay

ALL = "ALL"
STEAL_POSITIONS = frozenset({"CO", "BTN", "SB"})
BLIND_POSITIONS = frozenset({"SB", "BB"})
DEFAULT_BLUFF_THRESHOLD = 0.5
SIZE_BINS: tuple[str, ...] = ("<40", "40-60", "60-85", "85-115", ">115", "allin")
_SIZE_EDGES: tuple[tuple[int, str], ...] = ((40, "<40"), (60, "40-60"), (85, "60-85"), (115, "85-115"))
LINE_LETTERS = {"fold": "f", "check": "x", "call": "c", "bet": "b", "raise": "r"}
_PREVIOUS = {"flop": "preflop", "turn": "flop", "river": "turn"}


@dataclass(frozen=True)
class StatDef:
    """Definition of one frequency statistic (see the module docstring)."""

    key: str
    street: str
    opportunity: str
    success: str
    truth: str  # "decision", "compensator" or "none"

    @property
    def truth_evaluable(self) -> bool:
        """Can the statistic be compared with per-decision ``truth`` probabilities?"""
        return self.truth != "none"


def _build_definitions() -> dict[str, StatDef]:
    defs: dict[str, StatDef] = {}

    def add(key: str, street: str, opportunity: str, success: str, truth: str = "decision") -> None:
        defs[key] = StatDef(key, street, opportunity, success, truth)

    add("vpip", "preflop", "had at least one preflop decision", "any call, bet or raise preflop", "compensator")
    add("pfr", "preflop", "had at least one preflop decision", "any bet or raise preflop", "compensator")
    add("rfi", "preflop", "first preflop decision, every earlier preflop action a fold", "raise")
    add("limp", "preflop", "first preflop decision, no raise yet, positive amount to call", "call")
    add("cold_call", "preflop", "first preflop decision facing exactly one raise", "call")
    add("three_bet", "preflop", "first preflop decision facing exactly one raise by another player", "raise")
    add("fold_to_three_bet", "preflop", "made the open; next own decision faces exactly two raises", "fold")
    add("four_bet", "preflop", "first preflop decision facing exactly two raises, the last by another player", "raise")
    add("steal", "preflop", "rfi opportunity in CO, BTN or SB", "raise")
    add("fold_to_steal", "preflop", "SB/BB first decision facing only an RFI raise from CO/BTN/SB", "fold")
    for s in POSTFLOP_STREETS:
        chain = {"flop": "", "turn": " after a flop c-bet", "river": " after a turn c-bet"}[s]
        add(f"cbet_{s}", s, f"preflop aggressor, first own {s} decision with no bet yet{chain}", "bet")
        for verb in ("fold", "call", "raise"):
            add(f"{verb}_vs_cbet_{s}", s, f"first own {s} decision facing a bet, the c-bet being the only bet", verb)
        add(f"donk_{s}", s, f"first own {s} decision, no bet yet, previous street's aggressor still to act", "bet")
        add(f"check_raise_{s}", s, f"checked on the {s}, then faced a bet with the option to raise", "raise")
        add(f"bet_unopened_{s}", s, f"first own {s} decision with no bet yet", "bet")
        for verb in ("fold", "call", "raise"):
            add(f"{verb}_vs_bet_{s}", s, f"first own {s} decision facing exactly one bet and no raise", verb)
        add(f"af_{s}", s, f"every own bet, raise, call or fold on the {s}", "bet or raise", "none")
    add("wtsd", "showdown", "saw the flop", "reached showdown", "none")
    add("wsd", "showdown", "reached showdown", "collected a positive amount", "none")
    add("river_bluff", "river", "own river bet shown at showdown with known strength", "strength below the bluff threshold", "none")
    add("xr_flop_bet_turn", "turn", "check-raised the flop; first turn decision with no bet yet", "bet")
    return defs


STAT_DEFS: dict[str, StatDef] = _build_definitions()
STAT_KEYS: tuple[str, ...] = tuple(STAT_DEFS)


# --------------------------------------------------------------------------- observations


@dataclass(frozen=True)
class Observation:
    """One opportunity of one statistic for one player in one hand."""

    player: str
    stat: str
    position: str
    success: bool
    truth: float | None = None


@dataclass(frozen=True)
class SizingObservation:
    """One bet or raise binned by pot fraction."""

    player: str
    street: str
    kind: str  # "bet" or "raise"
    fraction: float | None
    bin: str


@dataclass
class HandObservations:
    """Everything :class:`StatsTable` takes from one hand."""

    hand_id: str
    players: dict[str, str]
    observations: list[Observation] = field(default_factory=list)
    sizing: list[SizingObservation] = field(default_factory=list)
    river_bets: list[tuple[str, float]] = field(default_factory=list)
    lines: dict[str, dict[str, str]] = field(default_factory=dict)
    cbet_chain: dict[str, dict[str, str]] = field(default_factory=dict)
    allin: list[tuple[str, float, float]] = field(default_factory=list)
    net_bb: dict[str, float] = field(default_factory=dict)


def truth_probability(decision: Decision, success_types: Iterable[str]) -> float | None:
    """Normalised ``truth`` probability of the success action types at a decision."""
    truth = decision.truth
    if not isinstance(truth, Mapping):
        return None
    wanted = set(success_types)
    total = hit = 0.0
    for action, prob in truth.items():
        try:
            p = float(prob)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(p) or p < 0:
            return None
        total += p
        if action in wanted:
            hit += p
    return hit / total if total > 0 else None


def compensator(decisions: Sequence[Decision], success_types: Iterable[str]) -> float | None:
    """Sum of success probabilities over decisions up to the first success.

    ``None`` when any of those decisions lacks ``truth``.
    """
    wanted = set(success_types)
    total = 0.0
    for d in decisions:
        p = truth_probability(d, wanted)
        if p is None:
            return None
        total += p
        if d.type in wanted:
            break
    return total


def size_bin(numerator: int, denominator: int, all_in: bool = False) -> str | None:
    """Pot-fraction bin of a bet/raise (exact integer comparisons, see module doc)."""
    if all_in:
        return "allin"
    if denominator <= 0 or numerator <= 0:
        return None
    for edge, label in _SIZE_EDGES:
        if numerator * 100 < edge * denominator:
            return label
    return ">115"


class _Collector:
    def __init__(self, rp: Replay) -> None:
        self.rp = rp
        self.items: list[Observation] = []

    def add(self, seat: int, stat: str, success: bool, truth: float | None = None) -> None:
        name = self.rp.names.get(seat)
        if name is None:
            return
        position = self.rp.positions.get(seat) or "UNKNOWN"
        self.items.append(Observation(name, stat, position, bool(success), truth))

    def decision(self, d: Decision, stat: str, success_types: Iterable[str]) -> bool:
        wanted = set(success_types)
        success = d.type in wanted
        self.add(d.seat, stat, success, truth_probability(d, wanted))
        return success


def _group(decisions: Sequence[Decision]) -> dict[int, list[Decision]]:
    grouped: dict[int, list[Decision]] = {}
    for d in decisions:
        grouped.setdefault(d.seat, []).append(d)
    return grouped


def _first(decisions: Iterable[Decision], predicate: Any) -> Decision | None:
    return next((d for d in decisions if predicate(d)), None)


def _preflop(rp: Replay, out: _Collector) -> None:
    decs = rp.by_street.get("preflop") or ()
    for seat, ds in _group(decs).items():
        out.add(seat, "vpip", any(d.type in VOLUNTARY for d in ds), compensator(ds, VOLUNTARY))
        out.add(seat, "pfr", any(d.type in AGGRESSIVE for d in ds), compensator(ds, AGGRESSIVE))
        first = ds[0]
        if all(p.type == "fold" for p in decs[: first.index]):
            out.decision(first, "rfi", AGGRESSIVE)
            if rp.positions.get(seat) in STEAL_POSITIONS:
                out.decision(first, "steal", AGGRESSIVE)
        if first.aggressions_before == 0 and first.to_call > 0:
            out.decision(first, "limp", {"call"})
        if first.aggressions_before == 1 and first.to_call > 0:
            out.decision(first, "cold_call", {"call"})
        facing = [d for d in ds if d.to_call > 0 and d.last_aggressor not in (None, seat)]
        three = _first(facing, lambda d: d.aggressions_before == 1)
        if three is not None:
            out.decision(three, "three_bet", AGGRESSIVE)
        four = _first(facing, lambda d: d.aggressions_before == 2)
        if four is not None:
            out.decision(four, "four_bet", AGGRESSIVE)
    _preflop_vs_open(rp, decs, out)


def _preflop_vs_open(rp: Replay, decs: Sequence[Decision], out: _Collector) -> None:
    opener = _first(decs, lambda d: d.type in AGGRESSIVE)
    if opener is None:
        return
    after = decs[opener.index + 1 :]
    reply = _first(after, lambda d: d.seat == opener.seat)
    if reply is not None and reply.aggressions_before == 2:
        out.decision(reply, "fold_to_three_bet", {"fold"})
    is_rfi = all(p.type == "fold" for p in decs[: opener.index])
    if not is_rfi or rp.positions.get(opener.seat) not in STEAL_POSITIONS:
        return
    for d in after:
        blind = rp.positions.get(d.seat) in BLIND_POSITIONS
        if blind and d.first_decision and d.seat != opener.seat and d.aggressions_before == 1:
            out.decision(d, "fold_to_steal", {"fold"})
        if d.type != "fold":
            break


def _postflop(rp: Replay, out: _Collector) -> None:
    pfa = rp.preflop_aggressor
    cbet_streets: set[str] = set()
    flop_check_raisers: set[int] = set()
    for street in POSTFLOP_STREETS:
        decs = rp.by_street.get(street) or ()
        if not decs:
            continue
        grouped = _group(decs)
        cbet_index = _cbet(street, pfa, grouped, cbet_streets, out)
        prev_aggr = rp.street_aggressor.get(_PREVIOUS[street])
        for seat, ds in grouped.items():
            first = ds[0]
            if first.aggressions_before == 0:
                out.decision(first, f"bet_unopened_{street}", AGGRESSIVE)
                if street == "turn" and seat in flop_check_raisers:
                    out.decision(first, "xr_flop_bet_turn", AGGRESSIVE)
            if _is_donk_spot(rp, street, decs, first, prev_aggr):
                out.decision(first, f"donk_{street}", AGGRESSIVE)
            _facing_bet(street, ds, cbet_index, out)
            if _check_raise(street, ds, out) and street == "flop":
                flop_check_raisers.add(seat)
            for d in ds:
                if d.type != "check":
                    out.add(seat, f"af_{street}", d.type in AGGRESSIVE)


def _cbet(
    street: str, pfa: int | None, grouped: dict[int, list[Decision]], done: set[str], out: _Collector
) -> int | None:
    """Record the c-bet opportunity; return the index of the c-bet when one was made."""
    if pfa is None or pfa not in grouped:
        return None
    if street != "flop" and _PREVIOUS[street] not in done:
        return None
    first = grouped[pfa][0]
    if first.aggressions_before != 0:
        return None
    if out.decision(first, f"cbet_{street}", AGGRESSIVE):
        done.add(street)
        return first.index
    return None


def _is_donk_spot(
    rp: Replay, street: str, decs: Sequence[Decision], first: Decision, prev_aggr: int | None
) -> bool:
    if prev_aggr is None or first.seat == prev_aggr or first.aggressions_before != 0:
        return False
    if prev_aggr in rp.folded_before[street] or prev_aggr in rp.allin_before[street]:
        return False
    return not any(p.seat == prev_aggr for p in decs[: first.index])


def _facing_bet(street: str, ds: Sequence[Decision], cbet_index: int | None, out: _Collector) -> None:
    facing = _first(ds, lambda d: d.to_call > 0)
    if facing is None or facing.aggressions_before != 1:
        return
    vs_cbet = cbet_index is not None and cbet_index < facing.index
    for verb in ("fold", "call", "raise"):
        out.decision(facing, f"{verb}_vs_bet_{street}", {verb})
        if vs_cbet:
            out.decision(facing, f"{verb}_vs_cbet_{street}", {verb})


def _check_raise(street: str, ds: Sequence[Decision], out: _Collector) -> bool:
    """Record the check-raise opportunity; return True when the player check-raised."""
    checked = next((i for i, d in enumerate(ds) if d.type == "check"), None)
    if checked is None:
        return False
    later = _first(ds[checked + 1 :], lambda d: d.to_call > 0)
    if later is None or not later.can_raise:
        return False
    return out.decision(later, f"check_raise_{street}", {"raise"})


def _showdown(rp: Replay, out: _Collector, threshold: float) -> list[tuple[str, float]]:
    hand = rp.hand
    shown = hh.showdown_seats(hand, rp)
    for seat in sorted(rp.alive_at("flop")):
        out.add(seat, "wtsd", seat in shown)
    collected = hh.collected_by_seat(hand)
    for seat in sorted(s for s in shown if s in rp.names):
        out.add(seat, "wsd", collected.get(seat, 0) > 0)
    strengths = hh.river_strength(hand)
    river_bets: list[tuple[str, float]] = []
    for d in rp.by_street.get("river") or ():
        if d.type == "bet" and d.seat in shown and d.seat in strengths:
            out.add(d.seat, "river_bluff", strengths[d.seat] < threshold)
            river_bets.append((rp.names[d.seat], strengths[d.seat]))
    return river_bets


def _sizing(rp: Replay) -> list[SizingObservation]:
    items: list[SizingObservation] = []
    for d in rp.decisions:
        frac = d.size_fraction()
        if frac is None:
            continue
        label = size_bin(frac[0], frac[1], d.all_in)
        if label is None or d.seat not in rp.names:
            continue
        value = frac[0] / frac[1] if frac[1] > 0 else None
        kind = "bet" if d.type == "bet" else "raise"
        items.append(SizingObservation(rp.names[d.seat], d.street, kind, value, label))
    return items


def _lines(rp: Replay) -> dict[str, dict[str, str]]:
    lines: dict[str, dict[str, str]] = {}
    for street in POSTFLOP_STREETS:
        for seat, ds in _group(rp.by_street.get(street) or ()).items():
            code = "".join(LINE_LETTERS.get(d.type, "?") for d in ds)
            lines.setdefault(rp.names.get(seat, str(seat)), {})[street] = code
    return lines


def _cbet_chain(rp: Replay) -> dict[str, dict[str, str]]:
    pfa = rp.preflop_aggressor
    if pfa is None or pfa not in rp.names:
        return {}
    states: dict[str, str] = {}
    for street in POSTFLOP_STREETS:
        ds = rp.decisions_of(pfa, street)
        if not ds:
            continue
        first = ds[0]
        if first.aggressions_before > 0:
            states[street] = "faced_bet"
        else:
            states[street] = "bet" if first.type in AGGRESSIVE else "check"
    return {rp.names[pfa]: states}


def _money(rp: Replay) -> tuple[dict[str, float], list[tuple[str, float, float]]]:
    hand = rp.hand
    try:
        bb = hh.big_blind(hand)
    except hh.HandError:
        return {}, []
    net = hh.net_by_seat(hand, rp)
    net_bb = {rp.names[s]: v / bb for s, v in net.items() if s in rp.names}
    allin: list[tuple[str, float, float]] = []
    info = hh.allin_info(hand)
    if info is not None:
        for seat, ev in hh.seat_map(info.get("ev_net")).items():
            if seat in rp.names and seat in net:
                allin.append((rp.names[seat], net[seat] / bb, float(ev) / bb))
    return net_bb, allin


def hand_observations(hand: hh.Hand, *, bluff_threshold: float = DEFAULT_BLUFF_THRESHOLD) -> HandObservations:
    """Replay one hand and extract every observation the counters need."""
    rp = hh.replay(hand)
    out = _Collector(rp)
    _preflop(rp, out)
    _postflop(rp, out)
    river_bets = _showdown(rp, out, bluff_threshold)
    net_bb, allin = _money(rp)
    return HandObservations(
        hand_id=str(hand.get("id", "")),
        players={rp.names[s]: rp.positions.get(s) or "UNKNOWN" for s in rp.seats},
        observations=out.items,
        sizing=_sizing(rp),
        river_bets=river_bets,
        lines=_lines(rp),
        cbet_chain=_cbet_chain(rp),
        allin=allin,
        net_bb=net_bb,
    )


# --------------------------------------------------------------------------- counters


def empirical_quantile(values: Sequence[float], q: float) -> float | None:
    """Inverse-CDF sample quantile: the ceil(q*n)-th smallest value (1-based)."""
    if not values:
        return None
    if not 0.0 <= q <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    ordered = sorted(values)
    j = max(1, math.ceil(q * len(ordered)))
    return ordered[j - 1]


class StatsTable:
    """Counters for many players and hands.

    ``get(player, stat, position)`` returns ``(opportunities, successes)``;
    ``position=None`` (or :data:`ALL`) means pooled over positions.
    """

    def __init__(self, bluff_threshold: float = DEFAULT_BLUFF_THRESHOLD) -> None:
        self.bluff_threshold = bluff_threshold
        self.hands: Counter[str] = Counter()
        self.positions_played: dict[str, Counter[str]] = defaultdict(Counter)
        self.net_bb: dict[str, float] = defaultdict(float)
        self.n_hands = 0
        self._counts: dict[tuple[str, str, str], list[int]] = {}
        self._truth: dict[tuple[str, str, str], list[float]] = {}
        self._positions: dict[str, set[str]] = defaultdict(set)
        self._sizing: dict[tuple[str, str, str], Counter[str]] = {}
        self._river: dict[str, list[float]] = defaultdict(list)
        self._transitions: dict[str, Counter[tuple[str, str, str]]] = defaultdict(Counter)
        self._allin: dict[str, list[tuple[float, float]]] = defaultdict(list)

    # -- building

    @classmethod
    def from_hands(cls, hands: Iterable[hh.Hand], *, bluff_threshold: float = DEFAULT_BLUFF_THRESHOLD) -> "StatsTable":
        table = cls(bluff_threshold)
        for hand in hands:
            table.add_hand(hand)
        return table

    def add_hand(self, hand: hh.Hand) -> HandObservations:
        obs = hand_observations(hand, bluff_threshold=self.bluff_threshold)
        self.add(obs)
        return obs

    def add(self, obs: HandObservations) -> None:
        self.n_hands += 1
        for player, position in obs.players.items():
            self.hands[player] += 1
            self.positions_played[player][position] += 1
        for o in obs.observations:
            self.add_observation(o)
        for s in obs.sizing:
            self._sizing.setdefault((s.player, s.street, s.kind), Counter())[s.bin] += 1
        for player, strength in obs.river_bets:
            self._river[player].append(strength)
        for player, by_street in obs.lines.items():
            self._add_chain(player, "line", by_street)
        for player, by_street in obs.cbet_chain.items():
            self._add_chain(player, "cbet", by_street)
        for player, net, ev in obs.allin:
            self._allin[player].append((net, ev))
        for player, net in obs.net_bb.items():
            self.net_bb[player] += net

    def add_observation(self, o: Observation) -> None:
        self.add_counts(o.player, o.stat, o.position, 1, int(o.success))
        if o.truth is not None:
            for key in ((o.player, o.stat, o.position), (o.player, o.stat, ALL)):
                acc = self._truth.setdefault(key, [0.0, 0])
                acc[0] += o.truth
                acc[1] += 1

    def add_counts(self, player: str, stat: str, position: str, n: int, k: int) -> None:
        """Add ``n`` opportunities with ``k`` successes at a position (and pooled)."""
        if position == ALL:
            raise ValueError("add counts at a concrete position; the pooled total is kept automatically")
        for key in ((player, stat, position), (player, stat, ALL)):
            acc = self._counts.setdefault(key, [0, 0])
            acc[0] += n
            acc[1] += k
        self._positions[stat].add(position)
        if player not in self.hands:
            self.hands[player] += 0

    def _add_chain(self, player: str, chain: str, by_street: Mapping[str, str]) -> None:
        streets = [s for s in POSTFLOP_STREETS if by_street.get(s)]
        for a, b in zip(streets, streets[1:]):
            if POSTFLOP_STREETS.index(b) == POSTFLOP_STREETS.index(a) + 1:
                self._transitions[player][(chain, f"{a}:{by_street[a]}", f"{b}:{by_street[b]}")] += 1

    # -- queries

    def players(self) -> list[str]:
        return sorted(self.hands)

    def stats(self) -> list[str]:
        """Statistic keys with at least one opportunity, in definition order."""
        present = {stat for (_, stat, _) in self._counts}
        return [k for k in STAT_KEYS if k in present] + sorted(present - set(STAT_KEYS))

    def positions(self, stat: str | None = None) -> list[str]:
        """Positions with at least one opportunity (for ``stat`` or any statistic)."""
        if stat is not None:
            found = set(self._positions.get(stat, set()))
        else:
            found = set()
            for positions in self._positions.values():
                found |= positions
        order = {p: i for i, p in enumerate(hh.POSITIONS)}
        return sorted(found, key=lambda p: (order.get(p, len(order)), p))

    def get(self, player: str, stat: str, position: str | None = None) -> tuple[int, int]:
        acc = self._counts.get((player, stat, position or ALL))
        return (acc[0], acc[1]) if acc else (0, 0)

    def frequency(self, player: str, stat: str, position: str | None = None) -> float | None:
        n, k = self.get(player, stat, position)
        return k / n if n else None

    def population(self, stat: str, position: str | None = None) -> list[tuple[str, int, int]]:
        """(player, n, k) for every player in the table (zero rows included)."""
        return [(p, *self.get(p, stat, position)) for p in self.players()]

    def truth(self, player: str, stat: str, position: str | None = None) -> tuple[float, int] | None:
        """(mean truth value, opportunities with truth) or ``None`` when unknown."""
        acc = self._truth.get((player, stat, position or ALL))
        if not acc or acc[1] == 0:
            return None
        return acc[0] / acc[1], int(acc[1])

    def sizing_counts(self, player: str, street: str, kind: str = "bet") -> dict[str, int]:
        found = self._sizing.get((player, street, kind), Counter())
        return {b: int(found.get(b, 0)) for b in SIZE_BINS}

    def river_strengths(self, player: str) -> list[float]:
        """Strengths of the player's river bets that were shown at showdown."""
        return list(self._river.get(player, []))

    def value_threshold(self, player: str, quantile: float = 0.1) -> float | None:
        """Lower ``quantile`` of shown value-bet strengths (no uncertainty attached)."""
        values = [s for s in self.river_strengths(player) if s >= self.bluff_threshold]
        return empirical_quantile(values, quantile)

    def transitions(self, player: str, chain: str = "line") -> Counter[tuple[str, str]]:
        """Counts of (from "street:line", to "street:line") for ``chain`` line or cbet."""
        return Counter({(a, b): n for (c, a, b), n in self._transitions.get(player, Counter()).items() if c == chain})

    def allin_records(self, player: str) -> list[tuple[float, float]]:
        """(actual net, all-in EV net) in big blinds for hands with ``derived.allin``."""
        return list(self._allin.get(player, []))

    def summary(self, player: str) -> dict[str, Any]:
        """JSON-friendly counters of one player."""
        stats: dict[str, Any] = {}
        for stat in self.stats():
            entry = {}
            for pos in [ALL, *self.positions(stat)]:
                n, k = self.get(player, stat, pos)
                if n:
                    entry[pos] = {"n": n, "k": k, "freq": k / n}
            if entry:
                stats[stat] = entry
        sizing = {
            f"{street}:{kind}": self.sizing_counts(player, street, kind)
            for (owner, street, kind) in self._sizing
            if owner == player
        }
        return {
            "player": player,
            "hands": self.hands.get(player, 0),
            "net_bb": self.net_bb.get(player, 0.0),
            "stats": stats,
            "sizing": sizing,
            "river_strengths": self.river_strengths(player),
        }


def compute_stats(hands: Iterable[hh.Hand], *, bluff_threshold: float = DEFAULT_BLUFF_THRESHOLD) -> StatsTable:
    """Build a :class:`StatsTable` from hands."""
    return StatsTable.from_hands(hands, bluff_threshold=bluff_threshold)


def definitions_markdown() -> str:
    """The definitions as a markdown table (one row per statistic key)."""
    rows = ["| key | street | opportunity | success | truth |", "|---|---|---|---|---|"]
    for d in STAT_DEFS.values():
        rows.append(f"| `{d.key}` | {d.street} | {d.opportunity} | {d.success} | {d.truth} |")
    return "\n".join(rows)
