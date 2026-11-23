"""Hand-written ``pe-hh/1`` fixtures with hand-checked chip accounting.

Blinds are 500/1000 milli unless stated. Every hand's ``total_pot`` equals
posts + actions - uncalled bets, which the replay tests verify.

=============  ===================================================================
id             scenario
=============  ===================================================================
HU-1           heads-up: BTN opens, BB calls, BB checks, BTN c-bets, BB folds.
               Synthetic, with ``truth`` on every action.
6M-3BET        6-max: HJ opens, BTN 3-bets, everyone folds (no ``derived``).
6M-BARREL      CO limps, BTN isolates, CO calls; flop c-bet called, turn barrel
               folded. Synthetic; ``truth`` only on the CO's actions.
6M-XR          UTG opens, BB calls; BB check-raises the flop, leads the turn.
3H-SD          3 dealt (+1 sitting out): BTN steals, BB calls; flop checks
               through; BB bets turn and river, called; river showdown with
               ``derived.river_strength`` (BB bluffed).
BOMB-1         PLO bomb pot: bomb antes, no preflop street, paired flop.
STRADDLE-1     UTG straddles, two limps, straddler checks his option; monotone
               flop checked through, straddler wins with a turn bet.
ALLIN-1        UTG opens, BTN 3-bets, UTG 4-bet jams, BTN calls: ``derived.allin``.
HU-DONK        heads-up on a 9-max table (seats 3 and 7): BB donks, BTN raises.
=============  ===================================================================
"""

from __future__ import annotations

import copy
from typing import Any

Hand = dict[str, Any]


def _player(seat: int, name: str, position: str | None, stack: int = 100_000, hole: list[str] | None = None, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"seat": seat, "name": name, "stack": stack, "hole": hole, "hero": False, "sitting_out": False}
    if position is not None:
        out["position"] = position
    out.update(extra)
    return out


def _act(seat: int, kind: str, amount: int = 0, to: int = 0, *, all_in: bool = False, truth: dict[str, float] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"seat": seat, "type": kind, "amount": amount, "to": to, "all_in": all_in}
    if truth is not None:
        out["truth"] = truth
    return out


def _hand(hand_id: str, **fields: Any) -> Hand:
    base: Hand = {
        "schema": "pe-hh/1",
        "id": hand_id,
        "site": "PokerStars",
        "game": "NLH",
        "limit": "NL",
        "currency": "USD",
        "time": "2026/11/05 21:00:00 ET",
        "table": "Fixture I",
        "max_seats": 6,
        "sb": 500,
        "bb": 1000,
        "ante": 0,
        "bomb_pot": False,
        "board_kind": "single",
        "boards": [],
        "shown": [],
        "collected": [],
        "uncalled": [],
        "rake": 0,
        "synthetic": False,
        "unknown_future_field": {"readers": "must ignore this"},
    }
    base.update(fields)
    return base


HU_1 = _hand(
    "HU-1",
    site="Synthetic",
    synthetic=True,
    max_seats=2,
    button=1,
    players=[
        _player(1, "Alice", "BTN", hole=["As", "Kd"]),
        _player(2, "Bob", "BB", hole=["7h", "7c"]),
    ],
    posts=[{"seat": 1, "kind": "sb", "amount": 500}, {"seat": 2, "kind": "bb", "amount": 1000}],
    streets=[
        {"name": "preflop", "actions": [
            _act(1, "raise", 2500, 3000, truth={"fold": 0.1, "call": 0.2, "raise": 0.7}),
            _act(2, "call", 2000, 3000, truth={"fold": 0.3, "call": 0.5, "raise": 0.2}),
        ]},
        {"name": "flop", "actions": [
            _act(2, "check", truth={"check": 0.8, "bet": 0.2}),
            _act(1, "bet", 3000, 3000, truth={"check": 0.4, "bet": 0.6}),
            _act(2, "fold", truth={"fold": 0.5, "call": 0.4, "raise": 0.1}),
        ]},
    ],
    boards=[["Ah", "8c", "2d"]],
    collected=[{"seat": 1, "amount": 6000, "pot": 0}],
    uncalled=[{"seat": 1, "amount": 3000}],
    total_pot=6000,
    derived={
        "ok": True, "issues": [], "players_dealt": 2, "preflop_aggressor": 1, "showdown_seats": [],
        "contrib": {"1": 3000, "2": 3000}, "net": {"1": 3000, "2": -3000}, "pots": [{"amount": 6000, "eligible": [1, 2]}],
        "allin": None, "river_strength": {},
    },
)

SIXMAX_3BET = _hand(
    "6M-3BET",
    button=4,
    players=[
        _player(1, "Carol", "UTG"),
        _player(2, "Dave", "HJ"),
        _player(3, "Erin", "CO"),
        _player(4, "Alice", "BTN", hole=["Ks", "Kh"], hero=True),
        _player(5, "Bob", "SB"),
        _player(6, "Frank", "BB"),
    ],
    posts=[{"seat": 5, "kind": "sb", "amount": 500}, {"seat": 6, "kind": "bb", "amount": 1000}],
    streets=[
        {"name": "preflop", "actions": [
            _act(1, "fold"),
            _act(2, "raise", 2500, 2500),
            _act(3, "fold"),
            _act(4, "raise", 8000, 8000),
            _act(5, "fold"),
            _act(6, "fold"),
            _act(2, "fold"),
        ]},
    ],
    collected=[{"seat": 4, "amount": 6500, "pot": 0}],
    uncalled=[{"seat": 4, "amount": 5500}],
    total_pot=6500,
)

SIXMAX_BARREL = _hand(
    "6M-BARREL",
    site="Synthetic",
    synthetic=True,
    button=1,
    players=[
        _player(1, "Alice", "BTN", hole=["Qc", "Qd"]),
        _player(2, "Bob", "SB", hole=["4c", "5c"]),
        _player(3, "Carol", "BB", hole=["Js", "Ts"]),
        _player(4, "Dave", "UTG", hole=["Ad", "6h"]),
        _player(5, "Erin", "HJ", hole=["7s", "7d"]),
        _player(6, "Frank", "CO", hole=["9h", "9d"]),
    ],
    posts=[{"seat": 2, "kind": "sb", "amount": 500}, {"seat": 3, "kind": "bb", "amount": 1000}],
    streets=[
        {"name": "preflop", "actions": [
            _act(4, "fold"),
            _act(5, "fold"),
            _act(6, "call", 1000, 1000, truth={"fold": 0.3, "call": 0.5, "raise": 0.2}),
            _act(1, "raise", 4000, 4000),
            _act(2, "fold"),
            _act(3, "fold"),
            _act(6, "call", 3000, 4000, truth={"fold": 0.2, "call": 0.6, "raise": 0.2}),
        ]},
        {"name": "flop", "actions": [
            _act(6, "check", truth={"check": 0.9, "bet": 0.1}),
            _act(1, "bet", 5000, 5000),
            _act(6, "call", 5000, 5000, truth={"fold": 0.3, "call": 0.6, "raise": 0.1}),
        ]},
        {"name": "turn", "actions": [
            _act(6, "check", truth={"check": 1.0}),
            _act(1, "bet", 15000, 15000),
            _act(6, "fold", truth={"fold": 0.7, "call": 0.3}),
        ]},
    ],
    boards=[["Kd", "8s", "3h", "2c"]],
    collected=[{"seat": 1, "amount": 19500, "pot": 0}],
    uncalled=[{"seat": 1, "amount": 15000}],
    total_pot=19500,
    derived={"ok": True, "issues": [], "preflop_aggressor": 1, "showdown_seats": []},
)

SIXMAX_XR = _hand(
    "6M-XR",
    button=3,
    players=[
        _player(1, "Erin", "HJ"),
        _player(2, "Frank", "CO"),
        _player(3, "Carol", "BTN"),
        _player(4, "Dave", "SB"),
        _player(5, "Alice", "BB", hole=["Th", "Tc"], hero=True),
        _player(6, "Bob", "UTG"),
    ],
    posts=[{"seat": 4, "kind": "sb", "amount": 500}, {"seat": 5, "kind": "bb", "amount": 1000}],
    streets=[
        {"name": "preflop", "actions": [
            _act(6, "raise", 3000, 3000),
            _act(1, "fold"),
            _act(2, "fold"),
            _act(3, "fold"),
            _act(4, "fold"),
            _act(5, "call", 2000, 3000),
        ]},
        {"name": "flop", "actions": [
            _act(5, "check"),
            _act(6, "bet", 4000, 4000),
            _act(5, "raise", 12000, 12000),
            _act(6, "call", 8000, 12000),
        ]},
        {"name": "turn", "actions": [
            _act(5, "bet", 20000, 20000),
            _act(6, "fold"),
        ]},
    ],
    boards=[["Th", "9h", "2c", "5s"]],
    collected=[{"seat": 5, "amount": 30500, "pot": 0}],
    uncalled=[{"seat": 5, "amount": 20000}],
    total_pot=30500,
    derived={"ok": True, "issues": [], "preflop_aggressor": 6, "showdown_seats": []},
)

THREE_HANDED_SHOWDOWN = _hand(
    "3H-SD",
    button=1,
    rake=500,
    players=[
        _player(1, "Alice", "BTN", hole=["Qh", "9h"]),
        _player(2, "Bob", "SB"),
        {"seat": 3, "name": "Zed", "stack": 50_000, "hole": None, "hero": False, "sitting_out": True},
        _player(4, "Carol", "BB", hole=["Js", "Ts"]),
    ],
    posts=[{"seat": 2, "kind": "sb", "amount": 500}, {"seat": 4, "kind": "bb", "amount": 1000}],
    streets=[
        {"name": "preflop", "actions": [
            _act(1, "raise", 2500, 2500),
            _act(2, "fold"),
            _act(4, "call", 1500, 2500),
        ]},
        {"name": "flop", "actions": [_act(4, "check"), _act(1, "check")]},
        {"name": "turn", "actions": [_act(4, "bet", 3000, 3000), _act(1, "call", 3000, 3000)]},
        {"name": "river", "actions": [_act(4, "bet", 11500, 11500), _act(1, "call", 11500, 11500)]},
    ],
    boards=[["Qs", "8d", "4c", "2h", "7s"]],
    shown=[
        {"seat": 4, "cards": ["Js", "Ts"], "description": "high card Queen"},
        {"seat": 1, "cards": ["Qh", "9h"], "description": "a pair of Queens"},
    ],
    collected=[{"seat": 1, "amount": 34000, "pot": 0}],
    total_pot=34500,
    derived={
        "ok": True, "issues": [], "players_dealt": 3, "preflop_aggressor": 1, "showdown_seats": [1, 4],
        "net": {"1": 17000, "2": -500, "4": -17000}, "river_strength": {"1": 0.85, "4": 0.2},
    },
)

BOMB_POT = _hand(
    "BOMB-1",
    game="PLO",
    limit="PL",
    bomb_pot=True,
    button=2,
    players=[
        _player(1, "Dave", "CO"),
        _player(2, "Erin", "BTN"),
        _player(3, "Frank", "SB"),
        _player(4, "Alice", "BB", hole=["As", "Ah", "Kd", "Qs"], hero=True),
    ],
    posts=[{"seat": s, "kind": "bomb", "amount": 2000} for s in (1, 2, 3, 4)],
    streets=[
        {"name": "flop", "actions": [
            _act(3, "check"),
            _act(4, "bet", 4000, 4000),
            _act(1, "fold"),
            _act(2, "call", 4000, 4000),
            _act(3, "fold"),
        ]},
        {"name": "turn", "actions": [_act(4, "check"), _act(2, "check")]},
        {"name": "river", "actions": [_act(4, "check"), _act(2, "bet", 16000, 16000), _act(4, "fold")]},
    ],
    boards=[["9c", "9d", "5s", "Kh", "2c"]],
    collected=[{"seat": 2, "amount": 16000, "pot": 0}],
    uncalled=[{"seat": 2, "amount": 16000}],
    total_pot=16000,
)

STRADDLE = _hand(
    "STRADDLE-1",
    button=6,
    players=[
        _player(1, "Bob", "SB"),
        _player(2, "Carol", "BB"),
        _player(3, "Dave", "UTG"),
        _player(4, "Erin", "HJ"),
        _player(5, "Frank", "CO"),
        _player(6, "Alice", "BTN", hole=["8c", "8d"], hero=True),
    ],
    posts=[
        {"seat": 1, "kind": "sb", "amount": 500},
        {"seat": 2, "kind": "bb", "amount": 1000},
        {"seat": 3, "kind": "straddle", "amount": 2000},
    ],
    streets=[
        {"name": "preflop", "actions": [
            _act(4, "fold"),
            _act(5, "call", 2000, 2000),
            _act(6, "call", 2000, 2000),
            _act(1, "fold"),
            _act(2, "fold"),
            _act(3, "check", 0, 2000),
        ]},
        {"name": "flop", "actions": [_act(3, "check"), _act(5, "check"), _act(6, "check")]},
        {"name": "turn", "actions": [_act(3, "bet", 5000, 5000), _act(5, "fold"), _act(6, "fold")]},
    ],
    boards=[["Ah", "Kh", "Qh", "3d"]],
    collected=[{"seat": 3, "amount": 7500, "pot": 0}],
    uncalled=[{"seat": 3, "amount": 5000}],
    total_pot=7500,
)

ALLIN = _hand(
    "ALLIN-1",
    button=5,
    players=[
        _player(1, "Frank", "BB"),
        _player(2, "Alice", "UTG", stack=50_000, hole=["Ah", "Ad"]),
        _player(3, "Bob", "HJ"),
        _player(4, "Carol", "CO"),
        _player(5, "Dave", "BTN", hole=["Ks", "Kd"]),
        _player(6, "Erin", "SB"),
    ],
    posts=[{"seat": 6, "kind": "sb", "amount": 500}, {"seat": 1, "kind": "bb", "amount": 1000}],
    streets=[
        {"name": "preflop", "actions": [
            _act(2, "raise", 2500, 2500),
            _act(3, "fold"),
            _act(4, "fold"),
            _act(5, "raise", 8000, 8000),
            _act(6, "fold"),
            _act(1, "fold"),
            _act(2, "raise", 47500, 50000, all_in=True),
            _act(5, "call", 42000, 50000),
        ]},
    ],
    boards=[["Kc", "7d", "2s", "9h", "3c"]],
    shown=[
        {"seat": 2, "cards": ["Ah", "Ad"], "description": "a pair of Aces"},
        {"seat": 5, "cards": ["Ks", "Kd"], "description": "three of a kind, Kings"},
    ],
    collected=[{"seat": 5, "amount": 101500, "pot": 0}],
    total_pot=101500,
    derived={
        "ok": True, "issues": [], "players_dealt": 6, "preflop_aggressor": 2, "showdown_seats": [2, 5],
        "net": {"1": -1000, "2": -50000, "5": 51500, "6": -500},
        "allin": {"street": "preflop", "seats": [2, 5], "equity": {"2": 0.82, "5": 0.18}, "ev_net": {"2": 33230, "5": -31730}},
        "river_strength": {"2": 0.62, "5": 0.97},
    },
)

HU_DONK = _hand(
    "HU-DONK",
    max_seats=9,
    button=3,
    players=[
        _player(3, "Carol", "BTN"),
        _player(7, "Dave", "BB"),
    ],
    posts=[{"seat": 3, "kind": "sb", "amount": 500}, {"seat": 7, "kind": "bb", "amount": 1000}],
    streets=[
        {"name": "preflop", "actions": [_act(3, "raise", 2000, 2500), _act(7, "call", 1500, 2500)]},
        {"name": "flop", "actions": [
            _act(7, "bet", 2500, 2500),
            _act(3, "raise", 10000, 10000),
            _act(7, "fold"),
        ]},
    ],
    boards=[["Jc", "Tc", "9d"]],
    collected=[{"seat": 3, "amount": 10000, "pot": 0}],
    uncalled=[{"seat": 3, "amount": 7500}],
    total_pot=10000,
)

_ALL = (HU_1, SIXMAX_3BET, SIXMAX_BARREL, SIXMAX_XR, THREE_HANDED_SHOWDOWN, BOMB_POT, STRADDLE, ALLIN, HU_DONK)


def all_hands() -> list[Hand]:
    """Fresh deep copies of every fixture, in a fixed order."""
    return [copy.deepcopy(h) for h in _ALL]


def by_id(hand_id: str) -> Hand:
    for h in _ALL:
        if h["id"] == hand_id:
            return copy.deepcopy(h)
    raise KeyError(hand_id)
