"""A tiny ``pe-hh/1`` hand generator used only by the tests.

It is *not* the project's simulator (that is ``pe hh simulate`` in Rust). Bots
play 6-max with fixed, card-independent action probabilities per situation
and record them as ``truth`` on every action, which is enough to exercise the
evaluation pipeline end to end and to cross-check the replay's chip
accounting against an independent implementation. Stacks are deep
(1000 big blinds) so nobody is ever all-in.
"""

from __future__ import annotations

import random
from typing import Any

from poker_ecosystem.hh import derive_positions

STYLES: dict[str, dict[str, float]] = {
    "nit": {"open": 0.10, "limp": 0.02, "call": 0.06, "threebet": 0.03, "fold3b": 0.75, "cbet": 0.50,
            "barrel": 0.35, "stab": 0.08, "fold_bet": 0.60, "raise_bet": 0.04},
    "TAG": {"open": 0.20, "limp": 0.02, "call": 0.10, "threebet": 0.08, "fold3b": 0.55, "cbet": 0.70,
            "barrel": 0.55, "stab": 0.20, "fold_bet": 0.45, "raise_bet": 0.10},
    "LAG": {"open": 0.34, "limp": 0.03, "call": 0.16, "threebet": 0.15, "fold3b": 0.40, "cbet": 0.80,
            "barrel": 0.65, "stab": 0.35, "fold_bet": 0.35, "raise_bet": 0.18},
    "station": {"open": 0.07, "limp": 0.30, "call": 0.45, "threebet": 0.02, "fold3b": 0.25, "cbet": 0.35,
                "barrel": 0.25, "stab": 0.10, "fold_bet": 0.15, "raise_bet": 0.03},
}
OPEN_FACTOR = {"UTG": 0.6, "HJ": 0.8, "CO": 1.0, "BTN": 1.5, "SB": 0.9, "BB": 0.6}
PREFLOP_ORDER = ("UTG", "HJ", "CO", "BTN", "SB", "BB")
POSTFLOP_ORDER = ("SB", "BB", "UTG", "HJ", "CO", "BTN")
SB, BB, STACK = 500, 1000, 1_000_000
DECK = [r + s for r in "23456789TJQKA" for s in "cdhs"]


def make_players(per_style: int = 3, seed: int = 7) -> list[dict[str, Any]]:
    """Bots of every style with +/-15% jitter on each profile parameter."""
    rng = random.Random(seed)
    players = []
    for style, params in STYLES.items():
        for i in range(per_style):
            profile = {k: round(min(0.95, max(0.01, v * rng.uniform(0.85, 1.15))), 4) for k, v in params.items()}
            players.append({"name": f"{style}-{i}", "style": style, "profile": profile})
    return players


def _normalise(probs: dict[str, float]) -> dict[str, float]:
    clean = {k: max(0.0, v) for k, v in probs.items()}
    total = sum(clean.values())
    return {k: round(v / total, 6) for k, v in clean.items()}


def _sample(rng: random.Random, probs: dict[str, float]) -> str:
    r = rng.random() * sum(probs.values())
    acc = 0.0
    chosen = ""
    for action, p in probs.items():
        if p <= 0:
            continue
        acc += p
        chosen = action
        if r < acc:
            break
    return chosen


def _preflop_probs(prof: dict[str, float], position: str, to_call: int, raises: int, is_opener: bool) -> dict[str, float]:
    if raises == 0:
        if to_call == 0:
            p = min(0.5, prof["open"] * 0.4)
            return {"check": 1.0 - p, "raise": p}
        p_raise = min(0.9, prof["open"] * OPEN_FACTOR[position])
        p_call = min(prof["limp"], 1.0 - p_raise)
        return {"fold": 1.0 - p_raise - p_call, "call": p_call, "raise": p_raise}
    if raises == 1:
        p3 = prof["threebet"]
        pc = min(prof["call"], 1.0 - p3)
        return {"fold": 1.0 - p3 - pc, "call": pc, "raise": p3}
    if is_opener and raises == 2:
        return {"fold": prof["fold3b"], "call": 1.0 - prof["fold3b"]}
    return {"fold": 0.9, "call": 0.1}


def _postflop_probs(prof: dict[str, float], street: str, to_call: int, raises: int, was_aggressor: bool) -> dict[str, float]:
    if to_call == 0:
        if was_aggressor:
            p = prof["cbet"] if street == "flop" else prof["barrel"]
        else:
            p = prof["stab"]
        return {"check": 1.0 - p, "bet": p}
    if raises == 1:
        f, r = prof["fold_bet"], prof["raise_bet"]
        return {"fold": f, "call": 1.0 - f - r, "raise": r}
    return {"fold": 0.5, "call": 0.5}


class _Game:
    def __init__(self, rng: random.Random, seats: list[int], positions: dict[int, str], profiles: dict[int, dict[str, float]]) -> None:
        self.rng = rng
        self.seats = seats
        self.positions = positions
        self.profiles = profiles
        self.contrib = {s: 0 for s in seats}
        self.street = {s: 0 for s in seats}
        self.folded: set[int] = set()
        self.pot = 0
        self.streets: list[dict[str, Any]] = []
        self.aggressor: dict[str, int | None] = {}
        self.uncalled: list[dict[str, int]] = []

    def alive(self) -> list[int]:
        return [s for s in self.seats if s not in self.folded]

    def put(self, seat: int, amount: int) -> int:
        amount = min(amount, STACK - self.contrib[seat])
        self.contrib[seat] += amount
        self.street[seat] += amount
        self.pot += amount
        return amount

    def raise_to(self, name: str, level: int, raises: int) -> int:
        if name == "preflop":
            if raises == 0:
                limpers = sum(1 for s in self.seats if self.street[s] == BB and self.positions[s] != "BB")
                return 2500 + BB * limpers
            return 3 * level
        if level == 0:
            return max(BB, int(self.pot * 0.6) // 10 * 10)
        return 3 * level

    def probs(self, name: str, seat: int, to_call: int, raises: int, is_opener: bool) -> dict[str, float]:
        prof = self.profiles[seat]
        if name == "preflop":
            return _preflop_probs(prof, self.positions[seat], to_call, raises, is_opener)
        previous = {"flop": "preflop", "turn": "flop", "river": "turn"}[name]
        return _postflop_probs(prof, name, to_call, raises, self.aggressor.get(previous) == seat)

    def betting(self, name: str, order: list[int]) -> None:
        actions: list[dict[str, Any]] = []
        level = max(self.street.values())
        raises = 0
        last: int | None = None
        opener: int | None = None
        queue = [s for s in order if s not in self.folded]
        while queue and len(self.alive()) > 1:
            seat = queue.pop(0)
            if seat in self.folded:
                continue
            to_call = level - self.street[seat]
            probs = _normalise(self.probs(name, seat, to_call, raises, seat == opener))
            choice = _sample(self.rng, probs)
            amount = 0
            if choice == "call":
                amount = self.put(seat, to_call)
            elif choice in ("bet", "raise"):
                amount = self.put(seat, self.raise_to(name, level, raises) - self.street[seat])
                level = self.street[seat]
                raises += 1
                last = seat
                if opener is None:
                    opener = seat
                i = order.index(seat)
                queue = [s for s in order[i + 1:] + order[:i] if s not in self.folded]
            elif choice == "fold":
                self.folded.add(seat)
            actions.append({
                "seat": seat, "type": choice, "amount": amount, "to": self.street[seat],
                "all_in": self.contrib[seat] >= STACK, "truth": probs,
            })
        self.aggressor[name] = last
        self.streets.append({"name": name, "actions": actions})
        self.return_uncalled()

    def return_uncalled(self) -> None:
        ranked = sorted(self.street.items(), key=lambda kv: kv[1], reverse=True)
        if len(ranked) >= 2 and ranked[0][1] > ranked[1][1]:
            seat, top = ranked[0]
            excess = top - ranked[1][1]
            self.pot -= excess
            self.contrib[seat] -= excess
            self.street[seat] -= excess
            self.uncalled.append({"seat": seat, "amount": excess})


def _play(rng: random.Random, table: int, number: int, seated: list[dict[str, Any]]) -> dict[str, Any]:
    seats = list(range(1, len(seated) + 1))
    button = seats[number % len(seats)]
    positions = derive_positions(seats, button)
    by_pos = {p: s for s, p in positions.items()}
    game = _Game(rng, seats, positions, {s: seated[s - 1]["profile"] for s in seats})
    posts = [{"seat": by_pos["SB"], "kind": "sb", "amount": SB}, {"seat": by_pos["BB"], "kind": "bb", "amount": BB}]
    game.put(by_pos["SB"], SB)
    game.put(by_pos["BB"], BB)
    cards = rng.sample(DECK, 2 * len(seats) + 5)
    hole = {s: cards[2 * i: 2 * i + 2] for i, s in enumerate(seats)}
    board = cards[2 * len(seats):]
    game.betting("preflop", [by_pos[p] for p in PREFLOP_ORDER])
    dealt = 0
    for name, n_cards in (("flop", 3), ("turn", 4), ("river", 5)):
        if len(game.alive()) < 2:
            break
        dealt = n_cards
        game.street = {s: 0 for s in seats}
        game.betting(name, [by_pos[p] for p in POSTFLOP_ORDER])
    alive = game.alive()
    derived: dict[str, Any] = {"ok": True, "issues": [], "players_dealt": len(seats), "preflop_aggressor": game.aggressor.get("preflop")}
    collected, shown = [], []
    if len(alive) >= 2:
        strength = {s: round(rng.random(), 4) for s in alive}
        best = max(strength.values())
        winners = [s for s in alive if strength[s] == best]
        share, rest = divmod(game.pot, len(winners))
        collected = [{"seat": s, "amount": share + (rest if i == 0 else 0), "pot": 0} for i, s in enumerate(winners)]
        derived["showdown_seats"] = alive
        derived["river_strength"] = {str(s): v for s, v in strength.items()}
        shown = [{"seat": s, "cards": hole[s], "description": ""} for s in alive]
    else:
        collected = [{"seat": alive[0], "amount": game.pot, "pot": 0}]
        derived["showdown_seats"] = []
    return {
        "schema": "pe-hh/1", "id": f"toy-{table}-{number}", "site": "Synthetic", "game": "NLH", "limit": "NL",
        "currency": "chips", "time": "", "table": f"toy-{table}", "max_seats": len(seats), "button": button,
        "sb": SB, "bb": BB, "ante": 0, "bomb_pot": False, "board_kind": "single",
        "players": [
            {"seat": s, "name": seated[s - 1]["name"], "stack": STACK, "position": positions[s], "hole": hole[s],
             "hero": False, "sitting_out": False}
            for s in seats
        ],
        "posts": posts, "streets": game.streets, "boards": [board[:dealt]] if dealt else [], "shown": shown,
        "collected": collected, "uncalled": game.uncalled, "total_pot": game.pot, "rake": 0,
        "derived": derived, "synthetic": True,
    }


def simulate(players: list[dict[str, Any]], hands_per_table: int, *, seed: int = 11, table_size: int = 6) -> list[dict[str, Any]]:
    """Seat the bots at tables of ``table_size`` (shuffled) and play hands."""
    rng = random.Random(seed)
    order = list(players)
    rng.shuffle(order)
    tables = [order[i:i + table_size] for i in range(0, len(order), table_size)]
    hands = []
    for t, seated in enumerate(tables):
        if len(seated) != table_size:
            continue
        for number in range(hands_per_table):
            hands.append(_play(rng, t, number, seated))
    return hands
