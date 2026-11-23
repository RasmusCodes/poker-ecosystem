"""Leak finder: compare a player's posteriors with a reference band.

For each statistic and position the player's posterior Beta(a, b) (from
:class:`~poker_ecosystem.opponent.OpponentModel`) is compared with a band
[lo, hi] of acceptable true frequencies:

* **population** (default): the ``quantiles`` (default 10% and 90%) of the
  fitted population (or population-by-position) Beta prior, i.e. the range
  of the middle 80% of players in this pool;
* **baseline**: a JSON object of target frequencies per statistic, either a
  single band for all positions or one per position, e.g.
  ``{"vpip": [0.20, 0.28], "rfi": {"UTG": [0.12, 0.18], "BTN": 0.45}}``. A
  bare number is a point target widened by +/- ``tolerance``; the key
  ``"ALL"`` is the pooled band.

Reported per statistic/position: P(theta < lo), P(theta > hi) and their sum
P(outside), the direction, the sample size, the expected distance outside
the band E[(lo - theta)+ + (theta - hi)+] (closed form with incomplete beta
functions), and an **EV-impact rank that is a heuristic**: expected distance
x opportunities per hand x a street weight (preflop 1, flop 1.5, turn 2,
river 3, showdown 1.5, reflecting typical pot growth). It orders leaks for
review; it is not an estimate of chips lost.

All-in luck uses ``derived.allin.ev_net``: luck = actual net - all-in EV net,
summed in big blinds over the player's all-in hands.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import hh
from .bayes import betainc
from .opponent import APPROXIMATIONS, OpponentModel
from .stats import ALL, STAT_DEFS, StatsTable, hand_observations

STREET_WEIGHT: dict[str, float] = {"preflop": 1.0, "flop": 1.5, "turn": 2.0, "river": 3.0, "showdown": 1.5}
HEURISTIC_NOTE = (
    "ev_impact_heuristic = expected distance outside the band x opportunities per hand x street weight "
    "(x100). A ranking aid only; it is not an EV estimate."
)

Baseline = Mapping[str, Any]


@dataclass(frozen=True)
class Leak:
    """One statistic/position compared with its reference band."""

    stat: str
    position: str
    mean: float
    lo: float
    hi: float
    n: int
    band: tuple[float, float]
    band_source: str
    p_low: float
    p_high: float
    expected_excess: float
    ev_impact_heuristic: float
    rank: int = 0

    @property
    def p_outside(self) -> float:
        return self.p_low + self.p_high

    @property
    def direction(self) -> str:
        return "high" if self.p_high >= self.p_low else "low"

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["band"] = list(self.band)
        out["p_outside"] = self.p_outside
        out["direction"] = self.direction
        return out


def load_baseline(path: str | Path) -> dict[str, Any]:
    """Read a baseline JSON file of target frequencies."""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("a baseline must be a JSON object keyed by statistic")
    return data


def _as_band(value: Any, tolerance: float) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        v = float(value)
        return max(0.0, v - tolerance), min(1.0, v + tolerance)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lo, hi = float(value[0]), float(value[1])
        if not 0.0 <= lo <= hi <= 1.0:
            raise ValueError(f"invalid band {value!r}")
        return lo, hi
    raise ValueError(f"a target must be a number or [lo, hi], got {value!r}")


def reference_band(
    model: OpponentModel,
    stat: str,
    position: str | None = None,
    baseline: Baseline | None = None,
    *,
    quantiles: tuple[float, float] = (0.1, 0.9),
    tolerance: float = 0.03,
) -> tuple[tuple[float, float], str] | None:
    """(band, source) for a statistic/position, or ``None`` when no reference exists."""
    pos = position or ALL
    if baseline is not None:
        target = baseline.get(stat)
        if target is None:
            return None
        if isinstance(target, Mapping):
            target = target.get(pos)
            if target is None:
                return None
        return _as_band(target, tolerance), "baseline"
    prior = model.population_prior(stat, None if pos == ALL else pos)
    if prior.degenerate:
        return None
    return (prior.quantile(quantiles[0]), prior.quantile(quantiles[1])), "population"


def expected_excess(a: float, b: float, lo: float, hi: float) -> float:
    """E[(lo - theta)+ + (theta - hi)+] for theta ~ Beta(a, b)."""
    mean = a / (a + b)
    below = lo * betainc(a, b, lo) - mean * betainc(a + 1.0, b, lo) if lo > 0 else 0.0
    above = mean * (1.0 - betainc(a + 1.0, b, hi)) - hi * (1.0 - betainc(a, b, hi)) if hi < 1 else 0.0
    return max(below, 0.0) + max(above, 0.0)


def find_leaks(
    model: OpponentModel,
    player: str,
    *,
    baseline: Baseline | None = None,
    stats: Iterable[str] | None = None,
    positions: bool = True,
    quantiles: tuple[float, float] = (0.1, 0.9),
    tolerance: float = 0.03,
    min_opportunities: int = 1,
) -> list[Leak]:
    """Compare every statistic (and position) with its band, ranked by the heuristic."""
    hands = max(model.table.hands.get(player, 0), 1)
    found: list[Leak] = []
    for stat in stats or model.stats():
        targets = [ALL, *(model.positions(stat) if positions else [])]
        for pos in targets:
            est = model.estimate(player, stat, None if pos == ALL else pos)
            if est.n < min_opportunities:
                continue
            ref = reference_band(model, stat, pos, baseline, quantiles=quantiles, tolerance=tolerance)
            if ref is None:
                continue
            (lo, hi), source = ref
            p_low = betainc(est.alpha, est.beta, lo) if lo > 0 else 0.0
            p_high = 1.0 - betainc(est.alpha, est.beta, hi) if hi < 1 else 0.0
            excess = expected_excess(est.alpha, est.beta, lo, hi)
            weight = STREET_WEIGHT.get(STAT_DEFS[stat].street if stat in STAT_DEFS else "", 1.0)
            impact = 100.0 * excess * (est.n / hands) * weight
            found.append(Leak(stat, pos, est.mean, est.lo, est.hi, est.n, (lo, hi), source, p_low, p_high, excess, impact))
    found.sort(key=lambda leak: (-leak.ev_impact_heuristic, -leak.p_outside, leak.stat, leak.position))
    return [replace(leak, rank=i + 1) for i, leak in enumerate(found)]


@dataclass(frozen=True)
class AllinLuck:
    """Actual versus all-in-EV results of one player, in big blinds."""

    player: str
    hands: int
    actual_bb: float
    ev_bb: float

    @property
    def luck_bb(self) -> float:
        return self.actual_bb - self.ev_bb

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["luck_bb"] = self.luck_bb
        return out


def allin_luck(source: StatsTable | Iterable[hh.Hand], player: str) -> AllinLuck:
    """All-in luck from a :class:`StatsTable` or directly from hands."""
    if isinstance(source, StatsTable):
        records = source.allin_records(player)
    else:
        records = [(net, ev) for hand in source for name, net, ev in hand_observations(hand).allin if name == player]
    return AllinLuck(player, len(records), sum(r[0] for r in records), sum(r[1] for r in records))


def leak_report(
    model: OpponentModel,
    player: str,
    *,
    baseline: Baseline | None = None,
    top: int | None = 20,
    min_p_outside: float = 0.0,
    **options: Any,
) -> dict[str, Any]:
    """JSON-ready leak report with explicit heuristic labelling."""
    leaks = [leak for leak in find_leaks(model, player, baseline=baseline, **options) if leak.p_outside >= min_p_outside]
    return {
        "player": player,
        "hands": model.table.hands.get(player, 0),
        "reference": "baseline" if baseline is not None else "population",
        "level": model.level,
        "leaks": [leak.as_dict() for leak in (leaks[:top] if top else leaks)],
        "allin_luck": allin_luck(model.table, player).as_dict(),
        "notes": [HEURISTIC_NOTE, *APPROXIMATIONS],
    }
