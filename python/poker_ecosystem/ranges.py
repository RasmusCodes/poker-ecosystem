"""Hold'em starting-hand classes and a strength ordering for range estimates.

The 169 classes are pairs (``"AA"``, 6 combos), suited hands (``"AKs"``, 4)
and offsuit hands (``"AKo"``, 12); 1326 combos in total.

:func:`load_ranking` reads ``research/vectors/preflop_ranking.json`` (a JSON
list of the 169 class names, strongest first) when the file exists. Otherwise
it falls back to the **Chen formula** ordering, a crude hand-strength score:

* highest card: A = 10, K = 8, Q = 7, J = 6, T..2 = half the face value;
* pairs: double that, minimum 5;
* suited: +2;
* gap between the ranks: 0 -> 0, 1 -> -1, 2 -> -2, 3 -> -4, 4+ -> -5;
* +1 for a 0- or 1-gap hand whose highest card is below a queen.

Classes are ordered by the unrounded score; ties go to pairs, then suited
hands, then the higher top card and the higher kicker. The Chen ordering is an
approximation of real preflop strength (it ignores equity realisation and
position) and is only a placeholder until a solver-derived ranking exists.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

RANKS = "AKQJT98765432"
TOTAL_COMBOS = 1326
_VALUE = {r: 14 - i for i, r in enumerate(RANKS)}
_HIGH_POINTS = {14: 10.0, 13: 8.0, 12: 7.0, 11: 6.0}
_GAP_PENALTY = {0: 0.0, 1: 1.0, 2: 2.0, 3: 4.0}


def hand_classes() -> list[str]:
    """All 169 class names (pairs, then suited/offsuit pairs of ranks)."""
    out: list[str] = []
    for i, hi in enumerate(RANKS):
        for j, lo in enumerate(RANKS):
            if i == j:
                out.append(hi + lo)
            elif i < j:
                out.extend((hi + lo + "s", hi + lo + "o"))
    return out


def normalise_class(name: str) -> str:
    """Canonical class name (``"kas"`` -> ``"AKs"``); raises ``ValueError`` if invalid."""
    text = str(name).strip()
    if len(text) not in (2, 3):
        raise ValueError(f"not a hand class: {name!r}")
    r1, r2 = text[0].upper(), text[1].upper()
    if r1 not in _VALUE or r2 not in _VALUE:
        raise ValueError(f"not a hand class: {name!r}")
    if _VALUE[r2] > _VALUE[r1]:
        r1, r2 = r2, r1
    if r1 == r2:
        if len(text) != 2:
            raise ValueError(f"a pair takes no suffix: {name!r}")
        return r1 + r2
    suffix = text[2:].lower()
    if suffix not in ("s", "o"):
        raise ValueError(f"non-pair class needs an 's' or 'o' suffix: {name!r}")
    return r1 + r2 + suffix


def class_combos(name: str) -> int:
    """Number of card combinations of a class (6, 4 or 12)."""
    canon = normalise_class(name)
    if len(canon) == 2:
        return 6
    return 4 if canon[2] == "s" else 12


def combos_of(classes: Iterable[str]) -> int:
    return sum(class_combos(c) for c in classes)


def chen_score(name: str) -> float:
    """Unrounded Chen formula score of a class."""
    canon = normalise_class(name)
    hi, lo = _VALUE[canon[0]], _VALUE[canon[1]]
    points = _HIGH_POINTS.get(hi, hi / 2.0)
    if hi == lo:
        return max(points * 2.0, 5.0)
    score = points + (2.0 if canon[2] == "s" else 0.0)
    gap = hi - lo - 1
    score -= _GAP_PENALTY.get(gap, 5.0)
    if gap <= 1 and hi < 12:
        score += 1.0
    return score


def chen_points(name: str) -> int:
    """Chen score rounded up to a whole number, as in the published formula."""
    return math.ceil(chen_score(name))


def chen_ranking() -> list[str]:
    """The 169 classes, strongest first, by Chen score (deterministic tie-breaks)."""

    def key(name: str) -> tuple[float, int, int, int, int]:
        pair = len(name) == 2
        suited = not pair and name[2] == "s"
        return (-chen_score(name), not pair, not suited, -_VALUE[name[0]], -_VALUE[name[1]])

    return sorted(hand_classes(), key=key)


def default_ranking_path() -> Path:
    """``research/vectors/preflop_ranking.json`` in the repository."""
    return Path(__file__).resolve().parents[2] / "research" / "vectors" / "preflop_ranking.json"


def validate_ranking(names: Sequence[str]) -> list[str]:
    """Check that a ranking lists each of the 169 classes exactly once."""
    canon = [normalise_class(n) for n in names]
    if len(canon) != 169 or set(canon) != set(hand_classes()):
        raise ValueError("a preflop ranking must list each of the 169 classes exactly once")
    return canon


def load_ranking(path: str | Path | None = None) -> tuple[list[str], str]:
    """(ranking, source). Uses the JSON ranking when present, else the Chen fallback.

    A file that exists but is malformed raises ``ValueError`` rather than
    silently falling back.
    """
    target = Path(path) if path is not None else default_ranking_path()
    if target.is_file():
        data = json.loads(target.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            data = data.get("ranking") or data.get("classes")
        if not isinstance(data, list):
            raise ValueError(f"{target}: expected a JSON list of 169 class names")
        return validate_ranking(data), str(target)
    return chen_ranking(), "chen-formula"


def top_fraction(ranking: Sequence[str], fraction: float) -> list[str]:
    """Strongest classes covering ``fraction`` of the 1326 combos.

    Classes are taken in ranking order; a class is included while at least
    half of its combos still fit inside the combo budget, so the result is
    the prefix whose combo share is closest to ``fraction`` from below/above.
    """
    budget = min(max(fraction, 0.0), 1.0) * TOTAL_COMBOS
    chosen: list[str] = []
    used = 0
    for name in ranking:
        combos = class_combos(name)
        if used + combos / 2.0 > budget:
            break
        chosen.append(name)
        used += combos
    return chosen


def range_fraction(classes: Iterable[str]) -> float:
    """Share of the 1326 combos covered by ``classes``."""
    return combos_of(classes) / TOTAL_COMBOS
