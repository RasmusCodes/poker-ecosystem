"""Player clustering on posterior frequency features.

* **Features**: the logit of the pooled posterior mean of each statistic in
  :data:`FEATURE_STATS`, standardised to z-scores across the clustered players
  (a feature with no spread becomes 0). Posterior means rather than raw
  frequencies keep small-sample players from landing at 0% or 100%.
* **k-means++**: seeded k-means++ initialisation, Lloyd iterations, the best
  of ``n_init`` restarts by inertia. Deterministic for a given seed.
* **Gaussian mixture**: diagonal covariances fitted by EM from a k-means
  start, variance floor ``var_floor``; BIC = -2 log L + p ln n with
  p = k * 2d + (k - 1) chooses k.
* **Names**: heuristic labels (nit, TAG, LAG, calling station, maniac,
  loose-passive, tight-passive) from cluster-mean VPIP, PFR and fold to flop
  c-bet, with thresholds meant for 6-max cash games (:func:`name_cluster`).
  They are descriptive conventions, not ground truth.
* **Adjusted Rand index** (Hubert and Arabie 1985) compares two labelings:
  1 for identical partitions, about 0 for chance agreement.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .bayes import logit

FEATURE_STATS: tuple[str, ...] = (
    "vpip", "pfr", "rfi", "limp", "cold_call", "three_bet", "fold_to_three_bet",
    "cbet_flop", "fold_vs_cbet_flop", "check_raise_flop", "af_flop", "wtsd",
)
NAME_STATS: tuple[str, ...] = ("vpip", "pfr", "fold_vs_cbet_flop", "af_flop", "wtsd")

Vector = list[float]


# --------------------------------------------------------------------------- features


@dataclass(frozen=True)
class FeatureMatrix:
    """Per-player feature rows (standardised logits) and the underlying means."""

    players: tuple[str, ...]
    stats: tuple[str, ...]
    probabilities: tuple[tuple[float, ...], ...]
    z: tuple[tuple[float, ...], ...]
    centre: tuple[float, ...]
    scale: tuple[float, ...]


def standardise(rows: Sequence[Sequence[float]]) -> tuple[list[Vector], list[float], list[float]]:
    """Column z-scores (population standard deviation); constant columns become 0."""
    if not rows:
        return [], [], []
    dims = len(rows[0])
    n = len(rows)
    centre = [sum(r[j] for r in rows) / n for j in range(dims)]
    scale = [math.sqrt(sum((r[j] - centre[j]) ** 2 for r in rows) / n) for j in range(dims)]
    z = [[(r[j] - centre[j]) / scale[j] if scale[j] > 1e-12 else 0.0 for j in range(dims)] for r in rows]
    return z, centre, scale


def feature_matrix(model: Any, players: Iterable[str] | None = None, stats: Sequence[str] = FEATURE_STATS) -> FeatureMatrix:
    """Features from an :class:`~poker_ecosystem.opponent.OpponentModel`."""
    names = tuple(players) if players is not None else tuple(model.players())
    probs = [tuple(model.estimate(p, s).mean for s in stats) for p in names]
    logits = [[logit(v) for v in row] for row in probs]
    z, centre, scale = standardise(logits)
    return FeatureMatrix(names, tuple(stats), tuple(probs), tuple(tuple(r) for r in z), tuple(centre), tuple(scale))


# --------------------------------------------------------------------------- k-means


def _sqdist(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _nearest(point: Sequence[float], centres: Sequence[Sequence[float]]) -> int:
    best, best_d = 0, math.inf
    for j, c in enumerate(centres):
        d = _sqdist(point, c)
        if d < best_d:
            best, best_d = j, d
    return best


def _kmeanspp(points: Sequence[Sequence[float]], k: int, rng: random.Random) -> list[Vector]:
    n = len(points)
    centres = [list(points[rng.randrange(n)])]
    d2 = [_sqdist(p, centres[0]) for p in points]
    while len(centres) < k:
        total = sum(d2)
        idx = rng.randrange(n)
        if total > 0:
            target, acc = rng.random() * total, 0.0
            for i, w in enumerate(d2):
                acc += w
                if acc >= target:
                    idx = i
                    break
        centres.append(list(points[idx]))
        d2 = [min(old, _sqdist(p, centres[-1])) for old, p in zip(d2, points)]
    return centres


def _lloyd(points: Sequence[Sequence[float]], centres: list[Vector], max_iter: int) -> tuple[list[int], list[Vector], float, int]:
    labels: list[int] = []
    iterations = 0
    for iterations in range(1, max_iter + 1):
        new = [_nearest(p, centres) for p in points]
        if new == labels:
            break
        labels = new
        for j in range(len(centres)):
            members = [p for p, lab in zip(points, labels) if lab == j]
            if members:
                centres[j] = [sum(col) / len(members) for col in zip(*members)]
            else:
                far = max(range(len(points)), key=lambda i: _sqdist(points[i], centres[labels[i]]))
                centres[j] = list(points[far])
    inertia = sum(_sqdist(p, centres[lab]) for p, lab in zip(points, labels))
    return labels, centres, inertia, iterations


@dataclass(frozen=True)
class KMeansResult:
    k: int
    labels: tuple[int, ...]
    centroids: tuple[tuple[float, ...], ...]
    inertia: float
    iterations: int


def kmeans(points: Sequence[Sequence[float]], k: int, *, n_init: int = 10, max_iter: int = 300, seed: int = 0) -> KMeansResult:
    """k-means with k-means++ starts; returns the restart with the lowest inertia."""
    if not points:
        raise ValueError("no points to cluster")
    k = max(1, min(k, len(points)))
    rng = random.Random(seed)
    best: KMeansResult | None = None
    for _ in range(max(1, n_init)):
        labels, centres, inertia, iters = _lloyd(points, _kmeanspp(points, k, rng), max_iter)
        if best is None or inertia < best.inertia - 1e-12:
            best = KMeansResult(k, tuple(labels), tuple(tuple(c) for c in centres), inertia, iters)
    assert best is not None
    return best


# --------------------------------------------------------------------------- Gaussian mixture


@dataclass(frozen=True)
class GMMResult:
    k: int
    labels: tuple[int, ...]
    weights: tuple[float, ...]
    means: tuple[tuple[float, ...], ...]
    variances: tuple[tuple[float, ...], ...]
    loglik: float
    bic: float
    iterations: int
    converged: bool


def _log_gauss(x: Sequence[float], mean: Sequence[float], var: Sequence[float]) -> float:
    return sum(-0.5 * (math.log(2.0 * math.pi * v) + (xi - m) ** 2 / v) for xi, m, v in zip(x, mean, var))


def _e_step(points: Sequence[Sequence[float]], weights: Sequence[float], means: Sequence[Vector], variances: Sequence[Vector]) -> tuple[float, list[Vector]]:
    loglik = 0.0
    resp: list[Vector] = []
    log_w = [math.log(w) if w > 0 else -math.inf for w in weights]
    for p in points:
        row = [lw + _log_gauss(p, m, v) for lw, m, v in zip(log_w, means, variances)]
        top = max(row)
        norm = top + math.log(sum(math.exp(r - top) for r in row))
        loglik += norm
        resp.append([math.exp(r - norm) for r in row])
    return loglik, resp


def _m_step(points: Sequence[Sequence[float]], resp: Sequence[Vector], means: list[Vector], variances: list[Vector], var_floor: float) -> list[float]:
    n, k = len(points), len(means)
    dims = len(points[0])
    weights = []
    for j in range(k):
        r = [row[j] for row in resp]
        total = sum(r)
        weights.append(total / n)
        if total < 1e-10:
            continue
        means[j] = [sum(ri * p[d] for ri, p in zip(r, points)) / total for d in range(dims)]
        variances[j] = [
            max(sum(ri * (p[d] - means[j][d]) ** 2 for ri, p in zip(r, points)) / total, var_floor)
            for d in range(dims)
        ]
    return weights


def gmm_diag(
    points: Sequence[Sequence[float]],
    k: int,
    *,
    seed: int = 0,
    max_iter: int = 500,
    tol: float = 1e-8,
    var_floor: float = 1e-3,
) -> GMMResult:
    """Diagonal-covariance Gaussian mixture by EM, initialised from k-means++."""
    if not points:
        raise ValueError("no points to cluster")
    n, dims = len(points), len(points[0])
    k = max(1, min(k, n))
    start = kmeans(points, k, seed=seed)
    means = [list(c) for c in start.centroids]
    overall = [max(sum((p[d] - sum(q[d] for q in points) / n) ** 2 for p in points) / n, var_floor) for d in range(dims)]
    variances: list[Vector] = []
    for j in range(k):
        members = [p for p, lab in zip(points, start.labels) if lab == j]
        if len(members) >= 2:
            variances.append([max(sum((p[d] - means[j][d]) ** 2 for p in members) / len(members), var_floor) for d in range(dims)])
        else:
            variances.append(list(overall))
    weights = [max(start.labels.count(j), 1) / n for j in range(k)]
    total_w = sum(weights)
    weights = [w / total_w for w in weights]
    previous = -math.inf
    converged = False
    iterations = 0
    loglik, resp = _e_step(points, weights, means, variances)
    for iterations in range(1, max_iter + 1):
        if iterations > 1 and loglik - previous <= tol * (1.0 + abs(loglik)):
            converged = True
            break
        previous = loglik
        weights = _m_step(points, resp, means, variances, var_floor)
        loglik, resp = _e_step(points, weights, means, variances)
    labels = tuple(max(range(k), key=lambda j: row[j]) for row in resp)
    params = k * 2 * dims + (k - 1)
    bic = -2.0 * loglik + params * math.log(n)
    return GMMResult(
        k, labels, tuple(weights), tuple(tuple(m) for m in means), tuple(tuple(v) for v in variances),
        loglik, bic, iterations, converged,
    )


def select_gmm(points: Sequence[Sequence[float]], k_values: Iterable[int] = range(1, 9), *, seed: int = 0) -> tuple[GMMResult, dict[int, float]]:
    """Fit a mixture for each k and return the lowest-BIC fit plus the BIC table."""
    best: GMMResult | None = None
    table: dict[int, float] = {}
    for k in k_values:
        if k < 1 or k > len(points):
            continue
        fit = gmm_diag(points, k, seed=seed)
        table[k] = fit.bic
        if best is None or fit.bic < best.bic - 1e-9:
            best = fit
    if best is None:
        raise ValueError("no admissible k")
    return best, table


# --------------------------------------------------------------------------- evaluation and naming


def adjusted_rand_index(labels_a: Sequence[Any], labels_b: Sequence[Any]) -> float:
    """Adjusted Rand index of two labelings of the same items."""
    if len(labels_a) != len(labels_b):
        raise ValueError("labelings must have the same length")
    n = len(labels_a)
    if n < 2:
        return 1.0

    def pairs(x: float) -> float:
        return x * (x - 1.0) / 2.0

    index = sum(pairs(v) for v in Counter(zip(labels_a, labels_b)).values())
    sum_a = sum(pairs(v) for v in Counter(labels_a).values())
    sum_b = sum(pairs(v) for v in Counter(labels_b).values())
    expected = sum_a * sum_b / pairs(n)
    maximum = (sum_a + sum_b) / 2.0
    if maximum == expected:
        return 1.0
    return (index - expected) / (maximum - expected)


def name_cluster(freqs: Mapping[str, float]) -> str:
    """Heuristic style label from frequencies (6-max cash thresholds).

    maniac: VPIP >= 45% and PFR >= 30%; nit: VPIP < 17%; calling station:
    VPIP >= 35%, PFR/VPIP < 0.35 and fold to flop c-bet <= 40%;
    loose-passive: VPIP >= 28% and PFR/VPIP < 0.55; LAG: VPIP >= 28%;
    TAG: PFR/VPIP >= 0.55; otherwise tight-passive.
    """
    vpip = float(freqs.get("vpip", 0.0))
    pfr = float(freqs.get("pfr", 0.0))
    ratio = pfr / vpip if vpip > 0 else 0.0
    fold_cbet = freqs.get("fold_vs_cbet_flop")
    if vpip >= 0.45 and pfr >= 0.30:
        return "maniac"
    if vpip < 0.17:
        return "nit"
    if vpip >= 0.35 and ratio < 0.35 and (fold_cbet is None or fold_cbet <= 0.40):
        return "calling station"
    if vpip >= 0.28 and ratio < 0.55:
        return "loose-passive"
    if vpip >= 0.28:
        return "LAG"
    if ratio >= 0.55:
        return "TAG"
    return "tight-passive"


def _unique_names(names: Sequence[str]) -> list[str]:
    seen: Counter[str] = Counter()
    totals = Counter(names)
    out = []
    for name in names:
        seen[name] += 1
        out.append(name if totals[name] == 1 else f"{name} #{seen[name]}")
    return out


@dataclass
class Clustering:
    """Result of :func:`cluster_players`."""

    method: str
    k: int
    players: list[str]
    labels: list[int]
    names: list[str]
    summary: list[dict[str, Any]]
    features: FeatureMatrix
    bic: dict[int, float] = field(default_factory=dict)

    def label_of(self, player: str) -> int:
        return self.labels[self.players.index(player)]

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "k": self.k,
            "assignments": {p: self.names[lab] for p, lab in zip(self.players, self.labels)},
            "clusters": self.summary,
            "bic": {str(k): v for k, v in self.bic.items()},
            "features": list(self.features.stats),
        }


def population_summary(model: Any, players: Sequence[str], labels: Sequence[int], k: int, stats: Sequence[str]) -> list[dict[str, Any]]:
    """Per-cluster size, members and mean posterior frequency of each statistic."""
    wanted = list(dict.fromkeys([*stats, *NAME_STATS]))
    summary = []
    for j in range(k):
        members = [p for p, lab in zip(players, labels) if lab == j]
        means = {s: sum(model.estimate(p, s).mean for p in members) / len(members) if members else None for s in wanted}
        summary.append({"cluster": j, "size": len(members), "players": members, "mean": means})
    return summary


def cluster_players(
    model: Any,
    *,
    method: str = "gmm",
    k: int | None = None,
    k_values: Iterable[int] = range(1, 9),
    stats: Sequence[str] = FEATURE_STATS,
    players: Iterable[str] | None = None,
    seed: int = 0,
) -> Clustering:
    """Cluster players of an opponent model and name the clusters heuristically."""
    fm = feature_matrix(model, players, stats)
    points = [list(r) for r in fm.z]
    bic: dict[int, float] = {}
    if method == "kmeans":
        result = kmeans(points, k or 4, seed=seed)
        labels, k_used = list(result.labels), result.k
    elif method == "gmm":
        if k is None:
            fit, bic = select_gmm(points, k_values, seed=seed)
        else:
            fit = gmm_diag(points, k, seed=seed)
            bic = {fit.k: fit.bic}
        labels, k_used = list(fit.labels), fit.k
    else:
        raise ValueError(f"unknown clustering method {method!r}")
    summary = population_summary(model, fm.players, labels, k_used, stats)
    names = _unique_names([name_cluster({s: v for s, v in c["mean"].items() if v is not None}) for c in summary])
    for entry, name in zip(summary, names):
        entry["name"] = name
    return Clustering(method, k_used, list(fm.players), labels, names, summary, fm, bic)
