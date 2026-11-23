"""Evaluate the opponent model on a synthetic corpus with ground truth.

Ground truth
------------
For player i and statistic s the *true value* is the mean, over all of i's
opportunities for s in the whole corpus, of the opportunity's truth value
(:mod:`poker_ecosystem.stats`: the bot's ``truth`` probability of the success
action types at the decision, or the compensator sum for VPIP/PFR). Only
statistics whose success is an action choice can be evaluated. Pairs with
fewer than ``min_truth`` truth-carrying opportunities are skipped, because
their truth average is itself noisy.

Protocol
--------
For each size N, every player with at least N hands contributes a random
subsample of N of their hands: the first N of a seeded permutation, so the
samples are nested across sizes. The opponent model (population priors
included) is refitted on the subsampled data only and compared with the
full-corpus truth:

* error of the posterior mean: MAE, RMSE and bias, pooled over positions and
  at position level;
* calibration: the share of credible intervals (default 90%) that contain the
  truth;
* shrinkage benefit: MAE of the raw frequency k/n minus MAE of the posterior
  mean on the same pairs (pairs with n >= 1);
* cluster recovery: adjusted Rand index between the true ``style`` in
  ``players.json`` and k-means (k = number of true styles) or the BIC-chosen
  diagonal Gaussian mixture, both on posterior features.

Outputs: ``research/results/opponent_model.json``, SVG figures in
``research/figures/`` and ``research/reports/opponent-model.md``.

Command line::

    python -m poker_ecosystem.evaluate --hands FILE [FILE ...] --players players.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import __version__, hh, svg
from .cluster import FEATURE_STATS, adjusted_rand_index, cluster_players, feature_matrix, kmeans, select_gmm
from .opponent import APPROXIMATIONS, OpponentModel
from .stats import ALL, STAT_DEFS, StatsTable, hand_observations

DEFAULT_SIZES: tuple[int, ...] = (50, 100, 250, 500, 1000, 2500)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO_ROOT / "research" / "results" / "opponent_model.json"
DEFAULT_FIGURES = REPO_ROOT / "research" / "figures"
DEFAULT_REPORT = REPO_ROOT / "research" / "reports" / "opponent-model.md"

SYNTHETIC_NOTES: tuple[str, ...] = (
    "The bots follow fixed, stationary policies: they do not adapt, tilt or change stakes. "
    "Errors on real opponents will be larger and the population is not a fixed set of archetypes.",
    "Ground truth (the bots' own action probabilities) exists only in simulation. "
    "On real hand histories the model can only be checked indirectly, e.g. by predicting held-out hands.",
    "Styles are discrete archetypes with modest per-bot jitter, which makes clustering easier than in real "
    "player pools where styles form a continuum.",
    "Population priors are fitted on the same players that are evaluated (no held-out players).",
    "Each true value is an average of the bot's decision probabilities over a finite number of spots, "
    "so it carries a little noise of its own.",
)


@dataclass(frozen=True)
class EvalConfig:
    sizes: tuple[int, ...] = DEFAULT_SIZES
    seed: int = 1
    level: float = 0.9
    min_truth: int = 20
    stats: tuple[str, ...] | None = None
    bluff_threshold: float = 0.5
    k_values: tuple[int, ...] = tuple(range(1, 9))


@dataclass(frozen=True)
class Record:
    """One (player, statistic, position) comparison at one sample size."""

    player: str
    stat: str
    position: str
    n: int
    k: int
    raw: float | None
    mean: float
    lo: float
    hi: float
    truth: float
    truth_n: int


def load_players_meta(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read ``players.json``: a list of ``{name, style, profile}`` (a dict keyed by name works too)."""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(data, dict) and isinstance(data.get("players"), list):
        data = data["players"]
    if isinstance(data, dict):
        return {str(name): dict(entry, name=str(name)) for name, entry in data.items()}
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of players")
    return {str(entry["name"]): dict(entry) for entry in data}


def evaluable_stats(stats: Iterable[str] | None = None) -> list[str]:
    """Statistics whose success can be compared with per-decision truth."""
    if stats is None:
        return [k for k, d in STAT_DEFS.items() if d.truth_evaluable]
    chosen = list(stats)
    bad = [s for s in chosen if s not in STAT_DEFS or not STAT_DEFS[s].truth_evaluable]
    if bad:
        raise ValueError(f"not truth-evaluable: {bad}")
    return chosen


# --------------------------------------------------------------------------- metrics


def _plain(value: float) -> float | int:
    """Integral floats as ints (a median of 500.0 reads as 500)."""
    return int(value) if float(value).is_integer() else value


def _clean(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else value


def _json_safe(obj: Any) -> Any:
    """Replace non-finite floats by ``None`` so the JSON output stays standard."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _error_summary(errors: Sequence[float]) -> dict[str, float | None]:
    if not errors:
        return {"mae": None, "rmse": None, "bias": None, "mae_se": None}
    absolute = [abs(e) for e in errors]
    mae = sum(absolute) / len(absolute)
    se = statistics.stdev(absolute) / math.sqrt(len(absolute)) if len(absolute) > 1 else None
    return {
        "mae": mae,
        "rmse": math.sqrt(sum(e * e for e in errors) / len(errors)),
        "bias": sum(errors) / len(errors),
        "mae_se": se,
    }


def summarise(records: Sequence[Record]) -> dict[str, Any]:
    """Error, calibration and shrinkage metrics for a set of records."""
    if not records:
        return {"pairs": 0}
    with_data = [r for r in records if r.n > 0]
    covered = [r.lo <= r.truth <= r.hi for r in records]
    coverage = sum(covered) / len(covered)
    raw = _error_summary([r.raw - r.truth for r in with_data if r.raw is not None])
    same = _error_summary([r.mean - r.truth for r in with_data])
    gain = raw["mae"] - same["mae"] if raw["mae"] is not None and same["mae"] is not None else None
    return {
        "pairs": len(records),
        "pairs_with_data": len(with_data),
        "mean_opportunities": sum(r.n for r in records) / len(records),
        "posterior": _error_summary([r.mean - r.truth for r in records]),
        "coverage": coverage,
        "coverage_se": math.sqrt(coverage * (1.0 - coverage) / len(records)),
        "mean_interval_width": sum(r.hi - r.lo for r in records) / len(records),
        "raw": raw,
        "posterior_same_pairs": same,
        "shrinkage_gain_mae": gain,
    }


def _records(model: OpponentModel, players: set[str], stats: set[str], truth: Mapping[tuple[str, str, str], list[float]], min_truth: int) -> list[Record]:
    out = []
    for (player, stat, pos) in sorted(truth):
        total, count = truth[(player, stat, pos)]
        if count < min_truth or player not in players or stat not in stats:
            continue
        est = model.estimate(player, stat, None if pos == ALL else pos)
        out.append(Record(player, stat, pos, est.n, est.k, est.raw, est.mean, est.lo, est.hi, total / count, int(count)))
    return out


def _cluster_recovery(model: OpponentModel, players: Sequence[str], meta: Mapping[str, Mapping[str, Any]], config: EvalConfig) -> dict[str, Any]:
    labelled = [p for p in players if meta.get(p, {}).get("style")]
    if len(labelled) < 3:
        return {"players": len(labelled), "k_true": None, "kmeans_ari": None, "gmm_k": None, "gmm_ari": None}
    truth = [str(meta[p]["style"]) for p in labelled]
    k_true = len(set(truth))
    points = [list(r) for r in feature_matrix(model, labelled).z]
    km = kmeans(points, k_true, seed=config.seed)
    gmm, bic = select_gmm(points, [k for k in config.k_values if k <= len(points)], seed=config.seed)
    return {
        "players": len(labelled),
        "k_true": k_true,
        "kmeans_ari": adjusted_rand_index(truth, km.labels),
        "gmm_k": gmm.k,
        "gmm_ari": adjusted_rand_index(truth, gmm.labels),
        "bic": {str(k): v for k, v in bic.items()},
    }


# --------------------------------------------------------------------------- evaluation


HandSource = Callable[[], Iterable[hh.Hand]]


def _index_players(source: HandSource) -> tuple[dict[str, list[int]], int, int]:
    hands_of: dict[str, list[int]] = defaultdict(list)
    total = synthetic = 0
    for i, hand in enumerate(source()):
        for name in hh.seat_names(hand).values():
            hands_of[name].append(i)
        total += 1
        synthetic += int(bool(hand.get("synthetic")))
    return hands_of, total, synthetic


def evaluate_corpus(
    hands: HandSource | Sequence[hh.Hand],
    players_meta: Mapping[str, Mapping[str, Any]],
    config: EvalConfig | None = None,
) -> dict[str, Any]:
    """Run the evaluation protocol; returns a JSON-ready results dictionary.

    ``hands`` is a list or a zero-argument callable returning a fresh
    iterator (the corpus is read twice, so large JSONL files are streamed).
    """
    config = config or EvalConfig()
    source: HandSource = hands if callable(hands) else (lambda: hands)  # type: ignore[assignment]
    stats = evaluable_stats(config.stats)
    keep = set(stats) | set(FEATURE_STATS)
    hands_of, total, synthetic = _index_players(source)
    rng = random.Random(config.seed)
    rank: dict[str, dict[int, int]] = {}
    for name in sorted(hands_of):
        order = list(hands_of[name])
        rng.shuffle(order)
        rank[name] = {h: r for r, h in enumerate(order)}
    sizes = sorted({int(s) for s in config.sizes if int(s) > 0})
    tables = {n: StatsTable(config.bluff_threshold) for n in sizes}
    truth: dict[tuple[str, str, str], list[float]] = {}
    for i, hand in enumerate(source()):
        obs = hand_observations(hand, bluff_threshold=config.bluff_threshold)
        member = {p: [n for n in sizes if rank[p][i] < n and len(hands_of[p]) >= n] for p in obs.players}
        for p, ns in member.items():
            for n in ns:
                tables[n].hands[p] += 1
        for o in obs.observations:
            if o.stat not in keep:
                continue
            if o.truth is not None and o.stat in stats:
                for key in ((o.player, o.stat, o.position), (o.player, o.stat, ALL)):
                    acc = truth.setdefault(key, [0.0, 0])
                    acc[0] += o.truth
                    acc[1] += 1
            for n in member.get(o.player, ()):
                tables[n].add_observation(o)

    results_by_size: list[dict[str, Any]] = []
    last: tuple[int, OpponentModel, list[str], list[Record]] | None = None
    for n in sizes:
        table = tables[n]
        players = [p for p in table.players() if table.hands.get(p, 0) >= n]
        if len(players) < 2:
            results_by_size.append({"hands": n, "players": len(players), "skipped": "fewer than 2 players with this many hands"})
            continue
        model = OpponentModel(table, level=config.level)
        records = _records(model, set(players), set(stats), truth, config.min_truth)
        pooled = [r for r in records if r.position == ALL]
        positional = [r for r in records if r.position != ALL]
        results_by_size.append({
            "hands": n,
            "players": len(players),
            "pooled": summarise(pooled),
            "by_position": summarise(positional),
            "by_stat": {s: summarise([r for r in pooled if r.stat == s]) for s in stats if any(r.stat == s for r in pooled)},
            "by_position_label": {
                pos: summarise([r for r in positional if r.position == pos])
                for pos in sorted({r.position for r in positional}, key=lambda p: hh.POSITIONS.index(p) if p in hh.POSITIONS else 99)
            },
            "clusters": _cluster_recovery(model, players, players_meta, config),
        })
        last = (n, model, players, pooled)

    styles = Counter(str(players_meta[p].get("style")) for p in hands_of if p in players_meta)
    counts = sorted(len(v) for v in hands_of.values())
    results: dict[str, Any] = {
        "schema": "pe-opponent-eval/1",
        "generated_by": f"poker_ecosystem.evaluate {__version__}",
        "synthetic_corpus": total > 0 and synthetic == total,
        "config": {**asdict(config), "stats": stats},
        "corpus": {
            "hands": total,
            "synthetic_hands": synthetic,
            "players": len(hands_of),
            "players_with_style": sum(styles.values()),
            "styles": dict(sorted(styles.items())),
            "hands_per_player": {
                "min": counts[0] if counts else 0,
                "median": _plain(statistics.median(counts)) if counts else 0,
                "max": counts[-1] if counts else 0,
            },
        },
        "sizes": results_by_size,
        "approximations": list(APPROXIMATIONS),
        "synthetic_notes": list(SYNTHETIC_NOTES),
    }
    if last is not None:
        results["largest"] = _largest_detail(*last, players_meta, config)
    return results


def _largest_detail(n: int, model: OpponentModel, players: list[str], pooled: list[Record], meta: Mapping[str, Mapping[str, Any]], config: EvalConfig) -> dict[str, Any]:
    clustering = cluster_players(model, method="gmm", k_values=[k for k in config.k_values if k <= len(players)], players=players, seed=config.seed)
    composition = []
    for entry in clustering.summary:
        styles = Counter(str(meta.get(p, {}).get("style", "unknown")) for p in entry["players"])
        composition.append({
            "cluster": entry["cluster"],
            "name": entry["name"],
            "size": entry["size"],
            "styles": dict(sorted(styles.items())),
            "mean": {s: entry["mean"].get(s) for s in ("vpip", "pfr", "three_bet", "cbet_flop", "fold_vs_cbet_flop", "wtsd")},
        })
    scatter: dict[str, list[list[float]]] = defaultdict(list)
    for p in players:
        style = str(meta.get(p, {}).get("style", "unknown"))
        scatter[style].append([model.estimate(p, "vpip").mean, model.estimate(p, "pfr").mean])
    return {
        "hands": n,
        "errors": [r.mean - r.truth for r in pooled],
        "clusters": composition,
        "bic": {str(k): v for k, v in clustering.bic.items()},
        "scatter_vpip_pfr": dict(sorted(scatter.items())),
    }


# --------------------------------------------------------------------------- outputs


def _series_values(sizes: Sequence[dict[str, Any]], *path: str) -> list[float | None]:
    out: list[float | None] = []
    for entry in sizes:
        value: Any = entry
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        out.append(_clean(value) if isinstance(value, (int, float)) else None)
    return out


def make_figures(results: Mapping[str, Any], figures_dir: str | Path) -> dict[str, Path]:
    """Write the evaluation SVGs; returns {figure key: path}."""
    out_dir = Path(figures_dir)
    sizes = [s for s in results["sizes"] if "pooled" in s]
    xs = [float(s["hands"]) for s in sizes]
    level = float(results["config"]["level"])
    figs: dict[str, str] = {}
    figs["error"] = svg.line_chart(
        [
            svg.Series("posterior mean, pooled", xs, _series_values(sizes, "pooled", "posterior", "mae")),
            svg.Series("raw frequency, pooled", xs, _series_values(sizes, "pooled", "raw", "mae")),
            svg.Series("posterior mean, by position", xs, _series_values(sizes, "by_position", "posterior", "mae")),
            svg.Series("raw frequency, by position", xs, _series_values(sizes, "by_position", "raw", "mae")),
        ],
        title="Estimation error vs hands observed",
        x_label="hands observed per player (log scale)",
        y_label="mean absolute error",
        log_x=True,
        description="Mean absolute error of posterior means and raw frequencies against the bots' true frequencies.",
    )
    figs["coverage"] = svg.line_chart(
        [
            svg.Series("pooled", xs, _series_values(sizes, "pooled", "coverage")),
            svg.Series("by position", xs, _series_values(sizes, "by_position", "coverage")),
        ],
        title=f"Calibration of {level:.0%} credible intervals",
        x_label="hands observed per player (log scale)",
        y_label="share of intervals containing the truth",
        log_x=True,
        y_min=0.0,
        y_max=1.0,
        reference=level,
        reference_label=f"nominal {level:.0%}",
    )
    figs["ari"] = svg.line_chart(
        [
            svg.Series("k-means, k = true styles", xs, _series_values(sizes, "clusters", "kmeans_ari")),
            svg.Series("Gaussian mixture, k by BIC", xs, _series_values(sizes, "clusters", "gmm_ari")),
        ],
        title="Style recovery (adjusted Rand index)",
        x_label="hands observed per player (log scale)",
        y_label="adjusted Rand index",
        log_x=True,
        y_max=1.0,
    )
    if sizes:
        for street, chart in _by_stat_charts(sizes[-1]).items():
            figs[f"by_stat_{street}"] = chart
    largest = results.get("largest")
    if largest:
        groups = list(largest["scatter_vpip_pfr"].items())
        if len(groups) > len(svg.PALETTE_LIGHT):
            head = groups[: len(svg.PALETTE_LIGHT) - 1]
            other = [pt for _, pts in groups[len(svg.PALETTE_LIGHT) - 1 :] for pt in pts]
            groups = [*head, ("other", other)]
        figs["styles"] = svg.scatter_chart(
            [(name, [tuple(p) for p in pts]) for name, pts in groups],
            title=f"Posterior VPIP vs PFR by true style ({largest['hands']} hands)",
            x_label="VPIP (posterior mean)",
            y_label="PFR (posterior mean)",
        )
        figs["error_hist"] = svg.histogram(
            largest["errors"],
            bins=30,
            title=f"Posterior error distribution ({largest['hands']} hands, pooled)",
            x_label="posterior mean minus truth",
        )
    return {key: svg.save(content, out_dir / f"opponent_model_{key}.svg") for key, content in figs.items()}


def _by_stat_charts(entry: Mapping[str, Any]) -> dict[str, str]:
    """One grouped-bar chart per street (small multiples keep the labels legible)."""
    by_stat = entry["by_stat"]
    streets: dict[str, list[str]] = defaultdict(list)
    for stat, summary in by_stat.items():
        if summary.get("pairs"):
            streets[STAT_DEFS[stat].street].append(stat)

    def band(mae: float | None, se: float | None) -> tuple[float, float] | None:
        if mae is None or se is None:
            return None
        return max(mae - 1.96 * se, 0.0), mae + 1.96 * se

    charts = {}
    for street in ("preflop", "flop", "turn", "river"):
        stats = streets.get(street)
        if not stats:
            continue
        labels = [s[: -len(street) - 1] if s.endswith(f"_{street}") else s for s in stats]
        post = [_clean(by_stat[s]["posterior"]["mae"]) for s in stats]
        raw = [_clean((by_stat[s].get("raw") or {}).get("mae")) for s in stats]
        charts[street] = svg.grouped_bar_chart(
            labels,
            [("posterior mean", post), ("raw frequency", raw)],
            errors=[
                [band(m, by_stat[s]["posterior"]["mae_se"]) for m, s in zip(post, stats)],
                [band(m, (by_stat[s].get("raw") or {}).get("mae_se")) for m, s in zip(raw, stats)],
            ],
            title=f"Mean absolute error, {street} statistics ({entry['hands']} hands, pooled)",
            y_label="mean absolute error",
            height=400,
            description="Error bars are 95% normal intervals of the mean absolute error.",
        )
    return charts


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _rel(path: Path, base: Path) -> str:
    try:
        return os.path.relpath(path, base).replace(os.sep, "/")
    except ValueError:
        return path.resolve().as_uri()


def render_report(results: Mapping[str, Any], figures: Mapping[str, Path], report_path: str | Path) -> str:
    """Markdown report of an evaluation."""
    base = Path(report_path).resolve().parent
    corpus = results["corpus"]
    level = results["config"]["level"]
    lines = ["# Opponent model evaluation", ""]
    if results.get("synthetic_corpus"):
        lines += [
            "> **This evaluation uses a synthetic corpus.** Every hand was generated by simulated bots with",
            "> known styles; the true frequencies come from the bots' own action probabilities. What that implies:",
            ">",
            *[f"> * {note}" for note in results["synthetic_notes"]],
            "",
        ]
    else:
        lines += [
            "> **Warning:** not every hand in this corpus is marked `synthetic`, yet the evaluation relies on",
            "> per-action `truth` probabilities that only simulated hands carry. Treat the numbers with suspicion.",
            "",
        ]
    lines += [
        "## Corpus",
        "",
        f"* hands: {corpus['hands']} ({corpus['synthetic_hands']} synthetic)",
        f"* players: {corpus['players']} ({corpus['players_with_style']} with a style in players.json)",
        f"* styles: {', '.join(f'{k} ({v})' for k, v in corpus['styles'].items()) or 'none'}",
        f"* hands per player: min {corpus['hands_per_player']['min']}, median {corpus['hands_per_player']['median']}, "
        f"max {corpus['hands_per_player']['max']}",
        f"* statistics evaluated: {len(results['config']['stats'])} (truth-evaluable ones; at least "
        f"{results['config']['min_truth']} truth-carrying opportunities per player and statistic)",
        "",
        "## Method",
        "",
        "Hierarchical empirical-Bayes Beta-binomial model (`poker_ecosystem.opponent`): population priors per",
        "statistic and per position fitted by maximum marginal likelihood with a weak boundary-avoiding prior,",
        "player-position priors shifted by the player's leave-one-position-out logit deviation (whose",
        "uncertainty widens the prior), conjugate posteriors with equal-tailed",
        f"{level:.0%} credible intervals. For each sample size N, every player with at least N hands is",
        "represented by a seeded random subsample of N hands (nested across sizes); the model is refitted on",
        "those subsamples and compared with each bot's true frequency over the full corpus.",
        "",
        "Approximations in the model:",
        "",
        *[f"* {a}" for a in results["approximations"]],
        "",
        "## Results",
        "",
        "| hands | players | pairs | MAE posterior | MAE raw (same pairs) | shrinkage gain | RMSE posterior | "
        f"coverage ({level:.0%}) | interval width | ARI k-means | ARI GMM (k) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for entry in results["sizes"]:
        if "pooled" not in entry:
            lines.append(f"| {entry['hands']} | {entry['players']} | skipped: {entry.get('skipped', '')} | | | | | | | | |")
            continue
        p, c = entry["pooled"], entry["clusters"]
        if not p.get("pairs"):
            lines.append(f"| {entry['hands']} | {entry['players']} | 0 | | | | | | | | |")
            continue
        lines.append(
            f"| {entry['hands']} | {entry['players']} | {p['pairs']} | {_fmt(p['posterior']['mae'])} | "
            f"{_fmt(p['raw']['mae'])} ({_fmt(p['posterior_same_pairs']['mae'])}) | {_fmt(p['shrinkage_gain_mae'])} | "
            f"{_fmt(p['posterior']['rmse'])} | {_fmt(p['coverage'])} | {_fmt(p['mean_interval_width'])} | "
            f"{_fmt(c.get('kmeans_ari'))} | {_fmt(c.get('gmm_ari'))} ({_fmt(c.get('gmm_k'))}) |"
        )
    lines += [
        "",
        "Pooled = one estimate per player and statistic over all positions. \"MAE raw (same pairs)\" compares the raw",
        "frequency k/n with the posterior mean on the pairs with at least one observation; the shrinkage gain is",
        "their difference (positive = the model beats raw frequencies).",
        "",
    ]
    lines += _position_table(results)
    lines += ["## Figures", ""]
    captions = {
        "error": "Mean absolute error against hands observed.",
        "coverage": "Share of credible intervals containing the true frequency.",
        "ari": "Recovery of the true styles by clustering.",
        "styles": "Posterior VPIP and PFR by true style (colour and marker shape).",
        "error_hist": "Distribution of posterior errors at the largest sample size.",
    }
    for key, path in figures.items():
        caption = captions.get(key)
        if caption is None and key.startswith("by_stat_"):
            caption = f"Error per {key[len('by_stat_'):]} statistic at the largest sample size (95% error bars)."
        caption = caption or key
        lines += [f"![{caption}]({_rel(Path(path), base)})", "", f"*{caption}*", ""]
    lines += _stat_table(results)
    lines += _cluster_table(results)
    lines += [
        "## Limitations",
        "",
        "* Only statistics whose success is an action choice are evaluated; WTSD, W$SD, aggression frequency,",
        "  sizing and the river bluff share have no per-decision truth.",
        "* Interval coverage below the nominal level points at over-shrinkage: a single Beta prior cannot follow",
        "  a pool of distinct styles (multimodal), and the Beta-binomial ignores the spot mix within a statistic.",
        "  A synthetic pool of a few sharply separated archetypes is close to the worst case for this.",
        "* Cluster names are heuristic labels from VPIP/PFR thresholds for 6-max cash games.",
        "",
    ]
    return "\n".join(lines)


def _position_table(results: Mapping[str, Any]) -> list[str]:
    rows = [e for e in results["sizes"] if "by_position" in e and e["by_position"].get("pairs")]
    if not rows:
        return []
    lines = [
        "Position-level estimates (one per player, statistic and position):",
        "",
        "| hands | pairs | MAE posterior | MAE raw (same pairs) | shrinkage gain | coverage |",
        "|---|---|---|---|---|---|",
    ]
    for e in rows:
        b = e["by_position"]
        lines.append(
            f"| {e['hands']} | {b['pairs']} | {_fmt(b['posterior']['mae'])} | {_fmt(b['raw']['mae'])} "
            f"({_fmt(b['posterior_same_pairs']['mae'])}) | {_fmt(b['shrinkage_gain_mae'])} | {_fmt(b['coverage'])} |"
        )
    return lines + [""]


def _stat_table(results: Mapping[str, Any]) -> list[str]:
    rows = [e for e in results["sizes"] if "by_stat" in e]
    if not rows:
        return []
    last = rows[-1]
    lines = [
        f"## Per statistic ({last['hands']} hands, pooled)",
        "",
        "| statistic | pairs | mean n | MAE posterior | MAE raw | coverage |",
        "|---|---|---|---|---|---|",
    ]
    for stat, s in last["by_stat"].items():
        lines.append(
            f"| `{stat}` | {s['pairs']} | {_fmt(s['mean_opportunities'], 1)} | {_fmt(s['posterior']['mae'])} | "
            f"{_fmt((s.get('raw') or {}).get('mae'))} | {_fmt(s['coverage'])} |"
        )
    return lines + [""]


def _cluster_table(results: Mapping[str, Any]) -> list[str]:
    largest = results.get("largest")
    if not largest:
        return []
    lines = [
        f"## Clusters at {largest['hands']} hands (Gaussian mixture, k by BIC)",
        "",
        "| cluster | heuristic name | size | true styles | VPIP | PFR | 3-bet | flop c-bet |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in largest["clusters"]:
        styles = ", ".join(f"{k} {v}" for k, v in c["styles"].items())
        m = c["mean"]
        lines.append(
            f"| {c['cluster']} | {c['name']} | {c['size']} | {styles} | {_fmt(m.get('vpip'))} | {_fmt(m.get('pfr'))} | "
            f"{_fmt(m.get('three_bet'))} | {_fmt(m.get('cbet_flop'))} |"
        )
    return lines + [""]


def write_outputs(
    results: Mapping[str, Any],
    *,
    results_path: str | Path = DEFAULT_RESULTS,
    figures_dir: str | Path = DEFAULT_FIGURES,
    report_path: str | Path = DEFAULT_REPORT,
) -> dict[str, Path]:
    """Write the JSON results, the SVG figures and the markdown report."""
    results_path, report_path = Path(results_path), Path(report_path)
    figures = make_figures(results, figures_dir)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(_json_safe(results), indent=2, allow_nan=False), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(results, figures, report_path), encoding="utf-8")
    return {"results": results_path, "report": report_path, **{f"figure:{k}": v for k, v in figures.items()}}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m poker_ecosystem.evaluate", description="Evaluate the opponent model on a synthetic corpus.")
    parser.add_argument("--hands", nargs="+", required=True, help=".json / .jsonl hand files")
    parser.add_argument("--players", required=True, help="players.json with name and style per bot")
    parser.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--level", type=float, default=0.9)
    parser.add_argument("--min-truth", type=int, default=20)
    parser.add_argument("--stats", help="comma-separated statistic keys (default: every truth-evaluable one)")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--figures", default=str(DEFAULT_FIGURES))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args(argv)
    try:
        config = EvalConfig(
            sizes=tuple(int(s) for s in args.sizes.split(",") if s.strip()),
            seed=args.seed,
            level=args.level,
            min_truth=args.min_truth,
            stats=tuple(s.strip() for s in args.stats.split(",") if s.strip()) if args.stats else None,
        )
        meta = load_players_meta(args.players)
        paths = [Path(p) for p in args.hands]
        results = evaluate_corpus(lambda: hh.iter_many(paths), meta, config)
        written = write_outputs(results, results_path=args.results, figures_dir=args.figures, report_path=args.report)
    except (OSError, ValueError, hh.HandError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for entry in results["sizes"]:
        p = entry.get("pooled") or {}
        post = (p.get("posterior") or {}).get("mae")
        print(f"N={entry['hands']:>5}  players={entry['players']:>4}  MAE={_fmt(post)}  coverage={_fmt(p.get('coverage'))}")
    for key, path in written.items():
        print(f"wrote {key}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
