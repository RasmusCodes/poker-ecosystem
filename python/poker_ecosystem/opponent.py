"""Hierarchical (empirical-Bayes) opponent model.

Model
-----
For a statistic ``s`` (see :mod:`poker_ecosystem.stats`) and a player ``i``
with ``n`` opportunities and ``k`` successes:

1. **Population prior.** theta_i ~ Beta(alpha_s, beta_s), fitted by maximum
   marginal likelihood across the players' pooled counts
   (:func:`bayes.fit_beta_prior`). The pooled posterior is
   Beta(alpha_s + k_i, beta_s + n_i - k_i).
2. **Population-by-position priors.** The same fit on the players' counts at
   each position p, with mean mu_sp.
3. **Player-position prior.** Centred at m = expit(logit(mu_sp) + delta_ip)
   where delta_ip is the player's overall deviation on the logit scale,
   estimated *without* the data at p (leave-one-position-out, so the data at
   p is not used twice): with N, K the player's opportunities and successes
   at the other positions and E = sum_q n_iq * mu_sq the successes expected
   from the population at the player's position mix,
   r = (K + kappa_s * E / N) / (N + kappa_s) (shrunk towards E / N with the
   population concentration kappa_s) and delta = logit(r) - logit(E / N).
   The uncertainty of that estimate is carried forward:
   v_ip = 1 / ((N + kappa_s + 1) r (1 - r)) is the logit-scale variance of
   the shrunk rate (Beta posterior variance, delta method). With N = 0,
   delta = 0 and v_ip uses r = mu_sp. The prior's logit variance is
   v_ip + s2_sp, where s2_sp, the position-specific spread the shift does not
   explain, is fitted by maximum marginal likelihood across players
   (:func:`bayes.fit_logit_variance`); with fewer than ``min_players``
   players it falls back to the logit variance of the population-position
   prior. The variance is turned into a Beta concentration with
   :func:`bayes.concentration_from_logit_variance`.
4. **Posterior.** Beta(m * kappa + k_ip, (1 - m) * kappa + n_ip - k_ip); we
   report the posterior mean and the equal-tailed credible interval
   (``level``, default 90%).

Approximations
--------------
See :data:`APPROXIMATIONS`; every report written by this package repeats
them. In short: hyperparameters are point estimates (empirical Bayes), the
player deviation's uncertainty enters only to first order (delta method),
opportunities are treated as exchangeable Bernoulli trials, ranges assume a
top-x% structure and the bluff share is learned from shown hands only.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from . import ranges
from .bayes import (
    BetaPrior,
    DirichletPrior,
    VarianceFit,
    beta_interval,
    concentration_from_logit_variance,
    expit,
    fit_beta_prior,
    fit_dirichlet_prior,
    fit_logit_variance,
    logit,
    order_statistic_interval,
)
from .hh import Hand, POSTFLOP_STREETS
from .stats import ALL, DEFAULT_BLUFF_THRESHOLD, SIZE_BINS, StatsTable, empirical_quantile

APPROXIMATIONS: tuple[str, ...] = (
    "Empirical Bayes: population priors are point estimates (maximum marginal likelihood plus a weak "
    "boundary-avoiding Gamma(2) prior on the between-player SD); their uncertainty is not propagated, "
    "so intervals are slightly too narrow when few players inform a prior.",
    "Each population prior is a single Beta distribution. A pool made of distinct styles is multimodal, "
    "so players far from the pool mean are over-shrunk until their own sample dominates.",
    "The player-position prior is centred on a plug-in estimate of the player's overall logit deviation "
    "(leave-one-position-out, shrunk); its uncertainty enters the prior variance only to first order (delta method).",
    "Opportunities are treated as exchangeable Bernoulli trials given the player's frequency; "
    "real spots differ (stacks, opponents, multiway) and af_* counts several correlated decisions per hand.",
    "Preflop ranges assume the player plays the top x% of hands by a fixed ranking "
    "(Chen formula unless research/vectors/preflop_ranking.json exists); mixed and polarised ranges are not modelled.",
    "River bluff share and the value threshold are learned only from bets that reached showdown "
    "with derived.river_strength; they describe called bets, not every bet.",
)

#: River statistics a solver needs to lock an opponent's river strategy.
EXPLOIT_STATS: dict[str, str] = {
    "bet_freq": "bet_unopened_river",
    "bluff_share": "river_bluff",
    "fold_to_bet": "fold_vs_bet_river",
    "raise_freq": "raise_vs_bet_river",
    "check_raise_freq": "check_raise_river",
}

#: Named multi-street tendencies and the statistic that measures each.
SEQUENCE_STATS: dict[str, str] = {
    "barrel_turn_after_flop_cbet": "cbet_turn",
    "bet_river_after_turn_barrel": "cbet_river",
    "bet_turn_after_flop_check_raise": "xr_flop_bet_turn",
}


@dataclass(frozen=True)
class Estimate:
    """Posterior summary of one statistic for one player at one position."""

    stat: str
    position: str
    mean: float
    lo: float
    hi: float
    level: float
    n: int
    k: int
    alpha: float
    beta: float
    prior_mean: float
    prior_concentration: float

    @property
    def raw(self) -> float | None:
        """Observed frequency k / n (``None`` without opportunities)."""
        return self.k / self.n if self.n else None

    def contains(self, value: float) -> bool:
        return self.lo <= value <= self.hi

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["raw"] = self.raw
        return out


@dataclass(frozen=True)
class SizingEstimate:
    """Dirichlet posterior over bet-size bins for one street and kind."""

    street: str
    kind: str
    bins: tuple[str, ...]
    counts: tuple[int, ...]
    mean: tuple[float, ...]
    intervals: tuple[tuple[float, float], ...]
    prior_mean: tuple[float, ...]
    prior_concentration: float
    level: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "street": self.street,
            "kind": self.kind,
            "level": self.level,
            "prior_concentration": self.prior_concentration,
            "bins": {
                b: {"count": c, "mean": m, "lo": iv[0], "hi": iv[1], "prior_mean": pm}
                for b, c, m, iv, pm in zip(self.bins, self.counts, self.mean, self.intervals, self.prior_mean)
            },
        }


@dataclass(frozen=True)
class RangeEstimate:
    """Top-x% preflop range implied by a posterior frequency (an approximation)."""

    stat: str
    position: str
    fraction: float
    interval: tuple[float, float]
    n: int
    classes: tuple[str, ...]
    classes_at_lo: int
    classes_at_hi: int
    combos: int
    source: str

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["classes"] = list(self.classes)
        out["interval"] = list(self.interval)
        return out


@dataclass(frozen=True)
class QuantileEstimate:
    """A sample quantile with a distribution-free order-statistic interval."""

    quantile: float
    value: float | None
    lo: float
    hi: float
    level: float
    n: int
    population_value: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class OpponentModel:
    """Posterior estimates for every player, statistic and position of a table."""

    def __init__(
        self,
        table: StatsTable,
        *,
        level: float = 0.9,
        min_players: int = 3,
        min_concentration: float = 0.05,
        max_concentration: float = 1e4,
        fallback_concentration: float = 2.0,
        value_quantile: float = 0.1,
        ranking: Iterable[str] | None = None,
        ranking_source: str | None = None,
    ) -> None:
        self.table = table
        self.level = level
        self.value_quantile = value_quantile
        self._fit = {
            "min_players": min_players,
            "min_concentration": min_concentration,
            "max_concentration": max_concentration,
        }
        self._fallback = fallback_concentration
        if ranking is None:
            loaded, source = ranges.load_ranking()
            self.ranking, self.ranking_source = loaded, source
        else:
            self.ranking = ranges.validate_ranking(list(ranking))
            self.ranking_source = ranking_source or "custom"
        self._priors: dict[tuple[str, str], BetaPrior] = {}
        self._shifts: dict[tuple[str, str, str], tuple[float, float]] = {}
        self._resid: dict[tuple[str, str], VarianceFit] = {}
        self._sizing: dict[tuple[str, str], DirichletPrior] = {}

    @classmethod
    def from_hands(cls, hands: Iterable[Hand], *, bluff_threshold: float = DEFAULT_BLUFF_THRESHOLD, **options: Any) -> "OpponentModel":
        return cls(StatsTable.from_hands(hands, bluff_threshold=bluff_threshold), **options)

    # -- population level

    def players(self) -> list[str]:
        return self.table.players()

    def stats(self) -> list[str]:
        return self.table.stats()

    def positions(self, stat: str) -> list[str]:
        return self.table.positions(stat)

    def population_prior(self, stat: str, position: str | None = None) -> BetaPrior:
        """Population (or population-by-position) Beta prior of a statistic."""
        key = (stat, position or ALL)
        prior = self._priors.get(key)
        if prior is None:
            data = [(n, k) for _, n, k in self.table.population(stat, position)]
            prior = fit_beta_prior(data, fallback_concentration=self._fallback, **self._fit)
            self._priors[key] = prior
        return prior

    # -- player-position prior

    def shift(self, player: str, stat: str, position: str) -> tuple[float, float]:
        """(prior mean at ``position``, logit variance of the estimated deviation)."""
        key = (player, stat, position)
        cached = self._shifts.get(key)
        if cached is not None:
            return cached
        n_other = k_other = 0
        expected = 0.0
        for q in self.table.positions(stat):
            if q == position:
                continue
            n, k = self.table.get(player, stat, q)
            if n:
                n_other += n
                k_other += k
                expected += n * self.population_prior(stat, q).mean
        mu = self.population_prior(stat, position).mean
        kappa = self.population_prior(stat).concentration
        delta, rate = 0.0, mu
        if n_other:
            base = expected / n_other
            rate = (k_other + kappa * base) / (n_other + kappa)
            delta = logit(rate) - logit(base)
        rate = min(max(rate, 1e-9), 1.0 - 1e-9)
        value = (expit(logit(mu) + delta), 1.0 / ((n_other + kappa + 1.0) * rate * (1.0 - rate)))
        self._shifts[key] = value
        return value

    def prior_mean_at(self, player: str, stat: str, position: str) -> float:
        """Mean of the player's prior at ``position`` (population mean shifted by delta)."""
        return self.shift(player, stat, position)[0]

    def residual_variance(self, stat: str, position: str) -> VarianceFit:
        """Logit-scale spread at ``position`` that the player's shift does not explain."""
        key = (stat, position)
        fit = self._resid.get(key)
        if fit is None:
            data = [
                (n, k, *self.shift(player, stat, position))
                for player, n, k in self.table.population(stat, position)
                if n
            ]
            fit = fit_logit_variance(
                data,
                min_players=self._fit["min_players"],
                min_concentration=self._fit["min_concentration"],
                max_concentration=self._fit["max_concentration"],
            )
            if fit.variance is None:
                prior = self.population_prior(stat, position)
                fallback = 1.0 / ((prior.concentration + 1.0) * prior.mean * (1.0 - prior.mean))
                fit = VarianceFit(fallback, f"fallback:{fit.method}", fit.players)
            self._resid[key] = fit
        return fit

    def prior_for(self, player: str, stat: str, position: str | None = None) -> tuple[float, float]:
        """(alpha, beta) of the prior used for a player's estimate."""
        if position in (None, ALL):
            prior = self.population_prior(stat)
            return prior.alpha, prior.beta
        mean, shift_var = self.shift(player, stat, position)
        kappa = concentration_from_logit_variance(
            mean,
            shift_var + (self.residual_variance(stat, position).variance or 0.0),
            min_concentration=self._fit["min_concentration"],
            max_concentration=self._fit["max_concentration"],
        )
        return mean * kappa, (1.0 - mean) * kappa

    # -- estimates

    def estimate(self, player: str, stat: str, position: str | None = None) -> Estimate:
        """Posterior mean and credible interval (pooled when ``position`` is None)."""
        pos = position or ALL
        a0, b0 = self.prior_for(player, stat, None if pos == ALL else pos)
        n, k = self.table.get(player, stat, None if pos == ALL else pos)
        a, b = a0 + k, b0 + n - k
        lo, hi = beta_interval(a, b, self.level)
        return Estimate(stat, pos, a / (a + b), lo, hi, self.level, n, k, a, b, a0 / (a0 + b0), a0 + b0)

    def estimates(self, player: str, stats: Iterable[str] | None = None, *, positions: bool = True) -> dict[str, dict[str, Estimate]]:
        """Estimates for many statistics, pooled and (optionally) per position."""
        out: dict[str, dict[str, Estimate]] = {}
        for stat in stats or self.stats():
            entry = {ALL: self.estimate(player, stat)}
            if positions:
                for pos in self.positions(stat):
                    entry[pos] = self.estimate(player, stat, pos)
            out[stat] = entry
        return out

    # -- sizing

    def sizing_prior(self, street: str, kind: str = "bet") -> DirichletPrior:
        key = (street, kind)
        prior = self._sizing.get(key)
        if prior is None:
            vectors = [self._bin_counts(p, street, kind) for p in self.players()]
            prior = fit_dirichlet_prior(vectors, size=len(SIZE_BINS))
            self._sizing[key] = prior
        return prior

    def _bin_counts(self, player: str, street: str, kind: str) -> list[int]:
        counts = self.table.sizing_counts(player, street, kind)
        return [counts[b] for b in SIZE_BINS]

    def sizing(self, player: str, street: str, kind: str = "bet") -> SizingEstimate:
        """Dirichlet posterior of the player's bet (or raise) sizes on a street."""
        prior = self.sizing_prior(street, kind)
        counts = self._bin_counts(player, street, kind)
        post = prior.posterior(counts)
        return SizingEstimate(
            street=street,
            kind=kind,
            bins=SIZE_BINS,
            counts=tuple(counts),
            mean=tuple(post.mean()),
            intervals=tuple(post.interval(j, self.level) for j in range(len(SIZE_BINS))),
            prior_mean=tuple(prior.mean),
            prior_concentration=prior.concentration,
            level=self.level,
        )

    # -- ranges

    def preflop_range(self, player: str, position: str, stat: str = "rfi") -> RangeEstimate:
        """Top-x% range for the posterior mean of ``stat`` (``rfi`` or ``vpip``) at a position."""
        est = self.estimate(player, stat, position)
        chosen = ranges.top_fraction(self.ranking, est.mean)
        return RangeEstimate(
            stat=stat,
            position=position,
            fraction=est.mean,
            interval=(est.lo, est.hi),
            n=est.n,
            classes=tuple(chosen),
            classes_at_lo=len(ranges.top_fraction(self.ranking, est.lo)),
            classes_at_hi=len(ranges.top_fraction(self.ranking, est.hi)),
            combos=ranges.combos_of(chosen),
            source=self.ranking_source,
        )

    def preflop_ranges(self, player: str, stat: str = "rfi") -> dict[str, RangeEstimate]:
        return {pos: self.preflop_range(player, pos, stat) for pos in self.positions(stat)}

    # -- river showdown information

    def bluff_share(self, player: str) -> Estimate:
        """Bluff share among the player's shown river bets (Beta posterior)."""
        return self.estimate(player, "river_bluff")

    def value_threshold(self, player: str, quantile: float | None = None) -> QuantileEstimate:
        """Lower quantile of shown value-bet strength with an order-statistic interval.

        Value bets are shown river bets with strength at or above the bluff
        threshold. The interval is distribution free; where the sample cannot
        bound it, the support limits (bluff threshold, 1) are reported.
        """
        q = quantile if quantile is not None else self.value_quantile
        floor = self.table.bluff_threshold
        values = sorted(v for v in self.table.river_strengths(player) if v >= floor)
        pooled = [v for p in self.players() for v in self.table.river_strengths(p) if v >= floor]
        lo, hi = floor, 1.0
        if values:
            low_i, high_i = order_statistic_interval(len(values), q, self.level)
            lo = values[low_i - 1] if low_i >= 1 else floor
            hi = values[high_i - 1] if high_i <= len(values) else 1.0
        return QuantileEstimate(q, empirical_quantile(values, q), lo, hi, self.level, len(values), empirical_quantile(pooled, q))

    # -- multi-street tendencies

    def tendencies(self, player: str, top: int = 10) -> dict[str, Any]:
        """Named sequence frequencies (posteriors) and raw line transition counts."""
        out: dict[str, Any] = {name: self.estimate(player, stat).as_dict() for name, stat in SEQUENCE_STATS.items()}
        out["check_raise"] = {s: self.estimate(player, f"check_raise_{s}").as_dict() for s in POSTFLOP_STREETS}
        out["line_transitions"] = _transition_rows(self.table.transitions(player, "line"), top)
        out["cbet_transitions"] = _transition_rows(self.table.transitions(player, "cbet"), top)
        return out

    def exploit_profile(self, player: str, position: str | None = None) -> dict[str, Any]:
        """River frequencies a solver needs to lock this opponent.

        ``bet_freq`` = bet when the river is unopened, ``bluff_share`` = share
        of shown river bets below the bluff threshold, ``fold_to_bet`` and
        ``raise_freq`` = response to a single river bet, ``check_raise_freq``
        = raise after checking the river. Each carries the posterior mean,
        credible interval and sample size.
        """
        out: dict[str, Any] = {"player": player, "street": "river", "level": self.level, "position": position or ALL}
        for name, stat in EXPLOIT_STATS.items():
            est = self.estimate(player, stat, position)
            out[name] = {
                "mean": est.mean,
                "lo": est.lo,
                "hi": est.hi,
                "n": est.n,
                "k": est.k,
                "stat": stat,
                "prior_mean": est.prior_mean,
            }
        out["value_threshold"] = self.value_threshold(player).as_dict()
        out["approximations"] = list(APPROXIMATIONS)
        return out

    def profile(self, player: str) -> dict[str, Any]:
        """Everything the model knows about one player, JSON-ready."""
        estimates = self.estimates(player)
        return {
            "player": player,
            "hands": self.table.hands.get(player, 0),
            "level": self.level,
            "estimates": {s: {p: e.as_dict() for p, e in by_pos.items()} for s, by_pos in estimates.items()},
            "sizing": {
                f"{street}:{kind}": self.sizing(player, street, kind).as_dict()
                for street in ("preflop", *POSTFLOP_STREETS)
                for kind in ("bet", "raise")
                if street != "preflop" or kind == "raise"
            },
            "ranges": {
                stat: {pos: r.as_dict() for pos, r in self.preflop_ranges(player, stat).items()}
                for stat in ("rfi", "vpip")
            },
            "river": {
                "bluff_share": self.bluff_share(player).as_dict(),
                "value_threshold": self.value_threshold(player).as_dict(),
            },
            "tendencies": self.tendencies(player),
            "exploit_profile": self.exploit_profile(player),
            "approximations": list(APPROXIMATIONS),
        }


def _transition_rows(counts: Mapping[tuple[str, str], int], top: int) -> list[dict[str, Any]]:
    from_totals: dict[str, int] = defaultdict(int)
    for (a, _), c in counts.items():
        from_totals[a] += c
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    return [{"from": a, "to": b, "count": c, "share_of_from": c / from_totals[a]} for (a, b), c in ordered]
