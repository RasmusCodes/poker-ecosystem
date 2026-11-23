"""Beta, Beta-binomial and Dirichlet-multinomial numerics (standard library only).

Accuracy notes
--------------
* :func:`betainc` evaluates the regularized incomplete beta function with the
  modified Lentz continued fraction (Numerical Recipes, section 6.4), using the
  symmetry I_x(a, b) = 1 - I_{1-x}(b, a) where the fraction converges faster.
  Relative accuracy is about 1e-14 for parameters up to a few hundred; with
  parameters in the thousands the cancellation inside log B(a, b) limits it
  to about 1e-12.
* :func:`digamma` / :func:`trigamma` use the recurrence up to x >= 10 and the
  asymptotic (Bernoulli) series; absolute error is below 1e-13.
* :func:`beta_ppf` brackets the root and alternates safeguarded Newton steps
  with bisection, so it always converges.

Empirical Bayes
---------------
:func:`fit_beta_prior` fits a population Beta(alpha, beta) prior to per-player
(opportunities, successes) pairs by maximising the Beta-binomial marginal
likelihood. It is an *approximation* to a full hierarchical model: the
hyperparameters are point estimates and their uncertainty is not propagated,
so posterior intervals built on them are slightly too narrow when few players
inform the prior. Degenerate populations (no data, fewer than ``min_players``
players with data, only single-trial players, no successes at all or only
successes) get a weak, documented fallback prior instead of a boundary fit.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

_ETA_MAX = 25.0  # |logit(mean)| bound used by the prior fits


# --------------------------------------------------------------------------- special functions


def _check_ab(a: float, b: float) -> None:
    if not (a > 0 and b > 0 and math.isfinite(a) and math.isfinite(b)):
        raise ValueError(f"Beta parameters must be positive and finite, got a={a}, b={b}")


def log_beta(a: float, b: float) -> float:
    """log B(a, b) via ``math.lgamma``."""
    _check_ab(a, b)
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def digamma(x: float) -> float:
    """psi(x) for x > 0."""
    if not x > 0:
        raise ValueError("digamma is implemented for x > 0")
    acc = 0.0
    while x < 10.0:
        acc -= 1.0 / x
        x += 1.0
    inv2 = 1.0 / (x * x)
    series = inv2 * (1 / 12 - inv2 * (1 / 120 - inv2 * (1 / 252 - inv2 * (1 / 240 - inv2 / 132))))
    return acc + math.log(x) - 0.5 / x - series


def trigamma(x: float) -> float:
    """psi'(x) for x > 0."""
    if not x > 0:
        raise ValueError("trigamma is implemented for x > 0")
    acc = 0.0
    while x < 10.0:
        acc += 1.0 / (x * x)
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    tail = inv * inv2 * (1 / 6 - inv2 * (1 / 30 - inv2 * (1 / 42 - inv2 * (1 / 30 - inv2 * 5 / 66))))
    return acc + inv + 0.5 * inv2 + tail


def _beta_cf(a: float, b: float, x: float, max_iter: int = 200_000, eps: float = 1e-15) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise ArithmeticError(f"incomplete beta continued fraction did not converge (a={a}, b={b}, x={x})")


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b) = Beta(a, b) CDF at x."""
    _check_ab(a, b)
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = a * math.log(x) + b * math.log1p(-x) - log_beta(a, b)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _beta_cf(a, b, x) / a
    return 1.0 - math.exp(log_front) * _beta_cf(b, a, 1.0 - x) / b


def beta_cdf(x: float, a: float, b: float) -> float:
    """Beta(a, b) cumulative distribution function."""
    return betainc(a, b, x)


def beta_logpdf(x: float, a: float, b: float) -> float:
    """log density of Beta(a, b) at x (``-inf`` outside [0, 1])."""
    _check_ab(a, b)
    if x < 0.0 or x > 1.0:
        return -math.inf
    if x == 0.0 or x == 1.0:
        edge = a if x == 0.0 else b
        if edge < 1.0:
            return math.inf
        if edge > 1.0:
            return -math.inf
        return -log_beta(a, b)
    return (a - 1.0) * math.log(x) + (b - 1.0) * math.log1p(-x) - log_beta(a, b)


def beta_pdf(x: float, a: float, b: float) -> float:
    """Density of Beta(a, b) at x."""
    value = beta_logpdf(x, a, b)
    return math.inf if value == math.inf else math.exp(value)


def beta_ppf(q: float, a: float, b: float, *, tol: float = 1e-13, max_iter: int = 1000) -> float:
    """Quantile function of Beta(a, b): the x with I_x(a, b) = q.

    The root is solved on the side of 0.5 where it lies: directly when
    q <= I_0.5(a, b), otherwise as 1 - y with I_y(b, a) = 1 - q. The solved
    variable is then at most 0.5 and carries full relative precision, so
    quantiles very close to 1 are as accurate as double precision allows. The
    root is kept inside a bracket; a Newton step is taken when it lands
    strictly inside, otherwise the bracket is bisected.
    """
    _check_ab(a, b)
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    if q == 0.0:
        return 0.0
    if q == 1.0:
        return 1.0
    if q > betainc(a, b, 0.5):
        return 1.0 - _ppf_lower(1.0 - q, b, a, tol, max_iter)
    return _ppf_lower(q, a, b, tol, max_iter)


def _ppf_lower(q: float, a: float, b: float, tol: float, max_iter: int) -> float:
    """Root of I_x(a, b) = q known to lie in (0, 0.5]."""
    lo, hi = 0.0, 0.5
    x = min(max(a / (a + b), 1e-9), 0.5)
    for _ in range(max_iter):
        err = betainc(a, b, x) - q
        if err == 0.0:
            return x
        if err > 0.0:
            hi = x
        else:
            lo = x
        if hi - lo <= tol * max(hi, 1e-300):
            return 0.5 * (lo + hi)
        dens = beta_pdf(x, a, b)
        candidate = x - err / dens if 0.0 < dens < math.inf else math.nan
        if lo < candidate < hi:
            if abs(candidate - x) <= tol * max(candidate, 1e-300):
                return candidate
            x = candidate
        else:
            x = 0.5 * (lo + hi)
    return x


def beta_mean(a: float, b: float) -> float:
    _check_ab(a, b)
    return a / (a + b)


def beta_var(a: float, b: float) -> float:
    _check_ab(a, b)
    s = a + b
    return a * b / (s * s * (s + 1.0))


def beta_interval(a: float, b: float, level: float = 0.9) -> tuple[float, float]:
    """Equal-tailed credible interval of Beta(a, b)."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")
    tail = (1.0 - level) / 2.0
    return beta_ppf(tail, a, b), beta_ppf(1.0 - tail, a, b)


# --------------------------------------------------------------------------- binomial family


def log_comb(n: int, k: int) -> float:
    """log of the binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def binom_logpmf(k: int, n: int, p: float) -> float:
    if k < 0 or k > n:
        return -math.inf
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return log_comb(n, k) + k * math.log(p) + (n - k) * math.log1p(-p)


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), via I_{1-p}(n - k, k + 1)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    return betainc(n - k, k + 1, 1.0 - p)


def betabinom_logpmf(k: int, n: int, a: float, b: float) -> float:
    """log P(K = k) for K ~ Beta-Binomial(n, a, b)."""
    if k < 0 or k > n:
        return -math.inf
    return log_comb(n, k) + log_beta(k + a, n - k + b) - log_beta(a, b)


def betabinom_pmf(k: int, n: int, a: float, b: float) -> float:
    return math.exp(betabinom_logpmf(k, n, a, b))


def beta_posterior(k: int, n: int, a: float, b: float) -> tuple[float, float]:
    """Conjugate update: Beta(a, b) prior plus k successes in n trials."""
    if k < 0 or n < k:
        raise ValueError("need 0 <= k <= n")
    return a + k, b + n - k


def posterior_predictive(k_new: int, m_new: int, a: float, b: float) -> float:
    """P(k_new successes in m_new future trials) under a Beta(a, b) posterior."""
    return betabinom_pmf(k_new, m_new, a, b)


def order_statistic_interval(n: int, q: float, level: float = 0.9) -> tuple[int, int]:
    """Distribution-free equal-tailed confidence interval for the q-quantile.

    Returns 1-based order-statistic indices ``(l, u)`` such that
    P(X_(l) <= xi_q) >= 1 - tail and P(X_(u) >= xi_q) >= 1 - tail with
    tail = (1 - level) / 2, using P(X_(j) <= xi_q) = P(Binomial(n, q) >= j)
    for continuous distributions. ``l == 0`` means the sample gives no finite
    lower bound, ``u == n + 1`` no finite upper bound.
    """
    if n < 0 or not 0.0 < q < 1.0 or not 0.0 < level < 1.0:
        raise ValueError("need n >= 0, 0 < q < 1 and 0 < level < 1")
    need = 1.0 - (1.0 - level) / 2.0
    lower = 0
    for j in range(n, 0, -1):
        if 1.0 - binom_cdf(j - 1, n, q) >= need:
            lower = j
            break
    upper = n + 1
    for j in range(1, n + 1):
        if binom_cdf(j - 1, n, q) >= need:
            upper = j
            break
    return lower, upper


# --------------------------------------------------------------------------- optimisation helpers

_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


def golden_section_max(f: Callable[[float], float], lo: float, hi: float, *, tol: float = 1e-7, max_iter: int = 200) -> tuple[float, float]:
    """Maximise a unimodal function on [lo, hi]; returns (argmax, max)."""
    a, b = lo, hi
    c = b - _GOLDEN * (b - a)
    d = a + _GOLDEN * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if abs(b - a) <= tol * (1.0 + abs(a) + abs(b)):
            break
        if fc >= fd:
            b, d, fd = d, c, fc
            c = b - _GOLDEN * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + _GOLDEN * (b - a)
            fd = f(d)
    candidates = [(fc, c), (fd, d), (f(lo), lo), (f(hi), hi)]
    best_f, best_x = max(candidates)
    return best_x, best_f


def _expit(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def logit(p: float, eps: float = 1e-12) -> float:
    """log(p / (1 - p)) with p clamped to [eps, 1 - eps]."""
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def expit(x: float) -> float:
    """Inverse of :func:`logit`."""
    return _expit(x)


# --------------------------------------------------------------------------- Beta prior fit


@dataclass(frozen=True)
class BetaPrior:
    """A Beta(alpha, beta) prior with the diagnostics of how it was obtained."""

    alpha: float
    beta: float
    players: int = 0
    opportunities: int = 0
    successes: int = 0
    loglik: float = math.nan
    method: str = "fixed"
    converged: bool = True
    at_bound: bool = False

    @classmethod
    def from_mean(cls, mean: float, concentration: float, **info: object) -> "BetaPrior":
        return cls(mean * concentration, (1.0 - mean) * concentration, **info)  # type: ignore[arg-type]

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def concentration(self) -> float:
        return self.alpha + self.beta

    @property
    def degenerate(self) -> bool:
        return self.method.startswith("degenerate")

    def posterior(self, k: int, n: int) -> tuple[float, float]:
        return beta_posterior(k, n, self.alpha, self.beta)

    def quantile(self, q: float) -> float:
        return beta_ppf(q, self.alpha, self.beta)

    def interval(self, level: float = 0.9) -> tuple[float, float]:
        return beta_interval(self.alpha, self.beta, level)

    def as_dict(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "mean": self.mean,
            "concentration": self.concentration,
            "players": self.players,
            "opportunities": self.opportunities,
            "successes": self.successes,
            "loglik": None if math.isnan(self.loglik) else self.loglik,
            "method": self.method,
            "converged": self.converged,
            "at_bound": self.at_bound,
        }


def _group_counts(data: Iterable[Sequence[int]]) -> "Counter[tuple[int, int]]":
    groups: Counter[tuple[int, int]] = Counter()
    for item in data:
        n, k = int(item[0]), int(item[1])
        if n < 0 or k < 0 or k > n:
            raise ValueError(f"invalid counts n={n}, k={k}")
        if n > 0:
            groups[(n, k)] += 1
    return groups


def marginal_loglik(groups: "Counter[tuple[int, int]]", alpha: float, beta: float) -> float:
    """Beta-binomial marginal log-likelihood, binomial coefficients omitted."""
    lb = log_beta(alpha, beta)
    kappa = alpha + beta
    total = 0.0
    for (n, k), m in groups.items():
        total += m * (math.lgamma(k + alpha) + math.lgamma(n - k + beta) - math.lgamma(n + kappa) - lb)
    return total


def boundary_penalty(mean: float, concentration: float) -> float:
    """log sd(theta) of Beta(mean * kappa, (1 - mean) * kappa).

    Adding it to the marginal log-likelihood is a Gamma(2, rate -> 0) prior on
    the between-player standard deviation (Chung, Rabe-Hesketh, Dorie, Gelman
    and Liu 2013, "A nondegenerate penalized likelihood estimator for variance
    parameters in multilevel models"): it keeps the fit off the degenerate
    boundary kappa -> infinity (no spread) and is negligible with many players.
    """
    return 0.5 * math.log(mean * (1.0 - mean)) - 0.5 * math.log1p(concentration)


def _ll_eta_tau(groups: "Counter[tuple[int, int]]", eta: float, tau: float, penalised: bool = False) -> float:
    kappa = math.exp(tau)
    value = marginal_loglik(groups, _expit(eta) * kappa, _expit(-eta) * kappa)
    if penalised:
        value += boundary_penalty(_expit(eta), kappa)
    return value


def _penalty_derivatives(eta: float, tau: float) -> tuple[float, float, float, float]:
    """(d/deta, d/dtau, d2/deta2, d2/dtau2) of :func:`boundary_penalty`; the cross term is 0."""
    mu, kappa = _expit(eta), math.exp(tau)
    return 0.5 * (1.0 - 2.0 * mu), -0.5 * kappa / (kappa + 1.0), -mu * (1.0 - mu), -0.5 * kappa / (kappa + 1.0) ** 2


def _derivatives(groups: "Counter[tuple[int, int]]", eta: float, tau: float) -> tuple[float, float, float, float, float]:
    """Gradient and Hessian of the marginal log-likelihood in (logit mean, log concentration)."""
    kappa = math.exp(tau)
    mu = _expit(eta)
    a, b = mu * kappa, _expit(-eta) * kappa
    dg_a, dg_b, dg_k = digamma(a), digamma(b), digamma(kappa)
    tg_a, tg_b, tg_k = trigamma(a), trigamma(b), trigamma(kappa)
    ga = gb = haa = hbb = hab = 0.0
    for (n, k), m in groups.items():
        dg_n, tg_n = digamma(n + kappa), trigamma(n + kappa)
        ga += m * (digamma(k + a) - dg_a - dg_n + dg_k)
        gb += m * (digamma(n - k + b) - dg_b - dg_n + dg_k)
        haa += m * (trigamma(k + a) - tg_a - tg_n + tg_k)
        hbb += m * (trigamma(n - k + b) - tg_b - tg_n + tg_k)
        hab += m * (tg_k - tg_n)
    s = mu * (1.0 - mu)
    ks = kappa * s
    g_eta = ks * (ga - gb)
    g_tau = a * ga + b * gb
    h_ee = ks * (1.0 - 2.0 * mu) * (ga - gb) + ks * ks * (haa - 2.0 * hab + hbb)
    h_et = ks * (ga - gb) + ks * ((haa * a + hab * b) - (hab * a + hbb * b))
    h_tt = g_tau + a * a * haa + 2.0 * a * b * hab + b * b * hbb
    return g_eta, g_tau, h_ee, h_et, h_tt


def _ascent_step(g1: float, g2: float, h11: float, h12: float, h22: float) -> tuple[float, float]:
    """Newton step, Levenberg-shifted when the Hessian is not negative definite."""
    half_tr = 0.5 * (h11 + h22)
    det = h11 * h22 - h12 * h12
    disc = math.sqrt(max(half_tr * half_tr - det, 0.0))
    lam_max, lam_min = half_tr + disc, half_tr - disc
    if not lam_max < 0.0:
        shift = lam_max + 1e-3 * (abs(lam_min) + 1.0)
        h11 -= shift
        h22 -= shift
        det = h11 * h22 - h12 * h12
    if not abs(det) > 1e-300:
        return g1, g2
    return (-g1 * h22 + g2 * h12) / det, (g1 * h12 - g2 * h11) / det


def _newton_fit(
    groups: "Counter[tuple[int, int]]",
    eta: float,
    tau: float,
    bounds: tuple[float, float],
    max_iter: int,
    penalised: bool,
) -> tuple[float, float, float, bool]:
    lo, hi = bounds
    eta = min(max(eta, -_ETA_MAX), _ETA_MAX)
    tau = min(max(tau, lo), hi)
    f = _ll_eta_tau(groups, eta, tau, penalised)
    scale = 1.0 + sum(groups.values())
    for _ in range(max_iter):
        g_e, g_t, h_ee, h_et, h_tt = _derivatives(groups, eta, tau)
        if penalised:
            pe, pt, pee, ptt = _penalty_derivatives(eta, tau)
            g_e, g_t, h_ee, h_tt = g_e + pe, g_t + pt, h_ee + pee, h_tt + ptt
        pinned = (tau <= lo and g_t < 0.0) or (tau >= hi and g_t > 0.0)
        g_free = 0.0 if pinned else g_t
        if max(abs(g_e), abs(g_free)) < 1e-8 * scale:
            return eta, tau, f, True
        if pinned:
            step = (-g_e / h_ee if h_ee < 0.0 else g_e, 0.0)
        else:
            step = _ascent_step(g_e, g_t, h_ee, h_et, h_tt)
        norm = math.hypot(*step)
        if norm > 2.0:
            step = (2.0 * step[0] / norm, 2.0 * step[1] / norm)
        t = 1.0
        while t >= 1e-10:
            e2 = min(max(eta + t * step[0], -_ETA_MAX), _ETA_MAX)
            t2 = min(max(tau + t * step[1], lo), hi)
            f2 = _ll_eta_tau(groups, e2, t2, penalised)
            if f2 >= f:
                break
            t *= 0.5
        else:
            return eta, tau, f, max(abs(g_e), abs(g_free)) < 1e-4 * scale
        eta, tau, f = e2, t2, f2
    return eta, tau, f, False


def _profile_golden_fit(groups: "Counter[tuple[int, int]]", bounds: tuple[float, float], penalised: bool) -> tuple[float, float, float]:
    def best_eta(tau: float) -> tuple[float, float]:
        return golden_section_max(lambda e: _ll_eta_tau(groups, e, tau, penalised), -_ETA_MAX, _ETA_MAX, tol=1e-9)

    tau, _ = golden_section_max(lambda t: best_eta(t)[1], bounds[0], bounds[1], tol=1e-8)
    eta, f = best_eta(tau)
    return eta, tau, f


def _moment_start(groups: "Counter[tuple[int, int]]", opportunities: int, successes: int, kmin: float, kmax: float) -> tuple[float, float]:
    """Method-of-moments start: pooled mean and the intra-class correlation."""
    players = sum(groups.values())
    mu = min(max(successes / opportunities, 1e-3), 1.0 - 1e-3)
    mean_rate = sum(m * k / n for (n, k), m in groups.items()) / players
    var = sum(m * (k / n - mean_rate) ** 2 for (n, k), m in groups.items()) / players
    inv_n = sum(m / n for (n, _), m in groups.items()) / players
    rho = 0.5 if inv_n >= 1.0 else (var / (mu * (1.0 - mu)) - inv_n) / (1.0 - inv_n)
    rho = min(max(rho, 1.0 / (1.0 + kmax)), 1.0 / (1.0 + kmin))
    return mu, min(max(1.0 / rho - 1.0, kmin), kmax)


def _degenerate_reason(groups: "Counter[tuple[int, int]]", players: int, n_total: int, k_total: int, min_players: int) -> str | None:
    if players < min_players:
        return "too-few-players"
    if max(n for n, _ in groups) <= 1:
        return "single-trials"
    if k_total == 0:
        return "no-successes"
    if k_total == n_total:
        return "all-successes"
    return None


def fit_beta_prior(
    data: Iterable[Sequence[int]],
    *,
    min_players: int = 3,
    min_concentration: float = 0.05,
    max_concentration: float = 1e4,
    fallback_concentration: float = 2.0,
    max_iter: int = 100,
    boundary_avoiding: bool = True,
) -> BetaPrior:
    """Empirical-Bayes Beta prior from per-player (opportunities, successes).

    Maximises the Beta-binomial marginal likelihood over (logit mean,
    log concentration) with a guarded Newton method started at the method of
    moments; the concentration is restricted to
    [``min_concentration``, ``max_concentration``] (``at_bound`` flags a fit
    on the boundary). With ``boundary_avoiding`` (default) the objective also
    contains :func:`boundary_penalty`, a weak Gamma(2) prior on the
    between-player standard deviation, so the fit is a posterior mode rather
    than a pure maximum-likelihood estimate; it prevents the classic
    collapse kappa -> infinity (every player shrunk to the mean) when a small
    sample cannot detect the spread. Set ``boundary_avoiding=False`` for pure
    maximum marginal likelihood. If Newton fails, a profile golden-section
    search is used. Players with zero opportunities carry no information and
    are ignored. Degenerate inputs return a weak prior with mean
    (K + 0.5) / (N + 1) and concentration ``fallback_concentration``
    (method ``degenerate:<reason>``).
    """
    groups = _group_counts(data)
    players = sum(groups.values())
    n_total = sum(n * m for (n, _), m in groups.items())
    k_total = sum(k * m for (_, k), m in groups.items())
    info = {"players": players, "opportunities": n_total, "successes": k_total}
    if players == 0:
        return BetaPrior(1.0, 1.0, method="degenerate:no-data", **info)
    reason = _degenerate_reason(groups, players, n_total, k_total, min_players)
    if reason is not None:
        mean = (k_total + 0.5) / (n_total + 1.0)
        return BetaPrior.from_mean(mean, fallback_concentration, method=f"degenerate:{reason}", **info)
    bounds = (math.log(min_concentration), math.log(max_concentration))
    mu0, kappa0 = _moment_start(groups, n_total, k_total, min_concentration, max_concentration)
    eta, tau, objective, converged = _newton_fit(groups, logit(mu0), math.log(kappa0), bounds, max_iter, boundary_avoiding)
    method = "newton"
    if not converged:
        eta_g, tau_g, objective_g = _profile_golden_fit(groups, bounds, boundary_avoiding)
        if objective_g >= objective:
            eta, tau, objective, method = eta_g, tau_g, objective_g, "golden"
        converged = True
    if boundary_avoiding:
        method += "+boundary-prior"
    kappa = math.exp(tau)
    at_bound = tau <= bounds[0] + 1e-6 or tau >= bounds[1] - 1e-6
    return BetaPrior(
        _expit(eta) * kappa,
        _expit(-eta) * kappa,
        loglik=_ll_eta_tau(groups, eta, tau),
        method=method,
        converged=converged,
        at_bound=at_bound,
        **info,
    )


@dataclass(frozen=True)
class ConcentrationFit:
    """Maximum-likelihood concentration around player-specific prior means."""

    concentration: float | None
    method: str
    players: int
    at_bound: bool = False


def fit_concentration(
    data: Iterable[tuple[int, int, float]],
    *,
    min_players: int = 3,
    min_concentration: float = 0.05,
    max_concentration: float = 1e4,
) -> ConcentrationFit:
    """Fit kappa in k_i ~ Beta-Binomial(n_i, m_i * kappa, (1 - m_i) * kappa).

    ``data`` holds (n_i, k_i, m_i) with known prior means m_i. Golden-section
    search on log kappa. Returns ``concentration=None`` when fewer than
    ``min_players`` players have data or every player has a single trial.
    """
    items = [(int(n), int(k), min(max(float(m), 1e-9), 1.0 - 1e-9)) for n, k, m in data if n > 0]
    if len(items) < min_players:
        return ConcentrationFit(None, "too-few-players", len(items))
    if max(n for n, _, _ in items) <= 1:
        return ConcentrationFit(None, "single-trials", len(items))

    def loglik(tau: float) -> float:
        kappa = math.exp(tau)
        lg_kappa = math.lgamma(kappa)
        total = 0.0
        for n, k, m in items:
            a, b = m * kappa, (1.0 - m) * kappa
            total += math.lgamma(k + a) + math.lgamma(n - k + b) - math.lgamma(n + kappa)
            total -= math.lgamma(a) + math.lgamma(b) - lg_kappa
        return total

    lo, hi = math.log(min_concentration), math.log(max_concentration)
    tau, _ = golden_section_max(loglik, lo, hi, tol=1e-8)
    at_bound = tau <= lo + 1e-6 or tau >= hi - 1e-6
    return ConcentrationFit(math.exp(tau), "ml", len(items), at_bound)


def concentration_from_logit_variance(
    mean: float, variance: float, *, min_concentration: float = 0.05, max_concentration: float = 1e4
) -> float:
    """Beta concentration whose variance matches a logit-scale variance (delta method).

    Var(theta) ~= (m (1 - m))^2 * Var(logit theta) and a Beta with mean m has
    Var = m (1 - m) / (kappa + 1), so kappa = 1 / (m (1 - m) Var(logit)) - 1,
    clamped to [min_concentration, max_concentration]. First-order only: for
    logit variances of order 1 or more it is a rough match.
    """
    m = min(max(mean, 1e-12), 1.0 - 1e-12)
    scaled = m * (1.0 - m) * max(variance, 0.0)
    kappa = 1.0 / scaled - 1.0 if scaled > 0 else max_concentration
    return min(max(kappa, min_concentration), max_concentration)


@dataclass(frozen=True)
class VarianceFit:
    """Maximum-likelihood residual variance on the logit scale."""

    variance: float | None
    method: str
    players: int
    at_bound: bool = False


def fit_logit_variance(
    data: Iterable[tuple[int, int, float, float]],
    *,
    min_players: int = 3,
    min_variance: float = 1e-6,
    max_variance: float = 10.0,
    min_concentration: float = 0.05,
    max_concentration: float = 1e4,
) -> VarianceFit:
    """Fit a shared residual variance s2 on the logit scale.

    ``data`` holds (n_i, k_i, m_i, v_i): counts, the prior mean and a known
    player-specific logit variance v_i (e.g. the uncertainty of an estimated
    shift). The model is k_i ~ Beta-Binomial(n_i, m_i kappa_i, (1 - m_i) kappa_i)
    with kappa_i = :func:`concentration_from_logit_variance` (m_i, v_i + s2).
    Golden-section search on log s2 in [min_variance, max_variance]; a fit at
    the lower bound means "no residual spread beyond v_i". Returns
    ``variance=None`` with fewer than ``min_players`` players or only single
    trials.
    """
    items = [(int(n), int(k), float(m), max(float(v), 0.0)) for n, k, m, v in data if n > 0]
    if len(items) < min_players:
        return VarianceFit(None, "too-few-players", len(items))
    if max(n for n, _, _, _ in items) <= 1:
        return VarianceFit(None, "single-trials", len(items))
    bounds = {"min_concentration": min_concentration, "max_concentration": max_concentration}

    def loglik(log_s2: float) -> float:
        s2 = math.exp(log_s2)
        total = 0.0
        for n, k, m, v in items:
            kappa = concentration_from_logit_variance(m, v + s2, **bounds)
            a, b = m * kappa, (1.0 - m) * kappa
            total += math.lgamma(k + a) + math.lgamma(n - k + b) - math.lgamma(n + kappa)
            total -= math.lgamma(a) + math.lgamma(b) - math.lgamma(kappa)
        return total

    lo, hi = math.log(min_variance), math.log(max_variance)
    log_s2, _ = golden_section_max(loglik, lo, hi, tol=1e-8)
    at_bound = log_s2 <= lo + 1e-6 or log_s2 >= hi - 1e-6
    return VarianceFit(math.exp(log_s2), "ml", len(items), at_bound)


# --------------------------------------------------------------------------- Dirichlet-multinomial


def dirichlet_multinomial_logpmf(counts: Sequence[int], alphas: Sequence[float]) -> float:
    """log P(counts) under a Dirichlet-multinomial with parameters ``alphas``."""
    if len(counts) != len(alphas):
        raise ValueError("counts and alphas must have the same length")
    if any(a <= 0 for a in alphas) or any(c < 0 for c in counts):
        raise ValueError("need positive alphas and non-negative counts")
    n = sum(counts)
    total_alpha = sum(alphas)
    out = math.lgamma(n + 1) - sum(math.lgamma(c + 1) for c in counts)
    out += math.lgamma(total_alpha) - math.lgamma(n + total_alpha)
    out += sum(math.lgamma(c + a) - math.lgamma(a) for c, a in zip(counts, alphas))
    return out


@dataclass(frozen=True)
class DirichletPosterior:
    """Dirichlet posterior over the probabilities of K categories."""

    alphas: tuple[float, ...]

    @property
    def concentration(self) -> float:
        return sum(self.alphas)

    def mean(self) -> list[float]:
        total = self.concentration
        return [a / total for a in self.alphas]

    def interval(self, j: int, level: float = 0.9) -> tuple[float, float]:
        """Marginal credible interval of category j (its marginal is a Beta)."""
        a = self.alphas[j]
        return beta_interval(a, self.concentration - a, level)

    def predictive_logpmf(self, counts: Sequence[int]) -> float:
        """log P(future counts) under the posterior predictive (Dirichlet-multinomial)."""
        return dirichlet_multinomial_logpmf(counts, self.alphas)


@dataclass(frozen=True)
class DirichletPrior:
    """Population Dirichlet prior with diagnostics."""

    alphas: tuple[float, ...]
    players: int = 0
    method: str = "fixed"
    at_bound: bool = False

    @property
    def concentration(self) -> float:
        return sum(self.alphas)

    @property
    def mean(self) -> list[float]:
        total = self.concentration
        return [a / total for a in self.alphas]

    def posterior(self, counts: Sequence[int]) -> DirichletPosterior:
        if len(counts) != len(self.alphas):
            raise ValueError("counts must have one entry per category")
        return DirichletPosterior(tuple(a + c for a, c in zip(self.alphas, counts)))


def fit_dirichlet_prior(
    count_vectors: Iterable[Sequence[int]],
    *,
    size: int | None = None,
    smoothing: float = 0.5,
    min_players: int = 2,
    min_concentration: float = 1.0,
    max_concentration: float = 1e4,
    fallback_concentration: float = 5.0,
) -> DirichletPrior:
    """Population Dirichlet prior for per-player category counts.

    The prior mean is fixed at the smoothed pooled proportions
    (c_j + smoothing) / (C + K * smoothing); the concentration is the maximum
    of the Dirichlet-multinomial marginal likelihood (profile, golden-section
    on log concentration) within [``min_concentration``,
    ``max_concentration``]. The default floor of 1 (the population is worth
    at least one observation) is a deliberate regularisation: with a few
    observations per player the likelihood prefers no pooling at all, which
    would turn three bets of one size into near-certainty. With fewer than
    ``min_players`` players with data the concentration is
    ``fallback_concentration`` (method ``degenerate``).
    """
    vectors = [[int(c) for c in v] for v in count_vectors]
    if size is None:
        if not vectors:
            raise ValueError("size is required when there are no count vectors")
        size = len(vectors[0])
    if any(len(v) != size for v in vectors):
        raise ValueError("all count vectors need the same length")
    totals = [sum(v[j] for v in vectors) for j in range(size)]
    grand = sum(totals)
    mean = [(t + smoothing) / (grand + size * smoothing) for t in totals]
    active = [v for v in vectors if sum(v) > 0]
    if len(active) < min_players or max(sum(v) for v in active) <= 1:
        alphas = tuple(m * fallback_concentration for m in mean)
        return DirichletPrior(alphas, players=len(active), method="degenerate:too-few-players")

    def loglik(log_s: float) -> float:
        s = math.exp(log_s)
        scaled = [s * m for m in mean]
        base = sum(math.lgamma(a) for a in scaled)
        total = 0.0
        for v in active:
            total += math.lgamma(s) - math.lgamma(sum(v) + s) - base
            total += sum(math.lgamma(c + a) for c, a in zip(v, scaled))
        return total

    lo, hi = math.log(min_concentration), math.log(max_concentration)
    log_s, _ = golden_section_max(loglik, lo, hi, tol=1e-8)
    s = math.exp(log_s)
    at_bound = log_s <= lo + 1e-6 or log_s >= hi - 1e-6
    return DirichletPrior(tuple(m * s for m in mean), players=len(active), method="ml", at_bound=at_bound)
