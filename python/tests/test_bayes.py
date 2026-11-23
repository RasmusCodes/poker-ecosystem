import itertools
import math
import random
import unittest
from collections import Counter

from poker_ecosystem import bayes

EULER = 0.5772156649015329


def beta23_cdf(x: float) -> float:
    """Closed-form CDF of Beta(2, 3): pdf 12 x (1 - x)^2."""
    return 6 * x**2 - 8 * x**3 + 3 * x**4


def solve_increasing(f, target: float, lo: float = 0.0, hi: float = 1.0) -> float:
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def simpson(f, a: float, b: float, n: int = 2000) -> float:
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += (4 if i % 2 else 2) * f(a + i * h)
    return total * h / 3


def binomial_draw(rng: random.Random, n: int, p: float) -> int:
    return sum(rng.random() < p for _ in range(n))


class SpecialFunctionTests(unittest.TestCase):
    def test_log_beta(self) -> None:
        self.assertAlmostEqual(bayes.log_beta(2, 3), math.log(1 / 12), places=14)
        self.assertAlmostEqual(bayes.log_beta(0.5, 0.5), math.log(math.pi), places=13)
        with self.assertRaises(ValueError):
            bayes.log_beta(0, 1)

    def test_digamma(self) -> None:
        self.assertAlmostEqual(bayes.digamma(1.0), -EULER, places=13)
        self.assertAlmostEqual(bayes.digamma(0.5), -EULER - 2 * math.log(2), places=13)
        for x in (0.3, 7.5, 12.0, 250.0):
            self.assertAlmostEqual(bayes.digamma(x + 1), bayes.digamma(x) + 1 / x, places=12)

    def test_trigamma(self) -> None:
        self.assertAlmostEqual(bayes.trigamma(1.0), math.pi**2 / 6, places=13)
        self.assertAlmostEqual(bayes.trigamma(0.5), math.pi**2 / 2, places=12)
        for x in (0.3, 7.5, 12.0, 250.0):
            self.assertAlmostEqual(bayes.trigamma(x + 1), bayes.trigamma(x) - 1 / x**2, places=12)

    def test_betainc_known_values(self) -> None:
        for x in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999):
            self.assertAlmostEqual(bayes.betainc(2, 3, x), beta23_cdf(x), places=13)
            self.assertAlmostEqual(bayes.betainc(1, 1, x), x, places=14)
            self.assertAlmostEqual(bayes.betainc(1, 4.5, x), 1 - (1 - x) ** 4.5, places=13)
            self.assertAlmostEqual(bayes.betainc(2, 3, x), 1 - bayes.betainc(3, 2, 1 - x), places=13)
        self.assertAlmostEqual(bayes.betainc(2, 3, 0.5), 0.6875, places=14)
        for a in (0.5, 3.0, 40.0):
            self.assertAlmostEqual(bayes.betainc(a, a, 0.5), 0.5, places=13)
        self.assertAlmostEqual(bayes.betainc(2500.0, 2500.0, 0.5), 0.5, delta=1e-11)  # documented ~1e-12 regime
        self.assertEqual(bayes.betainc(2, 3, 0.0), 0.0)
        self.assertEqual(bayes.betainc(2, 3, 1.0), 1.0)

    def test_betainc_matches_numerical_integration(self) -> None:
        a, b, x = 2.5, 3.5, 0.4
        integral = simpson(lambda t: bayes.beta_pdf(t, a, b), 0.0, x)
        self.assertAlmostEqual(bayes.betainc(a, b, x), integral, places=9)

    def test_beta_ppf(self) -> None:
        median = solve_increasing(beta23_cdf, 0.5)
        self.assertAlmostEqual(bayes.beta_ppf(0.5, 2, 3), median, places=11)
        self.assertAlmostEqual(bayes.beta_ppf(0.5, 2, 3), 0.38572756813238945, places=11)
        self.assertAlmostEqual(bayes.beta_ppf(0.5, 3, 3), 0.5, places=12)
        cases = [(2, 3), (0.3, 50), (500, 800), (0.5, 0.5), (1, 1), (25, 0.2), (1e4, 3e4)]
        for (a, b), q in itertools.product(cases, (1e-4, 0.05, 0.5, 0.95, 0.9999)):
            x = bayes.beta_ppf(q, a, b)
            y = bayes.beta_ppf(1 - q, b, a)
            self.assertAlmostEqual(x, 1 - y, delta=1e-12, msg=f"reflection {a},{b},{q}")
            # Relative accuracy of the tail probability on the side that carries the precision.
            if q <= bayes.betainc(a, b, 0.5):
                self.assertAlmostEqual(bayes.betainc(a, b, x), q, delta=1e-9 * q, msg=f"{a},{b},{q}")
            else:
                self.assertAlmostEqual(bayes.betainc(b, a, y), 1 - q, delta=1e-9 * (1 - q), msg=f"{a},{b},{q}")
        # The 0.9999 quantile of Beta(25, 0.2) is 1 - 2.6e-22: it rounds to 1.0, its mirror is exact.
        self.assertEqual(bayes.beta_ppf(0.9999, 25, 0.2), 1.0)
        self.assertGreater(bayes.beta_ppf(1e-4, 0.2, 25), 1e-23)
        self.assertEqual(bayes.beta_ppf(0.0, 2, 3), 0.0)
        self.assertEqual(bayes.beta_ppf(1.0, 2, 3), 1.0)
        with self.assertRaises(ValueError):
            bayes.beta_ppf(1.5, 2, 3)

    def test_beta_interval(self) -> None:
        lo, hi = bayes.beta_interval(20, 30, 0.9)
        self.assertLess(lo, 0.4)
        self.assertGreater(hi, 0.4)
        self.assertAlmostEqual(bayes.betainc(20, 30, lo), 0.05, places=10)
        lo2, hi2 = bayes.beta_interval(200, 300, 0.9)
        self.assertLess(hi2 - lo2, hi - lo)
        self.assertAlmostEqual(bayes.beta_mean(20, 30), 0.4)
        self.assertAlmostEqual(bayes.beta_var(2, 3), 6 / (25 * 6))
        with self.assertRaises(ValueError):
            bayes.beta_interval(2, 3, 1.0)


class BinomialTests(unittest.TestCase):
    def test_beta_binomial_sums_to_one(self) -> None:
        for n, (a, b) in itertools.product((0, 1, 10, 57), ((2.5, 0.7), (1.0, 1.0), (30.0, 12.0))):
            total = sum(bayes.betabinom_pmf(k, n, a, b) for k in range(n + 1))
            self.assertAlmostEqual(total, 1.0, places=12)
        self.assertEqual(bayes.betabinom_logpmf(-1, 5, 1, 1), -math.inf)

    def test_beta_binomial_special_cases(self) -> None:
        for k in range(8):
            self.assertAlmostEqual(bayes.betabinom_pmf(k, 7, 1, 1), 1 / 8, places=13)
        p, big = 0.3, 1e7
        for k in range(6):
            binom = math.exp(bayes.binom_logpmf(k, 5, p))
            self.assertAlmostEqual(bayes.betabinom_pmf(k, 5, big * p, big * (1 - p)), binom, places=6)

    def test_binomial_cdf(self) -> None:
        for n, p in ((10, 0.3), (25, 0.9), (1, 0.5)):
            running = 0.0
            for k in range(n + 1):
                running += math.exp(bayes.binom_logpmf(k, n, p))
                self.assertAlmostEqual(bayes.binom_cdf(k, n, p), running, places=12)
        self.assertEqual(bayes.binom_cdf(-1, 5, 0.5), 0.0)

    def test_posterior_predictive(self) -> None:
        a, b = bayes.beta_posterior(3, 10, 1.5, 2.0)
        self.assertEqual((a, b), (4.5, 9.0))
        self.assertAlmostEqual(sum(bayes.posterior_predictive(k, 6, a, b) for k in range(7)), 1.0, places=12)
        with self.assertRaises(ValueError):
            bayes.beta_posterior(5, 3, 1, 1)

    def test_order_statistic_interval(self) -> None:
        n, q = 50, 0.1
        lo, hi = bayes.order_statistic_interval(n, q, 0.9)
        self.assertGreaterEqual(1 - bayes.binom_cdf(lo - 1, n, q), 0.95)
        self.assertLess(1 - bayes.binom_cdf(lo, n, q), 0.95)
        self.assertGreaterEqual(bayes.binom_cdf(hi - 1, n, q), 0.95)
        self.assertLess(bayes.binom_cdf(hi - 2, n, q), 0.95)
        self.assertEqual(bayes.order_statistic_interval(3, 0.1, 0.9), (0, 2))


class PriorFitTests(unittest.TestCase):
    def test_recovers_known_prior(self) -> None:
        rng = random.Random(20261121)
        data = []
        for _ in range(600):
            theta = rng.betavariate(2.0, 5.0)
            n = rng.randint(20, 200)
            data.append((n, binomial_draw(rng, n, theta)))
        for boundary in (False, True):
            prior = bayes.fit_beta_prior(data, boundary_avoiding=boundary)
            self.assertTrue(prior.converged)
            self.assertTrue(prior.method.startswith("newton"))
            self.assertAlmostEqual(prior.mean, 2 / 7, delta=0.015)
            self.assertGreater(prior.concentration, 5.5)
            self.assertLess(prior.concentration, 9.0)
            self.assertEqual(prior.players, 600)
        self.assertEqual(prior.method, "newton+boundary-prior")
        # Zero-opportunity players carry no information.
        same = bayes.fit_beta_prior(data + [(0, 0)] * 50)
        self.assertAlmostEqual(same.alpha, prior.alpha, places=9)
        self.assertAlmostEqual(same.beta, prior.beta, places=9)

    def test_fit_is_a_local_maximum_and_golden_agrees(self) -> None:
        rng = random.Random(5)
        data = [(n, binomial_draw(rng, n, rng.betavariate(6, 4))) for n in (rng.randint(5, 60) for _ in range(150))]
        groups = Counter((n, k) for n, k in data)
        steps = ((1.01, 1.0), (0.99, 1.0), (1.0, 1.01), (1.0, 0.99), (1.01, 1.01), (0.99, 0.99))

        def score(a: float, b: float, boundary: bool) -> float:
            value = bayes.marginal_loglik(groups, a, b)
            return value + bayes.boundary_penalty(a / (a + b), a + b) if boundary else value

        for boundary in (False, True):
            prior = bayes.fit_beta_prior(data, boundary_avoiding=boundary)
            best = score(prior.alpha, prior.beta, boundary)
            for da, db in steps:
                self.assertLess(score(prior.alpha * da, prior.beta * db, boundary), best)
            golden = bayes.fit_beta_prior(data, max_iter=0, boundary_avoiding=boundary)
            self.assertTrue(golden.method.startswith("golden"))
            self.assertAlmostEqual(golden.mean, prior.mean, places=5)
            self.assertAlmostEqual(math.log(golden.concentration), math.log(prior.concentration), places=4)

    def test_boundary_prior_keeps_small_samples_off_the_boundary(self) -> None:
        # Three opportunities per player cannot reveal the spread between players.
        rng = random.Random(12)
        data = [(3, binomial_draw(rng, 3, rng.choice((0.03, 0.15)))) for _ in range(12)]
        pure = bayes.fit_beta_prior(data, boundary_avoiding=False)
        soft = bayes.fit_beta_prior(data)
        self.assertLessEqual(soft.concentration, pure.concentration)
        self.assertFalse(soft.at_bound)

    def test_degenerate_populations(self) -> None:
        empty = bayes.fit_beta_prior([])
        self.assertEqual((empty.alpha, empty.beta, empty.method), (1.0, 1.0, "degenerate:no-data"))
        self.assertEqual(bayes.fit_beta_prior([(10, 3), (0, 0)]).method, "degenerate:too-few-players")
        zeros = bayes.fit_beta_prior([(10, 0)] * 5)
        self.assertEqual(zeros.method, "degenerate:no-successes")
        self.assertAlmostEqual(zeros.mean, 0.5 / 51)
        self.assertAlmostEqual(zeros.concentration, 2.0)
        self.assertEqual(bayes.fit_beta_prior([(10, 10)] * 4).method, "degenerate:all-successes")
        self.assertEqual(bayes.fit_beta_prior([(1, 0), (1, 1), (1, 1)]).method, "degenerate:single-trials")
        with self.assertRaises(ValueError):
            bayes.fit_beta_prior([(3, 4)])

    def test_homogeneous_population_gets_a_large_concentration(self) -> None:
        rng = random.Random(9)
        data = [(50, binomial_draw(rng, 50, 0.3)) for _ in range(200)]
        prior = bayes.fit_beta_prior(data, max_concentration=1e4)
        self.assertAlmostEqual(prior.mean, 0.3, delta=0.02)
        self.assertGreater(prior.concentration, 100)

    def test_fit_concentration_recovers_kappa(self) -> None:
        rng = random.Random(3)
        kappa = 20.0
        data = []
        for _ in range(400):
            m = rng.uniform(0.1, 0.6)
            theta = rng.betavariate(m * kappa, (1 - m) * kappa)
            n = rng.randint(50, 150)
            data.append((n, binomial_draw(rng, n, theta), m))
        fit = bayes.fit_concentration(data)
        self.assertEqual(fit.method, "ml")
        self.assertGreater(fit.concentration, 14)
        self.assertLess(fit.concentration, 28)
        self.assertIsNone(bayes.fit_concentration(data[:2]).concentration)

    def test_logit_variance_link_and_fit(self) -> None:
        self.assertAlmostEqual(bayes.concentration_from_logit_variance(0.5, 0.04), 99.0)
        self.assertEqual(bayes.concentration_from_logit_variance(0.5, 0.0), 1e4)
        self.assertEqual(bayes.concentration_from_logit_variance(0.5, 1e3), 0.05)
        rng = random.Random(21)
        data = []
        for _ in range(400):
            m = rng.uniform(0.15, 0.5)
            p = bayes.expit(bayes.logit(m) + rng.gauss(0.0, 0.5))
            n = rng.randint(60, 160)
            data.append((n, binomial_draw(rng, n, p), m, 0.0))
        fit = bayes.fit_logit_variance(data)
        self.assertEqual(fit.method, "ml")
        self.assertGreater(fit.variance, 0.15)  # true logit variance 0.25 (delta-method link)
        self.assertLess(fit.variance, 0.40)
        # A known per-player variance absorbs that much of the spread.
        shifted = bayes.fit_logit_variance([(n, k, m, 0.1) for n, k, m, _ in data])
        self.assertLess(shifted.variance, fit.variance)
        self.assertIsNone(bayes.fit_logit_variance(data[:2]).variance)


class DirichletTests(unittest.TestCase):
    def test_pmf_sums_to_one_and_reduces_to_beta_binomial(self) -> None:
        alphas = (0.7, 2.0, 1.3)
        n = 4
        total = sum(
            math.exp(bayes.dirichlet_multinomial_logpmf(c, alphas))
            for c in itertools.product(range(n + 1), repeat=3)
            if sum(c) == n
        )
        self.assertAlmostEqual(total, 1.0, places=12)
        for k in range(6):
            dm = bayes.dirichlet_multinomial_logpmf((k, 5 - k), (2.5, 0.7))
            self.assertAlmostEqual(dm, bayes.betabinom_logpmf(k, 5, 2.5, 0.7), places=12)

    def test_posterior(self) -> None:
        prior = bayes.DirichletPrior((1.0, 2.0, 3.0))
        post = prior.posterior((4, 0, 1))
        self.assertEqual(post.alphas, (5.0, 2.0, 4.0))
        for got, want in zip(post.mean(), (5 / 11, 2 / 11, 4 / 11)):
            self.assertAlmostEqual(got, want)
        self.assertEqual(post.interval(0, 0.9), bayes.beta_interval(5.0, 6.0, 0.9))
        self.assertAlmostEqual(
            sum(math.exp(post.predictive_logpmf(c)) for c in itertools.product(range(3), repeat=3) if sum(c) == 2),
            1.0,
            places=12,
        )

    def test_fit_recovers_mean_and_concentration(self) -> None:
        rng = random.Random(17)
        mean, s = (0.5, 0.3, 0.2), 8.0
        vectors = []
        for _ in range(300):
            draws = [rng.gammavariate(s * m, 1.0) for m in mean]
            p = [d / sum(draws) for d in draws]
            counts = Counter(rng.choices(range(3), weights=p, k=30))
            vectors.append([counts[j] for j in range(3)])
        prior = bayes.fit_dirichlet_prior(vectors)
        self.assertEqual(prior.method, "ml")
        for got, want in zip(prior.mean, mean):
            self.assertAlmostEqual(got, want, delta=0.03)
        self.assertGreater(prior.concentration, 5.0)
        self.assertLess(prior.concentration, 13.0)

    def test_fit_degenerate(self) -> None:
        one = bayes.fit_dirichlet_prior([[3, 1, 0]])
        self.assertTrue(one.method.startswith("degenerate"))
        self.assertAlmostEqual(one.concentration, 5.0)
        empty = bayes.fit_dirichlet_prior([], size=4)
        self.assertEqual(len(empty.alphas), 4)
        with self.assertRaises(ValueError):
            bayes.fit_dirichlet_prior([])


if __name__ == "__main__":
    unittest.main()
