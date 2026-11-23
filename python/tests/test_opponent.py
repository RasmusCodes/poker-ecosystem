import json
import random
import tempfile
import unittest
from pathlib import Path

from poker_ecosystem import bayes, ranges
from poker_ecosystem.bayes import expit, logit
from poker_ecosystem.opponent import EXPLOIT_STATS, OpponentModel
from poker_ecosystem.stats import ALL, HandObservations, StatsTable

from . import fixtures

BASE_RFI = {"UTG": 0.15, "CO": 0.27, "BTN": 0.45}


def population_table(seed: int = 1, players: int = 40) -> tuple[StatsTable, dict[tuple[str, str], float]]:
    """Players with a logit-normal skill shift applied to position base rates."""
    rng = random.Random(seed)
    table = StatsTable()
    truth = {}
    for i in range(players):
        name = f"p{i:02d}"
        shift = rng.gauss(0.0, 0.5)
        for pos, base in BASE_RFI.items():
            p = expit(logit(base) + shift)
            n = rng.randint(20, 80)
            table.add_counts(name, "rfi", pos, n, sum(rng.random() < p for _ in range(n)))
            truth[(name, pos)] = p
    return table, truth


class HierarchyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.table, cls.truth = population_table()
        cls.table.add_counts("fresh", "rfi", "UTG", 0, 0)
        cls.model = OpponentModel(cls.table)

    def test_pooled_estimate_is_conjugate_update(self) -> None:
        prior = self.model.population_prior("rfi")
        self.assertEqual(prior.method, "newton+boundary-prior")
        n, k = self.table.get("p00", "rfi")
        est = self.model.estimate("p00", "rfi")
        self.assertAlmostEqual(est.alpha, prior.alpha + k)
        self.assertAlmostEqual(est.beta, prior.beta + n - k)
        self.assertEqual((est.n, est.k, est.position), (n, k, ALL))
        self.assertLess(est.lo, est.mean)
        self.assertLess(est.mean, est.hi)
        self.assertAlmostEqual(bayes.betainc(est.alpha, est.beta, est.lo), 0.05, places=9)

    def test_position_prior_mean_formula(self) -> None:
        player, position = "p07", "UTG"
        kappa = self.model.population_prior("rfi").concentration
        n_other = k_other = 0
        expected = 0.0
        for q in ("CO", "BTN"):
            n, k = self.table.get(player, "rfi", q)
            n_other += n
            k_other += k
            expected += n * self.model.population_prior("rfi", q).mean
        base = expected / n_other
        shrunk = (k_other + kappa * base) / (n_other + kappa)
        mu = self.model.population_prior("rfi", position).mean
        want = expit(logit(mu) + logit(shrunk) - logit(base))
        self.assertAlmostEqual(self.model.prior_mean_at(player, "rfi", position), want, places=12)

    def test_player_without_data_gets_population_position_prior(self) -> None:
        mu = self.model.population_prior("rfi", "UTG").mean
        self.assertAlmostEqual(self.model.prior_mean_at("fresh", "rfi", "UTG"), mu, places=12)
        est = self.model.estimate("fresh", "rfi", "UTG")
        self.assertAlmostEqual(est.mean, mu, places=12)
        self.assertEqual(est.n, 0)
        self.assertIsNone(est.raw)

    def test_shift_direction_and_residual_fit(self) -> None:
        loose = max(self.model.players(), key=lambda p: self.table.frequency(p, "rfi", "BTN") or 0)
        tight = min((p for p in self.model.players() if p != "fresh"), key=lambda p: self.table.frequency(p, "rfi", "BTN"))
        mu = self.model.population_prior("rfi", "UTG").mean
        self.assertGreater(self.model.prior_mean_at(loose, "rfi", "UTG"), mu)
        self.assertLess(self.model.prior_mean_at(tight, "rfi", "UTG"), mu)
        fit = self.model.residual_variance("rfi", "UTG")
        self.assertEqual(fit.method, "ml")
        self.assertGreaterEqual(fit.variance, 0.0)
        # Knowing the player's other positions makes the UTG prior tighter than the population's ...
        a, b = self.model.prior_for("p07", "rfi", "UTG")
        self.assertGreater(a + b, self.model.population_prior("rfi", "UTG").concentration)
        # ... and the uncertainty of the estimated shift keeps it finite even when the residual fit is 0.
        _, shift_var = self.model.shift("p07", "rfi", "UTG")
        self.assertGreater(shift_var, 0.0)
        self.assertLess(a + b, 1e4)

    def test_intervals_cover_most_true_values(self) -> None:
        hits = total = 0
        for (player, pos), p in self.truth.items():
            est = self.model.estimate(player, "rfi", pos)
            hits += est.contains(p)
            total += 1
        self.assertGreater(hits / total, 0.8)

    def test_shrinkage_strength_depends_on_sample_size(self) -> None:
        table = StatsTable()
        rng = random.Random(4)
        for i in range(30):
            n = 200
            table.add_counts(f"q{i}", "vpip", "BTN", n, sum(rng.random() < rng.uniform(0.2, 0.3) for _ in range(n)))
        table.add_counts("small", "vpip", "BTN", 4, 4)
        table.add_counts("large", "vpip", "BTN", 2000, 1800)
        model = OpponentModel(table)
        prior = model.population_prior("vpip").mean
        small = model.estimate("small", "vpip").mean
        large = model.estimate("large", "vpip").mean
        self.assertTrue(prior < small < 1.0)
        self.assertLess(abs(small - prior), 0.2)  # 4 hands barely move a tight population prior
        self.assertLess(abs(large - 0.9), abs(small - 1.0))  # 2000 hands dominate the prior


class FixtureModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = OpponentModel.from_hands(fixtures.all_hands())

    def test_exploit_profile(self) -> None:
        prof = self.model.exploit_profile("Carol")
        for key, stat in EXPLOIT_STATS.items():
            entry = prof[key]
            self.assertEqual(entry["stat"], stat)
            self.assertTrue(0.0 <= entry["lo"] <= entry["hi"] <= 1.0)
            self.assertTrue(0.0 < entry["mean"] < 1.0)
        self.assertEqual((prof["bluff_share"]["n"], prof["bluff_share"]["k"]), (1, 1))
        self.assertEqual((prof["bet_freq"]["n"], prof["bet_freq"]["k"]), (1, 1))
        self.assertEqual(prof["fold_to_bet"]["n"], 0)
        self.assertIn("value_threshold", prof)
        self.assertTrue(prof["approximations"])
        json.dumps(prof)

    def test_profile_is_json_ready(self) -> None:
        profile = self.model.profile("Alice")
        text = json.dumps(profile)
        self.assertIn("exploit_profile", profile)
        self.assertIn("rfi", profile["ranges"])
        self.assertGreater(len(text), 1000)
        self.assertEqual(profile["hands"], 8)

    def test_sizing_posterior(self) -> None:
        sizing = self.model.sizing("Alice", "flop", "bet")
        self.assertAlmostEqual(sum(sizing.mean), 1.0)
        self.assertEqual(sizing.counts[sizing.bins.index("40-60")], 3)
        self.assertEqual(max(range(len(sizing.mean)), key=lambda j: sizing.mean[j]), sizing.bins.index("40-60"))
        alphas = [m * sizing.prior_concentration + c for m, c in zip(sizing.prior_mean, sizing.counts)]
        total = sum(alphas)
        for (lo, hi), a in zip(sizing.intervals, alphas):
            # Equal-tailed marginal intervals. With a tiny alpha (or beta) the mean may lie outside
            # them and a quantile can sit closer to 0 (or 1) than a double can express.
            self.assertTrue(0.0 <= lo <= hi <= 1.0)
            if hi < 1.0:
                self.assertAlmostEqual(bayes.betainc(a, total - a, hi), 0.95, places=8)
            if lo > 0.0:
                self.assertAlmostEqual(bayes.betainc(a, total - a, lo), 0.05, places=8)
        self.assertGreaterEqual(self.model.sizing_prior("flop", "bet").concentration, 1.0)

    def test_preflop_range(self) -> None:
        rng = self.model.preflop_range("Alice", "BTN")
        self.assertEqual(rng.classes[0], "AA")
        self.assertEqual(rng.source, "chen-formula")
        self.assertLessEqual(abs(rng.combos / ranges.TOTAL_COMBOS - rng.fraction), 6 / ranges.TOTAL_COMBOS)
        self.assertLessEqual(rng.classes_at_lo, len(rng.classes))
        self.assertGreaterEqual(rng.classes_at_hi, len(rng.classes))

    def test_tendencies(self) -> None:
        tend = self.model.tendencies("Alice")
        self.assertEqual(tend["bet_turn_after_flop_check_raise"]["n"], 1)
        self.assertEqual(tend["barrel_turn_after_flop_cbet"]["k"], 1)
        rows = {(r["from"], r["to"]): r for r in tend["line_transitions"]}
        self.assertEqual(rows[("flop:xr", "turn:b")]["share_of_from"], 1.0)

    def test_value_threshold_interval(self) -> None:
        table = StatsTable()
        values = [0.55 + 0.009 * i for i in range(50)]
        table.add(HandObservations(hand_id="x", players={"P": "BTN"}, river_bets=[("P", v) for v in values]))
        table.add(HandObservations(hand_id="y", players={"Q": "BB"}, river_bets=[("Q", 0.3), ("Q", 0.9)]))
        model = OpponentModel(table)
        est = model.value_threshold("P", 0.1)
        self.assertEqual(est.n, 50)
        self.assertAlmostEqual(est.value, values[4])
        self.assertLessEqual(est.lo, est.value)
        self.assertLessEqual(est.value, est.hi)
        lo_i, hi_i = bayes.order_statistic_interval(50, 0.1, 0.9)
        self.assertAlmostEqual(est.lo, values[lo_i - 1])
        self.assertAlmostEqual(est.hi, values[hi_i - 1])
        small = model.value_threshold("Q", 0.1)
        self.assertEqual((small.n, small.value, small.lo, small.hi), (1, 0.9, 0.5, 1.0))


class RangeTests(unittest.TestCase):
    def test_classes_and_combos(self) -> None:
        classes = ranges.hand_classes()
        self.assertEqual(len(classes), 169)
        self.assertEqual(ranges.combos_of(classes), 1326)
        self.assertEqual(ranges.normalise_class("kas"), "AKs")
        self.assertEqual(ranges.class_combos("T9o"), 12)
        for bad in ("AK", "AAs", "XYz", "A"):
            with self.assertRaises(ValueError):
                ranges.normalise_class(bad)

    def test_chen_ordering(self) -> None:
        self.assertEqual(ranges.chen_score("AA"), 20)
        self.assertEqual(ranges.chen_score("22"), 5)
        self.assertEqual(ranges.chen_score("AKs"), 12)
        self.assertEqual(ranges.chen_score("T9s"), 8)
        self.assertEqual(ranges.chen_points("72o"), -1)
        order = ranges.chen_ranking()
        self.assertEqual(order[:4], ["AA", "KK", "QQ", "JJ"])
        self.assertEqual(order[4], "AKs")
        self.assertEqual(order[-1], "72o")
        self.assertEqual(len(set(order)), 169)

    def test_top_fraction(self) -> None:
        order = ranges.chen_ranking()
        self.assertEqual(ranges.top_fraction(order, 0.0), [])
        self.assertEqual(len(ranges.top_fraction(order, 1.0)), 169)
        top = ranges.top_fraction(order, 0.1)
        self.assertLessEqual(abs(ranges.range_fraction(top) - 0.1), 6 / 1326)

    def test_load_ranking_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preflop_ranking.json"
            reverse = list(reversed(ranges.chen_ranking()))
            path.write_text(json.dumps(reverse), encoding="utf-8")
            loaded, source = ranges.load_ranking(path)
            self.assertEqual(loaded, reverse)
            self.assertEqual(source, str(path))
            path.write_text(json.dumps(reverse[:-1]), encoding="utf-8")
            with self.assertRaises(ValueError):
                ranges.load_ranking(path)
            missing, source = ranges.load_ranking(Path(tmp) / "absent.json")
            self.assertEqual(source, "chen-formula")
            self.assertEqual(len(missing), 169)


if __name__ == "__main__":
    unittest.main()
