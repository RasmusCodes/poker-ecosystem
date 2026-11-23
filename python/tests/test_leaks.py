import json
import random
import tempfile
import unittest
from pathlib import Path

from poker_ecosystem import bayes, leaks
from poker_ecosystem.opponent import OpponentModel
from poker_ecosystem.stats import StatsTable

from . import fixtures
from .test_bayes import simpson


def vpip_population() -> OpponentModel:
    rng = random.Random(8)
    table = StatsTable()
    for i in range(30):
        name = f"reg{i}"
        n = 300
        table.add_counts(name, "vpip", "BTN", n, sum(rng.random() < rng.uniform(0.22, 0.30) for _ in range(n)))
        table.hands[name] = n
    table.add_counts("fish", "vpip", "BTN", 500, 300)
    table.hands["fish"] = 500
    return OpponentModel(table)


class ExcessTests(unittest.TestCase):
    def test_expected_excess_matches_integration(self) -> None:
        a, b, lo, hi = 3.0, 7.0, 0.2, 0.4
        below = simpson(lambda t: (lo - t) * bayes.beta_pdf(t, a, b), 0.0, lo)
        above = simpson(lambda t: (t - hi) * bayes.beta_pdf(t, a, b), hi, 1.0)
        self.assertAlmostEqual(leaks.expected_excess(a, b, lo, hi), below + above, places=8)
        self.assertEqual(leaks.expected_excess(a, b, 0.0, 1.0), 0.0)


class LeakTests(unittest.TestCase):
    def test_population_reference_flags_the_outlier(self) -> None:
        model = vpip_population()
        fish = leaks.find_leaks(model, "fish", positions=False)
        self.assertEqual(len(fish), 1)
        leak = fish[0]
        self.assertEqual((leak.stat, leak.position, leak.rank, leak.direction), ("vpip", "ALL", 1, "high"))
        self.assertGreater(leak.p_outside, 0.99)
        self.assertEqual(leak.band_source, "population")
        self.assertGreater(leak.ev_impact_heuristic, 0.0)
        regular = leaks.find_leaks(model, "reg0", positions=False)[0]
        self.assertLess(regular.p_outside, fish[0].p_outside)
        self.assertLess(regular.ev_impact_heuristic, leak.ev_impact_heuristic)

    def test_baseline_bands(self) -> None:
        model = vpip_population()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baseline.json"
            path.write_text(json.dumps({"vpip": {"ALL": [0.2, 0.3], "BTN": 0.5}, "pfr": 0.2}), encoding="utf-8")
            baseline = leaks.load_baseline(path)
        found = {(l.stat, l.position): l for l in leaks.find_leaks(model, "fish", baseline=baseline)}
        self.assertEqual(found[("vpip", "ALL")].band, (0.2, 0.3))
        for got, want in zip(found[("vpip", "BTN")].band, (0.47, 0.53)):
            self.assertAlmostEqual(got, want, places=12)
        self.assertEqual(found[("vpip", "BTN")].band_source, "baseline")
        self.assertNotIn(("pfr", "ALL"), found)  # no pfr data at all
        self.assertGreater(found[("vpip", "ALL")].p_high, 0.99)
        with self.assertRaises(ValueError):
            leaks.reference_band(model, "vpip", None, {"vpip": [0.5, 0.2]})

    def test_allin_luck(self) -> None:
        hands = fixtures.all_hands()
        alice = leaks.allin_luck(hands, "Alice")
        self.assertEqual(alice.hands, 1)
        self.assertAlmostEqual(alice.actual_bb, -50.0)
        self.assertAlmostEqual(alice.ev_bb, 33.23)
        self.assertAlmostEqual(alice.luck_bb, -83.23)
        table = StatsTable.from_hands(hands)
        self.assertAlmostEqual(leaks.allin_luck(table, "Dave").luck_bb, 83.23)
        self.assertEqual(leaks.allin_luck(table, "Bob").hands, 0)

    def test_report_is_labelled_heuristic(self) -> None:
        model = OpponentModel.from_hands(fixtures.all_hands())
        report = leaks.leak_report(model, "Alice", top=5)
        text = json.dumps(report)
        self.assertIn("heuristic", text)
        self.assertLessEqual(len(report["leaks"]), 5)
        self.assertEqual([l["rank"] for l in report["leaks"]], list(range(1, len(report["leaks"]) + 1)))
        self.assertAlmostEqual(report["allin_luck"]["luck_bb"], -83.23)


if __name__ == "__main__":
    unittest.main()
