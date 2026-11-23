import contextlib
import io
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from poker_ecosystem import evaluate, hh
from poker_ecosystem.stats import StatsTable

from . import toysim

PLAYERS = toysim.make_players(per_style=3, seed=7)
HANDS = toysim.simulate(PLAYERS, 300, seed=11)
META = {p["name"]: p for p in PLAYERS}
STATS = ("vpip", "pfr", "rfi", "three_bet", "cold_call", "cbet_flop", "fold_vs_bet_flop", "bet_unopened_flop")


class ToySimulatorTests(unittest.TestCase):
    def test_replay_matches_independent_accounting(self) -> None:
        for hand in HANDS:
            rp = hh.replay(hand)
            self.assertEqual(rp.final_pot, hand["total_pot"], hand["id"])
            self.assertEqual(sum(rp.returned.values()), sum(u["amount"] for u in hand["uncalled"]), hand["id"])

    def test_truth_matches_the_simulator_profile(self) -> None:
        table = StatsTable.from_hands(HANDS)
        checked = 0
        for p in PLAYERS:
            for position, factor in toysim.OPEN_FACTOR.items():
                truth = table.truth(p["name"], "rfi", position)
                if truth is None:
                    continue
                self.assertAlmostEqual(truth[0], min(0.9, p["profile"]["open"] * factor), places=4)
                checked += 1
            cold = table.truth(p["name"], "cold_call")
            want = min(p["profile"]["call"], 1 - p["profile"]["threebet"])
            self.assertAlmostEqual(cold[0], want, places=4)
        self.assertGreater(checked, 40)


class EvaluateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = evaluate.EvalConfig(sizes=(25, 100, 300), seed=3, min_truth=10, stats=STATS, k_values=(1, 2, 3, 4, 5))
        cls.results = evaluate.evaluate_corpus(HANDS, META, cls.config)

    def test_corpus_summary(self) -> None:
        r = self.results
        self.assertTrue(r["synthetic_corpus"])
        self.assertEqual(r["corpus"]["hands"], 600)
        self.assertEqual(r["corpus"]["players"], 12)
        self.assertEqual(r["corpus"]["styles"], {"LAG": 3, "TAG": 3, "nit": 3, "station": 3})
        self.assertEqual([s["hands"] for s in r["sizes"]], [25, 100, 300])
        self.assertEqual(r["config"]["stats"], list(STATS))

    def test_error_falls_with_sample_size(self) -> None:
        sizes = self.results["sizes"]
        small, large = sizes[0]["pooled"], sizes[-1]["pooled"]
        self.assertEqual(sizes[-1]["players"], 12)
        self.assertLess(large["posterior"]["mae"], small["posterior"]["mae"])
        self.assertLess(sizes[-1]["by_position"]["posterior"]["mae"], sizes[0]["by_position"]["posterior"]["mae"])
        self.assertGreater(small["shrinkage_gain_mae"], 0.0)
        for entry in sizes:
            for scope in ("pooled", "by_position"):
                self.assertGreaterEqual(entry[scope]["coverage"], 0.6)
                self.assertLessEqual(entry[scope]["coverage"], 1.0)
            for key in ("kmeans_ari", "gmm_ari"):
                self.assertGreaterEqual(entry["clusters"][key], -1.0)
                self.assertLessEqual(entry["clusters"][key], 1.0)
        self.assertGreater(sizes[-1]["clusters"]["kmeans_ari"], 0.3)
        self.assertEqual(set(sizes[-1]["by_stat"]), set(STATS))

    def test_outputs_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            written = evaluate.write_outputs(
                self.results,
                results_path=root / "results" / "opponent_model.json",
                figures_dir=root / "figures",
                report_path=root / "reports" / "opponent-model.md",
            )
            data = json.loads(written["results"].read_text(encoding="utf-8"))
            self.assertEqual(data["schema"], "pe-opponent-eval/1")
            figures = {k: p for k, p in written.items() if k.startswith("figure:")}
            expected = {"error", "coverage", "ari", "styles", "error_hist", "by_stat_preflop", "by_stat_flop"}
            self.assertEqual(set(figures), {f"figure:{k}" for k in expected})
            for path in figures.values():
                ET.fromstring(path.read_text(encoding="utf-8"))
            report = written["report"].read_text(encoding="utf-8")
            self.assertIn("synthetic corpus", report)
            self.assertIn("](../figures/opponent_model_error.svg)", report)
            self.assertIn("Approximations in the model", report)

    def test_command_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hands = root / "hands.jsonl"
            hands.write_text("\n".join(json.dumps(h) for h in HANDS[:300]), encoding="utf-8")
            players = root / "players.json"
            players.write_text(json.dumps(PLAYERS), encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = evaluate.main([
                    "--hands", str(hands), "--players", str(players), "--sizes", "20,150", "--stats", "vpip,pfr",
                    "--min-truth", "5", "--results", str(root / "r.json"), "--figures", str(root / "fig"),
                    "--report", str(root / "report.md"),
                ])
            self.assertEqual(code, 0)
            self.assertIn("N=  150", out.getvalue())
            self.assertTrue((root / "report.md").is_file())
            outputs = ["--results", str(root / "x.json"), "--figures", str(root / "xfig"), "--report", str(root / "x.md")]
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(evaluate.main(["--hands", str(hands), "--players", str(players), "--stats", "wtsd", *outputs]), 2)
            self.assertFalse((root / "x.json").exists())

    def test_helpers(self) -> None:
        with self.assertRaises(ValueError):
            evaluate.evaluable_stats(["wtsd"])
        self.assertIn("vpip", evaluate.evaluable_stats())
        self.assertNotIn("af_flop", evaluate.evaluable_stats())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "players.json"
            path.write_text(json.dumps({"bot1": {"style": "TAG"}}), encoding="utf-8")
            self.assertEqual(evaluate.load_players_meta(path), {"bot1": {"style": "TAG", "name": "bot1"}})


if __name__ == "__main__":
    unittest.main()
