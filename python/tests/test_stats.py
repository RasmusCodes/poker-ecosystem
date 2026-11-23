import unittest
from pathlib import Path

from poker_ecosystem import stats
from poker_ecosystem.stats import ALL, StatsTable, hand_observations, size_bin

from . import fixtures


def observed(hand_id: str) -> dict[tuple[str, str], int]:
    """(player, stat) -> success (0/1) for a single fixture; fails on duplicates."""
    out: dict[tuple[str, str], int] = {}
    for o in hand_observations(fixtures.by_id(hand_id)).observations:
        key = (o.player, o.stat)
        if key in out and not o.stat.startswith("af_"):
            raise AssertionError(f"duplicate opportunity {key}")
        out[key] = out.get(key, 0) + int(o.success) if o.stat.startswith("af_") else int(o.success)
    return out


def expect(player: str, spec: str) -> dict[tuple[str, str], int]:
    """'vpip+ pfr-' -> {(player, 'vpip'): 1, (player, 'pfr'): 0}."""
    return {(player, item[:-1]): int(item[-1] == "+") for item in spec.split()}


class ExactCountTests(unittest.TestCase):
    def test_heads_up(self) -> None:
        want = {
            **expect("Alice", "vpip+ pfr+ rfi+ steal+ limp- cbet_flop+ bet_unopened_flop+ af_flop+ wtsd-"),
            **expect("Bob", "vpip+ pfr- cold_call+ three_bet- fold_to_steal- bet_unopened_flop- donk_flop- "
                            "fold_vs_bet_flop+ fold_vs_cbet_flop+ call_vs_bet_flop- call_vs_cbet_flop- "
                            "raise_vs_bet_flop- raise_vs_cbet_flop- check_raise_flop- af_flop- wtsd-"),
        }
        self.assertEqual(observed("HU-1"), want)

    def test_three_bet_pot(self) -> None:
        want = {
            **expect("Carol", "vpip- pfr- rfi- limp-"),
            **expect("Dave", "vpip+ pfr+ rfi+ limp- fold_to_three_bet+ four_bet-"),
            **expect("Erin", "vpip- pfr- cold_call- three_bet-"),
            **expect("Alice", "vpip+ pfr+ cold_call- three_bet+"),
            **expect("Bob", "vpip- pfr- four_bet-"),
            **expect("Frank", "vpip- pfr- four_bet-"),
        }
        self.assertEqual(observed("6M-3BET"), want)

    def test_limp_iso_and_barrel(self) -> None:
        want = {
            **expect("Dave", "vpip- pfr- rfi- limp-"),
            **expect("Erin", "vpip- pfr- rfi- limp-"),
            **expect("Frank", "vpip+ pfr- rfi- steal- limp+ three_bet- bet_unopened_flop- donk_flop- "
                              "fold_vs_bet_flop- call_vs_bet_flop+ raise_vs_bet_flop- fold_vs_cbet_flop- "
                              "call_vs_cbet_flop+ raise_vs_cbet_flop- check_raise_flop- af_flop- "
                              "bet_unopened_turn- donk_turn- fold_vs_bet_turn+ call_vs_bet_turn- raise_vs_bet_turn- "
                              "fold_vs_cbet_turn+ call_vs_cbet_turn- raise_vs_cbet_turn- check_raise_turn- af_turn- wtsd-"),
            **expect("Alice", "vpip+ pfr+ limp- cbet_flop+ bet_unopened_flop+ af_flop+ cbet_turn+ "
                              "bet_unopened_turn+ af_turn+ wtsd-"),
            **expect("Bob", "vpip- pfr- cold_call- three_bet-"),
            **expect("Carol", "vpip- pfr- cold_call- three_bet-"),
        }
        self.assertEqual(observed("6M-BARREL"), want)

    def test_check_raise(self) -> None:
        got = observed("6M-XR")
        want = {
            **expect("Bob", "vpip+ pfr+ rfi+ limp- cbet_flop+ bet_unopened_flop+ fold_vs_bet_turn+ "
                            "call_vs_bet_turn- raise_vs_bet_turn- af_turn- wtsd-"),
            **expect("Alice", "vpip+ pfr- cold_call+ three_bet- bet_unopened_flop- donk_flop- fold_vs_bet_flop- "
                              "call_vs_bet_flop- raise_vs_bet_flop+ fold_vs_cbet_flop- call_vs_cbet_flop- "
                              "raise_vs_cbet_flop+ check_raise_flop+ af_flop+ bet_unopened_turn+ "
                              "xr_flop_bet_turn+ af_turn+ wtsd-"),
            **{k: v for name in ("Erin", "Frank", "Carol", "Dave") for k, v in expect(name, "vpip- pfr- cold_call- three_bet-").items()},
        }
        got_af = got.pop(("Bob", "af_flop"))
        self.assertEqual(got, want)
        self.assertEqual(got_af, 1)  # Bob: bet (aggressive) + call (passive) on the flop
        table = StatsTable.from_hands([fixtures.by_id("6M-XR")])
        self.assertEqual(table.get("Bob", "af_flop"), (2, 1))

    def test_river_showdown(self) -> None:
        want = {
            **expect("Alice", "vpip+ pfr+ rfi+ steal+ limp- cbet_flop- bet_unopened_flop- fold_vs_bet_turn- "
                              "call_vs_bet_turn+ raise_vs_bet_turn- af_turn- fold_vs_bet_river- call_vs_bet_river+ "
                              "raise_vs_bet_river- af_river- wtsd+ wsd+"),
            **expect("Bob", "vpip- pfr- cold_call- three_bet- fold_to_steal+"),
            **expect("Carol", "vpip+ pfr- cold_call+ three_bet- fold_to_steal- bet_unopened_flop- donk_flop- "
                              "bet_unopened_turn+ af_turn+ bet_unopened_river+ af_river+ wtsd+ wsd- river_bluff+"),
        }
        self.assertEqual(observed("3H-SD"), want)

    def test_bomb_pot_has_no_preflop_or_cbet_stats(self) -> None:
        want = {
            **expect("Frank", "bet_unopened_flop- fold_vs_bet_flop+ call_vs_bet_flop- raise_vs_bet_flop- "
                              "check_raise_flop- af_flop- wtsd-"),
            **expect("Alice", "bet_unopened_flop+ af_flop+ bet_unopened_turn- bet_unopened_river- fold_vs_bet_river+ "
                              "call_vs_bet_river- raise_vs_bet_river- check_raise_river- af_river- wtsd-"),
            **expect("Dave", "fold_vs_bet_flop+ call_vs_bet_flop- raise_vs_bet_flop- af_flop- wtsd-"),
            **expect("Erin", "fold_vs_bet_flop- call_vs_bet_flop+ raise_vs_bet_flop- af_flop- bet_unopened_turn- "
                             "bet_unopened_river+ af_river+ wtsd-"),
        }
        self.assertEqual(observed("BOMB-1"), want)

    def test_straddle(self) -> None:
        want = {
            **expect("Erin", "vpip- pfr- rfi- limp-"),
            **expect("Frank", "vpip+ pfr- rfi- steal- limp+ bet_unopened_flop- fold_vs_bet_turn+ call_vs_bet_turn- "
                              "raise_vs_bet_turn- af_turn- wtsd-"),
            **expect("Alice", "vpip+ pfr- limp+ bet_unopened_flop- fold_vs_bet_turn+ call_vs_bet_turn- "
                              "raise_vs_bet_turn- af_turn- wtsd-"),
            **expect("Bob", "vpip- pfr- limp-"),
            **expect("Carol", "vpip- pfr- limp-"),
            **expect("Dave", "vpip- pfr- bet_unopened_flop- bet_unopened_turn+ af_turn+ wtsd-"),
        }
        self.assertEqual(observed("STRADDLE-1"), want)

    def test_allin(self) -> None:
        want = {
            **expect("Alice", "vpip+ pfr+ rfi+ limp- fold_to_three_bet- four_bet+ wtsd+ wsd-"),
            **expect("Bob", "vpip- pfr- cold_call- three_bet-"),
            **expect("Carol", "vpip- pfr- cold_call- three_bet-"),
            **expect("Dave", "vpip+ pfr+ cold_call- three_bet+ wtsd+ wsd+"),
            **expect("Erin", "vpip- pfr- four_bet-"),
            **expect("Frank", "vpip- pfr- four_bet-"),
        }
        self.assertEqual(observed("ALLIN-1"), want)

    def test_donk(self) -> None:
        got = observed("HU-DONK")
        self.assertEqual(got.pop(("Dave", "af_flop")), 1)
        want = {
            **expect("Carol", "vpip+ pfr+ rfi+ steal+ limp- fold_vs_bet_flop- call_vs_bet_flop- raise_vs_bet_flop+ "
                              "af_flop+ wtsd-"),
            **expect("Dave", "vpip+ pfr- cold_call+ three_bet- fold_to_steal- bet_unopened_flop+ donk_flop+ wtsd-"),
        }
        self.assertEqual(got, want)


class TableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.table = StatsTable.from_hands(fixtures.all_hands())

    def test_pooled_and_position_counts(self) -> None:
        t = self.table
        self.assertEqual(t.get("Alice", "vpip"), (7, 7))
        self.assertEqual(t.get("Bob", "vpip"), (7, 2))
        self.assertEqual(t.get("Alice", "rfi"), (3, 3))
        self.assertEqual(t.get("Alice", "rfi", "BTN"), (2, 2))
        self.assertEqual(t.get("Alice", "rfi", "UTG"), (1, 1))
        self.assertEqual(t.get("Alice", "rfi", ALL), (3, 3))
        self.assertEqual(t.get("Alice", "limp"), (5, 1))  # HU-1, 6M-BARREL, 3H-SD, STRADDLE-1, ALLIN-1
        self.assertEqual(t.get("Dave", "donk_flop"), (1, 1))
        self.assertEqual(t.get("Carol", "fold_to_steal"), (1, 0))
        self.assertEqual(t.get("Nobody", "vpip"), (0, 0))
        self.assertIsNone(t.frequency("Nobody", "vpip"))
        self.assertEqual(t.hands["Alice"], 8)
        self.assertEqual(t.n_hands, 9)
        self.assertEqual(t.positions("rfi"), ["UTG", "HJ", "CO", "BTN"])

    def test_truth_values(self) -> None:
        t = self.table
        self.assertAlmostEqual(t.truth("Alice", "vpip")[0], 0.9)
        self.assertEqual(t.truth("Alice", "vpip")[1], 1)  # only HU-1 carries truth for Alice
        self.assertAlmostEqual(t.truth("Frank", "pfr", "CO")[0], 0.4)  # compensator 0.2 + 0.2
        self.assertAlmostEqual(t.truth("Frank", "vpip")[0], 0.7)
        self.assertAlmostEqual(t.truth("Bob", "check_raise_flop")[0], 0.1)
        self.assertAlmostEqual(t.truth("Frank", "raise_vs_cbet_turn")[0], 0.0)
        self.assertIsNone(t.truth("Alice", "wtsd"))

    def test_sizing_bins(self) -> None:
        t = self.table
        self.assertEqual(t.sizing_counts("Alice", "flop", "bet")["40-60"], 3)  # HU-1, 6M-BARREL, BOMB-1
        self.assertEqual(t.sizing_counts("Alice", "preflop", "raise"),
                         {"<40": 0, "40-60": 0, "60-85": 3, "85-115": 2, ">115": 0, "allin": 1})
        self.assertEqual(t.sizing_counts("Carol", "river", "bet")["85-115"], 1)
        self.assertEqual(t.sizing_counts("Alice", "flop", "raise")["40-60"], 1)

    def test_size_bin_edges_are_exact(self) -> None:
        self.assertEqual(size_bin(399, 1000), "<40")
        self.assertEqual(size_bin(400, 1000), "40-60")
        self.assertEqual(size_bin(600, 1000), "60-85")
        self.assertEqual(size_bin(849, 1000), "60-85")
        self.assertEqual(size_bin(850, 1000), "85-115")
        self.assertEqual(size_bin(1149, 1000), "85-115")
        self.assertEqual(size_bin(1150, 1000), ">115")
        self.assertEqual(size_bin(10, 1000, all_in=True), "allin")
        self.assertIsNone(size_bin(100, 0))

    def test_river_strength_bluffs_and_threshold(self) -> None:
        t = self.table
        self.assertEqual(t.river_strengths("Carol"), [0.2])
        self.assertEqual(t.get("Carol", "river_bluff"), (1, 1))
        self.assertIsNone(t.value_threshold("Carol"))
        strict = StatsTable.from_hands(fixtures.all_hands(), bluff_threshold=0.1)
        self.assertEqual(strict.get("Carol", "river_bluff"), (1, 0))
        self.assertEqual(strict.value_threshold("Carol"), 0.2)
        self.assertEqual(stats.empirical_quantile([0.9, 0.6, 0.7, 0.8], 0.25), 0.6)
        self.assertEqual(stats.empirical_quantile([0.9, 0.6, 0.7, 0.8], 0.5), 0.7)

    def test_transitions(self) -> None:
        t = self.table
        self.assertEqual(t.transitions("Alice", "line")[("flop:xr", "turn:b")], 1)
        self.assertEqual(t.transitions("Alice", "cbet")[("flop:bet", "turn:bet")], 1)
        self.assertEqual(t.transitions("Bob", "cbet")[("flop:bet", "turn:faced_bet")], 1)
        self.assertEqual(t.transitions("Erin", "line")[("turn:x", "river:b")], 1)

    def test_allin_and_net(self) -> None:
        t = self.table
        self.assertEqual(t.allin_records("Alice"), [(-50.0, 33.23)])
        self.assertEqual(t.allin_records("Dave"), [(51.5, -31.73)])
        # 6M-BARREL -1 (BB), 3H-SD -17, STRADDLE-1 -1 (BB), HU-DONK +5; folds elsewhere cost nothing.
        self.assertAlmostEqual(t.net_bb["Carol"], -14.0)

    def test_summary_is_json_ready(self) -> None:
        import json
        summary = self.table.summary("Alice")
        json.dumps(summary)
        self.assertEqual(summary["stats"]["vpip"][ALL], {"n": 7, "k": 7, "freq": 1.0})

    def test_add_counts(self) -> None:
        t = StatsTable()
        t.add_counts("P", "vpip", "BTN", 10, 4)
        t.add_counts("P", "vpip", "BB", 5, 1)
        self.assertEqual(t.get("P", "vpip"), (15, 5))
        self.assertEqual(t.players(), ["P"])
        with self.assertRaises(ValueError):
            t.add_counts("P", "vpip", ALL, 1, 1)


class DefinitionTests(unittest.TestCase):
    def test_every_stat_is_documented(self) -> None:
        doc = stats.__doc__ or ""
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
        for key in stats.STAT_KEYS:
            generic = key.rsplit("_", 1)[0] + "_S"
            self.assertTrue(f"`{key}`" in doc or f"`{generic}`" in doc or generic in doc, key)
            self.assertIn(f"`{key}`", readme, key)

    def test_truth_categories(self) -> None:
        self.assertEqual(stats.STAT_DEFS["vpip"].truth, "compensator")
        self.assertFalse(stats.STAT_DEFS["af_flop"].truth_evaluable)
        self.assertFalse(stats.STAT_DEFS["wtsd"].truth_evaluable)
        self.assertTrue(stats.STAT_DEFS["check_raise_river"].truth_evaluable)
        self.assertIn("| `rfi` |", stats.definitions_markdown())


if __name__ == "__main__":
    unittest.main()
