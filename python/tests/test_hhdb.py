import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from poker_ecosystem import hhdb

from . import fixtures


class DatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = hhdb.connect(":memory:")
        self.result = hhdb.import_hands(self.conn, fixtures.all_hands())

    def tearDown(self) -> None:
        self.conn.close()

    def ids(self, **filters: object) -> list[str]:
        return [h["id"] for h in hhdb.query_hands(self.conn, **filters)]

    def test_import_is_idempotent_and_atomic(self) -> None:
        self.assertEqual((self.result.inserted, self.result.skipped, self.result.failed), (9, 0, 0))
        again = hhdb.import_hands(self.conn, fixtures.all_hands())
        self.assertEqual((again.inserted, again.skipped), (0, 9))
        broken = fixtures.by_id("HU-1")
        del broken["id"]
        bad = hhdb.import_hands(self.conn, [broken])
        self.assertEqual(bad.failed, 1)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM hands").fetchone()[0], 9)
        counts = {t: self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("players", "boards")}
        self.assertEqual(counts, {"players": 6, "boards": 8})
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM actions").fetchone()[0], 81)

    def test_filters(self) -> None:
        self.assertEqual(self.ids(player="Alice", position="BTN"), ["HU-1", "6M-3BET", "6M-BARREL", "3H-SD", "STRADDLE-1"])
        self.assertEqual(self.ids(game="PLO"), ["BOMB-1"])
        self.assertEqual(self.ids(limit_type="PL"), ["BOMB-1"])
        self.assertEqual(self.ids(min_pot_bb=30), ["6M-XR", "3H-SD", "ALLIN-1"])
        self.assertEqual(self.ids(max_pot_bb=10), ["HU-1", "6M-3BET", "STRADDLE-1", "HU-DONK"])
        self.assertEqual(self.ids(street="turn"), ["6M-BARREL", "6M-XR", "3H-SD", "BOMB-1", "STRADDLE-1", "ALLIN-1"])
        self.assertEqual(self.ids(street="river"), ["3H-SD", "BOMB-1", "ALLIN-1"])
        self.assertEqual(self.ids(player="Alice", street="river"), ["3H-SD", "BOMB-1", "ALLIN-1"])
        self.assertEqual(self.ids(player="Frank", street="flop"), ["6M-BARREL", "BOMB-1", "STRADDLE-1"])
        self.assertEqual(self.ids(street="showdown"), ["3H-SD", "ALLIN-1"])
        self.assertEqual(self.ids(action="flop:raise"), ["6M-XR", "HU-DONK"])
        self.assertEqual(self.ids(player="Alice", action="flop:raise"), ["6M-XR"])
        self.assertEqual(self.ids(showdown=True), ["3H-SD", "ALLIN-1"])
        self.assertEqual(self.ids(allin=True), ["ALLIN-1"])
        self.assertEqual(self.ids(texture="paired"), ["BOMB-1"])
        self.assertEqual(self.ids(texture="monotone"), ["STRADDLE-1"])
        self.assertEqual(self.ids(texture="two-tone"), ["6M-XR", "HU-DONK"])
        self.assertEqual(self.ids(texture="rainbow"), ["HU-1", "6M-BARREL", "3H-SD", "BOMB-1", "ALLIN-1"])
        self.assertEqual(self.ids(synthetic=True), ["HU-1", "6M-BARREL"])
        self.assertEqual(len(self.ids(synthetic=False)), 7)
        self.assertEqual(self.ids(game="NLH", showdown=True, min_pot_bb=50), ["ALLIN-1"])
        self.assertEqual(self.ids(limit=2, offset=1), ["6M-3BET", "6M-BARREL"])
        with self.assertRaises(ValueError):
            self.ids(position="BTN")
        with self.assertRaises(ValueError):
            self.ids(action="flop-raise")
        with self.assertRaises(TypeError):
            self.ids(colour="red")

    def test_summary_fields(self) -> None:
        (hand,) = hhdb.query_hands(self.conn, player="Alice", allin=True)
        self.assertEqual(hand["position"], "UTG")
        self.assertEqual(hand["net_bb"], -50.0)
        self.assertEqual(hand["hole"], "Ah Ad")
        self.assertEqual(hand["winners"], ["Dave"])
        self.assertEqual(hand["board"], "Kc 7d 2s 9h 3c")
        self.assertEqual((hand["stakes"], hand["street"], hand["pot_bb"]), ("0.5/1", "river", 101.5))

    def test_aggregates(self) -> None:
        pooled = hhdb.player_stats(self.conn, player="Alice", stats=["vpip", "rfi"])
        self.assertEqual([(r["stat"], r["position"], r["n"], r["k"]) for r in pooled], [("vpip", "ALL", 7, 7), ("rfi", "ALL", 3, 3)])
        by_pos = hhdb.player_stats(self.conn, player="Alice", stats=["rfi"], by_position=True)
        self.assertEqual([(r["position"], r["n"], r["k"]) for r in by_pos], [("UTG", 1, 1), ("BTN", 2, 2)])
        players = hhdb.list_players(self.conn)
        self.assertEqual(players[0]["player"], "Alice")
        self.assertEqual((players[0]["hands"], players[0]["vpip"]), (8, 1.0))
        self.assertEqual([p["player"] for p in hhdb.list_players(self.conn, min_hands=7)], ["Alice", "Bob", "Carol", "Dave"])
        table = hhdb.stats_table(self.conn, player="Alice")
        self.assertEqual(table.get("Alice", "vpip"), (7, 7))
        self.assertEqual(table.n_hands, 8)
        self.assertEqual(hhdb.rebuild_stats_cache(self.conn), 9)
        self.assertEqual(hhdb.player_stats(self.conn, player="Alice", stats=["vpip"])[0]["n"], 7)


REVIEW = {
    "id": "3H-SD", "board": "Kh8d3c2s2h", "pot_bb": 20.5, "exploitability_pct": 0.3,
    "decisions": [
        {"index": 9, "seat": 1, "name": "Alice", "hand": "AhKd", "actions": ["Check", "Bet 33%", "Bet 75%"], "taken": 0, "best": 2,
         "ev_loss_bb": 1.25, "frequency": 0.04, "size_matched": True},
        {"index": 10, "seat": 2, "name": "Bob", "hand": "8c8h", "actions": ["Check", "Bet 33%"], "taken": 1, "best": 1,
         "ev_loss_bb": 0.0, "frequency": 0.9, "size_matched": True},
        {"index": 11, "seat": 3, "name": "Carol", "hand": None, "actions": ["Fold", "Call"], "taken": 1, "best": None,
         "ev_loss_bb": None, "frequency": None, "size_matched": True},
    ],
}


class ReviewTests(unittest.TestCase):
    def test_import_and_rank_river_reviews(self) -> None:
        conn = hhdb.connect(":memory:")
        hhdb.import_hands(conn, fixtures.all_hands())
        # the decision with unknown cards is skipped; re-importing replaces rows
        self.assertEqual(hhdb.import_reviews(conn, [REVIEW]), 2)
        self.assertEqual(hhdb.import_reviews(conn, [REVIEW]), 2)
        rows = hhdb.costliest_decisions(conn)
        self.assertEqual([(r["player"], r["action"], r["best"], r["ev_loss_bb"]) for r in rows], [("Alice", "Check", "Bet 75%", 1.25), ("Bob", "Bet 33%", "Bet 33%", 0.0)])
        self.assertEqual([r["id"] for r in hhdb.costliest_decisions(conn, player="Bob")], ["3H-SD"])
        linked = conn.execute("SELECT COUNT(*) FROM river_reviews WHERE hand_pk IS NOT NULL").fetchone()[0]
        self.assertEqual(linked, 2)


class CommandLineTests(unittest.TestCase):
    def run_cli(self, *args: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = hhdb.main(list(args))
        return code, out.getvalue(), err.getvalue()

    def test_cli_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "hands.sqlite")
            src = Path(tmp) / "hands.jsonl"
            src.write_text("\n".join(json.dumps(h) for h in fixtures.all_hands()), encoding="utf-8")
            code, out, _ = self.run_cli("import", db, str(src))
            self.assertEqual((code, out.strip()), (0, "inserted 9, skipped 0 (already present), failed 0"))
            code, out, _ = self.run_cli("import", db, str(src))
            self.assertIn("inserted 0, skipped 9", out)
            code, out, _ = self.run_cli("query", db, "--player", "Alice", "--street", "river", "--json")
            self.assertEqual([h["id"] for h in json.loads(out)], ["3H-SD", "BOMB-1", "ALLIN-1"])
            code, out, _ = self.run_cli("query", db, "--texture", "monotone", "--no-showdown")
            self.assertIn("STRADDLE-1", out)
            self.assertEqual(len(out.strip().splitlines()), 3)  # header, rule, one hand
            code, out, _ = self.run_cli("stats", db, "--player", "Alice", "--stat", "vpip", "--json")
            self.assertEqual(json.loads(out), [{"player": "Alice", "stat": "vpip", "position": "ALL", "n": 7, "k": 7, "freq": 1.0}])
            code, out, _ = self.run_cli("players", db, "--min-hands", "8")
            self.assertEqual(code, 0)
            self.assertIn("Alice", out)
            reviews = Path(tmp) / "reviews.json"
            reviews.write_text(json.dumps({"reviews": [REVIEW]}), encoding="utf-8")
            code, out, _ = self.run_cli("reviews", db, str(reviews))
            self.assertEqual((code, out.strip()), (0, "imported 2 reviewed river decisions"))
            code, out, _ = self.run_cli("costly", db, "--limit", "1", "--json")
            self.assertEqual([(r["player"], r["ev_loss_bb"]) for r in json.loads(out)], [("Alice", 1.25)])
            code, _, err = self.run_cli("query", db, "--position", "BTN")
            self.assertEqual(code, 2)
            self.assertIn("needs a player", err)


if __name__ == "__main__":
    unittest.main()
