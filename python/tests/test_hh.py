import json
import tempfile
import unittest
from pathlib import Path

from poker_ecosystem import hh

from . import fixtures


class LoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_json_array_and_jsonl_round_trip(self) -> None:
        hands = fixtures.all_hands()
        (self.dir / "a.json").write_text(json.dumps(hands), encoding="utf-8")
        (self.dir / "b.jsonl").write_text("\n".join(json.dumps(h) for h in hands) + "\n\n", encoding="utf-8")
        from_json = hh.load_hands(self.dir / "a.json")
        from_jsonl = list(hh.iter_hands(self.dir / "b.jsonl"))
        self.assertEqual([h["id"] for h in from_json], [h["id"] for h in hands])
        self.assertEqual(from_json, from_jsonl)
        self.assertEqual(len(list(hh.iter_many([self.dir / "a.json", self.dir / "b.jsonl"]))), 2 * len(hands))

    def test_single_object_file_and_bom(self) -> None:
        path = self.dir / "one.json"
        path.write_text(json.dumps(fixtures.by_id("HU-1")), encoding="utf-8-sig")
        self.assertEqual([h["id"] for h in hh.load_hands(path)], ["HU-1"])

    def test_bad_input_is_reported(self) -> None:
        path = self.dir / "bad.jsonl"
        path.write_text('{"schema": "pe-hh/1", "players": []}\n{not json}\n', encoding="utf-8")
        with self.assertRaises(hh.HandError) as ctx:
            list(hh.iter_hands(path))
        self.assertIn(":2:", str(ctx.exception))
        with self.assertRaises(hh.HandError):
            hh.check_hand({"schema": "pe-hh/2", "players": []})
        with self.assertRaises(hh.HandError):
            hh.check_hand({"schema": "pe-hh/1"})
        hh.check_hand({"schema": "pe-hh/1.1", "players": []})


class LookupTests(unittest.TestCase):
    def test_positions_agree_with_schema_table(self) -> None:
        for hand in fixtures.all_hands():
            seats = [p["seat"] for p in hh.dealt_players(hand)]
            derived = hh.derive_positions(seats, hand["button"])
            given = {p["seat"]: p["position"] for p in hh.dealt_players(hand)}
            self.assertEqual(derived, given, hand["id"])

    def test_derive_positions_ten_handed_and_fallback(self) -> None:
        labels = hh.derive_positions(list(range(1, 11)), 10)
        self.assertEqual(
            [labels[s] for s in range(1, 11)],
            ["SB", "BB", "UTG", "UTG1", "UTG2", "UTG3", "LJ", "HJ", "CO", "BTN"],
        )
        hand = fixtures.by_id("6M-3BET")
        for p in hand["players"]:
            del p["position"]
        self.assertEqual(hh.player_positions(hand)[4], "BTN")
        self.assertEqual(hh.player_positions(hand)[1], "UTG")

    def test_sitting_out_players_are_not_dealt(self) -> None:
        hand = fixtures.by_id("3H-SD")
        self.assertEqual([p["seat"] for p in hh.dealt_players(hand)], [1, 2, 4])
        self.assertNotIn(3, hh.replay(hand).seats)

    def test_groups_and_big_blinds(self) -> None:
        self.assertEqual(hh.position_group("UTG1"), "EP")
        self.assertEqual(hh.position_group("HJ"), "MP")
        self.assertEqual(hh.position_group("BTN"), "LP")
        self.assertEqual(hh.position_group("BB"), "BLINDS")
        self.assertIsNone(hh.position_group(None))
        hand = fixtures.by_id("HU-1")
        self.assertEqual(hh.to_bb(6000, hand), 6.0)
        hand["bb"] = 0
        self.assertEqual(hh.big_blind(hand), 1000)

    def test_textures(self) -> None:
        self.assertEqual(hh.flop_texture(["9c", "9d", "5s"]), {"paired": True, "monotone": False, "two_tone": False, "rainbow": True})
        self.assertTrue(hh.flop_texture(["Th", "9h", "2c", "5s"])["two_tone"])
        self.assertTrue(hh.flop_texture(["Ah", "Kh", "Qh"])["monotone"])
        self.assertTrue(hh.flop_texture(["Ah", "8c", "2d"])["rainbow"])
        self.assertIsNone(hh.flop_texture(["Ah"]))

    def test_seat_maps_and_derived_fields(self) -> None:
        hand = fixtures.by_id("ALLIN-1")
        self.assertEqual(hh.river_strength(hand), {2: 0.62, 5: 0.97})
        self.assertEqual(hh.allin_info(hand)["street"], "preflop")
        self.assertEqual(hh.collected_by_seat(hand), {5: 101500})
        self.assertEqual(hh.river_strength(fixtures.by_id("6M-3BET")), {})


class ReplayTests(unittest.TestCase):
    def test_final_pot_matches_total_pot(self) -> None:
        for hand in fixtures.all_hands():
            self.assertEqual(hh.replay(hand).final_pot, hand["total_pot"], hand["id"])

    def test_uncalled_bets_are_returned(self) -> None:
        self.assertEqual(hh.replay(fixtures.by_id("6M-3BET")).returned, {4: 5500})
        self.assertEqual(hh.replay(fixtures.by_id("HU-DONK")).returned, {3: 7500})
        self.assertEqual(hh.replay(fixtures.by_id("ALLIN-1")).returned, {})

    def test_pot_before_and_to_call(self) -> None:
        rp = hh.replay(fixtures.by_id("6M-XR"))
        flop = rp.by_street["flop"]
        self.assertEqual([d.pot_before for d in flop], [6500, 6500, 10500, 22500])
        self.assertEqual([d.to_call for d in flop], [0, 0, 4000, 8000])
        self.assertEqual([d.aggressions_before for d in flop], [0, 0, 1, 2])
        self.assertEqual(flop[2].size_fraction(), (8000, 14500))
        pots = hh.pot_before_actions(fixtures.by_id("HU-1"))
        self.assertEqual(pots[0], ("preflop", 0, 1, 1500))
        self.assertEqual(pots[3], ("flop", 1, 1, 6000))

    def test_straddle_and_facing_bet(self) -> None:
        rp = hh.replay(fixtures.by_id("STRADDLE-1"))
        pre = rp.by_street["preflop"]
        self.assertEqual([d.to_call for d in pre], [2000, 2000, 2000, 1500, 1000, 0])
        self.assertFalse(pre[-1].facing_bet)
        self.assertTrue(pre[-2].facing_bet)
        self.assertIsNone(rp.preflop_aggressor)
        self.assertEqual(rp.by_street["turn"][0].pot_before, 7500)

    def test_preflop_aggressor(self) -> None:
        self.assertEqual(hh.preflop_aggressor(fixtures.by_id("HU-1")), 1)
        self.assertEqual(hh.preflop_aggressor(fixtures.by_id("ALLIN-1")), 2)
        self.assertIsNone(hh.preflop_aggressor(fixtures.by_id("BOMB-1")))

    def test_bomb_pot_and_allin_runout(self) -> None:
        bomb = hh.replay(fixtures.by_id("BOMB-1"))
        self.assertEqual(bomb.streets_dealt, ("flop", "turn", "river"))
        self.assertEqual(bomb.by_street["flop"][0].pot_before, 8000)
        allin = hh.replay(fixtures.by_id("ALLIN-1"))
        self.assertEqual(allin.streets_dealt, ("preflop", "flop", "turn", "river"))
        self.assertEqual(allin.alive_at("flop"), frozenset({2, 5}))
        self.assertEqual(hh.showdown_seats(fixtures.by_id("ALLIN-1")), frozenset({2, 5}))
        jam = allin.by_street["preflop"][6]
        self.assertTrue(jam.all_in)
        self.assertFalse(allin.by_street["preflop"][7].can_raise)

    def test_can_raise_and_check_tracking(self) -> None:
        rp = hh.replay(fixtures.by_id("HU-1"))
        fold = rp.by_street["flop"][2]
        self.assertTrue(fold.can_raise)
        self.assertTrue(fold.checked_before)
        self.assertFalse(fold.first_decision)

    def test_showdown_fallback_and_net(self) -> None:
        hand = fixtures.by_id("3H-SD")
        del hand["derived"]
        self.assertEqual(hh.showdown_seats(hand), frozenset({1, 4}))
        self.assertEqual(hh.net_by_seat(hand), {1: 17000, 2: -500, 4: -17000})
        self.assertEqual(hh.showdown_seats(fixtures.by_id("HU-1")), frozenset())


if __name__ == "__main__":
    unittest.main()
