import random
import unittest

from poker_ecosystem import cluster
from poker_ecosystem.opponent import OpponentModel
from poker_ecosystem.stats import StatsTable


def blobs(seed: int = 0, per: int = 25) -> tuple[list[list[float]], list[int]]:
    rng = random.Random(seed)
    centres = [(0.0, 0.0, 0.0), (6.0, 6.0, 0.0), (0.0, 6.0, 6.0)]
    points, labels = [], []
    for j, c in enumerate(centres):
        for _ in range(per):
            points.append([x + rng.gauss(0, 0.6) for x in c])
            labels.append(j)
    return points, labels


class AriTests(unittest.TestCase):
    def test_known_values(self) -> None:
        self.assertEqual(cluster.adjusted_rand_index([0, 0, 1, 1], [0, 0, 1, 1]), 1.0)
        self.assertEqual(cluster.adjusted_rand_index([0, 0, 1, 1], ["b", "b", "a", "a"]), 1.0)
        self.assertAlmostEqual(cluster.adjusted_rand_index([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2]), 0.8 / 3.3, places=12)
        self.assertAlmostEqual(cluster.adjusted_rand_index([0, 0, 1, 1], [0, 0, 1, 2]), 4 / 7, places=12)
        self.assertEqual(cluster.adjusted_rand_index([1, 1, 1], [2, 2, 2]), 1.0)
        with self.assertRaises(ValueError):
            cluster.adjusted_rand_index([0], [0, 1])

    def test_random_labelings_score_near_zero(self) -> None:
        rng = random.Random(1)
        a = [rng.randrange(4) for _ in range(2000)]
        b = [rng.randrange(4) for _ in range(2000)]
        self.assertLess(abs(cluster.adjusted_rand_index(a, b)), 0.02)


class AlgorithmTests(unittest.TestCase):
    def test_kmeans_recovers_blobs_deterministically(self) -> None:
        points, labels = blobs()
        first = cluster.kmeans(points, 3, seed=42)
        second = cluster.kmeans(points, 3, seed=42)
        self.assertEqual(first.labels, second.labels)
        self.assertEqual(cluster.adjusted_rand_index(labels, first.labels), 1.0)
        self.assertEqual(cluster.kmeans(points, 1).inertia, cluster.kmeans(points, 1, seed=9).inertia)
        self.assertEqual(cluster.kmeans(points[:2], 5).k, 2)

    def test_gmm_bic_selects_three_components(self) -> None:
        points, labels = blobs(seed=3)
        best, table = cluster.select_gmm(points, range(1, 6), seed=1)
        self.assertEqual(best.k, 3)
        self.assertEqual(set(table), {1, 2, 3, 4, 5})
        self.assertEqual(min(table, key=table.get), 3)
        self.assertEqual(cluster.adjusted_rand_index(labels, best.labels), 1.0)
        self.assertAlmostEqual(sum(best.weights), 1.0)
        self.assertTrue(best.converged)

    def test_standardise(self) -> None:
        z, centre, scale = cluster.standardise([[1.0, 5.0], [3.0, 5.0]])
        self.assertEqual(z, [[-1.0, 0.0], [1.0, 0.0]])
        self.assertEqual(centre, [2.0, 5.0])
        self.assertEqual(scale, [1.0, 0.0])


class NamingTests(unittest.TestCase):
    def test_rules(self) -> None:
        name = cluster.name_cluster
        self.assertEqual(name({"vpip": 0.55, "pfr": 0.40}), "maniac")
        self.assertEqual(name({"vpip": 0.12, "pfr": 0.10}), "nit")
        self.assertEqual(name({"vpip": 0.45, "pfr": 0.08, "fold_vs_cbet_flop": 0.25}), "calling station")
        self.assertEqual(name({"vpip": 0.40, "pfr": 0.15, "fold_vs_cbet_flop": 0.55}), "loose-passive")
        self.assertEqual(name({"vpip": 0.32, "pfr": 0.25}), "LAG")
        self.assertEqual(name({"vpip": 0.22, "pfr": 0.18}), "TAG")
        self.assertEqual(name({"vpip": 0.22, "pfr": 0.06}), "tight-passive")

    def test_cluster_players_on_a_model(self) -> None:
        rng = random.Random(2)
        table = StatsTable()
        styles = {"tight": (0.14, 0.11), "loose": (0.45, 0.08), "aggro": (0.30, 0.24)}
        truth = {}
        for style, (vpip, pfr) in styles.items():
            for i in range(8):
                name = f"{style}{i}"
                truth[name] = style
                for stat, p in (("vpip", vpip), ("pfr", pfr)):
                    n = 400
                    table.add_counts(name, stat, "BTN", n, sum(rng.random() < p for _ in range(n)))
        model = OpponentModel(table)
        result = cluster.cluster_players(model, method="gmm", stats=("vpip", "pfr"), k_values=range(1, 6), seed=0)
        self.assertEqual(result.k, 3)
        self.assertEqual(sum(c["size"] for c in result.summary), 24)
        self.assertEqual(cluster.adjusted_rand_index([truth[p] for p in result.players], result.labels), 1.0)
        names = set(result.names)
        self.assertIn("nit", names)
        self.assertTrue(names & {"calling station", "loose-passive"})
        km = cluster.cluster_players(model, method="kmeans", k=3, stats=("vpip", "pfr"))
        self.assertEqual(km.k, 3)
        self.assertIn("assignments", km.as_dict())
        with self.assertRaises(ValueError):
            cluster.cluster_players(model, method="spectral")


if __name__ == "__main__":
    unittest.main()
