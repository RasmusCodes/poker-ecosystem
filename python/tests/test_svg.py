import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from poker_ecosystem import svg

NS = "{http://www.w3.org/2000/svg}"


def parse(doc: str) -> ET.Element:
    root = ET.fromstring(doc)
    assert root.tag == f"{NS}svg"
    return root


def texts(root: ET.Element) -> list[str]:
    return [t.text or "" for t in root.iter(f"{NS}text")]


class TickTests(unittest.TestCase):
    def test_ticks_and_formatting(self) -> None:
        self.assertEqual(svg.nice_ticks(0, 0.37), [0.0, 0.1, 0.2, 0.3, 0.4])
        self.assertEqual(svg.nice_ticks(0, 2500), [0.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0])
        self.assertEqual(svg.log_ticks(50, 2500), [50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0])
        self.assertEqual(svg.log_ticks(1, 1e4), [1.0, 10.0, 100.0, 1000.0, 10000.0])
        self.assertEqual(svg.format_number(2500), "2,500")
        self.assertEqual(svg.format_number(25000), "25K")
        self.assertEqual(svg.format_number(0.05), "0.05")
        self.assertEqual(svg.format_number(0.30000000000000004), "0.3")
        with self.assertRaises(ValueError):
            svg.log_ticks(0, 10)


class ChartTests(unittest.TestCase):
    def test_line_chart_with_log_axis_legend_and_reference(self) -> None:
        doc = svg.line_chart(
            [
                svg.Series("posterior", [50, 100, 250, 1000], [0.08, 0.06, 0.04, 0.02], lo=[0.06, 0.05, 0.03, 0.015], hi=[0.1, 0.07, 0.05, 0.025]),
                svg.Series("raw", [50, 100, 250, 1000], [0.12, None, 0.05, 0.025]),
            ],
            title="Error & <hands>",
            x_label="hands",
            y_label="MAE",
            log_x=True,
            reference=0.05,
            reference_label="target",
        )
        root = parse(doc)
        self.assertIn("currentColor", doc)
        self.assertIn("prefers-color-scheme: dark", doc)
        self.assertIn(':root:not([data-theme="light"]) .pe-chart .pe-l0', doc)  # inline follows the page's stamp
        self.assertIn(':root[data-theme="dark"] .pe-chart .pe-l1', doc)
        self.assertIn("svg.pe-chart:root>.pe-bg", doc)  # standalone files paint their own surface
        self.assertEqual(root.find(f"{NS}rect").get("class"), "pe-bg")
        labels = texts(root)
        self.assertIn("Error & <hands>", labels)
        self.assertIn("posterior", labels)
        self.assertIn("raw", labels)
        self.assertIn("1,000", labels)
        self.assertIn("target", labels)
        self.assertEqual(len(root.findall(f"{NS}circle")), 7)  # one marker per non-missing point
        self.assertTrue(all(c.find(f"{NS}title") is not None for c in root.findall(f"{NS}circle")))

    def test_single_series_has_no_legend(self) -> None:
        root = parse(svg.line_chart([svg.Series("only", [1, 2, 3], [1, 4, 9])], title="Squares"))
        self.assertNotIn("only", texts(root))

    def test_grouped_bars_with_error_bars(self) -> None:
        doc = svg.grouped_bar_chart(
            ["vpip", "pfr", "three_bet"],
            [("posterior", [0.03, 0.02, 0.01]), ("raw", [0.05, 0.04, None])],
            errors=[[(0.02, 0.04), None, (0.0, 0.02)], None],
            title="MAE by stat",
            y_label="MAE",
        )
        root = parse(doc)
        bars = [p for p in root.iter(f"{NS}path") if p.find(f"{NS}title") is not None]
        self.assertEqual(len(bars), 5)
        self.assertIn("A4,4", doc)  # rounded data end
        self.assertIn("three_bet", texts(root))

    def test_scatter_and_histogram(self) -> None:
        groups = [(f"style{i}", [(0.1 * i, 0.05 * i), (0.1 * i + 0.02, 0.05 * i + 0.01)]) for i in range(5)]
        root = parse(svg.scatter_chart(groups, title="VPIP vs PFR", x_label="VPIP", y_label="PFR"))
        self.assertGreaterEqual(len(root.findall(f"{NS}polygon")), 8)  # non-circle shapes for groups 2-5
        hist = parse(svg.histogram([0.1, 0.12, 0.5, 0.52, 0.55, 0.9], bins=4, value_range=(0.0, 1.0), title="Errors"))
        bars = [p for p in hist.iter(f"{NS}path") if p.find(f"{NS}title") is not None]
        self.assertEqual(len(bars), 3)

    def test_limits_and_empty_data(self) -> None:
        with self.assertRaises(ValueError):
            svg.line_chart([svg.Series(str(i), [1], [1]) for i in range(9)])
        root = parse(svg.line_chart([svg.Series("a", [], [])], title="Empty"))
        self.assertIn("no data", texts(root))
        self.assertIn("no data", texts(parse(svg.histogram([], title="Nothing"))))

    def test_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = svg.save(svg.histogram([1, 2, 2, 3]), Path(tmp) / "nested" / "h.svg")
            self.assertTrue(path.read_text(encoding="utf-8").startswith("<svg"))


if __name__ == "__main__":
    unittest.main()
