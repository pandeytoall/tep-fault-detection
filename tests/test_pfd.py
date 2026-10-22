"""Tests for the TEP process flow diagram (docs/figures/tep-pfd.svg).

Run with: python -m unittest tests.test_pfd   (or pytest). Standard library only.
Expected routing literals below come from research/tep-flowsheet.md,
section "Stream routing verified from the flowsheet figure".
"""
import re
import struct
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import check_pfd  # noqa: E402

SVG = ROOT / "docs" / "figures" / "tep-pfd.svg"
PNG = ROOT / "docs" / "figures" / "tep-pfd.png"
NS = "{http://www.w3.org/2000/svg}"


def _root():
    return ET.parse(SVG).getroot()


def _streams():
    return {p.get("id"): p for p in _root().iter(NS + "path") if "stream" in (p.get("class") or "").split()}


class TestCheckScript(unittest.TestCase):
    def test_real_diagram_has_no_problems(self):
        self.assertEqual(check_pfd.check(SVG), [])

    def test_duplicate_stream_is_reported(self):
        text = SVG.read_text(encoding="utf-8")
        dup = re.search(r'<path id="stream-3"[^>]*/>', text).group(0)
        problems = check_pfd.check_text(text.replace(dup, dup + dup))
        self.assertTrue(any("stream-3" in p for p in problems), problems)

    def test_missing_stream_is_reported(self):
        text = SVG.read_text(encoding="utf-8")
        gone = re.search(r'<path id="stream-13"[^>]*/>', text).group(0)
        problems = check_pfd.check_text(text.replace(gone, ""))
        self.assertTrue(any("stream-13" in p for p in problems), problems)

    def test_unknown_endpoint_is_reported(self):
        text = SVG.read_text(encoding="utf-8")
        bad = text.replace('id="stream-10"', 'id="stream-10" data-bogus="1"', 1)
        bad = re.sub(r'(id="stream-10"[^>]*data-to=")[^"]+', r"\1unit-nowhere", bad, count=1)
        problems = check_pfd.check_text(bad)
        self.assertTrue(any("unit-nowhere" in p for p in problems), problems)

    def test_wrong_routing_is_reported(self):
        # The classic mistake: stream 4 drawn into the reactor feed instead of the stripper.
        text = SVG.read_text(encoding="utf-8")
        bad = re.sub(r'(id="stream-4"[^>]*data-to=")[^"]+', r"\1junction-feed-header", text, count=1)
        problems = check_pfd.check_text(bad)
        self.assertTrue(any("stream-4" in p for p in problems), problems)

    def test_duplicate_unit_label_is_reported(self):
        text = SVG.read_text(encoding="utf-8")
        bad = text.replace('data-name="Condenser"', 'data-name="Reactor"', 1)
        problems = check_pfd.check_text(bad)
        self.assertTrue(any("Reactor" in p for p in problems), problems)

    def test_diagonal_stream_is_reported(self):
        text = SVG.read_text(encoding="utf-8")
        bad = re.sub(r'(id="stream-1"[^>]*d=")M ([\d.]+) ([\d.]+) H', r"\1M \2 \3 L 100 100 H", text, count=1)
        problems = check_pfd.check_text(bad)
        self.assertTrue(any("stream-1" in p and "orthogonal" in p for p in problems), problems)


class TestRouting(unittest.TestCase):
    """Acceptance criteria: routing matches the verified table."""

    EXPECTED = {
        "stream-1": ("inlet-feed-a", "junction-feed-header"),
        "stream-2": ("inlet-feed-d", "junction-feed-header"),
        "stream-3": ("inlet-feed-e", "junction-feed-header"),
        "stream-4": ("inlet-feed-c", "unit-stripper"),
        "stream-5": ("unit-stripper", "junction-feed-header"),
        "stream-6": ("junction-feed-header", "unit-reactor"),
        "stream-7": ("unit-reactor", "unit-condenser"),
        "stream-8": ("unit-compressor", "junction-feed-header"),
        "stream-9": ("unit-separator", "outlet-purge"),
        "stream-10": ("unit-separator", "unit-stripper"),
        "stream-11": ("unit-stripper", "outlet-product"),
        "stream-12": ("inlet-cw-reactor", "unit-reactor"),
        "stream-13": ("inlet-cw-condenser", "unit-condenser"),
        "stream-steam": ("inlet-steam", "unit-stripper"),
        "stream-condensate": ("unit-stripper", "outlet-condensate"),
        "stream-condenser-outlet": ("unit-condenser", "unit-separator"),
        "stream-separator-vapor": ("unit-separator", "unit-compressor"),
    }

    def test_routing_matches_verified_table(self):
        streams = _streams()
        for sid, (src, dst) in self.EXPECTED.items():
            with self.subTest(stream=sid):
                self.assertIn(sid, streams)
                self.assertEqual((streams[sid].get("data-from"), streams[sid].get("data-to")), (src, dst))

    def test_five_units_exactly(self):
        units = [g.get("id") for g in _root().iter(NS + "g") if g.get("class") == "unit"]
        self.assertEqual(
            sorted(units),
            ["unit-compressor", "unit-condenser", "unit-reactor", "unit-separator", "unit-stripper"],
        )

    def test_unit_names_are_generic(self):
        names = {g.get("id"): g.get("data-name") for g in _root().iter(NS + "g") if g.get("class") == "unit"}
        self.assertEqual(
            names,
            {
                "unit-reactor": "Reactor",
                "unit-condenser": "Condenser",
                "unit-separator": "Vapor-liquid separator",
                "unit-compressor": "Recycle compressor",
                "unit-stripper": "Stripper",
            },
        )

    def test_feed_labels(self):
        texts = {t.text for t in _root().iter(NS + "text")}
        for label in ("1: A feed", "2: D feed", "3: E feed", "4: C feed"):
            self.assertIn(label, texts)


class TestStyle(unittest.TestCase):
    def test_title(self):
        title = [t for t in _root().iter(NS + "text") if t.get("id") == "title"]
        self.assertEqual(len(title), 1)
        self.assertEqual(title[0].text, "Tennessee Eastman Process: 5 units, 52 readings, 20 faults")

    def test_canvas_is_wide_16_9(self):
        root = _root()
        self.assertEqual(root.get("viewBox"), "0 0 1600 900")

    def test_only_three_font_sizes_and_units_at_least_18(self):
        root = _root()
        sizes = {int(t.get("font-size")) for t in root.iter(NS + "text")}
        self.assertEqual(sizes, {14, 18, 32})
        for g in root.iter(NS + "g"):
            if g.get("class") == "unit":
                for t in g.iter(NS + "text"):
                    self.assertGreaterEqual(int(t.get("font-size")), 18, g.get("id"))

    def test_dashed_stream_types_have_distinct_patterns(self):
        patterns = check_pfd.stream_type_styles(SVG)
        dashed = {t: p for t, p in patterns.items() if p}
        self.assertGreaterEqual(len(dashed), 4)
        self.assertEqual(len(set(dashed.values())), len(dashed), dashed)

    def test_every_used_stream_type_is_in_legend(self):
        root = _root()
        used = {c for p in _streams().values() for c in p.get("class").split() if c.startswith("stream-")}
        legend = next(g for g in root.iter(NS + "g") if g.get("id") == "legend")
        in_legend = {c for p in legend.iter(NS + "path") for c in (p.get("class") or "").split() if c.startswith("stream-")}
        self.assertEqual(used, in_legend)

    def test_no_tags_ids_or_company_names_in_text(self):
        for t in _root().iter(NS + "text"):
            text = t.text or ""
            self.assertIsNone(re.search(r"\b[A-Z]{1,4}-?\d{2,4}\b", text), text)
            for banned in ("Exxon", "Chemical", "Downs", "Vogel", "IFAC", "Bathelt", "Ricker"):
                self.assertNotIn(banned, text)

    def test_self_contained(self):
        text = SVG.read_text(encoding="utf-8")
        for banned in ("<image", "<script", "<foreignObject", "href=", "url(http", "@import"):
            self.assertNotIn(banned, text)

    def test_text_is_not_duplicated(self):
        texts = [t.text for t in _root().iter(NS + "text") if t.get("id") != "title"]
        # Legend words and stream-label words may repeat in meaning, but exact strings must not.
        self.assertEqual(len(texts), len(set(texts)), [x for x in texts if texts.count(x) > 1])


class TestPng(unittest.TestCase):
    def test_png_is_at_least_1600_by_900(self):
        data = PNG.read_bytes()
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", data[16:24])
        self.assertGreaterEqual(width, 1600)
        self.assertGreaterEqual(height, 900)
        self.assertAlmostEqual(width / height, 16 / 9, places=2)


if __name__ == "__main__":
    unittest.main()
