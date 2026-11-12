"""Phase 5 tests on the built reports in results/report/ (skipped until `scripts/make_report.py` has been run)."""
import re

import pytest

from src import report, report_text
from src.data import ROOT

OUT = ROOT / "results" / "report"
built = pytest.mark.skipif(not (OUT / "report.md").exists(), reason="make_report.py has not been run")

SECTIONS = ["What this is and what to take from it", "Data and rules", "Detection", "Against the literature", "The unseen-fault test",
            "Diagnosis", "The assistant", "What we did not do and what is uncertain", "Reproduce it", "How this was made"]


def _sections(text):
    parts = re.split(r"^## ", text, flags=re.M)[1:]
    return {p.split("\n", 1)[0]: p for p in parts}


@built
def test_every_link_and_image_in_the_three_files_exists():
    for name in ("report.md", "operator-fault6.md", "README.md"):
        text = (OUT / name).read_text(encoding="utf8")
        assert report.broken_links(text, OUT) == [], name
    assert (OUT / "figures" / "detection-by-family.png").exists() and (OUT / "figures" / "operator-fault6.png").exists()


@built
def test_report_sections_are_in_the_asked_order():
    text = (OUT / "report.md").read_text(encoding="utf8")
    headings = re.findall(r"^## (.*)$", text, flags=re.M)
    assert headings == SECTIONS


@built
def test_the_summary_has_five_bullets():
    section = _sections((OUT / "report.md").read_text(encoding="utf8"))[SECTIONS[0]]
    assert len(re.findall(r"^- \*\*", section, flags=re.M)) == 5


@built
def test_each_caveat_sits_in_the_section_that_uses_its_number():
    sections = _sections((OUT / "report.md").read_text(encoding="utf8"))
    for prefix, phrases in report_text.CAVEATS.items():
        body = next(v for k, v in sections.items() if k.startswith(prefix)).lower()
        for phrase in phrases:
            assert phrase.lower() in body, (prefix, phrase)


@built
def test_no_claim_that_hard_faults_are_detected_and_no_cross_track_ranking():
    text = (OUT / "report.md").read_text(encoding="utf8").lower()
    assert "not detected" in text
    for bad in ("faults 3, 9 and 15 are detected", "better than the unsupervised", "beats the unsupervised", "outperforms pca",
                "reasoning alone reached", "reasoned better than"):
        assert bad not in text, bad


@built
def test_the_assistant_gain_is_attributed_to_the_classifier():
    section = _sections((OUT / "report.md").read_text(encoding="utf8"))["The assistant"]
    assert "a trained classifier helps" in section and "not better reasoning" in section
    assert "All 90 of 90" in section


@built
def test_operator_report_has_the_asked_sections_and_a_how_made_line():
    text = (OUT / "operator-fault6.md").read_text(encoding="utf8")
    for heading in ("## Situation", "## What the readings show", "## Where to look first", "## What this does and doesn't tell you",
                    "## Evidence", "## How this was made"):
        assert heading in text
    assert "peak RAM" in text.split("## How this was made")[1] and "laptop" in text.split("## How this was made")[1]


@built
def test_operator_report_states_the_shutdown_tail_with_its_hedge():
    text = (OUT / "operator-fault6.md").read_text(encoding="utf8")
    why = text.split("Why they differ:")[1].split("\n")[0]
    assert "main reason is the shutdown tail" in why and "not verified here" in why
    assert "shutdown limit" in text


@built
def test_pressure_table_file_lists_the_affected_faults():
    import pandas as pd
    table = pd.read_csv(OUT / "pressure_tail.csv", index_col="fault")
    assert len(table) == 20 and table.loc[6, "share_reached"] == 1.0
    assert set(table.index[table["share_reached"] > 0]) == {6, 12, 13, 18}
