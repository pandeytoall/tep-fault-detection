"""Phase 1 tests for src/style.py (shared chart colours)."""
import re

from src import style


def test_normal_and_fault_colours_are_okabe_ito_blue_and_vermillion():
    assert style.NORMAL == "#0072B2"
    assert style.FAULT == "#D55E00"


def test_palette_has_the_eight_okabe_ito_colours():
    assert len(style.OKABE_ITO) == 8
    assert len(set(style.OKABE_ITO.values())) == 8
    assert all(re.fullmatch(r"#[0-9A-F]{6}", c) for c in style.OKABE_ITO.values())
    assert style.OKABE_ITO["blue"] == style.NORMAL and style.OKABE_ITO["vermillion"] == style.FAULT
