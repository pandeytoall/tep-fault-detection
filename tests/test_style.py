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


def test_finish_figure_titles_left_hides_top_right_spines_and_saves_a_png(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = style.new_figure(figsize=(4, 3), font_size=12)
    ax.plot([0, 1], [0, 1])
    path = tmp_path / "sub" / "chart.png"
    style.finish_figure(fig, ax, "A finding as the title", path)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert ax.get_title(loc="left") == "A finding as the title"
    assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()
    assert ax.spines["left"].get_visible() and ax.spines["bottom"].get_visible()
    assert not plt.fignum_exists(fig.number)  # closed after saving
