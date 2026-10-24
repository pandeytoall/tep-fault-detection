"""Shared chart colours (the Okabe-Ito colour-blind-safe palette) and the chart boilerplate."""
from pathlib import Path

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}
NORMAL = OKABE_ITO["blue"]
FAULT = OKABE_ITO["vermillion"]


def new_figure(figsize=(8, 5.5), font_size=15):
    """(fig, ax) at 150 dpi with the project font size; readable at phone size."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": font_size})
    return plt.subplots(figsize=figsize, dpi=150)


def finish_figure(fig, ax, title, path, title_size=None, **title_kwargs):
    """Left-aligned finding title, no top/right spines, save to `path` (parents created), close the figure."""
    import matplotlib.pyplot as plt
    ax.set_title(title, loc="left", **({"fontsize": title_size} if title_size else {}), **title_kwargs)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
