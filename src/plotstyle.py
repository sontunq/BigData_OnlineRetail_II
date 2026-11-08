"""Shared matplotlib style for all figures (static PNG, light background, 200 dpi).

Colours follow a validated categorical palette; slots are assigned in fixed order and
stay with the entity (e.g. each year always has the same colour in every chart).
For scatter-like charts use at most 3 categorical colours (only the first three slots are
distinguishable in every pairing); fold the rest into "Other" or use small multiples.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import FIGURES_DIR  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_DARK = "#184f95"
BLUE_LIGHT = "#9ec5f4"
NEUTRAL = "#a8a7a2"

# Fixed year -> colour mapping, used when the two trading years are compared
YEAR_COLOR = {2010: SERIES[0], 2011: SERIES[1], 2009: SERIES[2]}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DPI = 200

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "text.color": TEXT,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2,
    "legend.frameon": False, "figure.dpi": 100,
})


def money(x, _pos=None):
    return f"${x / 1000:,.0f}k" if abs(x) >= 1000 else f"${x:,.0f}"


def save(fig, name: str):
    path = FIGURES_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path
