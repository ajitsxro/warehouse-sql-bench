"""PNG charts for the report. Colours follow one rule each: a warehouse always has
the same colour, outcomes take categorical slots in a fixed order, and the
feature heatmap uses a single light-to-dark blue."""

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - the backend must be chosen first
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from sqlbench.analysis.tables import WAREHOUSES, WarehouseAccuracy  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
WAREHOUSE_COLOURS = {"snowflake": "#2a78d6", "databricks": "#eb6834"}
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
OUTCOME_ORDER = [
    "syntax_error",
    "runtime_error",
    "wrong_result",
    "empty_result",
    "wrong_columns",
    "no_sql",
    "timeout",
]
DPI = 160


def style_axes(axes: Axes) -> None:
    axes.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(MUTED)
    axes.tick_params(colors=MUTED, length=0)
    axes.grid(axis="x", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)


def accuracy_chart(ranked: list[WarehouseAccuracy], path: Path) -> None:
    """Dots with 95% CI whiskers, one row per model, one colour per warehouse."""
    figure, axes = plt.subplots(figsize=(7.5, 0.55 * len(ranked) + 1.2), facecolor=SURFACE)
    style_axes(axes)
    offsets = {WAREHOUSES[0]: -0.15, WAREHOUSES[1]: 0.15}
    for position, entry in enumerate(ranked):
        for warehouse in WAREHOUSES:
            interval = entry.by_warehouse[warehouse]
            y = position + offsets[warehouse]
            colour = WAREHOUSE_COLOURS[warehouse]
            axes.plot([interval.low * 100, interval.high * 100], [y, y], color=colour, lw=2)
            axes.plot(interval.estimate * 100, y, "o", color=colour, ms=7, mec=SURFACE, mew=1.5)
    axes.set_yticks(range(len(ranked)), [entry.model_id for entry in ranked], color=TEXT)
    axes.invert_yaxis()
    # A little room past 100 so dots at 100% aren't cut in half.
    axes.set_xlim(0, 101.5)
    axes.set_xlabel("correct on main and edge (%), 95% bootstrap CI over questions", color=MUTED)
    handles = [
        Line2D([], [], color=WAREHOUSE_COLOURS[w], marker="o", lw=2, label=w) for w in WAREHOUSES
    ]
    axes.legend(handles=handles, loc="lower right", frameon=False, labelcolor=TEXT)
    figure.tight_layout()
    figure.savefig(path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)


def failure_chart(outcomes: dict[str, Counter[str]], path: Path) -> None:
    """One stacked bar per warehouse: failures by outcome."""
    present = [o for o in OUTCOME_ORDER if any(outcomes[w][o] for w in WAREHOUSES)]
    figure, axes = plt.subplots(figsize=(7.5, 2.4), facecolor=SURFACE)
    style_axes(axes)
    for position, warehouse in enumerate(WAREHOUSES):
        left = 0
        for slot, outcome in enumerate(present):
            count = outcomes[warehouse][outcome]
            if count == 0:
                continue
            axes.barh(
                position,
                count,
                left=left,
                color=CATEGORICAL[slot],
                edgecolor=SURFACE,
                linewidth=2,
                height=0.6,
                label=outcome if position == 0 else None,
            )
            left += count
        axes.text(left + 1, position, str(left), va="center", color=TEXT, fontsize=9)
    axes.set_yticks(range(len(WAREHOUSES)), list(WAREHOUSES), color=TEXT)
    axes.invert_yaxis()
    axes.set_xlabel("failed tasks (all models)", color=MUTED)
    handles = [
        Rectangle((0, 0), 1, 1, color=CATEGORICAL[slot], label=outcome)
        for slot, outcome in enumerate(present)
    ]
    axes.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.45),
        ncol=len(present),
        frameon=False,
        fontsize=8,
        labelcolor=TEXT,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)


def feature_chart(
    table: dict[str, dict[str, tuple[float, int]]], models: list[str], path: Path
) -> None:
    """Heatmap of pooled accuracy, features down, models across, value in each cell."""
    features = sorted({feature for row in table.values() for feature in row})
    values = [[table[model][feature][0] * 100 for model in models] for feature in features]
    colour_map = LinearSegmentedColormap.from_list("blue", SEQUENTIAL)
    figure, axes = plt.subplots(
        figsize=(1.1 * len(models) + 2.6, 0.38 * len(features) + 1.6), facecolor=SURFACE
    )
    axes.imshow(values, cmap=colour_map, vmin=0, vmax=100, aspect="auto")
    for row, feature_values in enumerate(values):
        for column, value in enumerate(feature_values):
            ink = SURFACE if value >= 55 else TEXT
            axes.text(column, row, f"{value:.0f}", ha="center", va="center", color=ink, fontsize=8)
    counts = [table[models[0]][feature][1] for feature in features]
    labels = [f"{feature} ({count})" for feature, count in zip(features, counts, strict=True)]
    axes.set_yticks(range(len(features)), labels, color=TEXT, fontsize=8)
    axes.set_xticks(range(len(models)), models, color=TEXT, fontsize=8, rotation=30, ha="right")
    axes.tick_params(length=0)
    for side in axes.spines.values():
        side.set_visible(False)
    axes.set_title("correct (%), both warehouses pooled; (questions)", color=MUTED, fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
