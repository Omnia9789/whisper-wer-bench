"""Chart generation — WER distribution and model comparison bar charts."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

COLORS = {"small": "#ffa657", "medium": "#bc8cff"}
BG = "#161b22"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#c9d1d9"
DIM_COLOR = "#8b949e"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(BG)
    ax.figure.set_facecolor(BG)
    ax.tick_params(colors=DIM_COLOR, labelsize=9)
    ax.spines[:].set_color(GRID_COLOR)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(DIM_COLOR)


def plot_per_clip_comparison(df: pd.DataFrame, output_path: str) -> None:
    """Grouped bar chart: WER per clip, grouped by model."""
    models = df["model"].unique()
    clips = df["clip"].unique()
    n = len(clips)
    x = range(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, model in enumerate(models):
        sub = df[df["model"] == model].set_index("clip").reindex(clips)
        bars = ax.bar(
            [xi + i * width for xi in x],
            sub["wer_pct"],
            width=width,
            color=COLORS.get(model, "#58a6ff"),
            label=model,
            zorder=3,
            edgecolor="#0d1117",
            linewidth=0.5,
        )
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.15,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=DIM_COLOR,
            )

    ax.set_xticks([xi + width / 2 for xi in x])
    ax.set_xticklabels(clips, rotation=30, ha="right")
    ax.set_ylabel("WER (%)", color=DIM_COLOR, fontsize=10)
    ax.set_title("Per-Clip WER — Model Comparison", color=TEXT_COLOR, fontsize=12, pad=12)
    ax.legend(facecolor=BG, edgecolor=GRID_COLOR, labelcolor=DIM_COLOR, fontsize=9)
    _style(ax)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✔ Saved {output_path}")


def plot_wer_distribution(df: pd.DataFrame, output_path: str) -> None:
    """KDE / histogram of WER distribution per model."""
    models = df["model"].unique()
    fig, ax = plt.subplots(figsize=(8, 4))

    for model in models:
        data = df[df["model"] == model]["wer_pct"]
        ax.hist(
            data,
            bins=8,
            alpha=0.65,
            color=COLORS.get(model, "#58a6ff"),
            label=model,
            edgecolor="#0d1117",
            linewidth=0.5,
            zorder=3,
        )

    ax.set_xlabel("WER (%)", color=DIM_COLOR, fontsize=10)
    ax.set_ylabel("Count", color=DIM_COLOR, fontsize=10)
    ax.set_title("WER Distribution by Model", color=TEXT_COLOR, fontsize=12, pad=12)
    ax.legend(facecolor=BG, edgecolor=GRID_COLOR, labelcolor=DIM_COLOR, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✔ Saved {output_path}")
