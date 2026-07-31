"""Plot Seoul Bike Sharing Demand experiment results."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
FIG_DIR = ROOT / "figures"
PREFIX = "seoul_bike_log_rentals_hour"
METHODS = ["LR", "Pooled", "DVCM", "ScalarRidge", "TL"]
LABELS = {
    "LR": "Target-only",
    "Pooled": "Pooled",
    "DVCM": "DVCM",
    "ScalarRidge": "Scalar Ridge",
    "TL": "TL",
}


def plot_mse_with_y_profile(result: pd.DataFrame, processed: pd.DataFrame) -> Path:
    colors = {
        "TL": "#d62728",
        "DVCM": "#1f77b4",
        "ScalarRidge": "#e6ab02",
        "LR": "#4d4d4d",
        "Pooled": "#7b3294",
    }
    markers = {"LR": "s", "Pooled": "D", "DVCM": "o", "ScalarRidge": "^", "TL": "o"}
    linestyles = {
        "LR": (0, (7, 2)),
        "Pooled": "-.",
        "DVCM": "--",
        "ScalarRidge": (0, (1, 1)),
        "TL": "-",
    }
    linewidths = {"TL": 1.2, "LR": 1.2, "Pooled": 1.2, "DVCM": 1.2, "ScalarRidge": 1.2}
    markersizes = {"TL": 4.8, "LR": 3.3, "Pooled": 3.3, "DVCM": 3.6, "ScalarRidge": 3.6}
    zorders = {"TL": 8, "ScalarRidge": 6, "LR": 5, "DVCM": 4, "Pooled": 3}

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.4, 4.45),
        gridspec_kw={"width_ratios": [2.5, 1.0], "wspace": 0.28},
    )
    ax, profile_ax = axes
    x = result["hour"].to_numpy()
    for method in METHODS:
        ax.plot(
            x,
            result[method].to_numpy(),
            color=colors[method],
            linestyle=linestyles[method],
            marker=markers[method],
            linewidth=linewidths[method],
            markersize=markersizes[method],
            alpha=1.0 if method == "TL" else 0.9,
            zorder=zorders[method],
            label=LABELS[method],
        )

    ax.set_yscale("log")
    ax.set_xlabel("Hour")
    ax.set_ylabel("MSE")
    ax.set_xticks(range(0, 24, 3))
    ax.grid(axis="y", which="both", alpha=0.25)

    profile_source = processed.copy()
    profile_source["rentals"] = np.expm1(profile_source["Y"].astype(float))
    profile = (
        profile_source.groupby("hour")["rentals"]
        .agg(
            mean="mean",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
        )
        .reset_index()
    )
    profile_ax.fill_between(
        profile["hour"].to_numpy(),
        profile["q25"].to_numpy(),
        profile["q75"].to_numpy(),
        color="#d9d9d9",
        alpha=0.85,
        linewidth=0,
    )
    profile_ax.plot(
        profile["hour"].to_numpy(),
        profile["mean"].to_numpy(),
        color="#252525",
        marker="o",
        markersize=3.2,
        linewidth=1.2,
    )
    profile_ax.set_xlabel("Hour")
    profile_ax.set_yscale("log")
    profile_ax.set_ylabel("Mean rentals")
    profile_ax.set_xticks(range(0, 24, 6))
    profile_ax.grid(axis="y", alpha=0.22)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=5,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.3,
    )
    fig.subplots_adjust(bottom=0.2)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    bandwidth = result["bandwidth"].iloc[0]
    if isinstance(bandwidth, str):
        bandwidth_label = bandwidth
    else:
        bandwidth_label = f"{float(bandwidth):.2f}"
    out = FIG_DIR / f"{PREFIX}_mse_with_y_profile_bandwidth_{bandwidth_label}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    cv_path = DATA_DIR / f"{PREFIX}_cv_bandwidth_result.csv"
    processed_path = DATA_DIR / f"processed_{PREFIX}.parquet"
    processed = pd.read_parquet(processed_path)
    result = pd.read_csv(cv_path).sort_values("hour")
    print(f"Saved {plot_mse_with_y_profile(result, processed)}")


if __name__ == "__main__":
    main()
