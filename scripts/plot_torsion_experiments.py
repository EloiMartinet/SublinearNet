import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Configuration (NeurIPS-like styling)
# ---------------------------
plt.rcParams.update({
    "font.size": 14,
    "font.family": "serif",
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2.5,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
})

# ---------------------------
# File paths
# ---------------------------
files = {
    "MFS": "res/csvs/history_MFS.csv",
    "PINN (config 1)": "res/csvs/history_PINN_config_1.csv",
    "PINN (config 2)": "res/csvs/history_PINN_config_2.csv",
    "PINN (config 3)": "res/csvs/history_PINN_config_3.csv",
    "Galerkin": "res/csvs/history_Galerkin.csv",
}

# ---------------------------
# Plot
# ---------------------------
fig, ax = plt.subplots(figsize=(7, 5))

for label, path in files.items():
    df = pd.read_csv(path)

    # Ensure correct columns exist
    assert "time" in df.columns and "deficit" in df.columns, \
        f"{path} must contain 'time' and 'deficit' columns"

    # Sort by time (important for log plots)
    df = df.sort_values("time")

    ax.plot(
        df["time"] + 1, # Artificially add 1sd everywhere for readability purposed on the log-log scale
        df["deficit"],
        label=label,
        marker=None
    )

# ---------------------------
# Log-log scale
# ---------------------------
ax.set_xscale("log")
ax.set_yscale("log")

# ---------------------------
# Labels and legend
# ---------------------------
ax.set_xlabel("Time (s)")
ax.set_ylabel("Isoperimetri Deficit")

ax.legend(frameon=True)

# ---------------------------
# Layout & save
# ---------------------------
plt.tight_layout()

# High-quality output for papers
plt.savefig("comparison_plot.pdf", dpi=300, bbox_inches="tight")
plt.savefig("comparison_plot.png", dpi=300, bbox_inches="tight")

plt.show()