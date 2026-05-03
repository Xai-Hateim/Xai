"""
benchmark_chart.py
==================
Produces the main benchmark summary figure from all CSVs in results/csv/.

Layout  :  2 pages (dim=5, dim=10), each page is:
           rows    = 4 key metrics (faithfulness, roar_faithfulness,
                                    shapley_corr, monotonicity)
           columns = 3 datasets
           x-axis  = rho (0.0 → 0.99)
           lines   = explainers, averaged across LR / DTREE / MLP

Output  :  results/plots/benchmark_dim=5.png
           results/plots/benchmark_dim=10.png
"""

import os, io, glob, collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Settings ──────────────────────────────────────────────────────────────────
CSV_DIR    = "results/csv"
OUTPUT_DIR = "results/plots"

KEY_METRICS = [
    ("faithfulness",     "Faithfulness ↑"),
    ("roar_faithfulness","ROAR Faithfulness ↑"),
    ("shapley_corr",     "Shapley Corr. ↑"),
    ("monotonicity",     "Monotonicity ↑"),
]

DATASETS = [
    "gaussianLinear",
    "gaussianNonLinearAdditive",
    "gaussianPiecewiseConstant",
]
DATASET_LABELS = {
    "gaussianLinear":            "Gaussian Linear",
    "gaussianNonLinearAdditive": "Gaussian\nNonLinear Additive",
    "gaussianPiecewiseConstant": "Gaussian\nPiecewise Const.",
}
MODELS = ["LR", "DTREE", "MLP"]
RHOS   = [0.0, 0.25, 0.5, 0.75, 0.99]

EXPLAINER_ORDER  = ["random", "breakdown", "shap", "brutekernelshap",
                    "shapr",  "maple",     "lime"]
EXPLAINER_LABELS = {
    "random":         "RANDOM",
    "breakdown":      "BreakDown",
    "shap":           "SHAP",
    "brutekernelshap":"BF-SHAP",
    "shapr":          "SHAPR",
    "maple":          "MAPLE",
    "lime":           "LIME",
}
# Distinct palette: colorblind-friendly
PALETTE = [
    "#e41a1c",  # red     – RANDOM
    "#ff7f00",  # orange  – BreakDown
    "#4daf4a",  # green   – SHAP
    "#984ea3",  # purple  – BF-SHAP
    "#377eb8",  # blue    – SHAPR
    "#a65628",  # brown   – MAPLE
    "#999999",  # grey    – LIME
]
EXPL_STYLE = {
    e: {"color": PALETTE[i], "marker": "oDsv^<>"[i], "zorder": 3 - (i == 0)}
    for i, e in enumerate(EXPLAINER_ORDER)
}


# ── CSV parser ────────────────────────────────────────────────────────────────
def parse_csv(path):
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    meta = {}
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s.startswith("#"):
            break
        body = s.lstrip("#").strip()
        if "," in body:
            k, v = body.split(",", 1)
            meta[k.strip()] = v.strip()
        i += 1
    blocks = {}
    while i < len(lines):
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break
        first = lines[i].strip().split(",")[0].strip()
        if not first:
            i += 1
            continue
        model_name = first
        i += 1
        chunk = []
        while i < len(lines) and lines[i].strip():
            chunk.append(lines[i])
            i += 1
        if not chunk:
            continue
        df = pd.read_csv(io.StringIO("".join(chunk)))
        df = df.rename(columns={df.columns[0]: "explainer"})
        df = df.dropna(subset=["explainer"])
        df["explainer"] = df["explainer"].astype(str).str.strip()
        blocks[model_name] = df
    return meta, blocks


# ── Load all CSVs ─────────────────────────────────────────────────────────────
def load_all(csv_dir):
    # data[dataset][dim][rho][model] = DataFrame
    data = collections.defaultdict(
        lambda: collections.defaultdict(
            lambda: collections.defaultdict(dict)
        )
    )
    for path in glob.glob(os.path.join(csv_dir, "*.csv")):
        meta, blocks = parse_csv(path)
        ds  = meta.get("dataset", "")
        dim = int(meta.get("dim", 0))
        rho = float(meta.get("rho", 0))
        if ds and blocks:
            for model, df in blocks.items():
                data[ds][dim][rho][model] = df
    return data


# ── Compute mean ± std across models ─────────────────────────────────────────
def get_series(data, dataset, dim, metric, explainer):
    """Return xs, means, stds averaged across all models."""
    xs, means, stds = [], [], []
    for rho in RHOS:
        vals = []
        for model in MODELS:
            df = data.get(dataset, {}).get(dim, {}).get(rho, {}).get(model)
            if df is None or metric not in df.columns:
                continue
            row = df[df["explainer"] == explainer]
            if row.empty:
                continue
            v = pd.to_numeric(row[metric].values[0], errors="coerce")
            if not np.isnan(v):
                vals.append(float(v))
        if vals:
            xs.append(rho)
            means.append(np.mean(vals))
            stds.append(np.std(vals))
    return np.array(xs), np.array(means), np.array(stds)


# ── Draw one page ─────────────────────────────────────────────────────────────
def draw_page(dim, data, output_dir):
    n_rows = len(KEY_METRICS)
    n_cols = len(DATASETS)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.2 * n_cols, 3.2 * n_rows),
        sharex=True,
        squeeze=False,
    )
    fig.patch.set_facecolor("#FAFAFA")

    # column headers
    for ci, ds in enumerate(DATASETS):
        axes[0][ci].set_title(DATASET_LABELS[ds], fontsize=11,
                              fontweight="bold", pad=8)

    # row y-labels
    for ri, (_, metric_label) in enumerate(KEY_METRICS):
        axes[ri][0].set_ylabel(metric_label, fontsize=9, labelpad=6)

    legend_handles, legend_labels = [], []

    for ri, (metric, _) in enumerate(KEY_METRICS):
        for ci, dataset in enumerate(DATASETS):
            ax = axes[ri][ci]
            ax.set_facecolor("white")
            ax.grid(True, linewidth=0.5, alpha=0.4, linestyle="--")
            ax.spines[["top", "right"]].set_visible(False)

            plotted = False
            for expl in EXPLAINER_ORDER:
                xs, means, stds = get_series(data, dataset, dim, metric, expl)
                if len(xs) == 0:
                    continue
                sty = EXPL_STYLE[expl]
                lbl = EXPLAINER_LABELS.get(expl, expl)
                line, = ax.plot(
                    xs, means,
                    color=sty["color"],
                    marker=sty["marker"],
                    markersize=5,
                    linewidth=1.8,
                    label=lbl,
                    zorder=sty["zorder"],
                )
                ax.fill_between(
                    xs,
                    means - stds,
                    means + stds,
                    color=sty["color"],
                    alpha=0.10,
                )
                if ri == 0 and ci == 0 and lbl not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(lbl)
                plotted = True

            ax.set_xticks(RHOS)
            ax.set_xticklabels([str(r) for r in RHOS], fontsize=7, rotation=30)
            ax.tick_params(axis="y", labelsize=7)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

            if ri == n_rows - 1:
                ax.set_xlabel("Feature correlation (rho)", fontsize=8)

    # Legend
    fig.legend(
        legend_handles, legend_labels,
        loc="lower center",
        ncol=len(EXPLAINER_ORDER),
        fontsize=9,
        frameon=True,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, -0.02),
        title="Explainer",
        title_fontsize=9,
    )

    fig.suptitle(
        f"XAI-Bench Results  |  dim = {dim}  |  Averaged across LR, DTREE, MLP",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"benchmark_dim={dim}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading CSVs from  : {CSV_DIR}")
    data = load_all(CSV_DIR)
    dims = sorted({d for ds_data in data.values() for d in ds_data})
    print(f"Dims found         : {dims}")

    written = []
    for dim in dims:
        path = draw_page(dim, data, OUTPUT_DIR)
        written.append(path)
        print(f"  [OK]  {path}")

    print(f"\nDone — {len(written)} benchmark figure(s) saved.")


if __name__ == "__main__":
    main()
