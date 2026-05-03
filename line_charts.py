"""
line_charts.py
==============
Generates line charts from all benchmark CSVs in results/csv/.

For each metric it produces one figure:
  rows    = datasets  (gaussianLinear, gaussianNonLinearAdditive, gaussianPiecewiseConstant)
  columns = models    (LR, DTREE, MLP)
  x-axis  = rho       (0.0 → 0.99)
  lines   = explainers (one coloured line per method)
  panels  = one per dim value found (separate PNG per dim)

Output: results/plots/linecharts/<metric>_dim=<D>.png
"""

import os
import io
import glob
import re
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ─────────────────────────────────────────────────────────────────────
CSV_DIR    = "results/csv"
OUTPUT_DIR = "results/plots/linecharts"

# ── Ordered display names ─────────────────────────────────────────────────────
EXPLAINER_ORDER = ["random", "breakdown", "shap", "brutekernelshap",
                   "shapr", "maple", "lime"]
EXPLAINER_LABELS = {
    "random":         "RANDOM",
    "breakdown":      "BreakDown",
    "shap":           "SHAP",
    "brutekernelshap":"BF-SHAP",
    "shapr":          "SHAPR",
    "maple":          "MAPLE",
    "lime":           "LIME",
}
COLORS = plt.cm.tab10(np.linspace(0, 1, len(EXPLAINER_ORDER)))
EXPL_COLOR = {e: COLORS[i] for i, e in enumerate(EXPLAINER_ORDER)}

DATASETS = [
    "gaussianLinear",
    "gaussianNonLinearAdditive",
    "gaussianPiecewiseConstant",
]
DATASET_LABELS = {
    "gaussianLinear":            "Gaussian Linear",
    "gaussianNonLinearAdditive": "Gaussian NonLinear Additive",
    "gaussianPiecewiseConstant": "Gaussian Piecewise Const.",
}
MODELS = ["LR", "DTREE", "MLP"]
RHOS   = [0.0, 0.25, 0.5, 0.75, 0.99]


# ── CSV parser ────────────────────────────────────────────────────────────────
def parse_csv(path):
    """Return (meta, {model: DataFrame}) from one benchmark CSV."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

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
        parts = lines[i].strip().split(",")
        if not parts or not parts[0].strip():
            i += 1
            continue
        model_name = parts[0].strip()
        i += 1
        chunk = []
        while i < len(lines) and lines[i].strip():
            chunk.append(lines[i])
            i += 1
        if not chunk:
            continue
        df = pd.read_csv(io.StringIO("".join(chunk)))
        c0 = df.columns[0]
        if str(c0).startswith("Unnamed") or c0 == "":
            df = df.rename(columns={c0: "explainer"})
        else:
            df = df.rename(columns={c0: "explainer"})
        df = df.dropna(subset=["explainer"])
        df["explainer"] = df["explainer"].astype(str).str.strip()
        blocks[model_name] = df

    return meta, blocks


# ── Load all CSVs into a nested dict ─────────────────────────────────────────
# data[dataset][dim][rho][model] = DataFrame
def load_all(csv_dir):
    data = collections.defaultdict(
        lambda: collections.defaultdict(
            lambda: collections.defaultdict(dict)
        )
    )
    metrics_found = set()
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

    for path in csv_files:
        meta, blocks = parse_csv(path)
        dataset = meta.get("dataset", "")
        dim     = int(meta.get("dim", 0))
        rho     = float(meta.get("rho", 0))
        if not dataset or not blocks:
            continue
        for model, df in blocks.items():
            data[dataset][dim][rho][model] = df
            for c in df.columns:
                if c != "explainer":
                    metrics_found.add(c)

    return data, sorted(metrics_found)


# ── Plot one figure per (metric, dim) ─────────────────────────────────────────
def make_figure(metric, dim, data, output_dir):
    n_rows = len(DATASETS)
    n_cols = len(MODELS)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.5 * n_rows),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle(f"{metric}   (dim = {dim})", fontsize=13, fontweight="bold", y=1.01)

    legend_handles = {}

    for ri, dataset in enumerate(DATASETS):
        for ci, model in enumerate(MODELS):
            ax = axes[ri][ci]

            # collect all explainers present in this panel
            explainers_here = set()
            for rho in RHOS:
                df = data.get(dataset, {}).get(dim, {}).get(rho, {}).get(model)
                if df is not None and metric in df.columns:
                    explainers_here.update(df["explainer"].tolist())

            # plot in fixed order (keeps colours consistent)
            ordered = [e for e in EXPLAINER_ORDER if e in explainers_here]
            ordered += sorted(explainers_here - set(EXPLAINER_ORDER))

            for expl in ordered:
                ys = []
                xs = []
                for rho in RHOS:
                    df = data.get(dataset, {}).get(dim, {}).get(rho, {}).get(model)
                    if df is None or metric not in df.columns:
                        continue
                    row = df[df["explainer"] == expl]
                    if row.empty:
                        continue
                    val = pd.to_numeric(row[metric].values[0], errors="coerce")
                    xs.append(rho)
                    ys.append(float(val) if not np.isnan(float(val)) else np.nan)

                if not xs:
                    continue
                color = EXPL_COLOR.get(expl, "grey")
                label = EXPLAINER_LABELS.get(expl, expl)
                line, = ax.plot(
                    xs, ys,
                    marker="o", markersize=4,
                    linewidth=1.6,
                    color=color,
                    label=label,
                )
                if label not in legend_handles:
                    legend_handles[label] = line

            ax.set_xticks(RHOS)
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
            ax.tick_params(axis="x", labelrotation=45, labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
            ax.grid(True, linewidth=0.4, alpha=0.5)

            if ri == 0:
                ax.set_title(model, fontsize=10, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(DATASET_LABELS.get(dataset, dataset), fontsize=8)
            if ri == n_rows - 1:
                ax.set_xlabel("rho", fontsize=8)

    # shared legend below figure
    if legend_handles:
        fig.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc="lower center",
            ncol=min(len(legend_handles), 7),
            fontsize=8,
            frameon=True,
            bbox_to_anchor=(0.5, -0.04),
        )

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{metric}_dim={dim}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading CSVs from {CSV_DIR} ...")
    data, metrics = load_all(CSV_DIR)

    dims = sorted({
        d
        for ds_data in data.values()
        for d in ds_data.keys()
    })

    print(f"  Datasets : {sorted(data.keys())}")
    print(f"  Dims     : {dims}")
    print(f"  Metrics  : {metrics}")
    print(f"  Output   : {OUTPUT_DIR}")
    print()

    written = []
    for metric in metrics:
        for dim in dims:
            path = make_figure(metric, dim, data, OUTPUT_DIR)
            written.append(path)
            print(f"  [OK]  {os.path.basename(path)}")

    print(f"\nWrote {len(written)} figures to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
