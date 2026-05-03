#!/usr/bin/env python3
"""Plot CSV outputs written by `parse_utils.save_results_csv`."""

from __future__ import annotations

import argparse
import glob
import io
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _find_csv_files(results_dir: str) -> list[str]:
    results_dir = os.path.normpath(results_dir)
    patterns = [
        os.path.join(results_dir, "csv", "*.csv"),
        os.path.join(results_dir, "**", "csv", "*.csv"),
    ]
    paths: set[str] = set()
    for p in patterns:
        paths.update(glob.glob(p, recursive=True))
    tag_re = re.compile(r".*_dim=.+_rho=.+.csv$", re.I)
    return sorted(p for p in paths if tag_re.search(os.path.basename(p)))


def _parse_meta_and_blocks(path: str) -> tuple[dict[str, str], list[tuple[str, list[str], pd.DataFrame]]]:
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    meta: dict[str, str] = {}
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

    blocks: list[tuple[str, list[str], pd.DataFrame]] = []
    while i < len(lines):
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break
        parts = lines[i].strip().split(",")
        if not parts or not parts[0]:
            i += 1
            continue
        model_name = parts[0].strip()
        perf = parts[1:]
        i += 1
        chunk: list[str] = []
        while i < len(lines) and lines[i].strip():
            chunk.append(lines[i])
            i += 1
        if not chunk:
            continue
        df = pd.read_csv(io.StringIO("".join(chunk)))
        c0 = df.columns[0]
        if str(c0).startswith("Unnamed") or c0 == "":
            df = df.rename(columns={c0: "explainer"})
        elif "explainer" not in df.columns:
            df = df.rename(columns={c0: "explainer"})
        blocks.append((model_name, perf, df))

    return meta, blocks


def _plot_one_csv(path: str, output_dir: str) -> str | None:
    meta, blocks = _parse_meta_and_blocks(path)
    if not blocks:
        return None

    metric_cols = []
    for _, _, df in blocks:
        for c in df.columns:
            if c != "explainer" and c not in metric_cols:
                metric_cols.append(c)
    if not metric_cols:
        return None

    n_models = len(blocks)
    n_metrics = len(metric_cols)
    fig, axes = plt.subplots(
        n_models,
        n_metrics,
        figsize=(max(4 * n_metrics, 8), max(3 * n_models, 4)),
        squeeze=False,
    )

    ds = meta.get("dataset", "dataset")
    dim = meta.get("dim", "?")
    rho = meta.get("rho", "?")
    fig.suptitle(f"{ds} (dim={dim}, rho={rho})", fontsize=12)

    for mi, (model_name, _perf, df) in enumerate(blocks):
        x = np.arange(len(df))
        for mj, mcol in enumerate(metric_cols):
            ax = axes[mi][mj]
            vals = pd.to_numeric(df[mcol], errors="coerce").fillna(0.0)
            ax.bar(x, vals.values, color=plt.cm.tab10(np.linspace(0, 1, len(df))))
            ax.set_xticks(x)
            ax.set_xticklabels(df["explainer"].astype(str), rotation=35, ha="right", fontsize=8)
            if mi == 0:
                ax.set_title(mcol, fontsize=9)
            if mj == 0:
                ax.set_ylabel(model_name, fontsize=9)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(output_dir, f"{base}_plot.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot benchmark CSV results.")
    ap.add_argument(
        "--results-dir",
        default="results/",
        help="Root directory to search for csv/**/*.csv experiment files.",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Where to write PNGs (default: <results-dir>/plots).",
    )
    args = ap.parse_args()
    results_dir = os.path.abspath(args.results_dir)
    output_dir = args.output_dir or os.path.join(results_dir, "plots")
    output_dir = os.path.abspath(output_dir)

    csv_files = _find_csv_files(results_dir)
    if not csv_files:
        raise SystemExit(
            f"No benchmark CSV files found under {results_dir!r} (expected **/csv/*_dim=*_rho=*.csv). "
            "Run main_driver.py without --no-logs first."
        )

    written: list[str] = []
    for path in csv_files:
        out = _plot_one_csv(path, output_dir)
        if out:
            written.append(out)

    print(f"Wrote {len(written)} figure(s) to {output_dir}:")
    for w in written:
        print(" ", w)


if __name__ == "__main__":
    main()
