import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
# plt.rc('pgf', texsystem='pdflatex')
try:
    plt.style.use('science')
except OSError:
    pass
plt.rcParams["figure.figsize"] = (17, 17)
import os
import pandas as pd
from collections import defaultdict
import numpy as np

mode = "regression"
datasets = {"regression": ["gaussianLinear", "gaussianNonLinearAdditive", "gaussianPiecewiseConstant"]}
metrics = ["faithfulness", "roar_faithfulness", "roar_monotonicity", "monotonicity"]
results_dir = "../results/"
output_dir = f"plots/{mode}/dim_sweep/"
exp_names = ["wed/exp-gaussian-new-1", "wed/exp-gaussian-new-2", "wed/exp-gaussian-new-3"]

# Sweep configuration: matches script.sh
dims = [5, 10, 20, 50, 100]
rhos = [0.0, 0.25, 0.5, 0.75, 0.99]
explainers = ["random", "shap", "shapr", "brutekernelshap", "maple", "lime", "l2x"]
explainer_mappings = {
    "random": "RANDOM",
    "shap": "SHAP",
    "shapr": "SHAPR",
    "brutekernelshap": "BF-SHAP",
    "maple": "MAPLE",
    "lime": "LIME",
    "l2x": "L2X",
}
models = ['LR', 'DTREE', 'MLP']


def collect_data(file_path, dim, scores, model_perfs):
    """Append the per-explainer dataframe and model perf row for one (dim) cell.

    The dim-aware CSV writer in `src/parse_utils.py` prepends three `#`-
    comment header lines (`dataset`, `dim`, `rho`); they do not match any
    model name, so the existing line scan skips them naturally.
    """
    if not os.path.exists(file_path):
        return
    lines = open(file_path, 'r').readlines()
    for idx, line in enumerate(lines):
        first = line.strip().split(",")[0]
        if first in models:
            model = first
            model_perfs[model][dim] = line.strip().split(",")[1:]
            df = pd.read_csv(file_path, skiprows=idx + 1, nrows=len(explainers))
            df.columns.values[0] = "explainer"
            scores[model][dim].append(df)


for rho in rhos:
    for metric in metrics:
        num_models, num_datasets = len(models), len(datasets[mode])
        fig, axs = plt.subplots(
            figsize=(5.0 * num_models, 4.0 * num_datasets),
            nrows=num_datasets,
            ncols=num_models,
        )

        lines, labels = [], []
        for data_idx, dataset in enumerate(datasets[mode]):
            scores = {model: defaultdict(list) for model in models}
            model_perfs = {model: defaultdict(list) for model in models}

            for dim in dims:
                for exp_name in exp_names:
                    file_path = os.path.join(
                        results_dir,
                        mode,
                        exp_name,
                        "more_csv/csv/",
                        f"{dataset}_dim={dim}_rho={rho}.csv",
                    )
                    collect_data(file_path, dim, scores, model_perfs)

            for idx, model in enumerate(models):
                ax = axs[data_idx][idx] if num_datasets > 1 else axs[idx]
                for explainer in explainers:
                    values = [[] for _ in range(len(exp_names))]
                    for exp in range(len(exp_names)):
                        per_dim = []
                        for d in dims:
                            runs = scores[model][d]
                            if exp < len(runs):
                                per_dim.append(
                                    runs[exp].query(f'explainer=="{explainer}"')[metric]
                                )
                            else:
                                per_dim.append(np.nan)
                        values[exp] = per_dim
                    values = np.squeeze(np.array(values, dtype=float))
                    if values.ndim == 1:
                        means = values
                        mins = values
                        maxs = values
                    else:
                        means = np.nanmean(values, axis=0)
                        mins = np.nanmin(values, axis=0)
                        maxs = np.nanmax(values, axis=0)
                    ax.plot(dims, means, linewidth=2, label=explainer)
                    ax.fill_between(dims, mins, maxs, alpha=0.2)
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.set_xlabel("D (number of features)")
                ax.set_xticks(dims)
                if idx == 0:
                    ax.set_ylabel(dataset, fontsize=14)
                if data_idx == 0:
                    ax.set_title(model)
                if model == "DTREE":
                    lines, labels = ax.get_legend_handles_labels()

        fig.suptitle(f"{metric} vs D (rho={rho})", fontsize=16)
        fig.subplots_adjust(bottom=0.12)
        if lines:
            fig.legend(
                lines,
                [explainer_mappings.get(a, a) for a in labels],
                frameon=True,
                shadow=True,
                loc="lower center",
                bbox_to_anchor=(0.43, 0),
                ncol=len(explainers),
                fontsize=14,
            )
        plt_save_path = os.path.join(output_dir, f"{metric}_rho={rho}.pdf")
        if not os.path.exists(os.path.dirname(plt_save_path)):
            os.makedirs(os.path.dirname(plt_save_path))
        plt.savefig(plt_save_path, bbox_inches='tight')
        plt.clf()
        plt.cla()
        plt.close(fig)
