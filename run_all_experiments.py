"""
run_all_experiments.py
======================
Runs xai-bench for all dataset × rho combinations needed for the paper.
Just run:  python run_all_experiments.py
Let it run overnight — it will produce all 15 CSVs in results/csv/

Total estimated time: 8-15 hours depending on your machine.
"""

import json, subprocess, sys, os, time, copy

# ── Base config (matches your exact xai-bench format) ────────────────────────
BASE_CONFIG = {
    "models": [
        {"name": "lr",    "model_kwargs": {}},
        {"name": "dtree", "model_kwargs": {}},
        {"name": "mlp",   "model_kwargs": {}}
    ],
    "explainers": [
        {"name": "random",    "expl_kwargs": {}},
        {"name": "breakdown", "expl_kwargs": {}},
        {"name": "shap",      "expl_kwargs": {}},
        {"name": "shapr",     "expl_kwargs": {"sigma": 0.4}},
        {"name": "maple",     "expl_kwargs": {"n_estimators": 200, "min_samples_leaf": 10}},
        {"name": "lime",      "expl_kwargs": {"kernel_width": 0.5}}
    ],
    "metrics": [
        "roar_faithfulness", "roar_monotonicity",
        "faithfulness", "monotonicity",
        "shapley", "shapley_corr", "infidelity"
    ],
    "conditional": "observational"
}

# ── All experiments needed for the paper ─────────────────────────────────────
EXPERIMENTS = []

for dataset, dim, weight in [
    ("gaussianLinear",            10, [4,3,2,1,0,0,0,0,0,0]),
    ("gaussianNonLinearAdditive", 10, [4,3,2,1,0,0,0,0,0,0]),
    ("gaussianPiecewiseConstant",  5, [4,3,2,1,0]),
]:
    for rho in [0.0, 0.25, 0.5, 0.75, 0.99]:
        EXPERIMENTS.append({
            "dataset": dataset,
            "dim":     dim,
            "rho":     rho,
            "weight":  weight,
        })

# ─────────────────────────────────────────────────────────────────────────────

def make_config(exp):
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["dataset"] = {
        "name": exp["dataset"],
        "data_kwargs": {
            "mu":               f"np.zeros({exp['dim']})",
            "dim":               exp["dim"],
            "rho":               exp["rho"],
            "weight":           f"np.array({exp['weight']})",
            "noise":             0.01,
            "num_train_samples": 1000,
            "num_val_samples":   100
        }
    }
    return cfg


def run_experiment(exp, idx, total):
    tag = f"{exp['dataset']}_dim={exp['dim']}_rho={exp['rho']}"
    print(f"\n{'='*65}")
    print(f"  [{idx}/{total}]  {tag}")
    print(f"{'='*65}")

    # Check if already done (skip if CSV exists)
    csv_path = os.path.join("results", "csv", f"{tag}.csv")
    if os.path.exists(csv_path):
        print(f"  Already exists — skipping.")
        return True

    # Write temp config
    cfg = make_config(exp)
    tmp_cfg = "configs/_tmp_experiment.jsonc"
    with open(tmp_cfg, "w") as f:
        # Write as JSON (jsonc comments not needed for programmatic use)
        json.dump(cfg, f, indent=4)

    # Run benchmark
    cmd = [
        sys.executable, "main_driver.py",
        "--mode",            "regression",
        "--seed",            "7",
        "--experiment",
        "--experiment-json", tmp_cfg,
        "--results-dir",     "results"
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = (time.time() - t0) / 60

    if result.returncode == 0:
        print(f"  Done in {elapsed:.1f} min")
        return True
    else:
        print(f"  FAILED (exit code {result.returncode})")
        return False


def main():
    os.makedirs("results/csv", exist_ok=True)
    os.makedirs("configs",     exist_ok=True)

    total   = len(EXPERIMENTS)
    success = 0
    failed  = []

    print(f"\nStarting {total} experiments for paper figures.")
    print(f"Results will be saved to: results/csv/")
    print(f"You can safely interrupt and restart — completed runs are skipped.\n")

    overall_start = time.time()

    for i, exp in enumerate(EXPERIMENTS, 1):
        ok = run_experiment(exp, i, total)
        if ok:
            success += 1
        else:
            failed.append(f"{exp['dataset']}_rho={exp['rho']}")

        # Estimated time remaining
        elapsed_total = (time.time() - overall_start) / 60
        avg_per_run   = elapsed_total / i
        remaining     = avg_per_run * (total - i)
        print(f"  Progress: {i}/{total} | "
              f"Elapsed: {elapsed_total:.0f}m | "
              f"Est. remaining: {remaining:.0f}m")

    print(f"\n{'='*65}")
    print(f"FINISHED:  {success}/{total} successful")
    if failed:
        print(f"Failed runs: {', '.join(failed)}")
    print(f"Total time: {(time.time()-overall_start)/60:.0f} minutes")
    print(f"\nNow run: python plot_xaibench.py --results-dir results/csv/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
