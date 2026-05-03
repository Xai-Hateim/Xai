"""
run_experiments_A.py  -  YOUR MACHINE  (i7-10700K, 7 workers)
Runs d=50 experiments.
NOTE: shapley and shapley_corr metrics are excluded at d=50
because XAI-Bench ground-truth computation requires 2^d memory,
which is physically impossible above d~25.
Metrics computed: faithfulness, roar_faithfulness, roar_monotonicity,
                  monotonicity, infidelity
"""
import json, subprocess, sys, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed

N_WORKERS   = 7
RESULTS_DIR = "results"
CSV_DIR     = os.path.join(RESULTS_DIR, "csv")
CFG_DIR     = "configs"

EXPLAINERS_D50 = [
    {"name": "random",    "expl_kwargs": {}},
    {"name": "breakdown", "expl_kwargs": {}},
    {"name": "shap",      "expl_kwargs": {}},
    {"name": "maple",     "expl_kwargs": {"n_estimators": 200, "min_samples_leaf": 10}},
    {"name": "lime",      "expl_kwargs": {"kernel_width": 0.5}},
]
EXPLAINERS_SHAPR = [
    {"name": "shapr", "expl_kwargs": {"sigma": 0.4}},
]
MODELS = [
    {"name":"lr",    "model_kwargs":{}},
    {"name":"dtree", "model_kwargs":{}},
    {"name":"mlp",   "model_kwargs":{}},
]

# Full metrics for d=5 (SHAPR — ground truth tractable)
METRICS_D5 = [
    "roar_faithfulness","roar_monotonicity",
    "faithfulness","monotonicity",
    "shapley","shapley_corr","infidelity",
]
# Reduced metrics for d=50 (no GT-Shapley — 2^50 is physically impossible)
METRICS_D50 = [
    "roar_faithfulness","roar_monotonicity",
    "faithfulness","monotonicity","infidelity",
]

DATASETS = ["gaussianLinear","gaussianNonLinearAdditive","gaussianPiecewiseConstant"]
RHOS     = [0.0, 0.25, 0.5, 0.75, 0.99]

def make_weight(dim):
    return [4,3,2,1,0] + [0]*(dim-5)

def already_done(dataset, dim, rho, suffix):
    for s in [f"_{suffix}", ""]:
        p = os.path.join(CSV_DIR, f"{dataset}_dim={dim}_rho={rho}{s}.csv")
        if os.path.exists(p): return True
    return False

def build_experiments():
    exps = []
    # d=50: all fast methods, no GT-Shapley
    for ds in DATASETS:
        for rho in RHOS:
            if not already_done(ds, 50, rho, "noSHAPR"):
                exps.append({
                    "tag":       f"{ds}_dim=50_rho={rho}_noSHAPR",
                    "dataset":   ds, "dim": 50, "rho": rho,
                    "weight":    make_weight(50),
                    "explainers":EXPLAINERS_D50,
                    "metrics":   METRICS_D50,
                })
            else:
                print(f"  SKIP: {ds}_dim=50_rho={rho}")

    # SHAPR d=5 (GT-Shapley tractable at d=5)
    for ds in DATASETS:
        for rho in RHOS:
            if not already_done(ds, 5, rho, "SHAPRonly"):
                exps.append({
                    "tag":       f"{ds}_dim=5_rho={rho}_SHAPRonly",
                    "dataset":   ds, "dim": 5, "rho": rho,
                    "weight":    make_weight(5),
                    "explainers":EXPLAINERS_SHAPR,
                    "metrics":   METRICS_D5,
                })
            else:
                print(f"  SKIP: {ds}_dim=5_rho={rho}_SHAPRonly")
    return exps

def write_config(exp, worker_id):
    cfg = {
        "dataset": {
            "name": exp["dataset"],
            "data_kwargs": {
                "mu":               f"np.zeros({exp['dim']})",
                "dim":               exp["dim"],
                "rho":               exp["rho"],
                "weight":           f"np.array({exp['weight']})",
                "noise":             0.01,
                "num_train_samples": 1000,
                "num_val_samples":   100,
            }
        },
        "models":     MODELS,
        "explainers": exp["explainers"],
        "metrics":    exp["metrics"],
        "conditional":"observational",
    }
    path = os.path.join(CFG_DIR, f"_tmp_worker{worker_id}.jsonc")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)
    return path

def run_one(exp, worker_id):
    csv_path = os.path.join(CSV_DIR, f"{exp['tag']}.csv")
    if os.path.exists(csv_path):
        return exp["tag"], "SKIPPED", 0.0
    cfg_path = write_config(exp, worker_id)
    cmd = [sys.executable, "main_driver.py",
           "--mode", "regression", "--seed", "7",
           "--experiment", "--experiment-json", cfg_path,
           "--results-dir", RESULTS_DIR]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = (time.time() - t0) / 60
    if result.returncode == 0:
        return exp["tag"], "OK", elapsed
    else:
        print(f"\n[FAIL] {exp['tag']}\n{result.stderr[-600:]}")
        return exp["tag"], "FAILED", elapsed

def main():
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(CFG_DIR, exist_ok=True)
    print("\nChecking which experiments are already done...")
    experiments = build_experiments()
    total = len(experiments)
    if total == 0:
        print("All experiments already done!")
        return
    print(f"\nRunning {total} experiments on {N_WORKERS} parallel workers")
    print(f"Note: GT-Shapley excluded at d=50 (2^50 allocation impossible)")
    print(f"Results -> {CSV_DIR}/\n{'='*60}")
    done, success, failed = 0, 0, []
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(run_one, exp, i % N_WORKERS): exp
                   for i, exp in enumerate(experiments)}
        for future in as_completed(futures):
            tag, status, mins = future.result()
            done += 1
            elapsed_total = (time.time() - t_start) / 60
            avg = elapsed_total / max(done, 1)
            rem = avg * (total - done)
            if status == "OK":
                success += 1
                print(f"  +[{done:2d}/{total}] {tag}  ({mins:.1f}m) | rem ~{rem:.0f}m")
            elif status == "SKIPPED":
                success += 1
                print(f"  -[{done:2d}/{total}] {tag}  (skipped)")
            else:
                failed.append(tag)
                print(f"  ![{done:2d}/{total}] {tag}  FAILED")
    print(f"\n{'='*60}")
    print(f"DONE: {success}/{total} | {(time.time()-t_start)/60:.0f} min total")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print(f"\nNext: zip results/csv/ and upload for graphs")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
