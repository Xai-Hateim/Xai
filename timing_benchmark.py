"""
timing_benchmark.py
===================
Measures wall-clock execution time per explainer AND
shows impact of increasing dimensionality (d=5, 10, 50).

Usage:
    .venv\Scripts\python.exe timing_benchmark.py

Output:
    timing_results.csv     — raw times
    timing_summary.txt     — formatted table for the paper
    timing_dim_scaling.csv — d=5 vs d=10 vs d=50 per method
"""

import time, math, warnings
import numpy as np
import pandas as pd
from itertools import combinations
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

# ── Data generator ────────────────────────────────────────────────────────────
def make_data(dim, rho=0.5, n_train=1000, n_test=100, seed=42):
    rng = np.random.default_rng(seed)
    cov = np.full((dim, dim), rho); np.fill_diagonal(cov, 1.0)
    X = rng.multivariate_normal(np.zeros(dim), cov, n_train + n_test)
    w = np.array([4,3,2,1,0] + [0]*(dim-5))
    y = X @ w + rng.normal(0, 0.1, n_train + n_test)
    y = (y - y.mean()) / y.std()
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]

# ── Explainer timers ──────────────────────────────────────────────────────────
def time_shap(model, X_train, X_test):
    import shap
    bg = shap.sample(X_train, 50)
    exp = shap.KernelExplainer(model.predict, bg)
    t0 = time.time()
    exp.shap_values(X_test, nsamples=200, silent=True)
    return time.time() - t0

def time_bfshap(model, X_train, X_test):
    import shap
    bg = shap.sample(X_train, 50)
    exp = shap.KernelExplainer(model.predict, bg)
    t0 = time.time()
    n = min(2**X_test.shape[1], 1024)
    exp.shap_values(X_test, nsamples=n, silent=True)
    return time.time() - t0

def time_lime(model, X_train, X_test):
    from lime.lime_tabular import LimeTabularExplainer
    exp = LimeTabularExplainer(
        X_train, mode="regression",
        feature_names=[f"x{i}" for i in range(X_train.shape[1])],
        discretize_continuous=False, kernel_width=0.5)
    t0 = time.time()
    for x in X_test:
        exp.explain_instance(x, model.predict,
                             num_features=X_train.shape[1],
                             num_samples=500)
    return time.time() - t0

def time_maple(model, X_train, y_train, X_test):
    y_soft = model.predict(X_train)
    rf = RandomForestRegressor(n_estimators=200, max_depth=5,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_soft)
    train_leaves = rf.apply(X_train)
    t0 = time.time()
    for x in X_test:
        x_leaves = rf.apply(x.reshape(1,-1))
        w = (train_leaves == x_leaves).mean(axis=1)
        W = np.diag(w)
        Z = X_train
        ZtW = Z.T @ W
        np.linalg.solve(ZtW @ Z + 0.001*np.eye(Z.shape[1]), ZtW @ y_soft)
    return time.time() - t0

def time_breakdown(model, X_train, X_test):
    t0 = time.time()
    baseline = model.predict(np.zeros((1, X_train.shape[1]))).item()
    for x in X_test:
        x_cur = np.zeros(X_train.shape[1])
        order = np.argsort(np.abs(x))[::-1]
        f_prev = baseline
        for j in order:
            x_cur[j] = x[j]
            f_new = model.predict(x_cur.reshape(1,-1)).item()
            f_prev = f_new
    return time.time() - t0

def time_shapr(model, X_train, X_test, dim, n_mc=50):
    cov = np.cov(X_train.T)
    mu  = X_train.mean(axis=0)
    all_f = list(range(dim))

    def cond_exp(xs, S, Sb):
        if not Sb:
            return model.predict(xs.reshape(1,-1)).item()
        if not S:
            Xmc = np.random.multivariate_normal(mu, cov, n_mc)
            return model.predict(Xmc).mean()
        S, Sb = np.array(S), np.array(Sb)
        Sbb = cov[np.ix_(Sb,Sb)]; Sbs = cov[np.ix_(Sb,S)]
        Sss = cov[np.ix_(S,S)];   Sinv = np.linalg.pinv(Sss)
        muc = mu[Sb] + Sbs@Sinv@(xs[S]-mu[S])
        Sc  = Sbb - Sbs@Sinv@Sbs.T
        Sc  = (Sc+Sc.T)/2 + 1e-8*np.eye(len(Sb))
        samp = np.random.multivariate_normal(muc, Sc, n_mc)
        Xmc = np.zeros((n_mc, dim))
        Xmc[:,S]=xs[S]; Xmc[:,Sb]=samp
        return model.predict(Xmc).mean()

    t0 = time.time()
    for xs in X_test:
        phi = np.zeros(dim)
        for j in range(dim):
            S_wo = [k for k in all_f if k!=j]
            for sz in range(dim):
                for S in combinations(S_wo, sz):
                    S = list(S)
                    Sb_j = [k for k in all_f if k not in S+[j]]
                    Sb   = [k for k in all_f if k not in S]
                    w = (math.factorial(len(S))*math.factorial(dim-len(S)-1)
                         /math.factorial(dim))
                    phi[j] += w*(cond_exp(xs,S+[j],Sb_j)-cond_exp(xs,S,Sb))
    return time.time() - t0

# ── Models ────────────────────────────────────────────────────────────────────
MODEL_CLASSES = {
    "LR":    lambda: LinearRegression(),
    "DTREE": lambda: DecisionTreeRegressor(max_depth=5, random_state=42),
    "MLP":   lambda: MLPRegressor(hidden_layer_sizes=(100,50),
                                   max_iter=500, random_state=42),
}

# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — Per-model timing at d=10 (all fast methods)
# ═════════════════════════════════════════════════════════════════════════════
results = []
RHO = 0.5

print("\n" + "="*60)
print("PART 1 — Per-model timing at d=10, rho=0.5, n_test=100")
print("="*60)

X_tr, y_tr, X_te, y_te = make_data(10, RHO)

for mname, mcls in MODEL_CLASSES.items():
    model = mcls(); model.fit(X_tr, y_tr)
    print(f"\n  Model: {mname}")

    for ename, efn in [
        ("SHAP",      lambda m=model: time_shap(m, X_tr, X_te)),
        ("BF-SHAP",   lambda m=model: time_bfshap(m, X_tr, X_te)),
        ("LIME",      lambda m=model: time_lime(m, X_tr, X_te)),
        ("MAPLE",     lambda m=model: time_maple(m, X_tr, y_tr, X_te)),
        ("Breakdown", lambda m=model: time_breakdown(m, X_tr, X_te)),
    ]:
        try:
            t = efn()
            p = t/len(X_te)*100
            print(f"    {ename:12s}  {p:8.2f}s / 100 instances")
            results.append({"method":ename,"model":mname,"dim":10,
                            "rho":RHO,"per_100_s":round(p,3)})
        except Exception as e:
            print(f"    {ename:12s}  ERROR: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — Dimensionality scaling: d=5, 10, 50 on MLP
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PART 2 — Dimensionality scaling (MLP, rho=0.5, n_test=100)")
print("="*60)

dim_results = []

for dim in [5, 10, 50]:
    X_tr_d, y_tr_d, X_te_d, y_te_d = make_data(dim, RHO)
    mlp = MLPRegressor(hidden_layer_sizes=(100,50), max_iter=500, random_state=42)
    mlp.fit(X_tr_d, y_tr_d)
    print(f"\n  d={dim}")

    for ename, efn in [
        ("SHAP",      lambda m=mlp,Xtr=X_tr_d,Xte=X_te_d:
                      time_shap(m,Xtr,Xte)),
        ("BF-SHAP",   lambda m=mlp,Xtr=X_tr_d,Xte=X_te_d:
                      time_bfshap(m,Xtr,Xte) if dim<=10 else None),
        ("LIME",      lambda m=mlp,Xtr=X_tr_d,Xte=X_te_d:
                      time_lime(m,Xtr,Xte)),
        ("MAPLE",     lambda m=mlp,Xtr=X_tr_d,ytr=y_tr_d,Xte=X_te_d:
                      time_maple(m,Xtr,ytr,Xte)),
        ("Breakdown", lambda m=mlp,Xtr=X_tr_d,Xte=X_te_d:
                      time_breakdown(m,Xtr,Xte)),
    ]:
        if dim > 10 and ename == "BF-SHAP":
            print(f"    {'BF-SHAP':12s}  SKIPPED (2^{dim} coalitions intractable)")
            dim_results.append({"method":ename,"dim":dim,"per_100_s":"N/A (intractable)"})
            continue
        try:
            t = efn()
            if t is None:
                continue
            p = t/len(X_te_d)*100
            print(f"    {ename:12s}  {p:8.2f}s / 100 instances")
            dim_results.append({"method":ename,"dim":dim,"per_100_s":round(p,3)})
        except Exception as e:
            print(f"    {ename:12s}  ERROR: {e}")

# SHAPR at d=5 only (10 instances, extrapolated)
print(f"\n  SHAPR at d=5 only (10 test instances, extrapolated to 100)")
X_tr5, y_tr5, X_te5, _ = make_data(5, RHO, n_test=10)
mlp5 = MLPRegressor(hidden_layer_sizes=(100,50), max_iter=500, random_state=42)
mlp5.fit(X_tr5, y_tr5)
try:
    t5 = time_shapr(mlp5, X_tr5, X_te5, 5)
    p5 = t5/10*100
    print(f"    {'SHAPR':12s}  {p5:8.2f}s / 100 instances (extrapolated)")
    dim_results.append({"method":"SHAPR","dim":5,"per_100_s":round(p5,3)})
    dim_results.append({"method":"SHAPR","dim":10,"per_100_s":">79200 (observed)"})
    dim_results.append({"method":"SHAPR","dim":50,"per_100_s":"N/A (intractable)"})
except Exception as e:
    print(f"    SHAPR  ERROR: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═════════════════════════════════════════════════════════════════════════════
df1 = pd.DataFrame(results)
df1.to_csv("timing_results.csv", index=False)

df2 = pd.DataFrame(dim_results)
df2.to_csv("timing_dim_scaling.csv", index=False)

print("\n" + "="*60)
print("SUMMARY — Per 100 instances, averaged across models (d=10)")
print("="*60)
pivot1 = df1.groupby(["method","model"])["per_100_s"].mean().unstack()
print(pivot1.round(2).to_string())

print("\n" + "="*60)
print("DIMENSIONALITY SCALING — MLP only (seconds / 100 instances)")
print("="*60)
df2_num = df2[pd.to_numeric(df2["per_100_s"],errors="coerce").notna()].copy()
df2_num["per_100_s"] = pd.to_numeric(df2_num["per_100_s"])
pivot2 = df2_num.groupby(["method","dim"])["per_100_s"].mean().unstack()
pivot2.columns = [f"d={c}" for c in pivot2.columns]
print(pivot2.round(2).to_string())

# Non-numeric rows
df2_str = df2[pd.to_numeric(df2["per_100_s"],errors="coerce").isna()]
if not df2_str.empty:
    print("\nIntractable cases:")
    print(df2_str.to_string(index=False))

with open("timing_summary.txt","w") as f:
    f.write("="*60 + "\n")
    f.write("Runtime per 100 test instances (seconds)\n")
    f.write("="*60 + "\n\n")
    f.write("Per-model breakdown at d=10:\n")
    f.write(pivot1.round(2).to_string())
    f.write("\n\nDimensionality scaling (MLP):\n")
    f.write(pivot2.round(2).to_string())
    f.write("\n\nNote: SHAPR intractable at d=10 (>22 hours observed in benchmark).\n")
    f.write("BF-SHAP intractable at d>10 (2^d coalitions required).\n")

print("\nSaved: timing_results.csv")
print("Saved: timing_dim_scaling.csv")
print("Saved: timing_summary.txt")
print("\nUpload timing_results.csv and timing_dim_scaling.csv here when done.")
