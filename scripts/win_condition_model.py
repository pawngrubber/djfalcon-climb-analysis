# /// script
# requires-python = ">=3.11"
# dependencies = ["scikit-learn", "scipy", "numpy"]
# ///
"""Honest P(win) model from djfalcon's own in-game stats.

L2 logistic regression on 97 ranked-solo (queue 420, non-remake) player-games,
standardized features, leave-one-out AUC, coefficients with 1000-bootstrap 95%
CIs (fixed seed), plus univariate rank-biserial + Mann-Whitney U per feature for
comparison. Also reports a correlation matrix / VIF to flag collinearity.

CAVEAT: these are correlations WITHIN his games, not causal levers. Outcome-laden
features (dragons, turret kills, gold) partly encode "the team was already
winning" rather than a knob he can independently turn.

Run: uv run scripts/win_condition_model.py
"""
import json
import os

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score

SEED = 12345
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "riot_games.json")

FEATURES = [
    "deaths_per_min",
    "cs_per_min",
    "kill_participation",
    "vision_per_min",
    "gold_per_min",
    "dmg_per_min",
    "turret_kills",
    "objectives_dragon",
    "first_blood",
    "duration_min",
]


def load():
    games = json.load(open(DATA))
    rows, y = [], []
    for g in games:
        if g["queue_id"] != 420 or g["remake"]:
            continue
        dm = g["duration_min"]
        rows.append([
            g["death"] / dm,
            g["cs_per_min"],
            g["kill_participation"] or 0.0,
            g["vision_score"] / dm,
            g["gold"] / dm,
            g["dmg_to_champs"] / dm,
            g["turret_kills"],
            g["objectives_dragon"],
            1.0 if g["first_blood"] else 0.0,
            dm,
        ])
        y.append(1 if g["win"] else 0)
    return np.array(rows, dtype=float), np.array(y, dtype=int)


def rank_biserial(a, b):
    """Rank-biserial effect size from MWU (a=wins, b=losses).

    r = 2*U1/(n1*n2) - 1; positive => feature higher in wins.
    """
    n1, n2 = len(a), len(b)
    U1, p = mannwhitneyu(a, b, alternative="two-sided")
    r = 2.0 * U1 / (n1 * n2) - 1.0
    return r, p


def vif(Xz):
    """Variance inflation factors on standardized features."""
    n, k = Xz.shape
    vifs = []
    for j in range(k):
        others = np.delete(Xz, j, axis=1)
        A = np.column_stack([np.ones(n), others])
        # regress feature j on the rest
        beta, *_ = np.linalg.lstsq(A, Xz[:, j], rcond=None)
        pred = A @ beta
        ss_res = np.sum((Xz[:, j] - pred) ** 2)
        ss_tot = np.sum((Xz[:, j] - Xz[:, j].mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        vifs.append(1.0 / (1.0 - r2) if r2 < 1 else np.inf)
    return np.array(vifs)


def main():
    X, y = load()
    n = len(y)
    print(f"n={n} ranked-solo games, wins={y.sum()} ({y.mean():.1%}), losses={n - y.sum()}")
    print()

    scaler = StandardScaler().fit(X)
    Xz = scaler.transform(X)

    # ---- Leave-one-out AUC ----
    C = 1.0
    loo = LeaveOneOut()
    oof = np.zeros(n)
    for tr, te in loo.split(Xz):
        # standardize inside the fold to avoid leakage
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=C, max_iter=5000)
        m.fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    loo_auc = roc_auc_score(y, oof)
    print(f"Leave-one-out AUC = {loo_auc:.3f}  (L2 logistic, C={C})")
    print()

    # ---- Full-data fit for coefficients ----
    full = LogisticRegression(C=C, max_iter=5000).fit(Xz, y)
    coef = full.coef_[0]

    # ---- Bootstrap CIs ----
    rng = np.random.default_rng(SEED)
    B = 1000
    boot = np.zeros((B, len(FEATURES)))
    for b in range(B):
        idx = rng.integers(0, n, n)
        Xb, yb = X[idx], y[idx]
        if yb.sum() == 0 or yb.sum() == n:
            boot[b] = np.nan
            continue
        sc = StandardScaler().fit(Xb)
        mb = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(Xb), yb)
        boot[b] = mb.coef_[0]
    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)

    # ---- Univariate rank-biserial + MWU ----
    print("Per-feature (standardized units):")
    print(f"{'feature':<20} {'coef':>7} {'95% CI':>18}   {'rank-bis':>8} {'MWU p':>8}")
    print("-" * 72)
    rows = []
    for j, f in enumerate(FEATURES):
        a = X[y == 1, j]
        bvals = X[y == 0, j]
        rb, p = rank_biserial(a, bvals)
        rows.append((f, coef[j], lo[j], hi[j], rb, p))
    # sort by coefficient descending for readability
    for f, c, l, h, rb, p in sorted(rows, key=lambda r: r[1], reverse=True):
        sig = "*" if (l > 0 or h < 0) else " "
        print(f"{f:<20} {c:>7.3f} [{l:>7.3f},{h:>7.3f}]{sig}  {rb:>8.3f} {p:>8.3f}")
    print("  (* = bootstrap 95% CI excludes 0)")
    print()

    # ---- Collinearity ----
    vifs = vif(Xz)
    print("VIF (standardized features):")
    for f, v in sorted(zip(FEATURES, vifs), key=lambda t: -t[1]):
        flag = "  <-- collinear" if v > 5 else ""
        print(f"  {f:<20} {v:>6.2f}{flag}")
    print()
    corr = np.corrcoef(Xz, rowvar=False)
    print("Notable feature correlations (|r| > 0.5):")
    for i in range(len(FEATURES)):
        for j in range(i + 1, len(FEATURES)):
            if abs(corr[i, j]) > 0.5:
                print(f"  {FEATURES[i]} ~ {FEATURES[j]}: r={corr[i, j]:+.2f}")
    print()
    print("CAVEAT: correlational within djfalcon's games, not causal. Outcome-laden")
    print("features (dragons, turret kills, gold/min) partly encode 'team was winning'.")


if __name__ == "__main__":
    main()
