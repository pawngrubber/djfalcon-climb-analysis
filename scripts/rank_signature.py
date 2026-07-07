# /// script
# requires-python = ">=3.11"
# dependencies = ["scikit-learn", "scipy", "numpy", "xgboost"]
# ///
"""Rank-signature model: predict Silver(1) vs Iron(0) player-games from behavior.

Pipeline:
  - Family A (controllable levers) vs A+B (adds outcome-contaminated). AUC gap = contamination share.
  - GroupKFold by match_id (never split a match). L2 logistic primary; xgboost optional ceiling.
  - Cluster bootstrap CIs on coefficients (resample MATCHES).
  - Discrete-time hazard for time-to-first-death (person-period logistic, tier HR).
  - Bronze ordinal sanity check (never trained; should land between Iron and Silver).
  - djfalcon application: score his games, lever ranking (std gap x weight) with CIs.
Assumptions are printed inline. Effective N is reported as #matches, not #rows.
"""
import csv, os, json, math, warnings
import numpy as np
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = f'{ROOT}/data/ranksig_features.csv'
RNG = np.random.default_rng(42)

# cspm_0_10 == cs10/10 (exact collinear duplicate) -> dropped to avoid double-counting.
FAM_A = ['cs10', 'cs14', 'cspm_10_20', 'cs10_delta', 'xp10_delta',
         'gold10_delta', 'deaths_pre10', 'first_death_min', 'died',
         'wards_pm_pre20', 'wardkills_pre20', 'cw_pre20', 'first_cw_min',
         'first_leg_min', 'early_kp']
FAM_B = ['b_team_dragons', 'b_team_towers', 'b_kda', 'b_gold_share', 'b_dmg_share',
         'b_kp', 'b_game_min', 'b_win']
ROLES = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']

def load():
    rows = list(csv.DictReader(open(CSV)))
    return rows

def to_float(v):
    if v == '' or v is None:
        return np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan

WINSOR_COLS = ['cs10', 'cs14', 'cs10_delta', 'xp10_delta', 'gold10_delta']

def winsor_bounds(rows, cols=WINSOR_COLS):
    b = {}
    for c in cols:
        col = np.array([to_float(r[c]) for r in rows])
        col = col[np.isfinite(col)]
        b[c] = (np.percentile(col, 1), np.percentile(col, 99))
    return b

def build_matrix(rows, feats, add_roles=True, impute=None, winsor=None):
    """Return X (median-imputed NaN + missingness flags for NaN-prone), colnames,
    groups(match_id). If `impute` (dict col->median) is given, use it (train stats)
    rather than per-subset medians — avoids leakage and all-NaN-column failures.
    Returns medians used as 4th value."""
    nan_prone = ['cspm_10_20', 'cs10_delta', 'xp10_delta', 'gold10_delta',
                 'first_death_min', 'first_cw_min', 'first_leg_min']
    data = {c: np.array([to_float(r[c]) for r in rows]) for c in feats}
    flag_cols = []
    for c in feats:
        if c in nan_prone:
            fc = c + '_miss'
            data[fc] = np.isnan(data[c]).astype(float)
            flag_cols.append(fc)
    med_used = {}
    for c in feats:
        col = data[c]
        if impute is not None and c in impute:
            med = impute[c]
        else:
            med = np.nanmedian(col)
            if not np.isfinite(med):
                med = 0.0
        med_used[c] = med
        filled = np.where(np.isnan(col), med, col)
        if winsor is not None and c in winsor:
            lo, hi = winsor[c]
            filled = np.clip(filled, lo, hi)
        data[c] = filled
    allcols = list(feats) + flag_cols
    if add_roles:
        for role in ROLES:
            data['role_' + role] = np.array([1.0 if r['position'] == role else 0.0 for r in rows])
            allcols.append('role_' + role)
    X = np.column_stack([data[c] for c in allcols])
    groups = np.array([r['match_id'] for r in rows])
    return X, allcols, groups, med_used

def cv_auc(X, y, groups, C=1.0, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    aucs = []
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=2000, penalty='l2')
        clf.fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        if len(set(y[te])) > 1:
            aucs.append(roc_auc_score(y[te], p))
    return np.array(aucs)

def fit_full(X, y):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), y)
    return sc, clf

def cluster_bootstrap_coefs(rows, feats, y, groups, cols, reps=500, impute=None, winsor=None):
    """Resample MATCHES with replacement; refit; collect standardized coefs."""
    match_ids = np.array(sorted(set(groups)))
    idx_by_match = {m: np.where(groups == m)[0] for m in match_ids}
    X, _, _, _ = build_matrix(rows, feats, impute=impute, winsor=winsor)
    coefs = []
    for _ in range(reps):
        samp = RNG.choice(match_ids, size=len(match_ids), replace=True)
        idx = np.concatenate([idx_by_match[m] for m in samp])
        yb = y[idx]
        if len(set(yb)) < 2:
            continue
        sc = StandardScaler().fit(X[idx])
        try:
            clf = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(X[idx]), yb)
            coefs.append(clf.coef_[0])
        except Exception:
            continue
    coefs = np.array(coefs)
    lo, md, hi = np.percentile(coefs, [2.5, 50, 97.5], axis=0)
    return dict(zip(cols, zip(md, lo, hi))), coefs

# ---------------- discrete-time hazard ----------------
def discrete_hazard(rows):
    """Person-period logistic: for each player-game, expand to minute bins 1..K until
    first death (event=1 at death minute) or censor at game end. Covariate: silver(1/0)+roles.
    Report tier hazard ratio = exp(beta_silver)."""
    iron_sil = [r for r in rows if r['tier'] in ('IRON', 'SILVER')]
    KMAX = 20
    Xp, yp, grp = [], [], []
    for r in iron_sil:
        glen = to_float(r['game_min'])
        fd = to_float(r['first_death_min'])
        last = int(min(KMAX, math.floor(glen if not math.isnan(glen) else KMAX)))
        death_bin = int(math.floor(fd)) + 1 if not math.isnan(fd) else None
        sil = 1.0 if r['tier'] == 'SILVER' else 0.0
        role = [1.0 if r['position'] == x else 0.0 for x in ROLES]
        for b in range(1, last + 1):
            event = 1 if (death_bin is not None and b == death_bin) else 0
            # bin dummies (linear time + time^2 for parsimony)
            Xp.append([sil, b, b * b] + role)
            yp.append(event)
            grp.append(r['match_id'])
            if event:
                break
            if death_bin is not None and b >= death_bin:
                break
    Xp = np.array(Xp); yp = np.array(yp); grp = np.array(grp)
    sc = StandardScaler(with_mean=False).fit(Xp)  # keep sil interpretable-ish
    clf = LogisticRegression(C=1e6, max_iter=3000).fit(Xp, yp)  # near-unpenalized for HR
    beta_sil = clf.coef_[0][0]
    # cluster bootstrap over matches for HR CI
    match_ids = np.array(sorted(set(grp)))
    idx_by = {m: np.where(grp == m)[0] for m in match_ids}
    hrs = []
    for _ in range(400):
        samp = RNG.choice(match_ids, size=len(match_ids), replace=True)
        idx = np.concatenate([idx_by[m] for m in samp])
        if len(set(yp[idx])) < 2:
            continue
        try:
            c = LogisticRegression(C=1e6, max_iter=1500).fit(Xp[idx], yp[idx])
            hrs.append(math.exp(c.coef_[0][0]))
        except Exception:
            continue
    lo, hi = np.percentile(hrs, [2.5, 97.5])
    return math.exp(beta_sil), lo, hi, len(iron_sil)

def reliability(y, p, nbins=5):
    order = np.argsort(p)
    out = []
    for chunk in np.array_split(order, nbins):
        out.append((float(np.mean(p[chunk])), float(np.mean(y[chunk])), len(chunk)))
    return out

def main():
    rows = load()
    print('LEGENDARY-based first_leg feature present. rows:', len(rows))
    print('by tier:', Counter(r['tier'] for r in rows))
    mtier = lambda t: len(set(r['match_id'] for r in rows if r['tier'] == t))
    print('matches:', {t: mtier(t) for t in ['IRON', 'SILVER', 'BRONZE', 'DJ']})

    # ---- training set: Iron vs Silver (cohort only) ----
    train = [r for r in rows if r['tier'] in ('IRON', 'SILVER') and r['source'] == 'cohort']
    y = np.array([1 if r['tier'] == 'SILVER' else 0 for r in train])
    print(f'\nTRAIN rows={len(train)} matches={len(set(r["match_id"] for r in train))} '
          f'silver_frac={y.mean():.3f}')

    train_win = winsor_bounds(train)  # [1,99] pct bounds on CS + laning-delta cols (smurf guard)
    Xa, colsA, gA, train_med = build_matrix(train, FAM_A, winsor=train_win)
    Xab, colsAB, gAB, _ = build_matrix(train, FAM_A + FAM_B, winsor=train_win)
    auc_a = cv_auc(Xa, y, gA)
    auc_ab = cv_auc(Xab, y, gAB)
    print(f'\n=== CV AUC (GroupKFold by match) ===')
    print(f'Family A only : {auc_a.mean():.3f} ± {auc_a.std():.3f}  (folds {np.round(auc_a,3)})')
    print(f'Family A+B    : {auc_ab.mean():.3f} ± {auc_ab.std():.3f}  (folds {np.round(auc_ab,3)})')
    gap = auc_ab.mean() - auc_a.mean()
    print(f'CONTAMINATION GAP (A+B − A) = {gap:+.3f}')

    # optional xgboost ceiling
    try:
        from xgboost import XGBClassifier
        gkf = GroupKFold(n_splits=5); xa = []
        for tr, te in gkf.split(Xa, y, gA):
            m = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                              subsample=0.8, eval_metric='logloss', verbosity=0,
                              tree_method='hist', device='cpu', n_jobs=2)
            m.fit(Xa[tr], y[tr])
            if len(set(y[te])) > 1:
                xa.append(roc_auc_score(y[te], m.predict_proba(Xa[te])[:, 1]))
        print(f'Family A xgboost ceiling: {np.mean(xa):.3f} ± {np.std(xa):.3f}')
    except Exception as e:
        print('xgboost skipped:', e)

    # ---- coefficients with cluster bootstrap ----
    print('\n=== Family-A standardized logistic coefficients (cluster bootstrap CI, resample matches) ===')
    coefCI, _ = cluster_bootstrap_coefs(train, FAM_A, y, gA, colsA, reps=500, winsor=train_win)
    ordered = sorted(coefCI.items(), key=lambda kv: -abs(kv[1][0]))
    for name, (md, lo, hi) in ordered:
        sig = '*' if (lo > 0 or hi < 0) else ' '
        print(f'  {sig} {name:20s} beta={md:+.3f}  [{lo:+.3f}, {hi:+.3f}]')

    # ---- hazard ----
    hr, hlo, hhi, nih = discrete_hazard(rows)
    print(f'\n=== Discrete-time hazard (time-to-first-death, person-period logistic) ===')
    print(f'Silver-vs-Iron hazard ratio = {hr:.3f}  [95% CI {hlo:.3f}, {hhi:.3f}]  (n player-games={nih})')
    print('  HR>1 => Silver higher per-minute first-death hazard; HR<1 => die later. CI vs 1.0 = significance.')

    # ---- fit full family-A model on Iron+Silver, score everything ----
    sc, clf = fit_full(Xa, y)
    def score(rws):
        X, _, _, _ = build_matrix(rws, FAM_A, impute=train_med, winsor=train_win)
        return clf.predict_proba(sc.transform(X))[:, 1]

    # calibration on training (in-sample reliability, plus note CV AUC above is the honest discrimination)
    p_train = score(train)
    print('\n=== Reliability (in-sample, family-A model, quintiles) ===')
    for pm, em, n in reliability(y, p_train):
        print(f'  pred={pm:.3f}  emp_silver={em:.3f}  n={n}')

    # ---- Bronze ordinal check ----
    def tier_rows(t): return [r for r in rows if r['tier'] == t and r['source'] == 'cohort']
    print('\n=== Bronze ordinal sanity check (family-A predicted P(Silver)) ===')
    res = {}
    for t in ['IRON', 'BRONZE', 'SILVER']:
        tr = tier_rows(t)
        p = score(tr)
        res[t] = p
        print(f'  {t:7s} mean={p.mean():.3f}  median={np.median(p):.3f}  n_rows={len(tr)}  n_matches={len(set(r["match_id"] for r in tr))}')
    print('  Expect IRON < BRONZE < SILVER if the signature is ordinal.')

    # ---- djfalcon application ----
    dj = [r for r in rows if r['tier'] == 'DJ']
    pdj = score(dj)
    # bootstrap over dj games for mean CI
    djmeans = []
    for _ in range(2000):
        s = RNG.choice(len(pdj), len(pdj), replace=True)
        djmeans.append(pdj[s].mean())
    dlo, dhi = np.percentile(djmeans, [2.5, 97.5])
    print(f'\n=== djfalcon Silver-likeness (family-A model) ===')
    print(f'  mean P(Silver)={pdj.mean():.3f}  [95% CI {dlo:.3f}, {dhi:.3f}]  n_games={len(dj)}')
    print(f'  cohort refs: IRON={res["IRON"].mean():.3f}  BRONZE={res["BRONZE"].mean():.3f}  SILVER={res["SILVER"].mean():.3f}')

    # most/least silver-like games
    dj_scored = sorted(zip(pdj, dj), key=lambda z: -z[0])
    print('\n  Most Silver-like dj games:')
    for p, r in dj_scored[:3]:
        print(f'    P={p:.3f}  {r["match_id"]}  {r["champion"]:12s} {r["position"]:8s} win={r["win"]}')
    print('  Most Iron-like dj games:')
    for p, r in dj_scored[-3:]:
        print(f'    P={p:.3f}  {r["match_id"]}  {r["champion"]:12s} {r["position"]:8s} win={r["win"]}')

    # ---- lever ranking: (dj mean − silver mean) in std units × coef, per family-A raw feature ----
    print('\n=== LEVER RANKING (standardized gap × model weight; associational) ===')
    silver = tier_rows('SILVER')
    # standardize using training scaler stats over the raw feature columns (first len(FAM_A))
    Xdj, cols_dj, _, _ = build_matrix(dj, FAM_A, impute=train_med, winsor=train_win)
    Xsl, _, _, _ = build_matrix(silver, FAM_A, impute=train_med, winsor=train_win)
    Xtr, _, _, _ = build_matrix(train, FAM_A, impute=train_med, winsor=train_win)
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
    coefvec = clf.coef_[0]
    # bootstrap lever contributions: resample dj games AND silver matches
    sil_groups = np.array([r['match_id'] for r in silver])
    sil_mids = np.array(sorted(set(sil_groups)))
    sil_idx_by = {m: np.where(sil_groups == m)[0] for m in sil_mids}
    contribs = defaultdict(list)
    base_gap = {}
    for j, name in enumerate(cols_dj[:len(FAM_A)]):  # raw family-A features only (skip flags/roles)
        gap_std = (Xdj[:, j].mean() - Xsl[:, j].mean()) / sd[j]
        base_gap[name] = gap_std * coefvec[j]
    for _ in range(1000):
        dj_s = RNG.choice(len(dj), len(dj), replace=True)
        sm = RNG.choice(sil_mids, len(sil_mids), replace=True)
        sidx = np.concatenate([sil_idx_by[m] for m in sm])
        for j, name in enumerate(cols_dj[:len(FAM_A)]):
            gap_std = (Xdj[dj_s, j].mean() - Xsl[sidx, j].mean()) / sd[j]
            contribs[name].append(gap_std * coefvec[j])
    lev = []
    for name in cols_dj[:len(FAM_A)]:
        arr = np.array(contribs[name])
        lo, hi = np.percentile(arr, [2.5, 97.5])
        lev.append((name, base_gap[name], lo, hi))
    # rank by how much closing the gap toward silver would RAISE predicted silverness:
    # contribution = coef*gap_std ; if dj is BELOW silver on a silver-positive feature, gap<0 & coef>0 => negative contribution => a lever to raise.
    lev.sort(key=lambda t: t[1])  # most negative first = biggest deficit vs silver
    print('  (negative = dj below Silver on a Silver-positive axis => raise it; positive = dj already Silver-like/above)')
    for name, c, lo, hi in lev:
        sig = '*' if (lo > 0 or hi < 0) else ' '
        print(f'  {sig} {name:16s} contrib={c:+.3f}  [{lo:+.3f}, {hi:+.3f}]')

    # save a json summary
    summary = {
        'auc_A': [float(auc_a.mean()), float(auc_a.std())],
        'auc_AB': [float(auc_ab.mean()), float(auc_ab.std())],
        'contamination_gap': float(gap),
        'hazard_ratio': [hr, hlo, hhi],
        'bronze_check': {t: [float(res[t].mean()), float(np.median(res[t]))] for t in res},
        'dj_silverness': [float(pdj.mean()), float(dlo), float(dhi)],
        'lever_ranking': [[n, float(c), float(lo), float(hi)] for n, c, lo, hi in lev],
        'n_matches': {t: mtier(t) for t in ['IRON', 'SILVER', 'BRONZE', 'DJ']},
    }
    json.dump(summary, open(f'{ROOT}/data/ranksig/model_summary.json', 'w'), indent=1)
    print('\nsaved data/ranksig/model_summary.json')

if __name__ == '__main__':
    main()
