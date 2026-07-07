# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Climb-trajectory predictor (issue 20).

Lagged panel: behavior in a 10-game window predicts the NEXT window's win rate,
across a 30-player IRON/BRONZE/SILVER panel, with mean-reversion controls
(current-window WR and tier). Cluster-robust + player-bootstrap inference.

Subcommands:
  panel   build data/ranksig/panel_windows.csv from cached match bodies + manifest
  fit     regress forward-WR on behavior features w/ mean-reversion controls
  dj      within-player: djfalcon churn hypothesis + June-crash drift (no API)
  all     panel -> fit -> dj

Bodies only (no timelines) — a deliberate budget limitation. Exploratory: small n.
"""
import json, os, sys, gzip, glob, math, datetime as dt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL = f'{ROOT}/data/ranksig/panel'
BODIES = f'{PANEL}/bodies'
CSV = f'{ROOT}/data/panel_windows.csv'  # top-level data/: committable (data/ranksig/ is gitignored)
LANES = {'TOP', 'MIDDLE', 'BOTTOM'}
WIN = 10
RNG = np.random.default_rng(20)

# ---------------------------------------------------------------- panel build
def _participant(body, puuid):
    for p in body['info']['participants']:
        if p.get('puuid') == puuid:
            return p
    return None

def _game_record(body, p):
    info = body['info']
    mins = info['gameDuration'] / 60.0
    tk = sum(x['kills'] for x in info['participants'] if x['teamId'] == p['teamId'])
    pos = p.get('teamPosition') or ''
    cs = p['totalMinionsKilled'] + p['neutralMinionsKilled']
    return {
        't': info.get('gameStartTimestamp') or info.get('gameCreation'),
        'win': 1 if p['win'] else 0,
        'pos': pos,
        'champ': p['championName'],
        'cs_min': cs / mins if mins else 0.0,
        'is_lane': pos in LANES,
        'vis_min': (p.get('visionScore') or 0) / mins if mins else 0.0,
        'deaths_min': p['deaths'] / mins if mins else 0.0,
        'ctrl_wards': p.get('visionWardsBoughtInGame') or 0,
        'kp': (p['kills'] + p['assists']) / tk if tk else None,
        'gv': info.get('gameVersion', ''),
        'dur': info['gameDuration'],
    }

def _entropy(counts):
    tot = sum(counts)
    if tot <= 1:
        return 0.0
    ps = [c / tot for c in counts if c > 0]
    h = -sum(p * math.log(p) for p in ps)
    k = len([c for c in counts if c > 0])
    return h / math.log(k) if k > 1 else 0.0

def _window_features(games):
    wr = np.mean([g['win'] for g in games])
    lane_cs = [g['cs_min'] for g in games if g['is_lane']]
    cs_min = float(np.mean(lane_cs)) if lane_cs else float('nan')
    vis = float(np.mean([g['vis_min'] for g in games]))
    dth = float(np.mean([g['deaths_min'] for g in games]))
    cw = float(np.mean([g['ctrl_wards'] for g in games]))
    kps = [g['kp'] for g in games if g['kp'] is not None]
    kp = float(np.mean(kps)) if kps else float('nan')
    champs = {g['champ'] for g in games}
    from collections import Counter
    rolec = Counter(g['pos'] for g in games if g['pos'])
    return {
        'wr': float(wr), 'cs_min': cs_min, 'vis_min': vis, 'deaths_min': dth,
        'ctrl_wards': cw, 'kp': kp, 'n_champs': len(champs),
        'n_roles': len(rolec), 'role_entropy': _entropy(list(rolec.values())),
        'gv': games[len(games)//2]['gv'],
    }

def build_panel():
    manifest = json.load(open(f'{PANEL}/manifest.json'))
    rows = []
    dropped_remake = 0
    for pidx, pl in enumerate(manifest):
        puuid, tier = pl['puuid'], pl['tier_cohort']
        recs = []
        for mid in pl['match_ids']:
            fp = f'{BODIES}/{mid}.json.gz'
            if not os.path.exists(fp):
                continue
            blob = json.load(gzip.open(fp, 'rt'))
            body = blob.get('body')
            if not body:
                continue
            if body['info'].get('gameDuration', 0) < 600:
                dropped_remake += 1
                continue
            p = _participant(body, puuid)
            if p is None:
                continue
            recs.append(_game_record(body, p))
        recs.sort(key=lambda r: r['t'])
        nwin = len(recs) // WIN
        feats = [_window_features(recs[i*WIN:(i+1)*WIN]) for i in range(nwin)]
        for w in range(nwin - 1):
            f, nxt = feats[w], feats[w+1]
            rows.append({
                'player': pidx, 'tier': tier, 'window': w,
                'cs_min': f['cs_min'], 'vis_min': f['vis_min'],
                'deaths_min': f['deaths_min'], 'ctrl_wards': f['ctrl_wards'],
                'kp': f['kp'], 'n_champs': f['n_champs'],
                'role_entropy': f['role_entropy'], 'n_roles': f['n_roles'],
                'cur_wr': f['wr'], 'fwd_wr': nxt['wr'], 'game_version': f['gv'],
            })
    cols = ['player', 'tier', 'window', 'cs_min', 'vis_min', 'deaths_min',
            'ctrl_wards', 'kp', 'n_champs', 'role_entropy', 'n_roles',
            'cur_wr', 'fwd_wr', 'game_version']
    with open(CSV, 'w') as fh:
        fh.write(','.join(cols) + '\n')
        for r in rows:
            fh.write(','.join(str(r[c]) for c in cols) + '\n')
    nplayers = len({r['player'] for r in rows})
    print(f'panel: {len(rows)} transitions from {nplayers} players '
          f'(dropped {dropped_remake} remakes) -> {CSV}')
    return rows

# ---------------------------------------------------------------- model
def _read_csv():
    lines = open(CSV).read().strip().split('\n')
    hdr = lines[0].split(',')
    out = []
    for ln in lines[1:]:
        d = dict(zip(hdr, ln.split(',')))
        out.append(d)
    return out

def _cluster_robust(X, y, groups):
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(groups):
        m = groups == g
        Xg, eg = X[m], resid[m]
        s = Xg.T @ eg
        meat += np.outer(s, s)
    G = len(np.unique(groups))
    adj = G / (G - 1) if G > 1 else 1.0
    cov = adj * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return beta, se

def _player_bootstrap(X, y, groups, names, B=3000):
    gids = np.unique(groups)
    betas = []
    for _ in range(B):
        samp = RNG.choice(gids, size=len(gids), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in samp])
        Xb, yb = X[idx], y[idx]
        try:
            b = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ yb
        except np.linalg.LinAlgError:
            continue
        betas.append(b)
    betas = np.array(betas)
    lo = np.percentile(betas, 2.5, axis=0)
    hi = np.percentile(betas, 97.5, axis=0)
    return lo, hi

def fit_model():
    rows = _read_csv()
    behav = ['cs_min', 'vis_min', 'deaths_min', 'ctrl_wards', 'kp',
             'n_champs', 'role_entropy']
    # impute NaN behavior columns with column mean
    raw = {c: np.array([float(r[c]) if r[c] not in ('nan', '') else np.nan
                        for r in rows]) for c in behav + ['cur_wr', 'fwd_wr']}
    n_nan = {}
    for c in behav:
        col = raw[c]
        nn = int(np.isnan(col).sum())
        if nn:
            col[np.isnan(col)] = np.nanmean(col)
            n_nan[c] = nn
    y = raw['fwd_wr']
    cur = raw['cur_wr']
    tiers = [r['tier'] for r in rows]
    groups = np.array([int(r['player']) for r in rows])

    # standardize behavior + cur_wr for comparable coefficients; keep raw cur_wr too
    Zbeh = {c: (raw[c] - raw[c].mean()) / (raw[c].std() + 1e-9) for c in behav}
    cur_z = (cur - cur.mean()) / (cur.std() + 1e-9)
    # tier dummies (IRON baseline)
    bronze = np.array([1.0 if t == 'BRONZE' else 0.0 for t in tiers])
    silver = np.array([1.0 if t == 'SILVER' else 0.0 for t in tiers])

    names = ['const'] + behav + ['cur_wr(std)', 'BRONZE', 'SILVER']
    cols = [np.ones(len(y))] + [Zbeh[c] for c in behav] + [cur_z, bronze, silver]
    X = np.column_stack(cols)
    beta, se = _cluster_robust(X, y, groups)
    lo, hi = _player_bootstrap(X, y, groups, names)

    # raw-units mean-reversion: fwd_wr on raw cur_wr (+tier), for interpretability
    Xr = np.column_stack([np.ones(len(y)), cur, bronze, silver])
    br, ser = _cluster_robust(Xr, y, groups)
    lor, hir = _player_bootstrap(Xr, y, groups, ['const', 'cur_wr_raw', 'BRONZE', 'SILVER'])

    # R2
    yhat = X @ beta
    r2 = 1 - np.sum((y - yhat)**2) / np.sum((y - y.mean())**2)

    print(f'\n=== PANEL MODEL: forward-window WR ===')
    print(f'transitions n={len(y)}  players(clusters)={len(np.unique(groups))}  '
          f'R2={r2:.3f}')
    if n_nan:
        print(f'imputed NaN behavior cells (mean): {n_nan}')
    print(f'{"term":14s} {"beta":>8s} {"clSE":>7s} {"boot95CI":>20s}')
    for i, nm in enumerate(names):
        print(f'{nm:14s} {beta[i]:8.4f} {se[i]:7.4f}  [{lo[i]:7.4f},{hi[i]:7.4f}]')
    print(f'\n--- mean-reversion in RAW units (fwd_wr ~ cur_wr + tier) ---')
    print(f'cur_wr_raw beta={br[1]:.4f}  clSE={ser[1]:.4f}  '
          f'boot95=[{lor[1]:.4f},{hir[1]:.4f}]')
    print(f'  interp: a window 10pp above its own mean -> forward WR '
          f'{br[1]*0.10*100:+.1f}pp')

    out = {'n_transitions': int(len(y)), 'n_players': int(len(np.unique(groups))),
           'r2': float(r2), 'terms': names,
           'beta': [float(b) for b in beta], 'cluster_se': [float(s) for s in se],
           'boot_lo': [float(x) for x in lo], 'boot_hi': [float(x) for x in hi],
           'meanrev_raw': {'beta': float(br[1]), 'se': float(ser[1]),
                           'ci': [float(lor[1]), float(hir[1])]},
           'imputed_nan': n_nan}
    json.dump(out, open(f'{PANEL}/model_fit.json', 'w'), indent=1)
    return out

# ---------------------------------------------------------------- dj within
def _perm_p(x, y, B=20000):
    x = np.asarray(x, float); y = np.asarray(y, float)
    r_obs = np.corrcoef(x, y)[0, 1]
    cnt = 0
    for _ in range(B):
        if abs(np.corrcoef(x, RNG.permutation(y))[0, 1]) >= abs(r_obs):
            cnt += 1
    return float(r_obs), (cnt + 1) / (B + 1)

def dj_within():
    games = json.load(open(f'{ROOT}/data/riot_games.json'))
    rk = [g for g in games if g['queue_id'] == 420 and not g['remake']]
    rk.sort(key=lambda g: g['game_creation_utc'])
    print(f'\n=== DJFALCON within-player: {len(rk)} ranked non-remake games ===')
    W, STEP = 10, 5
    wins = []
    i = 0
    while i + W <= len(rk):
        seg = rk[i:i+W]
        champs = {g['champion'] for g in seg}
        poss = [g['position'] for g in seg if g['position'] not in ('Invalid', '')]
        role_switches = sum(1 for a, b in zip(poss, poss[1:]) if a != b)
        lane_cs = [g['cs_per_min'] for g in seg
                   if g['position'] in LANES and g['cs_per_min'] > 0]
        wins.append({
            'start': seg[0]['game_creation_utc'][:10],
            'end': seg[-1]['game_creation_utc'][:10],
            'wr': np.mean([g['win'] for g in seg]),
            'n_champs': len(champs),
            'role_switches': role_switches,
            'cs_min': float(np.mean(lane_cs)) if lane_cs else float('nan'),
        })
        i += STEP
    # forward WR = next window's WR
    for k in range(len(wins) - 1):
        wins[k]['fwd_wr'] = wins[k+1]['wr']

    pairs = [w for w in wins if 'fwd_wr' in w]
    nchamp = [w['n_champs'] for w in pairs]
    rsw = [w['role_switches'] for w in pairs]
    fwd = [w['fwd_wr'] for w in pairs]
    r_c, p_c = _perm_p(nchamp, fwd)
    r_r, p_r = _perm_p(rsw, fwd)
    # also churn vs SAME-window wr as a sanity contrast
    r_cs, p_cs = _perm_p(nchamp, [w['wr'] for w in pairs])
    print(f'churn(distinct champs) -> forward WR: r={r_c:+.3f} perm_p={p_c:.3f}')
    print(f'role_switches         -> forward WR: r={r_r:+.3f} perm_p={p_r:.3f}')
    print(f'(contrast) churn -> SAME-window WR: r={r_cs:+.3f} perm_p={p_cs:.3f}')
    print(f'n_window_pairs={len(pairs)}')

    print(f'\n--- window series (June-crash drift; Iron1 Jun27 -> Iron4 Jul7) ---')
    print(f'{"win":>3s} {"start":>10s} {"end":>10s} {"wr":>5s} {"champs":>6s} '
          f'{"rolesw":>6s} {"cs/min":>6s} {"fwdWR":>5s}')
    for k, w in enumerate(wins):
        fw = f"{w.get('fwd_wr', float('nan')):.2f}" if 'fwd_wr' in w else '  -'
        cs = f"{w['cs_min']:.2f}" if not math.isnan(w['cs_min']) else ' nan'
        print(f'{k:3d} {w["start"]:>10s} {w["end"]:>10s} {w["wr"]:5.2f} '
              f'{w["n_champs"]:6d} {w["role_switches"]:6d} {cs:>6s} {fw:>5s}')

    out = {'n_pairs': len(pairs),
           'churn_fwd': {'r': r_c, 'perm_p': p_c},
           'roleswitch_fwd': {'r': r_r, 'perm_p': p_r},
           'churn_same': {'r': r_cs, 'perm_p': p_cs},
           'windows': [{k: (None if isinstance(v, float) and math.isnan(v) else v)
                        for k, v in w.items()} for w in wins]}
    json.dump(out, open(f'{PANEL}/dj_within.json', 'w'), indent=1)
    return out

if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('panel', 'all'):
        build_panel()
    if what in ('fit', 'all'):
        fit_model()
    if what in ('dj', 'all'):
        dj_within()
