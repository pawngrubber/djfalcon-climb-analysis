# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy"]
# ///
"""Test two hypotheses about djfalcon#NA1's role performance.

H1 -- "he does better as support because farming is removed"
H2 -- champion-specific farm dependence (Nasus/Annie etc.)

Local data only. Run: uv run scripts/role_hypothesis.py
"""
import json, os
from collections import defaultdict
from statistics import mean, pstdev

from scipy.stats import fisher_exact, mannwhitneyu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANES = ('TOP', 'MIDDLE', 'BOTTOM')


def load(name):
    return json.load(open(f'{ROOT}/data/{name}.json'))


def wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (p, max(0, c - h), min(1, c + h))


def pct_below(val, arr):
    """Percentile of val within arr (fraction strictly below + half ties)."""
    n = len(arr)
    below = sum(x < val for x in arr)
    ties = sum(x == val for x in arr)
    return 100.0 * (below + 0.5 * ties) / n


def rank_biserial(a, b):
    """rb effect size for a>b via Mann-Whitney. AUC = P(a>b). rb = 2*AUC-1."""
    if len(a) == 0 or len(b) == 0:
        return None
    U, p = mannwhitneyu(a, b, alternative='two-sided')
    auc = U / (len(a) * len(b))
    return 2 * auc - 1, p, auc


# ------------------------------------------------------------------ load
games = load('riot_games')
ranked = [g for g in games if g['queue_id'] == 420 and not g['remake']
          and g['position'] != 'Invalid']

cohort = load('cohort_iron') + load('cohort_bronze') + load('cohort_silver')
silver = load('cohort_silver')  # "goal peers"

# his per-game derived metrics
for g in ranked:
    g['deaths_per_min'] = g['death'] / g['duration_min']
    g['vision_per_min'] = g['vision_score'] / g['duration_min']
    g['kp'] = g['kill_participation']

print('=' * 72)
print('HYPOTHESIS 1 -- "better as support because farming is removed"')
print('=' * 72)

# ---- 1a win rate: UTILITY vs lanes ----------------------------------
util = [g for g in ranked if g['position'] == 'UTILITY']
lane = [g for g in ranked if g['position'] in LANES]
uw, un = sum(g['win'] for g in util), len(util)
lw, ln = sum(g['win'] for g in lane), len(lane)
up, ulo, uhi = wilson(uw, un)
lp, llo, lhi = wilson(lw, ln)
odds, fp = fisher_exact([[uw, un - uw], [lw, ln - lw]])
print('\n(1a) Ranked win rate')
print(f'  UTILITY : {uw}/{un} = {up:.1%}  [95% {ulo:.1%}, {uhi:.1%}]')
print(f'  LANES   : {lw}/{ln} = {lp:.1%}  [95% {llo:.1%}, {lhi:.1%}]')
print(f'  Fisher exact: OR={odds:.2f}, p={fp:.3f}')
# per-lane detail
for pos in LANES:
    gg = [g for g in ranked if g['position'] == pos]
    w = sum(g['win'] for g in gg)
    p, lo, hi = wilson(w, len(gg))
    print(f'    {pos:7s}: {w}/{len(gg)} = {p:.1%} [{lo:.1%},{hi:.1%}]')

# ---- 1b role-relative percentile profile ----------------------------
print('\n(1b) Percentile of his mean vs same-position cohort (Iron+Bronze+Silver pooled)')
print('     good-direction shown; deaths lower=better so we flag interpretation')
metrics_lane = ['deaths_per_min', 'kp', 'vision_per_min', 'cs_per_min']
metrics_util = ['deaths_per_min', 'kp', 'vision_per_min']


def cohort_pos(pos, metric):
    return [c[metric] for c in cohort if c['position'] == pos]


def role_profile(pos, metrics):
    hisg = [g for g in ranked if g['position'] == pos]
    out = {}
    for m in metrics:
        hv = mean(g[m] for g in hisg)
        arr = cohort_pos(pos, m)
        out[m] = (hv, pct_below(hv, arr), mean(arr))
    return len(hisg), out


LOWER_BETTER = {'deaths_per_min'}
profiles = {}
for pos, mets in [('TOP', metrics_lane), ('MIDDLE', metrics_lane),
                  ('BOTTOM', metrics_lane), ('UTILITY', metrics_util)]:
    n, out = role_profile(pos, mets)
    profiles[pos] = out
    print(f'\n  {pos} (n={n} his games)')
    for m, (hv, pc, cm) in out.items():
        # "good percentile": for lower-better invert
        good = 100 - pc if m in LOWER_BETTER else pc
        dist = abs(good - 50)
        print(f'    {m:15s} his={hv:6.3f} cohort_mean={cm:6.3f} '
              f'raw_pctile={pc:5.1f}  good_pctile={good:5.1f}  |dev50|={dist:4.1f}')

# closeness-to-average comparison on the non-CS metrics
print('\n  Non-CS metrics: mean |good_pctile - 50| (lower = closer to cohort-average)')
noncs = ['deaths_per_min', 'kp', 'vision_per_min']
for pos in ['TOP', 'MIDDLE', 'BOTTOM', 'UTILITY']:
    devs = []
    for m in noncs:
        hv, pc, cm = profiles[pos][m]
        good = 100 - pc if m in LOWER_BETTER else pc
        devs.append(abs(good - 50))
    print(f'    {pos:7s}: {mean(devs):.1f}   (per-metric {[round(d,1) for d in devs]})')
laneavg = mean([mean([abs((100 - profiles[p][m][1] if m in LOWER_BETTER else profiles[p][m][1]) - 50)
                      for m in noncs]) for p in LANES])
utilavg = mean([abs((100 - profiles['UTILITY'][m][1] if m in LOWER_BETTER else profiles['UTILITY'][m][1]) - 50)
                for m in noncs])
print(f'    LANES pooled avg |dev50| = {laneavg:.1f}   UTILITY = {utilavg:.1f}')
print(f'    Prediction (support closer to average) HOLDS' if utilavg < laneavg
      else '    Prediction FAILS (support NOT closer to average)')

# ---- 1c counterpoint: support vision gap vs lane CS gap (z effect sizes) ----
print('\n(1c) Effect-size gaps vs GOAL peers (Silver cohort, same position)')


def z_gap(pos, metric, hisg):
    hv = mean(g[metric] for g in hisg)
    arr = [c[metric] for c in silver if c['position'] == pos]
    sd = pstdev(arr) or 1e-9
    return hv, mean(arr), (hv - mean(arr)) / sd


utilg = [g for g in ranked if g['position'] == 'UTILITY']
hv, cm, zv = z_gap('UTILITY', 'vision_per_min', utilg)
print(f'  Support VISION : his={hv:.3f} silver={cm:.3f} z={zv:+.2f}  (his worst support stat)')
# lane CS gap pooled across lanes
lane_cs_z = []
for pos in LANES:
    gg = [g for g in ranked if g['position'] == pos]
    h, c, z = z_gap(pos, 'cs_per_min', gg)
    lane_cs_z.append(z)
    print(f'  {pos} CS       : his={h:.3f} silver={c:.3f} z={z:+.2f}')
print(f'  |z| support vision = {abs(zv):.2f}   vs mean |z| lane CS = {mean(abs(x) for x in lane_cs_z):.2f}')

# total gap to goal peers per role (sum |z| over shared metrics)
print('\n  Total gap to Silver goal-peers (sum |z| across metrics):')
gap_tot = {}
for pos, mets in [('TOP', metrics_lane), ('MIDDLE', metrics_lane),
                  ('BOTTOM', metrics_lane), ('UTILITY', metrics_util)]:
    gg = [g for g in ranked if g['position'] == pos]
    zs = {m: z_gap(pos, m, gg)[2] for m in mets}
    tot = sum(abs(v) for v in zs.values())
    gap_tot[pos] = (tot, zs)
    print(f'    {pos:7s} sum|z|={tot:.2f}  ' + ' '.join(f'{m.split("_")[0]}:{v:+.2f}' for m, v in zs.items()))
best = min(gap_tot, key=lambda k: gap_tot[k][0])
print(f'  Smallest total gap to goal peers: {best} (sum|z|={gap_tot[best][0]:.2f})')
print('  NOTE: lanes carry an extra CS metric (4 vs 3); comparison is directional not exact.')

print('\n' + '=' * 72)
print('HYPOTHESIS 2 -- champion-specific farm dependence')
print('=' * 72)

# ---- 2a CS-dependence per champion (lane roles), pooled cohorts ------
lanecoh = [c for c in cohort if c['position'] in LANES]
print(f'\n(2a) Pooled cohort lane player-games: n={len(lanecoh)}')
# baseline all-champion
winners = [c['cs_per_min'] for c in lanecoh if c['win']]
losers = [c['cs_per_min'] for c in lanecoh if not c['win']]
rb, p, auc = rank_biserial(winners, losers)
print(f'  BASELINE all-champ lane CS-dependence: rb={rb:+.3f} (AUC={auc:.3f}) '
      f'p={p:.2e}  n_win={len(winners)} n_lose={len(losers)}')

bychamp = defaultdict(list)
for c in lanecoh:
    bychamp[c['champion']].append(c)

results = []
for champ, rows in bychamp.items():
    if len(rows) < 8:
        continue
    w = [r['cs_per_min'] for r in rows if r['win']]
    l = [r['cs_per_min'] for r in rows if not r['win']]
    if not w or not l:
        continue
    rb, p, auc = rank_biserial(w, l)
    results.append((champ, len(rows), len(w), len(l), rb, p))

results.sort(key=lambda x: x[4], reverse=True)
print(f'\n  Champions with n>=8 lane games (both W&L present): {len(results)}')
print(f'  {"champ":12s} {"n":>3s} {"nW":>3s} {"nL":>3s} {"rb":>7s} {"p":>8s}')
for champ, n, nw, nl, rb, p in results:
    print(f'  {champ:12s} {n:3d} {nw:3d} {nl:3d} {rb:+7.3f} {p:8.3f}')
if results:
    print(f'\n  MOST CS-dependent : {results[0][0]} rb={results[0][4]:+.3f} (n={results[0][1]})')
    print(f'  LEAST CS-dependent: {results[-1][0]} rb={results[-1][4]:+.3f} (n={results[-1][1]})')

# Nasus / Annie specifically
for champ, his_cs in [('Nasus', None), ('Annie', None)]:
    rows = bychamp.get(champ, [])
    n = len(rows)
    hisg = [g for g in ranked if g['champion'] == champ]
    his_mean = mean(g['cs_per_min'] for g in hisg) if hisg else None
    print(f'\n  {champ}: cohort lane n={n}', end='')
    if n >= 8:
        w = [r['cs_per_min'] for r in rows if r['win']]
        l = [r['cs_per_min'] for r in rows if not r['win']]
        if w and l:
            rb, p, auc = rank_biserial(w, l)
            print(f'  rb={rb:+.3f} p={p:.3f} (nW={len(w)} nL={len(l)})')
        else:
            print('  (one outcome empty -> no test)')
    else:
        cm = mean(r['cs_per_min'] for r in rows) if rows else float('nan')
        print(f'  INSUFFICIENT (<8). cohort Nasus/champ mean cs={cm:.2f} '
              f'vs his {his_mean:.2f}' if rows else '  no cohort rows')

# ---- 2b his per-champion CS vs cohort same-champion ------------------
print('\n(2b) His CS vs cohort same-champion players (any lane role, n reported)')
his_by_champ = defaultdict(list)
for g in ranked:
    his_by_champ[g['champion']].append(g)
coh_by_champ = defaultdict(list)
for c in lanecoh:
    coh_by_champ[c['champion']].append(c)

rows_out = []
for champ, hg in his_by_champ.items():
    if hg[0]['position'] not in LANES:
        continue
    his_mean = mean(g['cs_per_min'] for g in hg)
    coh = coh_by_champ.get(champ, [])
    cm = mean(c['cs_per_min'] for c in coh) if coh else None
    rows_out.append((champ, len(hg), his_mean, len(coh), cm))
rows_out.sort(key=lambda x: -x[1])
print(f'  {"champ":12s} {"his_n":>5s} {"his_cs":>7s} {"coh_n":>5s} {"coh_cs":>7s} {"diff":>6s}')
for champ, hn, hm, cn, cm in rows_out:
    if cm is None:
        print(f'  {champ:12s} {hn:5d} {hm:7.2f} {cn:5d}   n/a     -   THIN(no cohort)')
    else:
        flag = '  THIN' if (hn < 5 or cn < 5) else ''
        print(f'  {champ:12s} {hn:5d} {hm:7.2f} {cn:5d} {cm:7.2f} {hm-cm:+6.2f}{flag}')

print('\n(2c) CAVEAT: CS-dependence here is CORRELATIONAL. Winners farm more partly')
print('     BECAUSE they are winning (longer leads, safer waves), not only vice-versa.')
print('     rb sign/size cannot be read as "farming causes the win".')
