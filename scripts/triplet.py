# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "scipy"]
# ///
"""Triplet analysis: djfalcon vs Iron II / Bronze II / Silver II cohorts.

For each fundamentals metric we place djfalcon against his own ranked lobbies
(the 873 non-target player-games) and against three rank-ladder cohorts, with
bootstrap CIs, percentiles, Mann-Whitney tests + rank-biserial effect sizes,
and a rank-ladder separability test (does the metric actually distinguish
Iron peers from Silver goal-peers?).

Renders charts/triplet_ladder.png and charts/triplet_gaps.png.
Run: uv run scripts/triplet.py
"""
import json, os

import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURFACE, INK, INK2, MUTED, GRID, BASE = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BLUE = '#2a78d6'
RED = '#e34948'
GRAY = '#898781'
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'sans-serif', 'text.color': INK, 'axes.edgecolor': BASE,
    'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
    'axes.axisbelow': True, 'figure.dpi': 150,
})

SEED = 20260707
LANES = ('TOP', 'MIDDLE', 'BOTTOM')

# ---------------------------------------------------------------- load & normalize
rg = json.load(open(f'{ROOT}/data/riot_games.json'))
him = [r for r in rg if r['queue_id'] == 420 and not r['remake']]
for r in him:
    r['deaths_per_min'] = r['death'] / r['duration_min']
    r['vision_per_min'] = r['vision_score'] / r['duration_min']
    r['kp'] = r['kill_participation']

lb = json.load(open(f'{ROOT}/data/lobby_benchmarks.json'))
lobbies = [r for r in lb if not r['is_target']]  # 873 other player-games
for r in lobbies:
    r['deaths_per_min'] = r['death'] / r['duration_min']
    r['vision_per_min'] = r['vision_score'] / r['duration_min']

cohorts = {t: json.load(open(f'{ROOT}/data/cohort_{t}.json'))
           for t in ('iron', 'bronze', 'silver')}

# metric -> (label, lane_only)
METRICS = [
    ('deaths_per_min', 'deaths / min', False),
    ('cs_per_min', 'CS / min (lanes)', True),
    ('kp', 'kill participation', False),
    ('vision_per_min', 'vision / min', False),
    ('control_wards', 'control wards', False),
]
# higher_better direction for orienting the gaps chart (deaths inverted)
HIGHER_BETTER = {'deaths_per_min': False, 'cs_per_min': True, 'kp': True,
                 'vision_per_min': True, 'control_wards': True}


def vals(pop, metric, lane_only):
    src = pop
    if lane_only:
        src = [r for r in pop if r['position'] in LANES]
    return np.array([float(r[metric]) for r in src if r.get(metric) is not None], dtype=float)


def boot_ci(x, n=10000, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def rank_biserial(a, b):
    """r for group a vs b: positive => a's values tend to be larger."""
    U1, p = mannwhitneyu(a, b, alternative='two-sided')
    rb = 2.0 * U1 / (len(a) * len(b)) - 1.0
    return p, rb


def percentile_of(value, dist):
    return 100.0 * float(np.mean(dist < value) + 0.5 * np.mean(dist == value))


POPS = [
    ('djfalcon', lambda m, lo: vals(him, m, lo)),
    ('his-lobbies', lambda m, lo: vals(lobbies, m, lo)),
    ('Iron II', lambda m, lo: vals(cohorts['iron'], m, lo)),
    ('Bronze II', lambda m, lo: vals(cohorts['bronze'], m, lo)),
    ('Silver II', lambda m, lo: vals(cohorts['silver'], m, lo)),
]

# ---------------------------------------------------------------- step 2 tables
print('=' * 96)
print('PER-METRIC POPULATION TABLES  (bootstrap 95% CI, 10k resamples, seed fixed)')
print('=' * 96)

ladder_data = {}   # metric -> dict of means/CI for chart
gaps_rb = {}       # metric -> him-vs-silver rank-biserial (oriented)

for metric, label, lane_only in METRICS:
    him_v = vals(him, metric, lane_only)
    him_mean = him_v.mean()
    print(f'\n### {label}   ({"lanes TOP/MID/BOT" if lane_only else "all roles"})')
    print(f'{"population":<13}{"n":>5}{"mean":>9}{"  95% CI":>18}'
          f'{"him %ile":>10}{"MWU p":>10}{"rank-bis":>10}')
    for name, getter in POPS:
        x = getter(metric, lane_only)
        m = x.mean()
        lo, hi = boot_ci(x)
        if name == 'djfalcon':
            print(f'{name:<13}{len(x):>5}{m:>9.3f}   [{lo:>6.3f},{hi:>6.3f}]'
                  f'{"—":>10}{"—":>10}{"—":>10}')
        else:
            pct = percentile_of(him_mean, x)
            p, rb = rank_biserial(him_v, x)
            print(f'{name:<13}{len(x):>5}{m:>9.3f}   [{lo:>6.3f},{hi:>6.3f}]'
                  f'{pct:>9.0f}%{p:>10.4f}{rb:>+10.3f}')
    # stash for charts
    d = {'him': him_mean, 'lobbies': vals(lobbies, metric, lane_only).mean()}
    for t, disp in (('iron', 'Iron II'), ('bronze', 'Bronze II'), ('silver', 'Silver II')):
        x = vals(cohorts[t], metric, lane_only)
        d[t] = (x.mean(), *boot_ci(x))
    ladder_data[metric] = d
    # oriented him-vs-silver effect
    _, rb_s = rank_biserial(him_v, vals(cohorts['silver'], metric, lane_only))
    gaps_rb[metric] = rb_s if HIGHER_BETTER[metric] else -rb_s

# ---------------------------------------------------------------- step 3 rank-ladder
print('\n' + '=' * 96)
print('RANK-LADDER SEPARATION  (does the metric distinguish cohorts?)')
print('=' * 96)
print(f'{"metric":<20}{"Iron vs Silver p":>18}{"rank-bis":>10}'
      f'{"Iron vs Bronze p":>18}{"rank-bis":>10}   verdict')
separates = {}
for metric, label, lane_only in METRICS:
    iron = vals(cohorts['iron'], metric, lane_only)
    silv = vals(cohorts['silver'], metric, lane_only)
    bron = vals(cohorts['bronze'], metric, lane_only)
    p_is, rb_is = rank_biserial(iron, silv)
    p_ib, rb_ib = rank_biserial(iron, bron)
    sep = p_is < 0.05
    separates[metric] = (sep, p_is)
    verdict = f'separates (p={p_is:.3f})' if sep else 'FLAT across ranks'
    print(f'{label:<20}{p_is:>18.4f}{rb_is:>+10.3f}{p_ib:>18.4f}{rb_ib:>+10.3f}   {verdict}')

# ---------------------------------------------------------------- step 4 role vision
print('\n' + '=' * 96)
print('ROLE-MATCHED VISION / MIN  (mean by position)')
print('=' * 96)
ROLES = ('TOP', 'MIDDLE', 'BOTTOM', 'UTILITY')
print(f'{"role":<9}{"djfalcon":>10}{"(n)":>7}{"his-lobbies":>13}{"Iron II":>11}{"Bronze II":>11}{"Silver II":>11}')
def role_mean(pop, metric, role):
    x = [float(r[metric]) for r in pop if r['position'] == role]
    return sum(x) / len(x) if x else float('nan')
for role in ROLES:
    hm = role_mean(him, 'vision_per_min', role)
    lm = role_mean(lobbies, 'vision_per_min', role)
    im = role_mean(cohorts['iron'], 'vision_per_min', role)
    bm = role_mean(cohorts['bronze'], 'vision_per_min', role)
    sm = role_mean(cohorts['silver'], 'vision_per_min', role)
    nhim = len([r for r in him if r['position'] == role])
    print(f'{role:<9}{hm:>10.3f}{nhim:>7}{lm:>13.3f}{im:>11.3f}{bm:>11.3f}{sm:>11.3f}')

# ---------------------------------------------------------------- chart: ladder
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
axes = axes.flatten()
cohort_names = ['Iron II', 'Bronze II', 'Silver II']
cohort_keys = ['iron', 'bronze', 'silver']
for ax, (metric, label, lane_only) in zip(axes, METRICS):
    d = ladder_data[metric]
    means = [d[k][0] for k in cohort_keys]
    los = [d[k][1] for k in cohort_keys]
    his = [d[k][2] for k in cohort_keys]
    xpos = np.arange(3)
    yerr = [[m - lo for m, lo in zip(means, los)], [hi - m for m, hi in zip(means, his)]]
    ax.bar(xpos, means, width=0.62, color=BLUE, alpha=0.85,
           yerr=yerr, capsize=4, ecolor=INK2, error_kw={'linewidth': 1.2})
    for x, m, hi in zip(xpos, means, his):
        ax.text(x, hi, f'  {m:.2f}', ha='center', va='bottom', fontsize=8.5, color=INK)
    # djfalcon mean = red line
    ax.axhline(d['him'], color=RED, linewidth=2, linestyle='-', zorder=5)
    ax.text(2.55, d['him'], f'him {d["him"]:.2f}', color=RED, fontsize=8.5,
            va='center', ha='left', fontweight='bold')
    # his-lobbies mean = gray marker
    ax.axhline(d['lobbies'], color=GRAY, linewidth=1.3, linestyle=(0, (4, 3)), zorder=4)
    ax.text(2.55, d['lobbies'], f'lobbies {d["lobbies"]:.2f}', color=GRAY, fontsize=7.5,
            va='center', ha='left')
    ax.set_xticks(xpos)
    ax.set_xticklabels(cohort_names, fontsize=9)
    ax.set_xlim(-0.6, 4.0)
    ax.grid(axis='x', visible=False)
    sep, p_is = separates[metric]
    tag = f'separates ranks p={p_is:.3f}' if sep else 'flat across ranks'
    ax.set_title(f'{label}\n{tag}', color=INK, fontsize=10, loc='left')
    top = max(his + [d['him'], d['lobbies']])
    ax.set_ylim(0, top * 1.18)
for ax in axes[len(METRICS):]:
    ax.axis('off')
fig.suptitle('djfalcon vs the rank ladder — cohort means (blue, 95% CI), his mean (red), his lobbies (gray)',
             color=INK, fontsize=12, x=0.02, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(f'{ROOT}/charts/triplet_ladder.png')
plt.close(fig)
print('\ncharts/triplet_ladder.png written')

# ---------------------------------------------------------------- chart: gaps forest
order = sorted(METRICS, key=lambda mm: gaps_rb[mm[0]])
labels = [m[1] for m in order]
vals_rb = [gaps_rb[m[0]] for m in order]
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ys = np.arange(len(order))
for y, v in zip(ys, vals_rb):
    c = RED if v < 0 else BLUE
    ax.plot([0, v], [y, y], color=c, linewidth=2.4, solid_capstyle='round')
    ax.plot(v, y, 'o', color=c, markersize=8)
    ax.text(v + (0.02 if v >= 0 else -0.02), y, f'{v:+.3f}',
            va='center', ha='left' if v >= 0 else 'right', color=INK2, fontsize=9)
ax.axvline(0, color=BASE, linewidth=1)
ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=9.5, color=INK2)
mx = max(0.05, max(abs(v) for v in vals_rb)) * 1.35
ax.set_xlim(-mx, mx)
ax.set_xlabel('rank-biserial effect, him vs Silver II  (oriented so + = above Silver level; deaths inverted)')
ax.set_title('Where djfalcon sits relative to Silver II peers', color=INK, fontsize=11, loc='left')
ax.grid(axis='y', visible=False)
fig.tight_layout()
fig.savefig(f'{ROOT}/charts/triplet_gaps.png')
plt.close(fig)
print('charts/triplet_gaps.png written')
