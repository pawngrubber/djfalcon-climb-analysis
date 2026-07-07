# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "scipy"]
# ///
"""Uncertainty bounds for every win-rate claim in the thread.

Wilson 95% intervals for proportions, Fisher exact tests for group contrasts,
Mann-Whitney U for the deaths difference. Renders charts/forest.png.

Run: uv run scripts/stats_ci.py
"""
import json, os
from collections import defaultdict
from datetime import datetime, timezone

from scipy.stats import fisher_exact, mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURFACE, INK, INK2, MUTED, GRID, BASE = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BLUE = '#2a78d6'
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'sans-serif', 'text.color': INK, 'axes.edgecolor': BASE,
    'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
    'axes.axisbelow': True, 'figure.dpi': 150,
})

def wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (p, max(0, c - h), min(1, c + h))

rows = json.load(open(f'{ROOT}/data/riot_games.json'))
for r in rows:
    r['dt'] = datetime.strptime(r['game_creation_utc'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
rows.sort(key=lambda r: r['dt'])
real = [r for r in rows if not r['remake']]
solo = [r for r in real if r['queue_id'] == 420]
norm = [r for r in real if r['queue_id'] == 400]

AS = [r for r in solo if r['champion'] in ('Annie', 'Sona')]
NAS = [r for r in solo if r['champion'] == 'Nasus']
OTH = [r for r in solo if r['champion'] not in ('Annie', 'Sona')]
HK = [r for r in solo if r['kill'] >= 10]

# sessions
sessions, cur, last = [], [], None
for r in real:
    if last and (r['dt'] - last).total_seconds() > 3 * 3600:
        sessions.append(cur); cur = []
    cur.append(r); last = r['dt']
sessions.append(cur)
early, late = [], []
for s in sessions:
    for i, r in enumerate(s):
        if r['queue_id'] == 420:
            (early if i < 3 else late).append(r)

wl = lambda g: (sum(r['win'] for r in g), len(g))
claims = [
    ('Ranked solo overall', *wl(solo)),
    ('Normal draft', *wl(norm)),
    ('Annie + Sona (ranked)', *wl(AS)),
    ('All other champs (ranked)', *wl(OTH)),
    ('Nasus (ranked)', *wl(NAS)),
    ('10+ kill games', *wl(HK)),
    ('Session games 1–3', *wl(early)),
    ('Session games 4+', *wl(late)),
]
print('claim, wins/n, wr, wilson95')
for name, w, n in claims:
    p, lo, hi = wilson(w, n)
    print(f'{name}: {w}/{n} = {100*p:.0f}%  [{100*lo:.0f}%, {100*hi:.0f}%]')

def fisher(g1, g2, label):
    w1, n1 = wl(g1); w2, n2 = wl(g2)
    odds, p = fisher_exact([[w1, n1 - w1], [w2, n2 - w2]])
    print(f'Fisher {label}: {w1}/{n1} vs {w2}/{n2} -> p = {p:.4f}')
    return p

print()
fisher(norm, solo, 'normals vs ranked')
fisher(AS, OTH, 'Annie+Sona vs other champs')
fisher(NAS, AS, 'Nasus vs Annie+Sona')
fisher(early, late, 'session 1-3 vs 4+')

W = [r['death'] for r in solo if r['win']]
L = [r['death'] for r in solo if not r['win']]
u, p = mannwhitneyu(W, L, alternative='less')
print(f'\nMann-Whitney deaths (wins < losses): U={u}, p = {p:.5f}')
print(f'deaths mean W {sum(W)/len(W):.2f} (n={len(W)}) vs L {sum(L)/len(L):.2f} (n={len(L)})')

# forest plot
fig, ax = plt.subplots(figsize=(8, 0.52 * len(claims) + 1.4))
names = [f'{c[0]}  ({c[1]}/{c[2]})' for c in claims][::-1]
stats = [wilson(c[1], c[2]) for c in claims][::-1]
ys = range(len(claims))
for y, (p, lo, hi) in zip(ys, stats):
    ax.plot([100 * lo, 100 * hi], [y, y], color=BLUE, linewidth=2, solid_capstyle='round')
    ax.plot(100 * p, y, 'o', color=BLUE, markersize=7)
    ax.text(100 * hi + 1.5, y, f'{100*p:.0f}% [{100*lo:.0f}–{100*hi:.0f}]', va='center', color=INK2, fontsize=8.5)
ax.set_yticks(list(ys)); ax.set_yticklabels(names, fontsize=9, color=INK2)
ax.axvline(50, color=BASE, linewidth=1, linestyle=(0, (3, 3)))
ax.set_xlim(0, 105); ax.set_xlabel('win rate (%), Wilson 95% interval')
ax.set_title('Every win-rate claim, with its uncertainty', color=INK, fontsize=11, loc='left')
ax.grid(axis='y', visible=False)
fig.tight_layout(); fig.savefig(f'{ROOT}/charts/forest.png'); plt.close(fig)
print('\nforest.png written')
