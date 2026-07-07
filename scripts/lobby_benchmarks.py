# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""Empirical benchmarks from djfalcon's own ranked lobbies (970 player-games).

The 873 other player-games are a matched sample of exactly his MMR pool —
better proxy data than any global 'average for Iron' table. Prints his
percentile on each fundamental; renders charts/lobby_benchmarks.png.

Run: uv run scripts/lobby_benchmarks.py
"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURFACE, INK, INK2, MUTED, GRID, BASE = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BLUE, RED = '#2a78d6', '#e34948'
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'sans-serif', 'text.color': INK, 'axes.edgecolor': BASE,
    'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
    'axes.axisbelow': True, 'figure.dpi': 150,
})

rows = json.load(open(f'{ROOT}/data/lobby_benchmarks.json'))
me = [r for r in rows if r['is_target']]
oth = [r for r in rows if not r['is_target']]
LANES = ('TOP', 'MIDDLE', 'BOTTOM')
mean = lambda xs: sum(xs) / len(xs) if xs else 0

def pctile(value, dist, lower_is_better=False):
    """share of the lobby distribution he beats"""
    if lower_is_better:
        return 100 * sum(1 for d in dist if d > value) / len(dist)
    return 100 * sum(1 for d in dist if d < value) / len(dist)

print(f'him: {len(me)} player-games; lobby cohort: {len(oth)} player-games')

# deaths (all roles)
his_d = mean([r['death'] for r in me])
oth_d = [r['death'] for r in oth]
print(f"deaths/gm: him {his_d:.2f} vs lobby mean {mean(oth_d):.2f} -> beats {pctile(his_d, oth_d, True):.0f}% of lobby player-games")

# cs/min (lane roles only, both sides)
his_cs = mean([r['cs_per_min'] for r in me if r['position'] in LANES])
oth_cs = [r['cs_per_min'] for r in oth if r['position'] in LANES]
print(f"cs/min (lanes): him {his_cs:.2f} vs lobby mean {mean(oth_cs):.2f} -> beats {pctile(his_cs, oth_cs):.0f}%")

# kill participation
his_kp = mean([r['kp'] for r in me if r['kp'] is not None])
oth_kp = [r['kp'] for r in oth if r['kp'] is not None]
print(f"KP: him {100*his_kp:.1f}% vs lobby mean {100*mean(oth_kp):.1f}% -> beats {pctile(his_kp, oth_kp):.0f}%")

# control wards
his_cw = mean([r['control_wards'] for r in me])
oth_cw = [r['control_wards'] for r in oth]
bought = 100 * sum(1 for c in oth_cw if c and c > 0) / len(oth_cw)
print(f"control wards/gm: him {his_cw:.2f} vs lobby mean {mean(oth_cw):.2f}; {bought:.0f}% of lobby player-games bought >=1")

# vision score
his_v = mean([r['vision_score'] for r in me])
oth_v = [r['vision_score'] for r in oth]
print(f"vision score: him {his_v:.1f} vs lobby mean {mean(oth_v):.1f} -> beats {pctile(his_v, oth_v):.0f}%")

# deaths of WINNERS in his lobbies (the actual target to hit)
win_d = [r['death'] for r in oth if r['win']]
lose_d = [r['death'] for r in oth if not r['win']]
print(f"lobby winners avg deaths: {mean(win_d):.2f}; lobby losers: {mean(lose_d):.2f}")

# chart: 4 panels, lobby distribution + his mean
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
panels = [
    ('Deaths / game', [r['death'] for r in oth], his_d, 'lower is better'),
    ('CS / min (lanes)', oth_cs, his_cs, ''),
    ('Kill participation', [100 * k for k in oth_kp], 100 * his_kp, '%'),
    ('Control wards / game', oth_cw, his_cw, ''),
]
for ax, (title, dist, his, note) in zip(axes, panels):
    ax.hist(dist, bins=20, color=BLUE, alpha=0.85, edgecolor=SURFACE)
    ax.axvline(his, color=RED, linewidth=2)
    ax.text(his, ax.get_ylim()[1] * 0.97, f' him: {his:.1f}', color=RED, fontsize=9, va='top', fontweight='bold')
    ax.set_title(title, color=INK, fontsize=10, loc='left')
    ax.set_yticks([])
fig.suptitle('djfalcon vs the 873 other player-games in his own ranked lobbies', color=INK, fontsize=11, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(f'{ROOT}/charts/lobby_benchmarks.png'); plt.close(fig)
print('lobby_benchmarks.png written')
