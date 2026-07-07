# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""Analysis for the djfalcon#NA1 climb question. Prints stats, renders charts/.

Run: uv run scripts/analyze.py
"""
import json, os
from collections import defaultdict
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTS = f'{ROOT}/charts'
os.makedirs(CHARTS, exist_ok=True)

# palette (light mode)
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
BASE = '#c3c2b7'
BLUE = '#2a78d6'   # slot 1 — wins / primary series
RED = '#e34948'    # slot 6 — losses

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'sans-serif', 'text.color': INK, 'axes.edgecolor': BASE,
    'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
    'axes.axisbelow': True, 'figure.dpi': 150,
})

QN = {420: 'Ranked Solo', 440: 'Ranked Flex', 400: 'Normal Draft', 430: 'Normal Blind',
      490: 'Quickplay', 450: 'ARAM', 1700: 'Arena', 480: 'Swiftplay', 1750: 'Arena'}

rows = json.load(open(f'{ROOT}/data/riot_games.json'))
for r in rows:
    r['dt'] = datetime.strptime(r['game_creation_utc'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
rows.sort(key=lambda r: r['dt'])

real = [r for r in rows if not r['remake']]
solo = [r for r in real if r['queue_id'] == 420]
norm = [r for r in real if r['queue_id'] == 400]
wr = lambda g: 100 * sum(r['win'] for r in g) / len(g) if g else 0

print(f"games: {len(rows)} total, {len(solo)} ranked solo (excl remakes), {len(norm)} normal draft")

# ---------- 1. win rate by queue ----------
fig, ax = plt.subplots(figsize=(7, 3.2))
qs = [('Ranked Solo', solo), ('Normal Draft', norm)]
names = [f'{n}\n({len(g)} games)' for n, g in qs]
vals = [wr(g) for n, g in qs]
bars = ax.barh(names[::-1], vals[::-1], height=0.45, color=BLUE)
for b, v in zip(bars, vals[::-1]):
    ax.text(v + 1.2, b.get_y() + b.get_height() / 2, f'{v:.0f}%', va='center', color=INK, fontsize=11, fontweight='bold')
ax.axvline(50, color=BASE, linewidth=1, linestyle=(0, (3, 3)))
ax.text(50, 1.62, '50% — breakeven', color=MUTED, fontsize=8, ha='center')
ax.set_xlim(0, 80); ax.set_xlabel('win rate (%)')
ax.set_title('Same player, 25-point gap: ranked solo vs normal draft', color=INK, fontsize=11, loc='left')
ax.grid(axis='y', visible=False)
fig.tight_layout(); fig.savefig(f'{CHARTS}/queue_winrate.png'); plt.close(fig)
print(f"solo WR {wr(solo):.1f}%  normals WR {wr(norm):.1f}%")

# ---------- 2. fundamentals ----------
W = [r for r in solo if r['win']]; L = [r for r in solo if not r['win']]
avg = lambda g, k: sum(r[k] for r in g) / len(g) if g else 0
print(f"deaths/gm solo: {avg(solo,'death'):.2f} (W {avg(W,'death'):.2f} / L {avg(L,'death'):.2f})")
print(f"cs/min solo: {avg(solo,'cs_per_min'):.2f}")
lanes = [r for r in solo if r['position'] in ('TOP', 'MIDDLE', 'BOTTOM')]
print(f"cs/min lanes only (no supp): {avg(lanes,'cs_per_min'):.2f}  (W {avg([r for r in lanes if r['win']],'cs_per_min'):.2f} / L {avg([r for r in lanes if not r['win']],'cs_per_min'):.2f})")
print(f"vision/gm: {avg(solo,'vision_score'):.1f}; control wards/gm: {avg(solo,'control_wards'):.2f}")
if 'kill_participation' in solo[0]:
    kp = [r for r in solo if r.get('kill_participation') is not None]
    print(f"KP: {100*avg(kp,'kill_participation'):.1f}%")

fig, ax = plt.subplots(figsize=(7, 3.4))
mets = [('Deaths / game', avg(W, 'death'), avg(L, 'death')),
        ('CS / min (lanes)', avg([r for r in lanes if r['win']], 'cs_per_min'), avg([r for r in lanes if not r['win']], 'cs_per_min')),
        ('Control wards / game', avg(W, 'control_wards'), avg(L, 'control_wards'))]
y = range(len(mets))
h = 0.32
ax.barh([i + h / 2 + 0.01 for i in y], [m[1] for m in mets], height=h, color=BLUE, label='in wins')
ax.barh([i - h / 2 - 0.01 for i in y], [m[2] for m in mets], height=h, color=RED, label='in losses')
ax.set_yticks(list(y)); ax.set_yticklabels([m[0] for m in mets], color=INK2)
for i, m in enumerate(mets):
    ax.text(m[1] + 0.08, i + h / 2 + 0.01, f'{m[1]:.1f}', va='center', color=INK, fontsize=9)
    ax.text(m[2] + 0.08, i - h / 2 - 0.01, f'{m[2]:.1f}', va='center', color=INK, fontsize=9)
ax.legend(frameon=False, loc='upper right', labelcolor=INK2)
ax.set_title('Ranked solo fundamentals, wins vs losses', color=INK, fontsize=11, loc='left')
ax.grid(axis='y', visible=False)
fig.tight_layout(); fig.savefig(f'{CHARTS}/fundamentals.png'); plt.close(fig)

# ---------- 3. champion pool ----------
c = defaultdict(lambda: [0, 0])  # champ -> [wins, losses]
for r in solo:
    c[r['champion']][0 if r['win'] else 1] += 1
pool = sorted(c.items(), key=lambda x: -(x[1][0] + x[1][1]))
print(f"distinct champs in {len(solo)} solo games: {len(pool)}")
print("champ table:", [(k, f'{v[0]}W-{v[1]}L') for k, v in pool])

fig, ax = plt.subplots(figsize=(7, 0.42 * len(pool) + 1.2))
names = [p[0] for p in pool][::-1]
wins = [p[1][0] for p in pool][::-1]
loss = [p[1][1] for p in pool][::-1]
ax.barh(names, wins, height=0.55, color=BLUE, label='wins')
ax.barh(names, loss, height=0.55, left=[w + 0.06 for w in wins], color=RED, label='losses')
for i, (w_, l_) in enumerate(zip(wins, loss)):
    ax.text(w_ + l_ + 0.35, i, f'{w_}W–{l_}L', va='center', color=INK2, fontsize=8)
ax.legend(frameon=False, loc='lower right', labelcolor=INK2)
ax.set_title(f'{len(pool)} different champions in {len(solo)} ranked games', color=INK, fontsize=11, loc='left')
ax.set_xlabel('games'); ax.grid(axis='y', visible=False)
ax.tick_params(axis='y', labelsize=8, labelcolor=INK2)
fig.tight_layout(); fig.savefig(f'{CHARTS}/champion_pool.png'); plt.close(fig)

# ---------- 4. timeline: games/day + rolling winrate ----------
days = defaultdict(int)
for r in real:
    days[r['dt'].strftime('%m-%d')] += 1
labels = sorted(days)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.6), sharex=False, height_ratios=[1, 1.3])
ax1.bar(labels, [days[d] for d in labels], color=BLUE, width=0.7)
ax1.set_title('Games per day (all queues)', color=INK, fontsize=10, loc='left')
ax1.tick_params(axis='x', labelsize=6, rotation=90)
ax1.grid(axis='x', visible=False)
roll, xs = [], []
for i in range(9, len(solo)):
    window = solo[i - 9:i + 1]
    roll.append(wr(window)); xs.append(i + 1)
ax2.plot(xs, roll, color=BLUE, linewidth=2)
ax2.axhline(50, color=BASE, linewidth=1, linestyle=(0, (3, 3)))
ax2.set_title('Ranked solo win rate, rolling 10-game window', color=INK, fontsize=10, loc='left')
ax2.set_xlabel('ranked game #'); ax2.set_ylim(0, 100)
ax2.grid(axis='x', visible=False)
fig.tight_layout(); fig.savefig(f'{CHARTS}/timeline.png'); plt.close(fig)

# ---------- 5. sessions / tilt ----------
sessions, cur, last = [], [], None
for r in real:
    if last and (r['dt'] - last).total_seconds() > 3 * 3600:
        sessions.append(cur); cur = []
    cur.append(r); last = r['dt']
sessions.append(cur)
by_idx = defaultdict(lambda: [0, 0])  # game idx in session -> [w, n] (ranked only)
for s in sessions:
    for i, r in enumerate(s):
        if r['queue_id'] == 420:
            b = by_idx[min(i + 1, 7)]
            b[0] += r['win']; b[1] += 1
print('session sizes:', sorted((len(s) for s in sessions), reverse=True)[:10])
print('solo WR by game # in session:', {k: f"{100*v[0]/v[1]:.0f}% (n={v[1]})" for k, v in sorted(by_idx.items())})

after_l = [0, 0]; after_w = [0, 0]
for i in range(1, len(solo)):
    prev, curr = solo[i - 1], solo[i]
    same_day = prev['dt'].date() == curr['dt'].date()
    if not same_day:
        continue
    b = after_w if prev['win'] else after_l
    b[0] += curr['win']; b[1] += 1
print(f"same-day WR after a win: {100*after_w[0]/after_w[1]:.0f}% (n={after_w[1]}) ; after a loss: {100*after_l[0]/after_l[1]:.0f}% (n={after_l[1]})")

fig, ax = plt.subplots(figsize=(7, 3.2))
idxs = sorted(by_idx)
vals = [100 * by_idx[i][0] / by_idx[i][1] for i in idxs]
ns = [by_idx[i][1] for i in idxs]
lab = [f'{i}' if i < 7 else '7+' for i in idxs]
bars = ax.bar(lab, vals, color=BLUE, width=0.55)
for b, v, n in zip(bars, vals, ns):
    ax.text(b.get_x() + b.get_width() / 2, v + 2, f'{v:.0f}%', ha='center', color=INK, fontsize=9, fontweight='bold')
    ax.text(b.get_x() + b.get_width() / 2, 3, f'n={n}', ha='center', color='#ffffff', fontsize=7)
ax.axhline(50, color=BASE, linewidth=1, linestyle=(0, (3, 3)))
ax.set_ylim(0, 80)
ax.set_xlabel('ranked game # within a play session (3h gap = new session)')
ax.set_title('Win rate by position in session', color=INK, fontsize=11, loc='left')
ax.grid(axis='x', visible=False)
fig.tight_layout(); fig.savefig(f'{CHARTS}/session.png'); plt.close(fig)

# ---------- 6. kills that don't convert ----------
high_k = [r for r in solo if r['kill'] >= 10]
print(f"10+ kill games: {len(high_k)}, record {sum(r['win'] for r in high_k)}W-{sum(not r['win'] for r in high_k)}L")
print(f"avg turret kills/gm: {avg(solo,'turret_kills'):.2f} (W {avg(W,'turret_kills'):.2f} / L {avg(L,'turret_kills'):.2f})")
if 'objectives_dragon' in solo[0]:
    print(f"team dragons: W {avg(W,'objectives_dragon'):.2f} / L {avg(L,'objectives_dragon'):.2f}")
long_g = [r for r in solo if r['duration_min'] >= 30]
print(f"games ≥30 min: {len(long_g)}, WR {wr(long_g):.0f}% ; games <30 min: {len(solo)-len(long_g)}, WR {wr([r for r in solo if r['duration_min']<30]):.0f}%")
print('CHARTS DONE')
