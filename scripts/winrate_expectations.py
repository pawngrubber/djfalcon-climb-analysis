# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "scipy"]
# ///
"""What is a 'good' win rate, quantitatively?

Three lenses:
1. Climb speed — LP/game and expected games to climb, as a function of true win rate
2. Detectability — how many games before a given true win rate is statistically
   distinguishable from 50%
3. Bayesian posterior — given an observed W-L record, the probability the TRUE
   win rate is above 50% (and above 55%)

Renders charts/winrate_meaning.png. Run: uv run scripts/winrate_expectations.py
"""
import os
from math import sqrt
from scipy.stats import beta
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

LP = 25          # typical net LP per win/loss at even MMR
TARGET = 325     # Iron IV 75 LP -> Bronze IV 0 LP: 25 + 3*100

print('=== 1. climb speed (assuming ±25 LP) ===')
print('true WR | LP/game | games to Bronze (325 LP)')
for p in (0.50, 0.51, 0.53, 0.55, 0.58, 0.60, 0.65, 0.70):
    lp_g = LP * (2 * p - 1)
    games = TARGET / lp_g if lp_g > 0 else float('inf')
    print(f'  {100*p:.0f}%  |  {lp_g:+.1f}  |  {games:.0f}' if lp_g > 0 else f'  {100*p:.0f}%  |  {lp_g:+.1f}  |  never')

print('\n=== 2. games needed for the 95% CI to exclude 50% ===')
for p in (0.51, 0.53, 0.55, 0.58, 0.60, 0.65):
    # normal approx: need 1.96*sqrt(p(1-p)/n) < p-0.5
    n = (1.96 ** 2) * p * (1 - p) / ((p - 0.5) ** 2)
    print(f'  true {100*p:.0f}%: ~{n:.0f} games')

print('\n=== 3. Bayesian read on observed records (uniform prior) ===')
records = [('Annie+Sona ranked', 15, 11), ('Normal draft', 21, 12),
           ('Ranked solo overall', 37, 60), ('Nasus', 6, 15)]
for name, w, l in records:
    post = beta(1 + w, 1 + l)
    print(f'  {name} ({w}W-{l}L): P(true>50%) = {1-post.cdf(0.5):.1%}, P(true>55%) = {1-post.cdf(0.55):.1%}, median = {post.ppf(0.5):.1%}')

# chart: two panels — climb speed curve; posterior densities
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

ps = [x / 1000 for x in range(455, 701)]
games = [TARGET / (LP * (2 * p - 1)) if p > 0.502 else None for p in ps]
ax1.plot([100 * p for p in ps], [g if g and g < 400 else None for g in games], color=BLUE, linewidth=2)
for p, lab in ((0.53, '53% → ~217 games'), (0.58, '58% → ~81 games'), (0.65, '65% → ~43 games')):
    g = TARGET / (LP * (2 * p - 1))
    ax1.plot(100 * p, g, 'o', color=INK, markersize=5)
    ax1.annotate(lab, (100 * p, g), textcoords='offset points', xytext=(8, 8), fontsize=8.5, color=INK2)
ax1.set_xlabel('true win rate (%)'); ax1.set_ylabel('games to climb 325 LP')
ax1.set_title('Why 58% is 3× better than 53%: climb time is nonlinear', color=INK, fontsize=10, loc='left')
ax1.set_ylim(0, 400)

xs = [x / 1000 for x in range(200, 901)]
for (name, w, l), color in zip(records[:3], (BLUE, '#1baf7a', RED)):
    post = beta(1 + w, 1 + l)
    ax2.plot([100 * x for x in xs], [post.pdf(x) for x in xs], color=color, linewidth=2, label=f'{name} ({w}W–{l}L)')
ax2.axvline(50, color=BASE, linewidth=1, linestyle=(0, (3, 3)))
ax2.legend(frameon=False, fontsize=8, labelcolor=INK2, loc='upper left')
ax2.set_xlabel('true win rate (%)'); ax2.set_yticks([])
ax2.set_title('What the records actually tell us (posterior densities)', color=INK, fontsize=10, loc='left')
fig.tight_layout(); fig.savefig(f'{ROOT}/charts/winrate_meaning.png'); plt.close(fig)
print('\nwinrate_meaning.png written')
