# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy"]
# ///
"""Role-normalized comparison: djfalcon vs same-position players in his own lobbies.

Part 1 -- for each position he plays (TOP, MIDDLE, UTILITY, BOTTOM), compare HIS
per-game distributions (lobby is_target rows) against the OTHER players in that
same position (lobby non-target rows) on: deaths/game, deaths/min, cs/min
(skip UTILITY), kill participation, vision score/min.

Part 2 -- champion-normalized: his ranked (queue 420) rows for Annie/Sona/Nasus
(from riot_games.json) vs same-position lobby cohort (Annie->MIDDLE, Sona->UTILITY,
Nasus->TOP). Same metrics.

For each metric: his mean, cohort mean, his-mean percentile in cohort distribution,
Mann-Whitney U two-sided p, rank-biserial effect size (his group vs cohort).

Run: uv run scripts/role_normalized.py
"""
import json, os
from scipy.stats import mannwhitneyu
from scipy.stats import percentileofscore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

lob = json.load(open(f'{ROOT}/data/lobby_benchmarks.json'))
riot = json.load(open(f'{ROOT}/data/riot_games.json'))
solo = [r for r in riot if r['queue_id'] == 420 and not r['remake']]

# Metric extractors. Each returns a per-game float from a row dict.
# kp lives in 'kp' (lobby) or 'kill_participation' (riot); handled per source.
def make_metrics(kp_key):
    return [
        ('deaths/game', lambda r: r['death']),
        ('deaths/min',  lambda r: r['death'] / r['duration_min']),
        ('cs/min',      lambda r: r['cs_per_min']),
        ('kp',          lambda r: r[kp_key]),
        ('vis/min',     lambda r: r['vision_score'] / r['duration_min']),
    ]

def mean(xs):
    return sum(xs) / len(xs)

def compare(his_vals, cohort_vals):
    """Return (his_mean, cohort_mean, percentile, p, rank_biserial)."""
    hm, cm = mean(his_vals), mean(cohort_vals)
    pct = percentileofscore(cohort_vals, hm, kind='mean')
    # Mann-Whitney: his group first so U1 -> positive rb means his values larger
    U1, p = mannwhitneyu(his_vals, cohort_vals, alternative='two-sided')
    n1, n2 = len(his_vals), len(cohort_vals)
    rb = 2 * U1 / (n1 * n2) - 1
    return hm, cm, pct, p, rb

def run_block(title, his_rows, cohort_rows, his_metrics, cohort_metrics, skip=()):
    print(f'\n### {title}  (his n={len(his_rows)}, cohort n={len(cohort_rows)})')
    print('| metric | his mean | cohort mean | his pctile | MWU p (2-sided) | rank-biserial |')
    print('|---|---|---|---|---|---|')
    for (name, hf), (_, cf) in zip(his_metrics, cohort_metrics):
        if name in skip:
            continue
        # drop rows where the metric is None (only kp/kill_participation has nulls)
        his_vals = [v for r in his_rows if (v := hf(r)) is not None]
        coh_vals = [v for r in cohort_rows if (v := cf(r)) is not None]
        note = ''
        if len(his_vals) < len(his_rows) or len(coh_vals) < len(cohort_rows):
            note = f' (n {len(his_vals)}/{len(coh_vals)})'
        hm, cm, pct, p, rb = compare(his_vals, coh_vals)
        print(f'| {name}{note} | {hm:.3f} | {cm:.3f} | {pct:.1f} | {p:.4g} | {rb:+.3f} |')

lob_metrics = make_metrics('kp')
riot_metrics = make_metrics('kill_participation')

POSITIONS = ['TOP', 'MIDDLE', 'UTILITY', 'BOTTOM']

print('=' * 70)
print('PART 1 -- ROLE-NORMALIZED (lobby is_target vs same-position non-target)')
print('=' * 70)
for pos in POSITIONS:
    his = [r for r in lob if r['is_target'] and r['position'] == pos]
    coh = [r for r in lob if not r['is_target'] and r['position'] == pos]
    skip = ('cs/min',) if pos == 'UTILITY' else ()
    run_block(f'Position {pos}', his, coh, lob_metrics, lob_metrics, skip=skip)

print('\n' + '=' * 70)
print('PART 2 -- CHAMPION-NORMALIZED (his ranked champ rows vs same-position cohort)')
print('=' * 70)
CHAMP_POS = [('Annie', 'MIDDLE'), ('Sona', 'UTILITY'), ('Nasus', 'TOP')]
for champ, pos in CHAMP_POS:
    his = [r for r in solo if r['champion'] == champ]
    coh = [r for r in lob if not r['is_target'] and r['position'] == pos]
    skip = ('cs/min',) if pos == 'UTILITY' else ()
    run_block(f'{champ} vs other {pos}', his, coh, riot_metrics, lob_metrics, skip=skip)

print('\n' + '=' * 70)
print('SANITY -- sample sizes')
print('=' * 70)
for pos in POSITIONS:
    his = [r for r in lob if r['is_target'] and r['position'] == pos]
    coh = [r for r in lob if not r['is_target'] and r['position'] == pos]
    flag = '  <-- SMALL, interpret with caution' if len(his) < 12 else ''
    print(f'{pos:8s} his n={len(his):3d}  cohort n={len(coh):3d}{flag}')
for champ, pos in CHAMP_POS:
    his = [r for r in solo if r['champion'] == champ]
    flag = '  <-- SMALL, interpret with caution' if len(his) < 12 else ''
    print(f'{champ:8s} his n={len(his):3d}  (vs {pos}){flag}')
