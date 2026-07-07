# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy"]
# ///
"""Detect repeated teammates (likely duo partners) for djfalcon#NA1.

Reads data/riot_games.json ranked-solo (queue_id 420) match_ids, fetches each
match from the Riot API, records the 4 teammate PUUIDs + win/loss per game, then
finds any teammate appearing in >=3 games. For each candidate duo: games together,
W-L together, and his W-L in games WITHOUT them, Wilson 95% CIs, Fisher exact.
Also overall: games with any repeat-teammate vs games with none.

Writes data/teammates.json. Run: uv run scripts/duo_analysis.py
(RIOT_API_KEY in env or ~/.env)
"""
import json, os, time, urllib.request, urllib.error
from collections import defaultdict

from scipy.stats import fisher_exact

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME_NAME, TAG_LINE = 'djfalcon', 'NA1'

KEY = os.environ.get('RIOT_API_KEY')
if not KEY and os.path.exists(os.path.expanduser('~/.env')):
    for line in open(os.path.expanduser('~/.env')):
        if line.startswith('RIOT_API_KEY='):
            KEY = line.strip().split('=', 1)[1]
if not KEY:
    raise SystemExit('Set RIOT_API_KEY in the environment or ~/.env')


def get(url, tries=6):
    for _ in range(tries):
        req = urllib.request.Request(url, headers={'X-Riot-Token': KEY, 'User-Agent': 'curl/8.5.0'})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = int(e.headers.get('Retry-After', '10'))
                print(f'429, sleeping {wait}s', flush=True)
                time.sleep(wait + 1)
            elif e.code >= 500:
                time.sleep(5)
            else:
                raise
    raise RuntimeError(f'gave up on {url}')


def wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (p, max(0, c - h), min(1, c + h))


acct = get(f'https://americas.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{GAME_NAME}/{TAG_LINE}')
PUUID = acct['puuid']
print(f'djfalcon puuid: {PUUID}', flush=True)

games = json.load(open(f'{ROOT}/data/riot_games.json'))
solo = [g for g in games if g.get('queue_id') == 420]
print(f'{len(solo)} ranked-solo (420) matches', flush=True)

# per-game record: match_id, win, list of 4 teammate puuids
per_game = []
name_of = {}  # puuid -> most recent riotIdGameName#tagLine (public, convenience only)
for i, g in enumerate(solo):
    m = get(f"https://americas.api.riotgames.com/lol/match/v5/matches/{g['match_id']}")
    info = m['info']
    parts = info['participants']
    me = next((x for x in parts if x['puuid'] == PUUID), None)
    if me is None:
        print(f"WARN {g['match_id']}: djfalcon not found, skipping", flush=True)
        continue
    win = bool(me['win'])
    mates = [p for p in parts if p['teamId'] == me['teamId'] and p['puuid'] != PUUID]
    mate_puuids = [p['puuid'] for p in mates]
    for p in mates:
        gn = p.get('riotIdGameName') or ''
        tl = p.get('riotIdTagline') or ''
        if gn:
            name_of[p['puuid']] = f'{gn}#{tl}' if tl else gn
    per_game.append({
        'match_id': g['match_id'],
        'game_creation_utc': g.get('game_creation_utc'),
        'win': win,
        'teammate_puuids': mate_puuids,
    })
    if (i + 1) % 20 == 0:
        print(f'{i + 1}/{len(solo)}', flush=True)

N = len(per_game)
overall_w = sum(1 for r in per_game if r['win'])
print(f'\nfetched {N} games, overall {overall_w}-{N - overall_w}', flush=True)

# teammate -> list of game indices
appearances = defaultdict(list)
for idx, r in enumerate(per_game):
    for pu in r['teammate_puuids']:
        appearances[pu].append(idx)

# candidate duos: teammate in >=3 games
candidates = {pu: idxs for pu, idxs in appearances.items() if len(idxs) >= 3}


def wl(idxs):
    w = sum(1 for i in idxs if per_game[i]['win'])
    return w, len(idxs)


def fisher_two(with_idxs, without_idxs):
    w1, n1 = wl(with_idxs)
    w2, n2 = wl(without_idxs)
    odds, p = fisher_exact([[w1, n1 - w1], [w2, n2 - w2]])
    return w1, n1, w2, n2, p


all_idx = set(range(N))
duo_results = []
for pu, idxs in sorted(candidates.items(), key=lambda kv: -len(kv[1])):
    with_set = set(idxs)
    without = sorted(all_idx - with_set)
    w1, n1, w2, n2, pval = fisher_two(idxs, without)
    pw, wlo, whi = wilson(w1, n1)
    pwo, wolo, wohi = wilson(w2, n2)
    duo_results.append({
        'puuid': pu,
        'name': name_of.get(pu),
        'games_together': n1,
        'wl_together': [w1, n1 - w1],
        'wr_together': pw,
        'wr_together_ci95': [wlo, whi],
        'wl_without': [w2, n2 - w2],
        'wr_without': pwo,
        'wr_without_ci95': [wolo, wohi],
        'fisher_p': pval,
    })

# overall: games with ANY repeat-teammate vs games with NONE
repeat_puuids = set(candidates)
with_any, with_none = [], []
for idx, r in enumerate(per_game):
    if any(pu in repeat_puuids for pu in r['teammate_puuids']):
        with_any.append(idx)
    else:
        with_none.append(idx)
aw, an, nw, nn, ap = fisher_two(with_any, with_none)
pa, alo, ahi = wilson(aw, an)
pn, nlo, nhi = wilson(nw, nn)

overall = {
    'games_with_any_repeat': {'wl': [aw, an - aw], 'wr': pa, 'ci95': [alo, ahi]},
    'games_with_none': {'wl': [nw, nn - nw], 'wr': pn, 'ci95': [nlo, nhi]},
    'fisher_p': ap,
}

# ---- save ----
out = {
    'djfalcon_puuid': PUUID,
    'n_games': N,
    'overall_wl': [overall_w, N - overall_w],
    'per_game': per_game,
    'repeat_teammates': duo_results,
    'repeat_vs_none': overall,
}
json.dump(out, open(f'{ROOT}/data/teammates.json', 'w'), indent=1)

# ---- print report ----
print('\n==== REPEAT TEAMMATES (>=3 games) ====')
if not duo_results:
    print('none — no teammate appeared in 3+ ranked-solo games')
for d in duo_results:
    nm = d['name'] or d['puuid'][:12]
    w, l = d['wl_together']
    w2, l2 = d['wl_without']
    print(f"{nm}: together {w}-{l} ({100*d['wr_together']:.0f}% "
          f"[{100*d['wr_together_ci95'][0]:.0f}-{100*d['wr_together_ci95'][1]:.0f}])  "
          f"without {w2}-{l2} ({100*d['wr_without']:.0f}% "
          f"[{100*d['wr_without_ci95'][0]:.0f}-{100*d['wr_without_ci95'][1]:.0f}])  "
          f"Fisher p={d['fisher_p']:.3f}")

print('\n==== ANY REPEAT-TEAMMATE vs NONE ====')
print(f"with any repeat: {aw}-{an - aw} ({100*pa:.0f}% [{100*alo:.0f}-{100*ahi:.0f}])")
print(f"with none:       {nw}-{nn - nw} ({100*pn:.0f}% [{100*nlo:.0f}-{100*nhi:.0f}])")
print(f"Fisher p={ap:.3f}")
print('\nDONE, wrote data/teammates.json')
