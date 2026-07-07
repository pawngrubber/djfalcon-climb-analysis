# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Proxy benchmark data: every OTHER player in djfalcon's ranked lobbies.

His matchmaking pool is the exact population 'normal for this rank' should mean —
so pull all 10 participants' stats from each ranked game (~930 player-games) to
build empirical distributions of deaths, CS/min, KP, vision, control wards.

Run: uv run scripts/lobby_pull.py   (RIOT_API_KEY in env or ~/.env)
"""
import json, os, time, urllib.request, urllib.error

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

acct = get(f'https://americas.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{GAME_NAME}/{TAG_LINE}')
PUUID = acct['puuid']

games = json.load(open(f'{ROOT}/data/riot_games.json'))
ranked = [g for g in games if g['queue_id'] == 420 and not g['remake']]
out = []
for i, g in enumerate(ranked):
    m = get(f"https://americas.api.riotgames.com/lol/match/v5/matches/{g['match_id']}")
    info = m['info']
    dur = info['gameDuration']
    mins = dur / 60 if dur else 1
    team_kills = {}
    for p in info['participants']:
        team_kills[p['teamId']] = team_kills.get(p['teamId'], 0) + p['kills']
    for p in info['participants']:
        tk = team_kills.get(p['teamId'], 0)
        out.append({
            'match_id': g['match_id'],
            'is_target': p['puuid'] == PUUID,
            'position': p.get('teamPosition') or p.get('individualPosition'),
            'win': p['win'],
            'kill': p['kills'], 'death': p['deaths'], 'assist': p['assists'],
            'kp': round((p['kills'] + p['assists']) / tk, 3) if tk else None,
            'cs_per_min': round((p['totalMinionsKilled'] + p['neutralMinionsKilled']) / mins, 2),
            'control_wards': p.get('visionWardsBoughtInGame'),
            'vision_score': p.get('visionScore'),
            'duration_min': round(mins, 1),
        })
    if (i + 1) % 25 == 0:
        print(f'{i + 1}/{len(ranked)}', flush=True)

json.dump(out, open(f'{ROOT}/data/lobby_benchmarks.json', 'w'), indent=1)
print('DONE', len(out), 'player-games from', len(ranked), 'ranked matches')
