# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Pull all matches for a Riot ID since a start date, via the official match-v5 API.

Usage:
    RIOT_API_KEY=RGAPI-... uv run riot_pull.py
    (or put RIOT_API_KEY=... in ~/.env)

The API key is read from the environment or ~/.env — never hardcode it.
Get a key at https://developer.riotgames.com (dev keys expire every 24h).
"""
import json, os, time, urllib.request, urllib.error

GAME_NAME = 'djfalcon'
TAG_LINE = 'NA1'
START = 1767225600  # 2026-01-01 00:00 UTC
OUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data'

KEY = os.environ.get('RIOT_API_KEY')
if not KEY and os.path.exists(os.path.expanduser('~/.env')):
    for line in open(os.path.expanduser('~/.env')):
        if line.startswith('RIOT_API_KEY='):
            KEY = line.strip().split('=', 1)[1]
if not KEY:
    raise SystemExit('Set RIOT_API_KEY in the environment or ~/.env')

def get(url, tries=6):
    for _ in range(tries):
        # Riot's edge blocks the default Python-urllib user agent, hence the override
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

ids, start = [], 0
while True:
    batch = get(f'https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{PUUID}/ids?startTime={START}&start={start}&count=100')
    ids += batch
    print(f'ids so far: {len(ids)}', flush=True)
    if len(batch) < 100:
        break
    start += 100

rows = []
for i, mid in enumerate(ids):
    m = get(f'https://americas.api.riotgames.com/lol/match/v5/matches/{mid}')
    info = m['info']
    p = next(x for x in info['participants'] if x['puuid'] == PUUID)
    dur = info['gameDuration']
    mins = dur / 60 if dur else 1
    # Arena games use subteams whose ids aren't in info['teams']
    team = next((t for t in info['teams'] if t['teamId'] == p['teamId']), None)
    team_kills = sum(x['kills'] for x in info['participants'] if x['teamId'] == p['teamId'])
    rows.append({
        'match_id': mid,
        'game_creation_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(info['gameCreation'] / 1000)),
        'queue_id': info['queueId'],
        'game_mode': info['gameMode'],
        'duration_s': dur,
        'duration_min': round(mins, 1),
        'remake': p.get('gameEndedInEarlySurrender', False),
        'win': p['win'],
        'champion': p['championName'],
        'position': p.get('teamPosition') or p.get('individualPosition'),
        'kill': p['kills'], 'death': p['deaths'], 'assist': p['assists'],
        'kda_ratio': round((p['kills'] + p['assists']) / p['deaths'], 2) if p['deaths'] else None,
        'kill_participation': round((p['kills'] + p['assists']) / team_kills, 3) if team_kills else None,
        'team_kills': team_kills,
        'cs_lane': p['totalMinionsKilled'],
        'cs_jungle': p['neutralMinionsKilled'],
        'cs_total': p['totalMinionsKilled'] + p['neutralMinionsKilled'],
        'cs_per_min': round((p['totalMinionsKilled'] + p['neutralMinionsKilled']) / mins, 2),
        'gold': p['goldEarned'],
        'dmg_to_champs': p['totalDamageDealtToChampions'],
        'dmg_taken': p['totalDamageTaken'],
        'vision_score': p.get('visionScore'),
        'wards_placed': p.get('wardsPlaced'),
        'control_wards': p.get('visionWardsBoughtInGame'),
        'level': p['champLevel'],
        'largest_multikill': p['largestMultiKill'],
        'first_blood': p.get('firstBloodKill', False),
        'turret_kills': p.get('turretKills', 0),
        'objectives_baron': team['objectives']['baron']['kills'] if team else None,
        'objectives_dragon': team['objectives']['dragon']['kills'] if team else None,
        'game_version': info['gameVersion'],
    })
    if (i + 1) % 20 == 0:
        print(f'details: {i + 1}/{len(ids)}', flush=True)

rows.sort(key=lambda r: r['game_creation_utc'])
json.dump(rows, open(f'{OUT_DIR}/riot_games.json', 'w'), indent=1)
cols = list(rows[0].keys())
with open(f'{OUT_DIR}/riot_games.csv', 'w') as f:
    f.write(','.join(cols) + '\n')
    for r in rows:
        f.write(','.join('' if r[c] is None else str(r[c]) for c in cols) + '\n')
print('DONE', len(rows), 'games;', rows[0]['game_creation_utc'], '->', rows[-1]['game_creation_utc'])
