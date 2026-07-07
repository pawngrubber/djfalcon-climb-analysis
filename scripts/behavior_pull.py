# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Pull behavior signals (AFK/leaves, surrenders) for every match in data/riot_games.json.

For each game: the target player's timePlayed vs game duration (a gap = left early),
the same for every teammate and enemy, and surrender flags.

Run: uv run scripts/behavior_pull.py   (RIOT_API_KEY in env or ~/.env)
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
out = []
for i, g in enumerate(games):
    m = get(f"https://americas.api.riotgames.com/lol/match/v5/matches/{g['match_id']}")
    info = m['info']
    dur = info['gameDuration']
    me = next(x for x in info['participants'] if x['puuid'] == PUUID)
    def brief(p):
        return {
            'time_played': p.get('timePlayed'),
            'missing_s': max(0, dur - (p.get('timePlayed') or dur)),
            'ended_in_surrender': p.get('gameEndedInSurrender', False),
            'early_surrender': p.get('gameEndedInEarlySurrender', False),
            'win': p['win'],
        }
    out.append({
        'match_id': g['match_id'],
        'queue_id': g['queue_id'],
        'duration_s': dur,
        'me': brief(me),
        'allies': [brief(p) for p in info['participants'] if p['teamId'] == me['teamId'] and p['puuid'] != PUUID],
        'enemies': [brief(p) for p in info['participants'] if p['teamId'] != me['teamId']],
    })
    if (i + 1) % 25 == 0:
        print(f'{i + 1}/{len(games)}', flush=True)

json.dump(out, open(f'{ROOT}/data/behavior.json', 'w'), indent=1)
print('DONE', len(out))
