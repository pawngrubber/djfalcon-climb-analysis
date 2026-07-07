# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Sample a rank cohort's ranked games and harvest every participant's stats.

Usage: uv run scripts/cohort_pull.py IRON|BRONZE|SILVER [division] [n_players] [games_each]
Writes data/cohort_<tier>.json. RIOT_API_KEY from env or ~/.env.

Sampling: league-exp-v4 entries for <TIER> <division> -> n players -> their recent
ranked (queue 420) matches -> all 10 participants per match. Participants in a
tier-X player's lobby are matched to tier-X MMR, mirroring how the 'his lobbies'
cohort was built — the same proxy logic for every tier keeps cohorts comparable.
"""
import json, os, sys, time, urllib.request, urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIER = sys.argv[1].upper()
DIV = sys.argv[2] if len(sys.argv) > 2 else 'II'
N_PLAYERS = int(sys.argv[3]) if len(sys.argv) > 3 else 12
GAMES_EACH = int(sys.argv[4]) if len(sys.argv) > 4 else 3

KEY = os.environ.get('RIOT_API_KEY')
if not KEY and os.path.exists(os.path.expanduser('~/.env')):
    for line in open(os.path.expanduser('~/.env')):
        if line.startswith('RIOT_API_KEY='):
            KEY = line.strip().split('=', 1)[1]
if not KEY:
    raise SystemExit('Set RIOT_API_KEY in the environment or ~/.env')

def get(url, tries=15):
    for _ in range(tries):
        req = urllib.request.Request(url, headers={'X-Riot-Token': KEY, 'User-Agent': 'curl/8.5.0'})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = int(e.headers.get('Retry-After', '15'))
                print(f'429, sleeping {wait}s', flush=True)
                time.sleep(wait + 2)
            elif e.code >= 500:
                time.sleep(5)
            elif e.code == 404:
                return None
            else:
                raise
    raise RuntimeError(f'gave up on {url}')

entries = get(f'https://na1.api.riotgames.com/lol/league-exp/v4/entries/RANKED_SOLO_5x5/{TIER}/{DIV}?page=1')
players = [e for e in entries if e.get('puuid')][:N_PLAYERS]
print(f'{TIER} {DIV}: {len(players)} players sampled', flush=True)

match_ids = []
for pl in players:
    ids = get(f"https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{pl['puuid']}/ids?queue=420&start=0&count={GAMES_EACH}") or []
    match_ids += ids
match_ids = sorted(set(match_ids))
print(f'{len(match_ids)} unique matches to fetch', flush=True)

out = []
for i, mid in enumerate(match_ids):
    m = get(f'https://americas.api.riotgames.com/lol/match/v5/matches/{mid}')
    if not m:
        continue
    info = m['info']
    dur = info['gameDuration']
    if dur < 600:  # skip remakes
        continue
    mins = dur / 60
    team_kills, team_dmg = {}, {}
    for p in info['participants']:
        team_kills[p['teamId']] = team_kills.get(p['teamId'], 0) + p['kills']
        team_dmg[p['teamId']] = team_dmg.get(p['teamId'], 0) + p['totalDamageDealtToChampions']
    for p in info['participants']:
        tk, td = team_kills[p['teamId']], team_dmg[p['teamId']]
        out.append({
            'match_id': mid, 'tier_cohort': TIER,
            'position': p.get('teamPosition') or p.get('individualPosition'),
            'champion': p['championName'], 'win': p['win'],
            'kill': p['kills'], 'death': p['deaths'], 'assist': p['assists'],
            'kp': round((p['kills'] + p['assists']) / tk, 3) if tk else None,
            'cs_per_min': round((p['totalMinionsKilled'] + p['neutralMinionsKilled']) / mins, 2),
            'gold_per_min': round(p['goldEarned'] / mins, 1),
            'dmg_share': round(p['totalDamageDealtToChampions'] / td, 3) if td else None,
            'vision_per_min': round((p.get('visionScore') or 0) / mins, 3),
            'control_wards': p.get('visionWardsBoughtInGame'),
            'deaths_per_min': round(p['deaths'] / mins, 3),
            'duration_min': round(mins, 1),
        })
    if (i + 1) % 10 == 0:
        print(f'{i + 1}/{len(match_ids)}', flush=True)

path = f'{ROOT}/data/cohort_{TIER.lower()}.json'
json.dump(out, open(path, 'w'), indent=1)
print(f'DONE {len(out)} player-games -> {path}')
