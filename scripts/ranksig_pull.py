# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Pull cohort (IRON/SILVER/BRONZE) match bodies + timelines and djfalcon timelines
for the rank-signature model. Resumable: skips files already on disk.

Sampling: league-exp-v4 RANKED_SOLO_5x5/{TIER}/II pages 2-4 (page 1 oversamples
grinders); ~15 players/tier spread across pages; 3 recent queue=420 match ids each;
dedupe; cap ~40 matches/tier. Each match -> match body + timeline, gzipped to
data/ranksig/cohort/{TIER}/{match_id}.json.gz  (body+timeline+tier in one blob).
Skip remakes (<600s). dj timelines -> data/ranksig/dj/{match_id}.tl.json.gz .
"""
import json, os, sys, time, gzip, urllib.request, urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f'{ROOT}/data/ranksig'
DJ_PUUID = 'i-wo_QJ1JiotACrlb6wlsPLYhfVCXCwDogb03ZcheMSsLMUY9MQknG3RqhXtWIwNoKtPuekiB6Oynw'
TIERS = ['IRON', 'SILVER', 'BRONZE']
PAGES = [2, 3, 4]
PLAYERS_PER_TIER = 15          # 5 per page
GAMES_EACH = 3
MATCH_CAP = 42

KEY = os.environ.get('RIOT_API_KEY')
if not KEY and os.path.exists(os.path.expanduser('~/.env')):
    for line in open(os.path.expanduser('~/.env')):
        if line.startswith('RIOT_API_KEY='):
            KEY = line.strip().split('=', 1)[1]
if not KEY:
    raise SystemExit('Set RIOT_API_KEY in the environment or ~/.env')

NCALL = 0
def get(url, tries=15):
    global NCALL
    for _ in range(tries):
        req = urllib.request.Request(url, headers={'X-Riot-Token': KEY, 'User-Agent': 'curl/8.5.0'})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                NCALL += 1
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = int(e.headers.get('Retry-After', '15'))
                print(f'  429, sleeping {wait}s', flush=True)
                time.sleep(wait + 2)
            elif e.code >= 500:
                time.sleep(5)
            elif e.code == 404:
                return None
            elif e.code in (401, 403):
                raise SystemExit(f'AUTH ERROR {e.code} — key likely expired: {url}')
            else:
                raise
    raise RuntimeError(f'gave up on {url}')

def wgz(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, 'wt') as f:
        json.dump(obj, f)

def cohort_pull():
    for tier in TIERS:
        outdir = f'{DATA}/cohort/{tier}'
        os.makedirs(outdir, exist_ok=True)
        # gather players spread across pages 2-4
        players = []
        per_page = max(1, PLAYERS_PER_TIER // len(PAGES))
        for pg in PAGES:
            entries = get(f'https://na1.api.riotgames.com/lol/league-exp/v4/entries/RANKED_SOLO_5x5/{tier}/II?page={pg}') or []
            got = [e for e in entries if e.get('puuid')][:per_page]
            players += got
            print(f'{tier} page {pg}: {len(got)} players', flush=True)
        players = players[:PLAYERS_PER_TIER]
        # collect match ids
        match_ids = []
        for pl in players:
            ids = get(f"https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{pl['puuid']}/ids?queue=420&start=0&count={GAMES_EACH}") or []
            match_ids += ids
        match_ids = sorted(set(match_ids))[:MATCH_CAP]
        print(f'{tier}: {len(match_ids)} unique matches (cap {MATCH_CAP})', flush=True)
        for i, mid in enumerate(match_ids):
            path = f'{outdir}/{mid}.json.gz'
            if os.path.exists(path):
                continue
            body = get(f'https://americas.api.riotgames.com/lol/match/v5/matches/{mid}')
            if not body:
                continue
            if body['info'].get('gameDuration', 0) < 600:
                continue  # remake
            tl = get(f'https://americas.api.riotgames.com/lol/match/v5/matches/{mid}/timeline')
            if not tl:
                continue
            wgz(path, {'tier': tier, 'match_id': mid, 'body': body, 'timeline': tl})
            if (i + 1) % 5 == 0:
                print(f'  {tier} {i+1}/{len(match_ids)} (calls={NCALL})', flush=True)

def dj_pull():
    outdir = f'{DATA}/dj'
    os.makedirs(outdir, exist_ok=True)
    games = json.load(open(f'{ROOT}/data/riot_games.json'))
    rk = [g for g in games if g['queue_id'] == 420 and not g['remake']]
    print(f'dj: {len(rk)} ranked non-remake games', flush=True)
    for i, g in enumerate(rk):
        mid = g['match_id']
        path = f'{outdir}/{mid}.tl.json.gz'
        if os.path.exists(path):
            continue
        tl = get(f'https://americas.api.riotgames.com/lol/match/v5/matches/{mid}/timeline')
        if not tl:
            continue
        # also grab body fresh so we have full participant list for opponent deltas
        body = get(f'https://americas.api.riotgames.com/lol/match/v5/matches/{mid}')
        wgz(path, {'match_id': mid, 'body': body, 'timeline': tl})
        if (i + 1) % 10 == 0:
            print(f'  dj {i+1}/{len(rk)} (calls={NCALL})', flush=True)

if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    t0 = time.time()
    if what in ('all', 'cohort'):
        cohort_pull()
    if what in ('all', 'dj'):
        dj_pull()
    print(f'DONE total calls={NCALL} in {int(time.time()-t0)}s', flush=True)
