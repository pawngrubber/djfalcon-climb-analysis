# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Lagged-panel data pull for the climb-trajectory model (issue 20).

Sampling: league-exp-v4 RANKED_SOLO_5x5/{TIER}/II pages 2-4 (page 1 oversamples
grinders), ~10 players/tier across IRON/BRONZE/SILVER. Per player: up to 40 recent
queue=420 match ids -> MATCH BODIES ONLY (no timelines; budget limitation).

Cache-first: skips any match already cached anywhere under ~/riot_cache (panel or
ranksig/cohort blobs). New bodies gzipped to data/ranksig/panel/bodies/{mid}.json.gz
AND mirrored to ~/riot_cache/panel/bodies/{mid}.json.gz. Player->match mapping and
tier written to data/ranksig/panel/manifest.json. Resumable.
"""
import json, os, sys, time, gzip, glob, urllib.request, urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL = f'{ROOT}/data/ranksig/panel'
BODIES = f'{PANEL}/bodies'
MIRROR = os.path.expanduser('~/riot_cache/panel/bodies')
CACHE_ROOT = os.path.expanduser('~/riot_cache')
TIERS = ['IRON', 'BRONZE', 'SILVER']
DIV = 'II'
PAGES = [2, 3, 4]
PLAYERS_PER_TIER = 10
GAMES_EACH = 40

KEY = os.environ.get('RIOT_API_KEY')
if not KEY and os.path.exists(os.path.expanduser('~/.env')):
    for line in open(os.path.expanduser('~/.env')):
        if line.startswith('RIOT_API_KEY='):
            KEY = line.strip().split('=', 1)[1].strip()
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
                time.sleep(0.4)  # base politeness; 429 handler covers the 2-min window
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

# --- build the set of match ids already cached anywhere, + a locator ---
def build_cache_index():
    """Return (cached_ids:set, locate:dict mid->filepath) across all cache dirs."""
    cached, locate = set(), {}
    patterns = [
        f'{MIRROR}/*.json.gz',
        f'{BODIES}/*.json.gz',
        f'{CACHE_ROOT}/ranksig/cohort/*/*.json.gz',
        f'{ROOT}/data/ranksig/cohort/*/*.json.gz',
    ]
    for pat in patterns:
        for fp in glob.glob(pat):
            mid = os.path.basename(fp).replace('.json.gz', '')
            cached.add(mid)
            locate.setdefault(mid, fp)
    return cached, locate

def load_body_from_cache(fp):
    d = json.load(gzip.open(fp, 'rt'))
    # panel blobs store body at top-level under 'body'; cohort blobs likewise
    return d.get('body')

def main():
    os.makedirs(BODIES, exist_ok=True)
    os.makedirs(MIRROR, exist_ok=True)
    cached, locate = build_cache_index()
    print(f'cache index: {len(cached)} match ids already on disk', flush=True)

    manifest = []
    # 1) gather players per tier across pages 2-4
    for tier in TIERS:
        players = []
        per_page = -(-PLAYERS_PER_TIER // len(PAGES))  # ceil
        for pg in PAGES:
            entries = get(f'https://na1.api.riotgames.com/lol/league-exp/v4/entries/RANKED_SOLO_5x5/{tier}/{DIV}?page={pg}') or []
            got = [e for e in entries if e.get('puuid')][:per_page]
            players += got
            print(f'{tier} page {pg}: {len(got)} players', flush=True)
        players = players[:PLAYERS_PER_TIER]
        for pl in players:
            puuid = pl['puuid']
            ids = get(f'https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?queue=420&start=0&count={GAMES_EACH}') or []
            manifest.append({'puuid': puuid, 'tier_cohort': tier, 'division': DIV,
                             'leaguePoints': pl.get('leaguePoints'),
                             'wins': pl.get('wins'), 'losses': pl.get('losses'),
                             'match_ids': ids})
            print(f'  {tier} player: {len(ids)} match ids', flush=True)
    json.dump(manifest, open(f'{PANEL}/manifest.json', 'w'), indent=1)
    print(f'manifest written: {len(manifest)} players', flush=True)

    # 2) pull bodies for every unique match id, cache-first
    all_ids = []
    for m in manifest:
        all_ids += m['match_ids']
    uniq = sorted(set(all_ids))
    print(f'{len(uniq)} unique match ids across panel', flush=True)

    fetched = skipped = 0
    for i, mid in enumerate(uniq):
        dest = f'{BODIES}/{mid}.json.gz'
        if os.path.exists(dest):
            skipped += 1
            continue
        if mid in cached:
            # reuse a body cached elsewhere (e.g. cohort blob); rewrite as panel blob
            body = load_body_from_cache(locate[mid])
            if body:
                blob = {'match_id': mid, 'body': body,
                        'game_version': body['info'].get('gameVersion'), 'reused_from': locate[mid]}
                wgz(dest, blob)
                wgz(f'{MIRROR}/{mid}.json.gz', blob)
                skipped += 1
                continue
        body = get(f'https://americas.api.riotgames.com/lol/match/v5/matches/{mid}')
        if not body:
            continue
        blob = {'match_id': mid, 'body': body, 'game_version': body['info'].get('gameVersion')}
        wgz(dest, blob)
        wgz(f'{MIRROR}/{mid}.json.gz', blob)
        fetched += 1
        if (fetched + skipped) % 25 == 0:
            print(f'  {i+1}/{len(uniq)} bodies (fetched={fetched} skipped={skipped} calls={NCALL})', flush=True)

    print(f'DONE fetched={fetched} skipped={skipped} total_calls={NCALL}', flush=True)

if __name__ == '__main__':
    main()
