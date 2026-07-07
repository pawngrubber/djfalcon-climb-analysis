# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Extract per-player-game features (Family A controllable + Family B outcome)
from cohort and djfalcon match bodies + timelines pulled by ranksig_pull.py.

Writes data/ranksig_features.csv (one row per player-game).
Family A = controllable behavior (lever set). Family B = outcome-contaminated.
"""
import gzip, json, os, glob, math, csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f'{ROOT}/data/ranksig'
DJ_PUUID = 'i-wo_QJ1JiotACrlb6wlsPLYhfVCXCwDogb03ZcheMSsLMUY9MQknG3RqhXtWIwNoKtPuekiB6Oynw'

# --- legendary item set from Data Dragon ---
items = json.load(open(f'{DATA}/ddragon_items.json'))['data']
LEGENDARY = set()
for iid, it in items.items():
    g = it.get('gold', {})
    tags = it.get('tags', [])
    if g.get('total', 0) >= 1600 and not it.get('into') and 'Consumable' not in tags and 'Trinket' not in tags:
        LEGENDARY.add(int(iid))

def frame_at(frames, target_ms):
    """participantFrames dict at frame whose timestamp is closest <= target (else nearest)."""
    best = frames[0]
    for fr in frames:
        if fr['timestamp'] <= target_ms:
            best = fr
        else:
            break
    return best

def pf_val(frames, target_ms, pid, key):
    fr = frame_at(frames, target_ms)
    return fr['participantFrames'][str(pid)].get(key)

def cs_at(frames, target_ms, pid):
    pf = frame_at(frames, target_ms)['participantFrames'][str(pid)]
    return (pf.get('minionsKilled', 0) or 0) + (pf.get('jungleMinionsKilled', 0) or 0)

def extract_match(tier, mid, body, timeline, source, only_puuid=None):
    info = body['info']
    tinfo = timeline['info']
    frames = tinfo['frames']
    dur = info['gameDuration']
    game_min = dur / 60.0
    parts = info['participants']
    # participantId -> participant, puuid map, team positions
    by_pid = {p['participantId']: p for p in parts}
    # lane opponent map: same teamPosition, opposite team
    opp = {}
    for p in parts:
        pos = p.get('teamPosition')
        if not pos or pos == 'Invalid':
            continue
        for q in parts:
            if q['teamId'] != p['teamId'] and q.get('teamPosition') == pos:
                opp[p['participantId']] = q['participantId']
                break
    # team-level tallies (whole game) from events
    team_dragons = {100: 0, 200: 0}
    team_towers = {100: 0, 200: 0}
    # per-pid event tallies
    kills_pre14 = {p['participantId']: 0 for p in parts}
    assists_pre14 = {p['participantId']: 0 for p in parts}
    team_kills_pre14 = {100: 0, 200: 0}
    deaths_pre10 = {p['participantId']: 0 for p in parts}
    first_death_ms = {}
    wards_pre20 = {p['participantId']: 0 for p in parts}
    wardkills_pre20 = {p['participantId']: 0 for p in parts}
    cw_pre20 = {p['participantId']: 0 for p in parts}
    first_cw_ms = {}
    first_legendary_ms = {}
    for fr in frames:
        for e in fr['events']:
            t = e['type']; ts = e.get('timestamp', 0)
            if t == 'CHAMPION_KILL':
                k = e.get('killerId'); v = e.get('victimId')
                if v in first_death_ms:
                    pass
                elif v is not None:
                    first_death_ms[v] = ts
                if v is not None and ts < 600000:
                    deaths_pre10[v] = deaths_pre10.get(v, 0) + 1
                if ts < 840000:
                    if k in by_pid:
                        kills_pre14[k] = kills_pre14.get(k, 0) + 1
                        team_kills_pre14[by_pid[k]['teamId']] += 1
                    for a in e.get('assistingParticipantIds', []) or []:
                        assists_pre14[a] = assists_pre14.get(a, 0) + 1
            elif t == 'ELITE_MONSTER_KILL' and e.get('monsterType') == 'DRAGON':
                tid = e.get('killerTeamId') or (by_pid.get(e.get('killerId'), {}).get('teamId'))
                if tid in team_dragons:
                    team_dragons[tid] += 1
            elif t == 'BUILDING_KILL' and e.get('buildingType') == 'TOWER_BUILDING':
                # killerTeamId is the team that lost? Riot: teamId = team that owned the building.
                owner = e.get('teamId')
                killer_team = 100 if owner == 200 else 200
                team_towers[killer_team] = team_towers.get(killer_team, 0) + 1
            elif t == 'WARD_PLACED' and ts < 1200000:
                c = e.get('creatorId')
                if c in wards_pre20:
                    wards_pre20[c] += 1
            elif t == 'WARD_KILL' and ts < 1200000:
                k = e.get('killerId')
                if k in wardkills_pre20:
                    wardkills_pre20[k] += 1
            elif t == 'ITEM_PURCHASED':
                pid = e.get('participantId'); iid = e.get('itemId')
                if iid == 2055 and ts < 1200000 and pid in cw_pre20:
                    cw_pre20[pid] += 1
                    if pid not in first_cw_ms:
                        first_cw_ms[pid] = ts
                if iid in LEGENDARY and pid not in first_legendary_ms:
                    first_legendary_ms[pid] = ts

    rows = []
    for p in parts:
        pid = p['participantId']
        pos = p.get('teamPosition') or 'Invalid'
        if only_puuid and p['puuid'] != only_puuid:
            continue
        cs10 = cs_at(frames, 600000, pid)
        cs14 = cs_at(frames, 840000, pid)
        cs20 = cs_at(frames, min(1200000, frames[-1]['timestamp']), pid)
        t20_min = min(20.0, frames[-1]['timestamp'] / 60000.0)
        cspm_0_10 = cs10 / 10.0
        cspm_10_20 = (cs20 - cs10) / max(1e-6, (t20_min - 10.0)) if t20_min > 10 else float('nan')
        xp10 = pf_val(frames, 600000, pid, 'xp') or 0
        gold10 = pf_val(frames, 600000, pid, 'totalGold') or 0
        # opponent deltas
        o = opp.get(pid)
        if o:
            cs10_delta = cs10 - cs_at(frames, 600000, o)
            xp10_delta = xp10 - (pf_val(frames, 600000, o, 'xp') or 0)
            gold10_delta = gold10 - (pf_val(frames, 600000, o, 'totalGold') or 0)
        else:
            cs10_delta = xp10_delta = gold10_delta = float('nan')
        # ttfd
        fd = first_death_ms.get(pid)
        first_death_min = fd / 60000.0 if fd else float('nan')
        died = 1 if fd else 0
        # early kp
        tk14 = team_kills_pre14[p['teamId']]
        ka = kills_pre14.get(pid, 0) + assists_pre14.get(pid, 0)
        early_kp = ka / tk14 if tk14 else 0.0
        # ward cadence
        wards_pm_pre20 = wards_pre20[pid] / 20.0
        first_cw_min = first_cw_ms[pid] / 60000.0 if pid in first_cw_ms else float('nan')
        first_leg_min = first_legendary_ms[pid] / 60000.0 if pid in first_legendary_ms else float('nan')
        # family B (outcome-contaminated)
        team = next((t for t in info['teams'] if t['teamId'] == p['teamId']), None)
        team_gold = sum(x['goldEarned'] for x in parts if x['teamId'] == p['teamId'])
        team_dmg = sum(x['totalDamageDealtToChampions'] for x in parts if x['teamId'] == p['teamId'])
        team_kills_final = sum(x['kills'] for x in parts if x['teamId'] == p['teamId'])
        kda = (p['kills'] + p['assists']) / p['deaths'] if p['deaths'] else (p['kills'] + p['assists'])
        rows.append({
            'source': source, 'tier': tier, 'match_id': mid, 'participant_id': pid,
            'is_dj': 1 if p['puuid'] == DJ_PUUID else 0,
            'position': pos, 'champion': p['championName'], 'win': int(p['win']),
            'game_version': info.get('gameVersion', ''),
            'game_min': round(game_min, 2),
            # Family A
            'cs10': cs10, 'cs14': cs14, 'cspm_0_10': round(cspm_0_10, 3),
            'cspm_10_20': round(cspm_10_20, 3) if cspm_10_20 == cspm_10_20 else '',
            'cs10_delta': cs10_delta if cs10_delta == cs10_delta else '',
            'xp10_delta': xp10_delta if xp10_delta == xp10_delta else '',
            'gold10_delta': gold10_delta if gold10_delta == gold10_delta else '',
            'deaths_pre10': deaths_pre10.get(pid, 0),
            'first_death_min': round(first_death_min, 3) if first_death_min == first_death_min else '',
            'died': died,
            'wards_pm_pre20': round(wards_pm_pre20, 4),
            'wardkills_pre20': wardkills_pre20[pid],
            'cw_pre20': cw_pre20[pid],
            'first_cw_min': round(first_cw_min, 3) if first_cw_min == first_cw_min else '',
            'first_leg_min': round(first_leg_min, 3) if first_leg_min == first_leg_min else '',
            'early_kp': round(early_kp, 4),
            # Family B
            'b_team_dragons': team_dragons[p['teamId']],
            'b_team_towers': team_towers.get(p['teamId'], 0),
            'b_kda': round(kda, 3),
            'b_gold_share': round(p['goldEarned'] / team_gold, 4) if team_gold else '',
            'b_dmg_share': round(p['totalDamageDealtToChampions'] / team_dmg, 4) if team_dmg else '',
            'b_kp': round((p['kills'] + p['assists']) / team_kills_final, 4) if team_kills_final else '',
            'b_game_min': round(game_min, 2),
            'b_win': int(p['win']),
        })
    return rows

def main():
    all_rows = []
    # cohort
    for f in sorted(glob.glob(f'{DATA}/cohort/*/*.json.gz')):
        d = json.load(gzip.open(f, 'rt'))
        try:
            all_rows += extract_match(d['tier'], d['match_id'], d['body'], d['timeline'], 'cohort')
        except Exception as e:
            print('SKIP cohort', d.get('match_id'), e)
    # dj (only dj's own row)
    for f in sorted(glob.glob(f'{DATA}/dj/*.tl.json.gz')):
        d = json.load(gzip.open(f, 'rt'))
        if not d.get('body'):
            print('SKIP dj no body', d.get('match_id')); continue
        try:
            all_rows += extract_match('DJ', d['match_id'], d['body'], d['timeline'], 'dj', only_puuid=DJ_PUUID)
        except Exception as e:
            print('SKIP dj', d.get('match_id'), e)
    cols = list(all_rows[0].keys())
    out = f'{ROOT}/data/ranksig_features.csv'
    with open(out, 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)
    from collections import Counter
    print('rows', len(all_rows))
    print('by tier', Counter(r['tier'] for r in all_rows))
    print('matches by tier', {t: len(set(r['match_id'] for r in all_rows if r['tier'] == t)) for t in ['IRON','SILVER','BRONZE','DJ']})
    print('LEGENDARY items', len(LEGENDARY))
    print('->', out)

if __name__ == '__main__':
    main()
