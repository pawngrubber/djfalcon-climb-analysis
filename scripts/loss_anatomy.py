# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
loss_anatomy.py -- classify djfalcon#NA1 SOLORANKED losses from op.gg op-score
timelines and measure ahead@15 conversion vs behind@15 comeback rates.

Data: op.gg server responses captured as flight_*.txt / flightB_*.txt (RSC
"flight" format). Each file has a row `N:{"data":[game,...]}`. Each game carries
chartData.data = [{time, up, down}] -- op.gg's proprietary "op score" composite
for the two teams over game time. At any timestamp exactly one of up/down is
populated (a single line that crosses the 5.0 midline); the value is his team's
op-score composite:  >5 = his team ahead, <5 = behind.  Verified empirically:
raw (his-team) orientation makes WIN games out-score LOSE games at every
checkpoint, whereas fixed-side orientations are ~50/50 (noise). op score is a
PROXY for game state, NOT true win probability -- see the sanity check below.

game_type is frequently an RSC back-reference string ($ROW:data:IDX:game_type)
that dedups to an earlier concrete value; we resolve it and keep SOLORANKED only.
Games are deduped by id across all files.
"""
from __future__ import annotations
import glob, json, math, os, re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# capture files live in the scratchpad dir two levels above the repo
SEARCH_DIRS = [os.path.join(ROOT, ".."), ROOT, os.path.join(ROOT, "data")]

ROW_RE = re.compile(r"^(\w+):(\{.*\})\s*$")
TYPEREF_RE = re.compile(r"^\$(\w+):data:(\d+):game_type$")


def find_files() -> list[str]:
    files = []
    for d in SEARCH_DIRS:
        files += glob.glob(os.path.join(d, "flight_*.txt"))
        files += glob.glob(os.path.join(d, "flightB_*.txt"))
    return sorted(set(os.path.abspath(f) for f in files))


def parse_file(path: str):
    """Return (rowmap, data_array) for one flight file."""
    rowmap, data = {}, None
    with open(path) as fh:
        for line in fh:
            m = ROW_RE.match(line)
            if not m:
                continue
            try:
                obj = json.loads(m.group(2))
            except json.JSONDecodeError:
                continue
            rowmap[m.group(1)] = obj
            if (isinstance(obj, dict) and isinstance(obj.get("data"), list)
                    and obj["data"] and isinstance(obj["data"][0], dict)
                    and "chartData" in obj["data"][0]):
                data = obj["data"]
    return rowmap, data


def resolve_type(game: dict, rowmap: dict):
    v = game.get("game_type")
    if isinstance(v, dict):
        return v.get("game_type")
    if isinstance(v, str):
        m = TYPEREF_RE.match(v)
        if m:
            arr = rowmap.get(m.group(1), {}).get("data")
            idx = int(m.group(2))
            if arr and idx < len(arr):
                tv = arr[idx].get("game_type")
                if isinstance(tv, dict):
                    return tv.get("game_type")
    return None


def load_games() -> dict:
    seen = {}
    for path in find_files():
        rowmap, data = parse_file(path)
        if not data:
            continue
        for g in data:
            if "chartData" not in g:
                continue
            g["_type"] = resolve_type(g, rowmap)
            seen[g["id"]] = g  # dedupe by id across files
    return seen


def series(game: dict):
    """List of (time, his_score) with score = up if present else down."""
    out = []
    for p in game["chartData"]["data"]:
        v = p["up"] if p.get("up") is not None else p.get("down")
        if v is not None:
            out.append((p["time"], v))
    return out


def score_at(s, t):
    """Last-observed score at or before time t (step interpolation)."""
    val = None
    for tm, v in s:
        if tm <= t:
            val = v
        else:
            break
    return val


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, (c - h) / d, (c + h) / d)


# time-window constants (seconds)
T10, T14, T20 = 600, 840, 1200


def classify_loss(s):
    """Mutually exclusive taxonomy, precedence = latest point of advantage.
       mid throw > late fade > early collapse > never competitive."""
    ahead_1420 = any(sc > 5 for tm, sc in s if T14 <= tm <= T20)
    even_ahead_20 = any(sc >= 5 for tm, sc in s if tm >= T20)
    ever_above = any(sc > 5 for tm, sc in s if tm > 0)
    if ahead_1420:
        return "mid_throw"
    if even_ahead_20:
        return "late_fade"
    # never above 5 after the 14-20 window from here on
    if ever_above:
        return "early_collapse"   # had an early lead/parity, gave it up, never recovered
    return "never_competitive"    # never once above the midline


def main():
    seen = load_games()
    solo = [g for g in seen.values() if g["_type"] == "SOLORANKED"]
    played = [g for g in solo if g["game_result"] in ("WIN", "LOSE")]
    remakes = [g for g in solo if g["game_result"] == "REMAKE"]
    wins = [g for g in played if g["game_result"] == "WIN"]
    losses = [g for g in played if g["game_result"] == "LOSE"]

    print(f"Files parsed: {len(find_files())}")
    print(f"Unique games (all types): {len(seen)}")
    print(f"  type breakdown: {dict(Counter(g['_type'] for g in seen.values()))}")
    print(f"SOLORANKED: {len(solo)}  (WIN {len(wins)}, LOSE {len(losses)}, REMAKE {len(remakes)})")
    base = len(wins) / len(played) if played else 0
    print(f"Base win rate (played): {len(wins)}/{len(played)} = {base*100:.1f}%")

    # ---- sanity: does op score end on the winner's side? ----
    agree = sum(1 for g in played if (score_at(series(g), 10**9) > 5) == (g["game_result"] == "WIN"))
    print(f"\nSANITY (winner ends >5, raw his-team orientation): "
          f"{agree}/{len(played)} = {agree/len(played)*100:.1f}%")
    for T, lab in ((T10, "10m"), (900, "15m"), (T20, "20m")):
        wv = [score_at(series(g), T) for g in wins]
        lv = [score_at(series(g), T) for g in losses]
        wv = [x for x in wv if x is not None]
        lv = [x for x in lv if x is not None]
        print(f"  op score @{lab}: WIN mean {sum(wv)/len(wv):.2f} | "
              f"LOSE mean {sum(lv)/len(lv):.2f}  (>5 => his team ahead)")

    # ---- loss taxonomy ----
    print("\nLOSS TAXONOMY (n losses = %d)" % len(losses))
    tax = Counter(classify_loss(series(g)) for g in losses)
    order = ["early_collapse", "mid_throw", "late_fade", "never_competitive"]
    label = {"early_collapse": "Early collapse (behind@10, never recovers)",
             "mid_throw": "Mid throw (ahead>5 in 14-20m, lost)",
             "late_fade": "Late fade (even/ahead at 20m+, lost)",
             "never_competitive": "Never competitive (never above midline)"}
    for k in order:
        n = tax.get(k, 0)
        print(f"  {label[k]:52s} {n:2d}  ({n/len(losses)*100:4.1f}%)")

    # ---- conversion @15 (ahead) and comeback @15 (behind), wins+losses ----
    print("\nAHEAD@15 CONVERSION vs BEHIND@15 COMEBACK (played games)")
    ahead = [g for g in played if (score_at(series(g), 900) or 5) > 5]
    behind = [g for g in played if (score_at(series(g), 900) or 5) <= 5]
    for grp, name in ((ahead, "Ahead@15 (>5)"), (behind, "Behind/even@15 (<=5)")):
        n = len(grp)
        w = sum(1 for g in grp if g["game_result"] == "WIN")
        p, lo, hi = wilson(w, n)
        rate = "conversion" if grp is ahead else "comeback"
        print(f"  {name:22s} n={n:2d}  won {w:2d}  {rate} {p*100:4.1f}%  "
              f"(Wilson 95% CI {lo*100:.1f}-{hi*100:.1f}%)")

    # ---- optional: write trimmed games-only JSON into the repo ----
    if os.environ.get("WRITE_DATA"):
        out = os.path.join(ROOT, "data", "opgg_chart_timelines.json")
        trimmed = [{
            "id": g["id"], "created_at": g.get("created_at"),
            "game_type": g["_type"], "game_result": g["game_result"],
            "game_length": g.get("game_length"),
            "summoner_team": g.get("summoner_team"),
            "chartData": g["chartData"]["data"],
        } for g in solo]
        with open(out, "w") as fh:
            json.dump(trimmed, fh)
        print(f"\nWrote {len(trimmed)} SOLORANKED timelines -> {out}")


if __name__ == "__main__":
    main()
