# /// script
# requires-python = ">=3.10"
# dependencies = ["scipy"]
# ///
"""Compare djfalcon#NA1's OWN play in normal draft (400) vs ranked solo (420).

Question: does he PLAY differently in normals (looser / better), or just WIN
more with the same play (pointing at opponent strength / draft differences)?
Remakes excluded. Local data only.
"""
import json
import os
import collections
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "riot_games.json")

NORMAL, RANKED = 400, 420


def load():
    games = json.load(open(DATA))
    return [g for g in games if g["queue_id"] in (NORMAL, RANKED) and not g["remake"]]


def deaths_per_min(g):
    return g["death"] / g["duration_min"] if g["duration_min"] else 0.0


def vision_per_min(g):
    return g["vision_score"] / g["duration_min"] if g["duration_min"] else 0.0


def gold_per_min(g):
    return g["gold"] / g["duration_min"] if g["duration_min"] else 0.0


METRICS = {
    "deaths/min": deaths_per_min,
    "cs/min": lambda g: g["cs_per_min"],
    "kill_participation": lambda g: g["kill_participation"],
    "vision/min": vision_per_min,
    "gold/min": gold_per_min,
    "duration_min": lambda g: g["duration_min"],
}


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2


def rank_biserial(a, b):
    """Effect size from Mann-Whitney U. Positive => group a tends higher."""
    U, _ = mannwhitneyu(a, b, alternative="two-sided")
    return 2 * U / (len(a) * len(b)) - 1


def compare(normals, ranked, label=""):
    print(f"\n{'='*72}\n{label}  (normals n={len(normals)}, ranked n={len(ranked)})\n{'='*72}")
    hdr = f"{'metric':<20}{'normal med':>12}{'ranked med':>12}{'r-biserial':>12}{'p (MWU)':>10}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for name, fn in METRICS.items():
        na = [v for g in normals if (v := fn(g)) is not None]
        ra = [v for g in ranked if (v := fn(g)) is not None]
        try:
            _, p = mannwhitneyu(na, ra, alternative="two-sided")
            rb = rank_biserial(na, ra)
        except ValueError:
            p, rb = float("nan"), float("nan")
        rows.append((name, median(na), median(ra), rb, p))
        star = "*" if p < 0.05 else " "
        print(f"{name:<20}{median(na):>12.3f}{median(ra):>12.3f}{rb:>12.3f}{p:>10.3f}{star}")
    return rows


def diversity(games, label):
    champs = set(g["champion"] for g in games)
    roles = set(g["position"] for g in games)
    n = len(games)
    print(f"{label:<10} games={n:<4} distinct champs={len(champs):<3} "
          f"({len(champs)/n:.2f}/game)  distinct roles={len(roles)} "
          f"({len(roles)/n:.2f}/game)")
    print(f"           champs: {sorted(champs)}")
    print(f"           roles:  {sorted(roles)}")


def main():
    games = load()
    normals = [g for g in games if g["queue_id"] == NORMAL]
    ranked = [g for g in games if g["queue_id"] == RANKED]

    print(f"Normal draft (400): {len(normals)} games, "
          f"{sum(g['win'] for g in normals)} wins "
          f"({sum(g['win'] for g in normals)/len(normals):.1%})")
    print(f"Ranked solo (420):  {len(ranked)} games, "
          f"{sum(g['win'] for g in ranked)} wins "
          f"({sum(g['win'] for g in ranked)/len(ranked):.1%})")

    print("\n--- CHAMPION / ROLE DIVERSITY ---")
    diversity(normals, "normals")
    diversity(ranked, "ranked")

    # Full-sample comparison
    compare(normals, ranked, "ALL GAMES per queue")

    # Same-champion paired view
    print(f"\n{'#'*72}\nSAME-CHAMPION view (champs played in BOTH queues)\n{'#'*72}")
    nchamp = collections.defaultdict(list)
    rchamp = collections.defaultdict(list)
    for g in normals:
        nchamp[g["champion"]].append(g)
    for g in ranked:
        rchamp[g["champion"]].append(g)
    shared = sorted(set(nchamp) & set(rchamp))
    print(f"Shared champions: {shared}")
    for c in shared:
        print(f"  {c}: normals n={len(nchamp[c])}, ranked n={len(rchamp[c])}")

    pooled_n = [g for c in shared for g in nchamp[c]]
    pooled_r = [g for c in shared for g in rchamp[c]]
    wn = sum(g["win"] for g in pooled_n)
    wr = sum(g["win"] for g in pooled_r)
    print(f"\nPooled shared-champ winrate: normals {wn}/{len(pooled_n)} "
          f"({wn/len(pooled_n):.1%})  ranked {wr}/{len(pooled_r)} "
          f"({wr/len(pooled_r):.1%})")
    compare(pooled_n, pooled_r, "SHARED-CHAMPION games only")


if __name__ == "__main__":
    main()
