# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy"]
# ///
"""Schedule & streak analysis for djfalcon#NA1 ranked solo (queue 420).

Does WHEN he plays (hour-of-day / day-of-week) or IN WHAT PATTERN (streaks,
autocorrelation) predict wins, or is it all i.i.d. variance?

Assumptions & method:
  - Ranked solo only (queue_id == 420), remakes excluded.
  - Timestamps in the data are UTC (`game_creation_utc`, ...Z). For human
    "hour of day" and "day of week" we convert to US/Pacific
    (America/Los_Angeles, DST-aware). This ASSUMES djfalcon is on Pacific time;
    if he plays from another zone the hour buckets shift accordingly.
  - Wilson 95% score intervals on every win-rate; buckets with n<10 flagged
    UNRELIABLE.
  - Streaks vs i.i.d. Bernoulli(p=overall WR): 10k fixed-seed shuffles.
  - Wald-Wolfowitz runs test (normal approx) for non-randomness of the
    W/L sequence; lag-1 autocorrelation with a permutation p-value.
  - Games-per-day vs that day's win rate: Spearman rho + p.

MULTIPLE-COMPARISONS CAVEAT: we test 4 hour buckets + 7 weekdays + streaks +
runs + autocorr + volume = ~15 hypotheses. At alpha=0.05 we EXPECT ~0.75
false positives by chance. Treat any single "significant" bucket skeptically;
the honest question is whether the WHOLE picture departs from randomness.

Run: uv run scripts/schedule_analysis.py
"""
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import norm, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACIFIC = ZoneInfo("America/Los_Angeles")
SEED = 20260707
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def hour_bucket(h):
    # Pacific local hour -> coarse daypart
    if 5 <= h < 12:
        return "morning (05-12)"
    if 12 <= h < 17:
        return "afternoon (12-17)"
    if 17 <= h < 22:
        return "evening (17-22)"
    return "night (22-05)"


def longest_run(seq, value):
    best = cur = 0
    for x in seq:
        if x == value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def runs_test(seq):
    """Wald-Wolfowitz runs test on a binary sequence. Returns (runs, z, p)."""
    n = len(seq)
    n1 = sum(seq)          # wins
    n0 = n - n1            # losses
    if n1 == 0 or n0 == 0:
        return (1, float("nan"), float("nan"))
    runs = 1 + sum(1 for i in range(1, n) if seq[i] != seq[i - 1])
    mu = 1 + (2 * n1 * n0) / n
    var = (2 * n1 * n0 * (2 * n1 * n0 - n)) / (n * n * (n - 1))
    if var <= 0:
        return (runs, float("nan"), float("nan"))
    z = (runs - mu) / (var ** 0.5)
    p = 2 * (1 - norm.cdf(abs(z)))
    return (runs, z, p)


def lag1_autocorr(seq):
    a = np.asarray(seq, dtype=float)
    a = a - a.mean()
    denom = np.dot(a, a)
    if denom == 0:
        return float("nan")
    return float(np.dot(a[:-1], a[1:]) / denom)


def main():
    rows = json.load(open(f"{ROOT}/data/riot_games.json"))
    for r in rows:
        r["dt_utc"] = datetime.strptime(
            r["game_creation_utc"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc)
        r["dt_pac"] = r["dt_utc"].astimezone(PACIFIC)
    solo = [r for r in rows if r["queue_id"] == 420 and not r["remake"]]
    solo.sort(key=lambda r: r["dt_utc"])

    n = len(solo)
    wins = sum(1 for r in solo if r["win"])
    p_overall, lo, hi = wilson(wins, n)
    print("=" * 70)
    print(f"djfalcon#NA1  ranked solo (queue 420, remakes excluded)")
    print(f"N = {n} games | wins = {wins} | overall WR = {p_overall:.3f} "
          f"[Wilson95 {lo:.3f}, {hi:.3f}]")
    print(f"date range (UTC): {solo[0]['dt_utc'].date()} .. {solo[-1]['dt_utc'].date()}")
    print("hour/day buckets use US/Pacific local time (assumption; see header)")
    print("=" * 70)

    # ---------- (1a) win rate by hour-of-day bucket ----------
    print("\n(1a) WIN RATE BY DAYPART (Pacific local hour)")
    print(f"{'bucket':<20}{'n':>4}{'wins':>6}{'WR':>8}{'   Wilson95':>18}   flag")
    order = ["morning (05-12)", "afternoon (12-17)", "evening (17-22)", "night (22-05)"]
    bkt = defaultdict(lambda: [0, 0])
    for r in solo:
        b = hour_bucket(r["dt_pac"].hour)
        bkt[b][0] += 1
        bkt[b][1] += int(r["win"])
    for b in order:
        cnt, w = bkt[b]
        pp, l, h = wilson(w, cnt)
        flag = "UNRELIABLE n<10" if cnt < 10 else ""
        print(f"{b:<20}{cnt:>4}{w:>6}{pp:>8.3f}   [{l:.3f}, {h:.3f}]   {flag}")

    # ---------- (1b) win rate by day-of-week ----------
    print("\n(1b) WIN RATE BY DAY-OF-WEEK (Pacific)")
    print(f"{'day':<6}{'n':>4}{'wins':>6}{'WR':>8}{'   Wilson95':>18}   flag")
    dow = defaultdict(lambda: [0, 0])
    for r in solo:
        d = r["dt_pac"].weekday()
        dow[d][0] += 1
        dow[d][1] += int(r["win"])
    for d in range(7):
        cnt, w = dow[d]
        pp, l, h = wilson(w, cnt)
        flag = "UNRELIABLE n<10" if cnt < 10 else ""
        print(f"{WEEKDAYS[d]:<6}{cnt:>4}{w:>6}{pp:>8.3f}   [{l:.3f}, {h:.3f}]   {flag}")

    # ---------- (2) streaks vs i.i.d. Bernoulli ----------
    seq = [int(r["win"]) for r in solo]
    obs_w = longest_run(seq, 1)
    obs_l = longest_run(seq, 0)
    rng = np.random.default_rng(SEED)
    NSIM = 10000
    base = np.array(seq)
    sim_w = np.empty(NSIM, dtype=int)
    sim_l = np.empty(NSIM, dtype=int)
    for i in range(NSIM):
        s = base.copy()
        rng.shuffle(s)
        sl = list(s)
        sim_w[i] = longest_run(sl, 1)
        sim_l[i] = longest_run(sl, 0)
    # p = P(sim >= observed) under shuffled (fixed-composition) null
    pw = (np.sum(sim_w >= obs_w) + 1) / (NSIM + 1)
    pl = (np.sum(sim_l >= obs_l) + 1) / (NSIM + 1)
    print("\n(2) STREAKS vs i.i.d. null (10k shuffles of the actual W/L multiset)")
    print(f"seed = {SEED}")
    print(f"longest WIN streak : observed {obs_w}  | "
          f"sim mean {sim_w.mean():.2f}, 95th pct {np.percentile(sim_w,95):.0f}, "
          f"max {sim_w.max()}  | P(sim>=obs) = {pw:.3f}")
    print(f"longest LOSS streak: observed {obs_l}  | "
          f"sim mean {sim_l.mean():.2f}, 95th pct {np.percentile(sim_l,95):.0f}, "
          f"max {sim_l.max()}  | P(sim>=obs) = {pl:.3f}")

    runs, z, p_runs = runs_test(seq)
    exp_runs = 1 + (2 * wins * (n - wins)) / n
    print("\n    Wald-Wolfowitz runs test (streakiness / non-randomness):")
    print(f"    runs observed = {runs} | expected under i.i.d. = {exp_runs:.1f} "
          f"| z = {z:.3f} | p = {p_runs:.3f}")
    print("    (fewer runs than expected -> clustering/streaky; more -> alternating)")

    ac1 = lag1_autocorr(seq)
    NPERM = 10000
    rng2 = np.random.default_rng(SEED + 1)
    perm = np.empty(NPERM)
    for i in range(NPERM):
        s = base.copy()
        rng2.shuffle(s)
        perm[i] = lag1_autocorr(list(s))
    p_ac = (np.sum(np.abs(perm) >= abs(ac1)) + 1) / (NPERM + 1)
    print("\n    Lag-1 autocorrelation of the W/L sequence:")
    print(f"    r1 = {ac1:+.3f} | two-sided permutation p = {p_ac:.3f} "
          f"(does a game's result predict the next?)")

    # ---------- (3) games-per-day vs that day's win rate ----------
    byday = defaultdict(lambda: [0, 0])  # pacific calendar date
    for r in solo:
        key = r["dt_pac"].date()
        byday[key][0] += 1
        byday[key][1] += int(r["win"])
    days = sorted(byday)
    vol = np.array([byday[d][0] for d in days])
    wr = np.array([byday[d][1] / byday[d][0] for d in days])
    rho, p_sp = spearmanr(vol, wr)
    print("\n(3) GAMES-PER-DAY vs THAT DAY'S WIN RATE")
    print(f"    active days = {len(days)} | games/day: min {vol.min()}, "
          f"median {np.median(vol):.0f}, max {vol.max()}")
    print(f"    Spearman rho = {rho:+.3f} | p = {p_sp:.3f}")
    print("    (positive -> more games that day tracks higher WR; "
          "negative -> tilt/fatigue pattern)")

    # ---------- verdict ----------
    print("\n" + "=" * 70)
    print("VERDICT")
    sig = []
    if pw < 0.05:
        sig.append("win-streak")
    if pl < 0.05:
        sig.append("loss-streak")
    if not (np.isnan(p_runs)) and p_runs < 0.05:
        sig.append("runs")
    if p_ac < 0.05:
        sig.append("autocorr")
    if p_sp < 0.05:
        sig.append("volume~WR")
    if sig:
        print(f"  Nominally significant (pre-correction): {', '.join(sig)}")
    else:
        print("  Nothing clears p<0.05 even before multiple-comparisons correction.")
    print("  With ~15 tests, expect ~0.75 false positives at alpha=0.05.")
    print("  Small-n dayparts/weekdays (many n<10) are noise, not signal.")
    print("=" * 70)


if __name__ == "__main__":
    main()
