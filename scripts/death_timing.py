# /// script
# requires-python = ">=3.10"
# dependencies = ["scipy", "numpy", "scikit-learn"]
# ///
"""
Death-timing analysis for djfalcon#NA1 climb study (issue 21).

Question: do death TIMINGS correlate with win/loss (dying in lane vs at 30 min),
holding death QUANTITY fixed? Uses cached Riot match bodies + timelines only.

Design notes:
  - Phase buckets 0-10 / 10-20 / 20-30 / 30+ min with per-phase EXPOSURE
    (a 20-30 death rate only exists for games that reach 20 min; rates are
    deaths / minutes actually played inside the phase).
  - Nexus-fall CENSORING: deaths in the final 90s of a game are a decided-game
    artifact (the last lost teamfight mechanically writes late deaths into every
    loss). Everything is computed censored AND uncensored.
  - Headline: logistic P(win) on the composition of deaths (share occurring at
    20+ min) CONTROLLING for total deaths/min and duration -> does timing carry
    signal beyond quantity? Cluster-bootstrap CIs + GroupKFold AUC, clustered by
    match. Effective n = matches.
  - Solo vs group deaths: a death is 'solo' if no ALLIED death within +/-W sec
    (W in {15,30,45}). Solo deaths in the 14-25 min pre-objective window are the
    hypothesized throw mechanism.
  - All correlational, within-game. Reverse causation is discussed in the report:
    deaths cluster while behind, and even censored, late deaths partly reflect
    already-losing states.

Population for inference = cohort player-games (full 10-player rosters, 5 win /
5 loss per match). dj's own games are held out for his personal profile.
"""
import glob
import gzip
import json
import os
import csv
from collections import defaultdict

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

CACHE = "/home/paul/riot_cache/ranksig"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV = os.path.join(REPO, "data", "death_features.csv")
DJ_PUUID = "i-wo_QJ1JiotACrlb6wlsPLYhfVCXCwDogb03ZcheMSsLMUY9MQknG3RqhXtWIwNoKtPuekiB6Oynw"

CENSOR_SEC = 90.0            # nexus-fall censor window
PHASES = [(0, 10), (10, 20), (20, 30), (30, 999)]
SOLO_WINDOWS = [15, 30, 45]  # seconds
MID_LO, MID_HI = 14.0, 25.0  # pre-objective throw window (minutes)


def load_matches():
    """Yield dicts: {match_id, tier, source, duration_min, parts, kills}."""
    files = []
    for tier in ("BRONZE", "SILVER", "IRON"):
        for f in sorted(glob.glob(f"{CACHE}/cohort/{tier}/*.json.gz")):
            files.append((f, tier, "cohort"))
    for f in sorted(glob.glob(f"{CACHE}/dj/*.tl.json.gz")):
        files.append((f, "DJ", "dj"))

    for f, tier, source in files:
        d = json.load(gzip.open(f))
        info = d["body"]["info"]
        dur_min = info["gameDuration"] / 60.0
        # skip remakes / very short games defensively
        if dur_min < 5:
            continue
        parts = {}
        for p in info["participants"]:
            parts[p["participantId"]] = {
                "win": bool(p["win"]),
                "team": p["teamId"],
                "pos": p.get("teamPosition") or p.get("individualPosition") or "",
                "champ": p["championName"],
                "puuid": p["puuid"],
                "body_deaths": p["deaths"],
            }
        kills = []  # (t_min, victimId, killerId)
        for fr in d["timeline"]["info"]["frames"]:
            for ev in fr.get("events", []):
                if ev.get("type") == "CHAMPION_KILL":
                    vid = ev.get("victimId")
                    if vid is None or vid == 0:
                        continue
                    kills.append((ev["timestamp"] / 60000.0,
                                  vid, ev.get("killerId")))
        # team gold diff at ~15 min for throw-label derivation
        gold15 = team_gold_diff(d["timeline"]["info"]["frames"], parts, 15)
        yield {"match_id": d["match_id"], "tier": tier, "source": source,
               "dur_min": dur_min, "parts": parts, "kills": kills,
               "gold15": gold15}


def team_gold_diff(frames, parts, target_min):
    """Total gold (team100 - team200) at frame nearest target_min. dict team->gold."""
    best = None
    for fr in frames:
        m = fr["timestamp"] / 60000.0
        if best is None or abs(m - target_min) < abs(best[0] - target_min):
            best = (m, fr)
    if best is None:
        return None
    tg = {100: 0, 200: 0}
    for pid_s, pf in best[1]["participantFrames"].items():
        pid = int(pid_s)
        team = parts.get(pid, {}).get("team")
        if team in tg:
            tg[team] += pf.get("totalGold", 0)
    return {100: tg[100], 200: tg[200]}


def phase_exposure(dur_min, a, b):
    """Minutes actually played inside phase [a,b)."""
    return max(0.0, min(dur_min, b) - a)


def build_features():
    rows = []
    for m in load_matches():
        dur = m["dur_min"]
        parts = m["parts"]
        censor_t = dur - CENSOR_SEC / 60.0
        # index deaths per player and per team (for solo detection)
        deaths_by_player = defaultdict(list)   # pid -> [t_min]
        deaths_by_team = defaultdict(list)     # team -> [(t_min, pid)]
        for t, vid, kid in m["kills"]:
            if vid not in parts:
                continue
            deaths_by_player[vid].append(t)
            deaths_by_team[parts[vid]["team"]].append((t, vid))

        for pid, meta in parts.items():
            # dj games: only emit dj's own row for the per-game CSV subject set,
            # but we still needed full roster above for solo/team detection.
            is_dj = (meta["puuid"] == DJ_PUUID)
            if m["source"] == "dj" and not is_dj:
                continue

            team = meta["team"]
            my_deaths = sorted(deaths_by_player.get(pid, []))
            ally_deaths = [(t, v) for (t, v) in deaths_by_team.get(team, [])
                           if v != pid]

            row = {
                "match_id": m["match_id"], "tier": m["tier"],
                "source": m["source"], "participant_id": pid,
                "is_dj": int(is_dj), "win": int(meta["win"]),
                "team_id": team, "position": meta["pos"],
                "champion": meta["champ"], "duration_min": round(dur, 3),
                "reached_20": int(dur >= 20), "reached_30": int(dur >= 30),
            }
            # gold lead (positive = this player's team ahead at 15m)
            g = m["gold15"]
            row["gold_lead_15"] = (g[team] - g[300 - team]) if g else 0

            for censored, pref in ((False, ""), (True, "c_")):
                dd = [t for t in my_deaths
                      if (not censored) or t <= censor_t]
                ad = [(t, v) for (t, v) in ally_deaths
                      if (not censored) or t <= censor_t]
                total = len(dd)
                row[pref + "deaths_total"] = total
                row[pref + "deaths_pm"] = total / dur
                # phase counts + exposure rates
                counts = {}
                for (a, b) in PHASES:
                    c = sum(1 for t in dd if a <= t < b)
                    counts[(a, b)] = c
                    tag = f"{a}_{b if b < 999 else 'p'}"
                    row[pref + f"d_{tag}"] = c
                    expo = phase_exposure(dur, a, b)
                    row[pref + f"rate_{tag}"] = (c / expo) if expo > 0 else np.nan
                d20p = counts[(20, 30)] + counts[(30, 999)]
                d30p = counts[(30, 999)]
                d010 = counts[(0, 10)]
                row[pref + "share_early"] = (d010 / total) if total else np.nan
                row[pref + "share_20p"] = (d20p / total) if total else np.nan
                row[pref + "share_30p"] = (d30p / total) if total else np.nan
                # mid-game 14-25 window
                dmid = [t for t in dd if MID_LO <= t < MID_HI]
                row[pref + "d_mid"] = len(dmid)
                # solo detection at each window
                for W in SOLO_WINDOWS:
                    ww = W / 60.0
                    solo_total = 0
                    solo_mid = 0
                    for t in dd:
                        near = any(abs(t - at) <= ww for (at, v) in ad)
                        if not near:
                            solo_total += 1
                            if MID_LO <= t < MID_HI:
                                solo_mid += 1
                    row[pref + f"solo_w{W}"] = solo_total
                    row[pref + f"solo_mid_w{W}"] = solo_mid
                    row[pref + f"solo_share_w{W}"] = (
                        solo_total / total) if total else np.nan
                    row[pref + f"solo_pm_w{W}"] = solo_total / dur
                    # solo rate within mid window per exposed mid-minute
                    mid_expo = phase_exposure(dur, MID_LO, MID_HI)
                    row[pref + f"solo_mid_pm_w{W}"] = (
                        solo_mid / mid_expo) if mid_expo > 0 else np.nan
            rows.append(row)
    return rows


# ----------------------------- inference helpers -----------------------------

def cluster_bootstrap_logit(X, y, groups, n_boot=2000, seed=0):
    """Resample whole matches with replacement; refit logit; return coef CIs.
    Returns (point_coef, lo, hi) arrays over columns of X (already scaled)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    base = LogisticRegression(max_iter=2000, C=1e6)
    base.fit(X, y)
    point = base.coef_[0]
    boots = []
    gidx = {g: np.where(groups == g)[0] for g in uniq}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([gidx[g] for g in pick])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        try:
            m = LogisticRegression(max_iter=2000, C=1e6).fit(X[idx], yb)
            boots.append(m.coef_[0])
        except Exception:
            continue
    boots = np.array(boots)
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    return point, lo, hi, boots


def groupkfold_auc(X, y, groups, seed=0, n_splits=5):
    clf = LogisticRegression(max_iter=2000, C=1.0)
    gkf = GroupKFold(n_splits=n_splits)
    pred = cross_val_predict(clf, X, y, cv=gkf, groups=groups,
                             method="predict_proba")[:, 1]
    return roc_auc_score(y, pred), pred


def run_composition_test(rows, pref, feats_extra=("deaths_pm", "duration_min"),
                         label=""):
    """Logistic P(win) ~ share_20p + deaths_pm + duration, cohort only,
    clustered by match. Drops zero-death games (share undefined)."""
    coh = [r for r in rows if r["source"] == "cohort"]
    key = pref + "share_20p"
    data = [r for r in coh if not (isinstance(r[key], float) and np.isnan(r[key]))]
    n_drop = len(coh) - len(data)
    y = np.array([r["win"] for r in data])
    groups = np.array([r["match_id"] for r in data])
    cols = [pref + "share_20p", pref + "deaths_pm", "duration_min"]
    Xraw = np.array([[float(r[c]) for c in cols] for r in data])
    sc = StandardScaler().fit(Xraw)
    X = sc.transform(Xraw)
    point, lo, hi, boots = cluster_bootstrap_logit(X, y, groups)
    auc_full, _ = groupkfold_auc(X, y, groups)
    # nested: quantity-only vs quantity+timing
    Xq = X[:, 1:]      # deaths_pm + duration
    auc_q, _ = groupkfold_auc(Xq, y, groups)
    # two-sided p for share_20p coef from bootstrap
    b0 = boots[:, 0]
    p_share = 2 * min((b0 <= 0).mean(), (b0 >= 0).mean())
    return {
        "label": label, "pref": pref, "cols": cols, "n": len(data),
        "n_matches": len(set(groups)), "n_drop_zero": n_drop,
        "coef": point.tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
        "p_share20p": p_share, "auc_full": auc_full, "auc_quant_only": auc_q,
        "auc_gain": auc_full - auc_q,
    }


def logit_uni(rows, pref, col, control_cols, source="cohort", label=""):
    """P(win) ~ col + controls, clustered bootstrap. Drops NaN in col."""
    dat = [r for r in rows if r["source"] == source]
    dat = [r for r in dat if not any(
        isinstance(r[c], float) and np.isnan(r[c]) for c in [col] + control_cols)]
    y = np.array([r["win"] for r in dat])
    groups = np.array([r["match_id"] for r in dat])
    cols = [col] + control_cols
    Xraw = np.array([[float(r[c]) for c in cols] for r in dat])
    X = StandardScaler().fit_transform(Xraw)
    point, lo, hi, boots = cluster_bootstrap_logit(X, y, groups)
    b0 = boots[:, 0]
    p = 2 * min((b0 <= 0).mean(), (b0 >= 0).mean())
    return {"label": label, "col": col, "controls": control_cols,
            "n": len(dat), "coef": point[0], "lo": lo[0], "hi": hi[0], "p": p}


def phase_winrate_table(rows, pref):
    """Winrate & per-exposed-minute death rate by phase for winners vs losers,
    cohort. Rates conditioned on games that reached the phase."""
    coh = [r for r in rows if r["source"] == "cohort"]
    out = {}
    for (a, b) in PHASES:
        tag = f"{a}_{b if b < 999 else 'p'}"
        rk = pref + f"rate_{tag}"
        w = [r[rk] for r in coh if r["win"] == 1 and not np.isnan(r[rk])]
        l = [r[rk] for r in coh if r["win"] == 0 and not np.isnan(r[rk])]
        out[tag] = {
            "win_rate_pm": float(np.mean(w)), "loss_rate_pm": float(np.mean(l)),
            "n_win": len(w), "n_loss": len(l),
            "mwu_p": float(stats.mannwhitneyu(w, l, alternative="two-sided")[1]),
        }
    return out


def dj_profile(rows, pref):
    coh = [r for r in rows if r["source"] == "cohort"]
    dj = [r for r in rows if r["is_dj"] == 1]
    cw = [r for r in coh if r["win"] == 1]
    cl = [r for r in coh if r["win"] == 0]
    djw = [r for r in dj if r["win"] == 1]
    djl = [r for r in dj if r["win"] == 0]

    def mean(rs, k):
        v = [r[k] for r in rs if not (isinstance(r[k], float) and np.isnan(r[k]))]
        return float(np.mean(v)) if v else float("nan")

    metrics = [pref + "deaths_pm", pref + "share_20p", pref + "solo_share_w30",
               pref + "solo_pm_w30", pref + "solo_mid_pm_w30", pref + "d_mid",
               pref + "solo_mid_w30"]
    prof = {}
    for k in metrics:
        prof[k] = {
            "dj_all": mean(dj, k), "dj_win": mean(djw, k), "dj_loss": mean(djl, k),
            "coh_win": mean(cw, k), "coh_loss": mean(cl, k),
        }
    return prof, {"dj_n": len(dj), "dj_w": len(djw), "dj_l": len(djl),
                  "coh_w": len(cw), "coh_l": len(cl)}


def run_composition_test_col(rows, pref, share_col, label=""):
    """Generic composition test on any share column, controlling deaths/min+dur."""
    coh = [r for r in rows if r["source"] == "cohort"]
    key = pref + share_col
    data = [r for r in coh if not (isinstance(r[key], float) and np.isnan(r[key]))]
    y = np.array([r["win"] for r in data])
    groups = np.array([r["match_id"] for r in data])
    cols = [pref + share_col, pref + "deaths_pm", "duration_min"]
    Xraw = np.array([[float(r[c]) for c in cols] for r in data])
    X = StandardScaler().fit_transform(Xraw)
    point, lo, hi, boots = cluster_bootstrap_logit(X, y, groups)
    auc_full, _ = groupkfold_auc(X, y, groups)
    auc_q, _ = groupkfold_auc(X[:, 1:], y, groups)
    b0 = boots[:, 0]
    p = 2 * min((b0 <= 0).mean(), (b0 >= 0).mean())
    return {"label": label, "col": share_col, "n": len(data),
            "coef_share": point[0], "lo": lo[0], "hi": hi[0], "p": p,
            "auc_full": auc_full, "auc_quant_only": auc_q,
            "auc_gain": auc_full - auc_q}


def dj_throw_sensitivity(rows, pref):
    """Count dj thrown-lead losses at several gold-lead@15 thresholds."""
    dj = [r for r in rows if r["is_dj"] == 1]
    losses = [r for r in dj if r["win"] == 0]
    out = {}
    for thr in (500, 1000, 1500, 2000, 2500):
        n = sum(1 for r in losses if r["gold_lead_15"] > thr)
        out[thr] = {"n_thrown": n, "n_loss": len(losses),
                    "pct": round(100 * n / len(losses), 1)}
    return out


def dj_throw_analysis(rows, pref, lead_thresh=1500):
    """Re-derive mid-throw labels from Riot gold timelines: dj's team ahead by
    >lead_thresh gold at 15 min but LOST = thrown mid-game lead. Compare
    solo-death shape of thrown vs non-thrown losses."""
    dj = [r for r in rows if r["is_dj"] == 1]
    losses = [r for r in dj if r["win"] == 0]
    thrown = [r for r in losses if r["gold_lead_15"] > lead_thresh]
    other_l = [r for r in losses if r["gold_lead_15"] <= lead_thresh]
    wins = [r for r in dj if r["win"] == 1]

    def mean(rs, k):
        v = [r[k] for r in rs if not (isinstance(r[k], float) and np.isnan(r[k]))]
        return float(np.mean(v)) if v else float("nan")

    ks = [pref + "solo_mid_w30", pref + "solo_mid_pm_w30", pref + "d_mid",
          pref + "solo_pm_w30", pref + "solo_share_w30"]
    res = {"n_loss": len(losses), "n_thrown": len(thrown),
           "n_other_loss": len(other_l), "n_win": len(wins),
           "lead_thresh": lead_thresh}
    for k in ks:
        res[k] = {"thrown": mean(thrown, k), "other_loss": mean(other_l, k),
                  "wins": mean(wins, k)}
    # test: solo_mid_pm in thrown losses vs wins
    a = [r[pref + "solo_mid_pm_w30"] for r in thrown
         if not np.isnan(r[pref + "solo_mid_pm_w30"])]
    b = [r[pref + "solo_mid_pm_w30"] for r in wins
         if not np.isnan(r[pref + "solo_mid_pm_w30"])]
    if a and b:
        res["mwu_thrown_vs_win_p"] = float(
            stats.mannwhitneyu(a, b, alternative="greater")[1])
    return res


def main():
    rows = build_features()
    # write CSV
    keys = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {OUT_CSV}: {len(rows)} player-games, {len(keys)} cols")

    report = {}
    for pref, name in (("", "uncensored"), ("c_", "censored")):
        print("\n" + "=" * 70)
        print(f"### {name.upper()} (pref='{pref}')")
        comp = run_composition_test(rows, pref, label=name)
        report[f"composition_{name}"] = comp
        print(f"COMPOSITION test (P(win) ~ share_20p + deaths_pm + duration)")
        print(f"  n={comp['n']} player-games, matches={comp['n_matches']}, "
              f"dropped zero-death={comp['n_drop_zero']}")
        for c, co, lo, hi in zip(comp["cols"], comp["coef"], comp["lo"], comp["hi"]):
            print(f"    {c:20s} coef={co:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]")
        print(f"  share_20p bootstrap p={comp['p_share20p']:.3f}")
        print(f"  AUC full={comp['auc_full']:.3f}  quant-only={comp['auc_quant_only']:.3f}"
              f"  gain={comp['auc_gain']:+.4f}")

        # early-death composition (complement of share_20p)
        comp_e = run_composition_test_col(rows, pref, "share_early", label=name)
        report[f"composition_early_{name}"] = comp_e
        print(f"  COMPOSITION share_early: coef={comp_e['coef_share']:+.3f} "
              f"CI[{comp_e['lo']:+.3f},{comp_e['hi']:+.3f}] p={comp_e['p']:.3f} "
              f"AUC gain={comp_e['auc_gain']:+.4f}")

        # phase winrate table
        pt = phase_winrate_table(rows, pref)
        report[f"phase_table_{name}"] = pt
        print("  phase death rate /exposed-min (winners vs losers):")
        for tag, v in pt.items():
            print(f"    {tag:8s} win={v['win_rate_pm']:.4f} loss={v['loss_rate_pm']:.4f}"
                  f"  MWU p={v['mwu_p']:.2e}")

        # solo-death tests (control total deaths/min)
        solo_tests = {}
        for W in SOLO_WINDOWS:
            r_all = logit_uni(rows, pref, pref + f"solo_pm_w{W}",
                              [pref + "deaths_pm", "duration_min"],
                              label=f"solo_pm_w{W}")
            r_mid = logit_uni(rows, pref, pref + f"solo_mid_pm_w{W}",
                              [pref + "deaths_pm", "duration_min"],
                              label=f"solo_mid_pm_w{W}")
            solo_tests[f"w{W}_all"] = r_all
            solo_tests[f"w{W}_mid"] = r_mid
            print(f"  SOLO w={W}s  all-game: coef={r_all['coef']:+.3f} "
                  f"CI[{r_all['lo']:+.3f},{r_all['hi']:+.3f}] p={r_all['p']:.3f}"
                  f"   mid14-25: coef={r_mid['coef']:+.3f} "
                  f"CI[{r_mid['lo']:+.3f},{r_mid['hi']:+.3f}] p={r_mid['p']:.3f}")
        report[f"solo_tests_{name}"] = solo_tests

        prof, ns = dj_profile(rows, pref)
        report[f"dj_profile_{name}"] = {"prof": prof, "ns": ns}
        print(f"  DJ profile (n_dj={ns['dj_n']} w={ns['dj_w']} l={ns['dj_l']}):")
        for k, v in prof.items():
            print(f"    {k:20s} dj={v['dj_all']:.4f} (W {v['dj_win']:.4f}/L {v['dj_loss']:.4f})"
                  f"  cohW={v['coh_win']:.4f} cohL={v['coh_loss']:.4f}")

        thr_sens = dj_throw_sensitivity(rows, pref)
        report[f"dj_throw_sensitivity_{name}"] = thr_sens
        print(f"  DJ throw-threshold sensitivity (losses with gold lead@15 > X):")
        for thr, v in thr_sens.items():
            print(f"    >{thr}g: {v['n_thrown']}/{v['n_loss']} = {v['pct']}%")

        throw = dj_throw_analysis(rows, pref)
        report[f"dj_throw_{name}"] = throw
        print(f"  DJ throw (gold>1500@15m & lost): n_thrown={throw['n_thrown']}"
              f"/{throw['n_loss']} losses, n_win={throw['n_win']}")
        for k in [pref + "solo_mid_pm_w30", pref + "solo_mid_w30", pref + "d_mid"]:
            v = throw[k]
            print(f"    {k:20s} thrown={v['thrown']:.4f} otherL={v['other_loss']:.4f}"
                  f" wins={v['wins']:.4f}")
        if "mwu_thrown_vs_win_p" in throw:
            print(f"    MWU solo_mid_pm thrown>win p={throw['mwu_thrown_vs_win_p']:.3f}")

    with open(os.path.join(REPO, "data", "death_timing_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)
    print("\nwrote data/death_timing_report.json")


if __name__ == "__main__":
    main()
