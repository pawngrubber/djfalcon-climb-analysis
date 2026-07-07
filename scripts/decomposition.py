# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy"]
# ///
"""
Decomposition analyst: separate BETWEEN-RANK, WITHIN-RANK W/L, and WITHIN-DJFALCON
effects for each core metric, then classify each as driver / marker / mixed / insufficient.

All effects are correlational. This decomposition separates *comparison types*
(does a metric track rank? does it track winning? does it track HIS winning?),
it does not establish causation.
"""
import json, os
from scipy.stats import mannwhitneyu

D = os.path.join(os.path.dirname(__file__), "..", "data")
def load(n): return json.load(open(os.path.join(D, n)))

iron   = load("cohort_iron.json")
bronze = load("cohort_bronze.json")
silver = load("cohort_silver.json")
riot   = load("riot_games.json")

LANES = {"TOP", "MIDDLE", "BOTTOM"}

def rb_mwu(a, b):
    """rank-biserial + MWU p for group a vs group b (positive rb => a stochastically larger)."""
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if len(a) < 3 or len(b) < 3:
        return None, None, len(a), len(b)
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    rb = 2.0 * U / (len(a) * len(b)) - 1.0
    return rb, p, len(a), len(b)

# ---- djfalcon ranked games (queue 420, non-remake) ----
his = []
for g in riot:
    if g["queue_id"] != 420 or g.get("remake"):
        continue
    dur = g["duration_min"]
    his.append({
        "position": g["position"],
        "win": g["win"],
        "cs_per_min": g["cs_per_min"],
        "vision_per_min": g["vision_score"] / dur if dur else None,
        "control_wards": g["control_wards"],
        "deaths_per_min": g["death"] / dur if dur else None,
        "kp": g["kill_participation"],
    })

METRICS = ["cs_per_min", "vision_per_min", "control_wards", "deaths_per_min", "kp"]

# reverse-causation risk (winning team mechanically inflates the per-game stat)
REVCAUSE = {
    "cs_per_min":     "moderate (winning lane snowballs farm, but also drives wins)",
    "vision_per_min": "moderate (leads/pushes create map access -> more vision)",
    "control_wards":  "low-ish (a purchase decision, less auto-inflated by score)",
    "deaths_per_min": "high (losing team dies more; strongly team-state driven)",
    "kp":             "high (winning teams get more kills to participate in)",
}

def vals(cohort, metric, roles=None, win=None):
    out = []
    for r in cohort:
        if roles is not None and r["position"] not in roles:
            continue
        if win is not None and r["win"] != win:
            continue
        out.append(r.get(metric))
    return out

pool = iron + bronze + silver

print("=" * 78)
print("AXIS 1  BETWEEN-RANK: Iron cohort vs Silver cohort (rb positive => Silver higher)")
print("=" * 78)
axis1 = {}
for m in METRICS:
    roles = LANES if m == "cs_per_min" else None
    rb, p, na, nb = rb_mwu(vals(silver, m, roles), vals(iron, m, roles))
    axis1[m] = (rb, p)
    print(f"  {m:15s} rb={rb:+.3f} p={p:.2e}  nSilver={na} nIron={nb}"
          + ("  [lanes only]" if m == "cs_per_min" else ""))

print()
print("=" * 78)
print("AXIS 2  WITHIN-RANK W/L: winners vs losers, POOLED across Iron+Bronze+Silver")
print("        (rb positive => winners higher)")
print("=" * 78)
axis2 = {}
for m in METRICS:
    roles = LANES if m == "cs_per_min" else None
    rb, p, na, nb = rb_mwu(vals(pool, m, roles, win=True), vals(pool, m, roles, win=False))
    axis2[m] = (rb, p)
    print(f"  {m:15s} rb={rb:+.3f} p={p:.2e}  nW={na} nL={nb}"
          + ("  [lanes only]" if m == "cs_per_min" else "")
          + f"   revcause={REVCAUSE[m]}")

print()
print("  -- robustness: per-tier W/L (rb, p) --")
for m in METRICS:
    roles = LANES if m == "cs_per_min" else None
    line = f"  {m:15s}"
    for name, coh in [("Iron", iron), ("Bronze", bronze), ("Silver", silver)]:
        rb, p, na, nb = rb_mwu(vals(coh, m, roles, win=True), vals(coh, m, roles, win=False))
        line += f"  {name}: rb={rb:+.2f} p={p:.2f}"
    print(line)

print()
print("  -- vision_per_min W/L per role (pooled) --")
for role in ["UTILITY", "TOP", "MIDDLE", "BOTTOM", "JUNGLE"]:
    rb, p, na, nb = rb_mwu(vals(pool, "vision_per_min", {role}, win=True),
                           vals(pool, "vision_per_min", {role}, win=False))
    if rb is not None:
        print(f"    {role:9s} rb={rb:+.3f} p={p:.2e}  nW={na} nL={nb}")

print()
print("=" * 78)
print("AXIS 3  WITHIN-DJFALCON: his ranked wins vs losses (rb positive => his wins higher)")
print("=" * 78)
axis3 = {}
hisW = [h for h in his if h["win"]]
hisL = [h for h in his if not h["win"]]
print(f"  his ranked games: {len(his)}  (W={len(hisW)} L={len(hisL)})")
for m in METRICS:
    if m == "cs_per_min":
        w = [h[m] for h in hisW if h["position"] in LANES]
        l = [h[m] for h in hisL if h["position"] in LANES]
        rb, p, na, nb = rb_mwu(w, l)
        tag = "  [lanes only]"
    else:
        rb, p, na, nb = rb_mwu([h[m] for h in hisW], [h[m] for h in hisL])
        tag = ""
    axis3[m] = (rb, p)
    print(f"  {m:15s} rb={rb:+.3f} p={p:.2e}  nW={na} nL={nb}{tag}")

# ---------------- classification rule ----------------
print()
print("=" * 78)
print("CLASSIFICATION RULE")
print("=" * 78)
RULE = """
  For each metric, orient sign so 'good' = the direction that helps (for deaths_per_min
  FEWER deaths is good, so we read its signs inverted when judging 'helps winning').
  DRIVER   : Axis-2 (within-rank W/L) AND Axis-3 (within-him W/L) point the SAME helpful
             direction, with p<0.05 on at least one of them and the other not contradicting
             (|rb|>=0.05 same sign OR ~0). Between-rank may or may not separate.
  MARKER   : Axis-1 (between-rank) is significant (p<0.05, |rb|>=0.10) BUT Axis-2 is
             effectively null (|rb|<0.07 or p>=0.05) -> tracks rank, not winning.
  NULL     : no axis separates (a1 not sig, a2 null, a3 null) -> not useful as a lever.
  MIXED    : win-axes disagree in sign / one says driver and the other contradicts.
  INSUFFICIENT: fewer than 3 usable obs on a relevant axis / degenerate (e.g. all-zero).
"""
print(RULE)

def helpful_sign(m, rb):
    # deaths: negative rb (winners die less) is helpful -> flip so + = helpful
    return -rb if m == "deaths_per_min" else rb

def classify(m):
    a1rb, a1p = axis1[m]
    a2rb, a2p = axis2[m]
    a3rb, a3p = axis3[m]
    h2 = helpful_sign(m, a2rb)
    h3 = helpful_sign(m, a3rb)
    # insufficient
    if a2rb is None or a3rb is None:
        return "insufficient"
    # driver: both win-axes helpful same direction, at least one sig, no contradiction
    both_helpful = h2 > 0 and h3 > 0
    at_least_one_sig = (a2p < 0.05) or (a3p < 0.05)
    contradiction = (h2 > 0.05 and h3 < -0.05) or (h2 < -0.05 and h3 > 0.05)
    if both_helpful and at_least_one_sig and not contradiction:
        return "driver"
    # marker: between-rank sig but within-rank W/L null
    a2_null = abs(a2rb) < 0.07 or a2p >= 0.05
    a1_sig = a1p < 0.05 and abs(a1rb) >= 0.10
    if a1_sig and a2_null:
        return "marker"
    if contradiction:
        return "mixed"
    # null: nothing separates on any axis
    a1_null = a1p >= 0.05 or abs(a1rb) < 0.10
    a3_null = (a3p is None) or (a3p != a3p) or a3p >= 0.05 or abs(a3rb) < 0.07
    if a1_null and a2_null and a3_null:
        return "null"
    return "mixed"

print("=" * 78)
print("VERDICTS")
print("=" * 78)
verdicts = {}
for m in METRICS:
    v = classify(m)
    verdicts[m] = v
    print(f"  {m:15s} between={axis1[m][0]:+.3f}(p{axis1[m][1]:.1e})  "
          f"WL={axis2[m][0]:+.3f}(p{axis2[m][1]:.1e})  "
          f"him={axis3[m][0]:+.3f}(p{axis3[m][1]:.1e})  => {v.upper()}")

# ---------------- special cases ----------------
print()
print("=" * 78)
print("SPECIAL CASES (illustrative)")
print("=" * 78)
def champ_wl(cohorts, champ, metric, roles=None):
    rows = [r for c in cohorts for r in c
            if r["champion"] == champ and (roles is None or r["position"] in roles)]
    w = [r[metric] for r in rows if r["win"]]
    l = [r[metric] for r in rows if not r["win"]]
    return rb_mwu(w, l), len(w), len(l)

# Nasus CS/min winners vs losers (pooled cohorts, lane restriction not needed, Nasus is TOP)
(rb, p, na, nb), nw, nl = champ_wl([iron, bronze, silver], "Nasus", "cs_per_min")
print(f"  Nasus cs_per_min W/L (pooled): rb={rb if rb is None else round(rb,3)} "
      f"p={p if p is None else round(p,3)}  nW={nw} nL={nl}")
# Nasus between-rank cs
nas_i = [r["cs_per_min"] for r in iron if r["champion"] == "Nasus"]
nas_s = [r["cs_per_min"] for r in silver if r["champion"] == "Nasus"]
rb, p, na, nb = rb_mwu(nas_s, nas_i)
print(f"  Nasus cs_per_min Iron vs Silver: rb={rb if rb is None else round(rb,3)} "
      f"p={p if p is None else round(p,3)}  nSilver={na} nIron={nb}")

# Sona / support vision
(rb, p, na, nb), nw, nl = champ_wl([iron, bronze, silver], "Sona", "vision_per_min")
print(f"  Sona vision_per_min W/L (pooled): rb={rb if rb is None else round(rb,3)} "
      f"p={p if p is None else round(p,3)}  nW={nw} nL={nl}")
rb, p, na, nb = rb_mwu(vals(pool, "vision_per_min", {"UTILITY"}, win=True),
                       vals(pool, "vision_per_min", {"UTILITY"}, win=False))
print(f"  Support(UTILITY) vision_per_min W/L (pooled): rb={rb:+.3f} p={p:.2e} nW={na} nL={nb}")

print()
print("REVERSE-CAUSATION CAVEATS")
for m in METRICS:
    print(f"  {m:15s} {REVCAUSE[m]}")
