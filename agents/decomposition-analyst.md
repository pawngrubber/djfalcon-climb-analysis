# Subagent: decomposition analyst (marker vs driver)
**Model:** Opus-class subagent · **Spawned by:** the main analysis session

## Instructions given
Methodological critique from the thread owner: the triplet compares rank populations, but between-rank differences are not the same as win drivers — e.g., cohort Nasus winners do not out-farm Nasus losers even though CS/min rises with rank. For every core metric (cs/min lanes, vision/min, control wards, deaths/min, kill participation), compute three effects side by side with CIs:
1. **Between-rank** — Iron vs Silver cohort separation (rank-biserial, MWU p);
2. **Within-rank winner-vs-loser** — inside each tier cohort, winners vs losers on that metric (per role where samples allow);
3. **Within-djfalcon** — across his own 97 ranked games, his wins vs his losses on that metric.
Classify each metric: *driver* (all three axes agree), *marker* (separates ranks but not winners from losers), *mixed*, or *insufficient data*. State plainly that all of it remains correlational — this decomposition separates comparison types, it does not establish causation. Post findings as a short signed comment on the subtask issue; full numbers back to the coordinator.
