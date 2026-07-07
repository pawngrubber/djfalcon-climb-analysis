# Subagent: loss-anatomy analyst
**Model:** Opus-class subagent · **Spawned by:** the main analysis session

## Instructions given
op.gg's per-game "op score" timelines (chartData in the captured server responses) show a win-probability-like score over game time for each of his recent ranked games. Parse them and classify each loss: early collapse (score below baseline by 10 min and never recovers), mid-game throw (ahead at 15-20 min then falls), or late fade. Also measure: in games where he was ahead at 15 min, how often does he win? Post a short signed comment on the issue with the loss taxonomy counts; full numbers back to the coordinator. Data caveat to state: op score is op.gg's proprietary composite, a proxy for game state, not actual win probability.
