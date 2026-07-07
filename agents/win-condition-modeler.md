# Subagent: win-condition modeler
**Model:** Opus-class subagent · **Spawned by:** the main analysis session

## Instructions given
Using his 97 ranked player-games (data/riot_games.json), fit a simple, honest model of P(win) from his in-game stats (deaths/min, cs/min, kp, vision/min, gold/min, dmg share, turret kills, team dragons, first blood, duration). L2 logistic regression with standardized features + leave-one-out AUC; report coefficients with bootstrap CIs, and univariate effect sizes for comparison. State clearly that these are correlations within his games, not causal claims. Post a short signed comment on the issue; full numbers back to the coordinator.
