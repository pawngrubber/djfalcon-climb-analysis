# Subagent: research strategist (critical review)
**Model:** Opus-class subagent · **Spawned by:** the main analysis session

## Instructions given
Read literally everything in this repo — every issue (1 through 21+) with all comments, every script and dataset filename — and act as a critical, statistics-oriented reviewer for the thread owner (a statistician). The job is not to run a new analysis but to propose what analyses should come *next*, and to say plainly what is NOT worth doing.

Deliverable: ONE new GitHub issue, "Research agenda: proposed next analyses (critical review)", containing a ranked list of 8-15 proposals ordered by expected value per unit effort. For each proposal state: (1) the question; (2) why it matters for the *actual* goal — dj climbing out of Iron; (3) data source and feasibility (Riot API, key expires ~8am PT Jul 8; op.gg MCP; cached data in `~/riot_cache/` and `data/`); (4) statistical design, pre-hardened against the confounds that have repeatedly burned this project — pseudo-replication (match/player clustering), outcome contamination (win/loss leaking into features), scope overreach (claims wider than the APIs can see), and marker-vs-driver conflation (between-rank separation ≠ within-rank win driver); (5) an honest expected-value assessment, explicitly labelling LOW-value ideas — including popular-sounding ones that won't survive the confounds — and why. Also flag any existing conclusion still under-supported and deserving re-examination.

Constraints: post NO interim comments on other issues; the single new issue is the entire output. Reference existing work concretely (write "issue 16" style, never bare #N). Sign the issue body with the standard subagent signature line linking back to this file.
