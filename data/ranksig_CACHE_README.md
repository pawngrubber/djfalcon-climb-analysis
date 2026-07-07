# Rank-signature raw cache & data-drift policy

The rank-signature model (`scripts/rank_signature.py`, issue 19) is built from Riot
match **bodies + timelines** for IRON/SILVER/BRONZE cohorts and djfalcon's ranked games.

## Where the raw data lives
Raw API responses are **not** committed to git (`.gitignore` excludes `data/ranksig/`
and `*.gz`). Two durable copies exist on `paul-desktop`:

- **Session working copy:** `data/ranksig/` (gzipped, one blob per match:
  `{tier, match_id, body, timeline}` for cohorts; `{match_id, body, timeline}` for dj).
- **Durable archive (source of truth):** `/home/paul/riot_cache/ranksig/` — mirror of the
  above, outside the session scratchpad. 216 matches (IRON 38 / SILVER 40 / BRONZE 40 /
  dj 98). Read from here; never re-fetch what is already cached.

Git tracks only: the scripts, the extracted `data/ranksig_features.csv`
(one row per player-game), and this README.

## Cache-first rule
If you need a match already pulled, **read it from disk** (`ranksig_pull.py` is resumable
and skips files already present). Do not re-hit the API for cached matches.

## Data-drift policy — what is immutable vs mutable
1. **Immutable (cache is permanent, keyed by match id):** completed match **bodies** and
   **timelines**. Matches only publish after they end, so their contents never change.
   Safe to cache forever and re-read.
2. **Mutable (always re-fetch live, never serve from cache):** match-ID lists
   (`by-puuid/.../ids`), league-exp entries / player rank/LP, and any op.gg or other
   aggregate. These track a moving population and go stale immediately.
3. **Patch is a covariate, not a nuisance to pool over.** Cohort pulls are patch-stamped
   snapshots of a *moving* population: a player's "3 most recent ranked games" can reach
   back months, so a single cohort spans many patches (this snapshot: cohort games span
   ~16.1–16.13; dj concentrated 16.11–16.13). `game_version` is stored on **every** cached
   match and is a column in `data/ranksig_features.csv`. Any future re-analysis must treat
   patch as a covariate (or stratify), **not** silently pool across patches — item/rune/
   objective changes shift the very behaviors the model reads.
