# djfalcon climb analysis

Data analysis of the League of Legends account **djfalcon#NA1** (NA), answering one question:

> "No matter what I do, I feel like I can never climb out of Iron. Why?"

The full question and the analysis live in the **[issue thread](../../issues/1)** — start there.

## What's here

| Path | What it is |
|---|---|
| `data/riot_games.csv` / `.json` | Every match played in 2026 (May 27 – Jul 7, 151 games), pulled from Riot's official match-v5 API. One row per game: result, champion, position, K/D/A, CS, gold, damage, vision, objectives. |
| `scripts/riot_pull.py` | Rebuilds the dataset from the Riot API. `uv run scripts/riot_pull.py` with `RIOT_API_KEY` in your env ([get a key](https://developer.riotgames.com)). |
| `scripts/analyze.py` | Computes every number and renders every chart in the issue. `uv run scripts/analyze.py` — dependencies resolve ephemerally via the PEP 723 header. |
| `charts/` | Rendered charts, embedded in the issue thread. |

## Reproducing

```bash
export RIOT_API_KEY=RGAPI-...   # your own key; never committed
uv run scripts/riot_pull.py     # ~4 min (rate-limited)
uv run scripts/analyze.py       # stats to stdout, charts to charts/
```
