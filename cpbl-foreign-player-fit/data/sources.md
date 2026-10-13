# Data sources

Analysis snapshot date: **2026-10-13**. Data was downloaded on **2026-10-13 (UTC)**. Nothing dated after the snapshot is used. `milb_loader.to_schema` and `milb_loader.game_logs` enforce this, and `test_data_consistency.py` tests it.

| # | Source | What we use | Period | Granularity | Access | Licence / terms | Integrity |
|---|---|---|---|---|---|---|---|
| 1 | **REBAS 野球革命 Open Data** (`rebas-tw/rebas.tw-open-data`, release tags `v0.1.0-2024`, `v0.1.0-2025`) | CPBL first-team regular season, pitch by pitch: pitch type (hand-tagged), velocity (km/h), location coords, handedness, PA results, batted-ball hardness/trajectory (manual) | 2024-03-30 → 2024-10-11, 2025-03-29 → 2025-10-06 (360 + 360 games) | pitch | GitHub release asset `CPBL-{yyyy}-OpenData.zip` | **ODC-By 1.0.** Attribution required: "資料來源：野球革命 Rebas Open Data". REBAS states that part of the data is recorded by hand and does not come from stadium instruments. | sha256 2024 zip `1711009c9d3bac68…`, 2025 zip `eb84d824e0fdce03…` |
| 2 | **armstjc/milb-data-repository** (GitHub releases `pbp`, `season_player_batting`, `season_player_pitching`, `game_player_stats`) | Mirror of the MLB Stats API MiLB game feed for AAA (International League + Pacific Coast League): Statcast pitch type, release speed, spin, movement, plate location, zone, EV/LA, pitch result codes, play descriptions. Also 2026 season totals and game logs. | Pitch level: 2024-03-29 → 2024-09-22 (2,232 games); 2025-03-28 → 2025-05-25 (743 games). The mirror publishes no later pitch-level files. Season totals: 2026, last modified 2026-10-01. Game logs: 2026-04-01 → 2026-09-20. | pitch / player-season / player-game | GitHub release assets `{yyyy}_{m}_aaa_pbp.csv` etc. | Repository is MIT-licensed. The underlying data is © MLB Advanced Media, which allows individual, non-commercial use. We do not redistribute the pitch-level files: they are git-ignored and re-downloaded by `src/fetch_data.py`. | The raw files contain duplicated games because of a scraper bug, so we de-duplicate on `(game_pk, at_bat_number, pitch_number)`. 2024: 1,398,450 → 675,688 rows. 2025: 464,388 → 220,758 rows. |
| 3 | **Retrosheet** event files via `chadwickbureau/retrosheet` (commit `ee82e50b`, 2026-08-13) | MLB **regular-season** event files `{yyyy}{TEAM}.EVA/.EVN` for 2024 and 2025. Postseason `.EVE`, All-Star and box-score `.EB*` files are excluded. | 2024 (2,429 games), 2025 (2,430 games) | player-game, built with `cwdaily` | `git clone --sparse` | "The information used here was obtained free of charge from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at www.retrosheet.org." | `tests/test_eligibility.py` checks the output against published lines (Ohtani 2024 731 PA, Judge 704 PA, Skubal 192.0 IP, Sale 177.2 IP) and against the game counts |
| 4 | **Chadwick Baseball Bureau register** (`chadwickbureau/register`, commit `781c175e`, 2026-10-02) | MLBAM ↔ Retrosheet ID crosswalk, birth year, `mlb_played_first/last` | as of 2026-10-02 | person | `git clone` | ODC-By 1.0 | none |
| 5 | **Chadwick tools 0.10.0** | `cwdaily`, which turns event files into player-game lines (B_PA, P_OUT = outs recorded) | n/a | n/a | GitHub release tarball, built locally | GPL (tooling only) | none |

## Sources investigated but not usable from this environment

| Source | Status | What it would add |
|---|---|---|
| `stats.cpbl.com.tw` (official CPBL advanced-stats site) | **Blocked** by the execution environment's egress policy (HTTP 403 at CONNECT). From third-party documentation in `ruan6047/cpbl-analytics/docs/CPBL_SITE_MAP.md` and a sample of 11 game payloads: the site exposes JSON (`/api/proxy/v1/games/{id}`, `/players/logs`, `/leaderboards/*`) with per-pitch TrackMan (speed, spin, extension, release, plate location, 3-D trajectory polynomial). Pitch type is only `fastball` / `breakingball` (TaggedPitchType/AutoPitchType). No CSV export was observed. | 2026 CPBL environment; spin and movement; instrument-measured velocity |
| `baseballsavant.mlb.com/statcast-search-minors` | **Blocked** (403) | 2025 Jun-Sep and 2026 AAA pitch-level data, xwOBA |
| `statsapi.mlb.com` | **Blocked** (403) | official MLB playing time; Retrosheet substitutes for it |
| `www.cpbl.com.tw` | **Blocked** (403). Third-party notes say it sits behind an anti-bot JS challenge since 2026-06. | official box scores |

To reproduce with the official sources, the environment owner can add these hosts under the network allow-list:

- `stats.cpbl.com.tw`
- `baseballsavant.mlb.com`
- `statsapi.mlb.com`

## Reproduce

```bash
pip install -e .[dev]              # plus statsmodels, scipy
python -m src.fetch_data           # downloads 1-5, builds chadwick tools (~2.5 GB raw CSV)
python -m src.eligibility          # MLB season lines (Retrosheet -> cwdaily)
python -m src.cpbl_profile         # CPBL/AAA environment profiles + figures 1-4
python -m src.data_audit           # reports/data_audit_summary.md
python -m src.evaluate             # candidates, baseline, Fit, Tests A-D, figures 5-6 (~10 min)
pytest -q
```
