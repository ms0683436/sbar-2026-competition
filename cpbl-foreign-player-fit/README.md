# CPBL Foreign Player Fit — POC

這是一個可重現的 Python POC，用公開數據檢驗一個問題：

> 「海外投打的技術特性與中職環境的匹配程度（CPBL Fit），能不能提供 AAA 傳統成績以外的招募資訊？」

最終依 2026 台灣棒球數據分析競賽的資格規則，推薦洋投與洋打各一名，另附候補名單。

- **分析快照**：2026-10-13
- **Go / No-Go**：有條件 GO（細球種可行；AAA 2025–26 逐球已補齊，但中職沒有 2026 逐球、也沒有轉速與位移）
- **修正 1（2026-10-13）**：官方 `statsapi.mlb.com` 開放後，補上 2025 下半季與 2026 AAA 逐球資料、MLB 40 人名單與異動事實，並以官方數據交叉核對出賽資格。所有權重在修正前已預先登錄、沒有改動（見 `config/scoring.yaml` 與 git 歷史）。

## 結論摘要

| | 結論 |
|---|---|
| **A. 技術可行？** | **可行，但有條件。** AAA 端改用官方 statsapi，2024–26 逐球完整；CPBL 端用 REBAS 2024–25（ODC-By）。中職官方站可連線，但使用條款要求書面授權，所以沒有使用。整條流程可以重現，有 46 個自動化測試。限制：中職缺 2026 逐球、轉速與位移；中職球種為人工標記。 |
| **B. 找到 baseline 沒有的適性資訊？** | **投手：有獨立且穩定的資訊，但效度未證實。** 投手 Fit 與 baseline 的 ρ = −0.05，2025 對 2026 跨季可靠度 0.82，主要內容是速球球速與球種組合。「中職怕高速球」這個訊號經投手內斜率檢驗，主要是投手品質混雜（findings D1）。<br>**打者：沒有可用的。** 打者 Fit 跨季可靠度 0.22，未達預先登錄的 0.30 門檻，H3 不成立。 |
| **C. 推薦與依據** | **洋投：Tyson Hardin**（RHP，25 歲，Nashville）。組合排名第 1，非 40 人名單中 bootstrap 首位機率 0.48、前 3 機率 0.80，客場第 4。依據**中等**。<br>**洋打：James Tibbs III**（左打 RF，24 歲，Oklahoma City）。非 40 人名單 baseline 第 1；客場整體第 7、非 40 人名單中第 1，但 bootstrap 首位機率只有 0.08。依據**弱**。<br>修正前的首選 Whisenhunt（巨人 60 天傷兵名單）與 Thompson（洛磯 40 人名單）都在 MLB 40 人名單上。兩位新首選都不在任何 40 人名單上，2026 也沒有 MLB 出賽；合約與意願**未評估**。候補見 [`reports/recommendations.md`](reports/recommendations.md)。 |
| **D. 下一步** | 見本頁最後一節。 |

## 專案結構

```
cpbl-foreign-player-fit/
├── README.md
├── pyproject.toml
├── config/scoring.yaml          # 所有權重與門檻（在任何排名產生前 commit，見 git log）
├── data/
│   ├── raw/                     # 下載檔（git-ignored，以 src.fetch_data 重建）
│   ├── processed/               # 彙總輸出（下方列出）
│   └── sources.md               # 來源、期間、授權、雜湊值
├── src/
│   ├── fetch_data.py            # 下載與建置（REBAS、MiLB 鏡像、Retrosheet、Chadwick）
│   ├── statsapi_pbp.py          # 修正 1：statsapi AAA 逐球（2025-05-20 起與 2026），鏡像格式
│   ├── roster_status.py         # 修正 1：40 人名單、2026 MLB 出賽、異動紀錄（事實，不計分）
│   ├── data_audit.py            # Phase 0 欄位覆蓋率 → reports/data_audit_summary.md
│   ├── eligibility.py           # MLB 2024/2025 PA、IP 資格（以出局數計，季別分開；statsapi 交叉核對）
│   ├── pitch_schema.py          # 兩聯盟共用的逐球欄位與指標定義
│   ├── rebas_loader.py          # CPBL 逐球 → 共用 schema
│   ├── milb_loader.py           # AAA 逐球（鏡像＋statsapi，去重）、季總計、逐場紀錄
│   ├── cpbl_profile.py          # Phase 1 環境輪廓、H1 GLM、fig1–4
│   ├── milb_candidates.py       # Phase 2 候選池與篩選紀錄
│   ├── baseline.py              # Phase 3 傳統排名
│   ├── fit_score.py             # Phase 4 Pitcher／Hitter Fit
│   └── evaluate.py              # Phase 5 Test A–D、bootstrap、fig5–6
├── tests/                       # 46 tests：資格邊界、計算、跨來源一致性、官方交叉核對
├── reports/
│   ├── data_feasibility.md      # Phase 0 稽核與 Go/No-Go
│   ├── data_audit_summary.md    # 自動產生的覆蓋率表
│   ├── findings.md              # H1–H3、Test A–D
│   ├── evaluation_tables.md     # 自動產生的 bootstrap 與環境表
│   ├── limitations.md
│   └── recommendations.md
└── figures/                     # fig1–fig6
```

## 主要輸出（`data/processed/`）

| 檔案 | 內容 |
|---|---|
| `cpbl_batter_profile.parquet` | 中職打者在各維度、各 cell 的指標：球種、球種群、投手手、對戰手、platoon、速球球速帶、進壘區、球種×投手手。每列都有分母（揮棒、好球帶外投球、打席）、估計值、Wilson CI、與聯盟的差、shrinkage 估計，以及小樣本旗標 |
| `cpbl_pitching_environment.parquet` | 球種用量（全體、LHP、RHP、SP、RP）、各球種球速分布、速球球速帶、左右投與左右打占比 |
| `aaa_batter_profile.parquet`、`aaa_pitching_environment.parquet` | AAA 對照組，定義完全相同 |
| `env_whiff_contrast.parquet`、`env_damage_contrast.parquet`、`fit_environment_cells.csv` | 中職 vs AAA 的對比 |
| `screening_log_pitchers.csv`、`screening_log_hitters.csv` | 每位 2026 AAA 球員的篩選結果、MLBAM ID、MLB 出賽數與證據、排除原因 |
| `baseline_pitchers.csv`、`baseline_hitters.csv` | 傳統排名，包含所有 z 分數 |
| `pitcher_fit_results.csv`、`hitter_fit_results.csv` | Fit 各成分、Fit／baseline／組合排名；招募可行性另立欄位 |
| `bootstrap_ranks_*.csv`、`evaluation_summary.json` | 穩定性與反證檢驗的原始結果 |
| `mlb_season_lines.csv`、`mlb_season_audit.csv` | Retrosheet 衍生的 MLB 2024–25 球員賽季 PA 與出局數 |
| `mlb_official_crosscheck_*.csv` | statsapi 官方 2024–25 PA 與出局數，以及據此重新判定的資格（與 Retrosheet 100% 一致） |
| `recruitment_status.csv` | 每位候選人 2026-10-13 的 40 人名單、2026 MLB 出賽、最新異動 |
| `recommendation_shortlist.csv` | 預先登錄規則下的整體前 10 名與非 40 人名單前 10 名，含 bootstrap 機率 |

## 重現

需求：Python 3.11 以上、gcc、make（用於建置 chadwick）。

```bash
cd cpbl-foreign-player-fit
pip install -e .[dev]
python -m src.fetch_data        # 約 2.5 GB 原始 CSV；Retrosheet、register clone 到 config.paths.external
python -m src.statsapi_pbp      # 3,681 場 AAA game feed（約 90 MB gzip，可續傳）
python -m src.eligibility       # MLB 2024/2025 球員賽季行
python -m src.cpbl_profile      # 環境輪廓 + fig1–4（約 1 分鐘）
python -m src.data_audit
python -m src.evaluate          # 候選人 → 名單事實 → baseline → Fit → Test A–D（300 次 bootstrap，約 30 分鐘）
pytest -q                       # 46 passed
```

各步驟都是 deterministic 的：bootstrap 的 seed 寫在 `config/scoring.yaml`。

## 方法要點

- **同一套定義**：兩聯盟都用「揮空 ÷ 揮棒」計算 Whiff%，觸擊排除，界外擦棒算接觸；wOBA 用同一組通用權重。所有分子與分母都由逐球原始計數重新計算。
- **Baseline 權重預先登錄**：權重在任何排名產生前就已 commit（git 第一個 commit）。z 分數在 ±2.5 截尾；AAA 兩聯盟分別做聯盟校正。
- **Fit 只看「形狀」**：中職與 AAA 的整體實力落差先扣除，再看球種、球速、左右手的相對匹配。AAA 平均型投手或打者的 Fit 恰為 0，有單元測試保證。
- **穩定性檢驗**：比賽層級的 Poisson bootstrap、權重 Dirichlet 擾動、2024 與 2025 分季、移除極端值。反證檢驗包含投手內／投手間球速斜率、分類合併、測速 ±1 mph、洋投代理、選樣偏差，以及只看客場。

## 資料授權與致謝

- **CPBL 逐球資料**：資料來源為野球革命 Rebas Open Data，ODC-By 1.0。
- **Retrosheet**：The information used here was obtained free of charge from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at www.retrosheet.org.
- **Chadwick Baseball Bureau register**：ODC-By 1.0。
- **AAA 資料**：源自 MLB Advanced Media，經 `armstjc/milb-data-repository`（MIT）鏡像取得。僅供非商業研究使用，逐球原始檔不隨本 repo 散布。

## 下一步：若要正式參賽，最值得投入的事

1. **取得中職 2026 逐球資料的授權**：這是目前最大的缺口。`stats.cpbl.com.tw` 已可連線，但數據由 robots.txt 禁止的 `/api/` 載入，使用條款要求書面授權。請先確認競賽規則是否已授權參賽者使用；若沒有，向中職申請。取得後可補上 2026 中職環境，以及 TrackMan 的轉速與位移，做「球路形狀（IVB、HB、轉速）」的匹配。
2. **建立 AAA→CPBL 歷史對照表**：整理 2015–2025 年來台洋將的中文名與 MLBAM ID，做時間正確的回溯驗證，檢驗「來台前一年的 Fit」能不能預測「來台後的 K-BB% 與 wOBA」。statsapi 已可查歷年 AAA 逐球，這一步現在做得到。沒有這一步，Fit 永遠只是假說。
3. **把投手 Fit 改寫成投手內的效應**：D1 顯示速球溢價主要是投手間的混雜。可改用投手固定效應或 Mundlak 分解估計中職環境對比，並把擊球傷害對比（兩季同向，ρ = 0.95）當作主要成分。
4. **打者 Fit 改用穩定得較快的成分**：例如對各球種的接觸率、好球帶內揮空率，並要求至少兩季樣本。
5. **人工查證招募可行性**：Hardin、Tibbs 與候補的合約類型、選擇權與意願；Hardin 9 月短暫進傷兵名單的原因。11 月小聯盟自由球員申報後，名單事實需要重查。
