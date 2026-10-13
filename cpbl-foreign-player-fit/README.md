# CPBL Foreign Player Fit — POC

這是一個可重現的 Python POC，用公開數據檢驗一個問題：

> 「海外投打的技術特性與中職環境的匹配程度（CPBL Fit），能不能提供 AAA 傳統成績以外的招募資訊？」

最終依 2026 台灣棒球數據分析競賽的資格規則，推薦洋投與洋打各一名，另附候補名單。

- **分析快照**：2026-10-13
- **Go / No-Go**：有條件 GO（細球種可行；AAA 2025–26 逐球已補齊，但中職沒有 2026 逐球、也沒有轉速與位移）
- **修正 5（2026-10-13）**：加入 AAA 球場校正（主場占一半場次）；投手 baseline 改以壓制力為主（K% 0.50、BB% 0.25、FIP 0.25）。規則在計算前 commit。
- **修正 3、4（2026-10-13）**：修正 3 把 Fit 的中職環境差異改在「同一投手、同一進壘位置」內估計，去掉投手品質混雜；修正 4 把 baseline 權重從人工判斷改成用 2025→2026 AAA 資料估計。兩項規則都在計算前 commit。
- **修正 2（2026-10-13）**：建立 REBAS 洋投 → MLBAM ID 對照表，並以 2024–25 來台洋投回測投手 Fit。規則在計算前先 commit；回測不改變推薦。
- **修正 1（2026-10-13）**：官方 `statsapi.mlb.com` 開放後，補上 2025 下半季與 2026 AAA 逐球資料、MLB 40 人名單與異動事實，並以官方數據交叉核對出賽資格。所有權重在修正前已預先登錄、沒有改動（見 `config/scoring.yaml` 與 git 歷史）。

## 結論摘要

| | 結論 |
|---|---|
| **A. 技術可行？** | **可行，但有條件。** AAA 端改用官方 statsapi，2024–26 逐球完整；CPBL 端用 REBAS 2024–25（ODC-By）。中職官方站可連線，但使用條款要求書面授權，所以沒有使用。整條流程可以重現，有 78 個自動化測試。限制：中職缺 2026 逐球、轉速與位移；中職球種為人工標記。 |
| **B. 找到 baseline 沒有的適性資訊？** | **投手：有獨立且穩定的資訊，但沒有證據能預測來台成績。** 投手 Fit 與 baseline 的 ρ = −0.07，跨季可靠度 0.74。修正 3 扣掉投手品質後，中職「怕高速球、擅打 SPL」的揮空面差異大多消失，剩下的差異跨季不穩定（ρ = 0.27）。<br>**打者：沒有可用的。** 打者 Fit 跨季可靠度 0.22，未達 0.30 門檻。<br>**回測**：修正 3 之後主要檢驗 ρ = 0.24（90% CI −0.29 至 0.70），無法判定；Fit 與來台後 K-BB% ρ = −0.36。見 [`reports/backtest.md`](reports/backtest.md)。<br>**baseline 本身**：AAA 傳統指標預測下一季 AAA 表現的能力很弱；打者的資料估權重與人工權重預測力相同（0.21），卻選出不同首選。 |
| **C. 推薦與依據** | **洋投：Tyson Hardin**（RHP，25 歲，Nashville）。不變。組合排名第 1；非 40 人名單中 FIP 第 1、K% 第 4；bootstrap 首位機率 0.40、前 3 機率 0.71；客場第 3。依據**中等**。<br>**洋打：Robert Moore**（兩打 2B，24 歲，Lehigh Valley）依登錄規則（資料估權重）；**James Tibbs III**（左打 RF，24 歲，Oklahoma City）在人工權重下為首選，球場校正後 wOBA .419。兩組權重預測力相同，依據都**很弱**；Moore 客場 wOBA 只有 .294。<br>合約與意願**未評估**。候補見 [`reports/recommendations.md`](reports/recommendations.md)。 |
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
│   ├── baseline_weights.py      # 修正 4：用 2025→2026 AAA 官方季成績估計 baseline 權重
│   ├── park.py                  # 修正 5：AAA 球場係數（主客場比較）與主場一半校正
│   ├── fit_score.py             # Phase 4 Pitcher／Hitter Fit
│   ├── evaluate.py              # Phase 5 Test A–D、bootstrap、fig5–6
│   └── backtest.py              # 修正 2：洋投回測（AAA 來台前 Fit vs 中職成績）
├── tests/                       # 78 tests：資格邊界、計算、跨來源一致性、官方交叉核對、回測規則、權重估計、球場係數
├── reports/
│   ├── data_feasibility.md      # Phase 0 稽核與 Go/No-Go
│   ├── data_audit_summary.md    # 自動產生的覆蓋率表
│   ├── findings.md              # H1–H3、Test A–D
│   ├── backtest.md              # 修正 2 回測（backtest_tables.md 為自動產生的表）
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
| `baseline_pitchers.csv`、`baseline_hitters.csv` | 傳統排名，包含所有 z 分數；`baseline_rank_judgment` 是人工權重下的排名 |
| `park_factors.csv` | 修正 5：每隊各事件（SO、BB、HBP、1B、2B、3B、HR）的球場係數與信度 |
| `baseline_weights.csv` | 修正 4 的估計權重、90% CI、樣本數、交叉驗證相關，以及人工權重對照 |
| `pitcher_fit_results.csv`、`hitter_fit_results.csv` | Fit 各成分、Fit／baseline／組合排名；招募可行性另立欄位 |
| `bootstrap_ranks_*.csv`、`evaluation_summary.json` | 穩定性與反證檢驗的原始結果 |
| `mlb_season_lines.csv`、`mlb_season_audit.csv` | Retrosheet 衍生的 MLB 2024–25 球員賽季 PA 與出局數 |
| `mlb_official_crosscheck_*.csv` | statsapi 官方 2024–25 PA 與出局數，以及據此重新判定的資格（與 Retrosheet 100% 一致） |
| `recruitment_status.csv` | 每位候選人 2026-10-13 的 40 人名單、2026 MLB 出賽、最新異動 |
| `backtest_screening.csv`、`backtest_cohort.csv`、`backtest_summary.json` | 回測：每位洋投的入選判定、來台前 AAA 的 Fit 與成績、來台後的中職成績、檢驗結果 |
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
python -m src.baseline_weights  # 修正 4 權重（evaluate 找不到時會自動執行）
python -m src.evaluate          # 候選人 → 名單事實 → baseline → Fit → Test A–D（300 次 bootstrap，約 30 分鐘）
python -m src.backtest          # 修正 2 回測；首次執行會另外下載 2023 AAA 2,156 場
pytest -q                       # 78 passed
```

各步驟都是 deterministic 的：bootstrap 的 seed 寫在 `config/scoring.yaml`。

## 方法要點

- **同一套定義**：兩聯盟都用「揮空 ÷ 揮棒」計算 Whiff%，觸擊排除，界外擦棒算接觸；wOBA 用同一組通用權重。所有分子與分母都由逐球原始計數重新計算。
- **Baseline 權重**：原本的人工權重在任何排名產生前就已 commit；修正 4 改用資料估計（以 2025 指標預測 2026 聯盟校正 FIP／wOBA 的非負迴歸），人工權重保留作為對照。z 分數在 ±2.5 截尾；AAA 兩聯盟分別做聯盟校正。
- **Fit 只看「形狀」**：中職與 AAA 的整體實力落差先扣除，再看球種、球速、左右手的相對匹配。AAA 平均型投手或打者的 Fit 恰為 0，有單元測試保證。
- **穩定性檢驗**：比賽層級的 Poisson bootstrap、權重 Dirichlet 擾動、2024 與 2025 分季、移除極端值。反證檢驗包含投手內／投手間球速斜率、分類合併、測速 ±1 mph、洋投代理、選樣偏差，以及只看客場。

## 資料授權與致謝

- **CPBL 逐球資料**：資料來源為野球革命 Rebas Open Data，ODC-By 1.0。
- **Retrosheet**：The information used here was obtained free of charge from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at www.retrosheet.org.
- **Chadwick Baseball Bureau register**：ODC-By 1.0。
- **AAA 資料**：源自 MLB Advanced Media，經 `armstjc/milb-data-repository`（MIT）鏡像取得。僅供非商業研究使用，逐球原始檔不隨本 repo 散布。

## 下一步：若要正式參賽，最值得投入的事

1. **決定洋打權重**：資料估權重（首選 Moore）與人工權重（首選 Tibbs）預測力相同。可以加入預測力更好的打者指標（例如逐球的擊球品質、好球帶內揮空），但新增指標必須先登錄規則。
2. **取得中職 2026 逐球資料的授權**：請先確認競賽規則是否已授權參賽者使用中職官方資料；若沒有，向中職申請。
3. **擴大回測樣本**：需要 2026 來台洋投的中職成績（REBAS 2026 尚未釋出，或開放 zh.wikipedia.org）。
4. **估權重樣本的存活偏差**：把 2026 升上大聯盟的球員以換算成績補進樣本。
5. **人工查證招募可行性**：Hardin 與候補的合約類型、選擇權與意願。
