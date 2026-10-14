# CPBL Foreign Player Fit — POC

這是一個可重現的 Python POC，用公開數據檢驗一個問題：

> 「海外投打的技術特性與中職環境的匹配程度（CPBL Fit），能不能提供 AAA 傳統成績以外的招募資訊？」

最終依 2026 台灣棒球數據分析競賽的資格規則，推薦洋投與洋打各一名，另附候補名單。

- **分析快照**：2026-10-13（可簽性分層的日期也是 2026-10-13）
- **Go / No-Go**：有條件 GO（細球種可行；AAA 2025–26 逐球已補齊，但中職沒有 2026 逐球、也沒有轉速與位移）
- **修正 12（2026-10-14，只做驗證，首選不變）**：用 Baseball-Reference 做 2018–2026 中職洋將普查。53 位前一季在 AAA 投過的洋投，AAA K-BB%、K%、BB% 的排序都能延續到中職（主要檢定 ρ 0.34，90% CI 0.11 至 0.54），AAA 的 K-BB% 優勢約剩一半，但預測不了準確數值。洋打只有 2 人符合條件，做不了檢定（[`reports/cpbl_census.md`](reports/cpbl_census.md)）。
- **wOBA 權重修正（2026-10-14，首選不變）**：Johnny 發現設定檔標示 2024、實際是 FanGraphs 2013 年的權重。改成 2024 後全部重跑，首選與登錄的回測結論都不變；Stuff+ 的 S3 改成勉強通過，是否加進排名待決定（findings 0.11 節）。
- **修正 11（2026-10-14，首選不變）**：Codex 指出舊回測用到未來資料。改成每位來台者只用來台前的資料（中職環境、球場係數、標準化母體、模型訓練都只用更早的資料），現在是主要回測。Fit 仍沒有被證實能預測來台成績（[`reports/backtest_chrono.md`](reports/backtest_chrono.md)）。
- **修正 10（2026-10-14，首選不變）**：Johnny 要求看氣候。AAA 投手熱天（≥ 85°F）的 K%、揮空率、球速差異沒有跨季穩定（ρ −0.03 至 0.10，門檻 0.30），不加進分數（[`reports/weather.md`](reports/weather.md)）。
- **修正 9（2026-10-14，首選不變）**：Johnny 問為什麼只看 AAA 先發。AAA 後援另列「轉先發池」：角色校正後用先發池的尺度算分，標出先發經驗，並回推比較三種做法。加入後援涵蓋更多實際來台者，但排序沒有變好，所以只當參考名單（[`reports/conversion_pool.md`](reports/conversion_pool.md)）。
- **修正 7、7b、8（2026-10-14，只做驗證，首選不變；經 Codex 審查並修正）**：Johnny 決定在沒有書面授權下使用中職官網 2026 成績做驗證（見 limitations L10）。修正 7 用 2026 洋將成績擴大回測；7b 做 AAA→中職換算與 REBAS 轉播球速對 Trackman 的校正；修正 8 仿 FanGraphs 自建 AAA Stuff+ 並檢驗它的預測力。規則都在計算前 commit。
- **修正 6（2026-10-14）**：Johnny 要求更貼近球團實際找人。候選池改成 2025–26 兩季 AAA；依 MLB 官方異動紀錄與 40 人名單把候選人分成 T1 自由、T2 可能到期、T3 受控制、T4 40 人名單，首選只從 T1＋T2 選；並做 2024→2025 的招募回推驗證。規則在計算前 commit。
- **修正 5（2026-10-13）**：加入 AAA 球場校正（主場占一半場次）；投手 baseline 改以壓制力為主（K% 0.50、BB% 0.25、FIP 0.25）。規則在計算前 commit。
- **修正 3、4（2026-10-13）**：修正 3 把 Fit 的中職環境差異改在「同一投手、同一進壘位置」內估計，去掉投手品質混雜；修正 4 把 baseline 權重從人工判斷改成用 2025→2026 AAA 資料估計。兩項規則都在計算前 commit。
- **修正 2（2026-10-13）**：建立 REBAS 洋投 → MLBAM ID 對照表，並以 2024–25 來台洋投回測投手 Fit。規則在計算前先 commit；回測不改變推薦。
- **修正 1（2026-10-13）**：官方 `statsapi.mlb.com` 開放後，補上 2025 下半季與 2026 AAA 逐球資料、MLB 40 人名單與異動事實，並以官方數據交叉核對出賽資格。所有權重在修正前已預先登錄、沒有改動（見 `config/scoring.yaml` 與 git 歷史）。

## 結論摘要

| | 結論 |
|---|---|
| **A. 技術可行？** | **可行，但有條件。** AAA 端改用官方 statsapi，2024–26 逐球完整；CPBL 端用 REBAS 2024–25（ODC-By）。中職官方站的使用條款要求書面授權；排名沒有用它，修正 7 起在沒有授權的情況下只用來驗證（limitations L10）。整條流程可以重現，有 107 個自動化測試。限制：中職缺 2026 逐球、轉速與位移；中職球種為人工標記。 |
| **B. 找到 baseline 沒有的適性資訊？** | **投手：有獨立且穩定的資訊，但沒有證據能預測來台成績。** 投手 Fit 與 baseline 幾乎無關（修正 6 後 ρ = 0.04），跨季可靠度 0.63（修正 5 為 0.74）。修正 3 扣掉投手品質後，中職「怕高速球、擅打 SPL」的揮空面差異大多消失，剩下的差異跨季不穩定（ρ = 0.27）。<br>**打者：沒有可用的。** 打者 Fit 跨季可靠度 0.09（修正 5 為 0.22），未達 0.30 門檻。<br>**回測**：修正 3 之後主要檢驗 ρ = 0.24（90% CI −0.29 至 0.70），無法判定；Fit 與來台後 K-BB% ρ = −0.36。見 [`reports/backtest.md`](reports/backtest.md)。修正 7 加入 2026 成績後 n = 17，所有檢驗仍無法判定（[`reports/backtest_2026.md`](reports/backtest_2026.md)）。修正 12 的 2018–2026 普查（n = 53）顯示 AAA 的 K%、BB% 排序能延續到中職，但沒有檢驗 Fit。<br>**自建 AAA Stuff+（修正 8）**：預測下一季 AAA 的方向為正，但沒有贏過 baseline，wOBA 權重修正後，在 baseline 之外的額外資訊勉強通過（S3 ρ = 0.14，90% CI 0.02 至 0.25），主要檢定 S1 仍未通過；是否加進排名待 Johnny 決定，目前不在排名內（[`reports/stuff_plus.md`](reports/stuff_plus.md)、findings 0.11 節）。第一版因實作錯誤誤判為通過，Codex 審查後修正（findings 0.6 節）。Gerber 的 2025 Stuff+ 為 112。<br>**baseline 本身**：AAA 傳統指標預測下一季 AAA 表現的能力很弱；打者的資料估權重與人工權重預測力相同（0.21），卻選出不同首選。 |
| **C. 推薦與依據** | **洋投：Joey Gerber**（RHP，29 歲，T1：2026-08 被大都會釋出）。可簽者 57 人中組合第 1、客場第 2，K% 24.9%；但 ERA 6.94、先發只有 10 場，bootstrap 首位機率 0.36。依據**弱**。<br>**洋打：Daz Cameron**（右打 LF，29 歲，T2：2026-09 被藍鳥 outright）。全體 baseline 第 1，球場校正後 wOBA .427、客場 .439；可簽者中首位機率 0.47、前 3 機率 0.76。只有 430 PA。依據**中等偏弱**。<br>修正 5 的首選 Hardin、Moore 都是 T3（仍受合約控制），已換掉。**回推驗證**：可簽性讓名單更像球團的選擇，但名單內的能力排序和隨機差不多（[`reports/recruitment_backtest.md`](reports/recruitment_backtest.md)）。<br>合約金額與來台意願**未評估**。候補見 [`reports/recommendations.md`](reports/recommendations.md)。 |
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
│   ├── signability.py           # 修正 6：可簽性分層（異動紀錄＋40 人名單，T1–T4）
│   ├── recruitment_backtest.py  # 修正 6：招募回推驗證（2024→2025、2025→2026）
│   ├── evaluate.py              # Phase 5 Test A–D、bootstrap、fig5–6
│   ├── backtest.py              # 修正 2：洋投回測（AAA 來台前 Fit vs 中職成績）
│   ├── cpbl_site.py             # 修正 7：中職官網公開頁（Markdown，不呼叫 /api/）：身分、球速表
│   ├── backtest_2026.py         # 修正 7、7b：2026 回測與 AAA→中職換算（官網匯出表）
│   ├── velocity_calibration.py  # 修正 7b：REBAS 轉播球速 vs 官網 Trackman
│   ├── statsapi_release.py      # 修正 8：AAA 出手點與轉軸（statsapi game feed，2023–25）
│   ├── stuff_plus.py            # 修正 8：自建 AAA Stuff+ 與驗證
│   ├── conversion_pool.py       # 修正 9：AAA 後援轉先發池、角色校正、三種做法的回推
│   ├── weather.py               # 修正 10：AAA 逐場天氣與熱天個人差異的跨季可靠度
│   ├── backtest_chrono.py       # 修正 11：只用事前資料的投手回測
│   └── cpbl_census.py           # 修正 12：2018–2026 中職洋將普查與 AAA 延續性檢定（Baseball-Reference）
├── tools/cpbl_site_export.js    # 修正 7：以無頭瀏覽器操作官網排行榜的「匯出表單」
├── tests/                       # 107 tests：資格邊界、計算、跨來源一致性、官方交叉核對、回測規則、權重估計、球場係數、可簽性、官網匯出解析
├── reports/
│   ├── data_feasibility.md      # Phase 0 稽核與 Go/No-Go
│   ├── data_audit_summary.md    # 自動產生的覆蓋率表
│   ├── findings.md              # H1–H3、Test A–D
│   ├── backtest.md              # 修正 2 回測（backtest_tables.md 為自動產生的表）
│   ├── recruitment_backtest.md  # 修正 6 招募回推驗證（自動產生）
│   ├── backtest_2026.md         # 修正 7、7b 回測與換算（自動產生）
│   ├── velocity_calibration.md  # 修正 7b 球速校正（自動產生）
│   ├── stuff_plus.md            # 修正 8 Stuff+ 驗證（自動產生）
│   ├── conversion_pool.md       # 修正 9 轉先發池（自動產生）
│   ├── weather.md               # 修正 10 氣候（自動產生）
│   ├── backtest_chrono.md       # 修正 11 只用事前資料的回測（自動產生）
│   ├── cpbl_census.md           # 修正 12 中職洋將普查（自動產生）
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
| `recommendation_shortlist.csv` | 預先登錄規則下的整體前 10 名、非 40 人名單前 10 名與可簽者前 10 名，含 bootstrap 機率 |
| `screening_log_*.csv` 的 `sign_tier` 等欄 | 修正 6：每位候選人的可簽性層級、第一個職業球季、最後一份合約、最後一次異動 |
| `amendment6_old_vs_new.csv`、`amendment5_frozen/` | 修正 5 與修正 6 的排名對照；修正 5 結果的凍結副本 |
| `recruitment_backtest.json`、`park_factors_2023_2024.csv`、`park_factors_2024_2025.csv` | 修正 6 回推驗證的結果與各 fold 自己的球場係數 |
| `backtest2026_*.csv`、`backtest2026_summary.json` | 修正 7：2026 洋將的入選判定、合併回測名單、檢驗與換算結果 |
| `velocity_calibration_pairs.csv`、`velocity_calibration.json` | 修正 7b：同一投手的 REBAS 與 Trackman 速球均速 |
| `stuff_plus_pitchers.csv`、`stuff_plus_validation.csv`、`stuff_plus_summary.json` | 修正 8：每位 AAA 投手每季的 Stuff+、驗證用資料、模型與檢驗結果 |
| `conversion_pool_pitchers.csv`、`conversion_pool_summary.json`、`conversion_cohort.csv` | 修正 9：先發池＋轉先發池的分數、先發經驗標記、回推與中職回測結果 |
| `weather_splits.csv`、`weather_summary.json` | 修正 10：每位投手每季的熱天差異與跨季可靠度 |
| `backtest_chrono_cohort.csv`、`backtest_chrono_summary.json`、`park_factors_2023.csv` 等 | 修正 11：只用事前資料的預測值、檢定與滾動預測結果，以及單季球場係數 |
| `cpbl_census.csv`、`cpbl_census_pitchers.csv`、`cpbl_census_hitters.csv`、`cpbl_census_summary.json` | 修正 12：2018–2026 中職洋將名單與來源、洋投／洋打檢定名單、檢定與換算結果 |

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
python -m src.recruitment_backtest  # 修正 6 招募回推驗證（2023–25 AAA 季成績與異動紀錄）
python -m src.cpbl_site         # 修正 7：官網公開頁（約每秒一頁，不呼叫 /api/）
NODE_PATH=$(npm root -g) node tools/cpbl_site_export.js 投手 0 data/raw/cpbl_site/exports/pr_pitcher_2026.xlsx  # 其餘表見腳本開頭
python -m src.statsapi_release 2025 2023 2024  # 修正 8：約 6,500 場 AAA game feed
python -m src.stuff_plus        # 修正 8 Stuff+ 與驗證（約 1 分鐘）
python -m src.velocity_calibration
python -m src.backtest_2026     # 修正 7、7b（需要上面的匯出表與 Stuff+）
python -m src.conversion_pool   # 修正 9 轉先發池
python -m src.weather           # 修正 10 氣候（AAA 賽程天氣）
python -m src.backtest_chrono   # 修正 11 只用事前資料的回測（需要 backtest 與 backtest_2026 的輸出）
python -m src.cpbl_census       # 修正 12 中職洋將普查（Baseball-Reference，每分鐘 ≤ 20 次，首次約 1 小時）
pytest -q                       # 107 passed
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
- **中職官網（stats.cpbl.com.tw）**：只用於修正 7、7b 的驗證，未取得書面授權（limitations L10）；原始頁面與匯出檔不隨本 repo 散布。
- **AAA 資料**：源自 MLB Advanced Media，經 `armstjc/milb-data-repository`（MIT）鏡像取得。僅供非商業研究使用，逐球原始檔不隨本 repo 散布。
- **中職洋將普查（修正 12）**：資料來源 Baseball-Reference（© Sports Reference）。原始頁面不隨本 repo 散布，只保留衍生的普查表。

## 下一步：若要正式參賽，最值得投入的事

1. **改善洋打權重**：2026-10-14 決定照登錄規則用資料估權重（首選 Moore）；人工權重（首選 Tibbs）的預測力其實相同。可以加入預測力更好的打者指標（例如逐球的擊球品質、好球帶內揮空），但新增指標必須先登錄規則。
2. **擴大回測樣本**：修正 7 已加入 2026 成績，但只多 2 人（n = 17）。要再擴大，需要回測 AAA 後援型與日韓職來台的洋將，或等更多季。正式發表前應取得中職書面授權，或改用 REBAS 2026 重跑。
3. **估權重樣本的存活偏差**：把 2026 升上大聯盟的球員以換算成績補進樣本。
4. **人工查證招募可行性**：Gerber、Cameron 與候補的合約狀態、身價與來台意願；T2 要等 11 月初小聯盟自由球員名單公布確認。
5. **後援型洋投**：修正 9 已另列轉先發池。下一步是更好的轉先發校正（例如用大聯盟或 AAA 的角色轉換歷史、球種數、單場用球數），並用 2027 年新來台的洋投確認。
6. **Stuff+**：修正 8 的 S1、S3 原本都沒通過；wOBA 權重修正後 S3 勉強通過，依登錄規則可以在之後的修正加進排名，待 Johnny 決定。若要再試，可以加多季 MLB／AAA 資料訓練，或用 statsapi 的球種分類重建全部欄位，但都要先登錄新規則。
