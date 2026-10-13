# CPBL Foreign Player Fit — POC

這是一個可重現的 Python POC，用公開數據檢驗一個問題：

> 「海外投打的技術特性與中職環境的匹配程度（CPBL Fit），能不能提供 AAA 傳統成績以外的招募資訊？」

最終依 2026 台灣棒球數據分析競賽的資格規則，推薦洋投與洋打各一名，另附候補名單。

- **分析快照**：2026-10-13
- **Go / No-Go**：有條件 GO（細球種可行；但沒有 2026 年逐球資料，CPBL 端也沒有轉速與位移）

## 結論摘要

| | 結論 |
|---|---|
| **A. 技術可行？** | **可行，但有條件。** 官方網站（stats.cpbl.com.tw、Savant、statsapi）都被執行環境擋掉，改用三組有授權的替代來源：REBAS 2024–25 中職逐球（ODC-By）、MLB feed 的 AAA 鏡像、Retrosheet。整條流程可以重現，也有 41 個自動化測試。限制是：缺 2026 年逐球資料、CPBL 沒有轉速與位移、CPBL 球種為人工標記。 |
| **B. 找到 baseline 沒有的適性資訊？** | **投手：有獨立資訊，但效度未證實。** 投手 Fit 與 baseline 的 ρ = 0.11，跨季可靠度 0.53–0.68，主要內容是速球球速與球種組合。不過「中職怕高速球」這個最強的訊號，經投手內斜率檢驗後，主要是投手品質混雜（findings D1）。<br>**打者：沒有。** 打者 Fit 跨季可靠度 ρ ≈ −0.05，等同雜訊，H3 不成立。 |
| **C. 推薦與依據** | **洋投：Carson Whisenhunt**（LHP，26 歲）。baseline 與 Fit 一致，bootstrap 首位機率 0.54，不受球場影響。但 2026-07-25 之後沒有 AAA 出賽，原因需查證。<br>**洋打：Sterlin Thompson**（左打外野，25 歲）。baseline 首位，權重擾動 81%，客場排名第 3。但追蹤樣本小。<br>兩人的依據都屬「中等或中等偏弱」，**招募可行性未評估**。候補見 [`reports/recommendations.md`](reports/recommendations.md)。 |
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
│   ├── data_audit.py            # Phase 0 欄位覆蓋率 → reports/data_audit_summary.md
│   ├── eligibility.py           # MLB 2024/2025 PA、IP 資格（以出局數計，季別分開）
│   ├── pitch_schema.py          # 兩聯盟共用的逐球欄位與指標定義
│   ├── rebas_loader.py          # CPBL 逐球 → 共用 schema
│   ├── milb_loader.py           # AAA 逐球（含去重）、季總計、逐場紀錄
│   ├── cpbl_profile.py          # Phase 1 環境輪廓、H1 GLM、fig1–4
│   ├── milb_candidates.py       # Phase 2 候選池與篩選紀錄
│   ├── baseline.py              # Phase 3 傳統排名
│   ├── fit_score.py             # Phase 4 Pitcher／Hitter Fit
│   └── evaluate.py              # Phase 5 Test A–D、bootstrap、fig5–6
├── tests/                       # 41 tests：資格邊界、計算、跨來源一致性
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

## 重現

需求：Python 3.11 以上、gcc、make（用於建置 chadwick）。

```bash
cd cpbl-foreign-player-fit
pip install -e .[dev]
python -m src.fetch_data        # 約 2.5 GB 原始 CSV；Retrosheet、register clone 到 config.paths.external
python -m src.eligibility       # MLB 2024/2025 球員賽季行
python -m src.cpbl_profile      # 環境輪廓 + fig1–4（約 1 分鐘）
python -m src.data_audit
python -m src.evaluate          # 候選人 → baseline → Fit → Test A–D（300 次 bootstrap，約 10 分鐘）
pytest -q                       # 41 passed
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

1. **取得官方逐球資料**：這是最大的槓桿。請環境管理者開放 `stats.cpbl.com.tw`、`baseballsavant.mlb.com`、`statsapi.mlb.com`。
   - 補上 2026 年中職環境，以及 TrackMan 量測的轉速與位移。
   - 補上 2025 年 6 月以後與 2026 年的 AAA 逐球資料。
   - 這樣才能把剩下 75 名投手、109 名打者也納入 Fit 計算，並做「球路形狀（IVB、HB、轉速）」的匹配，這是目前完全做不到的部分。
2. **建立 AAA→CPBL 歷史對照表**：整理 2015–2025 年來台洋將的中文名與 MLBAM ID，做時間正確的回溯驗證，檢驗「來台前一年的 Fit」能不能預測「來台後的 K-BB% 與 wOBA」。沒有這一步，Fit 永遠只是假說。
3. **把投手 Fit 改寫成投手內的效應**：D1 顯示速球溢價主要是投手間的混雜。可改用投手固定效應或 Mundlak 分解估計中職環境對比，並把擊球傷害對比（兩季同向、較穩定）當作主要成分。
4. **放棄打者 Fit，或改成高可靠度的指標**：例如對各球種的接觸率、好球帶內揮空率這類穩定得快的成分，並要求至少 2 個賽季的樣本。
5. **人工查證招募可行性**：查證 Whisenhunt 7/25、Veen 8/9 之後沒有出賽的原因，以及合約與 40 人名單狀態。
