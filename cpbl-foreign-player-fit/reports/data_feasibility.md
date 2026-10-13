# Phase 0：資料可行性稽核（Data Feasibility Audit）

- 分析快照：2026-10-13
- 資料取得：2026-10-13（UTC）
- 機器產生的欄位覆蓋率：[`data_audit_summary.md`](data_audit_summary.md)
- 來源與授權：[`../data/sources.md`](../data/sources.md)

## 0. 執行環境限制（先講結論）

本次執行環境的對外網路政策**擋掉了題目指定的三個官方來源**：

- `stats.cpbl.com.tw`、`www.cpbl.com.tw`
- `baseballsavant.mlb.com`
- `statsapi.mlb.com`

以上連線都在 CONNECT 階段收到 HTTP 403。網頁抓取工具也無法解析這些網域。

我們沒有繞過這個限制，改用以下三組**公開、有授權**的替代來源：

1. **CPBL 端**：REBAS 野球革命 Open Data（ODC-By 1.0），GitHub release 上有 2024、2025 兩季完整的逐球資料。
2. **AAA 端**：GitHub 上鏡像 MLB Stats API MiLB game feed 的 `armstjc/milb-data-repository`，內容包含 Statcast 球種、球速、轉速、位移、擊球初速。
3. **MLB 出賽資格**：Retrosheet 例行賽事件檔，經 Chadwick 工具 `cwdaily` 計算，再用 Chadwick register 對應 MLBAM ID。

如果環境管理者把上列官方網域加入允許清單，就能補上第 4 節列出的缺口。

## 1. CPBL（中職一軍例行賽）

| 題目要求的項目 | 官方 `stats.cpbl.com.tw` | 實際使用：REBAS Open Data | 可行？ |
|---|---|---|---|
| 2026 一軍例行賽打者數據 | 有，但被環境擋掉 | **沒有 2026**（REBAS 只釋出到 2025） | ⚠️ 用 2024–2025 代替 |
| 不同球種的打者 Whiff% | 官方只分 fastball／breakingball 兩類（依第三方文件與 11 場樣本 payload） | ✅ 可以自己算。細球種有 FF/SI/FC/SL/CU/CH/FO/FS/KN/EP，**人工標記**。原始計數：揮棒數、揮空數 | ✅ |
| 不同球種的打者 Chase% | 官方有「追打%」，定義未取得 | ✅ 可算。進壘座標為**人工依轉播畫面記錄**，2024 與 2025 皆有 99.9% 覆蓋。好球帶用規則尺寸（\|x\|≤50、\|y\|≤60），與 Statcast zone 1–9 的定義一致 | ✅（有量測誤差） |
| 不同球種的 wOBA | 官方只有整體 wOBA PR 值 | ✅ 以打席最後一球的球種歸屬計算（Savant 慣例），兩聯盟使用同一組通用權重 | ✅ |
| 投手球種使用量 | 官方只分兩類 | ✅ 細球種 | ✅ |
| 球速 | TrackMan | ✅ km/h，2024 有 96.9%、2025 有 94.5%。**來源為轉播測速，並非 TrackMan**，打錯的值（如 463、12）已清掉 | ✅（量測系統不同） |
| 轉速 | TrackMan 有 | ❌ REBAS 沒有 | ❌ |
| 位移（movement） | TrackMan 軌跡多項式 | ❌ REBAS 沒有 | ❌ |
| 左右打、左右投對戰 | 有 | ✅ 每個打席都有 `batterHand` 與 `pitcherHand`（2025 投手手別缺 3%） | ✅ |
| 進壘區域 | 有 | ✅ 座標（人工記錄） | ✅（有誤差） |
| 合法 CSV 匯出 | 未觀察到；第三方文件指出有 JSON API | ✅ ODC-By 授權的 JSON zip | ✅ |
| 原始計數，而非只有比率 | API 為逐球 | ✅ 逐球，所有分子分母都能自行計算 | ✅ |
| 擊球品質 | TrackMan EV | ⚠️ 只有人工標記的 hardness（H/M/S）與彈道，**不能與 AAA 的 EV 直接比較** | ⚠️ 只在聯盟內使用 |

樣本量：

| 季別 | 場數 | 投球數 | 打席數 | 投手 | 打者 |
|---|---|---|---|---|---|
| 2024 | 360 | 103,620 | 27,399 | 168 | 173 |
| 2025 | 360 | 102,488 | 27,057 | 168 | 174 |

REBAS 的資料問題與處理方式：

- **重複資料**：zip 內同時有整季 JSON 與逐場 JSON，內容完全重疊，以 `(seasonId, seq)` 去重。
- **只有姓名、沒有 ID**：打席資料只記姓名，用同場 box score 的「隊伍＋姓名」對回 `playerId`，對應率 100%。

## 2. MiLB（AAA）

| 題目要求的項目 | 官方 Savant | 實際使用：MLB feed 鏡像 | 可行？ |
|---|---|---|---|
| Pitch type、velocity | 有，被擋 | ✅ 2024 全季；2025 只有 3/28–5/25 | ⚠️ 缺 2025 下半季與整個 2026 |
| Movement（pfx）、spin、extension | 有 | ✅ 覆蓋率 97–99.9% | ✅（同上期間） |
| Whiff%、K%、BB% | 有 | ✅ 由 pitch code 與打席描述計算；2026 整季 K%、BB% 用賽季總計 | ✅ |
| EV、HardHit% | 有 | ✅ 2024–25 的 EV 覆蓋 99% | ✅（2024–25） |
| Barrel%、xwOBA | 有 | ❌ 鏡像中 `estimated_woba`、`launch_speed_angle`、`delta_run_exp` 欄位**全是空值** | ❌ |
| 逐球或分組資料 | 有 | ✅ 逐球 | ✅ |
| 投手角色、先發工作量 | 有 | ✅ 2026 賽季總計有 GS、IP；逐球資料可判斷每場第一任投手 | ✅ |
| 2026 整季傳統數據 | 有 | ✅ 季總計檔最後更新 2026-10-01，171,799 打席，30 隊 | ✅ |
| 2026 逐場紀錄 | 有 | ✅ 2026-04-01 至 09-20 | ✅（缺 3 月開季幾天，約 3% 打席） |

鏡像的資料問題與處理方式：

- **重複列（重大品質問題）**：爬蟲程式碼會把每場成功解析的比賽 concat 兩次，失敗時又重複附加上一場。原始逐球列數因此約為真實值的 2.1 倍。已用 `(game_pk, at_bat_number, pitch_number)` 去重，並有測試覆蓋。
- **球數記錄時點**：`balls`、`strikes` 是投球**之後**的球數，已平移為投球之前。
- **打席結果需解析**：`events` 欄位是空的，打席結果改由 `des` 描述文字解析，取最左邊符合的動詞。有 11 個解析案例的測試。
- **兩個聯盟環境差很多**：International League 與 Pacific Coast League 打擊環境差距大，baseline 以球員所屬聯盟的加權平均做聯盟校正。**沒有做球場校正**，第 5 階段 D7 以客場成績另行檢查。

## 3. MLB 出賽資格

- **計算方式**：Retrosheet 2024、2025 例行賽事件檔，只取 `{yyyy}{TEAM}.EVA/.EVN`，排除季後賽 `.EVE`、明星賽與 `.EB*`。經 `cwdaily` 得到每位球員每場的 `B_PA` 與 `P_OUT`（出局數＝投球局數×3）。
- **IP 精度**：以出局數計算，1/3 局與 2/3 局都精確。
- **場數**：2024 年 2,429 場、2025 年 2,430 場，與 Retrosheet 該季 game log（`gl2024.txt`、`GL2025.TXT`）的場數一致。
- **抽樣核對**：Ohtani 2024 731 PA、Judge 704 PA、Skubal 192.0 IP、Sale 177.2 IP，與公開數據一致。
- **ID 對應**：MLBAM 經 Chadwick register 對到 Retrosheet ID。
- **「零出賽」的確認規則**：
  1. 有 Retrosheet ID，但某季沒有任何紀錄：該季事件檔是完整的全季檔，所以判定為 0。
  2. 沒有 Retrosheet ID：必須同時滿足以下兩點，才判定為 0；只要任一點不成立就判定為 `unknown`（不合格）。
     - register 的 `mlb_played_last` 早於 2024 或為空值；
     - 姓名不在 2024、2025 任何 MLB `.ROS` 名單上。
- **本次結果**：候選人中 `unknown` 為 0 人。
- **與官方的差異**：本流程不是 `statsapi.mlb.com` 官方 API，而是 Retrosheet 的整理版本。兩者 PA／IP 在極少數判例上可能相差 1，例如妨礙打擊或中斷的打席。距離門檻 ±2 以內的球員應人工複核；本次候選人中沒有落在這個範圍的。

## 4. Go / No-Go 判定

| 閘門 | 判定 | 說明 |
|---|---|---|
| CPBL 是否有細球種數據 | **GO（細球種）** | REBAS 有 7 類可用球種，每類樣本在 2 季合計 6,000 次揮棒以上。代價是球種為人工標記，分類可能與 Statcast 不一致，見 findings Test D3。 |
| 按球種拆分且樣本足夠 | **GO** | 環境層級的每個 cell 最少：CPBL FB94+ 有 1,803 次揮棒、AAA SPL 有 6,696 次揮棒。 |
| 2026 CPBL 環境 | **NO-GO → 以 2024–25 代替** | 2026 只能從官方站取得（被擋），且官方只分兩類球種。 |
| AAA 2026 球員級球種能力 | **NO-GO → 以 2024 全季＋2025 前兩個月代替** | 鏡像在 2025-05 之後就沒有逐球檔。2026 只有傳統數據。 |
| 轉速、位移適性 | **NO-GO** | CPBL 端沒有轉速與位移，無法比對「球路形狀」。 |
| xwOBA、Barrel% | **NO-GO** | 鏡像中這些欄位是空的。 |
| MLB 資格 | **GO** | 完整且可稽核，見 `data/processed/screening_log_*.csv`。 |

**整體判定：有條件 GO。**

可以做的：以「球種（7 類）× 速球球速帶 × 投打左右手」做適性匹配。

主要 blocker：

1. 沒有 2026 CPBL 與 2026 AAA 的逐球資料。
2. CPBL 沒有轉速、位移。
3. CPBL 的球種與進壘點為人工標記，球速為轉播測速。

受影響的部分：

- **適性指數只能算「有 2024–25 AAA 追蹤資料」的球員**：投手 47/122、打者 87/196。
- 2026 才升上 AAA 的年輕球員只有 baseline 排名，沒有 Fit。這是**選樣偏差**，見 findings D6。
