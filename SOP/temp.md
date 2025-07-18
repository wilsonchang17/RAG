# 【Etching Chamber Temperature Drift SOP—Extended】

> **適用範圍**：Dry／ICP／RIE 等蝕刻機台  
> **目的**：在不穩定溫度造成良率下滑前，快速判斷、處置、預防

---

## 1. 問題判斷

1. 讀取即時 FDC 溫度；若偏離 recipe 設定 ± 3 °C 以上，立即啟動緊急處理流程。  
2. 檢視過去 25 lots 的 SPC 趨勢。若 Cpk 低於 1.33，或平均溫度持續上升，啟動 PM 評估。  
3. 比對同一批中中心與邊緣溫度差 ΔT。若超過 2 °C，先檢查 ESC 背壓與 He 冷卻。

---

## 2. 緊急處理流程（製程中）

1. **立刻暫停或中止** 當前 lot，將 wafer 保留在 chamber 內。  
2. 進行五點量測（中心＋四角）確認溫度均勻度。  
3. 下載過去 60 分鐘的 Temperature Trend log 以利後續分析。  
4. 將 heater 切換至備用 profile（如有設定），空載跑 3 片 dummy wafer 觀察數據。  
5. 若溫度仍異常，轉入「設備檢查與校正」。

---

## 3. 設備檢查與校正

### 3.1 ESC／Heater

- 量測各 zone 電阻，應在基準值 ± 5 % 內。  
- 讀取 He 背壓；低於 2 Torr 或高於 9 Torr 代表 O-ring 或控制閥可能損壞。  
- 以紅外熱像判斷 ESC 表面是否出現局部熱斑。

### 3.2 溫度量測鏈

- 使用外掛標準 TC 交叉比對 on-board TC，兩者差值需低於 1 °C。  
- 檢查 TC 線材有無磨損、接地不良或鬆脫。

### 3.3 背面冷卻路徑

- He 流量需大於 1 slm，壓降不應超過 10 %。  
- 以 He sniffer 測漏，洩漏率須低於 1 × 10⁻⁸ atm·cc/s。

### 3.4 內部清潔

- 量測 ESC 表面 polymer 厚度；若超過 5 µm，執行化學濕洗或更換 ESC。  
- 清除 showerhead、RF 接地彈片等處累積沉積物。

---

## 4. Wafer 處理決策

- **溫度偏差大於 ± 6 °C**：該批 wafer 歸類為報廢或送 QA 進行 SEM／EDS 特檢。  
- **溫度落在 ± 3 ~ 6 °C**：同樣送 QA 特檢，並在後段製程加強 CD／Overlay 抽測。  
- **溫度偏差不超過 ± 3 °C**：可 Run-As-Is，但需標記 lot 並於後段抽測確認。

---

## 5. 後續改善行動

1. **Preventive Maintenance**  
   - 短期內將 ESC O-ring 換線週期由 1,000 wafers 縮短至 500 wafers。  
   - 每月進行 heater zone balance tuning。  
2. **FDC 規則更新**  
   - 新增 Rate-of-Rise 監控：溫度連續 30 秒上升速率 > 0.2 °C/s 即警報。  
3. **SPC 與 Early-Warning**  
   - 導入 EWMA 管制圖（λ = 0.3），自動計算並更新 UCL／LCL。  
4. **培訓與紀錄**  
   - 事故發生三日內完成根因分析（5-Why 與 FTA），上傳 MES 供追溯。

---

## 6. 典型 Root-Cause 與對策

- **He 背壓異常**  
  - *根因*：O-ring 硬化、閥體磨損或雜質堵塞。  
  - *即時對策*：更換 O-ring、清潔或更換控制閥。  
  - *長期改善*：改用高耐熱 Kalrez® O-ring，並加裝 in-line filter。  

- **長期溫度緩慢上升**  
  - *根因*：ESC 表面汙染或 RF 接地接觸不良。  
  - *即時對策*：清洗或直接替換 ESC；檢查接地彈片。  
  - *長期改善*：週期性霧化乾洗並建立 ESC 清潔履歷。  

- **單一 zone 溫差過大**  
  - *根因*：Heater zone 部分失效。  
  - *即時對策*：停機更換 Heater。  
  - *長期改善*：導入多 zone PID 控制並記錄回饋參數。  

- **批內中心-邊緣 ΔT 過大**  
  - *根因*：背面冷卻流場不均。  
  - *即時對策*：重新校正 He 流量及背壓。  
  - *長期改善*：增設多點背壓 sensor 進行閉迴路控制。  

---

## 7. 快速解法與建議

1. 定期更換 He sealing O-ring 與背壓閥體，避免背壓飄移。  
2. 若溫度 trend 緩慢爬升，多半是 ESC 汙染；執行專用溶劑濕洗或直接換 ESC。  
3. 對於經常溫漂的機台，將 TC 校正頻率提升至每週一次，並保留校正紀錄做趨勢分析。  
4. 若製程允收範圍窄，可加裝三點 TC＋多點 IR 交叉校驗，或導入 AI-Based 預測模型。  
5. 任何 Heater／ESC 維修後，必須跑 10-片 Golden Lot 做 CD／Overlay 確認，合格後方可量產。

---

> **備註**：所有數值與週期僅為範例，實際設定請依公司規範與特定製程需求調整。