# 【Etching Chamber Temperature Drift SOP】

## 1. 問題判斷：
- Chamber 溫度是否超出製程設定容忍值（±3°C）？
  - 是 → 進入緊急處理流程。
  - 否 → 持續監控趨勢變化 → 判斷是否需執行 Preventive Maintenance。

## 2. 緊急處理流程（製程中）：
- 暫停該批蝕刻製程（Pause/Abort）。
- 檢查 Heater / ESC 運作狀況。
- 確認溫度 Sensor 是否正常。
- 儲存並檢視該批次溫度 Trend。

## 3. 設備檢查與校正：
- 執行 ESC 或 Chuck 的溫控模組檢查。
- 檢查 backside He cooling 是否正常（壓力、流量、是否漏氣）。
- 若異常屬感測器，安排 Thermocouple 校正作業。
- 清理 chamber 內部堆積物或異常熱源區域。

## 4. Wafer 處理決策：
- 溫度偏差大但已執行中止：判定為報廢或送 QA 特別量測（依站別規定）。
- 偏差小於 ±3°C：視情況判定是否進入 Run As Is 流程，並加強後段 SEM / CD 檢查。

## 5. 後續改善行動：
- 啟動 Preventive Maintenance，定期清理與檢查熱控模組。
- 更新 FDC 設備異常偵測規則。
- 加入 SPC 監控，設定 Early Warning 門檻。

---

# 【解法與建議】

- 若為 He 背壓異常導致 wafer 升溫，建議定期更換 sealing O-ring 與背壓控制閥。
- 若發現 trend 呈現長期溫度慢慢上升，可能為 ESC 表面汙染，建議使用專用清潔流程或更換 ESC。
- 對於頻繁發生溫漂機台，應安排更密集的 TC 校正與記錄，以防止良率波動。
- 若製程允收範圍小，可評估導入多點溫控 sensor 系統進行交叉校驗。