# 【Etching FDC 壓力異常 SOP 彙整】

---

## 【Case 1：Chamber 壓力不穩定（Oscillation / Drift）】

### 1. 問題判斷：
- 是否在穩壓區段觀察到壓力振盪或趨勢飄移？
  - 是 → 進入動作流程。
  - 否 → 記錄觀察並持續追蹤 FDC Trend。

### 2. 處理流程：
- 確認是否有 recipe 中壓力 ramp 不一致設定。
- 檢查 APC（Automatic Pressure Controller）門開度是否抖動或異常延遲。
- 檢查 roughing pump/roots pump 真空效能是否降低（漏氣、老化等）。

### 3. 設備檢查與維護：
- 重新校正壓力 sensor（Capacitance manometer / Pirani）。
- 檢查 APC actuator 驅動元件（是否卡住、動作延遲）。
- 排查 vacuum line 是否堵塞或有 particle 堆積。

### 解法與建議：
- 壓力振盪常與 APC 控制回路參數（PID tuning）有關，建議請設備工程師重新調整 loop。
- 若短時間內頻繁出現壓力不穩，建議暫停該機台製程，避免造成 CD variation。

---

## 【Case 2：Pump 故障導致壓力偏高】

### 1. 問題判斷：
- 是否在 pump 開啟後 chamber 壓力無法拉低至指定 base pressure？
  - 是 → 疑似 Pump 故障或 load lock 泄漏。
  - 否 → 進入一般檢查。

### 2. 處理流程：
- 確認 pump 電流是否異常上升（馬達卡死或軸承問題）。
- 確認 chamber 是否過度髒污，影響真空抽氣效率。
- 確認 foreline 有無液體反吸（chemical backstream）。

### 3. 設備處理：
- 執行 pump 更換或清洗保養（PM）。
- 確認 foreline trap 有無定期更換。
- 若為 dry pump，確認隔膜或 dry stage 是否失效。

### 解法與建議：
- 避免長時間 idle 導致 pump 冷凝污染，建議加裝加熱保溫系統。
- 排程定期 base pressure test 以提早偵測 pump 劣化。

---

## 【Case 3：閥門（MFC / APC）反應遲緩導致壓力偏移】

### 1. 問題判斷：
- 壓力變化是否伴隨 flow 波動？
  - 是 → 檢查 MFC 是否老化或校正失準。
  - 否 → 檢查 APC 控制回應時間與 hysteresis。

### 2. 處理流程：
- 檢查氣體 flow 是否超過 MFC 正常線性控制區。
- 檢查 APC 電磁閥是否延遲開啟 / 關閉。

### 3. 維護與預防：
- 對 MFC 進行 zero/span 校正。
- 替換異常 slow-response 的 APC 閥門。
- 確保排氣路徑壓力 feedback sensor 無延遲。

### 解法與建議：
- Flow 與壓力耦合控制（Gas/Pressure Loop）需同時檢測，建議建立 Cross FDC rule。
- 若有中斷式 flow profile，建議 recipe 改為緩降型 pressure profile 降低衝擊。

---

## 【Case 4：He 背壓（Backside Pressure）異常】

### 1. 問題判斷：
- 是否 ESC 背面無法維持設定 He 壓力？
  - 是 → 有可能是 wafer contact 不良 或 O-ring 損壞。
  - 否 → 繼續監測。

### 2. 處理流程：
- 檢查 ESC 表面是否有異物或顆粒，影響 wafer flatness。
- 檢查 backside channel 是否堵塞或 valve leak。
- 驗證 He 供氣系統是否有恆壓源。

### 3. 設備保養：
- 更換 deteriorated O-ring 或 sealing。
- 清理 ESC 表面與 He 管路。
- 檢查 He 壓力 sensor 與 leak detector。

### 解法與建議：
- 若連續發生 backside leak，建議檢查 wafer 規格與 flatness 公差是否偏差。
- 可裝設 inline leak monitor 提早偵測壓力 drop。

---

## 【Case 5：突然 loss of vacuum（真空瞬間掉壓）】

### 1. 問題判斷：
- 是否壓力突然瞬間升高並跳機？
  - 是 → 疑似大漏氣或 power interlock 問題。
  - 否 → 判斷為暫態 spike，可進行一次性追蹤。

### 2. 處理流程：
- 檢查 chamber door 是否關閉密合。
- 檢查 gate valve 是否異常動作。
- 檢查 power loss 導致 rough pump 停止抽氣。

### 3. 緊急處理：
- 立即隔離該機台，並使用 helium sniffer 檢測漏點。
- 若為多次跳機，考慮更換 gate valve 或 door seal。

### 解法與建議：
- loss of vacuum 多伴隨 wafer scrap，建議搭配 wafer presence sensor 防止誤操作。
- 若 chamber 髒污導致 seal 閉合不良，建議增加 plasma clean 頻率。

---