## 【Case 1：MFC Flow 無法達成設定值（偏低 / 飄移 / 零輸出）】

### 📌 異常徵象（FDC Pattern）：
- MFC Set > 0，但 Output 為 0 或長時間偏低（> ±5～10%）
- Octotools 工具顯示該段 Flow Output 有明顯下降斜率（由 STL 解析）
- chamber 壓力與等離子體功率同步異常（可能無法達到 process condition）

### 🔍 分析依據（Octotools 可判斷的條件）：
- `flow_output` vs `flow_setpoint` gap 明顯拉大
- `flow_output_slope` < -Threshold（預設可從 STL 模型輸出）
- `anomaly_span_duration` 超過 3 秒以上
- 可用欄位：Step ID、Gas Name、Channel ID、Setpoint、Output

---

### 1. 問題判斷：
- 是否特定氣體 flow 無法達到 recipe 設定值 ±5%？
  - 是 → 檢查該路 MFC 與 upstream 氣瓶壓力。
  - 否 → 判定為誤報或短期擾動。
- 是否 Output 為 0 且 Setpoint 有值？
  - 是 → 優先檢查供氣端與 MFC 本體。

---

### 2. 處理流程：
- 查看 MFC Flow vs Setpoint trend，確認時間對齊。
- 檢查氣瓶是否已空或壓力不足（是否有 Auto Changeover）。
- 檢查氣體為高腐蝕性者（如 Cl₂、BCl₃）是否造成 MFC 阻塞。
- 若 MFC 為 Digital type，確認通訊回傳值與實際對應。

---

### 3. 設備處理：
- 對異常 MFC 進行 Zero / Span 校正。
- 若完全無輸出 → 更換 MFC（sensor 或內部 valve 損壞）。
- 檢查 upstream shut-off valve 是否延遲開啟或閥座磨損。
- 若其他機台同時出現類似現象，需回溯 bulk supply 壓力。

---

### 解法與建議：
- 建議將「Flow Set > 20 sccm 且 Output < 5 sccm」超過 3 秒時觸發 FDC alarm。
- 每條 flow signal 應與 chamber pressure 建立 cross correlation rule。
- 腐蝕性氣體使用的 MFC 建議定期更換（PM ≤ 3 個月）。
- 將異常 span 與對應 flow profile 儲存於事件記錄供機台比對。

---