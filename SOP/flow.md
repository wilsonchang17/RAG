# 【FDC 異常分析結果】

### 🔍 偵測到異常訊號特徵：
- 變數名稱：`flow_output`
- 異常範圍：Index 15 ~ 18
- 訊號特性：
  - 前段（Index 8~13）Flow 穩定在 1749 sccm
  - 後段 Flow 掉回 1 / -1.4（異常值）
  - 對應 Step 12，應為 Etching 主反應階段

---

## 🧠 可能異常原因（預測）：
**MFC Flow 異常中斷 / Signal Fail**
- Flow Signal 在 process 中段突然掉落，無對應 Setpoint 改變
- `-1.4` 為常見 MFC sensor fault default 值
- 若 MFC output 持續為 -1.4 / 0，極可能為：
  - MFC 感測器故障
  - 閥體卡住 / 壓力不足
  - 上游切瓶動作失敗（氣瓶沒切成功）

---

# 【對應 SOP】

## Case：MFC Flow 無法達成設定值（偏低 / 漏失）

### 1. 問題判斷：
- 是否 MFC Set > 0 但 Output 為 0、-1.4 或極低？
  - ✅ 是 → 優先檢查氣瓶壓力與 MFC 狀態。

### 2. 處理流程：
- 查看 Flow trend，確認是否在 process zone 出現中斷。
- 檢查 MFC 通訊或 controller 是否重啟 / reset。
- 檢查氣瓶是否已空 → Auto Changeover 是否動作成功？

### 3. 設備處理：
- 嘗試執行 MFC Zero / Span 校正。
- 若 Flow 恆為 -1.4 → 更換 MFC 或檢查 signal 線路。
- 若同批次其他 MFC 也異常 → 疑似 bulk supply 壓力不足。

### 解法與建議：
- 異常值 `-1.4` 建議直接作為 sensor-fail 的判別門檻。
- 可設定「Set > 100 且 Output < 10」持續 2 秒為 FDC 異常。
- 建議對腐蝕性氣體定期更換 MFC。
- 發生異常時記錄：Step ID / Flow Channel / Setpoint / Output / Anomaly Index。

