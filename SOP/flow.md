# 【FDC 異常分析結果】

### 🔍 偵測到異常訊號特徵：
- 正常訊號特性：
  - 前段 Output 為 1 / -1.4（表示無 output 或 idle）。
  - Flow 穩定於 1749 sccm（Setpoint 相符，Process 中段）。
  - 對應 Step 12，為 Etching 主反應階段。

- 異常徵兆：
  - Flow 在原本穩定 1749 sccm 時，突然出現高頻振盪或下降。
  - 沒有對應 Setpoint 改變，Output 自動下降或跳成 -1.4。
  - 信號維持 1749 不變但實際未送氣，可能為 sensor 假訊號。

---

## 🧠 可能異常原因（預測）：
**MFC Flow 異常中斷 / Signal Fail**
- Flow Signal 在 process 中段突然掉落，無對應 Setpoint 改變。
- `1749外訊號` 為常見 MFC sensor 故障預設值（sensor lock value）。
- 出現特徵包括：
  - MFC 感測器 internal fault，僅回報 default 值。
  - MFC 控制閥卡住，雖有開度但實際無 flow。
  - Bulk supply 或上游切瓶系統供氣異常。
  - 通訊中斷，signal frozen。

---

# 【對應 SOP】

## Case：MFC Flow 無法達成設定值（偏低 / 漏失 / 不穩）

### 1. 問題判斷：

- 是否 MFC Setpoint > 0，Output 卻為：
  - ❌ -1.4
  - ❌ 0
  - ❌ 高頻噪訊（波動 ±200 sccm）
  - ❌ 或維持 1749 但 Reaction Chamber 無壓力變化

  ✅ 若符合上述任何一項，啟動以下處理流程。

---

### 2. 初步處理流程：

a. 觀察 Trend：
   - 確認 Flow 是否在 process zone 中段瞬間跳變。
   - 對應 Step ID、Process Time、Recipe Parameter。
   - 是否只影響單一 MFC，或同批次其他通道也異常。

b. 設備檢查項目：
   - 確認氣瓶是否切換（切瓶中斷導致瞬間掉壓）。
   - 驗證 MFC Controller 是否 Reset（檢查 Alarm Log）。
   - 檢查 MFC 的電源狀態與 Signal 線路接頭。
   - 檢查 MFC Valve 是否卡住（無法 fully open）。
   - 若 Output 值仍為 1749，但反應腔未達預期壓力，疑似 MFC 感測器 lock。

---

### 3. 設備維修建議：

- 執行以下動作驗證 MFC 功能：
  1. 送 100% Setpoint，量測實際流量與 Output 是否一致。
  2. 執行 Zero / Span 校正（僅限已冷卻且非 corrosive gas）。
  3. 若 Output 恆為 1749（無變化）→ 更換 MFC sensor 模組。

- 若 flow 出現「-1.4」且未恢復，通常為：
  - 控制線中斷或 Controller Crash。
  - MFC control loop disable。
  - 設備進入 idle，但未正確 reset。

---

### 4. FDC 規則建議：

- 建議設定以下異常偵測邏輯：
  - Setpoint > 100 且 Output < 10，持續超過 2 秒。
  - Output 在 Step 中段突然下降 > 80%。
  - Output 恆為 1749 ± 1，且對應 Valve Position = 0（疑似 lock value）。

---

### 5. 紀錄與通報項目：

- 若發生 MFC Flow 異常，應記錄以下資訊：
  - Step ID
  - MFC Channel 編號（Gas Type）
  - Setpoint、Output、異常時間戳
  - 是否發生切瓶 / 換氣瓶時間點
  - 是否與其他批次同樣異常模式

- 通報對象：
  - Module Owner
  - Gas Supply 團隊（若為氣瓶相關）
  - FDC Owner 進行 alarm rule 優化

---

### 📘 附註：

- 若異常出現在 Etching 主反應 Step，需通報 Process Owner 檢查 wafer impact。
- 若使用腐蝕性氣體（如 Cl₂、CHF₃），建議半年更換 MFC 或排檢一次。