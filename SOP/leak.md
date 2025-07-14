【Dry Etcher - Leak Rate 監控與異常處理標準作業流程】

1. 監控對象：
   - 設備模組：Dry Etching Chamber（含 L/L、Process Chamber、Transfer Module）。
   - 監控參數：Leak Rate（洩漏率），單位：Torr/min。

2. 正常條件定義：
   - 正常狀況下，Leak Rate 應接近 0（通常定義為 < 0.01 Torr/min）。
   - 若 Leak Rate = 0，視為氣密良好。
   - Leak Rate 記錄於 Pumping 結束後 5 秒內之系統報表，並自動上傳 FDC 系統。

3. 異常判定條件：
   - Leak Rate > 0.01 Torr/min，視為異常。
   - 若連續三批次 Leak Rate 不為 0，或突升超過 0.1 Torr/min，需立即排查。
   - 若 FDC 觸發異常事件（如 Six Sigma、Z-score 超標），強制中斷後續製程。

4. 處理流程：

   a. 停機與隔離：
      - 暫停目前 Chamber 製程，標記該模組為「Engineering Mode」。
      - 通知 Shift Engineer 與 Tool Owner 進行處理。

   b. 硬體檢查順序建議：
      1. 確認 Chamber 門是否關閉確實，有無 Particle 卡阻。
      2. 檢查 Load Lock O-ring 是否硬化、脫落、或表面有灰塵。
      3. 檢查 Process Chamber 閥件（如 Throttle Valve、MFC inlet、Gate Valve）是否鬆動。
      4. 真空管路與接頭是否有鬆脫或彎折。
      5. 是否近期更換過 Component（Sensor、Liner、ESC），有未鎖緊之虞。

   c. 使用測試工具：
      - 啟用 Leak Check 模式（Recipe: LeakTest01）觀察壓力變化趨勢。
      - 若設備支援，使用 He Leak Detector 進行局部漏氣檢查。

   d. 修復與驗證：
      - 修復後，重新 Pump Down 並執行 Leak Rate 驗證。
      - Leak Rate 測試需連續 2 次小於 0.01 Torr/min 才可視為修復完成。
      - 更新設備保養記錄並回報 PM（Preventive Maintenance）排程。

5. 記錄與通報：

   - 將異常紀錄上傳至：
     - FDC 系統（含：時間、Leak Rate 數值、批號、Chamber ID）。
     - Tool History Log（登記異常類型與處理結果）。
     - 若異常發生於量產批次，需通知 Process Owner 評估產品風險。

   - 異常通知對象：
     - 值班工程師（Shift Engineer）
     - Module Owner（設備負責人）
     - 設備 PM 團隊（若與定期維護相關）

6. 復機條件：
   - 完成 Leak Rate 驗證（< 0.01 Torr/min）。
   - Process Owner 與 Tool Owner 簽核確認可恢復生產。
   - 於 MES 系統中解除 Chamber Engineering Hold。