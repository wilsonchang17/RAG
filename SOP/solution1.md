

# 溫度

溫度上升
對應 SOP：《Etching Chamber Temperature Drift SOP—Extended》
	•	觸發條件符合 SOP 1、5 條件：
	•	“SPC 顯示平均溫度緩慢上升，達 +3.8°C”，對應 SOP 第 1 點「若 Cpk 低於 1.33 或平均溫度持續上升，啟動 PM 評估」
	•	趨勢沒達到即時告警，但符合 SOP 建議引入 EWMA。
	•	處理對應 SOP 第 3、4、6 條：
	•	使用熱像儀判斷 ESC 表面熱斑 → SOP 3.1
	•	判定表面污染 → 對應 Root-Cause「ESC 表面汙染」與建議「溶劑清洗／更換」
	•	改善對應 SOP 5 條：
	•	建議縮短 ESC 清洗週期 → 呼應 SOP 5 的 Preventive Maintenance 計畫


# 壓力 

對應 SOP：《Etching FDC 壓力異常 SOP 彙整／Case 1》
	•	觸發條件符合 SOP 第 1 條：
	•	穩壓階段壓力振盪 ±0.5 mTorr，符合「是否在穩壓區段觀察到壓力振盪」
	•	處理符合 SOP 第 2 條與設備調整步驟：
	•	確認 Recipe 與 APC 門開度控制異常 → SOP 建議檢查 ramp 與 APC 控制元件
	•	調整 PID loop（P 值） → SOP 建議由設備工程師重新 tuning loop
	•	後續改善建議符合 SOP 建議：
	•	要求新裝 APC 上傳 loop 參數 → 類似 SOP 的「建議請工程師重新調整 loop」


# 漏率

對應 SOP：《Dry Etcher - Leak Rate 監控與異常處理標準作業流程》
	•	觸發條件完全符合 SOP 異常條件：
	•	Leak Rate > 0.01，且連續 3 批非 0 → 符合 SOP「連續三批不為 0，需排查」
	•	處理符合 SOP 的 b.硬體檢查 → c.測試工具 → d.修復驗證邏輯：
	•	使用 LeakTest01 → SOP 建議 recipe 名稱一致
	•	找出 Liner 安裝不良 → 對應 SOP 建議的「Component 更換未鎖緊」風險
	•	重測兩次通過 < 0.01 → 符合 SOP 的驗收條件
	•	改善對應 SOP 的「更新 PM Checklist」與 Tool History Log


# 流量
對應 SOP：《FDC Flow 異常（MFC 無法達成設定值）》
	•	觸發條件符合 SOP 問題判斷：
	•	Flow Output 固定為 1749、但壓力未變化 → 符合 SOP 所述 Lock Value 假訊號
	•	初步處理符合 SOP 第 2 條：
	•	檢查 Trend、Alarm Log → 與 SOP 建議一致
	•	設備處理符合 SOP 第 3 條：
	•	更換 sensor 模組，控制器 reset → 對應 SOP 的具體建議
	•	FDC Rule 改善建議符合 SOP 第 4 條：
	•	新增 Cross Check 條件「Valve = 0」時即告警 → 完全對應 SOP 建議的異常邏輯

