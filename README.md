# RAG
LangChain + FAISS 快速實作 RAG 系統範例

系統架構概述 (Retrieval-Augmented Generation 流程)

**檢索增強生成（RAG）**系統通常包含兩大階段： ￼ ￼
	1.	文件索引階段（離線處理）：將SOP文件資料載入後，先切分成較小的段落（chunks），再對每個段落計算向量嵌入並建立向量索引（通常使用向量資料庫，如FAISS） ￼。這個階段相當於把所有知識點預先編碼存入資料庫，方便日後快速相似度查詢。
	2.	問題檢索與回答階段（線上查詢）：針對使用者輸入的自然語言問題，系統先利用Retriever在向量索引中檢索最相關的幾個內容段落 ￼。接著將問題與這些檢索到的段落一併送入LLM（大型語言模型）進行答案生成，得到最終回答 ￼。LLM 會根據檢索到的文件內容來產生答案，確保回答有依據於SOP文件知識。

以上流程確保了AI Agent在回答時引入外部知識：先從SOP文件中查找相關內容，再由LLM基於這些內容生成答案。下面將說明各步驟的實作細節。

文件準備與切分嵌入

1. 資料準備：將所有SOP文件內容整理為純文字格式（例如對PDF圖像先執行OCR得到文字，或直接使用提供的文字版）。可以手動將SOP文件彙整成文字檔，或利用LangChain的文件載入器（Document Loader）來讀取PDF等格式。確保每份文件內容都以文字形式可取得。

2. 切分文件（Chunking）：由於LLM對單次輸入的長度有限制，而且直接對長文搜尋效率低，因此需要將長文件切分成多個片段 ￼。可使用 LangChain 提供的文字切分器（如RecursiveCharacterTextSplitter）來將文件切成適當大小的chunk（例如每個chunk約5001000個字元），並可設定一定程度的重疊（overlap，例如50100字元）以保留段落銜接的上下文。切分後，每個文件chunk會作為獨立的語意單位，方便後續向量檢索。這步驟能避免段落過長無法放入LLM上下文窗口的問題 ￼。

3. 向量嵌入（Embedding）：接下來，對每個文件chunk計算其向量表示。選擇一個適當的文字嵌入模型（embedding model）將文本轉換為高維度向量。常用做法是使用OpenAI提供的文字嵌入模型（例如text-embedding-ada-002），透過OpenAI的SDK介面來取得向量表示。每個chunk會對應到一個向量，高維向量能夠捕捉該段文字的語意。LangChain中可以直接使用OpenAIEmbeddings類別來生成嵌入向量 ￼。例如：
```python
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()  # 使用預設的OpenAI嵌入模型 (text-embedding-ada-002)
```
上述程式會為每段文本產生向量表示，維度通常為1536（以ADA模型為例）。

建立向量資料庫（FAISS）

有了文件chunk及其向量，下一步是建立向量索引資料庫以支援相似度檢索。這裡我們採用FAISS（Facebook AI Similarity Search）作為本地的向量資料庫。FAISS能在記憶體中高效地執行向量相似度搜尋，無需外部服務，適合快速開發原型。

1. 建立FAISS索引：透過LangChain的向量庫介面，可以很方便地將嵌入向量存入FAISS。使用FAISS.from_documents方法，傳入文件片段列表和對應的嵌入模型，即可自動建構出FAISS索引 ￼。例如：
```python
from langchain.vectorstores import FAISS
```
# docs 為前述切分後的 Document 清單
vector_store = FAISS.from_documents(docs, embeddings)

上述程式會計算所有docs的向量（若尚未計算）並建立一個FAISS索引，將每個chunk向量存入（同時保存對應的文本內容）。完成後，我們就擁有一個本地的向量資料庫，可用於相似度檢索。

2. （可選）儲存與載入索引：若SOP文件量較大，建索引可能耗時。FAISS支援將索引保存到本地檔案，日後直接載入使用，而不必每次重建。例如：
```python
vector_store.save_local("sop_faiss_index")
# 之後可用 FAISS.load_local 載入
new_store = FAISS.load_local("sop_faiss_index", embeddings)
```
這樣可以將索引建立與查詢解耦，提高系統啟動速度（本範例簡單起見可不特別實作保存）。

檢索與問答整合 (Retriever + LLM Chain)

向量索引建立後，就可以使用它來進行檢索，並串接LLM產生答案。

1. 構建 Retriever：LangChain 向量庫物件提供了as_retriever()方法，可將FAISS索引包裝成Retriever介面。可以設定檢索時參數，如search_type="similarity"以及返回文檔數量k等。例えば：
```python
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
```
此設定代表每次查詢會返回前5個與問題最相關的文本片段。檢索器會計算使用者問題的向量，與資料庫中所有chunk向量計算相似度，找出分數最高的幾個。

2. 建立問答鏈（QA Chain）：LangChain提供了高階鏈（Chain）來整合檢索與LLM。最簡單的是使用RetrievalQA鏈：將LLM模型和上面建立的retriever傳入即可。 ￼此鏈在執行時會自動完成「檢索→組合提示→LLM產生回答」的過程。我們可以使用公司內部的LLM接口（OpenAI SDK相容）來初始化一個LLM對象。假設公司的模型以OpenAI API方式提供，可以使用ChatOpenAI類別（搭配對應的API金鑰與參數）來呼叫。例如：
```python
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model="gpt-4", temperature=0)  # 可換成內部模型名稱
qa_chain = RetrievalQA(llm=llm, retriever=retriever)
```
上述qa_chain即為一個結合檢索器與LLM的問答鏈。當呼叫qa_chain.run(<使用者問題>)時，鏈會先用Retriever從FAISS中找出相關的SOP段落，再將問題+段落內容一起組裝提示給LLM，讓LLM產生包含根據文件資訊的答案 ￼。這實現了RAG流程中「檢索相關知識後生成回答」的自動化 ￼。開發者也可自行調整提示模板，要求模型在回答中引用文件內容或來源，但基本原理相同。

範例程式碼樣板

下面提供一個完整的簡易RAG系統程式碼範例（使用LangChain + FAISS），包含文件載入、建立向量索引，以及QA查詢整合。請依需求安裝相應套件並替換API金鑰與文件路徑：

# 安裝必要套件： langchain, openai, faiss-cpu, PyPDF2 (如需PDF載入)


```python
!pip install langchain openai faiss-cpu PyPDF2

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# 設定OpenAI API金鑰（若使用公司內部接口，需確保環境變數或OpenAI SDK指向正確的endpoint）
os.environ["OPENAI_API_KEY"] = "你的-openai-api-key"

# 步驟1: 載入SOP文件並轉成文字文件 (此處以 PDF 為例)
loader = PyPDFLoader("你的_SOP文件.pdf")            # 將路徑替換為你的SOP文件
documents = loader.load()                          # 讀取PDF內容為 Document 物件列表

# 步驟2: 將長文檔切分為較小的 chunk
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)    # 切分後的 Document 清單

# 步驟3: 為每個 chunk 計算向量嵌入並建立 FAISS 向量索引
embeddings = OpenAIEmbeddings()                    # 使用 OpenAI 的文字向量嵌入模型
vector_store = FAISS.from_documents(docs, embeddings)

# 步驟4: 構建檢索器與QA鏈
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
llm = ChatOpenAI(model="gpt-4", temperature=0)     # 使用公司內部的 LLM（OpenAI SDK 介面），此處假設為GPT-4
qa_chain = RetrievalQA(llm=llm, retriever=retriever)

# 步驟5: 提供介面進行問答 (簡易示例)
query = "客戶要退款怎麼處理？"                    # 使用者的自然語言問題
result = qa_chain.run(query)
print("Q:", query)
print("A:", result)
```


執行上述程式後，系統會從預先建立的SOP向量資料庫中檢索與問題相關的內容段落，並由內部的LLM生成答案。例如，對於問題「客戶要退款怎麼處理？」，AI Agent可能會在SOP文件中找到退款流程的相關段落，然後給出一步一步的處理說明。整套流程快速整合了檢索與生成： ￼文本片段經OpenAI嵌入產生向量並存入FAISS，查詢時檢索相關片段後由LLM根據片段內容產生回答 ￼。這樣的RAG架構實作簡潔且有效，工程師可以在此基礎上擴展功能，例如增加更多文件、調整檢索參數或自訂回答格式等。

來源: 參考了LangChain官方文件與範例程式碼 ￼ ￼ ￼ ￼，以上內容說明了如何使用LangChain結合FAISS及OpenAI接口，快速實作一個能查詢SOP文件並回答問題的RAG系統。


很好，你的問題是核心問題之一：「如何將圖片格式的SOP轉成能被RAG使用的可檢索文字格式」。

⸻

✅ 目標：將 SOP 流程圖（圖像）→轉成一組「可 chunk 的結構化文字」，讓 RAG 系統能檢索與回答。

⸻

🧩 解法流程（你只要做一次準備）：

1. 將圖轉文字（手動 or OCR）

以這張圖為例，你可以手動或用 OCR 工具（如 Tesseract）將內容轉成文字。

⸻

2. 轉成「條列式」或「規則式」文字描述

以下是推薦的格式來寫，這樣最適合給向量化、切 chunk、檢索。

⸻

✅ 建議文字格式（可直接丟給 LangChain）

Customer Inquiries SOP:

1. 問題判斷：
   - 客戶是否對產品有抱怨？
     - 若是，請繼續第 2 步。
     - 若否，請找出來電原因並使用通訊錄將客戶轉接到正確負責人。

2. 若有抱怨，詢問：
   a. 客戶是否希望更換產品？
      - 若是：直接轉接銷售部。
      - 若否：紀錄客訴並使用團隊 App 傳給銷售部。

   b. 客戶是否希望退費？
      - 若是：直接轉接財務部。
      - 若否：紀錄客訴並使用團隊 App 傳給銷售部。


⸻

🧠 原因與原理：
	•	✅ 條列式比自然段落更適合向量檢索（因為語意清楚、邏輯明確）
	•	✅ 清楚提問條件（如是否退費）會讓 retriever 更容易 match 用戶 query（例如：「客戶抱怨但不要退費怎麼辦？」）
	•	✅ 每一段可以被 chunk 切分出來（如每一個 if-else 流程），再轉成嵌入向量 → 放進 FAISS。

⸻

📦 可選進階格式（轉成 Q&A）

也可以補充 Q&A 文件來增強回答品質（搭配 Retriever）

Q: 客戶來電表示產品有問題，但他不要換貨也不要退費，怎麼辦？
A: 請記錄客訴並透過團隊App傳送給銷售部。

Q: 客戶來電抱怨，想要退款，該轉給誰？
A: 請直接轉接給財務部門（Accounts team）。

這種格式適合加進 RAG 的資料庫中，作為「語意強化」的 chunk。

⸻

🛠️ 怎麼做（步驟摘要）
	1.	手動或OCR把圖片轉成文字
	2.	用條列方式描述 SOP 的邏輯流程
	3.	存成 .txt 檔或其他可讀格式
	4.	用 LangChain 的 DocumentLoader 載入
	5.	用 TextSplitter 切段
	6.	產生嵌入並建 FAISS 向量庫

⸻

🧪 範例練習資料（你可以複製下面這段當第一個測試）

【Customer Inquiries SOP】

1. 問題判斷：
   - 是否有產品抱怨？
     - 是 → 問是否要換貨或退費。
     - 否 → 查明來電原因 → 使用通訊錄將客戶轉到正確部門。

2. 如果要換貨：
   - 是 → 轉接銷售部。
   - 否 → 記錄並傳送給銷售部（用團隊App）。

3. 如果要退費：
   - 是 → 轉接財務部。
   - 否 → 記錄並傳送給銷售部（用團隊App）。


⸻

需要我幫你用 OCR 工具提取其他圖片的話，也可以幫忙處理。
有了這個文字後，你就可以走 LangChain 的 standard RAG 流程了。要不要我也幫你生一份 .txt 版可以直接載入的文件？

很好，你的問題是核心問題之一：「如何將圖片格式的SOP轉成能被RAG使用的可檢索文字格式」。

⸻

✅ 目標：將 SOP 流程圖（圖像）→轉成一組「可 chunk 的結構化文字」，讓 RAG 系統能檢索與回答。

⸻

🧩 解法流程（你只要做一次準備）：

1. 將圖轉文字（手動 or OCR）

以這張圖為例，你可以手動或用 OCR 工具（如 Tesseract）將內容轉成文字。

⸻

2. 轉成「條列式」或「規則式」文字描述

以下是推薦的格式來寫，這樣最適合給向量化、切 chunk、檢索。

⸻

✅ 建議文字格式（可直接丟給 LangChain）

Customer Inquiries SOP:

1. 問題判斷：
   - 客戶是否對產品有抱怨？
     - 若是，請繼續第 2 步。
     - 若否，請找出來電原因並使用通訊錄將客戶轉接到正確負責人。

2. 若有抱怨，詢問：
   a. 客戶是否希望更換產品？
      - 若是：直接轉接銷售部。
      - 若否：紀錄客訴並使用團隊 App 傳給銷售部。

   b. 客戶是否希望退費？
      - 若是：直接轉接財務部。
      - 若否：紀錄客訴並使用團隊 App 傳給銷售部。


⸻

🧠 原因與原理：
	•	✅ 條列式比自然段落更適合向量檢索（因為語意清楚、邏輯明確）
	•	✅ 清楚提問條件（如是否退費）會讓 retriever 更容易 match 用戶 query（例如：「客戶抱怨但不要退費怎麼辦？」）
	•	✅ 每一段可以被 chunk 切分出來（如每一個 if-else 流程），再轉成嵌入向量 → 放進 FAISS。

⸻

📦 可選進階格式（轉成 Q&A）

也可以補充 Q&A 文件來增強回答品質（搭配 Retriever）

Q: 客戶來電表示產品有問題，但他不要換貨也不要退費，怎麼辦？
A: 請記錄客訴並透過團隊App傳送給銷售部。

Q: 客戶來電抱怨，想要退款，該轉給誰？
A: 請直接轉接給財務部門（Accounts team）。

這種格式適合加進 RAG 的資料庫中，作為「語意強化」的 chunk。

⸻

🛠️ 怎麼做（步驟摘要）
	1.	手動或OCR把圖片轉成文字
	2.	用條列方式描述 SOP 的邏輯流程
	3.	存成 .txt 檔或其他可讀格式
	4.	用 LangChain 的 DocumentLoader 載入
	5.	用 TextSplitter 切段
	6.	產生嵌入並建 FAISS 向量庫

⸻

🧪 範例練習資料（你可以複製下面這段當第一個測試）
```
【Customer Inquiries SOP】

1. 問題判斷：
   - 是否有產品抱怨？
     - 是 → 問是否要換貨或退費。
     - 否 → 查明來電原因 → 使用通訊錄將客戶轉到正確部門。

2. 如果要換貨：
   - 是 → 轉接銷售部。
   - 否 → 記錄並傳送給銷售部（用團隊App）。

3. 如果要退費：
   - 是 → 轉接財務部。
   - 否 → 記錄並傳送給銷售部（用團隊App）。

```
⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻
# STL

核心原理先講清楚
	1.	異質分段 (Regime) 先處理
	•	你的數列明顯有三段不同均值／變異度。
	•	若不先分段就直接算整體平均 → 高低段會互相稀釋，任何閾值都會偏離真相。
	•	可用已知分段；若實務上分段未知，可用 PELT、Binary Segmentation 或 Bayesian Online Change-Point Detection 找轉折點。
	2.	在「同質」段內定義 Outlier
	•	一段數據可以近似「固定位置 + 噪音」。
	•	Outlier 就是「噪音」遠超過典型範圍的點。必要要件：
	1.	位置統計量：平均或中位數。
	2.	離散度統計量：標準差、IQR、MAD (對極端值最不敏感)。
	•	通常選用 MAD（Median Absolute Deviation）再轉成 Robust Z-Score：
\[
z_i^\* = \frac{\,x_i - \text{median}\,}{1.4826 \times \text{MAD}}
\]
只要 \(|z_i^\*| > 3\)（或業務上更嚴格的 2.5、3.5）即可標記為異常。
	3.	簡明步驟
```python
import numpy as np

def robust_outliers(segment, thresh=3):
    med = np.median(segment)
    mad = np.median(np.abs(segment - med))
    z = 1.4826 * (segment - med) / (mad if mad != 0 else 1e-9)
    return np.where(np.abs(z) > thresh)[0]  # 回傳 index
```
	•	對三段各跑一次 robust_outliers；回傳的索引就是異常點在該段中的位置。
	•	若段長很短，MAD 可能為 0，需加上微小值避免除零。

	4.	其他同樣好用（視場景挑選）

思路	方法	適合情況
距離基準	IQR (箱形圖)	小樣本、分佈對稱
機率模型	ARIMA／Prophet 殘差檢測	有季節性或趨勢
樹模型	Isolation Forest	多維特徵、數列不須強序性
密度模型	LOF／DBSCAN	有聚簇結構、多維資料
統計檢定	GESD / Grubbs	完全假設常態、樣本 > 25


	5.	實務建議
	•	先用最簡單 MAD → 人眼核對 → 再決定要不要換更複雜模型。
	•	閾值不要死守 3；可用業務容忍度或 K-sigma 法調整。
	•	若資料量大或分段很多，可把上述流程封裝成函式，批次跑完再人工複核。

這樣就能在三段各自判定並標記異常點，避免高、中、低段互相干擾。

整套「先分段再找異常」的全自動流程

適用情境：序列中段落（regime）數量未知，且各段平均值/變異度差異明顯。
思路：① 用變點偵測（Change-Point Detection）先切段 → ② 於每一段內跑魯棒異常值檢測。

⸻

1. 變點偵測：自動把序列切成同質段

類型	推薦演算法	長處	何時用
離線（一次性）	PELT、Binary Seg、Bottom-Up（皆含於 ruptures）	O(n) ~ O(n log n)，速度快，可設定懲罰自動決定段數	批次資料、長度上萬也 OK
線上（串流）	Bayesian Online Change-Point Detection（bocd 套件）	到點即判斷、遞迴更新，適合即時監控	需要邊收邊判斷時

示範（ruptures + PELT）：  ￼
```python
import numpy as np, ruptures as rpt

ts = np.array([...])               # 你的原始序列
model = "rbf"                      # 均值或變異同時改變時常用
algo  = rpt.Pelt(model=model).fit(ts)

# pen = β 控制段數；可用 BIC ≈ 3*log(n) 或手動調
break_pts = algo.predict(pen=3*np.log(len(ts)))
# e.g., [8, 14, 21] 代表 0–7, 8–13, 14–20 為三段
```
若需線上： pip install bocd → 參考範例即可  ￼。

調 β（penalty）的簡易做法
	•	從大到小掃描 β，畫「段數 vs β」折線；折線明顯拐點通常是合理段數（elbow method）。
	•	或先估一個最大可接受段數 max_k，直接 algo.predict(k=max_k) 然後人工調低。

⸻

2. 段內異常值：用 MAD Robust Z-Score
```python
def robust_z_outliers(arr, thresh=3):
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    z = 1.4826 * (arr - med) / (mad or 1e-9)
    return np.where(np.abs(z) > thresh)[0]          # 回傳在段內的索引

segments = np.split(ts, break_pts[:-1])             # 切段
global_outliers = []
start = 0
for seg in segments:
    idx_local = robust_z_outliers(seg)
    global_outliers += list(start + idx_local)      # 轉成全域索引
    start += len(seg)
print("Outlier indices:", global_outliers)

為何選 MAD？ 對極端值最不敏感；小樣本也穩定。
必要時改用 IQR、Isolation Forest、或針對季節性殘差做檢測。
```
⸻

3. 還有兩個常見加強點
	1.	迭代式修正
	•	先切段 → 段內去掉異常 → 重跑變點偵測（可去除「單點極端值造成假變點」的誤判）。
	2.	聯合模型
	•	直接建「階梯均值 + 稀疏異常項」的貝式模型（如 R-BEAST 或 Bayesian piecewise regression），一次輸出段落及 outlier  ￼。
	•	好處：變點與異常互不干擾，但計算較重。

⸻

一頁結論
	1.	先用 ruptures.Pelt 自動切段（β ≈ 3 log n 起手）。
	2.	每段跑 MAD-Z > 3 判定 outlier。
	3.	如有串流需求，改用 bocd 線上偵測。
	4.	段數、β、Z 閾值都可依業務風險做微調；工具選擇簡：ruptures → bocd → 進階貝式。

照此流程，程式即可 自動決定段落 + 段內異常值，不用事先知道到底有幾段。祝開發順利!
