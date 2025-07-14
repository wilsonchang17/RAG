Implementing a RAG Tool for SOP Retrieval (Local Setup)

To build a Retrieval-Augmented Generation (RAG) system in your local OctoTools-based agent, you can follow a lightweight approach using a vector search (e.g. FAISS) for document retrieval. The idea is to embed your SOP text files into vector representations, then search for relevant SOP snippets when a question is asked. This way, the LLM can be provided with the relevant SOP content to generate accurate answers. Below are suggestions and an example implementation:

Overview of the RAG Approach
	•	Document Embeddings: Convert each SOP text (or text chunk) into a numerical vector using an embedding model. For a local setup, you can use an open-source model (e.g. a SentenceTransformer) that supports both Chinese and English (since your SOP content is bilingual). This step is done offline or during initialization.
	•	Vector Store (FAISS): Store all document vectors in a FAISS index for fast similarity search. FAISS is lightweight and runs locally.
	•	Retrieval Step: When a user asks a question (e.g. “溫度有沒有異常? 有的話該怎麼解決?”), the agent (after detecting an anomaly via your tools) will form a query (possibly the user’s question or a refined query with anomaly context) and embed this query with the same embedding model. Then use FAISS to find the top-K most similar SOP chunks.
	•	Augment LLM Prompt: Take the retrieved SOP text (e.g. the steps and recommendations from the relevant SOP) and include it in the LLM’s prompt (or provide it to a tool that generates the answer). For example: “根據SOP資料，Chamber溫度漂移的處理步驟如下: … Given this, here’s how to resolve the issue…”.
	•	Local LLM Generation: Finally, use your LLM (via the Generalist_Solution_Generator_Tool or directly) to generate the answer, ensuring it references the SOP instructions or recommendations found.

This approach ensures the LLM’s answer is grounded in the actual SOP documents, reducing hallucination and providing accurate solutions.

Steps to Implement the RAG Pipeline
	1.	Prepare SOP Documents: Organize your SOP text files (for different anomaly types) in a folder. Each file can be chunked into smaller passages (e.g. by paragraph or a fixed token length) if they are long, to improve retrieval granularity.
	2.	Choose an Embedding Model: Use a lightweight embedding model that can run locally. For example, a Sentence-Transformer model like "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" works for both English/Chinese text. You can install sentence-transformers or use HuggingFace Transformers to load such a model.
	3.	Embed the Documents: Iterate over each SOP file (or each chunk) and compute its embedding. Store all embeddings in an array (and keep references to which file/chunk they belong to).
	4.	Build FAISS Index: Use FAISS to create an index from the embeddings. (For a small number of documents, you could even do a simple brute-force similarity search without FAISS, but FAISS is efficient and easy to use).
	5.	Implement a Retrieval Tool: Integrate this into your agent as a tool (e.g. SOPRetrievalTool) that, given a query, will embed the query and retrieve the most relevant SOP text. The tool’s output can be the text of the top result (or multiple results) which the agent can then use to answer the question.
	6.	Agent Integration: Update the agent’s planner/solver logic to use the retrieval tool after anomaly detection. For example: if the anomaly analysis tool finds a temperature drift, the agent can call SOPRetrievalTool with a query like “温度 漂移 SOP 解決方法” (or simply use the user’s question) to get the relevant SOP content. Then feed this content into the Generalist_Solution_Generator_Tool (LLM) to compose the final answer with step-by-step solutions from the SOP.

Example Implementation


## 用sentence-transfoemers的方風
Below is a simplified example of how you might implement the SOP retrieval using FAISS and an embedding model. This code can be adapted into an OctoTools BaseTool subclass for integration into your agent:
```python
!pip install faiss-cpu sentence-transformers   # install FAISS and embedding model package

import faiss
from sentence_transformers import SentenceTransformer

# 1. Load embedding model (multilingual model for Chinese/English)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 2. Read and prepare SOP documents
import os
sop_folder = "path/to/sop_texts"
documents = []       # store text of each chunk
doc_embeddings = []  # store embedding vectors

for filename in os.listdir(sop_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(sop_folder, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        # Optionally split long text into chunks
        # Here we split by double newline as an example (paragraphs)
        chunks = text.split("\n\n")
        for chunk in chunks:
            if chunk.strip():
                documents.append(chunk)
                emb = model.encode(chunk)  # get embedding vector (numpy array)
                doc_embeddings.append(emb)

# Convert embeddings list to array
import numpy as np
emb_array = np.vstack(doc_embeddings).astype('float32')

# 3. Build FAISS index
dimension = emb_array.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance index (could also use cosine similarity)
index.add(emb_array)
print(f"Indexed {index.ntotal} SOP chunks.")
```
At this point, you have a FAISS index of your SOP content. Now, implement a function (or tool) to query this index:
```python
def retrieve_sop(query, top_k=2):
    """Return top_k relevant SOP text chunks for the given query."""
    # Embed the query using the same model
    q_vec = model.encode(query).astype('float32')
    q_vec = np.expand_dims(q_vec, axis=0)  # shape (1, dim)
    # Search FAISS index
    distances, indices = index.search(q_vec, top_k)
    results = []
    for idx in indices[0]:
        sop_text = documents[idx]
        results.append(sop_text)
    return results

# Example usage:
query = "Chamber 溫度異常 解決"  # example query in Chinese/English mix
top_chunks = retrieve_sop(query, top_k=1)
print("Top retrieved SOP content:\n", top_chunks[0])

This function will return the most relevant SOP chunk(s) for a given query. You can integrate this logic into an OctoTools BaseTool as follows:

from octotools.tools.base import BaseTool

class SOPRetrievalTool(BaseTool):
    def __init__(self, sop_folder="path/to/sop_texts"):
        super().__init__(
            tool_name="SOP_Retrieval_Tool",
            tool_description="Retrieves relevant SOP instructions given a query about an anomaly or issue.",
            tool_version="1.0.0",
            input_types={"query": "str - User question or anomaly description to search SOP knowledge base."},
            output_type="str - Relevant SOP text or instructions related to the query.",
        )
        # Load embedding model
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        # Build the knowledge index
        self.documents = []
        doc_embeddings = []
        for filename in os.listdir(sop_folder):
            if filename.endswith(".txt"):
                text = open(os.path.join(sop_folder, filename), 'r', encoding='utf-8').read().strip()
                chunks = text.split("\n\n")
                for chunk in chunks:
                    if chunk.strip():
                        self.documents.append(chunk)
                        doc_embeddings.append(self.model.encode(chunk))
        emb_array = np.vstack(doc_embeddings).astype('float32')
        dim = emb_array.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(emb_array)

    def execute(self, query):
        # Embed query and search
        q_vec = self.model.encode(query).astype('float32').reshape(1, -1)
        distances, indices = self.index.search(q_vec, 1)  # get the top 1 result
        if indices.size > 0:
            best_idx = indices[0][0]
            return self.documents[best_idx]
        else:
            return "No relevant SOP found."
```
This SOPRetrievalTool can be added to your agent’s toolbox. When executed with a query (such as a description of the anomaly), it returns the most relevant SOP snippet. For example, if the temperature drift anomaly is detected, the agent might call:

```python
result = sop_tool.execute(query="腔體溫度飄移 SOP 解決方案")
```
And get back a text like “1. 問題判斷：Chamber 溫度是否超出製程設定容忍值（±3°C）？是 → 進入緊急處理流程。…” (from your example SOP).

Integrating Retrieval with the Agent’s Workflow
	•	After Anomaly Detection: Once your anomaly analysis tool identifies an outlier (e.g., temperature drift), construct a query for the retrieval tool. You might use the anomaly name or the user’s original question. In this case, if the user asks “溫度有沒有異常資料? 有的話該怎麼解決”, the agent detects an anomaly in temperature and then queries the knowledge base for “temperature drift SOP solution”.
	•	Use Retrieved Text in Answer Generation: Take the output of SOPRetrievalTool and include it in the prompt for the LLM. For instance, you can feed a prompt like: “根據以下SOP內容，回答使用者的問題：\n{retrieved_SOP_text}\n\n使用者問題: 温度有異常，要如何處理?”. The Generalist_Solution_Generator_Tool can then produce a step-by-step answer citing the SOP and recommendations.
	•	Resource Considerations: This approach is lightweight. The main overhead is computing embeddings for your SOP docs (done once at startup). FAISS lookups are fast even on CPU. Ensure your embedding model is small enough for local use (the suggested MiniLM model is only ~120MB and quite fast). Since your LLM is local too, you avoid external API calls completely.
	•	Maintenance: If SOP documents update or new ones are added, you’ll need to re-run the embedding and index-building step. This can be automated on startup or whenever files change. For a small number of documents, this is very quick.

Additional Suggestions & Considerations
	•	Chunking Strategy: Store each numbered step or paragraph as separate chunks in the knowledge base. This yields more precise search results (rather than retrieving a whole long document when only a part is relevant).
	•	Multilingual Support: Your SOP content contains Chinese and technical English terms. The chosen embedding model should handle this. The multilingual MiniLM works well for Chinese; alternatively, you could use a Chinese-specific model if needed, but mixing languages in one model is convenient here.
	•	Answer Composition: When the LLM generates the final answer, it might be useful to reference the SOP (e.g., “根據《Etching Chamber Temperature Drift SOP》, 建議採取以下措施…”). This increases user trust that the answer is from an official procedure. You can include the SOP title in the retrieved text or as metadata.
	•	Testing: Test the retrieval by varying query phrasing to ensure the correct SOP is found (e.g., “溫度飄移”, “溫控異常”, “He 背壓 溫度 升溫”). If needed, you can add some keyword-based filtering or synonyms to improve recall. However, a good embedding model should capture the semantic similarity.
	•	No Fancy Frameworks Needed: This setup avoids complex frameworks – it uses direct embedding computation and FAISS for similarity search. It fits well with your minimal-resource requirement and can be integrated into the OctoTools agent by treating it as just another tool or a pre-processing step before calling the LLM.

By implementing the above, your agent will be able to retrieve the correct SOP and solution steps for a detected anomaly and present them to the engineer, all running locally. This ensures that when the engineer provides FDC abnormal data, the agent responds with a solid SOP-based solution. Good luck with your RAG tool implementation!

## 用sentence-transfoemers的方風
```python
# file: octotools/tools/sop_retrieval_tool/tool.py
import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from octotools.tools.base import BaseTool

class SOP_Retrieval_Tool(BaseTool):
    """
    A lightweight Retrieval-Augmented tool for fetching the most relevant SOP snippet
    (from local txt files) given an engineer's anomaly query.
    """

    require_llm_engine = False   # 只做檢索，不直接呼叫 LLM

    def __init__(
        self,
        sop_folder: str = "data/sop_texts",
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        top_k: int = 1,
    ):
        """
        Args:
            sop_folder: 本地存放多個 .txt SOP 檔案的資料夾。
            model_name: Sentence-Transformer embedding 模型名稱。
            top_k: 每次查詢要回傳前 k 個最相關片段。
        """
        super().__init__(
            tool_name="SOP_Retrieval_Tool",
            tool_description=(
                "Retrieves the most relevant SOP paragraph(s) for a given anomaly-related "
                "query using local FAISS vector search."
            ),
            tool_version="1.0.0",
            input_types={
                "query": "str - The anomaly description or user question to search for.",
            },
            output_type="str - The retrieved SOP paragraph(s).",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="腔體溫度飄移 如何處理")',
                    "description": "Retrieve SOP steps for chamber temperature drift."
                },
                {
                    "command": 'execution = tool.execute(query="MFC 流量異常 SOP")',
                    "description": "Retrieve SOP for MFC flow anomaly."
                },
            ],
            user_metadata={
                "limitation": (
                    "本工具僅依照向量相似度取前 k 個片段，若 SOP 文檔極度冗長或關鍵字差異很大，"
                    "可能需要調整 chunking 或加入同義詞。"
                ),
                "best_practice": (
                    "1) 保持 SOP txt 檔案最新；2) SOP 文字建議依段落或條列分段，提升檢索精度；"
                    "3) 若新增/修改 SOP 檔案，重新載入工具或呼叫 rebuild_index(True)。"
                )
            },
        )

        self.sop_folder = sop_folder
        self.model_name = model_name
        self.top_k = top_k

        # 延遲載入：第一次真正查詢時才建 index，可縮短 agent 啟動時間
        self._index_built = False
        self._index = None
        self._doc_texts = []
        self._emb_model = None

    # ------------------------------------------------------------------
    # Public API —— agent 只會呼叫 execute()
    # ------------------------------------------------------------------
    def execute(self, query: str):
        """
        Return top-k relevant SOP text chunks for the given query.
        """
        if not query or not isinstance(query, str):
            return "Error: `query` must be a non-empty string."

        # 如有需要，先建索引（僅一次）
        if not self._index_built:
            ok, msg = self._build_index()
            if not ok:
                return f"Error building SOP index: {msg}"

        # 1. 對 query 做 embedding
        q_vec = self._emb_model.encode(query).astype("float32").reshape(1, -1)

        # 2. 在 FAISS 中搜尋
        try:
            distances, indices = self._index.search(q_vec, self.top_k)
        except Exception as e:
            return f"Error while searching FAISS index: {str(e)}"

        # 3. 組回覆
        retrieved = []
        for idx in indices[0]:
            if idx < len(self._doc_texts):
                retrieved.append(self._doc_texts[idx])

        if not retrieved:
            return "No relevant SOP found."

        # 用 --- 分隔多段，回傳給上層 LLM prompt 或直接展出
        return "\n---\n".join(retrieved)

    # ------------------------------------------------------------------
    # Helper: build / rebuild index (可在代碼內部或手動呼叫)
    # ------------------------------------------------------------------
    def _build_index(self):
        """
        Load all txt files, split into chunks, embed, and build FAISS index.
        Returns (ok: bool, message: str)
        """
        if not os.path.isdir(self.sop_folder):
            return False, f"SOP folder not found: {self.sop_folder}"

        # 1) 讀檔 & chunk
        texts = []
        for fname in os.listdir(self.sop_folder):
            if fname.endswith(".txt"):
                path = os.path.join(self.sop_folder, fname)
                with open(path, "r", encoding="utf-8") as f:
                    txt = f.read().strip()
                # 以兩個以上換行或 '【' 標題切段
                chunks = [c.strip() for c in txt.split("\n\n") if c.strip()]
                texts.extend(chunks)

        if not texts:
            return False, "No SOP txt files or chunks found."

        # 2) 產生 embedding
        try:
            self._emb_model = SentenceTransformer(self.model_name)
            embs = self._emb_model.encode(texts, batch_size=32, show_progress_bar=False)
            embs = np.asarray(embs, dtype="float32")
        except Exception as e:
            return False, f"Embedding failed: {str(e)}"

        # 3) 建 FAISS index
        try:
            dim = embs.shape[1]
            index = faiss.IndexFlatL2(dim)  # L2 distance; 可改 cosine
            index.add(embs)
        except Exception as e:
            return False, f"Building FAISS index failed: {str(e)}"

        # 4) 設定內部狀態
        self._doc_texts = texts
        self._index = index
        self._index_built = True
        return True, "Index built successfully."

    # ------------------------------------------------------------------
    # 方便測試：暴露 metadata
    # ------------------------------------------------------------------
    def get_metadata(self):
        metadata = super().get_metadata()
        metadata["sop_folder"] = self.sop_folder
        metadata["index_built"] = self._index_built
        metadata["num_chunks"] = len(self._doc_texts)
        return metadata

# ----------------------------------------------------------------------
# CLI / 獨立測試區段
# ----------------------------------------------------------------------
if __name__ == "__main__":
    """
    Test command:

    cd octotools/tools/sop_retrieval_tool
    python tool.py
    """

    # 取得當前腳本路徑
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {script_dir}")

    # 假設 SOP txt 檔放在 ./examples/sops/
    example_sop_dir = os.path.join(script_dir, "examples", "sops")

    tool = SOP_Retrieval_Tool(sop_folder=example_sop_dir, top_k=2)

    # 查看 metadata
    metadata = tool.get_metadata()
    print(json.dumps(metadata, indent=2, ensure_ascii=False))

    # 執行查詢
    try:
        execution = tool.execute(query="腔體溫度飄移 SOP 解決方案")
        print("Retrieved SOP Snippet(s):")
        print(execution)
    except Exception as e:
        print(f"Execution failed: {e}")

    print("Done!")
```

## 用TF-IDF / BM25
```python
# file: octotools/tools/sop_retrieval_tool/tool.py
"""
SOP_Retrieval_Tool  —  TF-IDF / BM25 版本

* 依賴：
    pip install scikit-learn faiss-cpu           # 如果要 BM25 再多裝 rank_bm25

此版本移除 sentence-transformers，改用 **文字統計向量**：
1. 預設採用 TF-IDF + 內積（對 L2-norm 後的向量即為 cosine 相似度）。
2. 若將 `use_bm25=True`，則改用 rank_bm25 套件計算 BM25 分數，不需要 FAISS。（BM25 適合小語料，直接線性掃描即可）

適用情境：公司環境無 CUDA / 不想下載大模型，但 SOP 數量不多。
"""

import os
import json
from typing import List

import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from rank_bm25 import BM25Okapi  # optional
except ImportError:
    BM25Okapi = None

from octotools.tools.base import BaseTool


class SOP_Retrieval_Tool(BaseTool):
    """Lightweight SOP 檢索：TF-IDF -> FAISS  or  BM25 (純 Python)"""

    require_llm_engine = False

    def __init__(
        self,
        sop_folder: str = "data/sop_texts",
        top_k: int = 1,
        use_bm25: bool = False,
    ):
        super().__init__(
            tool_name="SOP_Retrieval_Tool",
            tool_description=(
                "Retrieve relevant SOP paragraph(s) for an anomaly query using TF-IDF/"
                "BM25. Default TF-IDF + FAISS; set use_bm25=True for BM25 scan."
            ),
            tool_version="2.0.0",
            input_types={
                "query": "str - Anomaly description or user question to search.",
            },
            output_type="str - Retrieved SOP paragraph(s).",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="腔體溫度飄移 如何處理")',
                    "description": "Retrieve SOP for chamber temperature drift."
                },
                {
                    "command": 'execution = tool.execute(query="MFC 流量異常 SOP")',
                    "description": "Retrieve SOP for MFC flow anomaly."
                },
            ],
            user_metadata={
                "limitation": (
                    "基於關鍵字的向量，對同義詞/語序變化較敏感；若 SOP 過長或語料大，可考慮升級為 Embedding 版本。"
                ),
                "best_practice": (
                    "1) SOP 文字以段落切分；2) 新增或修改檔案後重建索引；"
                    "3) 若語意準確度不足，再換成 SentenceTransformer/BERT 版本。"
                ),
            },
        )

        self.sop_folder = sop_folder
        self.top_k = max(1, top_k)
        self.use_bm25 = use_bm25 and (BM25Okapi is not None)

        # 延遲建索引
        self._index_built = False
        self._doc_texts: List[str] = []
        self._vectorizer = None
        self._tfidf_mat = None  # sparse matrix
        self._faiss_index = None  # 只有 TF-IDF 模式會用到
        self._bm25 = None        # 只有 BM25 模式會用到

    # ----------------------------------------------------------------------
    def execute(self, query: str):
        if not query or not isinstance(query, str):
            return "Error: `query` must be a non-empty string."

        if not self._index_built:
            ok, msg = self._build_index()
            if not ok:
                return f"Error building SOP index: {msg}"

        if self.use_bm25:
            return self._bm25_search(query)
        else:
            return self._tfidf_faiss_search(query)

    # ----------------------------------------------------------------------
    # TF-IDF + FAISS 路徑
    # ----------------------------------------------------------------------
    def _tfidf_faiss_search(self, query: str):
        # 將 query 轉 sparse → dense → normalize → 搜
        q_vec = self._vectorizer.transform([query]).toarray().astype("float32")
        if q_vec.sum() == 0:
            return "No relevant SOP found."  # 全新字沒被向量化
        # L2 normalize -> cosine similarity = dot product
        faiss.normalize_L2(q_vec)
        distances, indices = self._faiss_index.search(q_vec, self.top_k)
        retrieved = [self._doc_texts[i] for i in indices[0] if i < len(self._doc_texts)]
        return "\n---\n".join(retrieved) if retrieved else "No relevant SOP found."

    # ----------------------------------------------------------------------
    # BM25 路徑
    # ----------------------------------------------------------------------
    def _bm25_search(self, query: str):
        tokens = query.split()  # 中文環境可接 jieba.lcut
        scores = self._bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][: self.top_k]
        retrieved = [self._doc_texts[i] for i in top_idx if scores[i] > 0]
        return "\n---\n".join(retrieved) if retrieved else "No relevant SOP found."

    # ----------------------------------------------------------------------
    # 建索引 (兩種模式共用) 
    # ----------------------------------------------------------------------
    def _build_index(self):
        if not os.path.isdir(self.sop_folder):
            return False, f"SOP folder not found: {self.sop_folder}"

        # 讀 txt + chunk
        texts: List[str] = []
        for fname in os.listdir(self.sop_folder):
            if fname.endswith(".txt"):
                with open(os.path.join(self.sop_folder, fname), "r", encoding="utf-8") as f:
                    txt = f.read().strip()
                chunks = [c.strip() for c in txt.split("\n\n") if c.strip()]
                texts.extend(chunks)
        if not texts:
            return False, "No SOP txt files or chunks found."

        self._doc_texts = texts

        if self.use_bm25:
            if BM25Okapi is None:
                return False, "rank_bm25 not installed. Run `pip install rank_bm25`."
            tokenized = [t.split() for t in texts]  # 中文可換 jieba 分詞
            self._bm25 = BM25Okapi(tokenized)
        else:
            # TF-IDF vectorizer
            self._vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
            X = self._vectorizer.fit_transform(texts)
            dense = X.toarray().astype("float32")
            faiss.normalize_L2(dense)  # 先正規化，方便用內積算 cosine
            dim = dense.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._faiss_index.add(dense)

        self._index_built = True
        return True, "Index built successfully."

    # ------------------------------------------------------------------
    def get_metadata(self):
        data = super().get_metadata()
        data.update(
            {
                "sop_folder": self.sop_folder,
                "index_built": self._index_built,
                "num_chunks": len(self._doc_texts),
                "mode": "BM25" if self.use_bm25 else "TF-IDF",
            }
        )
        return data


# ----------------------------------------------------------------------
# Stand-alone test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    example_dir = os.path.join(script_dir, "examples", "sops")

    tool = SOP_Retrieval_Tool(sop_folder=example_dir, top_k=2, use_bm25=False)

    # metadata
    print(json.dumps(tool.get_metadata(), indent=2, ensure_ascii=False))

    # query# file: octotools/tools/sop_retrieval_tool/tool.py
"""
SOP_Retrieval_Tool  —  TF-IDF + FAISS 精簡版（完全移除 BM25）

依賴：
    pip install scikit-learn faiss-cpu

核心流程：
1. 以 TF-IDF 向量化每段 SOP 文字（支援中文，可自行替換 tokenizer）。
2. L2 normalize → 使用 `faiss.IndexFlatIP`，內積即 Cosine 相似度。
3. 查詢時回傳前 `top_k` 最相關段落（用 `---` 分隔）。

適用情境：SOP 文件量有限，需在純 CPU、本地環境快速部署 RAG。 
"""

import os
import json
from typing import List

import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

from octotools.tools.base import BaseTool


class SOP_Retrieval_Tool(BaseTool):
    """Lightweight SOP 檢索：TF-IDF → FAISS（無 BM25）。"""

    require_llm_engine = False

    def __init__(
        self,
        sop_folder: str = "data/sop_texts",
        top_k: int = 1,
    ):
        super().__init__(
            tool_name="SOP_Retrieval_Tool",
            tool_description=(
                "Retrieve relevant SOP paragraph(s) for an anomaly query using TF-IDF + FAISS "
                "(cosine similarity)."
            ),
            tool_version="2.1.0",
            input_types={
                "query": "str - Anomaly description or user question to search.",
            },
            output_type="str - Retrieved SOP paragraph(s).",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="腔體溫度飄移 如何處理")',
                    "description": "Retrieve SOP for chamber temperature drift."
                },
                {
                    "command": 'execution = tool.execute(query="MFC 流量異常 SOP")',
                    "description": "Retrieve SOP for MFC flow anomaly."
                },
            ],
            user_metadata={
                "limitation": (
                    "基於關鍵字的向量，對同義詞/語序變化較敏感；若精度不足，可改用嵌入式模型。"
                ),
                "best_practice": (
                    "1) 將 SOP 以段落切分；2) 新增/修改檔案後重建索引；"
                    "3) 查詢可加入關鍵詞，如『SOP』『處理步驟』以提高命中。"
                ),
            },
        )

        self.sop_folder = sop_folder
        self.top_k = max(1, top_k)

        # 延遲建索引
        self._index_built = False
        self._doc_texts: List[str] = []
        self._vectorizer = None
        self._faiss_index = None

    # ----------------------------------------------------------------------
    def execute(self, query: str):
        if not query or not isinstance(query, str):
            return "Error: `query` must be a non-empty string."

        if not self._index_built:
            ok, msg = self._build_index()
            if not ok:
                return f"Error building SOP index: {msg}"

        return self._tfidf_faiss_search(query)

    # ----------------------------------------------------------------------
    # TF-IDF + FAISS 查詢
    # ----------------------------------------------------------------------
    def _tfidf_faiss_search(self, query: str):
        q_vec = self._vectorizer.transform([query]).toarray().astype("float32")
        if q_vec.sum() == 0:
            return "No relevant SOP found."
        faiss.normalize_L2(q_vec)  # cosine
        _, indices = self._faiss_index.search(q_vec, self.top_k)
        retrieved = [self._doc_texts[i] for i in indices[0] if i < len(self._doc_texts)]
        return "\n---\n".join(retrieved) if retrieved else "No relevant SOP found."

    # ----------------------------------------------------------------------
    # 建索引
    # ----------------------------------------------------------------------
    def _build_index(self):
        if not os.path.isdir(self.sop_folder):
            return False, f"SOP folder not found: {self.sop_folder}"

        texts: List[str] = []
        for fname in os.listdir(self.sop_folder):
            if fname.endswith(".txt"):
                with open(os.path.join(self.sop_folder, fname), "r", encoding="utf-8") as f:
                    txt = f.read().strip()
                chunks = [c.strip() for c in txt.split("\n\n") if c.strip()]
                texts.extend(chunks)
        if not texts:
            return False, "No SOP txt files or chunks found."

        self._doc_texts = texts

        # TF-IDF vectorizer
        self._vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        X = self._vectorizer.fit_transform(texts)
        dense = X.toarray().astype("float32")
        faiss.normalize_L2(dense)
        dim = dense.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(dense)

        self._index_built = True
        return True, "Index built successfully."

    # ------------------------------------------------------------------
    def get_metadata(self):
        data = super().get_metadata()
        data.update(
            {
                "sop_folder": self.sop_folder,
                "index_built": self._index_built,
                "num_chunks": len(self._doc_texts),
                "mode": "TF-IDF",
            }
        )
        return data


# ----------------------------------------------------------------------
# Stand-alone test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    example_dir = os.path.join(script_dir, "examples", "sops")

    tool = SOP_Retrieval_Tool(sop_folder=example_dir, top_k=2)

    print(json.dumps(tool.get_metadata(), indent=2, ensure_ascii=False))

    ans = tool.execute(query="腔體溫度飄移 SOP 解決方案")
    print("\nRetrieved:\n", ans)
```
    ans = tool.execute(query="腔體溫度飄移 SOP 解決方案")
    print("\nRetrieved:\n", ans)

```

## 更新
```python
