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