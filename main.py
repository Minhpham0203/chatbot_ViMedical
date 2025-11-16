import os
import faiss
import numpy as np
import pandas as pd
import torch
import pickle
from pyvi import ViTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
import httpx # Dùng httpx để gọi API Gemini
import gc

# --- 0. Cấu hình & Tải Mô hình/Dữ liệu ---
load_dotenv() 
print("--- BẮT ĐẦU KHỞI ĐỘNG API RAG ---")

# --- 0a. Cấu hình Đường dẫn ---
MODEL_PATH_BI_ENCODER = './models/triplet-finetuned-model-v1/'
MODEL_PATH_CROSS_ENCODER = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
INDEX_FAISS_PATH = './artifacts/medical_index_B.faiss'
INDEX_BM25_PATH = './artifacts/medical_index_B.bm25'
CORPUS_PATH = './corpus_with_id.csv'

# --- 0b. Cấu hình Tham số Truy xuất ---
K_RETRIEVE_HYBRID = 50 # Số lượng ứng cử viên
RRF_K = 60

# --- 0c. Cấu hình LLM (Đã đổi sang Gemini) ---
GEMINI_API_KEY = os.getenv("LLM_API_KEY")
# Sử dụng mô hình Flash, nhanh và rẻ
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

if not GEMINI_API_KEY:
    print("CẢNH BÁO: GEMINI_API_KEY không tìm thấy trong file .env. Bước G sẽ thất bại.")

# --- 0d. Tải Mô hình & Dữ liệu ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"--- Đang chạy trên thiết bị: {device} ---")

print("Đang tải Bi-Encoder (Model B)...")
bi_encoder = SentenceTransformer(MODEL_PATH_BI_ENCODER, device=device)

print("Đang tải Cross-Encoder (Re-Ranker)...")
cross_encoder = CrossEncoder(MODEL_PATH_CROSS_ENCODER, max_length=512, device=device)

print("Đang tải FAISS Index (Dense)...")
index_faiss = faiss.read_index(INDEX_FAISS_PATH)

print("Đang tải BM25 Index (Sparse)...")
with open(INDEX_BM25_PATH, 'rb') as f:
    index_bm25 = pickle.load(f)

print("Đang tải Dữ liệu Corpus...")
corpus_df = pd.read_csv(CORPUS_PATH)
index_to_id_map = {i: doc_id for i, doc_id in enumerate(corpus_df['doc_id'].tolist())}
id_to_text_map = pd.Series(corpus_df.Question.values, index=corpus_df.doc_id).to_dict()

print("--- Tải hoàn tất. API Sẵn sàng. ---")

# --- 0e. Hàm phụ trợ ---
def tokenize_vietnamese(text):
    return ViTokenizer.tokenize(text).split()

# --- 1. Định nghĩa API ---
app = FastAPI(title="Medical RAG API (Hybrid + Re-Ranker + Gemini)")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class ChatResponse(BaseModel):
    answer: str
    retrieved_context: list[str]

# --- 2. Lõi Logic RAG (Pipeline 3 Giai đoạn) ---

async def perform_retrieval(query: str, top_k_final: int) -> list[str]:
    """
    (Hàm này giữ nguyên như cũ, dùng Hybrid Search + Re-Ranker)
    """
    # (Code của hàm retrieve_hybrid_candidates + rerank_candidates)
    # ...
    # --- Giai đoạn 1: Lấy nét (Retrieve) ---
    query_vec = bi_encoder.encode([query], convert_to_tensor=False, normalize_embeddings=True)
    D, I_faiss = index_faiss.search(query_vec.astype(np.float32), K_RETRIEVE_HYBRID)
    
    tokenized_query = tokenize_vietnamese(query)
    bm25_scores = index_bm25.get_scores(tokenized_query)
    I_bm25 = np.argsort(bm25_scores)[::-1][:K_RETRIEVE_HYBRID]
    
    rrf_scores = {}
    for rank, faiss_index in enumerate(I_faiss[0]):
        doc_id = index_to_id_map[faiss_index]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        
    for rank, bm25_index in enumerate(I_bm25):
        doc_id = index_to_id_map[bm25_index]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        
    hybrid_candidates = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    candidate_doc_ids = [doc_id for doc_id, score in hybrid_candidates[:K_RETRIEVE_HYBRID]]
    
    # --- Giai đoạn 2: Xếp hạng lại (Re-Rank) ---
    pairs = []
    for doc_id in candidate_doc_ids:
        doc_text = id_to_text_map.get(doc_id, "")
        pairs.append([query, doc_text])
        
    with torch.no_grad():
        scores = cross_encoder.predict(pairs, show_progress_bar=False, convert_to_tensor=True)
    
    reranked_results = list(zip(candidate_doc_ids, scores))
    reranked_results.sort(key=lambda x: x[1], reverse=True)
    
    # --- Giai đoạn 3: Chọn lọc (Select) ---
    final_doc_ids = [doc_id for doc_id, score in reranked_results[:top_k_final]]
    retrieved_docs = [id_to_text_map[doc_id] for doc_id in final_doc_ids]
    
    del query_vec, D, I_faiss, bm25_scores, I_bm25, rrf_scores, pairs, scores
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
        
    return retrieved_docs


async def perform_generation_gemini(query: str, context: list[str]) -> str:
    """
    (Đã cập nhật) Tạo prompt và gọi API Gemini.
    """
    if not GEMINI_API_KEY:
        return "Lỗi: GEMINI_API_KEY chưa được cấu hình."
    if not context:
        return "Xin lỗi, tôi không tìm thấy thông tin triệu chứng nào liên quan."

    # --- Prompt Engineering ---
    context_str = "\n".join([f"- {doc}" for doc in context])
    # Prompt này kết hợp System Prompt và User Prompt thành một
    full_prompt = f"""Bạn là một trợ lý thông tin y tế AI. Nhiệm vụ của bạn là:
1.  Phân tích [Thông tin tham khảo] được cung cấp.
2.  Nếu có sự tương đồng, hãy liệt kê các triệu chứng chính và đề cập đến các tình trạng sức khỏe *có thể liên quan* được gợi ý trong thông tin tham khảo, **nhấn mạnh rằng đây KHÔNG phải là chẩn đoán.**
3.  **KHÔNG BAO GIỜ đưa ra chẩn đoán cuối cùng.**
4.  Luôn kết thúc bằng lời khuyên nên tham khảo ý kiến bác sĩ hoặc chuyên gia y tế để có chẩn đoán chính xác.
Trả lời bằng tiếng Việt.

---
[THÔNG TIN THAM KHẢO]
{context_str}
---

[CÂU HỎI CỦA NGƯỜI DÙNG]
{query}

[Phân tích và Thông tin liên quan]
"""

    # --- Gọi API Gemini bằng httpx ---
    async with httpx.AsyncClient(timeout=45.0) as client:
        try:
            # Gửi request đến URL API, truyền key trong params
            response = await client.post(
                GEMINI_API_URL,
                params={"key": GEMINI_API_KEY},
                json={
                    "contents": [{"parts": [{"text": full_prompt}]}],
                    "generationConfig": {
                        "temperature": 0.6,
                        "maxOutputTokens": 1024,
                    }
                }
            )
            response.raise_for_status() # Báo lỗi nếu API trả về 4xx/5xx
            result = response.json()

            # Xử lý nếu bị Safety Block (như bạn gặp trước đó)
            if 'candidates' not in result or not result['candidates']:
                 if 'promptFeedback' in result and 'blockReason' in result['promptFeedback']:
                     return f"Lỗi: API Gemini đã chặn prompt vì lý do an toàn: {result['promptFeedback']['blockReason']}"
                 return "Lỗi: API Gemini trả về cấu trúc không hợp lệ."

            # Trích xuất văn bản từ JSON response của Gemini
            answer = result['candidates'][0]['content']['parts'][0]['text']
            return answer.strip()
        
        except httpx.HTTPStatusError as e:
            print(f"Lỗi HTTP API Gemini: {e.response.text}")
            return f"Lỗi khi gọi API Gemini: {e.response.status_code}"
        except Exception as e:
            print(f"Lỗi không xác định khi gọi Gemini: {e}")
            return "Đã xảy ra lỗi hệ thống trong quá trình tạo câu trả lời."

# --- 3. Endpoint API (Cập nhật) ---

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: QueryRequest):
    """
    Main RAG endpoint (Hybrid + Re-Ranker + Gemini Generator)
    """
    # Bước R (Retrieval + Reranking)
    retrieved_context = await perform_retrieval(request.query, request.top_k)
    
    # Bước G (Generation) - Đổi sang hàm Gemini
    answer = await perform_generation_gemini(request.query, retrieved_context) 
    
    return ChatResponse(
        answer=answer,
        retrieved_context=retrieved_context
    )

@app.get("/")
async def root():
    return {"message": "Medical RAG API (Hybrid + Re-Ranker + Gemini) is running."}

# --- Cách chạy (Trong terminal) ---
# uvicorn main:app --reload