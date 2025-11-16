# chatbot_ViMedical
# Project Chatbot RAG Y tế

Đây là project chatbot hỗ trợ thông tin y tế, sử dụng pipeline RAG (Retrieval-Augmented Generation) tiên tiến.

## 1. Dành cho Nhóm Phát triển (Cách chạy API)

Để chạy API server trên máy local của bạn:

### Bước 1: Clone và Tải Hiện vật

1.  Clone repo này:
    `git clone [ĐƯỜNG DẪN REPO CỦA BẠN]`
2.  Tải các thư mục `models/`, `artifacts/`, và `data/` từ [https://drive.google.com/drive/folders/1hF5g3O0jhlZWub8uw-WYuMIMG1X5Pvv1?usp=drive_link]
3.  Đặt các thư mục đó vào thư mục project sao cho cấu trúc giống như sau:
    ```
    /project/
    ├── main.py
    ├── requirements.txt
    ├── .env
    ├── data/
    │   └── corpus_with_id.csv
    ├── models/
    │   └── triplet-finetuned-model-v1/
    └── artifacts/
        ├── index_Model_B_Triplet_1Epoch.faiss
        └── medical_index_B.bm25
    ```

### Bước 2: Cài đặt Môi trường

1.  (Khuyến nghị) Tạo môi trường ảo: `python -m venv .venv` và kích hoạt nó.
2.  Cài đặt thư viện: `pip install -r requirements.txt`

### Bước 3: Cấu hình API Key

1.  Tạo một file tên là `.env` trong thư mục gốc.
2.  Thêm API Key của bạn vào (ví dụ: Gemini):
    `GEMINI_API_KEY=YOUR_KEY_HERE`

### Bước 4: Chạy Server

1.  Mở terminal và chạy:
    `uvicorn main:app --reload`
2.  Server API đang chạy tại: `http://127.0.0.1:8000`

### Bước 5: Sử dụng API (Dành cho Frontend)

* **Endpoint:** `POST /chat`
* **Địa chỉ:** `http://127.0.0.1:8000/chat`
* **JSON Input:**
    ```json
    {
      "query": "tôi bị sốt và đau họng",
      "top_k": 5
    }
    ```
* **JSON Output:**
    ```json
    {
      "answer": "Dựa trên thông tin tham khảo... (LLM trả lời)",
      "retrieved_context": [
        "Triệu chứng A...",
        "Triệu chứng B...",
        ...
      ]
    }
    ```
* Giao diện test (Swagger UI): `http://127.0.0.1:8000/docs`

---

## 2. Dành cho Nhóm Báo cáo (Câu chuyện của Project) 📊

Phần này giải thích **tại sao** chúng ta lại làm các bước phức tạp này. Đây là hành trình MLOps của chúng ta:

### Mục tiêu

Xây dựng một hệ thống RAG có khả năng truy xuất (Retrieve) thông tin triệu chứng chính xác từ 12.060 tài liệu.

### Hành trình Tối ưu hóa (Data-Driven Decisions)

Chúng ta đã thực hiện 3 thử nghiệm lớn để tìm ra pipeline tốt nhất. "Tốt nhất" được đo bằng chỉ số **Recall@5** (khả năng tìm thấy đáp án đúng trong 5 kết quả đầu).

#### Thử nghiệm 1: Baseline (Model A) vs. Fine-tuning (Model B)

* **Model A (Gốc):** Dùng mô hình `paraphrase-multilingual-mpnet-base-v2` có sẵn.
* **Model B (Fine-tuned):** Dạy lại (fine-tune) Model A trên dữ liệu của chúng ta bằng `TripletLoss` trong 1 epoch.
* **Kết quả:** Model B thắng (Recall@5: **10.4%** vs 8.5% của Model A).
* **Quyết định:** Fine-tuning hiệu quả. Chúng ta chọn **Model B** làm nền tảng.

#### Thử nghiệm 2: Baseline (Model B) vs. Hybrid Search (Chiến lược 1)

* **Vấn đề:** Model B (chỉ FAISS) có thể bỏ lỡ các từ khóa y tế quan trọng.
* **Giải pháp:** Kết hợp Model B (ngữ nghĩa) với BM25 (từ khóa) bằng thuật toán RRF.
* **Kết quả:** Hybrid Search thắng (Recall@5: 10.4%, Recall@10: **16.1%** vs 10.4%).
* **Quyết định:** Hybrid Search cải thiện đáng kể khả năng tìm kiếm (đặc biệt là Recall@10 tăng 54%).

#### Thử nghiệm 3: Hybrid Search vs. Hybrid + Re-Ranker (Chiến lược 3)

* **Vấn đề:** Hybrid Search tìm *ra* nhiều kết quả tốt, nhưng *sắp xếp* chúng chưa tối ưu (Recall@3 không tăng).
* **Giải pháp:** Thêm một tầng Re-Ranker (Cross-Encoder) để đọc 50 kết quả của Hybrid Search và chấm điểm lại, chọn ra 5 kết quả tốt nhất.
* **Kết quả:** Hybrid + Re-Ranker **thắng tuyệt đối** (Recall@5: **13.3%** vs 10.4%; Recall@3: **11.4%** vs 7.6%).

### Kết luận (Kiến trúc cuối cùng)

Pipeline cuối cùng của chúng ta (`main.py`) là kiến trúc tốt nhất sau 3 vòng lặp MLOps, kết hợp:
1.  **Model B (Fine-tuned):** Để hiểu ngữ nghĩa y tế.
2.  **BM25:** Để bắt từ khóa chính xác.
3.  **Re-Ranker:** Để xếp hạng (tăng độ chính xác) Top 5.
4.  **Gemini/OpenAI:** Để giao tiếp với người dùng.
