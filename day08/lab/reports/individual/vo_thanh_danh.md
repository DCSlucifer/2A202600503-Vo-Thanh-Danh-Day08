# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Võ Thanh Danh  
**MSSV:** 2A202600503  
**Vai trò trong nhóm:** Sprint 1 Owner (Indexing) + Sprint 2 Owner (Baseline RAG) + Bonus UI. Ngoài phần được phân công, bản thân tự chủ động hoàn thành thêm Sprint 3 (Sparse) và Sprint 4 (Hybrid) trong repo cá nhân để tự học và nắm rõ toàn bộ pipeline  
**Ngày nộp:** 2026-04-13  

---

## 1. Tôi đã làm gì trong lab này?

### Sprint 1 — Indexing (`index.py`)

Implement toàn bộ pipeline index từ raw text đến ChromaDB:

- **`preprocess_document()`**: Parse header metadata (Source, Department, Effective Date, Access) từ mỗi file `.txt`. Gặp dòng `=== ... ===` thì kết thúc header và bắt đầu thu thập nội dung. Giữ lại alias/ghi chú trong phần content thay vì bỏ đi — quyết định này về sau giúp BM25 match được "Approval Matrix for System Access".

- **`chunk_document()`**: Chiến lược hai tầng — split theo section heading `=== ... ===` trước, sau đó mới split theo size (400 tokens ≈ 1600 ký tự, overlap 80 tokens ≈ 320 ký tự). Mỗi chunk giữ đúng `section` metadata của section gốc để sau này có thể filter.

- **`get_embedding()`**: Gọi OpenAI `text-embedding-3-small`. Mặc định `EMBEDDING_PROVIDER=openai`, có fallback sang `sentence-transformers` local nếu không có API key.

- **`build_index()`**: Xóa collection cũ bằng `client.delete_collection()` trước khi `get_or_create_collection()` — tránh duplicate chunk khi rebuild. Batch upsert vào ChromaDB với cosine space.

- Metadata đầy đủ 5 fields: `source`, `section`, `department`, `effective_date`, `access`.

### Sprint 2 — Baseline RAG (`rag_answer.py`)

Implement retrieval dense và generation có grounding:

- **`retrieve_dense()`**: Query ChromaDB bằng embedding của query. Chuyển cosine distance → similarity score bằng `score = 1.0 - dist`. Trả về list chunk với text, metadata, score.

- **`build_grounded_prompt()`**: Prompt theo 4 quy tắc: evidence-only, abstain rule tường minh (`"Không đủ dữ liệu..."`), citation instruction (`[1]`, `[2]`), trả lời ngắn gọn cùng ngôn ngữ với câu hỏi. Abstain phải viết tường minh — nếu chỉ ghi "answer from context" thì model vẫn hallucinate.

- **`call_llm()`**: Gọi `gpt-4o-mini` với `temperature=0` để output ổn định cho eval. Có hỗ trợ Gemini nếu cần qua `LLM_PROVIDER=gemini`.

- **`rag_answer()`**: Orchestrate pipeline: retrieve → (optional rerank) → build context → generate. Return dict chuẩn với `answer`, `sources`, `chunks_used`, `config`.

- **`build_context_block()`**: Format chunks thành numbered context với header `[i] source | section | score=x.xx`.

---

## 2. Bonus — Web UI (`app.py` + `templates/` + `static/`)

Ngoài Sprint 1 và 2, tôi xây thêm Flask web UI để test toàn bộ pipeline trực quan thay vì chỉ chạy script.

**Tính năng:**

- **Chat tab**: Nhập query, chọn retrieval mode (dense / sparse / hybrid), top-k search/select, toggle rerank. Hiển thị answer + danh sách chunks được dùng (source, section, score).

- **Index tab**: Xem trạng thái ChromaDB (tổng chunks, phân bố department, danh sách docs). Có nút rebuild index không cần terminal.

- **Eval tab**: Chạy scorecard baseline hoặc variant trực tiếp từ UI, xem kết quả từng câu + average score theo 4 metrics.

- **Retrieval Compare**: Nhập một query, UI tự chạy cả 3 mode (dense / sparse / hybrid) song song và hiển thị kết quả side-by-side để thấy sự khác biệt rõ ràng.

**Stack:** Flask + vanilla JS + CSS tĩnh. Không dùng React hay framework nặng vì mục tiêu là demo nhanh, không phải production app.

**API endpoints chính:**
| Endpoint | Mô tả |
|---|---|
| `POST /api/chat` | Chạy RAG pipeline, trả JSON |
| `GET /api/index/status` | Thống kê ChromaDB |
| `POST /api/index/rebuild` | Rebuild index |
| `POST /api/eval/run` | Chạy scorecard |
| `POST /api/retrieval/compare` | So sánh 3 mode |

### Hướng dẫn test UI

#### Khởi động

```bash
# 1. Đảm bảo đã có .env với OPENAI_API_KEY
# 2. Cài dependencies
pip install flask rank-bm25 chromadb openai sentence-transformers

# 3. Index tài liệu trước (nếu chưa có ChromaDB)
python index.py

# 4. Chạy Flask server
python app.py
# Server chạy tại http://127.0.0.1:5000
```

#### Tab 1 — Chat

1. Mở `http://127.0.0.1:5000` → chọn tab **Chat**.
2. Nhập query vào ô text, ví dụ: `"SLA cho Priority 1 là bao nhiêu?"`.
3. Chọn **Retrieval Mode**: `dense` / `sparse` / `hybrid` (khuyến nghị `hybrid` để thấy đủ kết quả).
4. Đặt **Top-K Search** = `10`, **Top-K Select** = `3`.
5. Bật/tắt **Rerank** để so sánh thứ tự chunk trước và sau cross-encoder.
6. Nhấn **Send** → quan sát:
   - Phần **Answer**: câu trả lời được grounding từ context.
   - Phần **Sources**: danh sách chunks `[i] source | section | score=x.xx`.
   - Trường hợp abstain: nhập câu không có trong tài liệu, ví dụ `"Giá cổ phiếu hôm nay là bao nhiêu?"` → kỳ vọng model trả lời `"Không đủ dữ liệu..."`.

#### Tab 2 — Index

1. Chọn tab **Index** → trang hiển thị ngay trạng thái ChromaDB:
   - Tổng số chunks đã index.
   - Phân bố theo `department` (IT, HR, Finance…).
   - Danh sách tài liệu nguồn.
2. Nhấn **Rebuild Index** để xóa collection cũ và index lại từ đầu — không cần mở terminal.
3. Sau khi rebuild xong, F5 lại tab Index để xác nhận số chunk không thay đổi (không bị duplicate).

#### Tab 3 — Eval

1. Chọn tab **Eval**.
2. Chọn **Config**: `baseline` (dense, top-k=10) hoặc `variant` (hybrid, top-k=10).
3. Nhấn **Run Scorecard** → UI hiển thị tiến trình chạy 10 câu hỏi.
4. Kết quả hiển thị theo từng câu:
   - Query, answer, và 4 score: `groundedness`, `relevance`, `completeness`, `abstain_correctness`.
5. Cuối trang có **Average Score** theo từng metric.
6. File kết quả tự động ghi vào `results/scorecard_baseline.md` hoặc `results/scorecard_variant.md`.

#### Tab 4 — Retrieval Compare

1. Chọn tab **Retrieval Compare**.
2. Nhập một query bất kỳ, ví dụ: `"Quy trình xử lý ticket P1?"`.
3. Nhấn **Compare** → UI gọi cả 3 mode song song (`dense` / `sparse` / `hybrid`).
4. Ba cột kết quả hiển thị side-by-side, mỗi cột có danh sách chunk với score tương ứng.
5. Dùng tab này để trực quan hóa:
   - `dense` mạnh ở semantic similarity.
   - `sparse` mạnh ở exact keyword (mã lỗi, tên chính sách).
   - `hybrid` (RRF) cân bằng hai cái trên.

#### Kiểm tra nhanh bằng `curl`

```bash
# Chat
curl -X POST http://127.0.0.1:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "SLA P1 là bao lâu?", "mode": "hybrid", "top_k": 5}'

# Index status
curl http://127.0.0.1:5000/api/index/status

# Compare retrieval
curl -X POST http://127.0.0.1:5000/api/retrieval/compare \
  -H "Content-Type: application/json" \
  -d '{"query": "access control policy"}'
```

---

## Phần tự học hỏi thêm (Sprint 3 & Sprint 4)

> Phần này nằm ngoài phạm vi phân công, được thực hiện trong repo cá nhân để tự tìm hiểu sâu hơn toàn bộ pipeline.

### Sprint 3 — Sparse & Hybrid Retrieval (`rag_answer.py`)

Extend `rag_answer.py` để hỗ trợ thêm hai retrieval mode ngoài dense:

- **`retrieve_sparse()`**: BM25 keyword search dùng `rank-bm25`. Build `BM25Okapi` từ toàn bộ chunks trong ChromaDB, tokenize bằng `lower().split()`. Normalize score sang `[0, 1]` bằng cách chia cho max score của batch. Mạnh ở: exact keyword (mã lỗi `ERR-403-AUTH`, ticket level `P1`, tên tài liệu).

- **`retrieve_hybrid()`**: Kết hợp dense và sparse bằng **Reciprocal Rank Fusion (RRF)**. Công thức:
  ```
  RRF_score(doc) = dense_weight / (60 + dense_rank)
                 + sparse_weight / (60 + sparse_rank)
  ```
  Dùng rank position thay vì normalize score giữa hai hệ thống — tránh vấn đề scale mismatch. Mặc định `dense_weight=0.6`, `sparse_weight=0.4`.

- **`rerank()`**: Cross-encoder rerank với `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers). Có fallback về top-k nếu không có thư viện hoặc model.

- **`transform_query()`**: Query expansion dùng LLM để sinh 2 phiên bản paraphrase của câu hỏi gốc. Ý tưởng để tăng recall, chưa tích hợp vào pipeline chính.

- **`compare_retrieval_strategies()`**: So sánh dense vs hybrid với cùng query để justify lý do chọn hybrid làm variant config trong eval.

### Sprint 4 — Evaluation & LLM-as-Judge (`eval.py`)

Implement toàn bộ evaluation pipeline:

- **LLM-as-Judge**: Dùng `gpt-4o-mini` làm judge chấm mỗi câu trả lời theo 4 metrics, trả về score 0–2 cho từng metric:
  | Metric | Ý nghĩa |
  |---|---|
  | `groundedness` | Câu trả lời có bám vào context không (không hallucinate) |
  | `relevance` | Có trả lời đúng câu hỏi không |
  | `completeness` | Có đủ thông tin không |
  | `abstain_correctness` | Nếu phải abstain thì có abstain không |

- **`run_scorecard()`**: Chạy toàn bộ 10 test questions qua pipeline với một config (baseline hoặc variant), gọi judge cho từng câu, tính average score mỗi metric, ghi ra `.md` và `.json`.

- **A/B Comparison**: `BASELINE_CONFIG` (dense, top-k=10) vs `VARIANT_CONFIG` (hybrid, top-k=10). Biến thay đổi duy nhất là `retrieval_mode` để kết quả so sánh fair. Kết quả ghi ra `results/ab_comparison.csv`.

- **Data**: 10 test questions trong `data/test_questions.json` bao gồm câu hỏi về SLA, policy hoàn tiền, access control, và intentionally unanswerable questions để test abstain behavior.

---

## 3. Điều tôi hiểu rõ hơn sau lab này

**Chunking strategy ảnh hưởng lớn đến retrieval**: Ban đầu tôi nghĩ chunk nhỏ hơn thì tốt hơn vì precision cao. Thực tế với corpus có cấu trúc section-based như helpdesk SOP, split theo section heading trước rồi mới split theo size tốt hơn nhiều — giữ được context trong cùng một điều khoản thay vì cắt ngang giữa câu.

**Abstain là kỹ thuật, không phải mặc định**: LLM không tự động abstain khi thiếu context — phải ép bằng prompt tường minh. Câu "If the context is insufficient, explicitly say '...'" phải có đủ phần quoted string để model copy y chang, không diễn giải lại.

**ChromaDB cache và rebuild**: Nếu chỉ dùng `get_or_create_collection()` mà không delete trước, collection cũ sẽ bị append — gây duplicate chunk, distort retrieval score. Phải `delete_collection()` + `get_or_create_collection()` mỗi khi rebuild.

---

## 4. Điều tôi gặp khó khăn

**Encoding tiếng Việt trên Windows**: Flask chạy trên Windows không tự handle UTF-8 output — Vietnamese characters bị lỗi khi print ra console. Fix: wrap `sys.stdout` và `sys.stderr` với `io.TextIOWrapper(encoding='utf-8')` ở đầu `app.py`.

**Cosine distance vs similarity**: ChromaDB trả về `distances` với cosine space, nhưng giá trị là distance (thấp = gần), không phải similarity. Phải convert `score = 1.0 - distance` để có số có nghĩa khi hiển thị trên UI.

**BM25 tokenization tiếng Việt**: `rank-bm25` tokenize bằng `split()` (split theo space) — không hiểu morphology tiếng Việt. Với query "cấp quyền hệ thống", mỗi syllable là một token riêng, nhưng vẫn đủ để match được exact term trong tài liệu. Đây là giới hạn đã chấp nhận — cải thiện cần thư viện riêng như `underthesea`.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì?

**Streaming response trên UI**: Hiện tại `/api/chat` trả toàn bộ JSON khi LLM xong. Thêm Server-Sent Events (SSE) để stream từng token ra UI sẽ cải thiện UX đáng kể — không cần chờ 3-5 giây nhìn màn hình trắng.

**Persistent chat history**: UI hiện stateless — mỗi query là một pipeline độc lập. Thêm session-based conversation history để cho phép follow-up questions ("câu trả lời trên lấy từ section nào?") mà không cần nhập lại context.

**Export eval results**: Nút export kết quả scorecard ra CSV/JSON từ UI thay vì phải vào terminal đọc `logs/grading_run.json`.
