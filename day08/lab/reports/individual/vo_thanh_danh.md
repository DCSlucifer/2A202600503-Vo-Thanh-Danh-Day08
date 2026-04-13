# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Võ Thanh Danh  
**MSSV:** 2A202600503  
**Vai trò trong nhóm:** Sprint 1 Owner (Indexing) + Sprint 2 Owner (Baseline RAG) + Bonus UI  
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
