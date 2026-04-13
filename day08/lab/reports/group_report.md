# Báo Cáo Nhóm — Lab Day 08: RAG Pipeline

**Ngày nộp:** 13/04/2026
**Số thành viên:** 2 người

| Thành viên | Vai trò |
|------------|---------|
| Võ Thành Danh | Tech Lead / Retrieval Owner |
| Trương Hầu Minh Kiệt | Eval Owner / Documentation Owner |

---

## 1. Tổng quan dự án

### Mục tiêu
Xây dựng RAG pipeline hoàn chỉnh cho trợ lý nội bộ CS + IT Helpdesk, trả lời câu hỏi về chính sách hoàn tiền, SLA ticket, quy trình cấp quyền, và HR policy dựa trên 5 tài liệu nội bộ. Pipeline phải đảm bảo:
- **Grounded**: Mọi câu trả lời bám sát tài liệu, có citation `[1][2]...`
- **Abstain**: Không bịa đặt khi thiếu thông tin
- **Tunable**: So sánh A/B giữa các cấu hình retrieval

### Cấu trúc repo
```
lab/
├── index.py                    # Sprint 1 — Build ChromaDB index
├── rag_answer.py               # Sprint 2+3 — Retrieval + Generation
├── eval.py                     # Sprint 4 — LLM-as-Judge evaluation
├── data/docs/                  # 5 tài liệu nội bộ (.txt)
├── data/test_questions.json    # 10 câu hỏi test
├── chroma_db/                  # ChromaDB persistent storage
├── docs/architecture.md        # Mô tả kiến trúc pipeline
├── docs/tuning-log.md          # Log thí nghiệm A/B
├── results/scorecard_baseline.md
├── results/scorecard_variant.md
├── results/ab_comparison.csv
└── reports/
    ├── group_report.md         # Báo cáo nhóm (file này)
    └── individual/             # Báo cáo cá nhân
```

---

## 2. Phân chia công việc

### Võ Thành Danh — Tech Lead / Retrieval Owner

| Sprint | Công việc cụ thể |
|--------|-------------------|
| Sprint 1 | Thiết kế chunking strategy (heading-based → paragraph fallback), implement `preprocess_document()` và `chunk_document()` trong `index.py` |
| Sprint 2 | Implement `retrieve_dense()` — query ChromaDB với cosine similarity, `retrieve_sparse()` — BM25 keyword search với `rank_bm25` |
| Sprint 3 | Implement `retrieve_hybrid()` — RRF merge (Dense 60% + BM25 40%), `rerank()` — LLM-as-Judge reranking, `transform_query()` — query expansion/decomposition/HyDE |
| Xuyên suốt | Thiết kế kiến trúc tổng thể pipeline, quyết định tech stack (ChromaDB, Sentence Transformers, GPT-4o-mini) |

**Quyết định kỹ thuật tiêu biểu:**
- Chọn `paraphrase-multilingual-MiniLM-L12-v2` làm embedding model vì hỗ trợ tiếng Việt, chạy local không cần API key, dimension 384 phù hợp với dataset nhỏ.
- RRF formula: `RRF_score = 0.6 × 1/(60 + dense_rank) + 0.4 × 1/(60 + sparse_rank)` — ưu tiên dense (semantic) nhưng vẫn giữ sparse (exact term) cho các câu hỏi có keyword chuyên ngành.
- LLM rerank thay vì cross-encoder vì môi trường lab không download được model cross-encoder; LLM rerank đơn giản, preview 300 ký tự/chunk để tiết kiệm token.

### Trương Hầu Minh Kiệt — Eval Owner / Documentation Owner

| Sprint | Công việc cụ thể |
|--------|-------------------|
| Sprint 2 | Implement `build_context_block()` và `build_grounded_prompt()` trong `rag_answer.py` — thiết kế prompt template ép citation và abstain |
| Sprint 4 | Implement toàn bộ `eval.py`: 4 scoring functions (`score_faithfulness`, `score_answer_relevance`, `score_context_recall`, `score_completeness`), `run_scorecard()`, `compare_ab()`, `generate_scorecard_summary()` |
| Documentation | Viết `docs/architecture.md` (kiến trúc pipeline, chunking decision, retrieval config, failure mode checklist, Mermaid diagram) và `docs/tuning-log.md` (3 experiments, hypothesis, config, expected vs actual results) |
| Xuyên suốt | Thiết kế 10 test questions (`data/test_questions.json`) bao gồm easy/medium/hard, abstain case, alias query |

**Quyết định kỹ thuật tiêu biểu:**
- Dùng LLM-as-Judge thay vì heuristic cho evaluation — prompt LLM chấm theo thang 1-5 với rubric rõ ràng, parse JSON response.
- Context Recall dùng fuzzy matching (normalize tên file, ignore extension/dash) để so sánh expected_sources vs retrieved_sources — tránh false negative do khác format path.
- Grounded prompt ép 4 quy tắc: evidence-only, abstain, citation `[n]`, ngắn gọn + cùng ngôn ngữ câu hỏi.

---

## 3. Sprint Deliverables — Kết quả từng Sprint

### Sprint 1: Indexing (`index.py`)
- **`python index.py` chạy không lỗi** ✅
- Tạo ChromaDB index từ 5 tài liệu: `policy_refund_v4.txt`, `sla_p1_2026.txt`, `access_control_sop.txt`, `it_helpdesk_faq.txt`, `hr_leave_policy.txt`
- Chunking: heading-based (`=== Section ===`) → paragraph fallback, chunk_size=400 tokens, overlap=80 tokens
- **Mỗi chunk có 5 metadata fields**: `source`, `section`, `department`, `effective_date`, `access` ✅
- Tổng số chunks: ~31 chunks

### Sprint 2: Retrieval + Answer (`rag_answer.py`)
- **`rag_answer("SLA ticket P1?")` trả về answer có citation `[1]`** ✅
- Dense retrieval qua ChromaDB cosine similarity, `score = 1 - distance`
- Grounded prompt template ép citation và abstain
- **Query "ERR-403-AUTH" → pipeline trả về abstain** ✅: _"Không tìm thấy thông tin này trong tài liệu nội bộ."_
- LLM: GPT-4o-mini, temperature=0, max_tokens=512

### Sprint 3: Variant Implementation
- **Hybrid retrieval (Dense + BM25 với RRF)** ✅
- **LLM-as-Judge Rerank** ✅
- **Query Transformation (expansion, decomposition, HyDE)** — implement sẵn, không bật mặc định
- Scorecard baseline và variant đều có số liệu thực ✅

### Sprint 4: Evaluation (`eval.py`)
- **`python eval.py` chạy end-to-end với 10 test questions không crash** ✅
- LLM-as-Judge cho 4 metrics: Faithfulness, Relevance, Context Recall, Completeness
- A/B comparison với delta rõ ràng ✅
- Output: `scorecard_baseline.md`, `scorecard_variant.md`, `ab_comparison.csv`

---

## 4. Kết quả Evaluation — A/B Comparison

### Tổng quan metrics

| Metric | Baseline (Dense) | Variant (Hybrid+Rerank) | Delta | Winner |
|--------|-------------------|-------------------------|-------|--------|
| Faithfulness | **3.90**/5 | 3.60/5 | −0.30 | Baseline |
| Relevance | **4.10**/5 | 3.80/5 | −0.30 | Baseline |
| Context Recall | 5.00/5 | 5.00/5 | 0.00 | Tie |
| Completeness | **4.40**/5 | 4.30/5 | −0.10 | Baseline |

### Phân tích per-question

| ID | Difficulty | Baseline (F/R/Rc/C) | Variant (F/R/Rc/C) | Nhận xét |
|----|-----------|----------------------|---------------------|----------|
| q01 | easy | 5/5/5/5 ✅ | 5/5/5/5 ✅ | Cả hai hoàn hảo |
| q02 | easy | 5/5/5/5 ✅ | 5/5/5/5 ✅ | Cả hai hoàn hảo |
| q03 | medium | 4/5/5/5 | 4/5/5/5 | Cả hai thêm "IT Security" — minor hallucination |
| q04 | medium | 5/5/5/5 ✅ | 5/5/5/5 ✅ | Cả hai hoàn hảo |
| q05 | easy | 5/5/5/5 ✅ | 4/5/5/5 | Variant thêm chi tiết unlock không có trong context |
| q06 | medium | 5/5/5/5 ✅ | 5/5/5/5 ✅ | Cả hai hoàn hảo |
| q07 | hard | 3/4/5/3 ⚠️ | 1/1/5/1 ❌ | Baseline khá hơn, Variant abstain sai |
| q08 | easy | 5/5/5/5 ✅ | 5/5/5/5 ✅ | Cả hai hoàn hảo |
| q09 | hard | 1/1/5/4 | 1/1/5/4 | Cả hai abstain đúng nhưng LLM-judge cho điểm thấp vì context vẫn retrieve được chunks |
| q10 | hard | 1/1/5/2 | 1/1/5/3 | Cả hai abstain thay vì giải thích từ context |

### Kết luận A/B

**Baseline (Dense) thắng tổng thể** với các câu hỏi dễ-trung bình. Tuy nhiên:

1. **Variant không cải thiện được q07 (Approval Matrix / alias query)** — đây là kỳ vọng chính khi thêm hybrid. Nguyên nhân: mặc dù BM25 tìm được chunk chứa "Approval Matrix", LLM rerank có thể đã loại nhầm chunk này (false negative trong rerank step). Baseline dense tuy không tìm đúng alias nhưng vẫn tìm được chunk liên quan đến access control → trả lời được phần nào.

2. **q09 và q10 — Abstain handling**: Cả hai cấu hình đều abstain đúng (không bịa) nhưng quá "cộc" — chỉ nói "không tìm thấy" mà không giải thích từ context đã retrieve. Đây là hạn chế của prompt template hiện tại.

3. **Context Recall = 5.00 ở cả hai** — retriever đều tìm được đúng source, vấn đề nằm ở generation step (LLM không tận dụng hết context).

**Biến tác động nhiều nhất:** `use_rerank` — rerank bằng LLM gây side effect là loại nhầm chunk quan trọng ở một số câu hard.

---

## 5. Quyết định kiến trúc quan trọng

### 5.1 Chunking Strategy
- **Heading-based (`=== Section ===`) trước → paragraph fallback**: Tôn trọng cấu trúc tài liệu gốc, mỗi chunk là một đơn vị ngữ nghĩa (1 điều khoản, 1 section).
- **Chunk size 400 tokens, overlap 80 tokens**: Đủ ngữ cảnh cho 1 điều khoản policy, overlap tránh mất thông tin ở ranh giới.
- **5 metadata fields**: `source`, `section`, `department`, `effective_date`, `access` — đủ cho citation, filter, và access control.

### 5.2 Retrieval Config
- **Baseline**: Dense cosine similarity, top_k_search=10, top_k_select=3, no rerank
- **Variant**: Hybrid RRF (Dense 60% + BM25 40%), top_k_search=10, top_k_select=3, LLM rerank
- **Lý do chọn hybrid**: Corpus có cả câu tự nhiên (chính sách hoàn tiền) lẫn keyword chuyên ngành (SLA P1, Level 3, Approval Matrix). Dense tốt cho semantic, BM25 tốt cho exact term.

### 5.3 Generation
- **Grounded prompt 4 quy tắc**: evidence-only, abstain, citation, ngắn gọn
- **Temperature = 0**: Output ổn định cho evaluation, reproducible
- **GPT-4o-mini**: Cân bằng chi phí và chất lượng

---

## 6. Khó khăn và bài học

### Khó khăn gặp phải

| Vấn đề | Nguyên nhân | Giải pháp |
|---------|-------------|-----------|
| ChromaDB distance vs similarity | ChromaDB trả `distances` (0=identical, 2=opposite), không phải similarity score | Fix: `score = 1 - distance` để chuyển về cosine similarity 0-1 |
| LLM rerank loại nhầm chunk | Rerank preview 300 ký tự có thể không đủ context cho LLM đánh giá relevance chính xác | Cân nhắc tăng preview length hoặc dùng cross-encoder nếu có thể |
| Abstain quá "cộc" | Prompt chỉ instruct "nói không biết" nhưng không yêu cầu giải thích từ context có sẵn | Cải tiến prompt: "Nếu không đủ → giải thích tại sao và gợi ý nguồn khác" |
| q03 hallucinate "IT Security" | Cả 2 config đều thêm "IT Security" vào danh sách approver dù context không đề cập | LLM dùng model knowledge — cần prompt mạnh hơn hoặc post-processing check |

### Bài học rút ra

1. **Hybrid ≠ luôn tốt hơn Dense.** Trong test set của chúng tôi, baseline dense thắng tổng thể. Hybrid chỉ có lợi khi query dùng alias/tên cũ, nhưng kết hợp với LLM rerank có thể gây side effect (loại nhầm chunk).

2. **LLM-as-Judge có bias.** Câu q09 (abstain case), pipeline trả lời đúng "không tìm thấy" nhưng LLM-judge cho Faithfulness=1 vì context có chunks (irrelevant) — judge không hiểu rằng abstain là hành vi mong muốn.

3. **Evaluation framework quan trọng không kém pipeline.** Nếu chỉ chạy query thủ công, chúng tôi sẽ nghĩ pipeline chạy tốt. LLM-as-Judge scorecard phát hiện được các failure mode ẩn (q07, q09, q10).

---

## 7. Đề xuất cải tiến

| Cải tiến | Evidence từ scorecard | Impact dự kiến |
|----------|----------------------|----------------|
| **HyDE (Hypothetical Document Embedding)** | q10 faithfulness=1 — model abstain thay vì giải thích từ context | Tăng recall cho câu hỏi mơ hồ / phrasal mismatch |
| **Metadata filter theo department** | q03 có thể retrieve chunk từ department không liên quan → noise | Giảm noise, tăng faithfulness |
| **Cải tiến prompt abstain** | q09, q10 đều abstain quá cộc, không tận dụng context đã retrieve | Tăng completeness cho abstain cases |
| **Cross-encoder rerank thay LLM rerank** | q07 variant bị LLM rerank loại nhầm chunk quan trọng | Giảm false negative trong rerank, tăng faithfulness |

---

## 8. Cấu hình tốt nhất (Best Config)

Dựa trên kết quả A/B, cấu hình tốt nhất cho use case hiện tại:

```python
BEST_CONFIG = {
    "retrieval_mode": "dense",    # Baseline thắng tổng thể
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": False,          # LLM rerank gây side effect
}
```

> **Lưu ý:** Nếu corpus mở rộng thêm nhiều tài liệu có alias/tên cũ, hybrid sẽ cần thiết. Nhưng với 5 tài liệu hiện tại, dense đủ tốt và ổn định hơn.

---
