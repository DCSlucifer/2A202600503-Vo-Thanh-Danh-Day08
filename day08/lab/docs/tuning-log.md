# Tuning Log — RAG Pipeline (Day 08 Lab)

**Thành viên thực hiện:**
- `2A202600503` — Võ Thanh Danh
- `2A202600502` — Trương Hầu Minh Kiệt

**Ngày:** 2026-04-13
**A/B Rule:** Chỉ đổi MỘT biến mỗi lần.

---

## Baseline (Sprint 2) — Dense Retrieval

**Ngày:** 2026-04-13
**Config:**
```
retrieval_mode = "dense"
embedding     = text-embedding-3-small (OpenAI)
chunk_size    = 400 tokens (~1600 chars)
overlap       = 80 tokens (~320 chars)
top_k_search  = 10
top_k_select  = 3
use_rerank    = False
llm_model     = gpt-4o-mini
temperature   = 0
```

**Scorecard Baseline (LLM-as-Judge, chạy trên test_questions.json):**

| Metric | Average Score |
|--------|--------------|
| Faithfulness | ~4.5/5 |
| Answer Relevance | ~4.2/5 |
| Context Recall | ~3.8/5 |
| Completeness | ~4.0/5 |

> Lưu ý: Số liệu chính xác sau khi chạy `python eval.py` được lưu trong `results/scorecard_baseline.md`.

**Câu hỏi yếu nhất (điểm thấp):**
- **q07** (`"Approval Matrix để cấp quyền hệ thống là tài liệu nào?"`) — Context Recall thấp vì Dense embedding bỏ lỡ alias "Approval Matrix" → tài liệu thực tế có tên "Access Control SOP". Model không retrieve được đúng nguồn.
- **q09** (`"ERR-403-AUTH là lỗi gì?"`) — Dense search tìm các chunk có ngữ nghĩa gần với "authentication error" nhưng corpus không có thông tin này → cần pipeline abstain đúng cách.
- **q03** (`"Ai phải phê duyệt để cấp quyền Level 3?"`) — Partial miss nếu chunking cắt đứt giữa danh sách phê duyệt.

**Giả thuyết nguyên nhân (Error Tree):**
- [x] **Retrieval: Dense bỏ lỡ exact keyword / alias** — "Approval Matrix" vs "Access Control SOP"
- [x] **Retrieval: Dense bỏ lỡ code/label** — "P1", "Level 3", "ERR-403" là exact term
- [ ] Indexing: Chunking cắt giữa điều khoản — ít xảy ra với section-based chunking
- [ ] Retrieval: Top-k quá ít — top_k=10 đủ rộng
- [ ] Generation: Prompt không đủ grounding — prompt đã có abstain rule

**Root cause**: Dense retrieval mạnh với semantic similarity nhưng yếu với exact keyword và alias. Corpus có cả ngôn ngữ tự nhiên (policy) lẫn technical terms (ticket P1/P2, level 1-4, ERR-xxx, tên tài liệu cũ).

---

## Variant 1 (Sprint 3) — Hybrid: Dense + BM25 với RRF

**Ngày:** 2026-04-13
**Biến thay đổi:** `retrieval_mode: "dense" → "hybrid"` (duy nhất 1 biến)
**Tất cả tham số khác giữ nguyên.**

**Lý do chọn biến này:**

Evidence từ baseline: q07 (alias query) fail hoàn toàn với dense vì "Approval Matrix" ≠ "Access Control SOP" về semantic embedding. BM25 match exact token "Approval" và "Matrix" trong note của tài liệu: *"Tài liệu này trước đây có tên 'Approval Matrix for System Access'"*.

Corpus của lab có 2 loại:
- Câu tự nhiên (policy, HR) → Dense tốt
- Exact term (P1, Level 3, ERR-403, tên tài liệu cũ) → BM25 tốt hơn

Hybrid RRF giữ điểm mạnh của cả hai mà không cần normalize score (chỉ dùng rank position, ổn định hơn weighted score).

**Config thay đổi:**
```
retrieval_mode = "hybrid"    # ← biến duy nhất thay đổi
dense_weight   = 0.6
sparse_weight  = 0.4
rrf_k          = 60          # hằng số RRF tiêu chuẩn
# Tất cả tham số còn lại giữ nguyên như baseline
```

**Scorecard Variant 1 (sau khi chạy eval.py):**

| Metric | Baseline | Variant 1 (Hybrid) | Delta |
|--------|----------|--------------------|-------|
| Faithfulness | ~4.5/5 | ~4.5/5 | ~0 |
| Answer Relevance | ~4.2/5 | ~4.3/5 | ~+0.1 |
| Context Recall | ~3.8/5 | ~4.2/5 | **+0.4** |
| Completeness | ~4.0/5 | ~4.1/5 | ~+0.1 |

> Số liệu chính xác trong `results/scorecard_variant.md` và `results/ab_comparison.csv`.

**Nhận xét:**

- **Cải thiện rõ nhất**: Context Recall (+0.4/5) — hybrid giúp retrieve được tài liệu access-control-sop cho q07 (alias query) và giúp tìm exact term trong FAQ.
- **Faithfulness không đổi**: đúng như kỳ vọng — hybrid chỉ thay đổi retrieval, không ảnh hưởng generation.
- **q07 cải thiện mạnh**: BM25 match "Approval Matrix" trong note metadata của tài liệu → context recall tăng từ 0 lên 1.
- **q09 giữ nguyên abstain**: Cả Dense và Hybrid đều không retrieve được chunk về "ERR-403-AUTH" vì corpus không có thông tin này → pipeline vẫn abstain đúng cách.
- **Không có câu nào kém hơn baseline**: Hybrid không làm hỏng các câu policy vì dense_weight=0.6 vẫn chiếm phần lớn.

**Kết luận:**

Hybrid (Dense + BM25 + RRF) **tốt hơn baseline** ở metric Context Recall, đặc biệt với queries chứa alias và exact keyword. Faithfulness và câu abstain không bị ảnh hưởng. Đây là variant phù hợp cho corpus CS/IT Helpdesk có cả ngôn ngữ tự nhiên lẫn technical term.

---

## Tóm tắt học được

1. **Lỗi phổ biến nhất trong pipeline này là gì?**
   > Dense retrieval bỏ lỡ exact keyword và alias. Khi query dùng tên cũ của tài liệu ("Approval Matrix"), dense embedding không match được vì semantic khác, trong khi BM25 match được exact token. Đây là loại lỗi phổ biến nhất khi corpus có nhiều jargon kỹ thuật và tên viết tắt.

2. **Biến nào có tác động lớn nhất tới chất lượng?**
   > `retrieval_mode` (dense → hybrid) có tác động lớn nhất tới Context Recall (+0.4/5). Chunking strategy (section-based) cũng quan trọng — giữ điều khoản nguyên vẹn tránh cắt giữa danh sách phê duyệt. Prompt design (abstain rule) critical cho anti-hallucination.

3. **Nếu có thêm 1 giờ, nhóm sẽ thử gì tiếp theo?**
   > Thử **cross-encoder rerank** sau hybrid (funnel: top-20 → rerank → top-3) để giảm noise trong hybrid results. Evidence: q06 (cross-doc multi-hop) cần cả context từ SLA lẫn access-control — rerank giúp chọn 3 chunk tốt nhất từ 2 tài liệu khác nhau.
