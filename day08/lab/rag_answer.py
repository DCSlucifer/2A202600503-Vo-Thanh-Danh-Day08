"""
rag_answer.py — Sprint 2 + Sprint 3: Retrieval & Grounded Answer
================================================================
Sprint 2: Dense retrieval + grounded answer với citation
Sprint 3: Hybrid retrieval (Dense + BM25 với RRF) để tăng recall
          cho corpus có cả câu tự nhiên lẫn từ kỹ thuật/mã lỗi.

Lý do chọn Hybrid:
  Corpus gồm: policy (ngôn ngữ tự nhiên) + SLA (mã ticket P1/P2) +
  access control (level 1-4, alias tên cũ) + helpdesk FAQ (mã lỗi ERR-xxx).
  Dense tốt với ngữ nghĩa nhưng bỏ lỡ exact keyword.
  BM25 bắt được exact term nhưng yếu với paraphrase.
  RRF fusion giữ điểm mạnh của cả hai.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CẤU HÌNH
# =============================================================================

TOP_K_SEARCH = 10    # Số chunk lấy từ vector store trước rerank (search rộng)
TOP_K_SELECT = 3     # Số chunk gửi vào prompt sau select (top-3 sweet spot)

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Cache BM25 để không rebuild mỗi lần query
_bm25_index = None
_bm25_chunks = None


# =============================================================================
# RETRIEVAL — DENSE (Vector Search)
# =============================================================================

def retrieve_dense(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    """
    Dense retrieval: tìm kiếm theo embedding similarity trong ChromaDB.

    Returns:
        List chunks với text, metadata, score (cosine similarity).
    """
    import chromadb
    from index import get_embedding, CHROMA_DB_DIR

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_collection("rag_lab")

    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB cosine distance = 1 - similarity; score cao = tốt hơn
        score = 1.0 - dist
        chunks.append({
            "text": doc,
            "metadata": meta,
            "score": score,
        })

    return chunks


# =============================================================================
# RETRIEVAL — SPARSE / BM25 (Keyword Search)
# =============================================================================

def _build_bm25_index() -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Build BM25 index từ toàn bộ chunks trong ChromaDB.
    Cache để tái sử dụng.
    """
    global _bm25_index, _bm25_chunks

    if _bm25_index is not None:
        return _bm25_index, _bm25_chunks

    import chromadb
    from rank_bm25 import BM25Okapi
    from index import CHROMA_DB_DIR

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_collection("rag_lab")

    # Lấy tất cả chunks
    results = collection.get(include=["documents", "metadatas"])
    all_chunks = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        all_chunks.append({"text": doc, "metadata": meta, "score": 0.0})

    # Tokenize: lowercase + split (đủ tốt cho tiếng Việt không dấu và tiếng Anh)
    tokenized_corpus = [chunk["text"].lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    _bm25_index = bm25
    _bm25_chunks = all_chunks
    return bm25, all_chunks


def retrieve_sparse(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    """
    Sparse retrieval: BM25 keyword search.
    Mạnh ở: exact term, mã lỗi, tên riêng (P1, ERR-403, Level 3).
    """
    from rank_bm25 import BM25Okapi

    bm25, all_chunks = _build_bm25_index()

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Lấy top_k indices theo score cao nhất
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        chunk = all_chunks[idx].copy()
        # Normalize BM25 score sang [0, 1] để dễ so sánh với dense
        max_score = max(scores) if max(scores) > 0 else 1.0
        chunk["score"] = float(scores[idx]) / max_score
        results.append(chunk)

    return results


# =============================================================================
# RETRIEVAL — HYBRID (Dense + Sparse với Reciprocal Rank Fusion)
# =============================================================================

def retrieve_hybrid(
    query: str,
    top_k: int = TOP_K_SEARCH,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval: kết hợp dense và sparse bằng Reciprocal Rank Fusion (RRF).

    RRF formula:
        RRF_score(doc) = dense_weight / (60 + dense_rank)
                       + sparse_weight / (60 + sparse_rank)

    Ưu điểm: không cần normalize score giữa hai hệ thống khác nhau,
    chỉ dùng rank position → ổn định hơn.
    """
    # Lấy top-k rộng từ cả hai
    dense_results = retrieve_dense(query, top_k=top_k * 2)
    sparse_results = retrieve_sparse(query, top_k=top_k * 2)

    # Build RRF score dict
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict] = {}

    for rank, chunk in enumerate(dense_results):
        key = chunk["text"][:100]  # dùng đầu text làm key
        rrf_scores[key] = rrf_scores.get(key, 0) + dense_weight / (60 + rank + 1)
        chunk_map[key] = chunk

    for rank, chunk in enumerate(sparse_results):
        key = chunk["text"][:100]
        rrf_scores[key] = rrf_scores.get(key, 0) + sparse_weight / (60 + rank + 1)
        if key not in chunk_map:
            chunk_map[key] = chunk

    # Sort theo RRF score
    sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

    results = []
    for key in sorted_keys[:top_k]:
        chunk = chunk_map[key].copy()
        chunk["score"] = rrf_scores[key]
        results.append(chunk)

    return results


# =============================================================================
# RERANK (Sprint 3 — không dùng trong variant chính nhưng có sẵn)
# =============================================================================

def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = TOP_K_SELECT,
) -> List[Dict[str, Any]]:
    """
    Rerank bằng cross-encoder nếu sentence-transformers có CrossEncoder.
    Fallback về top_k đầu nếu không có.
    """
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [[query, chunk["text"]] for chunk in candidates]
        scores = model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in ranked[:top_k]]
    except Exception:
        return candidates[:top_k]


# =============================================================================
# QUERY TRANSFORMATION
# =============================================================================

def transform_query(query: str, strategy: str = "expansion") -> List[str]:
    """
    Query expansion: thêm alias/tên đồng nghĩa cho query.
    """
    if strategy == "expansion":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            prompt = (
                f"Given this query in Vietnamese: '{query}'\n"
                "Generate 2 alternative phrasings or related Vietnamese terms.\n"
                "Output as JSON array of strings only. Example: [\"alt1\", \"alt2\"]"
            )
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=100,
            )
            import json
            alternatives = json.loads(response.choices[0].message.content)
            return [query] + alternatives
        except Exception:
            return [query]
    return [query]


# =============================================================================
# GENERATION — GROUNDED ANSWER FUNCTION
# =============================================================================

def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """
    Đóng gói danh sách chunks thành context block để đưa vào prompt.
    Mỗi chunk có số thứ tự [1], [2], ... để model dễ trích dẫn.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        section = meta.get("section", "")
        score = chunk.get("score", 0)
        text = chunk.get("text", "")

        header = f"[{i}] {source}"
        if section:
            header += f" | {section}"
        if score > 0:
            header += f" | score={score:.2f}"

        context_parts.append(f"{header}\n{text}")

    return "\n\n".join(context_parts)


def build_grounded_prompt(query: str, context_block: str) -> str:
    """
    Grounded prompt theo 4 quy tắc:
    1. Evidence-only: chỉ trả lời từ retrieved context
    2. Abstain: thiếu context thì nói không đủ dữ liệu
    3. Citation: gắn [số] khi trích dẫn
    4. Short, clear, stable: ngắn gọn, nhất quán
    """
    prompt = f"""Answer only from the retrieved context below.
If the context is insufficient to answer the question, explicitly say "Không đủ dữ liệu trong tài liệu để trả lời câu hỏi này." and do not make up any information.
Cite the source number in brackets like [1] or [2] when referencing information.
Keep your answer short, clear, and factual.
Respond in the same language as the question (Vietnamese if question is in Vietnamese).

Question: {query}

Context:
{context_block}

Answer:"""
    return prompt


def call_llm(prompt: str) -> str:
    """
    Gọi LLM để sinh câu trả lời.
    Dùng OpenAI gpt-4o-mini (temperature=0 để output ổn định cho eval).
    """
    provider = os.getenv("LLM_PROVIDER", "openai")

    if provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text

    else:
        # Default: OpenAI
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=512,
        )
        return response.choices[0].message.content


def rag_answer(
    query: str,
    retrieval_mode: str = "dense",
    top_k_search: int = TOP_K_SEARCH,
    top_k_select: int = TOP_K_SELECT,
    use_rerank: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Pipeline RAG hoàn chỉnh: query → retrieve → (rerank) → generate.

    Args:
        query: Câu hỏi
        retrieval_mode: "dense" | "sparse" | "hybrid"
        top_k_search: Số chunk lấy từ vector store
        top_k_select: Số chunk đưa vào prompt sau select
        use_rerank: Có dùng cross-encoder rerank không
        verbose: In thêm thông tin debug

    Returns:
        Dict với answer, sources, chunks_used, query, config.
    """
    config = {
        "retrieval_mode": retrieval_mode,
        "top_k_search": top_k_search,
        "top_k_select": top_k_select,
        "use_rerank": use_rerank,
    }

    # --- Bước 1: Retrieve ---
    if retrieval_mode == "dense":
        candidates = retrieve_dense(query, top_k=top_k_search)
    elif retrieval_mode == "sparse":
        candidates = retrieve_sparse(query, top_k=top_k_search)
    elif retrieval_mode == "hybrid":
        candidates = retrieve_hybrid(query, top_k=top_k_search)
    else:
        raise ValueError(f"retrieval_mode không hợp lệ: {retrieval_mode}")

    if verbose:
        print(f"\n[RAG] Query: {query}")
        print(f"[RAG] Retrieved {len(candidates)} candidates (mode={retrieval_mode})")
        for i, c in enumerate(candidates[:3]):
            print(f"  [{i+1}] score={c.get('score', 0):.3f} | {c['metadata'].get('source', '?')} | {c['metadata'].get('section', '')}")

    # --- Bước 2: Rerank hoặc truncate ---
    if use_rerank:
        candidates = rerank(query, candidates, top_k=top_k_select)
    else:
        candidates = candidates[:top_k_select]

    if verbose:
        print(f"[RAG] After select: {len(candidates)} chunks")

    # --- Bước 3: Build context và prompt ---
    context_block = build_context_block(candidates)
    prompt = build_grounded_prompt(query, context_block)

    if verbose:
        print(f"\n[RAG] Context block preview:\n{context_block[:300]}...\n")

    # --- Bước 4: Generate ---
    answer = call_llm(prompt)

    # --- Bước 5: Extract sources ---
    sources = list({
        c["metadata"].get("source", "unknown")
        for c in candidates
    })

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "chunks_used": candidates,
        "config": config,
    }


# =============================================================================
# SPRINT 3: SO SÁNH BASELINE VS VARIANT
# =============================================================================

def compare_retrieval_strategies(query: str) -> None:
    """
    So sánh dense vs hybrid với cùng một query.
    Dùng để justify tại sao chọn hybrid cho Sprint 3.
    A/B Rule: chỉ đổi retrieval_mode, giữ nguyên mọi tham số khác.
    """
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)

    strategies = ["dense", "hybrid"]

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        try:
            result = rag_answer(query, retrieval_mode=strategy, verbose=False)
            print(f"Answer: {result['answer'][:200]}")
            print(f"Sources: {result['sources']}")
        except Exception as e:
            print(f"Lỗi: {e}")


# =============================================================================
# MAIN — Demo và Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 2 + 3: RAG Answer Pipeline")
    print("=" * 60)

    test_queries = [
        "SLA xử lý ticket P1 là bao lâu?",
        "Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?",
        "Ai phải phê duyệt để cấp quyền Level 3?",
        "ERR-403-AUTH là lỗi gì?",  # Abstain test
    ]

    print("\n--- Sprint 2: Test Baseline (Dense) ---")
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            result = rag_answer(query, retrieval_mode="dense", verbose=True)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['sources']}")
        except Exception as e:
            print(f"Lỗi: {e}")

    print("\n--- Sprint 3: So sánh Dense vs Hybrid ---")
    compare_retrieval_strategies("Approval Matrix để cấp quyền hệ thống là tài liệu nào?")
    compare_retrieval_strategies("ERR-403-AUTH")
