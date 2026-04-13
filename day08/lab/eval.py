"""
eval.py — Sprint 4: Evaluation & Scorecard
==========================================
Mục tiêu Sprint 4 (60 phút):
  - Chạy 10 test questions qua pipeline
  - Chấm điểm bằng LLM-as-Judge (bonus) theo 4 metrics
  - So sánh baseline vs variant (hybrid)
  - Ghi kết quả ra scorecard

Bonus: LLM-as-Judge tự động hoá việc chấm thay vì chấm thủ công.
"""

import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from rag_answer import rag_answer

# =============================================================================
# CẤU HÌNH
# =============================================================================

TEST_QUESTIONS_PATH = Path(__file__).parent / "data" / "test_questions.json"
RESULTS_DIR = Path(__file__).parent / "results"
LOGS_DIR = Path(__file__).parent / "logs"

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Cấu hình baseline (Sprint 2): Dense, không rerank
BASELINE_CONFIG = {
    "retrieval_mode": "dense",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": False,
    "label": "baseline_dense",
}

# Cấu hình variant (Sprint 3): Hybrid (Dense + BM25/RRF), không rerank
# Biến thay đổi duy nhất: retrieval_mode = "hybrid"
VARIANT_CONFIG = {
    "retrieval_mode": "hybrid",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": False,
    "label": "variant_hybrid",
}


# =============================================================================
# LLM-AS-JUDGE HELPER
# =============================================================================

def _call_judge_llm(prompt: str) -> str:
    """Gọi LLM để chấm điểm (judge)."""
    provider = os.getenv("LLM_PROVIDER", "openai")

    if provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    else:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        return response.choices[0].message.content


# =============================================================================
# SCORING FUNCTIONS — LLM-AS-JUDGE (Bonus)
# =============================================================================

def score_faithfulness(
    answer: str,
    chunks_used: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Faithfulness (LLM-as-Judge): Câu trả lời có bám đúng retrieved context không?
    Thang 1-5: 5 = hoàn toàn grounded, 1 = phần lớn bịa.
    """
    if not answer or answer.startswith("PIPELINE"):
        return {"score": None, "notes": "Pipeline error — skip"}

    context_preview = "\n".join(
        f"[{i+1}] {c.get('text', '')[:200]}" for i, c in enumerate(chunks_used)
    )

    prompt = f"""You are an evaluation judge for a RAG (Retrieval-Augmented Generation) system.

Retrieved context:
{context_preview}

Model answer:
{answer}

Rate the FAITHFULNESS of the answer on a scale of 1-5:
5 = Every claim in the answer is directly supported by the retrieved context
4 = Almost entirely grounded, one minor detail is uncertain
3 = Mostly grounded but some information may come from model knowledge
2 = Several claims are not found in the retrieved context
1 = Answer is mostly fabricated, not grounded in context

If the answer says "Không đủ dữ liệu" (insufficient data) and the context indeed lacks the answer, give score 5.

Output ONLY a JSON object: {{"score": <integer 1-5>, "reason": "<one sentence>"}}"""

    try:
        raw = _call_judge_llm(prompt)
        # Clean up markdown code blocks if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        return {
            "score": int(result["score"]),
            "notes": result.get("reason", ""),
        }
    except Exception as e:
        return {"score": None, "notes": f"Judge error: {e}"}


def score_answer_relevance(
    query: str,
    answer: str,
) -> Dict[str, Any]:
    """
    Answer Relevance (LLM-as-Judge): Answer có trả lời đúng câu hỏi không?
    Thang 1-5: 5 = trả lời trực tiếp và đầy đủ.
    """
    if not answer or answer.startswith("PIPELINE"):
        return {"score": None, "notes": "Pipeline error — skip"}

    prompt = f"""You are an evaluation judge for a RAG system.

Question: {query}
Answer: {answer}

Rate the RELEVANCE of the answer to the question on a scale of 1-5:
5 = Directly and completely answers the question
4 = Answers correctly but missing some minor details
3 = Related but not fully on target
2 = Partially off-topic
1 = Does not answer the question

Output ONLY a JSON object: {{"score": <integer 1-5>, "reason": "<one sentence>"}}"""

    try:
        raw = _call_judge_llm(prompt)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        return {
            "score": int(result["score"]),
            "notes": result.get("reason", ""),
        }
    except Exception as e:
        return {"score": None, "notes": f"Judge error: {e}"}


def score_context_recall(
    chunks_used: List[Dict[str, Any]],
    expected_sources: List[str],
) -> Dict[str, Any]:
    """
    Context Recall: Retriever có mang về đủ evidence cần thiết không?
    Tính theo tỷ lệ expected sources được retrieve.
    """
    if not expected_sources:
        return {"score": 5, "recall": 1.0, "notes": "No expected sources (abstain case) — full score"}

    retrieved_sources = {
        c.get("metadata", {}).get("source", "")
        for c in chunks_used
    }

    found = 0
    missing = []
    for expected in expected_sources:
        # Partial match theo tên file (bỏ qua path prefix)
        expected_name = expected.split("/")[-1].replace(".pdf", "").replace(".md", "")
        # Cũng match theo stem của source path trong metadata
        matched = any(
            expected_name.lower() in r.lower() or
            expected.lower() in r.lower()
            for r in retrieved_sources
        )
        if matched:
            found += 1
        else:
            missing.append(expected)

    recall = found / len(expected_sources)

    return {
        "score": round(recall * 5),
        "recall": recall,
        "found": found,
        "missing": missing,
        "notes": f"Retrieved {found}/{len(expected_sources)} expected sources" +
                 (f". Missing: {missing}" if missing else ""),
    }


def score_completeness(
    query: str,
    answer: str,
    expected_answer: str,
) -> Dict[str, Any]:
    """
    Completeness (LLM-as-Judge): Answer có bao phủ đủ điểm quan trọng không?
    So sánh với expected_answer.
    """
    if not answer or answer.startswith("PIPELINE"):
        return {"score": None, "notes": "Pipeline error — skip"}

    if not expected_answer:
        return {"score": None, "notes": "No expected answer to compare"}

    prompt = f"""You are an evaluation judge for a RAG system.

Question: {query}
Expected answer (reference): {expected_answer}
Model answer: {answer}

Rate the COMPLETENESS of the model answer compared to the expected answer on a scale of 1-5:
5 = Covers all key points from the expected answer
4 = Missing one minor detail
3 = Missing some important information
2 = Missing many important points
1 = Missing most core content

If both the expected and model answer indicate "insufficient data", give score 5.

Output ONLY a JSON object: {{"score": <integer 1-5>, "reason": "<one sentence>"}}"""

    try:
        raw = _call_judge_llm(prompt)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        return {
            "score": int(result["score"]),
            "notes": result.get("reason", ""),
        }
    except Exception as e:
        return {"score": None, "notes": f"Judge error: {e}"}


# =============================================================================
# SCORECARD RUNNER
# =============================================================================

def run_scorecard(
    config: Dict[str, Any],
    test_questions: Optional[List[Dict]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Chạy toàn bộ test questions qua pipeline và chấm điểm bằng LLM-as-Judge.
    """
    if test_questions is None:
        with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            test_questions = json.load(f)

    results = []
    label = config.get("label", "unnamed")

    print(f"\n{'='*70}")
    print(f"Chạy scorecard: {label}")
    print(f"Config: retrieval_mode={config.get('retrieval_mode')} | "
          f"top_k={config.get('top_k_select')} | rerank={config.get('use_rerank')}")
    print('='*70)

    for q in test_questions:
        question_id = q["id"]
        query = q["question"]
        expected_answer = q.get("expected_answer", "")
        expected_sources = q.get("expected_sources", [])
        category = q.get("category", "")

        if verbose:
            print(f"\n[{question_id}] {query}")

        # --- Gọi pipeline ---
        try:
            result = rag_answer(
                query=query,
                retrieval_mode=config.get("retrieval_mode", "dense"),
                top_k_search=config.get("top_k_search", 10),
                top_k_select=config.get("top_k_select", 3),
                use_rerank=config.get("use_rerank", False),
                verbose=False,
            )
            answer = result["answer"]
            chunks_used = result["chunks_used"]

        except Exception as e:
            answer = f"PIPELINE_ERROR: {e}"
            chunks_used = []

        # --- Chấm điểm bằng LLM-as-Judge ---
        faith = score_faithfulness(answer, chunks_used)
        relevance = score_answer_relevance(query, answer)
        recall = score_context_recall(chunks_used, expected_sources)
        complete = score_completeness(query, answer, expected_answer)

        row = {
            "id": question_id,
            "category": category,
            "query": query,
            "answer": answer,
            "expected_answer": expected_answer,
            "faithfulness": faith["score"],
            "faithfulness_notes": faith["notes"],
            "relevance": relevance["score"],
            "relevance_notes": relevance["notes"],
            "context_recall": recall["score"],
            "context_recall_notes": recall["notes"],
            "completeness": complete["score"],
            "completeness_notes": complete["notes"],
            "config_label": label,
        }
        results.append(row)

        if verbose:
            print(f"  Answer: {answer[:120]}...")
            print(f"  F={faith['score']} | R={relevance['score']} | "
                  f"Rc={recall['score']} | C={complete['score']}")

    # Tính averages
    print(f"\n--- Average Scores ({label}) ---")
    for metric in ["faithfulness", "relevance", "context_recall", "completeness"]:
        scores = [r[metric] for r in results if r[metric] is not None]
        avg = sum(scores) / len(scores) if scores else None
        if avg:
            print(f"  {metric}: {avg:.2f}/5")
        else:
            print(f"  {metric}: N/A")

    return results


# =============================================================================
# A/B COMPARISON
# =============================================================================

def compare_ab(
    baseline_results: List[Dict],
    variant_results: List[Dict],
    output_csv: Optional[str] = None,
) -> None:
    """
    So sánh baseline vs variant theo từng metric và từng câu hỏi.
    """
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]

    print(f"\n{'='*70}")
    print("A/B Comparison: Baseline (Dense) vs Variant (Hybrid)")
    print('='*70)
    print(f"{'Metric':<22} {'Baseline':>10} {'Variant':>10} {'Delta':>8}")
    print("-" * 55)

    for metric in metrics:
        b_scores = [r[metric] for r in baseline_results if r[metric] is not None]
        v_scores = [r[metric] for r in variant_results if r[metric] is not None]

        b_avg = sum(b_scores) / len(b_scores) if b_scores else None
        v_avg = sum(v_scores) / len(v_scores) if v_scores else None
        delta = (v_avg - b_avg) if (b_avg is not None and v_avg is not None) else None

        b_str = f"{b_avg:.2f}" if b_avg is not None else "N/A"
        v_str = f"{v_avg:.2f}" if v_avg is not None else "N/A"
        d_str = f"{delta:+.2f}" if delta is not None else "N/A"

        print(f"{metric:<22} {b_str:>10} {v_str:>10} {d_str:>8}")

    # Per-question comparison
    print(f"\n{'ID':<6} {'Baseline F/R/Rc/C':<22} {'Variant F/R/Rc/C':<22} {'Winner':<10}")
    print("-" * 65)

    b_by_id = {r["id"]: r for r in baseline_results}
    for v_row in variant_results:
        qid = v_row["id"]
        b_row = b_by_id.get(qid, {})

        b_scores_str = "/".join([str(b_row.get(m, "?")) for m in metrics])
        v_scores_str = "/".join([str(v_row.get(m, "?")) for m in metrics])

        b_total = sum(b_row.get(m, 0) or 0 for m in metrics)
        v_total = sum(v_row.get(m, 0) or 0 for m in metrics)
        winner = "Variant" if v_total > b_total else ("Baseline" if b_total > v_total else "Tie")

        print(f"{qid:<6} {b_scores_str:<22} {v_scores_str:<22} {winner:<10}")

    # Export CSV
    if output_csv:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = RESULTS_DIR / output_csv
        combined = baseline_results + variant_results
        if combined:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=combined[0].keys())
                writer.writeheader()
                writer.writerows(combined)
            print(f"\n✓ Kết quả A/B đã lưu vào: {csv_path}")


# =============================================================================
# SCORECARD MARKDOWN GENERATOR
# =============================================================================

def generate_scorecard_summary(results: List[Dict], label: str) -> str:
    """Tạo báo cáo scorecard dạng markdown."""
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]
    averages = {}
    for metric in metrics:
        scores = [r[metric] for r in results if r[metric] is not None]
        averages[metric] = sum(scores) / len(scores) if scores else None

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    md = f"""# Scorecard: {label}
Generated: {timestamp}
Scoring method: LLM-as-Judge (gpt-4o-mini, temperature=0)

## Summary

| Metric | Average Score (1-5) |
|--------|---------------------|
"""
    for metric, avg in averages.items():
        avg_str = f"{avg:.2f}/5" if avg is not None else "N/A"
        md += f"| {metric.replace('_', ' ').title()} | {avg_str} |\n"

    md += "\n## Per-Question Results\n\n"
    md += "| ID | Category | Answer (preview) | Faithful | Relevant | Recall | Complete |\n"
    md += "|----|----------|-----------------|----------|----------|--------|----------|\n"

    for r in results:
        answer_preview = r.get("answer", "")[:60].replace("|", "/")
        md += (
            f"| {r['id']} | {r['category']} | {answer_preview}... | "
            f"{r.get('faithfulness', 'N/A')} | {r.get('relevance', 'N/A')} | "
            f"{r.get('context_recall', 'N/A')} | {r.get('completeness', 'N/A')} |\n"
        )

    md += "\n## Notes\n"
    for r in results:
        if r.get("faithfulness_notes"):
            md += f"- [{r['id']}] Faithfulness: {r['faithfulness_notes']}\n"

    return md


# =============================================================================
# GRADING LOG GENERATOR (Tạo logs/grading_run.json)
# =============================================================================

def run_grading(
    grading_questions_path: str,
    retrieval_mode: str = "hybrid",
    output_path: str = None,
) -> List[Dict]:
    """
    Chạy pipeline với grading_questions.json và ghi log.
    Dùng cấu hình tốt nhất (hybrid).
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    with open(grading_questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    log = []
    for q in questions:
        print(f"  Running: {q['id']} — {q['question'][:50]}...")
        try:
            result = rag_answer(
                q["question"],
                retrieval_mode=retrieval_mode,
                verbose=False,
            )
            log.append({
                "id": q["id"],
                "question": q["question"],
                "answer": result["answer"],
                "sources": result["sources"],
                "chunks_retrieved": len(result["chunks_used"]),
                "retrieval_mode": result["config"]["retrieval_mode"],
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            log.append({
                "id": q["id"],
                "question": q["question"],
                "answer": f"PIPELINE_ERROR: {e}",
                "sources": [],
                "chunks_retrieved": 0,
                "retrieval_mode": retrieval_mode,
                "timestamp": datetime.now().isoformat(),
            })

    if output_path is None:
        output_path = str(LOGS_DIR / "grading_run.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Grading log lưu tại: {output_path}")
    return log


# =============================================================================
# MAIN — Chạy evaluation
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 4: Evaluation & Scorecard (LLM-as-Judge)")
    print("=" * 60)

    # Load test questions
    print(f"\nLoading test questions từ: {TEST_QUESTIONS_PATH}")
    with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        test_questions = json.load(f)
    print(f"Tìm thấy {len(test_questions)} câu hỏi")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Chạy Baseline ---
    print("\n--- Chạy Baseline (Dense) ---")
    baseline_results = run_scorecard(
        config=BASELINE_CONFIG,
        test_questions=test_questions,
        verbose=True,
    )
    baseline_md = generate_scorecard_summary(baseline_results, "baseline_dense")
    scorecard_baseline_path = RESULTS_DIR / "scorecard_baseline.md"
    scorecard_baseline_path.write_text(baseline_md, encoding="utf-8")
    print(f"\n✓ Scorecard baseline lưu tại: {scorecard_baseline_path}")

    # --- Chạy Variant (Hybrid) ---
    print("\n--- Chạy Variant (Hybrid) ---")
    variant_results = run_scorecard(
        config=VARIANT_CONFIG,
        test_questions=test_questions,
        verbose=True,
    )
    variant_md = generate_scorecard_summary(variant_results, "variant_hybrid")
    scorecard_variant_path = RESULTS_DIR / "scorecard_variant.md"
    scorecard_variant_path.write_text(variant_md, encoding="utf-8")
    print(f"\n✓ Scorecard variant lưu tại: {scorecard_variant_path}")

    # --- A/B Comparison ---
    if baseline_results and variant_results:
        compare_ab(
            baseline_results,
            variant_results,
            output_csv="ab_comparison.csv",
        )

    # --- Grading questions (nếu có file) ---
    grading_path = Path(__file__).parent / "data" / "grading_questions.json"
    if grading_path.exists():
        print(f"\n--- Chạy Grading Questions ---")
        run_grading(
            grading_questions_path=str(grading_path),
            retrieval_mode="hybrid",
        )
    else:
        print(f"\n[INFO] {grading_path} chưa có — sẽ chạy khi file được public lúc 17:00.")

    print("\n✓ Sprint 4 hoàn thành!")
