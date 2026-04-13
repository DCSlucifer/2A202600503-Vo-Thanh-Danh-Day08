"""
app.py — Web UI for Full RAG Pipeline Testing
===============================================
Flask-based web interface to test all components of the RAG pipeline:
  - Chat: Ask questions and see retrieval + generation results
  - Index: View/rebuild the ChromaDB index
  - Eval: Run scorecard evaluation and see results
"""

import json
import os
import sys
import io
import traceback
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
import threading

# Fix Windows console encoding for Vietnamese characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()

# Import pipeline modules
from index import (
    build_index, list_chunks, inspect_metadata_coverage,
    DOCS_DIR, CHROMA_DB_DIR, preprocess_document, chunk_document
)
from rag_answer import (
    rag_answer, retrieve_dense, retrieve_sparse, retrieve_hybrid,
    build_context_block, TOP_K_SEARCH, TOP_K_SELECT
)

app = Flask(__name__, static_folder="static", template_folder="templates")

# =============================================================================
# ROUTES
# =============================================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Run RAG pipeline on a user query and return structured results."""
    data = request.json
    query = data.get("query", "").strip()
    retrieval_mode = data.get("retrieval_mode", "dense")
    top_k_search = int(data.get("top_k_search", TOP_K_SEARCH))
    top_k_select = int(data.get("top_k_select", TOP_K_SELECT))
    use_rerank = data.get("use_rerank", False)

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    try:
        result = rag_answer(
            query=query,
            retrieval_mode=retrieval_mode,
            top_k_search=top_k_search,
            top_k_select=top_k_select,
            use_rerank=use_rerank,
            verbose=False,
        )

        # Format chunks for frontend
        chunks_data = []
        for i, chunk in enumerate(result["chunks_used"]):
            chunks_data.append({
                "index": i + 1,
                "text": chunk["text"],
                "score": round(chunk.get("score", 0), 4),
                "source": chunk["metadata"].get("source", "unknown"),
                "section": chunk["metadata"].get("section", ""),
                "department": chunk["metadata"].get("department", ""),
                "effective_date": chunk["metadata"].get("effective_date", ""),
            })

        return jsonify({
            "answer": result["answer"],
            "sources": result["sources"],
            "chunks": chunks_data,
            "config": result["config"],
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc(),
        }), 500


@app.route("/api/index/status", methods=["GET"])
def api_index_status():
    """Get current index status — doc count, chunk count, metadata coverage."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        collection = client.get_collection("rag_lab")
        count = collection.count()

        # Get metadata stats
        results = collection.get(include=["metadatas"], limit=count)
        departments = {}
        sections = {}
        for meta in results["metadatas"]:
            dept = meta.get("department", "unknown")
            departments[dept] = departments.get(dept, 0) + 1
            sec = meta.get("section", "General")
            sections[sec] = sections.get(sec, 0) + 1

        # Get doc files
        doc_files = [f.name for f in DOCS_DIR.glob("*.txt")]

        return jsonify({
            "total_chunks": count,
            "doc_files": doc_files,
            "departments": departments,
            "sections": sections,
            "db_path": str(CHROMA_DB_DIR),
        })

    except Exception as e:
        return jsonify({
            "total_chunks": 0,
            "error": str(e),
            "doc_files": [f.name for f in DOCS_DIR.glob("*.txt")],
        })


@app.route("/api/index/chunks", methods=["GET"])
def api_index_chunks():
    """Get all chunks from the index."""
    try:
        import chromadb
        n = int(request.args.get("limit", 20))
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        collection = client.get_collection("rag_lab")
        results = collection.get(limit=n, include=["documents", "metadatas"])

        chunks = []
        for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
            chunks.append({
                "index": i + 1,
                "text": doc[:300],
                "full_text": doc,
                "source": meta.get("source", "N/A"),
                "section": meta.get("section", "N/A"),
                "department": meta.get("department", "N/A"),
                "effective_date": meta.get("effective_date", "N/A"),
                "access": meta.get("access", "N/A"),
            })

        return jsonify({"chunks": chunks})
    except Exception as e:
        return jsonify({"chunks": [], "error": str(e)})


@app.route("/api/index/rebuild", methods=["POST"])
def api_index_rebuild():
    """Rebuild the index from scratch."""
    try:
        build_index()
        return jsonify({"success": True, "message": "Index rebuilt successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/eval/run", methods=["POST"])
def api_eval_run():
    """Run evaluation scorecard for a given config."""
    data = request.json
    config_type = data.get("config", "baseline")

    from eval import run_scorecard, BASELINE_CONFIG, VARIANT_CONFIG, TEST_QUESTIONS_PATH

    config = BASELINE_CONFIG if config_type == "baseline" else VARIANT_CONFIG

    try:
        with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            test_questions = json.load(f)

        results = run_scorecard(config=config, test_questions=test_questions, verbose=False)

        # Calculate averages
        metrics = ["faithfulness", "relevance", "context_recall", "completeness"]
        averages = {}
        for metric in metrics:
            scores = [r[metric] for r in results if r[metric] is not None]
            averages[metric] = round(sum(scores) / len(scores), 2) if scores else None

        return jsonify({
            "results": results,
            "averages": averages,
            "config": config,
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/eval/questions", methods=["GET"])
def api_eval_questions():
    """Get test questions."""
    from eval import TEST_QUESTIONS_PATH
    try:
        with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            questions = json.load(f)
        return jsonify({"questions": questions})
    except Exception as e:
        return jsonify({"questions": [], "error": str(e)})


@app.route("/api/retrieval/compare", methods=["POST"])
def api_retrieval_compare():
    """Compare dense vs hybrid retrieval for the same query."""
    data = request.json
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    try:
        results = {}
        for mode in ["dense", "sparse", "hybrid"]:
            try:
                result = rag_answer(
                    query=query,
                    retrieval_mode=mode,
                    verbose=False,
                )
                chunks_data = []
                for i, chunk in enumerate(result["chunks_used"]):
                    chunks_data.append({
                        "index": i + 1,
                        "text": chunk["text"][:200],
                        "score": round(chunk.get("score", 0), 4),
                        "source": chunk["metadata"].get("source", "unknown"),
                        "section": chunk["metadata"].get("section", ""),
                    })

                results[mode] = {
                    "answer": result["answer"],
                    "chunks": chunks_data,
                    "sources": result["sources"],
                }
            except Exception as e:
                results[mode] = {"error": str(e)}

        return jsonify({"query": query, "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  RAG Pipeline UI — http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, port=5000, host="0.0.0.0")
