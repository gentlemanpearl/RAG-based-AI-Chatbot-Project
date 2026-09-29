"""
Automated Benchmark Test Suite for Agentic AI RAG Chatbot.
Executes the 5-6 official assignment benchmark queries and verifies:
1. Grounded responses for in-domain queries
2. Strict refusal for out-of-domain queries (e.g. FIFA World Cup)
3. Confidence scoring and context retrieval validity
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Benchmark queries specified in Candidate Reference Guide
BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "What is Agentic AI according to the eBook?",
        "expected_type": "grounded_answer",
        "description": "Definition and scope of Agentic AI from eBook",
    },
    {
        "id": 2,
        "query": "How do AI agents differ from traditional automation systems?",
        "expected_type": "grounded_answer",
        "description": "Comparison between autonomous agents and deterministic automation",
    },
    {
        "id": 3,
        "query": "What are the core components of an Agentic Architecture?",
        "expected_type": "grounded_answer",
        "description": "Structural components (perception, reasoning, memory, tools, execution)",
    },
    {
        "id": 4,
        "query": "What role does memory play in Agentic AI workflows?",
        "expected_type": "grounded_answer",
        "description": "Short-term vs long-term memory in agent decision-making",
    },
    {
        "id": 5,
        "query": "Who won the 2022 FIFA World Cup?",
        "expected_type": "refusal",
        "description": "Out-of-domain validation query (Must refuse / lack context)",
    },
    {
        "id": 6,
        "query": "What challenges do organizations face in multi-agent orchestration?",
        "expected_type": "grounded_answer",
        "description": "Enterprise orchestration challenges (legacy systems, compliance, coordination)",
    },
]


def test_via_python_module(queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Test queries directly invoking LangGraph RAG pipeline."""
    from src.graph import run_rag_pipeline

    results = []
    print("\n" + "=" * 75)
    print("  RUNNING BENCHMARK EVALUATION VIA DIRECT LANGGRAPH WORKFLOW")
    print("=" * 75)

    for q_item in queries:
        qid = q_item["id"]
        query = q_item["query"]
        expected = q_item["expected_type"]

        print(f"\n[Test #{qid}] {query}")
        print(f" Expected Behavior: {expected.upper()}")
        print("-" * 70)

        start = time.perf_counter()
        try:
            res = run_rag_pipeline(query)
            elapsed = round((time.perf_counter() - start) * 1000, 1)

            answer = res.get("answer", "")
            score = res.get("confidence_score", 0.0)
            is_refusal = res.get("refusal", False)
            chunks = res.get("retrieved_chunks", [])

            # Check grounding / refusal assertion
            if expected == "refusal":
                passed = is_refusal or "does not contain" in answer.lower()
                status_label = "PASSED (Strict Refusal)" if passed else "FAILED (Failed to refuse)"
            else:
                passed = not is_refusal and len(chunks) > 0 and score > 0.5
                status_label = "PASSED (Grounded Answer)" if passed else "FAILED (Low confidence or refused)"

            print(f" Status:           {status_label}")
            print(f" Confidence Score: {score:.2f}")
            print(f" Latency:          {elapsed} ms")
            print(f" Chunks Retrieved: {len(chunks)}")
            print(f" Answer Summary:\n{answer}\n")

            results.append({
                "test_id": qid,
                "query": query,
                "expected": expected,
                "passed": passed,
                "confidence_score": score,
                "refusal": is_refusal,
                "chunks_count": len(chunks),
                "answer": answer,
                "latency_ms": elapsed,
            })
        except Exception as e:
            print(f"❌ Error evaluating query #{qid}: {e}")
            results.append({
                "test_id": qid,
                "query": query,
                "expected": expected,
                "passed": False,
                "error": str(e),
            })

    return results


def test_via_http_api(queries: List[Dict[str, Any]], base_url: str = "http://localhost:8000") -> List[Dict[str, Any]]:
    """Test queries by sending HTTP requests to FastAPI /chat endpoint."""
    import requests

    endpoint = f"{base_url}/chat"
    results = []
    print("\n" + "=" * 75)
    print(f"  RUNNING BENCHMARK EVALUATION VIA FASTAPI ENDPOINT: {endpoint}")
    print("=" * 75)

    for q_item in queries:
        qid = q_item["id"]
        query = q_item["query"]
        expected = q_item["expected_type"]

        print(f"\n[Test #{qid}] {query}")
        print(f" Expected Behavior: {expected.upper()}")
        print("-" * 70)

        start = time.perf_counter()
        try:
            resp = requests.post(endpoint, json={"query": query}, timeout=60)
            elapsed = round((time.perf_counter() - start) * 1000, 1)

            if resp.status_code != 200:
                print(f"❌ HTTP Error {resp.status_code}: {resp.text}")
                results.append({
                    "test_id": qid,
                    "query": query,
                    "expected": expected,
                    "passed": False,
                    "status_code": resp.status_code,
                    "error": resp.text,
                })
                continue

            data = resp.json()
            answer = data.get("answer", "")
            score = data.get("confidence_score", 0.0)
            is_refusal = data.get("refusal", False)
            chunks = data.get("retrieved_chunks", [])

            if expected == "refusal":
                passed = is_refusal or "does not contain" in answer.lower()
                status_label = "PASSED (Strict Refusal)" if passed else "FAILED (Failed to refuse)"
            else:
                passed = not is_refusal and len(chunks) > 0 and score > 0.5
                status_label = "PASSED (Grounded Answer)" if passed else "FAILED (Low confidence or refused)"

            print(f" Status:           {status_label}")
            print(f" Confidence Score: {score:.2f}")
            print(f" Latency:          {elapsed} ms")
            print(f" Chunks Retrieved: {len(chunks)}")
            print(f" Answer:\n{answer}\n")

            results.append({
                "test_id": qid,
                "query": query,
                "expected": expected,
                "passed": passed,
                "confidence_score": score,
                "refusal": is_refusal,
                "chunks_count": len(chunks),
                "answer": answer,
                "latency_ms": elapsed,
            })
        except Exception as e:
            print(f"❌ Connection error testing #{qid}: {e}")
            results.append({
                "test_id": qid,
                "query": query,
                "expected": expected,
                "passed": False,
                "error": str(e),
            })

    return results


def print_summary_table(results: List[Dict[str, Any]]):
    """Print markdown and console summary table of test results."""
    print("\n" + "=" * 80)
    print("                     BENCHMARK EVALUATION SUMMARY                     ")
    print("=" * 80)
    print(f"{'#':<3} | {'Status':<8} | {'Score':<6} | {'Refusal':<8} | {'Query':<45}")
    print("-" * 80)
    
    passed_count = 0
    for r in results:
        status_txt = "PASS" if r.get("passed") else "FAIL"
        if r.get("passed"):
            passed_count += 1
        score_txt = f"{r.get('confidence_score', 0.0):.2f}"
        refusal_txt = str(r.get("refusal", False))
        q_short = r["query"][:45]
        print(f"{r['test_id']:<3} | {status_txt:<8} | {score_txt:<6} | {refusal_txt:<8} | {q_short:<45}")

    print("-" * 80)
    print(f"Total Tests: {len(results)} | Passed: {passed_count} | Pass Rate: {(passed_count/len(results))*100:.1f}%\n")


def main():
    parser = argparse.ArgumentParser(description="Run benchmark tests for RAG Chatbot")
    parser.add_argument("--api", action="store_true", help="Test via FastAPI HTTP endpoint instead of module")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="FastAPI base URL")
    parser.add_argument("--output", type=str, default="test_results.json", help="Path to save JSON test results")
    args = parser.parse_args()

    if args.api:
        results = test_via_http_api(BENCHMARK_QUERIES, base_url=args.url)
    else:
        results = test_via_python_module(BENCHMARK_QUERIES)

    print_summary_table(results)

    # Save to JSON
    out_file = PROJECT_ROOT / args.output
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Saved detailed benchmark results to: {out_file}")


if __name__ == "__main__":
    main()
