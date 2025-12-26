# scripts/run_eval.py
# Quantitative evaluation runner (vector vs hybrid) aligned with notebook sampling.

from __future__ import annotations

import os
import json
from pathlib import Path

from dotenv import load_dotenv

from amazon_skincare_graphrag.config import get_neo4j_driver, get_openai_client
from amazon_skincare_graphrag.embeddings import get_embedding
from amazon_skincare_graphrag.eval import fetch_test_dataset, calculate_metrics
from amazon_skincare_graphrag.llm import generate_synthetic_query
from amazon_skincare_graphrag.retrieval import run_search, run_hybrid_query


def main() -> None:
    load_dotenv(dotenv_path=".env")

    # Clients / connections
    driver = get_neo4j_driver()
    client = get_openai_client()

    # Settings
    sample_size = int(os.getenv("EVAL_SAMPLE_SIZE", "50"))
    top_k = 10  # fixed to match notebook (HitRate@10, NDCG@10)
    use_synthetic = os.getenv("USE_SYNTHETIC_QUERY", "1").strip() not in {"0", "false", "False"}

    print(f"[Eval] sample_size={sample_size}, top_k={top_k}")
    # Debug: show effective eval settings (mask secrets)
    print("[ENV] EVAL_SAMPLE_SIZE =", os.getenv("EVAL_SAMPLE_SIZE"))
    print("[ENV] EVAL_TOP_K       =", os.getenv("EVAL_TOP_K"))
    print("[ENV] USE_SYNTHETIC_QUERY =", os.getenv("USE_SYNTHETIC_QUERY"))
    print("[ENV] NEO4J_URI prefix =", (os.getenv("NEO4J_URI") or "")[:25] + "...")
    print("[ENV] OPENAI_API_KEY set =", bool(os.getenv("OPENAI_API_KEY")))

    # Test cases: user_id, target_product, raw_review_text, loyalty_score
    test_cases = fetch_test_dataset(driver, sample_size=sample_size)
    print("[Eval] fetched test_cases =", len(test_cases))
    test_cases = test_cases[:sample_size]
    print("[Eval] using test_cases   =", len(test_cases))

    # Initialize result containers
    results = {
        "vector": {"hit": [], "ndcg": []},
        "hybrid": {"hit": [], "ndcg": []},
    }

    for i, case in enumerate(test_cases, start=1):
        user_id = case["user_id"]
        target = case["target_product"]
        raw_text = (case.get("raw_review_text") or "").strip()
        if not raw_text:
            continue

        query_text = raw_text if not use_synthetic else generate_synthetic_query(raw_review_text=raw_text, client=client)
        emb = get_embedding(query_text, client)

        # Vector-only
        v_rows = run_search(driver, query_embedding=emb, k=top_k)
        v_items = [r.get("product_name") for r in v_rows if r.get("product_name")]
        v_m = calculate_metrics(v_items, target_item=target, k=top_k)
        results["vector"]["hit"].append(v_m["hit"])
        results["vector"]["ndcg"].append(v_m["ndcg"])

        # Hybrid
        loyalty = float(case.get("loyalty_score", 0.0) or 0.0)
        h_rows = run_hybrid_query(
            driver, query_embedding=emb, user_id=user_id, loyalty_score=loyalty, k=top_k
        )
        h_items = [r.get("product_name") for r in h_rows if r.get("product_name")]
        h_m = calculate_metrics(h_items, target_item=target, k=top_k)
        results["hybrid"]["hit"].append(h_m["hit"])
        results["hybrid"]["ndcg"].append(h_m["ndcg"])

        if i % 10 == 0:
            v_hit = sum(results["vector"]["hit"]) / max(1, len(results["vector"]["hit"]))
            h_hit = sum(results["hybrid"]["hit"]) / max(1, len(results["hybrid"]["hit"]))
            print(f"[Progress] {i}/{len(test_cases)} | Hit@{top_k}: vector={v_hit:.3f}, hybrid={h_hit:.3f}")

    # Aggregate
    v_hit = sum(results["vector"]["hit"]) / max(1, len(results["vector"]["hit"]))
    v_ndcg = sum(results["vector"]["ndcg"]) / max(1, len(results["vector"]["ndcg"]))
    h_hit = sum(results["hybrid"]["hit"]) / max(1, len(results["hybrid"]["hit"]))
    h_ndcg = sum(results["hybrid"]["ndcg"]) / max(1, len(results["hybrid"]["ndcg"]))

    print("=== FINAL EVAL ===")
    print(f"Hit@10:  vector={v_hit:.4f} | hybrid={h_hit:.4f} | diff={h_hit - v_hit:.4f}")
    print(f"NDCG@10: vector={v_ndcg:.4f} | hybrid={h_ndcg:.4f} | diff={h_ndcg - v_ndcg:.4f}")

    # Build payload for export (overwrite)
    payload = {
        "sample_size": sample_size,
        "top_k": top_k,
        "n_cases_used": len(results["vector"]["hit"]),  # actual evaluated cases (skips applied)
        "avg_vector_hit_rate_at_10": float(v_hit * 100.0),
        "avg_vector_ndcg_at_10": float(v_ndcg * 100.0),
        "avg_hybrid_hit_rate_at_10": float(h_hit * 100.0),
        "avg_hybrid_ndcg_at_10": float(h_ndcg * 100.0),
        "improvement_hit_rate_pp": float((h_hit - v_hit) * 100.0),
        "improvement_ndcg_pp": float((h_ndcg - v_ndcg) * 100.0),
    }

    # Save (overwrite) to reports/eval_quantitative.json
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "eval_quantitative.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    driver.close()


if __name__ == "__main__":
    main()
