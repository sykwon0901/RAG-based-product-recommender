 # scripts/run_pipeline.py
 # Smoke-test entrypoint: verify existing graph/index/embeddings  retrieval sanity check.

from __future__ import annotations

import os
from getpass import getpass
from typing import Any, Dict, List, Optional

from amazon_skincare_graphrag.config import get_neo4j_driver, get_openai_client
from amazon_skincare_graphrag.embeddings import get_embedding
from amazon_skincare_graphrag.neo4j_io import init_neo4j_schema
from amazon_skincare_graphrag.retrieval import run_hybrid_query

def _check_counts(driver) -> Dict[str, int]:
    q = """
    MATCH (u:User)
    WITH count(u) AS users
    MATCH (p:Product)
    WITH users, count(p) AS products
    MATCH (r:Review)
    RETURN users, products, count(r) AS reviews
    """
    with driver.session() as s:
        r = s.run(q).single()
        return {
            "User": int(r["users"]),
            "Product": int(r["products"]),
            "Review": int(r["reviews"]),
        }


def _check_embedding_coverage(driver) -> Dict[str, Any]:
    q = """
    MATCH (r:Review)
    WITH count(r) AS total,
         sum(CASE WHEN r.embedding IS NULL THEN 0 ELSE 1 END) AS embedded
    RETURN total, embedded
    """
    with driver.session() as s:
        r = s.run(q).single()
        total = int(r["total"] or 0)
        embedded = int(r["embedded"] or 0)
        pct = (embedded / total) if total else 0.0
        return {"total": total, "embedded": embedded, "pct": pct}


def _check_vector_index(driver, index_name: str) -> Dict[str, Any]:
    q_show = """
    SHOW INDEXES
    YIELD name, type, state, populationPercent
    WHERE name = $name
    RETURN name, type, state, populationPercent
    """
    q_call = """
    CALL db.indexes()
    YIELD name, type, state, populationPercent
    WHERE name = $name
    RETURN name, type, state, populationPercent
    """
    with driver.session() as s:
        try:
            r = s.run(q_show, name=index_name).single()
        except Exception:
            r = s.run(q_call, name=index_name).single()

        if not r:
            return {"name": index_name, "type": None, "state": "MISSING", "populationPercent": None}
        return {
            "name": r["name"],
            "type": r["type"],
            "state": r["state"],
            "populationPercent": r["populationPercent"],
        }


def _get_fallback_embedding(driver) -> Optional[List[float]]:
    q = """
    MATCH (r:Review)
    WHERE r.embedding IS NOT NULL
    RETURN r.embedding AS emb
    LIMIT 1
    """
    with driver.session() as s:
        r = s.run(q).single()
        return r["emb"] if r else None




def main() -> None:
    driver = get_neo4j_driver()
    try:
        print(f"[OK] Neo4j connected: {os.getenv('NEO4J_URI')}")
        init_neo4j_schema(driver)

        # Counts
        counts = _check_counts(driver)
        print(f"[Check] counts: {counts}")

        # Embedding coverage
        cov = _check_embedding_coverage(driver)
        print(f"[Check] Review.embedding coverage: {cov['embedded']}/{cov['total']} ({cov['pct']:.1%})")

        # Index status
        index_name = os.getenv("VECTOR_INDEX_NAME", "review_embedding_index")
        idx = _check_vector_index(driver, index_name=index_name)
        print(f"[Check] vector index '{index_name}': state={idx.get('state')} population={idx.get('populationPercent')}")

        # Smoke retrieval (OpenAI key optional)
        query = os.getenv("PIPELINE_QUERY", "I need a soothing moisturizer for dry and sensitive skin.").strip()
        top_k = int(os.getenv("PIPELINE_TOP_K", "5"))

        embedding = None
        if os.getenv("OPENAI_API_KEY"):
            client = get_openai_client()
            embedding = get_embedding(query, client)
        else:
            embedding = _get_fallback_embedding(driver)

        if not embedding:
            raise RuntimeError(
                "No embedding available for smoke test. "
                "Either set OPENAI_API_KEY or ensure Review.embedding is populated."
            )

        rows = run_hybrid_query(driver, query_embedding=embedding, user_id=None, loyalty_score=0.0, k=top_k, index_name=index_name)
        print("\\n[Smoke Test] Top results:")
        for i, r in enumerate(rows[:top_k], start=1):
            print(f"{i}. {r.get('product_name')} | {r.get('brand_name')} | score={r.get('final_score', r.get('vector_score'))}")
    finally:
        driver.close()

if __name__ == "__main__":
    main()
