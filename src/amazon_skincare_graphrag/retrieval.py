# src/amazon_skincare_graphrag/retrieval.py

from __future__ import annotations

from typing import Any, Dict, List, Optional


def run_search(
    driver,
    query_embedding: List[float],
    k: int = 10,
    index_name: str = "review_embedding_index",
) -> List[Dict[str, Any]]:
    """
    Vector-only retrieval: returns product candidates from similar reviews.
    """
    query = f"""
    CALL db.index.vector.queryNodes('{index_name}', $k, $embedding)
    YIELD node AS similar_review, score AS vector_score

    MATCH (similar_review)-[:EVALUATED]->(product:Product)-[:MADE_BY]->(brand:Brand)

    RETURN product.title AS product_name,
           product.asin AS asin,
           brand.name AS brand_name,
           vector_score
    ORDER BY vector_score DESC
    LIMIT $k
    """
    with driver.session() as session:
        res = session.run(query, k=k, embedding=query_embedding)
        return [r.data() for r in res]


def analyze_user_loyalty(driver, user_id: str) -> tuple[float, Optional[str], List[str]]:
    """
    Returns (loyalty_score, top_brand, purchased_asins)
    """
    query = """
    MATCH (u:User {user_id: $user_id})-[:WROTE]->(:Review)-[:EVALUATED]->(p:Product)-[:MADE_BY]->(b:Brand)
    WITH b.name AS brand, count(*) AS cnt
    ORDER BY cnt DESC
    WITH collect({brand: brand, cnt: cnt}) AS stats

    WITH stats,
         reduce(total = 0, x IN stats | total + x.cnt) AS total_cnt,
         CASE WHEN size(stats) > 0 THEN stats[0].brand ELSE NULL END AS top_brand,
         CASE WHEN size(stats) > 0 THEN stats[0].cnt ELSE 0 END AS top_cnt

    MATCH (u:User {user_id: $user_id})-[:WROTE]->(:Review)-[:EVALUATED]->(p2:Product)
    RETURN
      CASE WHEN total_cnt = 0 THEN 0.0 ELSE toFloat(top_cnt) / toFloat(total_cnt) END AS loyalty_score,
      top_brand AS top_brand,
      collect(DISTINCT p2.asin) AS purchased_asins
    """
    with driver.session() as session:
        rec = session.run(query, user_id=user_id).single()
        if not rec:
            return 0.0, None, []
        d = rec.data()
        return float(d["loyalty_score"]), d["top_brand"], d["purchased_asins"] or []


def get_user_last_review_text(driver, user_id: str) -> Optional[str]:
    query = """
    MATCH (u:User {user_id: $user_id})-[:WROTE]->(r:Review)
    RETURN r.text AS text
    ORDER BY r.timestamp DESC
    LIMIT 1
    """
    with driver.session() as session:
        rec = session.run(query, user_id=user_id).single()
        return rec["text"] if rec else None


def run_hybrid_query(
    driver,
    query_embedding: List[float],
    user_id: Optional[str] = None,
    loyalty_score: float = 0.0,
    k: int = 5,
    exclude_purchased: bool = False,
    return_context_string: bool = False,
    index_name: str = "review_embedding_index",
    brand_weight: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid RAG retrieval:
    - Vector candidates from review embeddings
    - Optional re-ranking boost using user-brand history (history_count)
    """
    # Match notebook logic: dynamic weight from loyalty_score
    if brand_weight is None:
        brand_weight = float(loyalty_score) * 0.5

    filter_clause = ""
    if exclude_purchased and user_id:
        filter_clause = """
        WHERE NOT EXISTS {
            MATCH (u2:User {user_id: $user_id})-[:WROTE]->(:Review)-[:EVALUATED]->(product)
        }
        """

    query = f"""
    // 1) Vector search
    CALL db.index.vector.queryNodes('{index_name}', 150, $embedding)
    YIELD node AS similar_review, score AS vector_score

    // 2) Graph traversal
    MATCH (similar_review)-[:EVALUATED]->(product:Product)-[:MADE_BY]->(brand:Brand)
    {filter_clause}

    // 3) Personalization signal
    OPTIONAL MATCH (u:User {{user_id: $user_id}})-[:WROTE]->(:Review)-[:EVALUATED]->(:Product)-[:MADE_BY]->(brand)
    WITH product, brand, similar_review, vector_score, count(u) AS history_count

    // 4) Re-ranking
    WITH product, brand, similar_review, vector_score, history_count,
         (vector_score * (1 + (log(1 + history_count) * $brand_weight))) AS final_score

    ORDER BY final_score DESC
    LIMIT $k

    RETURN product.title AS product_name,
           product.asin AS asin,
           brand.name AS brand_name,
           similar_review.text AS review_text,
           vector_score,
           history_count,
           final_score
    """

    params = {
        "embedding": query_embedding,
        "user_id": user_id,
        "k": k,
        "brand_weight": brand_weight,
    }

    with driver.session() as session:
        rows = [r.data() for r in session.run(query, **params)]

    if return_context_string:
        # Optional: return a compact context string for LLM prompting
        for r in rows:
            r["context"] = (r.get("review_text") or "")[:500]
    return rows


