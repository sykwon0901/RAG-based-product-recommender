# src/amazon_skincare_graphrag/eval.py

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


def calculate_metrics(recommended_items: List[str], target_item: str, k: int = 10) -> Dict[str, float]:
    """
    Compute Hit@k and NDCG@k for a single test case.
    recommended_items: list of product titles or asins
    """
    topk = recommended_items[:k]
    hit = 1.0 if target_item in topk else 0.0

    if hit == 0.0:
        return {"hit": 0.0, "ndcg": 0.0}

    rank = topk.index(target_item) + 1
    ndcg = 1.0 / math.log2(rank + 1)
    return {"hit": hit, "ndcg": ndcg}


def fetch_test_dataset(driver, sample_size: int = 50) -> List[Dict[str, Any]]:
    """
    Fetch 'Core Users' and their 'Last Review' to create a Ground Truth dataset.

    Returns rows with:
    - user_id
    - target_product (Product.title)
    - raw_review_text (Review.text)
    - loyalty_score (top-brand share for the user)
    """
    query = """
    MATCH (u:User)-[:WROTE]->(r:Review)-[:EVALUATED]->(p:Product)
    WITH u, count(r) AS review_count
    WHERE review_count >= 5
    WITH u
    ORDER BY rand() LIMIT $limit

    MATCH (u)-[:WROTE]->(r:Review)-[:EVALUATED]->(target_p:Product)-[:MADE_BY]->(target_b:Brand)
    WITH u, target_p, target_b, r
    ORDER BY r.timestamp DESC
    WITH u,
         head(collect(target_p.title)) AS target_product,
         head(collect(r.text)) AS raw_review_text,
         head(collect(target_b.name)) AS target_brand

    MATCH (u)-[:WROTE]->(:Review)-[:EVALUATED]->(p2:Product)-[:MADE_BY]->(b2:Brand)
    WITH u, target_product, raw_review_text, count(p2) AS total, b2, count(b2) AS b_count
    ORDER BY b_count DESC
    WITH u, target_product, raw_review_text, total,
         head(collect(b2.name)) AS top_brand,
         head(collect(b_count)) AS top_count

    RETURN u.user_id AS user_id,
           target_product,
           raw_review_text,
           (CASE WHEN total = 0 THEN 0.0 ELSE (toFloat(top_count) / toFloat(total)) END) AS loyalty_score
    """
    with driver.session() as session:
        return [rec.data() for rec in session.run(query, limit=sample_size)]


def get_existing_col(candidates: List[str], columns: List[str]) -> Optional[str]:
    for c in candidates:
        if c in columns:
            return c
    return None
