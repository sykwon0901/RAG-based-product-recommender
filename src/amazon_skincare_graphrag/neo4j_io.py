# src/amazon_skincare_graphrag/neo4j_io.py

from __future__ import annotations

from typing import Any, Dict, List, Optional


def init_neo4j_schema(driver) -> None:
    """
    Initialize schema constraints used by the notebook queries.
    Safe to re-run.
    Note: Use unnamed constraints to avoid name collisions with constraints created in notebooks.

    """
    cyphers = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (b:Brand) REQUIRE b.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Product) REQUIRE p.asin IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Review) REQUIRE r.id IS UNIQUE",
    ]
    with driver.session() as session:
        for q in cyphers:
            session.run(q)


def get_reviews_without_embeddings(driver, limit: int = 1000):
    query = """
    MATCH (r:Review)
    WHERE r.embedding IS NULL
    RETURN r.id AS review_id, r.text AS text
    LIMIT $limit
    """
    with driver.session() as session:
        return [record.data() for record in session.run(query, limit=limit)]


def update_review_embeddings(driver, rows):
    query = """
    UNWIND $rows AS row
    MATCH (r:Review {id: row.review_id})
    SET r.embedding = row.embedding
    """
    with driver.session() as session:
        session.run(query, rows=rows)


# Optional loaders (keep minimal; wire later to your CSVs if needed)
def import_products(driver, products: List[Dict[str, Any]]) -> None:
    """
    products: list of dicts with at least {asin, title, brand_name, category(optional), features(optional)}
    """
    query = """
    UNWIND $rows AS row
    MERGE (p:Product {asin: row.asin})
    SET p.title = row.title,
        p.features = coalesce(row.features, p.features),
        p.category = coalesce(row.category, p.category)
    MERGE (b:Brand {name: row.brand_name})
    MERGE (p)-[:MADE_BY]->(b)
    """
    with driver.session() as session:
        session.run(query, rows=products)


def import_reviews(driver, reviews: List[Dict[str, Any]]) -> None:
    """
    reviews: list of dicts with at least {review_id, user_id, asin, text, timestamp(optional)}
    """
    query = """
    UNWIND $rows AS row
    MERGE (u:User {user_id: row.user_id})
    MERGE (p:Product {asin: row.asin})
    MERGE (r:Review {id: row.review_id})
    SET r.text = row.text,
        r.timestamp = coalesce(row.timestamp, r.timestamp)
    MERGE (u)-[:WROTE]->(r)
    MERGE (r)-[:EVALUATED]->(p)
    """
    with driver.session() as session:
        session.run(query, rows=reviews)
