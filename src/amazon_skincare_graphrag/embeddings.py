# src/amazon_skincare_graphrag/embeddings.py

from __future__ import annotations

import os
from typing import List, Optional


def get_embedding(text: str, client, model: Optional[str] = None) -> List[float]:
    """
    Return an embedding vector for a given text.
    """
    model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    text = (text or "").strip()
    if not text:
        return []

    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding


def create_vector_index(
    driver,
    index_name: str = "review_embedding_index",
    label: str = "Review",
    property_name: str = "embedding",
    dims: int = 1536,
    similarity: str = "cosine",
) -> None:
    """
    Create Neo4j vector index (Neo4j 5+). Safe to re-run.
    """
    query = f"""
    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
    FOR (n:{label})
    ON (n.{property_name})
    OPTIONS {{
      indexConfig: {{
        `vector.dimensions`: {dims},
        `vector.similarity_function`: '{similarity}'
      }}
    }}
    """
    with driver.session() as session:
        session.run(query)


def generate_embeddings(
    texts: List[str],
    client,
    model: Optional[str] = None,
) -> List[List[float]]:
    """
    Batch embedding (simple loop; optimize later if needed).
    """
    return [get_embedding(t, client=client, model=model) for t in texts]