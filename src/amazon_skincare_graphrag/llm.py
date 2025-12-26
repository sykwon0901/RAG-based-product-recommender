# src/amazon_skincare_graphrag/llm.py

from __future__ import annotations

import os
from typing import List, Optional


CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def generate_synthetic_query(raw_review_text: str, client, model: Optional[str] = None) -> str:
    """
    Paraphrase a raw review into a generic user intent query (anti-leakage).
    """
    model = model or CHAT_MODEL
    prompt = f"""
Convert the following product review into a generic e-commerce search query.
Do not mention brand names, ASINs, or unique product identifiers.

Review:
{raw_review_text}

Output: one concise user query sentence.
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return (resp.choices[0].message.content or "").strip()


def generate_answer(user_query: str, contexts: List[str], client, model: Optional[str] = None) -> str:
    """
    Generate a recommendation answer grounded in retrieved contexts.
    """
    model = model or CHAT_MODEL
    context_block = "\n".join([f"- {c}" for c in contexts if c])
    prompt = f"""
User Query: {user_query}

Contexts:
{context_block}

Based only on the contexts, recommend the best product(s) and explain briefly.
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return (resp.choices[0].message.content or "").strip()


def generate_ragas_answer(user_query: str, contexts: List[str], client, model: Optional[str] = None) -> str:
    """
    Thin wrapper for Ragas answer generation.
    """
    return generate_answer(user_query=user_query, contexts=contexts, client=client, model=model)
