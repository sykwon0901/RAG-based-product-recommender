from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from amazon_skincare_graphrag.config import get_neo4j_driver
from amazon_skincare_graphrag.embeddings import get_embedding
from amazon_skincare_graphrag.retrieval import run_search, run_hybrid_query, analyze_user_loyalty


def load_env() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=repo_root / ".env", override=True)


@st.cache_resource
def get_clients():
    load_env()

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env or export it in your shell.")

    client = OpenAI(api_key=openai_key)
    driver = get_neo4j_driver()
    return client, driver


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = (time.perf_counter() - t0) * 1000.0
    return out, dt


def add_rank(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for i, r in enumerate(rows or [], start=1):
        rr = dict(r)
        rr["Rank"] = i
        out.append(rr)
    return out


def main() -> None:
    st.set_page_config(page_title="Amazon Skincare Hybrid RAG Demo", layout="wide")

    # Styles
    st.markdown(
        """
        <style>
        /* Sidebar background */
        [data-testid="stSidebar"] {
            background-color: #000000;
        }

        /* Sidebar text color */
        [data-testid="stSidebar"] * {
            color: #ffffff;
        }

        /* Inputs background in sidebar */
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] .stSelectbox,
        [data-testid="stSidebar"] .stSlider,
        [data-testid="stSidebar"] .stTextInput,
        [data-testid="stSidebar"] .stTextArea {
            background-color: #111111 !important;
            color: #ffffff !important;
        }

        /* Checkbox label color */
        [data-testid="stSidebar"] label {
            color: #ffffff !important;
        }

        /* Info/alert boxes: make background black */
        div[data-testid="stAlert"],
        div[data-testid="stAlert"] > div,
        div.stAlert,
        div.stAlert > div {
            background: #000000 !important;
            color: #ffffff !important;
            border: 1px solid #222222 !important;
        }

        /* Ensure all text inside is white */
        div[data-testid="stAlert"] *,
        div.stAlert * {
            color: #ffffff !important;
        }

        /* Slightly darken dataframe container */
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Amazon Skincare Recommender by Hybrid RAG")
    st.caption("Vector-only vs Hybrid RAG(Graph & Vector) side-by-side")

    client, driver = get_clients()

    with st.sidebar:
        query = st.text_area(
            "Query",
            value="I need a soothing moisturizer for dry and sensitive skin.",
            height=90,
        )

        user_id = st.text_input(
            "User ID (optional)",
            value="AEXCLMGS3Y7SRW5CMLJBNYI2HBZQ",
        )

        top_k = st.slider("Top K", min_value=3, max_value=20, value=10, step=1)
        exclude_purchased = st.checkbox("Exclude purchased (only if user_id is set)", value=False)

        run_btn = st.button("Run", type="primary")

        st.caption("Try inputting one of these user ids for demo")
        st.text("AESTJLOI5LB2WRFCVBXCXURB2FHA")
        st.text("AGLCYFK2SZYTSYVMHA744ZC6AXBQ")
        st.text("AGAHANLSS7DG4ZHNPP5S56W4SKHA")
        st.text("AFNRGRORFAGSB7WGANETBDDYDEAQ")
        st.text("AE6MKX6TWGESBF2HBPH6F7YQPFWQ")

    if not query.strip():
        st.warning("Query is empty.")
        return

    if not run_btn:
        st.info("Showing default results. Edit the query (and optional user_id), then click Run to refresh.")

    user_id_val: Optional[str] = user_id.strip() or None

    # 1) Embed query
    q_emb, emb_ms = timed(get_embedding, query, client)

    # 2) Vector-only
    vec_rows, vec_ms = timed(run_search, driver, query_embedding=q_emb, k=top_k)

    # 3) Hybrid (Dynamic: brand_weight = loyalty_score * 0.5)
    loyalty_score = 0.0
    top_brand = None
    purchased_asins = []

    if user_id_val:
        loyalty_score, top_brand, purchased_asins = analyze_user_loyalty(driver, user_id_val)

    brand_weight = float(loyalty_score) * 0.5

    hyb_rows, hyb_ms = timed(
        run_hybrid_query,
        driver,
        query_embedding=q_emb,
        user_id=user_id_val,
        loyalty_score=loyalty_score,
        k=top_k,
        exclude_purchased=exclude_purchased,
        brand_weight=brand_weight,
    )

    # Display
    st.write(
        f"Embedding latency: {emb_ms:,.1f} ms | "
        f"Vector latency: {vec_ms:,.1f} ms | "
        f"Hybrid latency: {hyb_ms:,.1f} ms"
    )

    if user_id_val:
        st.write(
            f"User: `{user_id_val}` | top_brand: `{top_brand}` | "
            f"loyalty_score: {loyalty_score:.3f} | brand_weight: {brand_weight:.3f} | "
            f"purchased_asins: {len(purchased_asins)}"
        )

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Vector-only")
        df = pd.DataFrame(add_rank(vec_rows))
        show_cols = [c for c in ["Rank", "product_name", "vector_score", "brand_name", "asin"] if c in df.columns]
        st.dataframe(
            df[show_cols] if len(df) and show_cols else df,
            use_container_width=True,
            hide_index=True,
            height=420,
        )

    with col2:
        st.subheader("Hybrid RAG(Vector & Graph)")
        df = pd.DataFrame(add_rank(hyb_rows))
        show_cols = [
            c
            for c in ["Rank", "product_name", "final_score", "brand_name", "asin", "vector_score", "history_count"]
            if c in df.columns
        ]
        st.dataframe(
            df[show_cols] if len(df) and show_cols else df,
            use_container_width=True,
            hide_index=True,
            height=420,
        )

    st.divider()


if __name__ == "__main__":
    main()
