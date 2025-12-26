# Hybrid GraphRAG: Personalized Skincare Recommendation Engine

[![Open In Nbviewer](https://img.shields.io/badge/render-nbviewer-orange)](https://nbviewer.org/github/sykwon0901/amazon-skincare-graphrag/tree/main/)
[![Tech Stack](https://img.shields.io/badge/Stack-Neo4j%20|%20OpenAI%20|%20LangChain%20|%20Ragas-blue)]()

This repository demonstrates a **Hybrid RAG** approach for e-commerce recommendations, designed as a **dynamic alternative** to traditional recommenders (e.g., Collaborative Filtering) when you want:

- **Semantic retrieval** for relevance (vector search)
- **Graph-based signals** for structure + explainability (Neo4j Knowledge Graph)
- **A decoupled personalization layer** that can be tuned, A/B tested, and governed as policy

> Target use case: ambiguous product intent + user-specific preference signals (e.g., brand loyalty).

---

## 1) Executive Summary

### Objective
Build a Hybrid RAG architecture that **separates Retrieval from Personalization**:

- **Hybrid Search (Retrieval Layer):** Vector Search (semantic relevance) + Knowledge Graph signals (structural context)
- **Dynamic Re-ranking (Personalization Layer):** Log-weighted user affinity score (brand loyalty) applied on top of retrieved candidates

### Problem Statement
**Vector Search limitation (semantic ambiguity):** Similar reviews can come from irrelevant categories (e.g., “face cream” vs “foot cream”) when retrieval lacks structural constraints.

**Goal:** Use **graph structure** to reduce category mismatch and add **dynamic re-ranking** to boost ranking quality (Hit Rate, NDCG).

### Data & Stack
- **Data:** Amazon Facial Skincare Reviews (subset of Amazon Reviews 2023; Hou et al.)
- **Tech stack:** Neo4j (graph + native vector index), OpenAI (embeddings/LLM), LangChain, Ragas

---

## 2) Key Results

**Question:** Does adding a Knowledge Graph improve recommendation accuracy?

The Hybrid RAG architecture outperforms the Vector Search baseline by combining structural retrieval with dynamic personalization.

### Quantitative Results (N=50)
| Metric      | Vector Only | Hybrid RAG | Improvement  |
|             | (Baseline)  | (Proposed) | Improvement  |
|-------------|------------:|-----------:|-------------:|
| Hit Rate@10 |      21.51% | **41.94%** | **+20.43%p** |
| NDCG@10     |      14.64% | **30.92%** | **+16.28%p** |
> Note: Metrics may vary across runs due to random sampling (unless a fixed seed/split is enforced).

**Interpretation**
- **Structural boost:** graph context reduces category mismatches (semantic ambiguity)
- **Affinity boost:** re-ranking acts as a personalized tie-breaker, pushing preferred brands upward

### Qualitative Results (Ragas + GPT Judge)
- **Answer Relevancy:** 0.8187
- **Faithfulness:** 0.7619
- **Context Precision:** 0.2560
- **Ragas run success rate:** 100.0% (84/84)
> Note: Metrics may vary across runs due to random sampling (unless a fixed seed/split is enforced).
---

## 3) Why This Matters for AI Transformation Teams

### A) Decoupled Personalization = Governance + Speed
Brand loyalty is **not hard-coded into the graph schema**. It is applied as a **policy layer** (re-ranking) on top of retrieved candidates.
- Easier to **tune** and **A/B test**
- Easier to implement **country/channel-specific rules**
- Easier to align with **business constraints** (e.g., margin, inventory, compliance)

### B) Explainability as a first-class feature
Graph paths + history_count provide human-auditable rationale:
- “Recommended because you have prior purchases/reviews for this brand line”
This is valuable when stakeholders need transparency beyond embedding similarity.

### C) Operational simplicity via a unified engine
Neo4j hosts both:
- **Persistent ANN vector index** (Lucene-based)
- **Knowledge Graph traversal**
This avoids synchronization overhead between a graph DB and an external vector DB.

---

## 4) Method & Architecture

### Retrieval + Re-ranking
1. **Candidate generation:** vector search over review embeddings
2. **Re-ranking:** apply a log-weighted user-brand affinity boost

**Scoring (conceptual):**
\[
FinalScore = VectorScore \times (1 + \log(1 + HistoryCount) \times w)
\]

### Cold Start vs Personalized
- **Cold start:** vector-only (or hybrid retrieval without loyalty boost)
- **Personalized:** dynamic boost based on user history

---

## 5) Evaluation Design (Anti-Leakage + Reliability)

### Anti-Leakage
To avoid trivial matching between reviews and queries, the pipeline uses **LLM-based paraphrasing** to convert raw review text into **generic user intent queries**.

### Metrics
- **Quantitative:** Hit Rate@10, NDCG@10  
- **Qualitative:** Ragas (faithfulness, relevancy, context precision) with GPT judge

### Reliability (“Safe Mode”)
Evaluation calls can fail due to timeouts / rate limits. This repo includes a **fail-soft** evaluation mode that logs failures and reports success rate.

---

## 6) Demo

### Notebooks
- `notebooks/00_demo_end_to_end.ipynb`  
  End-to-end demo with a small set of final outputs (Top-K comparison + rationale + metrics)
- `notebooks/01_pipeline_clean.ipynb`  
  Clean reproducible pipeline (minimal outputs)

### Streamlit
A Streamlit demo compares **Vector-only vs Hybrid RAG** side-by-side and show:
- Ranking differences
- “Why it changed” (history_count / loyalty signal)
- Latency

### Live demo (on request)
Not publicly deployed due to paid API usage (OpenAI) and private Neo4j connectivity.
Available upon request.
---

## 7) How to Run

### 7-1)Install
```bash
pip install -r requirements.txt

```bash
python -m pip install -e .

### 7-2) Environment setup (.env)

### Environment (.env)
Copy and edit:
```bash
cp .env.example .env

### 7-3) Run scripts (reproducible entry point)

### Run: pipeline verification (no re-loading by default)
```bash
python scripts/run_pipeline.py

### 7-4) Run Streamlit demo
python -m streamlit run app/streamlit_app.py

### 7-5) Cost controls (OpenAI Cost)

### Cost controls
`run_eval.py` may call OpenAI for:
- synthetic query generation
- embedding

To reduce cost:
- Set `USE_SYNTHETIC_QUERY=0` in `.env`
- Reduce `EVAL_SAMPLE_SIZE` (default: 50)

## 7-6)Reports (overwrite policy)
All exports are written to `reports/` root and overwrite existing files (no date folders).

- `eval_quantitative.json`
- `interactive_demo_log.jsonl`
- `ragas_export.xlsx` (sheets: raw / clean / top3)
- `ragas_report.txt`