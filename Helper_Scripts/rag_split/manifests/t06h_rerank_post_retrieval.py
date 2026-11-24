"""TASK-13950.24 PR H: reranking moves into ``pipeline/rerank.py``; the web fallback, source explanations,
citations and evidence chains move into ``pipeline/post_retrieval.py``. Spans are at this PR's base
(2cdbd180eb); the ``# ========== X ==========`` headers are the anchors.
"""

MANIFEST = {
    "base": "2cdbd180eb",
    "moves": [
        {"name": "select_reranker", "module": "rerank", "span": [5659, 5797],
         "doc": "Pick the strategy, resolve the LLM reranker credentials and build the reranking config."},
        {"name": "apply_rerank", "module": "rerank", "span": [5798, 5879], "noqa": [5823],
         "doc": "Run the reranker (two-tier degrades to hybrid on failure) and record snapshots and timings."},
        {"name": "calibrate_rerank", "module": "rerank", "span": [5881, 5968],
         "doc": "Record the reranker's calibration metadata or compute a learned-fusion probability."},
        {"name": "run_rerank", "module": "rerank", "span": [5627, 6001],
         "doc": "Reranking under the rag.rerank span."},
        {"name": "run_web_fallback", "module": "post_retrieval", "span": [6005, 6084],
         "doc": "Self-correcting RAG web search fallback."},
        {"name": "explain_and_expand_sources", "module": "post_retrieval", "span": [6087, 6235],
         "doc": "Why-these-sources metadata, sibling inclusion and the document cap."},
        {"name": "build_citations_and_chains", "module": "post_retrieval", "span": [6237, 6440],
         "doc": "Citations, knowledge strips and evidence chains."},
    ],
}
