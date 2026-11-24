"""TASK-13950.24 PR J: the retrieval cache lookup and document retrieval move into ``pipeline/retrieval.py``.
Spans are at this PR's base (2c10b2a6f3); the ``# ========== X ==========`` headers are the anchors.
"""

MANIFEST = {
    "base": "2c10b2a6f3",
    "moves": [
        {"name": "lookup_retrieval_cache", "module": "retrieval", "span": [3677, 3737],
         "doc": "Exact then semantic retrieval cache lookup, scoped by the per-owner cache namespace."},
        {"name": "prepare_retriever", "module": "retrieval", "span": [3766, 3906],
         "doc": "Build the multi-database retriever and config, then run the base retrieval."},
        {"name": "merge_hyde_and_variants", "module": "retrieval", "span": [3912, 4056],
         "doc": "Media DB fallback, HyDE merge and query-variant expansion retrieval."},
        {"name": "run_prf", "module": "retrieval", "span": [4061, 4116],
         "doc": "Optional PRF second-pass retrieval to fill remaining slots."},
        {"name": "run_decomposition", "module": "retrieval", "span": [4119, 4319],
         "doc": "Guided query decomposition with bounded concurrent sub-query retrieval."},
        {"name": "record_retrieval_diagnostics", "module": "retrieval", "span": [4321, 4371],
         "doc": "Workspace filtering, retrieval guidance, sources and timing metrics."},
        {"name": "run_retrieval", "module": "retrieval", "span": [3740, 4468], "noqa": [4378],
         "doc": "Document retrieval under the rag.retrieval span."},
    ],
}
