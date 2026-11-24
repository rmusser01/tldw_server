"""TASK-13950.24 PR I1: table/VLM processing and evidence accumulation move into ``pipeline/processing.py``;
document grading and the query-rewriting loop move into ``pipeline/self_correct.py``. Spans are at this PR's
base (a8c9583a77); the ``# ========== X ==========`` headers are the anchors.
"""

MANIFEST = {
    "base": "a8c9583a77",
    "moves": [
        {"name": "process_tables_and_vlm", "module": "processing", "span": [5055, 5235],
         "doc": "Table processing and VLM late chunking of the retrieved documents."},
        {"name": "run_evidence_accumulation", "module": "processing", "span": [5238, 5343],
         "doc": "Progressive evidence accumulation over the retrieved documents."},
        {"name": "grade_documents", "module": "self_correct", "span": [5358, 5445],
         "doc": "Self-correcting RAG stage 1: grade documents for relevance."},
        {"name": "run_rewrite_loop", "module": "self_correct", "span": [5449, 5626],
         "doc": "Self-correcting RAG stage 2: rewrite the query and retry retrieval on low relevance."},
    ],
}
