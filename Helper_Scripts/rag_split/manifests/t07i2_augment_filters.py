"""TASK-13950.24 PR I2: multi-vector passages, the numeric boost, gap analysis and the keyword filter move into
``pipeline/augment.py``; instruction-injection, chunk-type and content-policy filters and the security filter move
into ``pipeline/filters.py``. Spans are at this PR's base (ecf95947f6); the ``# ========== X ==========`` headers
are the anchors.
"""

MANIFEST = {
    "base": "ecf95947f6",
    "moves": [
        {"name": "augment_passages", "module": "augment", "span": [4482, 4603],
         "doc": "Multi-vector passages and the numeric/table-aware boost (pre-rerank)."},
        {"name": "run_gap_analysis", "module": "augment", "span": [4606, 4784],
         "doc": "Gap analysis and follow-ups, then the keyword filter."},
        {"name": "apply_content_filters", "module": "filters", "span": [4787, 4913],
         "doc": "Instruction-injection filtering, the chunk-type filter and content policy sanitation."},
        {"name": "apply_security_filter", "module": "filters", "span": [4916, 5054],
         "doc": "Security filtering by sensitivity level, recording its own failures."},
    ],
}
