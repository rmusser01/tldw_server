"""TASK-13950.24 PR K: source scoping, query preparation and the research loop move into ``pipeline/scope.py``,
``pipeline/query_prep.py`` and ``pipeline/research.py``.
Spans are at this PR's base (0ce815b0b9); the ``# ========== X ==========`` headers are the anchors.
"""

MANIFEST = {
    "base": "0ce815b0b9",
    "moves": [
        {"name": "resolve_sources", "module": "scope", "span": [2660, 2735],
         "doc": "Resolve data sources and explicit source selection; raises EarlyReturn on an invalid source."},
        {"name": "clarify_and_spellcheck", "module": "query_prep", "span": [2739, 2792],
         "doc": "Pre-retrieval clarification (may raise EarlyReturn) and spell check."},
        {"name": "run_query_classification", "module": "query_prep", "span": [2794, 3021], "noqa": [2986, 3020],
         "doc": "Query classification and reformulation, with the routed external-docs prefetch."},
        {"name": "run_research_loop", "module": "research", "span": [3026, 3246], "noqa": [3244],
         "doc": "Agentic research loop."},
        {"name": "apply_guardrail_defaults", "module": "query_prep", "span": [3251, 3314],
         "doc": "Env-based production defaults and generation guardrails."},
        {"name": "run_query_expansion", "module": "query_prep", "span": [3317, 3426],
         "doc": "Query expansion, scoped to the user's cache namespace."},
        {"name": "route_and_weight", "module": "query_prep", "span": [3429, 3546],
         "doc": "Intent routing, dynamic granularity and intent-based weighting."},
        {"name": "prepare_hyde_and_temporal_filters", "module": "query_prep", "span": [3549, 3674],
         "doc": "HyDE preparation and automatic temporal filters."},
    ],
}
