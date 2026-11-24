"""TASK-13950.24 PR E: the post-generation stages move into ``pipeline/post_processing.py``.

Spans are at this PR's base (1e65d8f5f1); the ``# ========== X ==========`` headers are the anchors.
"""

MANIFEST = {
    "base": "1e65d8f5f1",
    "moves": [
        {
            "name": "finish_answer",
            "module": "post_processing",
            "span": [8068, 8254],
            "doc": "Evidence-chain refresh, utility grading, user feedback, result highlighting, cost tracking, workspace filtering.",
        },
        {
            "name": "store_and_observe",
            "module": "post_processing",
            "span": [8256, 8337],
            "doc": "Cache storage, observability (retrieval metrics, faithfulness) and performance analysis.",
        },
        {
            "name": "finalize_run",
            "module": "post_processing",
            "span": [8366, 8569],
            "doc": "The pipeline's `finally` body: total time, metrics, observability and debug logging.",
            "noqa": [8407, 8548],
        },
    ],
}
