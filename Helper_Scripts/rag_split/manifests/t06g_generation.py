"""TASK-13950.24 PR G: answer generation moves into ``pipeline/generation.py``. Spans are at this PR's base
(bc8818374e); the ``# ========== ANSWER GENERATION ==========`` header is the anchor.
"""

MANIFEST = {
    "base": "bc8818374e",
    "moves": [
        {"name": "prepare_generation", "module": "generation", "span": [6441, 6496],
         "doc": "Calibration gating, the effective enable flag and the retrieval-only result."},
        {"name": "synthesize_multi_turn", "module": "generation", "span": [6616, 6791],
         "doc": "Draft, critique and refine the answer within a time budget."},
        {"name": "generate_standard_answer", "module": "generation", "span": [6794, 6860],
         "doc": "Run the generation phase and fold its result into the pipeline result."},
        {"name": "finish_generation", "module": "generation", "span": [6861, 6922], "noqa": [6921],
         "doc": "Timings, metrics, the span attribute and follow-up suggestions."},
        {"name": "answer_when_gated", "module": "generation", "span": [6956, 7024],
         "doc": "Record the gate, apply the abstention policy."},
        {"name": "run_generation", "module": "generation", "span": [6498, 7024],
         "doc": "Answer generation under the rag.generation span, or the gated path."},
    ],
}
