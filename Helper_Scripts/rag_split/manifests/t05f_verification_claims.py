"""TASK-13950.24 PR F: citations, claims, numeric fidelity and post-verification move into
``pipeline/verification.py`` and ``pipeline/claims.py``. Spans are at this PR's base (6d4f1e3f03).
"""

MANIFEST = {
    "base": "6d4f1e3f03",
    "moves": [
        {"name": "check_citations_and_groundedness", "module": "verification", "span": [7024, 7116],
         "doc": "Hard and quote-level citations, the fast groundedness check and the claims-skip flag."},
        {"name": "run_numeric_fidelity", "module": "verification", "span": [7551, 7701],
         "doc": "Verify the numeric tokens of the answer against the sources."},
        {"name": "verify_answer", "module": "verification", "span": [7714, 7816],
         "doc": "Run the post-generation verifier and record its outcome."},
        {"name": "adaptive_rerun", "module": "verification", "span": [7820, 8042], "allow_locals": True,
         "doc": "Re-run the pipeline with adapted settings and keep the better answer."},
        {"name": "run_post_verification", "module": "verification", "span": [7703, 8067],
         "doc": "Post-generation verification with an optional adaptive rerun."},
        {"name": "prepare_claims_job", "module": "claims", "span": [7201, 7339],
         "doc": "Build the claims engine and its closures."},
        {"name": "execute_claims", "module": "claims", "span": [7341, 7521],
         "doc": "Extract, verify and report claims."},
        {"name": "run_claims", "module": "claims", "span": [7117, 7549],
         "doc": "Claims and factuality stage."},
    ],
}
