# Independent rereview — UAT105 / TASK13260.46

## Verdict

Clear for this bounded correction. The original P2 is addressed; no new actionable finding in the reviewed scope.

## Resolution

`Summarization_General_Lib.py:364–375` now removes only well-formed flat reasoning blocks and rejects residual recognized reasoning delimiters. The original nested case can no longer expose the outer reasoning tail as final analysis: the remaining outer delimiter makes the result unusable, entering the existing safe failure contract. Nested, crossed, mismatched, unclosed and malformed delimiters also reject safely. Plain final text, uppercase well-formed tags, and multiple closed reasoning blocks with real final text remain supported.

Permanent adapter tests cover the original shape in both typed and legacy modes, asserting bounded errors and no synthetic sentinel in output/logs. The actual plaintext tests cover nested closed/unclosed and mismatched cases: source survives, status is Warning, top-level/chunk analysis is absent and the sentinel is not returned.

The existing real terminal integration controls remain unchanged and pass. They execute WorkerSDK → media worker → document persistence → actual summarizer with isolated provider/config seams, then read durable job/MediaDatabase/DocumentVersions state. Length-truncated output retains source with Warning and no analysis; the valid stop control retains source and expected analysis with Success. These are real persistence tests with a mocked provider, not a manufactured terminal-job result.

## Independent verification

Activated project `.venv`, then ran:

```sh
python -m pytest tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_plaintext_analysis_outcomes.py -q --tb=short
```

Result: **93 passed, 8 warnings in 3.90s**; no skips/failures. Log: `/private/tmp/cycle4-analysis-outcome-independent-rereview-tests.log`.

Scoped `git diff --check` passes. Reviewed updated implementation report, production correction, added negative/positive tests and retained plaintext/worker controls. The broader seven-file 120-test set passed in the prior independent review; this rereview reran only the two covering files for the correction.

## Limits

No live inference, browser/runtime/configuration changes, repository edits, staging or commits. No new recursive/multi-chunk or native acceptance claim. Conservative rejection of literal/malformed reasoning markup is intentional and documented; original source text is never normalized by this code. Author's current Bandit/Ruff baseline distinctions remain accurately documented; this read-only rereview did not rerun those tools.
