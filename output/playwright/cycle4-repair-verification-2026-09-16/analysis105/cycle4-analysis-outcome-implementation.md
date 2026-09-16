# Cycle4 Task3 — UAT105 / TASK13260.46 implementation

Ready for independent review; task remains In Progress pending review and native verification.

## Scope and behavior

Production change only in `tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py`.

- Removes `extract_response_content(response) or str(response)` envelope serialization.
- Non-streaming analysis requires nonblank string answer text. Unsupported content types, malformed/empty responses and reasoning-only output use the existing safe failure contract.
- Removes only well-formed flat tagged reasoning (`think`, `reason`, `reasoning`, `thought`) from analysis text. Any remaining reasoning delimiter makes the answer ambiguous and is rejected safely, including nested, mismatched and unclosed blocks. Valid text following a well-formed closed block remains the answer.
- Rejects the selected completion's `finish_reason: length`, including nonempty partial text, with an explicit legacy truncation warning. Typed callers receive sanitized `SummaryProviderError(provider_failure)`.
- No provider response/details are interpolated into failure messages or logs. Shared Chat extractor and streaming behavior were not modified.
- Existing plaintext caller preserves source, omits bad analysis/chunk analysis, and returns Warning. Real WorkerSDK terminal persistence preserves that Warning and source; valid controls retain Success and stored analysis.

Files changed (plus official Backlog record):
1. `tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py`
2. `tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py`
3. `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_plaintext_analysis_outcomes.py`

## Permanent tests and RED/GREEN

New tests exercise actual `analyze`, not an AST reconstruction. Lower provider/config seams are replaced; no inference/network is used. Existing adapter credentials, typed/legacy exceptions, and streaming controls remain in the same suite.

17 invalid cases in both legacy/typed modes: null/empty/whitespace responses, malformed/missing choices/message, null/blank/object/list content, reasoning-only structured or closed/unclosed tagged strings, empty/nonempty length truncation. Positive plain text, normal envelope with separate reasoning, and tagged reasoning followed by final text controls. Sentinel must not appear in return/exception or captured Loguru output.

Three real summarizer→plaintext failure cases preserve source, warnings and null chunk analysis. Two integration controls execute real WorkerSDK→media worker→document persistence→plaintext→summarizer→temporary Media DB; read durable terminal job result and DocumentVersions analysis. Length outcome: completed orchestration + Warning, source intact, no stored analysis. Stop outcome: completed + Success with expected analysis. Provider sentinel/envelope absent from job/media/version data.

First permanent-test RED before production edits: **32 failed, 36 passed**, `/private/tmp/cycle4-uat105-red.log`.

Final permanent-test RED replay with original HEAD summarizer, restored in `finally`: **33 failed, 37 passed**, `/private/tmp/cycle4-uat105-confirmed-red-final.log`. This includes real terminal-job truncation returning Success against the old implementation; positive job control passed. This replay also verifies corrected integration setup, after initial test-only staging, isolated credential lookup and DB read-interface mistakes were corrected. Those setup failures were not counted as product RED evidence.

Final expanded GREEN: **120 passed, 8 warnings in 8.00s**, `/private/tmp/cycle4-uat105-final-tests.log`.

Exact command (after `source .venv/bin/activate`):

```sh
python -m pytest \
  tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py \
  tldw_Server_API/tests/LLM_Calls/test_summarization_runtime_credentials.py \
  tldw_Server_API/tests/LLM_Calls/test_local_summarization_config.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_plaintext_analysis_outcomes.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_chunk_consistency.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_metadata_contract.py \
  -q --tb=short
```

RED replay used the first and fourth paths only. Both original and repaired runs reported existing warning noise; pytest also reports unsuccessful cleanup of an old `test_kokoro_constructor_direct3` temporary directory. No test skipped or disabled.

## Static verification

- Black applied to changed ranges; unrelated blank-line edits reverted.
- Ruff all three touched files: only pre-existing `UP028` at `test_summarization_adapter.py:150`. Original HEAD stdin check confirms identical diagnostic. `/private/tmp/cycle4-uat105-ruff-final.json`, `-ruff-baseline.json`. No new diagnostics.
- `git diff --check` on three touched files: exit0.
- Production Bandit: `python -m bandit tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py -f json -o /private/tmp/cycle4-uat105-bandit.json`: exit0, zero findings, no skips.
- Test Bandit: same two test paths with `-s B101` (intentional test assertions only excluded), JSON `/private/tmp/cycle4-uat105-bandit-tests.json`: initial receipt reported zero; current scoped rerun identifies one baseline B105 on the pre-existing synthetic SECRET sentinel at line41, verified against original HEAD. No new test finding; see correction receipt below.

## Risks / explicit limits

- No live inference, runtime/browser work, global extractor changes, staged changes or commits. Existing UAT data untouched.
- Native source-success/analysis-warning UI verification and independent review remain pending; mocked-provider tests are not real-model acceptance.
- Scope is the existing string/OpenAI-style first-choice answer contract. Unsupported rich content blocks are safely rejected rather than introducing a new provider-format conversion layer.
- Typed errors use the existing bounded provider_failure code, while legacy plaintext callers receive actionable truncation/empty-output messages.
- Literal reasoning tags in generated analysis are treated as reasoning markup and stripped. This follows the existing UI tag vocabulary; source text is never changed by this normalization.
- Recursive/multi-chunk caller logic itself was not changed. The regression explicitly verifies single-source/chunk failure preservation and existing chunk persistence tests; it does not claim a new recursive multi-chunk native pass.

Tracking: official Backlog CLI set TASK13260.46 In Progress and appended implementation/verification notes. MCP task-view timed out/hung and was terminated; no manual Backlog edits.


## Independent review correction — nested reasoning markup

The independent review found a P2: the first implementation's non-greedy expression consumed an outer opening tag through an inner closing tag, then accepted the reasoning tail. The exact reviewer case is now permanently covered through both typed/legacy actual `analyze` and real plaintext ingestion. Review evidence was read at `/private/tmp/cycle4-analysis-outcome-independent-review.md` and its two tag-probe logs.

Added nine invalid provider fixtures in both error modes: nested same-tag reasoning, nested unclosed outer block, answer-looking prefix with unclosed reasoning, mismatched closing tag, stray close, malformed opening attributes, incomplete opening delimiter, whitespace-malformed closing delimiter, and crossed tags. Added uppercase and multiple closed-block positive controls. Added three plaintext outcomes for nested closed/unclosed and mismatched cases; source survives, analysis/chunk analysis absent, safe Warning, no sentinel.

Before correction: **19 failed,74 passed**, `/private/tmp/cycle4-uat105-tag-correction-red.log`. These are actual behavioral failures against the first repair, not a harness/setup failure.

Small local correction: match only flat, correctly paired reasoning blocks, then reject any residual recognized reasoning delimiter. No global parser/extractor change. Safe rejection of nested/malformed markup is intentional; it avoids treating a reasoning tail as a completed answer. Valid ordinary text after well-formed reasoning remains supported.

After correction: **93 passed,8 warnings in3.62s**, `/private/tmp/cycle4-uat105-tag-correction-green.log`.

Command after activating `.venv`:

```sh
python -m pytest tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_plaintext_analysis_outcomes.py -q --tb=short
```

Only covering suites rerun, as requested; the original expanded120-case suite was also independently verified before this bounded correction. Production Bandit0findings/no skips (`cycle4-uat105-tag-correction-bandit.json`); test Bandit has one pre-existing B105 on synthetic SECRET at line41, with only B101 asserts excluded (`...-bandit-tests.json`). Original HEAD stdin run reproduces the identical B105 (`cycle4-uat105-tag-correction-bandit-test-baseline.json`), so no new finding. Earlier zero-test-findings wording in task notes was corrected after inspecting the JSON result. Ruff still only original UP028 at test_summarization_adapter.py:150 (`...-ruff.json`). Scoped diff whitespace check passes.

No runtime/inference/browser/commit/staging changes. Native acceptance remains pending. Ready for scoped rereview; editing stopped after report/task update.
