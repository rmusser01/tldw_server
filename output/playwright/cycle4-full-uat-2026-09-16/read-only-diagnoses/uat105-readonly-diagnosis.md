# UAT105 / TASK13260.46 — read-only normalization diagnosis

## Confirmed root cause

**The non-streaming summarization fallback serializes the entire provider response when no answer text was extracted.** At `tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py:348`, `_summarize_via_adapter` returns `extract_response_content(response) or str(response)`. This converts empty/null content or an unrecognized response dictionary into a nonempty Python dictionary representation. It includes provider metadata and `reasoning_content`; it is not valid user-facing analysis.

The retained Cedar result proves this precisely: `processing.analysis` is15,865 characters and equals `str(ast.literal_eval(processing.analysis))` byte-for-byte. Its first choice has `message.content == ''`, `finish_reason == 'length'`,4096completion tokens, and14,670reasoning characters. Source content is289characters. The reasoning is not repeated in this report or probe output.

Provider truncation explains why no final answer arrived. It does **not** justify saving the envelope or claiming successful analysis. The same bad fallback also triggers for an empty stop completion, null content and a malformed dictionary; it is not specific to Gemma, one account, or the length finish reason.

## Actual call/data path

1. The document job calls `process_document_like_item` from `app/services/media_ingest_jobs_worker.py:586`.
2. `core/Ingestion_Media_Processing/persistence.py:5188–5209` selects the real plaintext document processor; common arguments enable analysis and supply provider/prompt configuration.
3. `Plaintext/Plaintext_Files.py:507–525` calls `analyze` per chunk. It rejects `Error:` or blank/non-string results; otherwise it accepts a nonempty string, adds it to chunk summaries and chunk metadata.
4. `Summarization_General_Lib.py` direct `analyze` → `_dispatch_to_api` → `_summarize_via_adapter:348` produces the bad string. `Chat/chat_helpers.py:544–562` itself reads only `choices[0].message.content` or a plain string; it does not copy reasoning. The summarization fallback defeats that boundary.
5. `Plaintext_Files.py:576–581` joins/saves the accepted analysis. `:592–599` marks Success because there are no warnings; empty warning lists become null.
6. Document persistence retains the processor outcome (`persistence.py:5445`), permits Success/Warning source writes (`:5537`), selects analysis separately from source (`:5604–5605`), and sends it as `analysis_content` (`:5778`). There is no second answer-normalization check at that point.
7. The worker copies processor status and warnings into the terminal job result (`media_ingest_jobs_worker.py:642–648`). The retained job has result.status Success, warnings null, error null. Job-level completed indicates completed orchestration; the bad analysis-success outcome originated upstream.

The local adapter also correctly retains response structure for callers: `_LocalAdapterBase.chat` returns `_call_handler` at `providers/local_adapters.py:2073–2074`. Rewriting every adapter or the Media renderer is unnecessary to address this failure.

## Offline reproduction

Files:

- `/private/tmp/uat105-offline-normalization-probe.py`
- `/private/tmp/uat105-offline-normalization-result.json`
- `/private/tmp/uat105-offline-normalization-red.log`

Command from repository root:

```sh
source .venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 python /private/tmp/uat105-offline-normalization-probe.py
```

The probe compiles unchanged AST function bodies from the frozen extractor, summarization adapter/dispatch/analyze, and plaintext processor. Only environment/config, logger/metrics, converter and provider adapter seams are substituted; the adapter returns retained data with no network. It executes the real normalization and caller acceptance/status logic together. It does not import application modules, invoke provider inference, connect to a database or alter runtime storage. The final assertion intentionally fails to preserve a RED for the expected warning/no-analysis contract.

Results:

| Returned adapter response | Current caller result | Envelope/reasoning outcome |
| --- | --- | --- |
| Actual retained Cedar empty+length | Success, no warnings | Exact full envelope stored; reasoning included |
| Empty content, stop | Success, no warnings | Envelope and synthetic reasoning included |
| Null content, stop | Success, no warnings | Envelope and synthetic reasoning included |
| Malformed dictionary | Success, no warnings | Dictionary stored |
| Recorded positive single-user Markdown, wrapped in a synthetic normal completion | Success | Only354characters of answer retained |
| Plain answer string | Success | Only answer retained |
| Nonempty partial answer, length | Success, no warnings | No envelope leak, but truncation goes unreported |
| Content-block list | Warning, no analysis | Existing caller rejects unexpected list result |

The positive replay uses the retained single-user Markdown in a synthetic ordinary completion envelope; it does not claim the single evidence file contains that original raw provider response. Parent reports the same provider/model was used; this comparison establishes the differing answer-normalization branch without a new provider call.

An additional existing-failure-path control supplies a safe `Error: Provider returned no usable answer.` result to the real plaintext caller: status Warning, analysis null, source preserved, chunk warning present. All eight adapter cases preserve source. Thus a bounded safe failure result can use existing ingest preservation rather than turn the entire source ingest into a hard failure.

## Bounded repair proposal after frozen cycle4 ends

Primary production scope: `core/LLM_Calls/Summarization_General_Lib.py` at the non-streaming adapter-result boundary. Remove arbitrary object-to-string fallback. Accept only supported, nonblank user-facing answer text; reject empty/null/malformed output with the established sanitized failure/typed-error contract. Do not promote reasoning or metadata as answer text, and do not log the raw response. Preserve successful plain-string and normal completion text, existing credentials, cancellation and provider error behavior.

The task also requires truthful truncation handling. Check the completion finish reason at this boundary before discarding metadata. **Recommended bounded contract:** treat length-truncated completion as unsuccessful analysis with a safe explicit warning, preserving source via the current caller path. Whether to retain a labeled partial answer instead is a product decision; do not silently mark it complete. This diagnosis makes no implementation decision or product edit.

Do not start with a shared `chat_helpers.extract_response_content` rewrite: other Chat consumers use it and the observed defect is the summarizer's fallback. A small local normalizer/validation helper may be enough. No new dependency, generic provider framework, model/prompt retuning, renderer patch or retroactive mutation of existing stored Cedar analysis is required for the initial repair.

### Regression scope

- Extend existing `tests/LLM_Calls/test_summarization_adapter.py` with actual summarization calls at a mocked transport/adapter boundary: empty/null/whitespace content, malformed/missing choices/message, reasoning-only empty answer, length-truncated empty and nonempty answer, successful text and plain-string controls. Use a short reasoning/metadata sentinel and assert it never reaches output, errors or captured logs. Exercise legacy and typed-error modes. Preserve credential and streaming controls.
- Extend `tests/MediaIngestion_NEW/unit/test_plaintext_analysis_outcomes.py` beyond its current stubbed `analyze` Error/empty cases: real summarization plus real plaintext caller with the lower provider seam substituted. Assert Warning, retained source, null bad analysis/chunk metadata, safe warnings; preserve good analysis and recursive multi-chunk behavior as applicable.
- Add one targeted persistence/job integration control using existing fixtures: successful source is stored, no provider envelope enters `analysis_content`/versions/chunk metadata, and the Warning is retained through the terminal job result. No live provider calls are needed.
- Existing frontend `services/tldw/ingest-job-results.ts:62–76` already reads Warning messages; `completedIngestJobIndicatesFailure` also treats such text as a failure outcome. Therefore verify the existing UI projection with a real warning result before claiming source-success/analysis-warning UX; this diagnosis does not establish a separate frontend regression or justify an orchestration rewrite.
- After approval/unfreeze: targeted pytest, Bandit for changed Python, independent review, and native file-ingest controls. Increasing token limits is not a substitute for rejecting raw envelopes.

## Integrity and limits

Read-only diagnosis only. Source hashes are retained in the offline result, with a separate frozen-revision comparison. No product/test edits, provider calls, browser/runtime changes or commits. Only private probe/report files and official TASK13260.46 notes are authorized. This is not a fix or a new native pass; the full cycle4 matrix remains in progress.
