# Synchronous EPUB summarization Service Prompt

Approved scope: TASK-13174. Add independent `media.ebook.summarization`
system instructions to the existing Service Prompts catalog and shared Settings.

The authenticated `/media/process-ebooks` adapter resolves one owner-specific
string before uploads/downloads, only when analysis and a provider are enabled
and no explicit system prompt was supplied. Explicit empty multipart values
also win. Reuse that string across books, chapters and the final summary.
Close the prompt database connection on its lookup worker, including failures.
Invalid saved overrides fail closed before processing inputs.

Reuse the shared resolver, including deployment-specific analyzer defaults,
and existing save/reset APIs. Normalize canonical `api_provider` ahead of legacy
`api_name`, as in the PDF route; credentials remain server-configured.
Keep user prompts, parser options, chunking, summary fallbacks and disabled
analysis unchanged. Queued/persisted ingestion and direct core callers are
outside this slice. No new storage, endpoint or Settings subsystem.

Verification: real multipart EPUB processing with isolated owner databases and
a stubbed LLM boundary; owner separation, independent media prompts, explicit
empty/text precedence, no-read disabled paths, frozen batch/recursive snapshots,
corrupt overrides and deployment defaults. Extend generic catalog/save/reset
and shared Settings coverage; run focused regressions, Ruff and Bandit.

## Verification results

- Baseline: 108 backend tests passed before implementation.
- Test-first: 10 backend failures and the new EPUB Settings test failed on the
  expected missing behavior, then passed after implementation.
- Final backend: 120 prompt/processing tests plus 45 adjacent EPUB ingestion,
  safe-path, chunking-contract and usage tests passed (165 non-overlapping tests).
- Shared frontend: 196 Settings/service/domain tests passed; 4 targeted Settings
  checks also passed under the WebUI configuration.
- Ruff lint/format and compilation passed. Bandit found no issues in touched
  production Python (report: /tmp/bandit_ebook_service_prompt.json).
- Official OpenAPI export, TypeScript generation and fingerprint check passed.
  Removing only the new EPUB form api_provider field reproduces the base
  fingerprint exactly.
- Independent read-only code review: no actionable findings.

Full repository suites, full frontend typechecking and live browser/provider
end-to-end runs were not performed. Existing runtime/dependency warnings remain.

## PR review follow-up

Qodo's multipart comment is addressed by restoring explicit empty system text
inside the EPUB form dependency before model validation. A real multipart parser
regression failed on empty text before this move; omitted, empty and literal
values now pass. The endpoint consumes only the validated prompt field.

Removed redundant owner lookup sequences from ownership/provider tests. Retained
no-read and single-lookup checks because those are explicit accepted requirements.
Authenticated database acquisition and worker-local cleanup remain API adapter
responsibilities; override validation and deployment defaults remain in the
existing core resolver. Independent review confirmed these boundaries.

Follow-up validation: 123 backend tests passed; Ruff, compilation and Bandit
(zero findings) passed. The official OpenAPI fingerprint check remained unchanged.
