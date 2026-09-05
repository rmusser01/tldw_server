# Synchronous PDF summarization Service Prompt

Approved scope: TASK-13161. Add an independent `media.pdf.summarization`
definition to the existing owner-scoped Service Prompts catalog and shared
WebUI/browser-extension Settings. Its only editable part is literal system
instructions; document and PDF overrides do not affect one another.

At `/api/v1/media/process-pdfs`, preserve explicit system prompts, including
empty multipart fields. Otherwise, if analysis and a provider are enabled,
resolve the authenticated owner's saved override once before input processing.
With no override, snapshot the existing server-configured analyzer default.
Reuse the string across all PDFs, chunks and recursive passes through the
existing processor argument. Close prompt-database connections on their worker.

Keep custom user instructions, recursive suffixes, parser/OCR/VLM options,
provider configuration, summary fallbacks and queued/persisted ingestion unchanged.
Disabled analysis and absent providers do not access prompt storage. Corrupt
overrides fail before upload/download processing rather than silently reverting.
No new storage, Settings layout, processor interface or shared processor behavior.

The HTTP regressions exposed a prerequisite bug: the PDF multipart dependency
discarded `api_name`, so the existing processor never received the selected
provider. Parse both canonical `api_provider` and legacy `api_name`, preferring
the canonical value, and forward through the existing processor argument.
Request credentials remain unsupported; keys come from server
configuration. No PDF processor changes are needed.

Review follow-up: resolve deployment defaults in the shared core resolver so
Settings detail/reset and runtime instructions agree. The same defect affected
document summaries, so both summary definitions use this path. Static packaged
defaults remain available separately; editable effective parts use the server
configuration. Reset still performs no post-deletion owner-storage reread.

## Stage 1: Compatibility contract

Goal: Confirm approved scope and current behavior.
Success criteria: Existing registry/document baseline is green.
Tests: Registry and document Service Prompt suites: 64 passed, 2 warnings.
Status: Complete

## Stage 2: Implementation

Goal: Test-first PDF route resolution and shared Settings metadata.
Success criteria: Independent save/use/reset, explicit precedence, owner isolation,
stable batch/chunk/recursive snapshots and unchanged disabled/default behavior.
Tests: Real PDF extraction and request handling with model-boundary fakes;
generic prompt API, catalog and shared Settings tests.
Status: Complete

## Stage 3: Verification

Goal: Validate the bounded change and review before handoff.
Success criteria: Focused backend/frontend tests, lint/type checks, Bandit and
independent code review pass; commands and limitations recorded in TASK-13161.
Tests: New PDF behavior plus existing document/PDF/Service Prompts regressions.
Status: Complete

Verification: PDF/registry/API suites: 89 passed; broader PDF/document,
chunking, input-contract, usage-event and permission regressions: 95 passed;
shared Settings/service/domain suites: 195 passed. Ruff lint/format and
touched-file ESLint passed; extension compile passed. Bandit found no issues
or scan errors in the three touched production Python files. Independent
review found no actionable issues. Full repository suites, full shared-UI
typechecking and live-provider/browser end-to-end tests were not run.

Qodo follow-up: four failing regressions reproduced canonical-provider omission
and mismatched Settings defaults. After fixes, the combined PDF/document and
Service Prompts registry/API suites pass all 108 tests. Independent review of
the fixes found no actionable issues. The OpenAPI fingerprint and local frontend
types were regenerated with the official tooling; a fresh drift check passes.
Removing only the two new PDF provider properties reproduces the original
fingerprint, confirming there is no unrelated schema drift.

Rebased onto `dev` commit `3bc8c6a98c` after PR #2868 merged. Only the generated
OpenAPI fingerprint conflicted; application code was unchanged by the rebase.
Regeneration preserves the new Sync contract (2068 paths, 3133 schemas), and
removing only the two PDF provider properties reproduces the new `dev`
fingerprint. Fresh verification: 108 focused backend tests and the OpenAPI
drift check pass; frontend API types regenerated successfully. The new base
also contains an unrelated task with the same numeric ID; this work's record
is specifically `task-13161 - Expose-synchronous-PDF-summarization-in-Service-Prompts.md`.
Do not edit the unrelated Personal Context task through ambiguous ID lookup.
