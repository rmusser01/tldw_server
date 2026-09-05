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
provider. Parse and forward that existing schema field, matching document and
ebook routes. Request credentials remain unsupported; keys come from server
configuration. No PDF processor changes are needed.

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
