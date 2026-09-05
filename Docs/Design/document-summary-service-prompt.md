# Synchronous document summarization Service Prompt

Approved scope: TASK-13159. Add `media.document.summarization` to the existing
Service Prompts catalog and shared WebUI/browser-extension Settings.

Expose the system instructions that already govern document analysis. Keep the
optional request-specific user instruction and recursive-summary suffix unchanged.
Do not add another prompt store, renderer, settings page, or processor interface.

For `/api/v1/media/process-documents`, explicit `system_prompt` wins (including
an explicitly empty string). Otherwise, when analysis and a provider are enabled,
resolve the authenticated owner's saved override once before processing inputs.
If no override exists, snapshot the existing server-configured analyzer default.
Reuse that string for every document, chunk, and recursive pass. Settings displays
the packaged default; resetting an override restores the server's configured default.
An explicit `custom_prompt` remains unchanged and supplements the system guidance.

Do not resolve prompts for disabled analysis or absent providers. Fail closed on
invalid saved overrides. Preserve provider settings and processing behavior.
Persisted/queued ingestion, PDFs, ebooks, and the unused legacy service are outside
this slice; they must not acquire this behavior through a shared-processor change.

## Stage 1: Compatibility contract

Goal: Record the scope and verify existing registry behavior.
Success criteria: Approved contract above; registry baseline passes.
Tests: Existing Service Prompts unit suite (49 passed).
Status: Complete

## Stage 2: Implementation

Goal: Test-first request-scoped resolution and shared Settings metadata.
Success criteria: Save/use/reset, explicit precedence, owner isolation, stable
multi-document/chunk/recursive instructions, unchanged disabled/default behavior.
Tests: Focused live document-route tests and registry/API/Settings tests.
Status: Complete

## Stage 3: Verification

Goal: Review and validate the bounded change.
Success criteria: Focused tests, frontend checks, and touched-scope Bandit pass.
Tests: Record commands and results in TASK-13159.
Status: Complete

Verification: 119 backend tests and 194 shared frontend tests passed; extension
TypeScript compile, Ruff lint/format, and touched-scope Bandit passed (zero findings).
Independent review identified the multipart empty-string normalization issue; a
real HTTP test reproduced it before the boundary fix. Re-review found no further
issues. Full repository tests and browser end-to-end tests were not run.

PR review follow-up: real SQLite regressions reproduced retained connections from
database initialization and successful/failed liveness probes. Those temporary
connections now close on their originating worker; cached instances remain usable.
The expanded focused backend suite passed all 138 tests, and Bandit found no issues
in the three touched production files. New functions also have docstrings and
explicit parameter/return annotations. Prompt lookup remains lazy at the request
boundary, consistent with Notes title generation and the approved bypass behavior.
