---
id: TASK-13159
title: Expose synchronous document summarization in Service Prompts
status: In Progress
assignee: []
created_date: '2026-09-05 01:52'
updated_date: '2026-09-05 02:28'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2867'
documentation:
  - Docs/Design/document-summary-service-prompt.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the user-approved next Service Prompts slice for synchronous /api/v1/media/process-documents. Reuse the existing owner-scoped storage and shared WebUI/extension Settings. Exclude persisted/queued ingestion, PDFs, ebooks and unused legacy document_processing_service.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Document summarization is editable through the existing Service Prompts settings.
- [x] #2 Each synchronous analysis request resolves owner-scoped instructions once and reuses them across documents, chunks and the recursive summary.
- [x] #3 Explicit request instructions, server defaults, provider configuration and analysis-disabled behavior remain compatible.
- [x] #4 Focused behavioral tests, frontend checks and touched-scope Bandit pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Capture approved design and compatibility contracts; establish baseline tests. 2. Add failing registry and live-route tests, implement minimal request-scoped resolution and shared Settings metadata. 3. Verify precedence, owner isolation, chunk reuse, defaults, disabled analysis and security; review and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approved in this task conversation. Isolated branch codex/document-summary-service-prompt starts at dev c5dfe0ff73. Backlog MCP search has not returned; using official CLI fallback.

Implemented literal document system guidance using the existing registry, storage, shared Settings and processor arguments. Resolved once before uploads/downloads; explicit request system prompt (including multipart empty string) wins, custom user instructions remain unchanged, and no override snapshots the deployment default. RED: 9 initial backend tests failed; UI localization regression failed; multipart empty-field HTTP regression failed. GREEN final: pytest document summary + registry/API + JSON document + media usage suites: 119 passed (9 existing warnings); Vitest Settings + Service Prompts service/domain suites: 194 passed; extension tsc --noEmit -p tsconfig.compile.json: passed; Ruff check and format check: passed; ESLint touched shared files: exit 0 (Next pages-directory configuration notice only); Bandit two production Python files: zero findings/errors, report /tmp/bandit_document_summary_service_prompt.json. Independent review found multipart normalization issue, fixed test-first and re-reviewed with no further findings. Full repository tests and browser end-to-end tests not run. Queued/persisted ingestion, PDFs and ebooks remain intentionally excluded. Local implementation only; no PR created or merge attempted.

User selected push/create PR. Opened PR #2867 against dev: https://github.com/rmusser01/tldw_server/pull/2867 . The PR records the focused verification and intentionally excluded workflows. Human-written Change summary is pending before merge; no merge or recurring monitoring requested in this step.

Qodo posted four review findings on the implementation commit: route/core prompt-policy separation, missing docstrings, missing test annotations, and potential thread-local connection retention in Prompts_DB_Deps health checks. Verifying and addressing the findings under this task before merge.

Qodo fixes: reproduced three real SQLite lifecycle failures (successful probe, failed probe, initial cached-instance setup retained handles), then closed each temporary connection on its originating worker. Expanded focused backend suite: 138 passed, 9 warnings. Added docstrings and explicit parameter/return annotations to new helpers/tests. Ruff lint passed on changed Python scope; format check passed on endpoint/tests (existing dependency-file formatting left unchanged). Bandit on three touched production files: zero findings/errors, /tmp/bandit_document_summary_review.json. Retaining lazy request-boundary orchestration consistent with existing Notes title resolver and approved explicit/disabled/no-provider bypass behavior; independent review pending. Fresh origin/dev fetch has zero commits missing from branch. Required CI and human Change summary still pending.

Independent review of the Qodo patch found no Critical or Important issues and confirmed retaining the request-boundary orchestration is justified by the existing Notes pattern and bypass requirements. Post-format lifecycle/document regression rerun: 20 passed. Publishing fixes and replying to all four Qodo threads; CI and human Change summary remain merge gates.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added Document summarization to shared Service Prompts Settings and owner-scoped synchronous document analysis. Reused existing prompt storage and processor arguments to avoid a separate configuration system, preserved server defaults and explicit requests, and froze one value per request to prevent mid-analysis saves from changing later chunks. Verified with focused backend/frontend suites, compiler, lint and security checks.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
