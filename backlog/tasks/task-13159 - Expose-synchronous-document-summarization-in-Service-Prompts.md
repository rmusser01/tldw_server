---
id: TASK-13159
title: Expose synchronous document summarization in Service Prompts
status: Done
assignee: []
created_date: '2026-09-05 01:52'
updated_date: '2026-09-05 02:06'
labels: []
dependencies: []
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
