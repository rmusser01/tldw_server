---
id: TASK-13161
title: Expose synchronous PDF summarization in Service Prompts
status: Done
assignee: []
created_date: '2026-09-05 04:03'
updated_date: '2026-09-05 04:21'
labels: []
dependencies: []
documentation:
  - Docs/Design/pdf-summary-service-prompt.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved bounded PDF slice: independent media.pdf.summarization settings entry, owner-scoped once-per-request system prompt, explicit including empty precedence, server defaults, no lookup for disabled analysis or missing provider. Keep OCR/VLM, extraction, custom user instructions, processor interfaces and queued ingestion unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PDF summarization is independently editable via shared WebUI and browser-extension Service Prompts settings.
- [x] #2 One authenticated-owner system prompt is reused across all PDFs, chunks and recursive summary passes in a synchronous request.
- [x] #3 Explicit empty and nonempty prompts, server defaults, reset, custom instructions, disabled analysis and absent providers remain compatible; invalid saved overrides fail before input processing.
- [x] #4 Focused backend and shared Settings regressions, lint/type checks and touched-scope Bandit pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record approved scope and baseline. 2. Add failing PDF route/processor and Settings regressions; implement the smallest existing-pattern extension. 3. Run focused compatibility/security checks, independent review and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approved in this conversation. Fresh worktree codex/pdf-summary-service-prompt from dev a5aa0c8e67. Baseline registry plus document Service Prompt tests: 64 passed, 2 warnings. Read-only searches found no existing PDF Service Prompt task/worktree. MCP search did not return; using official CLI fallback. Local worktree task-ID scan tops out at 13160.

Implemented the approved independent PDF literal-system entry with shared Settings metadata and golden defaults. HTTP tests exposed an existing multipart bug: api_name was discarded, preventing analysis. Added only parsing/forwarding of that existing field; credentials remain server-only. RED: 7 backend contract failures and 1 shared Settings failure before implementation. GREEN: PDF/registry/API 89 passed; broader PDF/document, chunking, input-contract, usage-event and permission regressions 95 passed (8 warnings); Settings/service/domain 195 passed. Ruff lint/format and touched-file ESLint passed; ESLint emitted only the shared-file Next pages-directory notice. Extension tsc --noEmit -p tsconfig.compile.json passed (not a full shared-UI typecheck). Bandit scanned the three changed production Python files: zero findings and errors, report /tmp/bandit_pdf_summary_service_prompt.json. Independent review found no Critical, Important or Minor issues. git diff --check passed. Full repository suites, full shared-UI typecheck and live provider/browser E2E were not run. No dependency changes; temporary dependency symlinks removed before commit. Local implementation complete; PR creation/integration awaits the user choice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Synchronous PDF summaries now use an independently configurable owner-scoped Service Prompt across each request, preserving explicit empty/nonempty prompts, server defaults, reset and disabled/no-provider behavior. Real multipart regressions cover owner isolation, batch/chunk/recursive snapshot consistency and corrupt overrides before input processing. Fixed the existing PDF provider form field omission needed to reach analysis. Reused existing storage, shared Settings and processor interfaces.
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
