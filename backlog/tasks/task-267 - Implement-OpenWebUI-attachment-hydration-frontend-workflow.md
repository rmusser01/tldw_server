---
id: TASK-267
title: Implement OpenWebUI attachment hydration frontend workflow
status: Done
assignee: []
created_date: '2026-05-11 16:59'
updated_date: '2026-05-11 17:21'
labels:
  - chatbooks
  - openwebui
  - frontend
  - implementation
dependencies:
  - TASK-266
references:
  - >-
    Docs/superpowers/plans/2026-05-11-openwebui-attachment-hydration-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 7 of the OpenWebUI attachment hydration plan: add frontend API client methods and a preview-first Chatbooks import workflow for server-local OpenWebUI attachment hydration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Frontend API client exposes preview, create job, and get job methods for OpenWebUI hydration
- [x] #2 Chatbooks import UI shows discoverable hydration controls near OpenWebUI import
- [x] #3 Preview requires a data root and sends process_supported_files false by default
- [x] #4 Opt-in processing toggles process_supported_files true and job creation is gated on successful preview
- [x] #5 Preview/job status counts and warnings render using existing UI patterns
- [x] #6 Focused Vitest/client/UI checks and diff checks are recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Stage 7 frontend workflow: added OpenWebUI hydration client methods and import-tab hydration controls for data root, conversation ids, optional source user, preview, opt-in supported-file processing, job creation, job refresh, summary counts, and warnings.

Verification: bun run test src/services/__tests__/tldw-api-client.chatbooks-openwebui.test.ts src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx from apps/packages/ui passed, 13 tests. bun run verify:openapi passed with reviewed baseline exception paths. git diff --check passed. Package-wide bunx tsc -p tsconfig.json --noEmit still fails on existing unrelated baseline TypeScript errors; no reported error referenced the new Chatbooks hydration files. Bandit not applicable for this frontend-only slice. User docs are owned by Stage 8.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the Stage 7 frontend workflow for OpenWebUI attachment hydration. The UI package now exposes preview/create/get hydration client methods and the Chatbooks import panel shows OpenWebUI hydration controls with data-root entry, imported conversation id scope, optional source user, default-off supported-file processing, preview summaries/warnings, gated job creation, and job status refresh. Verification: focused client/page Vitest suite passed with 13 tests, verify:openapi passed, git diff --check passed. Package-wide TypeScript still has unrelated baseline failures outside this slice; Bandit was not applicable because this slice touched frontend TypeScript only.
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
