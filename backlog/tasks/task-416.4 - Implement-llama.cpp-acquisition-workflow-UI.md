---
id: TASK-416.4
title: Implement llama.cpp acquisition workflow UI
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-18 04:07
labels:
- llamacpp
- webui
- local-llm
dependencies: []
references:
- https://github.com/rmusser01/tldw_server/pull/1833
- https://github.com/rmusser01/tldw_server/pull/1836
documentation:
- Docs/superpowers/plans/2026-05-16-llamacpp-model-acquisition-import-workflows-plan.md
- 'PR #1836 review thread sweep: verifying and fixing current review findings for
  acquisition workflow UI.'
- 'PR #1836 review fix verification: focused Vitest passed 33 tests; git diff --check
  passed; broad bunx tsc remains blocked by pre-existing repo-wide TypeScript baseline
  outside this touched slice; Bandit not run because review fix touches only TypeScript/TSX
  and Backlog metadata.'
parent_task_id: TASK-416
priority: high
modified_files:
- apps/packages/ui/src/types/llamacpp-admin.ts
- apps/packages/ui/src/services/tldw/domains/models-audio.ts
- apps/packages/ui/src/components/Option/Admin/LlamacppAssetsPanel.tsx
- apps/packages/ui/src/components/Option/Admin/LlamacppAdminPage.tsx
- apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx
- apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx
- apps/packages/ui/src/services/__tests__/tldw-api-client.models-normalization.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 4 from the llama.cpp model acquisition/import workflow plan: expose local import preview, confirmed folder import, remote asset download queue/status/cancel, and asset refresh in the existing Admin assets WebUI without creating or starting profiles.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Local import preview renders counts/warnings and requires an explicit import action before mutation.
- [x] #2 Download form prevents empty submissions and queues Jobs-backed asset downloads through the shared API client.
- [x] #3 Queued/running/completed/failed downloads render as a compact status list with progress and cancel where applicable.
- [x] #4 Completed downloads refresh the normal llama.cpp asset inventory and do not create/start/wire profiles.
- [x] #5 Focused shared UI client/component tests cover the workflow.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Task 4 in Docs/superpowers/plans/2026-05-16-llamacpp-model-acquisition-import-workflows-plan.md. Use TDD: add failing Vitest/client tests, implement client types/methods and compact LlamacppAssetsPanel workflow, then verify focused tests and diff hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added llama.cpp acquisition API client types and methods, converted Admin asset folder import to preview then explicit confirm, added a compact remote download queue/status/cancel workflow, and refresh asset inventory once per newly completed download job. Download handling deliberately does not create saved profiles, start runtime profiles, or wire Chat. Verification: focused Vitest passed 28 tests; git diff --check passed; broad tsc remains blocked by unrelated repo-wide baseline type errors; Bandit not run because this slice touches only TypeScript/TSX and Backlog metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the llama.cpp acquisition workflow UI and addressed PR #1836 review feedback: preview confirmations now require the current previewed folder, stale previews are cleared on input changes, completed download refreshes avoid redundant initial scans and retry after failed asset refreshes, download-list errors clear after successful reloads, and duplicate warning strings render with unique keys.
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
