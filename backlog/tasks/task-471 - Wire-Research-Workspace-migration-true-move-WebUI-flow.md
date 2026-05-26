---
id: TASK-471
title: Wire Research Workspace migration true-move WebUI flow
status: In Progress
labels:
- research-workspace
- migration
- webui
references:
- Docs/superpowers/specs/2026-05-23-research-workspace-hard-replacement-roadmap-design.md
- Docs/Design/Research_Workspace_Legacy_Storage_Inventory.md
- Docs/Design/Research_Workspace_Migration_Protocol_API.md
documentation:
- Docs/superpowers/plans/2026-05-26-research-workspace-migration-true-move-webui-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the client-side /research-workspace migration driver that uses the existing legacy inventory gate and backend migration protocol to create sessions, upload chunk receipts, finalize, wait for client_delete_eligible, delete only approved local content payloads, write a tombstone, and send client-delete-ack. No /workspace-playground aliases or redirects.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Typed WebUI API client methods exist for create/list/get chunk/finalize/client-delete-ack migration protocol calls with focused tests.
- [ ] #2 Legacy migration ignores the obsolete workspace_migrated flag and builds a deterministic manifest/chunk plan from known content-bearing localStorage and IndexedDB surfaces.
- [ ] #3 Unknown workspace-prefixed localStorage keys or unknown tldw-workspace-storage IndexedDB stores block local content deletion by default.
- [ ] #4 Migration driver creates or resumes an idempotent session, records chunk receipts, finalizes, fetches recovery state, and returns a recoverable status without blocking workspace load.
- [ ] #5 Local content deletion, tombstone write, and client-delete-ack happen only when both local inventory eligibility and server client_delete_eligible are true.
- [ ] #6 Current backend-ineligible finalize state is surfaced as retained-local-data recovery copy rather than silently deleting or claiming migration success.
- [ ] #7 Focused Vitest and live backend + WebUI + CDP validation prove the migration flow, old-route no-redirect behavior, and no workspace-playground UI regression.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan: Docs/superpowers/plans/2026-05-26-research-workspace-migration-true-move-webui-plan.md

Scope: Client migration protocol methods, safe legacy storage manifest/chunk planning, driver state machine, contextual /research-workspace UI status, and live validation. True local deletion remains gated by server client_delete_eligible; current backend false eligibility must retain local content and show recovery state.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
