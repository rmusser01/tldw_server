---
id: TASK-471
title: Wire Research Workspace migration true-move WebUI flow
status: Done
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
- [x] #1 Typed WebUI API client methods exist for create/list/get chunk/finalize/client-delete-ack migration protocol calls with focused tests.
- [x] #2 Legacy migration ignores the obsolete workspace_migrated flag and builds a deterministic manifest/chunk plan from known content-bearing localStorage and IndexedDB surfaces.
- [x] #3 Unknown workspace-prefixed localStorage keys or unknown tldw-workspace-storage IndexedDB stores block local content deletion by default.
- [x] #4 Migration driver creates or resumes an idempotent session, records chunk receipts, finalizes, fetches recovery state, and returns a recoverable status without blocking workspace load.
- [x] #5 Local content deletion, tombstone write, and client-delete-ack happen only when both local inventory eligibility and server client_delete_eligible are true.
- [x] #6 Current backend-ineligible finalize state is surfaced as retained-local-data recovery copy rather than silently deleting or claiming migration success.
- [x] #7 Focused Vitest and live backend + WebUI + CDP validation prove the migration flow, old-route no-redirect behavior, and no workspace-playground UI regression.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan: Docs/superpowers/plans/2026-05-26-research-workspace-migration-true-move-webui-plan.md

Scope: Client migration protocol methods, safe legacy storage manifest/chunk planning, driver state machine, contextual /research-workspace UI status, and live validation. True local deletion remains gated by server client_delete_eligible; current backend false eligibility must retain local content and show recovery state.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added typed workspace migration API methods and focused request path tests.
- Replaced the old flag-based migration helper with deterministic manifest/chunk planning and tombstone helpers; the obsolete workspace_migrated flag is ignored.
- Added the runResearchWorkspaceMigration state machine with local deletion gated by both inventory eligibility and server client_delete_eligible.
- Wired /research-workspace to detect legacy localStorage, infer IndexedDB offload stores from split-storage pointers, start migration once per active workspace/signature, and show compact retained-local-data copy in the existing status bar without adding a new trust/banner bar.
- Fixed two live-validation issues found after initial wiring: React StrictMode now reattaches to an in-flight migration promise instead of leaving the UI stuck at "Legacy workspace data found", and local-inventory-blocked sessions now show both "Server receipt saved" and "Local data retained".
- Live backend + WebUI + CDP validation on 2026-05-26 used current-checkout backend `127.0.0.1:18001` and WebUI `127.0.0.1:3001`. Seeded legacy localStorage plus an unknown workspace-prefixed key; `/research-workspace` created a migration session, accepted 3 chunk receipts, finalized with `client_delete_eligible=false`, fetched the recovery session, retained all local legacy data, showed compact recovery copy, did not render a workspace trust bar, and did not mention `/workspace-playground`.
- Old-route validation: `/workspace-playground` returned 404 and stayed at `/workspace-playground`; no redirect or alias was present.
- Backend follow-up recorded as `TASK-515`: server-side verification is still needed before migration sessions can safely emit `client_delete_eligible=true`.
- Verification: `bunx vitest run src/services/tldw/domains/__tests__/workspace-api.status-capabilities.test.ts src/store/__tests__/workspace-migration.test.ts src/store/__tests__/research-workspace-legacy-storage-inventory.test.ts src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx --maxWorkers=1 --no-file-parallelism` passed with 4 files / 53 tests; `git diff --check` passed.
- TypeScript package check required `NODE_OPTIONS=--max-old-space-size=8192`; touched-code type errors were resolved. Remaining failures are pre-existing baseline issues in `src/components/Option/Characters/__tests__/CharacterListContent.design-system.test.tsx` and `src/routes/__tests__/sidepanel-flashcards.test.tsx`.
- Bandit skipped: this slice changed TypeScript UI code, docs, and Backlog records only; no Python backend files were touched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the safe WebUI true-move migration flow for `/research-workspace`. The client now builds deterministic migration manifests, uploads migration chunks, finalizes/rechecks recovery state, and retains local data unless both local inventory and server deletion eligibility allow deletion. Live validation proved current backend behavior remains recovery-only (`client_delete_eligible=false`), with the backend eligibility follow-up tracked separately in `TASK-515`.
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
