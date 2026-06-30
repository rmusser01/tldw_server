---
id: TASK-468
title: Add Research Workspace legacy storage inventory and migration safety gate
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-23 22:25'
labels:
  - frontend
  - research-workspace
  - migration
  - storage
  - safety
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-23-research-workspace-hard-replacement-roadmap-design.md
  - Docs/Design/Research_Workspace_Legacy_Storage_Inventory.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement slice A of the Research Workspace migration roadmap: create a non-destructive legacy local-storage/IndexedDB inventory and schema-mapping gate that classifies content, metadata, UI-only, derived, obsolete, and unsupported surfaces. Add tests that unknown or unmapped content is not deletion-eligible. No local deletion, no migration API endpoints, and no /workspace-playground route aliases or redirects.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

- Added a non-mutating Research Workspace legacy storage inventory module that classifies known localStorage keys, split workspace snapshot/chat keys, IndexedDB offload stores, UI-only preferences, metadata, derived runtime surfaces, and obsolete flags.
- Added a fail-closed deletion eligibility evaluator. Content-bearing or unsupported surfaces block deletion unless covered by a migration manifest; unknown workspace-prefixed localStorage keys and unknown stores in the workspace IndexedDB database also block deletion. UI-only/local diagnostic keys are retained and do not block content migration.
- Documented the storage inventory and deletion rules in Docs/Design/Research_Workspace_Legacy_Storage_Inventory.md. This slice does not delete local data, add server migration endpoints, or restore /workspace-playground aliases/redirects.
- Verification: `cd apps/packages/ui && bunx vitest run src/store/__tests__/research-workspace-legacy-storage-inventory.test.ts` passed 7 tests. `git diff --check` on tracked docs/task paths and `git diff --no-index --check /dev/null ...` for the two new TypeScript files produced no whitespace diagnostics. Bandit skipped because this slice changed frontend TypeScript/tests and documentation only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the Research Workspace legacy storage inventory and migration safety gate. Future true-move migration work now has a code-level classifier and deletion-eligibility evaluator that fails closed for unmapped content or unknown workspace storage, plus a documentation table that defines which local surfaces require server receipt before deletion and which UI-only surfaces should be retained.
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
