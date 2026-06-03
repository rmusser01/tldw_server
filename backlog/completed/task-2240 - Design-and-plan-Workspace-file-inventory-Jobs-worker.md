---
id: TASK-2240
title: Design and plan Workspace file inventory Jobs worker
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-03 20:32'
labels: []
dependencies: []
references:
  - TASK-2235
documentation:
  - >-
    Docs/superpowers/specs/2026-06-03-canonical-workspace-core-project-model-design.md
  - >-
    Docs/superpowers/plans/2026-06-03-workspace-core-contract-implementation-plan.md
  - >-
    Docs/superpowers/specs/2026-06-03-workspace-primary-root-attach-api-design.md
  - Docs/superpowers/specs/2026-06-03-workspace-file-inventory-jobs-design.md
  - >-
    Docs/superpowers/plans/2026-06-03-workspace-file-inventory-jobs-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create an implementation-ready design spec and plan for the next Project Workspace slice: a Jobs-backed file inventory metadata scanner for attached primary roots. Scope includes metadata-only scanning, ignore policy, bounded diagnostics, partial success, root state projection, job enqueue/status contracts, redaction, and tests. Exclude file-content indexing, Git operations beyond optional status placeholders, UI file tree implementation, MCP trusted-root mutation, and Sandbox volume lifecycle.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 File inventory design spec defines metadata-only scan scope, safety boundaries, Jobs ownership, DB/API contracts, and non-goals.
- [x] #2 Implementation plan decomposes the work into reviewable TDD slices with exact files, commands, and verification gates.
- [x] #3 Plan preserves primary-root redaction, no automatic content indexing, and fail-closed behavior for missing/unready roots.
- [x] #4 Backlog task records verification/review notes and the current status of any open implementation questions.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Design/spec notes:
- Added Docs/superpowers/specs/2026-06-03-workspace-file-inventory-jobs-design.md.
- Added Docs/superpowers/plans/2026-06-03-workspace-file-inventory-jobs-implementation-plan.md.
- Recommended Jobs for execution/live progress and Workspace tables for durable scan/item state.
- Recommended job payloads that contain workspace/root/scan ids but no absolute root paths.
- Preserved metadata-only scope: no content reads except bounded ignore-policy files, no content hashes, no chunking, no embeddings.
- Host-local scans revalidate Workspace root-binding policy in the worker; sandbox-volume scans fail closed until a mounted-path resolver exists.
- Root validation failures should be represented as durable failed scan status, not opaque request-only errors.
- Open implementation questions resolved for first slice in the design spec.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the Workspace file inventory Jobs design slice. Added an implementation-ready design spec and TDD implementation plan covering metadata-only scan scope, Jobs ownership, durable Workspace scan/item state, redacted job/API contracts, ignore policy, bounded diagnostics, partial success, fail-closed root handling, and context/capability integration. Verification: git diff --check passed; ASCII scan passed; no accidental local path leaks found except intentional no-redirect wording; the completed record was renumbered to TASK-2240 after a TASK-2236 shared-index collision and verified by direct file inspection. Bandit/Python tests were not run because this task changed only docs and Backlog records. Spec subagent review was not dispatched in this slice because no fresh subagent authorization was available; a self-review pass tightened force semantics, context resolvability wording, and root-validation error mapping.
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
