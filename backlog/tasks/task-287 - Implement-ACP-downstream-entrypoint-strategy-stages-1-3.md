---
id: TASK-287
title: Implement ACP downstream entrypoint strategy stages 1-3
status: In Progress
assignee: []
created_date: '2026-05-12 03:51'
updated_date: '2026-05-12 04:43'
labels:
  - ACP
  - implementation
  - certification
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1563'
  - 'https://github.com/rmusser01/tldw_server/issues/1564'
documentation:
  - >-
    Docs/superpowers/specs/2026-05-12-acp-downstream-entrypoint-strategy-design.md
  - Docs/Development/ACP_Compatibility_Matrix.md
  - Docs/Development/ACP_Certification_Checklist.md
  - tldw_Server_API/Config_Files/agents.yaml
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved ACP downstream entrypoint strategy design for the first product slice. This work adds explicit ACP entrypoint strategy metadata, classification, profile-specific certification manifests, and setup/status/API visibility while keeping live certification, downstream agent installation, and adapter implementation out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Registry entries, YAML rows, API registration/update schemas, and DB-backed dynamic registrations preserve entrypoint strategy metadata with conservative defaults for legacy rows.
- [ ] #2 A deterministic classifier reports probe state, ACP command/args, primary blocker, blockers, status message, and docs URL without running live agent commands.
- [ ] #3 Certification smoke helper can render profile-specific dry-run manifests for native, adapter-backed, documented-candidate, and custom-template profiles and refuses unsafe live runs without required env.
- [ ] #4 ACP agents, health, and setup-guide surfaces expose strategy and blocker metadata consistently for YAML, API-backed, runner, and static fallback rows.
- [ ] #5 Focused unit, helper, integration, docs, and security checks pass for the touched scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation plan saved at Docs/superpowers/plans/2026-05-12-acp-entrypoint-strategy-implementation-plan.md.

Plan review completed after two fix rounds. Added explicit no-inference guardrail tests, legacy DB migration/default coverage, dynamic API parity tests, and initialize-gated session/new plus session/prompt manifest sequencing. Final reviewer status: approved.

Task 1 complete. Added registry/API/DB ACP entrypoint strategy metadata, built-in YAML seeds, migration/default handling, and focused tests. Reviews: spec compliant and code quality approved after null-clearing and mutable-default fixes. Final scoped Task 1 tests: 67 passed, 5 warnings. Bandit reports only existing ACP_Sessions_DB.py baseline findings outside changed lines.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
