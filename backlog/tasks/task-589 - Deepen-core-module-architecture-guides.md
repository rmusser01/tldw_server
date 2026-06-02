---
id: TASK-589
title: Deepen core module architecture guides
status: In Progress
assignee: []
created_date: '2026-06-02 00:18'
updated_date: '2026-06-02 00:21'
labels: []
dependencies:
  - TASK-588
documentation:
  - Docs/superpowers/specs/2026-06-01-core-module-readme-refresh-design.md
  - Docs/superpowers/plans/2026-06-01-core-module-readme-refresh-inventory.md
  - >-
    Docs/superpowers/plans/2026-06-02-core-module-architecture-guides-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 2 follow-up to TASK-588. Review all 88 top-level tldw_Server_API/app/core modules as architecture-guide candidates, prioritize high-risk/high-complexity modules, expand only the modules that need deeper contributor guidance, and explicitly record sufficient modules without padding them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 High-risk and high-complexity core modules are reviewed first using source, endpoint, schema, config, and test evidence.
- [ ] #2 Expanded architecture guides document concrete flows, boundaries, data/config surfaces, extension points, operational/security gotchas, and verification paths without speculative content.
- [ ] #3 Modules that do not warrant expansion are recorded as sufficient with a concrete reason.
- [ ] #4 Verification records changed files, Markdown/link sanity, placeholder scan, and docs-only Bandit skip or relevant security check.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan: Docs/superpowers/plans/2026-06-02-core-module-architecture-guides-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created after PR #2216 merged TASK-588 into dev. Worktree: .worktrees/core-architecture-guides on branch codex/core-architecture-guides from origin/dev merge commit c7b2e66400492614914e1a8f0e4abe939b031b64.

2026-06-02: Drafted Phase 2 implementation plan at Docs/superpowers/plans/2026-06-02-core-module-architecture-guides-implementation-plan.md. Plan reviews all 88 modules, prioritizes the 47 high-risk/high-complexity modules first, and records sufficient decisions for modules that do not need expansion.
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
