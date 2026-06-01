---
id: TASK-588
title: Refresh core module README developer docs
status: In Progress
assignee: []
created_date: '2026-06-01 07:18'
updated_date: '2026-06-01 07:21'
labels: []
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-06-01-core-module-readme-refresh-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source-informed documentation pass for all 88 top-level tldw_Server_API/app/core modules. Phase 1 creates or refreshes concise contributor-oriented README.md files from actual source, endpoint, schema, configuration, and test context. Phase 2 follows with deeper architecture-guide review for all 88 modules as candidates, prioritized by risk and complexity without padding simple modules unnecessarily.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every top-level app/core module has a README.md.
- [ ] #2 README content is source-informed and avoids placeholder scaffolding.
- [ ] #3 Existing strong READMEs are preserved or tightened rather than rewritten wholesale.
- [ ] #4 Verification records README coverage and markdown/link sanity checks.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-06-01: Design approved by user: source-informed orientation pass for all 88 top-level core modules first, followed by a deeper architecture-guide pass.

2026-06-01: Pre-plan review found and corrected spec risks: Phase 2 now aligns with the user's 'approach 3 after approach 2' direction by treating all 88 modules as deep-guide candidates; scope now explicitly covers all immediate non-cache app/core directories; implementation inventory and local verification fallback are documented.
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
