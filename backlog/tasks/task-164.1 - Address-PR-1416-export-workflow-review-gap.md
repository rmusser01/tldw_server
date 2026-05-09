---
id: TASK-164.1
title: 'Address PR #1416 export workflow review gap'
status: In Progress
assignee: []
created_date: '2026-05-09 16:00'
updated_date: '2026-05-09 16:01'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1416#discussion_r3213384370'
  - 'https://github.com/rmusser01/tldw_server/pull/1416'
documentation:
  - Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md
parent_task_id: TASK-164
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Persona Visuals no-pack empty state explicitly includes export/import review as a post-draft workflow.
- [x] #2 Focused VisualPackEditor test fails before the copy change and passes after implementation.
- [ ] #3 PR review thread is resolved after verification and push.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RED: bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx failed with the expected missing export wording in persona-visual-pack-empty.

GREEN: bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx passed with 7 tests after adding import/export copy.

RELATED VERIFICATION: bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx src/utils/__tests__/persona-garden-route.test.ts passed with 31 tests.

HYGIENE: git diff --check passed.

BANDIT: not applicable; touched code is frontend TypeScript plus Backlog metadata only.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
