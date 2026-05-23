---
id: TASK-45.44.11
title: >-
  Migrate design-system product state: Character, Persona, and presentation
  surfaces
status: Done
assignee: []
created_date: '2026-05-14 03:20'
updated_date: '2026-05-23'
labels:
  - design-system
  - webui
  - extension
  - product-state
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1668'
  - 'https://github.com/rmusser01/tldw_server/pull/1709'
  - 'https://github.com/rmusser01/tldw_server/pull/1759'
  - 'https://github.com/rmusser01/tldw_server/pull/1832'
  - >-
    Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45.44
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Mirror the linked GitHub product-area migration issue. Closure requires zero current product-state baseline exceptions for the owned path map area and the verification gates recorded in the GitHub issue.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The linked GitHub issue owns current count and public status.
- [x] #2 Implementation PR tasks are created under this child when the area is too broad for one PR.
- [x] #3 Backlog notes record PR links and before/after count evidence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created and completed TASK-45.44.11.1 for the Presentation Studio ExtensionStartPanel Ready-label migration. PR: https://github.com/rmusser01/tldw_server/pull/1709. This removed the two ExtensionStartPanel canonical-state-label baseline entries.
- Created and completed TASK-45.44.11.2 for the Persona Garden AssistantVoiceCard wake-warning Alert migration. PR: https://github.com/rmusser01/tldw_server/pull/1759. This removed the AssistantVoiceCard AntD Alert baseline entry and addressed review follow-up by migrating the inherited QuickIngest review-step offline warning.
- Created and completed TASK-45.44.11.3 for the CharacterDialogs Alert migration. PR: https://github.com/rmusser01/tldw_server/pull/1832. This removed the ten CharacterDialogs AntD Alert baseline entries and recorded touched-file TypeScript cleanup evidence.
- Closeout 2026-05-23: verified current `apps/packages/ui/scripts/design-system-product-state-baseline.json` has zero rows for the Character/Persona/presentation owned path map (`src/components/Option/Characters`, `src/components/Option/PresentationStudio`, and `src/components/PersonaGarden`). GitHub issue #1668 was refreshed from the original Total 22 baseline debt to Total 0 with PR links for the three completed child slices. Current full `bun run verify:design-system-state` exits 1 on unrelated repo-wide product-state drift outside this owned path map, so this closeout does not claim global verifier cleanliness. Bandit skipped for this closeout because only Backlog markdown is changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed the Character, Persona, and presentation design-system product-state tracker. The area was completed through child tasks TASK-45.44.11.1, TASK-45.44.11.2, and TASK-45.44.11.3, with merged PRs #1709, #1759, and #1832. The current baseline has zero rows for the owned paths and GitHub issue #1668 now records the public zero-count status. No application code changed in this closeout.
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
