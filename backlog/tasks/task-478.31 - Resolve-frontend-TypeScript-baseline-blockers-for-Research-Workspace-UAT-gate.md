---
id: TASK-478.31
title: Resolve frontend TypeScript baseline blockers for Research Workspace UAT gate
status: To Do
labels:
- research-workspace
- typescript
- verification
- frontend
priority: Medium
milestone: Research Workspace UAT Remediation
ordinal: 31
parent_task_id: TASK-478
references:
- Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md
- TASK-478.25
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove or reclassify the unrelated frontend TypeScript baseline blockers that repeatedly prevent a clean Research Workspace UAT verification gate. Initial known blockers include the CharacterListContent design-system density typing mismatch and historical sidepanel/e2e type failures recorded by prior child tasks. Scope should be limited to restoring a trustworthy project-level type check or documenting a narrower owned gate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Current project-level TypeScript blockers are reproduced and classified as Research Workspace-owned, unrelated baseline, or obsolete skip.
- [ ] #2 Known unrelated blockers such as `CharacterListContent.design-system.test.tsx` density typing are fixed or moved to an explicit non-Research Workspace verification owner.
- [ ] #3 Research Workspace UAT has a trustworthy repeatable TypeScript gate, either project-level clean or a documented focused gate with rationale.
- [ ] #4 Backlog notes and UAT matrix no longer describe stale Watchlists/e2e blockers as current unless they still reproduce.
- [ ] #5 Verification command output is recorded with exact command and outcome.
<!-- AC:END -->

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
