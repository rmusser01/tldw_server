---
id: TASK-13243
title: Assess Persona Workspace parity with Chatbook dev and plan issue 2950
status: Done
assignee: []
created_date: '2026-09-13 18:12'
updated_date: '2026-09-13 18:23'
labels:
  - persona
  - workspaces
  - planning
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2952'
documentation:
  - Docs/Design/2026-09-13-persona-workspace-parity-assessment.md
  - >-
    Docs/superpowers/plans/2026-09-13-persona-workspace-parity-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Audit pinned server and Chatbook dev implementations for Persona Workspace assistant defaults; write an evidence-backed parity assessment and backend-first staged implementation plan, including Research Workspace adoption and tool-profile dependencies.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pin both dev revisions and classify contract parity, confirmed gaps, platform distinctions, and unverified behavior with code/test evidence.
- [x] #2 Provide a backend-first staged plan with exact target files, meaningful regression cases, dependencies, and explicit acceptance gates.
- [x] #3 Review the assessment and plan, verify referenced paths and diff hygiene, and link the artifacts from issue 2950.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Assessment and five-stage plan written and self-reviewed; stage tasks TASK-13244, TASK-13245, TASK-13246, TASK-13248, TASK-13247 created. Verified 39 existing server path references and document links; no missing paths. Baseline: 32 passed, 1 failed (legacy v48 test rewinds current DB into Notes v59 registry collision). Code and Chatbook files unchanged; docs-only Bandit skip. MCP resource/search stalled; official CLI fallback used.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the pinned Persona Workspace/Chatbook parity assessment and reviewed five-stage plan. Draft PR #2952 and artifacts linked from #2950. Stage tasks remain To Do with dependencies: 13244, 13245, 13246, 13248, 13247. Verified 39 server path references and document links, Python example syntax, and staged whitespace. Baseline: 32 passed, 1 pre-existing migration-fixture failure, 6 warnings; tracked Stage 1 repair. Chatbook/runtime UAT not run. Bandit skipped for docs-only changes; production code unchanged. Implementation and parity certification remain pending.
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
