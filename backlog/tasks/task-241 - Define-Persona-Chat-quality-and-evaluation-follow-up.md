---
id: TASK-241
title: Define Persona Chat quality and evaluation follow-up
status: Done
assignee:
  - Codex
created_date: '2026-05-10 19:32'
updated_date: '2026-05-10 19:35'
labels:
  - persona
  - chat
  - stage-2
  - planning
  - evaluations
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1510'
  - 'https://github.com/rmusser01/tldw_server/issues/1543'
  - 'https://github.com/rmusser01/tldw_server/issues/635'
documentation:
  - Docs/Reviews/PERSONA_BUDDY_CURRENT_STATE_AUDIT_2026_05_10.md
  - Docs/Reviews/PERSONA_CHAT_QUALITY_EVAL_FOLLOWUP_2026_05_10.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define the Stage 2 Persona Chat quality/evaluation follow-up from GitHub issue #1543. Recheck the current persona-backed chat implementation, prompt assembly, exemplar behavior, memory mode, UI entry points, and existing tests against the preserved #635 references. Produce a repo-grounded planning artifact that separates Persona Chat quality/evaluation work from Buddy/Persona Live reliability, proposes PR-sized follow-up tasks, and keeps VN/CYOA work out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current Persona Chat backend, frontend, prompt assembly, exemplar, memory, and test contracts are rechecked from source.
- [x] #2 Preserved #635 references are summarized as inputs or evaluation inspiration without making them Stage 1 Buddy/Live requirements.
- [x] #3 A durable design/planning artifact defines Persona Chat quality/evaluation criteria, risks, non-goals, and PR-sized follow-up slices.
- [x] #4 The artifact explicitly separates Persona Chat quality/evaluation from Buddy/Persona Live reliability and from VN/CYOA runtime work.
- [x] #5 Relevant GitHub tracker notes are updated or a clear tracker update recommendation is recorded.
- [x] #6 Docs verification, git diff hygiene, and Bandit skip rationale are recorded for this docs-only slice.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect current Persona Chat backend, frontend, prompt assembly, and tests.
2. Inspect Stage 0 audit and preserved #635 reference links from issue #1543.
3. Write a durable Stage 2 Persona Chat quality/evaluation planning artifact with repo evidence and PR-sized follow-up slices.
4. Update Backlog notes, run documentation verification and git diff hygiene, and package the branch for PR review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created Docs/Reviews/PERSONA_CHAT_QUALITY_EVAL_FOLLOWUP_2026_05_10.md as the Stage 2 Persona Chat quality/evaluation definition artifact for GitHub issue #1543. Rechecked ordinary persona-backed chat backend projection, conversation identity fields, persona prompt assembly, memory mode/writeback, frontend create/restore/picker/settings contracts, existing Chat/Chat_NEW/Evaluations tests, and prior Persona Role-Play PRD/evaluation docs.

The artifact preserves #635 references as inspiration only, separates Persona Chat quality/evaluation from Buddy/Persona Live reliability and VN/CYOA runtime work, defines quality axes and risks, and proposes five PR-sized follow-up slices: trace/error taxonomy, deterministic fixtures, telemetry label normalization, effective context preview, and optional calibrated LLM-as-judge evaluation.

Verification: placeholder/deferred-marker scan for the new doc returned no matches; git diff --check passed. Runtime tests were not run because this slice is docs/Backlog-only. Bandit is not applicable because no Python files changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Defined the Stage 2 Persona Chat quality/evaluation follow-up for issue #1543. Added a repo-grounded planning artifact that inventories current persona-backed chat contracts and tests, preserves legacy #635 references, defines quality axes and risks, separates this work from Buddy/Live reliability and VN/CYOA, and recommends PR-sized next slices. Verification passed for documentation marker scan and git diff hygiene; Bandit was skipped for docs-only changes.
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
