---
id: TASK-225
title: Tighten Persona/Buddy roadmap design after review
status: Done
assignee:
  - Codex
created_date: '2026-05-10 06:47'
labels:
  - persona
  - buddy
  - roadmap
  - design
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/635'
  - 'https://github.com/rmusser01/tldw_server/issues/1449'
documentation:
  - >-
    Docs/superpowers/specs/2026-05-10-persona-buddy-assistant-maturity-roadmap-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Patch the approved Persona/Buddy assistant maturity roadmap spec with the self-review clarifications requested by the user before continuing into implementation planning. Scope is docs/backlog only: strengthen Stage 0 evidence requirements, clarify Stage 1 boundaries, defer detailed issue creation until after audit evidence, preserve #635 reference migration, add Stage 3 policy/security gate, and require human error-analysis before eval/judge work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec includes stronger Stage 0 evidence table and current contract inventory requirements.
- [x] #2 Spec clearly limits Stage 1 to reliability diagnostics and UX hardening without new persona capabilities.
- [x] #3 Spec defers broad child issue creation until Stage 0 evidence identifies concrete gaps.
- [x] #4 Spec requires preserving useful #635 links/comments before tracker rewrite or closure.
- [x] #5 Spec adds policy/security review gate before Stage 3 runtime/MCP trigger expansion.
- [x] #6 Spec requires human error analysis and representative traces before Stage 2 LLM judge/eval recipes.
- [x] #7 Verification and non-code Bandit skip are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Patch the roadmap spec to add explicit Stage 0 audit evidence requirements and contract inventory outputs.
2. Clarify that Stage 1 is limited to reliability diagnostics and UX hardening of existing flows, not new persona capability expansion.
3. Revise issue hygiene so only the epic and Stage 0 audit issue are immediate; Stage 0 evidence should produce concrete Stage 1 child issues.
4. Add tracker hygiene instructions to preserve useful `#635` links/comments before rewriting or closing it.
5. Add Stage 3 policy/security review gate before broader runtime/MCP triggers.
6. Add Stage 2 error-analysis-first eval guidance before optional LLM judges or eval recipes.
7. Run docs verification, record non-code Bandit skip, and commit the docs/backlog follow-up.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Patched `Docs/superpowers/specs/2026-05-10-persona-buddy-assistant-maturity-roadmap-design.md` with stronger Stage 0 contract/evidence deliverables and a minimum report shape.
- Clarified Stage 1 as reliability diagnostics, recovery, copy, and tests only; new runtime triggers, renderer work, and Persona Chat intelligence stay out of the first implementation slice.
- Revised issue guidance so only the epic plus Stage 0 audit issue are immediate; candidate child issues are created only after audit evidence identifies concrete gaps.
- Added `#635` reference/comment preservation before tracker rewrite or closure.
- Added policy/security review before Stage 3 runtime/MCP trigger expansion.
- Added human error analysis and representative traces before Stage 2 optional LLM judges/eval recipes.
- Verification: `git diff --check` passed; targeted `rg` checks confirmed the new design gates are present.
- Bandit: skipped because this is a docs/backlog-only follow-up with no touched Python code.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Tightened the approved Persona/Buddy roadmap design after self-review. The spec now requires evidence-backed Stage 0 audit outputs, narrows Stage 1 to existing-flow reliability/UX hardening, defers broad issue creation until audit evidence exists, preserves useful `#635` context during tracker cleanup, gates Stage 3 runtime/MCP expansion on policy/security review, and makes Stage 2 eval work start from human-reviewed traces before optional LLM judges.
<!-- SECTION:FINAL_SUMMARY:END -->
