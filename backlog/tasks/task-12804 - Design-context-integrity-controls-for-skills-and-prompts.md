---
id: TASK-12804
title: Design context integrity controls for skills and prompts
status: Done
labels:
- security
- skills
- prompts
- design
documentation:
- Docs/superpowers/specs/2026-06-25-context-integrity-skills-prompts-design.md
modified_files:
- Docs/superpowers/specs/2026-06-25-context-integrity-skills-prompts-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a design spec for signed trust manifests, quarantine, and audit controls around skill files and prompt-bearing assets so users can detect and contain offline tampering across server restarts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec covers user and bundled/plugin/repo skill files plus prompt-bearing files and DB prompt records.
- [x] #2 Spec defines the default threat model, trust anchors, quarantine policy, data flow, and rollout modes.
- [x] #3 Spec documents limitations for full filesystem compromise and the need for external/admin-held manifests or OS/hardware-backed keys.
- [x] #4 Spec is self-reviewed for placeholders, contradictions, ambiguity, and scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Brainstorming approved architecture, components, data flow, policy, testing, and rollout sections with user review on 2026-06-25.

PR #2523 review follow-up: renumbered from TASK-2363 to TASK-12015 after the dev rebase exposed a duplicate TASK-2363 record, and marked completed AC/DoD to match the recorded final summary.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and amended Docs/superpowers/specs/2026-06-25-context-integrity-skills-prompts-design.md covering shared context integrity controls for skills, prompt-bearing files, and DB prompt versions. The spec defines the threat model, signed trust manifest, OS/external trust anchors, anti-rollback anchoring, TOCTOU-safe runtime resolver requirements, quarantine policy, startup/runtime flows, initial enrollment, rollout modes, and verification strategy. Verification: searched the spec for TBD/TODO/FIXME/placeholder markers and self-reviewed for contradictions, ambiguity, and scope. Bandit skipped: documentation-only design change.
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
