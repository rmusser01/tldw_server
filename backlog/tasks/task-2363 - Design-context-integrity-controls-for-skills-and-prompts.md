---
id: TASK-2363
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
- [ ] #1 Spec covers user and bundled/plugin/repo skill files plus prompt-bearing files and DB prompt records.
- [ ] #2 Spec defines the default threat model, trust anchors, quarantine policy, data flow, and rollout modes.
- [ ] #3 Spec documents limitations for full filesystem compromise and the need for external/admin-held manifests or OS/hardware-backed keys.
- [ ] #4 Spec is self-reviewed for placeholders, contradictions, ambiguity, and scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Brainstorming approved architecture, components, data flow, policy, testing, and rollout sections with user review on 2026-06-25.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and amended Docs/superpowers/specs/2026-06-25-context-integrity-skills-prompts-design.md covering shared context integrity controls for skills, prompt-bearing files, and DB prompt versions. The spec defines the threat model, signed trust manifest, OS/external trust anchors, anti-rollback anchoring, TOCTOU-safe runtime resolver requirements, quarantine policy, startup/runtime flows, initial enrollment, rollout modes, and verification strategy. Verification: searched the spec for TBD/TODO/FIXME/placeholder markers and self-reviewed for contradictions, ambiguity, and scope. Bandit skipped: documentation-only design change.
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
