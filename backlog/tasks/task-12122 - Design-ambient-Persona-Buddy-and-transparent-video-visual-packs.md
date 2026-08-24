---
id: TASK-12122
title: Design ambient Persona Buddy and transparent-video visual packs
status: Done
created_date: 2026-08-24 04:39
labels:
- persona
- persona-visuals
- buddy
- design
priority: High
references:
- https://github.com/PC2005-cloud/dsh-pet
- https://github.com/PC2005-cloud/dsh-pet/blob/main/DESIGN.md
documentation:
- Docs/superpowers/specs/2026-08-23-persona-ambient-companion-transparent-video-design.md
modified_files:
- Docs/superpowers/specs/2026-08-23-persona-ambient-companion-transparent-video-design.md
- backlog/tasks/task-12122 - Design-ambient-Persona-Buddy-and-transparent-video-visual-packs.md
updated_date: 2026-08-24 04:44
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Consolidate the approved architectural design for staged Persona Buddy ambient behavior, the shared renderer-neutral companion engine, native transparent-video packs with mandatory raster fallback, local conversion Jobs, dsh-pet import mapping, and Chatbook-compatible fallback export. This task is documentation/design only; implementation planning follows after human review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design documents Stage 1 ambient behavior and Stage 2 transparent-video creation/import as separately shippable work.
- [x] #2 Design captures approved Off/Expressive/Roaming behavior, focus, interaction, accessibility, preference layering, runtime precedence, and no-runtime-model-call boundaries.
- [x] #3 Design specifies native video manifest, pack-level companion behavior metadata, immutable activation, required sprite/static fallback, capability/failure handling, and reduced-motion behavior.
- [x] #4 Design specifies review-first local conversion Jobs, cleanup/retain-source policy, dsh-pet adapter mapping, and current Chatbook fallback-only export.
- [x] #5 Design records risks, testing strategy, migration concerns, security constraints, and explicit non-goals.
- [x] #6 Design is self-reviewed against the brainstorming decisions and committed without implementation changes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved through terminal-only architectural brainstorming on 2026-08-23. Chatbook source reviewed from sibling tldw_chatbook origin/dev Persona_Visual and Persona_Buddy modules.
Verification: self-reviewed the design against the approved brainstorming decisions; parsed both JSON examples successfully; confirmed required stage, mode, fallback, immutability, renderer, Jobs, dsh-pet, Chatbook, accessibility, and no-model-call boundaries with targeted text checks. Bandit and runtime tests skipped because this task changes documentation only. No known blockers.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Documented the approved two-stage Persona Buddy design: idle-only ambient behavior first, then native transparent-video creation/import. The spec defines the shared renderer-neutral engine, modes and preference precedence, adaptive controls and accessibility, immutable visual-pack lifecycle, strict raster fallback, fallback-first video renderer, review-first local conversion Jobs, safe dsh-pet mapping, current Chatbook fallback projection, security boundaries, tests, risks, and delivery gates. No implementation code was changed.
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
