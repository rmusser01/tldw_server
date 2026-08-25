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
updated_date: 2026-08-24 05:06
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
Reopened on 2026-08-23 after post-spec review identified approved amendments: authenticated asset loading, explicit video/behavior persistence and migrations, deterministic mode/state semantics, transient roaming position, truly static reduced-motion fallback, and conversion/import hardening. Applying design-only revisions before implementation planning.
Revision verification: both JSON examples parsed; targeted assertions confirmed authenticated asset loading, explicit video/behavior storage, fail-closed preference handling, movement timing, non-animated PNG fallback, dsh-pet moves mapping, subprocess hardening, dual-database migration, and reuse of the existing binding invariant. Placeholder scan and scoped git diff --check passed. Bandit and runtime tests remain not applicable because only design/Backlog Markdown changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Documented and post-review hardened the approved two-stage Persona Buddy design: idle-only ambient behavior first, then native transparent-video creation/import. The revised spec closes authenticated asset delivery, explicit storage and migrations, deterministic mode/state precedence, transient roaming persistence, true reduced-motion stills, conversion validation/cancellation, exact dsh-pet mapping, and Chatbook fallback-projection gaps while preserving the shared renderer-neutral engine, immutable activation, mandatory raster fallback, no-runtime-model boundary, and current archive envelope. No implementation code changed.
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
