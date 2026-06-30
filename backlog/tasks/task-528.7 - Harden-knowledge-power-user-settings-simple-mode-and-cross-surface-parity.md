---
id: TASK-528.7
title: Harden /knowledge power-user settings simple mode and cross-surface parity
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-07 05:39'
labels:
  - webui
  - extension
  - knowledge
  - ux
  - accessibility
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-06-07-knowledge-power-user-settings-parity-plan.md
parent_task_id: TASK-528
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Harden /knowledge for experienced users with many sources: Basic versus Expert RAG settings, simple versus detailed layouts, provider/model controls, evidence inspection, compact workflow efficiency, and WebUI/extension parity. Do not add flashcard behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Basic and Expert RAG settings are usable, reversible, keyboard-accessible, and focus-safe.
- [x] #2 Simple/compact mode still exposes source scope, profiles, preset, web fallback, model/provider, and settings efficiently.
- [x] #3 Provider/model controls handle loading, default, manual entry, and failure states.
- [x] #4 WebUI and extension behavior is aligned except for documented setup/config differences.
- [x] #5 Responsive and accessibility checks cover desktop, mobile, and extension options surfaces.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-06-07-knowledge-power-user-settings-parity-plan.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Compact simple mode now opens a source scope and profiles dialog that reuses shared KnowledgeContextBar controls for source categories, exact document/note scope, saved profiles, preset, web fallback, answer model/provider, and advanced settings.
- CompactToolbar now summarizes exact selected document/note scope, labels source/profile access for assistive tech, and respects web fallback availability when rendering the web toggle.
- Settings/provider coverage now exercises reset defaults, Escape close, focus return, Expert All Options filtering, boolean setting updates, provider failure redaction, server default clearing, manual model entry, and restored manual model summary.
- WebUI and extension empty-recovery E2E specs were updated for the compact source scope flow. WebUI Chromium E2E passed. Extension E2E was attempted, but the WXT production build stalled before any browser test started; the stuck process tree was terminated and this is recorded as a known verification blocker.
- Bandit is not applicable for TASK-528.7 because only frontend and E2E files were touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened Knowledge QA power-user controls by giving compact/simple mode access to the same source scope, saved profile, preset, web fallback, answer model, and settings controls used in detailed mode. Added focused accessibility, compact parity, provider/model, and recovery-flow coverage. Targeted Vitest and WebUI Playwright checks pass; extension runtime E2E remains blocked by a WXT production build stall before browser execution.
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
