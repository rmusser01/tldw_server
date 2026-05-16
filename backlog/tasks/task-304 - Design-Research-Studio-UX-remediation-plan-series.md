---
id: TASK-304
title: Design Research Studio UX remediation plan series
status: Done
assignee:
  - Codex
created_date: '2026-05-12 15:35'
updated_date: '2026-05-13 00:30'
labels:
  - design
  - research-studio
  - ux
  - webui
  - extension
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-12-research-studio-ux-remediation-design.md
  - >-
    Docs/superpowers/plans/2026-05-12-research-studio-ux-remediation-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a repo-grounded design spec for addressing the Research Studio audit findings.

The agreed direction is:
- Make `/research-studio` the canonical user-facing route while preserving `/workspace-playground` and `/workspace-studio` as aliases.
- Keep internal persisted workspace-playground identifiers stable unless separately migrated.
- Support `?tab=studio` mobile deep links.
- Hide planned work products until actionable and shift Studio to a work-product-first model.
- Improve no-source/progressive-disclosure states.
- Stage degraded-health handling without blocking on unresolved backend capability semantics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design spec is written under `Docs/superpowers/specs/` and covers every audit finding plus the user's resolved open-question decisions.
- [x] #2 Spec decomposes remediation into independently reviewable stages with dependencies and verification expectations.
- [x] #3 Spec explicitly preserves stable internal storage/export/telemetry identifiers while changing user-facing route and naming to Research Studio.
- [x] #4 Spec separates immediate degraded-health pass-through from later capability-aware health semantics.
- [x] #5 Spec includes WebUI and extension routing considerations, mobile `?tab=studio` deep-linking, work-product-first Studio IA, hidden planned products, no-source state, returning-user efficiency, accessibility, docs, and CDP/test verification scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write `Docs/superpowers/specs/2026-05-12-research-studio-ux-remediation-design.md` as the formal design spec for the approved staged remediation series.
2. Ground the spec in current repository evidence: route constants, Next/extension routing, tutorial definitions, StudioPane/WorkProductTemplateChooser behavior, readiness gate behavior, and the browser/CDP audit observations.
3. Include the revised stage breakdown: baseline/tracking, route compatibility, naming sweep, mobile `?tab=studio`, degraded-health pass-through, later capability-aware health semantics, work-product-first Studio IA, no-source/progressive disclosure, returning-user efficiency, accessibility/docs/CDP verification.
4. Explicitly call out non-goals and migration constraints, especially preserving internal storage/export/telemetry identifiers unless a separate migration is approved.
5. Review the written spec for missing audit findings, over-broad stages, and unsafe migration assumptions before committing only the spec and task record.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-05-12: Wrote `Docs/superpowers/specs/2026-05-12-research-studio-ux-remediation-design.md`. The spec covers the audit findings and user decisions, splits remediation into independently reviewable stages, preserves internal workspace-playground storage/export/telemetry identifiers, separates degraded-health pass-through from later capability-aware semantics, and includes WebUI/extension routing, `?tab=studio`, work-product-first IA, hidden planned products, no-source state, returning-user efficiency, accessibility, docs, and CDP/test verification scope. Verification: `git diff --check -- Docs/superpowers/specs/2026-05-12-research-studio-ux-remediation-design.md "backlog/tasks/task-304 - Design-Research-Studio-UX-remediation-plan-series.md"` passed. Bandit skipped: documentation/task-only change.

2026-05-12: Design spec and task record were committed together. Per brainstorming workflow, implementation planning remains gated on user review of the written spec. Spec-review subagent was not dispatched because this session's tool policy only allows spawning subagents when explicitly requested by the user.

2026-05-12: Follow-up design review found and patched four planning gaps before implementation planning: route aliases must preserve extension `search`/`hash` state; redirect-only legacy pages can be blocked by `ServerReadinessGate` unless degraded-health pass-through lands first or aliases bypass narrowly; naming sweep must inventory known handoff/entry-point callers before path edits; docs updates should target source docs rather than generated `Docs/site` unless the docs pipeline rebuild is in scope. Verification: `git diff --check -- Docs/superpowers/specs/2026-05-12-research-studio-ux-remediation-design.md` passed.

Implementation planning child `TASK-304.1` is complete. Plan artifact: `Docs/superpowers/plans/2026-05-12-research-studio-ux-remediation-implementation-plan.md`. The plan orders degraded-health pass-through before browser-visible route alias verification and splits the work into independently reviewable stages.

2026-05-13: Completed the staged Research Studio remediation series through TASK-304.10. Child tasks now cover degraded-health pass-through, canonical /research-studio aliases, mobile ?tab=studio state, user-facing naming, work-product-first Studio IA, no-source progressive disclosure, returning-user tab persistence, health-capability documentation, and release docs/verification. Final release verification is recorded in TASK-304.10.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the staged Research Studio UX remediation series. The branch now makes /research-studio canonical while preserving /workspace-playground and /workspace-studio aliases, supports mobile ?tab=studio routing and tab persistence, shifts Studio toward a work-product-first/current-only model, improves no-source/source-readiness feedback, documents degraded-health capability semantics, and updates current release docs/tests/inventories with focused verification evidence.
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
