---
id: TASK-12125
title: Implement Stage 1 Persona ambient companion
status: In Progress
assignee: []
created_date: 2026-08-24 05:42
updated_date: 2026-08-24 11:57
labels:
- persona
- persona-visuals
- buddy
- implementation
dependencies:
- TASK-12123
documentation:
- Docs/superpowers/specs/2026-08-23-persona-ambient-companion-transparent-video-design.md
- Docs/superpowers/plans/2026-08-23-persona-ambient-companion-stage-1-implementation-plan.md
priority: high
modified_files:
- apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellDock.tsx
- apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellHost.tsx
- apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellPopover.tsx
- apps/packages/ui/src/components/Common/PersonaBuddy/SpriteFrameRenderer.tsx
- apps/packages/ui/src/components/Common/PersonaBuddy/personaCompanionEngine.ts
- apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualDiagnostics.ts
- apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualRenderers.tsx
- apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualTypes.ts
- apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx
- apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.integration.test.tsx
- apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/SpriteFrameRenderer.test.tsx
- apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaCompanionEngine.test.ts
- apps/packages/ui/src/components/PersonaGarden/BuddyDraftReviewPanel.tsx
- apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx
- apps/packages/ui/src/services/persona-buddy.ts
- apps/packages/ui/src/services/persona-visuals.ts
- apps/packages/ui/src/store/persona-buddy-shell.ts
- apps/packages/ui/src/types/persona-buddy.ts
- apps/packages/ui/src/types/persona-visuals.ts
- tldw_Server_API/app/api/v1/endpoints/persona.py
- tldw_Server_API/app/api/v1/schemas/persona.py
- tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py
- tldw_Server_API/app/core/Persona/visual_service.py
- tldw_Server_API/tests/Persona/test_persona_visuals_api.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved Stage 1 implementation plan end to end using subagent-driven development and strict TDD: persistence/lifecycle hardening, behavior metadata and reviews, versioned preferences and APIs, authenticated raster asset loading, deterministic idle-only companion engine, adaptive interactions, reduced motion, grounded transient roaming, E2E coverage, documentation, Bandit, and verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Schema migrations, stores, behavior validation, immutable review fingerprints, and atomic activation satisfy Stage 1 plan Tasks 1-3 in SQLite and PostgreSQL-supported paths.
- [ ] #2 Frontend authenticated raster loading, deterministic companion engine, adaptive controls, reduced motion, transient grounded roaming, and focused Persona behavior satisfy Stage 1 plan Tasks 4-6.
- [ ] #3 Every implementation task records red-green TDD evidence, focused tests, an implementation commit, and independent specification/code-quality review.
- [ ] #4 Focused backend, frontend, E2E, lint/typecheck, Bandit, and diff verification required by Task 7 pass or any environment skip/blocker is explicitly documented.
- [ ] #5 Documentation is updated, the final whole-branch review is resolved, and the branch is ready for the repository integration workflow.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute the seven tasks in Docs/superpowers/plans/2026-08-23-persona-ambient-companion-stage-1-implementation-plan.md sequentially. Use the plan's exact interfaces and commands, with the approved design as binding authority. Stage 2 video implementation is out of scope.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 6 implementation started in isolated worktree on 2026-08-24. Controller rulings #7-#9 apply. First strict-TDD slice covers owner-scoped per-Persona preference GET/clear semantics, stale-before-copy visual revision fork API/service, and typed frontend service helpers before renderer/host/editor integration.
Task 6 implementation complete in the isolated worktree. Added the single hook-owned ambient engine host integration, authenticated/fenced raster renderer, reduced-motion still path, gesture/keyboard/touch arbitration, quiet chrome, layered optimistic Buddy settings (including per-Persona GET/Use-global clear and 409 recovery), review-before-activate Visual Garden flow, active revision fork API/editor behavior, and typed frontend helpers. Verification: backend Persona API/service suites 125/125; frontend Task 6 aggregate 238/238; touched Task 6 TypeScript diagnostics 0 (51 unrelated project baseline); Bandit 0 findings/0 errors at /tmp/bandit_persona_ambient_task6.json; git diff --check clean. Full ignored report: .superpowers/sdd/2026-08-23-persona-ambient-companion-stage-1-implementation-plan/task-6-report.md.
Task 6 review round 1 fix work started from commit 3375a3d732. Scope is limited to the seven Important findings in task-6-review-round-1.md; the two Minor findings remain deferred.
Task 6 review round 1 fixes implemented for all seven Important findings; the two deferred Minor findings remain out of scope. Real-hook integration now proves focused Space, reduced-motion authored PNG reaction, completed drag after drag suspension, and reactive transient resize re-clamping without anchor persistence. Persona preference reads/writes and click/nudge work are Persona/pack/engine-generation fenced; pointercancel discards partial drag; invalid reduced-motion transitions release animated Blobs; and the HTTP fork route admits only immutable active sources before service/file access. Verification: frontend Task 6 aggregate 258/258 across 14 files (real-host suite 4/4), backend aggregate 128/128, touched TS diagnostics 0 with unrelated project baseline remaining, Bandit 0 findings/0 errors at /tmp/bandit_persona_ambient_task6_review1.json, and git diff --check clean.
Task 6 review round 2 fixes completed for both Important rereview findings; the two deferred Minors and Task 7 remain out of scope. Layered reads now have coordination independent from user-scoped global and Persona-scoped mutations, so an in-flight global save neither strands the only Persona read nor loses its result across focus changes. Deferred drag completion now carries stable interaction identity plus engine generation and is cleared/rejected across identity replacement, cancel, new pointer, and unmount. Strict RED evidence: the two preference races failed as Off/Expressive instead of durable Roaming, and actual-engine instrumentation recorded stale drag on Persona B generation 4. GREEN: focused Host suites 56/56; full Task 6 frontend aggregate 261/261 across 14 files; touched TS diagnostics 0 against 51 unrelated baseline diagnostics; git diff --check clean. Bandit not rerun because round 2 touched no Python; prior /tmp/bandit_persona_ambient_task6_review1.json remains 0 findings/0 errors.
Task 6 review round 3 fixes the single Important A→B→A stale Persona mutation finding without touching global preference coordination, drag fencing, deferred Minors, or Task 7. Each Persona mutation now captures a focus epoch in addition to Persona ID and mutation generation; success, rollback/error, conflict refresh, and messages fail closed after any focus cycle. Strict RED: delayed error and success both overwrote the returning A read (`Off`, version 9). GREEN: both preserve `Off`, emit no stale message, and the next save uses `expected_version: 9`. Verification: focused Host suites 58/58; Task 6 frontend aggregate 263/263 across 14 files; touched TS diagnostics 0 against the unchanged 51-error unrelated baseline; git diff check clean. Bandit not rerun because no Python changed; prior Task 6 result remains 0 findings/0 errors.
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
