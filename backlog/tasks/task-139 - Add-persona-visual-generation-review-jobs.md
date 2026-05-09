---
id: TASK-139
title: Add persona visual generation review jobs
status: Done
assignee: []
created_date: '2026-05-09 00:57'
updated_date: '2026-05-09 01:12'
labels:
  - backend
  - frontend
  - jobs
  - persona
  - visuals
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 9 from Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md: Jobs-backed generated visual candidate creation, candidate review API support, optional worker startup wiring, and Persona Garden editor candidate controls. This builds on the existing persona visual service/API/editor and must keep generated candidates review-gated rather than auto-activating packs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Creates persona visual generation Jobs in the persona_visuals domain with deterministic idempotency keys
- [x] #2 Rejects generation requests for packs outside the current user and persona scope
- [x] #3 Worker fails clearly when no image-generation backend is configured
- [x] #4 Worker stores generated assets and review candidates when a fake image adapter succeeds
- [x] #5 Optional worker startup keeps persona visual generation disabled by default and registers it when enabled
- [x] #6 Candidate list/detail/review APIs return authenticated generated asset preview URLs
- [x] #7 Accepting a candidate updates a draft manifest without activating it
- [x] #8 Editor supports generation prompt target state enqueue candidate preview and accept/reject controls
- [x] #9 Backend and frontend focused tests cover the generation review workflow
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented persona visual generation Jobs helpers, worker, optional startup registration, API job/candidate endpoints, review-gated candidate merge behavior, and Persona Garden generation/review controls.

Verification: backend focused pytest for persona visual jobs/API/startup passed 14 tests; existing startup optional worker pytest passed 11 tests; VisualPackEditor vitest passed 4 tests; touched-file TypeScript filter produced no diagnostics; git diff --check passed; Bandit JSON report at /tmp/bandit_persona_visual_jobs.json had results [].

Added generated candidate detail endpoint and assertion after the first verification pass so the API surface matches the accepted list/detail/review plan text.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added Jobs-backed persona visual generation candidates with review-gated manifest merge, generated asset preview responses for list/detail/review APIs, optional worker startup wiring, and Persona Garden controls for enqueueing generation jobs plus accepting/rejecting candidates. Focused backend/frontend tests and Bandit passed.
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
