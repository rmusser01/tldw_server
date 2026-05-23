---
id: TASK-488
title: Implement Watchlists server-backed output delivery audio presets
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-23 02:01'
labels:
  - watchlists
  - frontend
  - backend
  - presets
  - ux
dependencies: []
documentation:
  - Docs/superpowers/plans/2026-05-22-watchlists-server-backed-output-presets.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement durable per-user Watchlists presets for monitor output, delivery, and audio configuration inside /watchlists, including backend CRUD/apply semantics, frontend service/UI support, preservation of raw advanced output prefs, focused tests, and verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Backend persists per-user Watchlists output presets with CRUD and user isolation.
- [x] #2 Apply semantics replace known output/delivery/audio fields while preserving unknown advanced output_prefs fields.
- [x] #3 Frontend /watchlists monitor form can load, save, apply, update, and delete server-backed output presets without changing cadence, scope, filters, source rules, or dedupe settings.
- [x] #4 Focused backend and frontend tests cover CRUD/apply behavior and raw preference preservation.
- [x] #5 Verification and Bandit results are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add per-user Watchlists output preset persistence with user isolation, default handling, and server-side apply semantics that preserve unknown advanced output_prefs fields.
2. Expose CRUD/apply API routes and typed schemas for saved output, delivery, and audio presets.
3. Add /watchlists JobFormModal controls for loading, saving, applying, updating, and deleting presets without mutating cadence, scope, filters, source rules, or dedupe settings.
4. Cover backend CRUD/apply behavior, frontend merge/service/modal behavior, and review-discovered edge cases with focused regression tests.
5. Record verification, Bandit results, and PR review follow-up notes before closing the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented per-user Watchlists output preset persistence in Watchlists DB with SQLite/PostgreSQL schema bootstrap, CRUD helpers, default exclusivity, user scoping, and server-side apply semantics. Added /api/v1/watchlists/job-output-presets CRUD/apply routes and schemas. Added shared UI types/services, a frontend merge helper aligned with backend behavior, and JobFormModal controls to load, save, apply, update, and delete presets inside /watchlists while preserving raw advanced output_prefs and leaving scope, filters, source rules, dedupe, and cadence unchanged. Self-review added regression coverage for legacy scalar nested output prefs and a confirmation gate before durable preset deletion.

PR review follow-up: narrowed output preset row projection error handling to log and re-raise corrupt JSON/non-object prefs instead of silently returning an empty object; mapped DB unique-index races for output preset names back to output_preset_name_exists so API routes continue returning 409; changed the frontend preset clone helper to prefer structuredClone with JSON fallback; made apply requests reject explicit null base_output_prefs while still allowing omission; added regression tests for the review items. Verification: backend Watchlists DB/API pytest suite passed with 18 tests; frontend preset/service/modal Vitest suite passed with 36 tests; static guard passed with 3 tests; Bandit wrote /tmp/bandit_watchlists_output_presets_review.json with zero findings; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented durable server-backed Watchlists output/delivery/audio presets on codex/watchlists-server-presets and addressed PR review follow-ups, including non-null apply request semantics for base_output_prefs. Verification passed: backend Watchlists DB/API pytest suite (18 tests), frontend watchlists preset/service/modal Vitest suite (36 tests), watchlists static guard (3 tests), git diff --check, and Bandit on touched backend files with zero findings in /tmp/bandit_watchlists_output_presets_review.json.
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
