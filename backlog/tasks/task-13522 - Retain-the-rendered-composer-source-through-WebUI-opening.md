---
id: TASK-13522
title: Retain the rendered composer source through WebUI opening
status: In Progress
labels:
- chat
- ux
- webui
- extension
dependencies:
- TASK-13457
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Approved Stage4 root continuation. Actual WebUI opener captures source text before configuration waits but lacks real form revision/H1/rendered-source lifetime; document preparation recaptures live source and bypasses guards without files. Header bypasses actual composer hydration and completion handlers can focus/report after retirement. Reuse existing Form captureHandoffGuard and actual draftReady; capture exact draft/setup once and carry the same internal predicate through document/config waits. Register the actual hydrated Form action for the header using the existing callback-ready pattern. Internal source fields stay outside serialized payload overrides. No new owner registry/transfer protocol. Automatic destination options/document qualification/receipts remain parent13457.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both header and composer WebUI open actions wait for actual owned draft hydration and refuse stale rendered New/ABA/H1/input actions.
- [ ] #2 Exact draft and initial setup plus rendered document selection stay captured through document preparation and configuration waits; retirement prevents stale opening and stale errors.
- [ ] #3 Current opening succeeds once; delayed retired or unmounted completion cannot close/focus/report in another view; no internal predicates/state leak into the URL payload.
- [ ] #4 Failure-first meaningful checks, both clients, no-added lint, independent reviews and canonical task/master/commit gates pass with native/authority limits recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Fresh ID census67 open PRs max13518/local13521/dev13510; latest remote dev7ba48 remains included. Plan Docs/Design/IMPLEMENTATION_PLAN_chat_webui_open_source_13522.md. ADR required:no, reuse ADR049/050 shared owned history/composer and existing callback-ready pattern. Bandit N/A TS/TSX/docs scope.
Red evidence: original source/input/H1 12 failures, readiness 2, retired feedback/focus 4; independent review cases 3; shared-entry retirement 2; deferred focus 1. Third registration implementation separates callback publication from stable unregistration and updates readiness only on actual transitions. green3 has all opening/lifecycle checks passing; eight broad cold-recovery assertions also fail against committed HEAD baseline because real auth model-cache invalidation writes a tombstone. Recovery assertions now validate/exclude only that legitimate tombstone, retaining all other writes; temporary baseline fixture removed. No cache implementation changed. Final matrix exposed mock reset placement outside original describe; moved reset to global beforeEach so the added behavior regressions cannot see prior preparation calls.
Verified bounded checkpoint: seven suites253 checks pass; raw payload exclusion and opener hook-dependency recheck10 checks pass. WebUI typecheck and extension entrypoint/config compile pass; no added lint across8 touched TS/TSX files; task normalize/diff checks pass. Both independent final reviewers clear the bounded source and declared fixture boundaries. Internal source state/predicate never serializes. Bandit not applicable: authored implementation TS/TSX, no Python code changed. Task remains In Progress while integrating new live dev3ca1ff055b and reassessing impacted chat contracts.
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
