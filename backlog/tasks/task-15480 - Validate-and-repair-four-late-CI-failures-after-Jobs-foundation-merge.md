---
id: TASK-15480
title: Validate and repair four late CI failures after Jobs foundation merge
status: In Progress
labels:
- ci
- testing
- jobs-followup
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Validate the four failing broad CI shards on PR #3323 against live dev. Repair confirmed unowned issues without weakening assertions or duplicating fixes already owned by PR #3406. Publish a focused follow-up PR against dev with exact verification and outstanding dependency evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each of the four original failures has a verified root cause and current-dev disposition.
- [x] #2 Confirmed unowned defects have focused regressions and minimal fixes; existing PR #3406 fixes are independently validated and linked rather than duplicated.
- [ ] #3 Affected CI selections, security/lint and repository guards are verified with failures and environment limitations explicitly recorded.
- [ ] #4 A follow-up PR against dev links all four dispositions and remains merge-blocked pending its own human Change summary and hosted gates.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan and complete evidence: Docs/superpowers/plans/2026-11-07-pr3323-late-ci-followup.md. Base dev: dfac3ca7f34a296a2dc3a3fbc340efdd1a0bae5c. Original PR #3323 run 41692726213, jobs 126208300683 (Logging), 126208312136 (AuthNZ), 126208349817 (MCP), 126208352624 (integrations).

ADR required: no. The repairs restore existing lock configuration and generated-artifact contracts without a durable architecture change. ADR-059 governs task editing; ADR-085 governs the separately owned membership fixture repair.

Logging: ordinary baseline 21 tests pass, but a deterministic queued-writer initialization during shared reload reproduces the historical five-second symptom (5.0022s against <0.5s). A new held-lock regression fails before repair because the fixture reconfigures the live held lock. Loading the unchanged source into a private module namespace fixes fixture isolation, preserving the 0.2s environment timeout, <0.5s assertion and lock ownership/cleanup checks. Independent reviewer found no actionable findings and reproduced 5.0126s before / 0.2156s after; live settings, queue, worker and sinks remain intact. The historical hosted initiating thread was not captured.

Inventory: guard fails on current dev; canonical refresh changes exactly three RAG import line anchors in each generated artifact (3387->3244 and 3416->3273), from ancestor commit 9e7019ccc585acd8e544b11a9afad3045301450f. All 261 records/nine targets are otherwise identical. Ten guard tests, exact scanner parity and second-refresh determinism pass. No scanner/guard change.

Overlapping repairs are already owned by open PR #3406 and are not duplicated. Independent disposable-checkout validation of exact test patches from 2c4a9852343eaed5b229c090e20bfca087cea43a: MCP 36 setup errors -> 36 passes, full accounting file 63 passes; AuthNZ six failures/six passes -> 12 SQLite/PostgreSQL passes, no skips. Both missing membership-write contexts and the MCP forbidden-effect guards retain mutation sensitivity. Neither repair is on dev. Optional AuthNZ containing-file run was interrupted on coordination request and had an event-loop-closed finalizer error; it is not a completed-file pass. Durable sidecar report: pr3323-sidecar-15466-20261107/REPORT.md in the chat visualization artifacts.

Qualification: final combined changed scope 32 passes (110 warnings), quality/task ratchets 33 passes (113 warnings), Docs 212 passes (653 warnings). Scoped Ruff, changed-line Black, compilation, whitespace, unique IDs and canonical task normalization pass. Bandit retains three inherited B105 dummy-secret findings, zero new findings or scan errors; B101 assertions excluded for baseline and head. No production/helper/workflow/dependency/security-baseline changes.

Limitations: two exact core-server-smoke runs each have 228 passes plus the unchanged Hypothesis input-generation too_slow health-check failure before Logging executes (25.76s and 8.85s draws). First reported seed passes with the explicit Hypothesis plugin; initial missing-plugin CLI argument error was corrected, not counted as a test result. Bounded read-only attribution remains in progress; no suppressions or deadline changes. Repository-wide fail-fast: 88,180 collected; 840 passes, 12 collection skips, one existing Admin test_shared_audit_refuses_stale_per_user_file[backup] failure (11,648 warnings), already tracked by PR #3406. No full-repository green claim. Exact integrations selection is still running.

Landing: publish a focused draft against dev; human Change summary waiver for #3323 does not carry over. Current eight required gates include E2E Critical (Tier 1 + Journeys) in addition to backend/security/coverage/frontend/e2e/container/license. MERGE_QUEUE variable is unset. No auto-merge or administrative bypass is authorized for this new PR.
Bounded Hypothesis attribution completed without edits: the installed engine uses a five-second cumulative draw budget for deadline=1000 and subtracts GC time. Original-plugin-set seeded probe passes100validexamples/117draws; slowest adjusted draw4.63s, with4.90s under lazy constant discovery,1.76s Unicode/charmap initialization,0.21sGC. Imported-module source/AST constant scanning supports suite-state-sensitive generation overhead. Exact historical failed draws were not profiled; CPU load was not established. Do not call the smoke group green or change its thresholds/suppression. No contextual-test/core source changed.
Renumbered this task from TASK-15466 to TASK-15480 before first publication because concurrently opened PR3436 allocated TASK-15466 through TASK-15476. Only this follow-up task is renamed. Earlier local evidence directories/reports retain the original 15466 run identifier; their exact source hashes and results remain applicable. No other PR or task was changed.
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
