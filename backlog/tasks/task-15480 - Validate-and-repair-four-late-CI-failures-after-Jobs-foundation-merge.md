---
id: TASK-15480
title: Validate and repair four late CI failures after Jobs foundation merge
status: In Progress
labels:
- ci
- testing
- jobs-followup
references:
- https://github.com/rmusser01/tldw_server/pull/3438
- https://github.com/rmusser01/tldw_server/pull/3323
- https://github.com/rmusser01/tldw_server/pull/3406
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Validate the four failing broad CI shards on PR #3323 against live dev. Repair confirmed unowned issues without weakening assertions or duplicating fixes already owned by PR #3406. Publish a focused follow-up PR against dev with exact verification and outstanding dependency evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each of the four original failures has a verified root cause and current-dev disposition.
- [x] #2 Confirmed unowned defects have focused regressions and minimal fixes; existing PR #3406 fixes are independently validated and linked rather than duplicated.
- [x] #3 Affected CI selections, security/lint and repository guards are verified with failures and environment limitations explicitly recorded.
- [x] #4 A follow-up PR against dev links all four dispositions and remains merge-blocked pending its own human Change summary and hosted gates.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Publication delivered: draft PR #3438 on codex/jobs-postmerge-ci-followup, source commit 1d0a61a9b3fb9c24876ab2684214df9aa18c5357, verified base dfac3ca7f34a296a2dc3a3fbc340efdd1a0bae5c. Keep task In Progress for landing: the new PR needs its own human Change summary and hosted checks. No merge or auto-merge is authorized by the follow-up-creation request.

Plan/evidence: Docs/superpowers/plans/2026-11-07-pr3323-late-ci-followup.md. Original run 41692726213; jobs 126208300683 (Logging), 126208312136 (AuthNZ), 126208349817 (MCP), 126208352624 (integrations). Renumbered from local TASK-15466 before publication after concurrent PR #3436 allocated TASK-15466 through TASK-15476. Earlier local evidence paths retain their original 15466 run identifier; verified source hashes are unchanged. No other task or PR was edited.

ADR required: no. Restore existing fixture/configuration and generated-artifact contracts; ADR-059 governs task editing, ADR-085 governs the separately owned membership fixture contract.

Logging: a deterministic queued-writer initialization during shared reload reproduces the historical five-second symptom; the held-lock isolation regression fails before repair. The fixture now loads the unchanged production source into a private module namespace, preserving the 0.2s environment timeout, <0.5s assertion and ownership/cleanup checks. Independent review found no actionable findings and reproduced 5.0126s before / 0.2156s after. The exact historical initiating thread was not captured.

Inventory: canonical regeneration changes exactly three RAG import line anchors in each artifact (3387->3244; 3416->3273), inherited from ancestor 9e7019ccc585acd8e544b11a9afad3045301450f. All 261 records/nine targets are otherwise identical. Scanner byte parity and second-refresh determinism pass. No production, generator, guard, CI policy, dependency or security baseline changed.

The other two original failures remain owned by pending PR #3406, verified at 2c4a9852343eaed5b229c090e20bfca087cea43a. Independent exact-test-patch validation on current dev: MCP 36 setup errors -> 36 passes, containing file 63 passes; AuthNZ six failures/six passes -> 12 real SQLite/PostgreSQL passes, no skips. Mutating either membership context or the forbidden-effect guard fails as expected. Neither fix is on dev and neither is duplicated here. The optional full AuthNZ file run was interrupted and had an event-loop-closed finalizer error; it is not counted as a completed-file pass. Durable report: pr3323-sidecar-15466-20261107/REPORT.md in the chat visualization artifacts.

Final qualification: exact integrations selection 3,490 passed / 28 skipped / zero errors or failures (22,418 warnings, 871.64s). Skips: 17 optional curl-cffi, nine opt-in external API, two disabled-router cases; no unavailable-PostgreSQL skip. Combined changed scope 32 passes (110 warnings); quality/task ratchets 33 passes (113 warnings); Docs 212 passes (653 warnings); committed-task guards 15 passes (59 warnings). Counts overlap. Scoped Ruff/Black, compilation, whitespace, unique IDs and canonical task normalization pass. Bandit retains three inherited B105 dummy-secret findings, zero new findings/errors; B101 excluded in both baseline/head scans.

Broader limitations remain explicit: two core-server-smoke selections each have 228 passes plus an unchanged Hypothesis draw-generation health-check failure before Logging executes. Isolated seeded probes pass; installed-source profiling supports lazy imported-module constant scanning overhead, not a contextual-retrieval assertion regression. Exact failed draws were not profiled; no CPU-load explanation established and no deadlines/checks were relaxed. Whole-repository fail-fast collected 88,180 tests, then stopped at 840 passes, 12 collection skips and the existing Admin test_shared_audit_refuses_stale_per_user_file[backup] settings-generation fixture failure, separately tracked in #3406. No full-repository green claim.

Landing: draft, auto-merge off, human summary pending, exact-current-head hosted gates pending. Eight live required gates include E2E Critical (Tier 1 + Journeys) plus backend/security/coverage/frontend/e2e/container/license. MERGE_QUEUE is unset. No unrelated CI was cancelled or hand-dispatched. Publication records are committed with the related repairs.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Validated all four original late CI failures. Draft PR #3438 repairs the unowned Logging fixture reload race and stale WebScraping import inventory; independently reproduced and verified the existing MCP/AuthNZ fixes in PR #3406 without duplication. Exact integrations: 3,490 passed, 28 optional/environment skips. Focused 32, quality/task 33 and Docs 212 pass; independent review approved and Bandit has no new findings. Broader local Hypothesis and inherited Admin limitations are recorded honestly. Await this PR own human Change summary and hosted qualification before landing.
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
