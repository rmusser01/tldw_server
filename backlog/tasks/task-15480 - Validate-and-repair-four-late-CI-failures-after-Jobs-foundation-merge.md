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
- [x] #4 PR #3438 records the requester explicit Change summary waiver and authorization for normal auto-merge while preserving all live review and required-check gates.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Current authorization: the human requester explicitly stated "Change summary waived for PR #3438." and requested that validated review comments and failures be addressed, the branch rebased if needed, and normal auto-merge enabled. This supersedes the initial publication restrictions below; no administrative merge, required-check bypass, hand-dispatched required gate or unrelated CI cancellation is authorized.

PR #3438, branch codex/jobs-postmerge-ci-followup. Latest verified dev cb70f0c5d134ac585c70fc30b5a99bf0ced98562 advanced from initial dfac3ca7f34a296a2dc3a3fbc340efdd1a0bae5c through unrelated WebUI PR #3400 (12 paths, no overlap with the five follow-up paths). Rebased both published commits cleanly; range-diff proves both patches unchanged and repaired test/generated-artifact blobs match previous published c3ccd5a3298b82d9b4f68deced61192e7795f5e4. Auto-merge was off before the rebase; verified remote head was c3ccd5a3298b82d9b4f68deced61192e7795f5e4. Publication will use an explicit SHA-specific lease.

Live review audit: full paginated discussion, submitted reviews and inline comments read; GraphQL confirms zero review threads and no more pages. Only bot feedback is CodeRabbit skipping the draft, not a completed review. No failing current-head check was reported; required gates passed on the old head while broader CI was still queued/running. Old-head results will not satisfy the rebased head. Existing independent source review remains applicable by exact blob identity. Refresh review/check state before arming normal auto-merge. MERGE_QUEUE is currently unset.

Plan and durable evidence: Docs/superpowers/plans/2026-11-07-pr3323-late-ci-followup.md. Original run 41692726213; jobs 126208300683 (Logging), 126208312136 (AuthNZ), 126208349817 (MCP), 126208352624 (integrations). Renumbered from local TASK-15466 before first publication after concurrent PR #3436 allocated TASK-15466 through TASK-15476; earlier local evidence paths retain their run ID. ADR required: no new architecture rule; ADR-059 governs task edits, ADR-085 the separately owned membership fixture contract.

Original repairs: private-source-module fallback lock fixture avoids reloading live writer globals, preserving real lock behavior and 0.2s/<0.5s assertions. Held-lock regression and deterministic queued-writer race reproduce red before repair, green after; independent review found no actionable findings. Canonical inventory refresh changes only three RAG import line anchors per artifact, preserving all 261 records/nine targets and scanner/guard semantics. No production, helper, workflow, dependency or security-baseline change.

Overlapping MCP/AuthNZ repairs remain separately owned by PR #3406, independently validated at 2c4a9852343eaed5b229c090e20bfca087cea43a: MCP 36 setup errors -> 36 passes, containing file 63 passes; AuthNZ six failures/six passes -> 12 real SQLite/PostgreSQL passes, no skips. Both context-removal and forbidden-effect mutations fail as expected. Neither repair is duplicated here. The optional full AuthNZ containing-file run was interrupted and had an event-loop-closed finalizer error; it is not a completed-file pass. Report: pr3323-sidecar-15466-20261107/REPORT.md in chat visualization artifacts.

Initial published-head qualification remains historical evidence: exact integrations 3,490 passed/28 skipped (17 optional curl-cffi, nine opt-in external API, two disabled-router; no unavailable-PG skips), focused scope 32 passed, quality/task ratchets 33 passed, Docs 212 passed, final committed Docs/task group 227 passed. Scoped Ruff/Black, compile, whitespace, canonical task normalization and unique IDs passed. Bandit retained three inherited B105 dummy-secret findings with zero new findings/errors, B101 excluded from baseline/head.

Broader limitations stay disclosed: two historical exact core-server-smoke runs each had 228 passes plus the unchanged Hypothesis draw-generation health check before Logging executes. Seeded probes and installed-source profiling support lazy imported-module constant scanning overhead; exact failed draws were not profiled, no CPU-load explanation established, no threshold/plugin suppression change. Historical whole-repository fail-fast: 88,180 collected; 840 passed, 12 collection skips, existing Admin backup settings-generation fixture failure separately tracked in #3406. No full-repository green claim.

Fresh affected verification will qualify the rebased source and waiver/authorization metadata before publication. Keep task In Progress until actual landing; GitHub normal auto-merge must wait for its exact-head protected gates.
Fresh post-rebase verification passed 277 tests, no failures/errors/skips (848 existing warnings): 22 Logging, ten inventory, 212 Docs and 33 quality/task-ratchet cases. Ruff, scoped Black, compilation, whitespace, canonical task normalization and unique IDs pass. Bandit against exact current dev retains three inherited B105 synthetic-secret findings with zero new findings/errors, B101 excluded consistently. Rebase source/helper/config inputs are byte-identical to prior qualified Python trees. Historical old-head hosted core-server-smoke is now confirmed SUCCESS, with no failing check reported; new-head hosted checks remain independently required. PR #3406 remains open at the same independently validated head. Final publication uses the verified c3ccd5a3298b82d9b4f68deced61192e7795f5e4 lease; mark ready and arm normal auto-merge only after the final live review/check refresh.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Validated the four original late CI failures and published PR #3438 with two unowned repairs; independently validated the other two fixture fixes in #3406 without duplication. Requester has now explicitly waived the Change summary gate for #3438 and authorized normal auto-merge after review/failure remediation and any needed rebase. Latest dev rebase is clean and patch-identical; fresh qualification and normal protected auto-merge follow, with historical verification limitations retained.
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
