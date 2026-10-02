---
id: TASK-13408
title: Prepare and publish 0.1.46 from latest dev
status: In Progress
assignee: []
created_date: '2026-10-02 00:22'
updated_date: '2026-10-02 04:05'
labels:
  - release
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Release remote dev 3304f6cf0c2cab8836c6a73308d7210534a759d9 through PR #3073 as 0.1.46 using existing metadata, protected-source records, CI and publication workflows. No new collectors or broader certification gates; preserve unrelated local work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Version, changelog and protected-source records agree on the frozen source
- [ ] #2 Existing required checks pass on the release candidate
- [ ] #3 Merge main, publish v0.1.46 using existing workflows and synchronize main into dev
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Freeze latest dev and prepare release metadata/source record. 2. Run existing release/licensing tests and PR CI; fix actual release failures. 3. Publish reviewed main merge and synchronize dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
CLI task creation failed with Maximum call stack size exceeded; using official Python MCP with explicit isolated project path.
Prepared0.1.46 from5f3ed81e88ec44750a5baa7838f7c69672e32694. Protected source: `5f3ed81e88ec44750a5baa7838f7c69672e32694`. Protected manifest SHA-256: `2f10da6c5c91e30356d15568d427e3e8dc2750123ebd48d1e5c2580cf958ab47`.7451files; proposed date2026-10-01/Countdown2028-10-01T12:00:00Z. Existing helper baseline46tests pass. Added only version/changelog/license records, compact merged-change inventory and release plan; existing publication workflows unchanged. No v0.1.46 tag/release exists. Python minimum changed by included dev commits to3.12, documented as upgrade requirement.
80 existing release/helper/Makefile/docs/licensing tests passed including strict MkDocs and TLDW_VERIFY_RELEASE_SOURCE=1 checkout equality. Touched main.py version change compiles; Bandit0findings/errors. Ruff9baseline findings are unchanged; no unrelated formatting repair. Metadata tests use existingPython3.11venv; runtimePython3.12 validation uses existingCI. Preparing main releasePR; no application features/workflow changes/new certification gates.
Opened release PR https://github.com/rmusser01/tldw_server/pull/3074; attached to release chat. Latest dev moved to 3304f6cf0c2cab8836c6a73308d7210534a759d9 via backlog-only PR #3073. Incorporate that merge and update frozen source/inventory before final CI. Protected source content remains unchanged.
Merged latest dev PR #3073 (backlog-only), refreshed protected source revision to 3304f6cf0c2cab8836c6a73308d7210534a759d9 and inventory to 980 commits. Protected manifest unchanged. All 80 existing release/docs/licensing/helper contracts passed again with source verification enabled.
Requester instructed: 'update the changelog and use that'. Use expanded 0.1.46 changelog for PR Change summary and GitHub release notes; requester explicitly chose this instead of a separate human-written summary. Do not falsely label the text human-authored. CI currently reports three CodeQL JavaScript alerts (2695-2697); investigate under the existing release checks stage.
Expanded 0.1.46 CHANGELOG with merged Persona Buddy/startup, VN recovery, authenticated Email uploads, encrypted agent-message storage slice, Python 3.12/FastAPI upgrade, Resource Governance, Chat/PostgreSQL/auth/workflow and CI/platform repairs. Reused it for PR Change summary and prepared GitHub release notes as requester directed. Existing changelog/docs/helper tests: 62 passed, four inherited warnings. No Python changes; prior Bandit zero findings remains applicable. Independently traced CodeQL 2695-2697: no code change justified. All production handoff scopes explicitly remove apiKey (useHomeMilestoneScope); SHA256 journey expressions assert ownership fingerprints, not password verifiers. Recorded false-positive dismissals with rationale. Generic FNV scope migration is excluded because it is unrelated and would invalidate persisted sessions. Legal-date review remains unresolved.
Requester explicitly approved release date 2026-10-01 and Countdown start 2028-10-01T12:00:00Z ('dates approved'). Earlier requester instruction selects updated changelog as the release summary. Proceed to normal protected merge/publication once required checks pass on d659f77d7ecd5975858634912c89c19bc08fa61b. Later dev PR #3072 is backlog-only and landed after the frozen candidate; release source remains reviewed snapshot 3304f6cf0c2cab8836c6a73308d7210534a759d9. Commit tracking closure with main-to-dev synchronization to avoid rerunning CI solely for approval bookkeeping.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Existing admin privacy tests: 9 passed; audio authentication/status tests: 16 passed. Qodo traceback suggestion conflicts with reproduced UAT309/UAT446 redaction controls; no-auth audio suggestion conflicts with intentional PR #3058 security policy, with Chatbook follow-up TASK-13416 already tracked on dev. Existing make pypi-check (installed tools, --no-isolation) passes wheel/sdist, Twine and backend-only artifact boundary. Optional Watchlists Extension E2E navigation check passed on prior head with identical application/test source; reran failed job once without changing source or gate policy. Python CodeQL scan has reported another result; investigate before publication.

Python CodeQL trace: 2698 upload sanitization uses server-owned staged files whose callers enforce temp-root containment; 2699 canonicalizes before component-based base containment; 2701 SHA256 is bounded in-memory validated-principal cache metadata, not a password verifier. No code change justified for those paths. Alert2700 is actionable for automatic initialization output: make generated secret display opt-in, keep automatic missing-key generation quiet, and preserve explicitly requested interactive manual copy output. Add focused output regressions, run initializer/path/cache checks, scoped Ruff/Bandit, and one independent patch review before pushing.

Minimal CodeQL2700 fix hides values for default/automatic generated keys and preserves explicitly selected interactive manual display. Two regressions failed before the fix; all 17 initializer tests pass after it. Production Ruff and Bandit clean; test Ruff I001 matches HEAD, Bandit assertions are expected existing pytest style (B101). Existing storage/governance tests46passed and upload security7passed support false-positive dismissals2698/2699/2701. Approved legal dates now recorded in plan and changelog-derived release notes include this fix. Await independent candidate review and final metadata tests before commit.

Independent reviewer demonstrated a surviving secret leak on automatic .env write failure: outer CLI logger.exception attaches generated-key frame locals, while str(exception) can also carry sensitive text. Address the same secret-output invariant by logging/printing only exception class without traceback; exercise the actual CLI handler with a failing initialization control before final commit.

Final focused verification: 18 initializer tests pass, including three output regressions; 80 metadata/docs/licensing/helper checks pass again. Single independent review reproduced CLI failure traceback disclosure; bounded exception-class console/log diagnostics close the confirmed sibling path, exit1 preserved. No reproduced generated-key leak from the separate Postgres handler, so left unchanged. Final production initializer Ruff and Bandit clean; new test assertions are expected B101, inherited test I001 unchanged. Push final candidate and await normal six gates, trusted license and CodeQL result before merge/publication.

Final-head backend, coverage, E2E, security and container gates pass; frontend eight shards pass, reporter pending. CodeQL dynamic/default replacement2702 points only to explicit guarded manual-copy output and is mitigated with 18 output controls; both PR-head/merge refs have no open findings. Full-CI sync-pc-rest has one failed concurrency handshake among447passed: installer committed.wait(5) unset. Exact test passes locally (1passed24deselected); owning-file reproduction in progress. Direct job retry is blocked by GitHub HTTP403 until enclosing run finishes, so do not repeat request before state changes. No Sync source or tests edited; investigate/confirm before any repair.

Sync concurrency investigation: isolated test1passed; initial owning-file run stopped on Docker container cleanup timeout rather than a Sync assertion. Supported TLDW_TEST_NO_DOCKER=1 fixture mode completed owning file with19passed6fixture-reportedPostgreSQLskips. No source/test edits and no guessed timeout increase. Await enclosing CI completion to perform one targeted Linux3.12 failed-job retry; preserve source and investigate if repeat fails. Container gate nowgreen, frontendreporter running sharedUIchecks; other required gates green.

Verified all223 full-CI jobs via all REST pages: only queued FullSuiteLinux3.12 summary remained; all test jobs complete, sole failure sync-pc-rest. Canceled that summary-only attempt (no tests running/queued) to unlock native failed-job rerun, preserving completed results. Requested one specific Sync job retry with debug logging and dependent summary; immutable candidate b5c50a9b7ce16665b94c5f5e57be89783479f151 unchanged.

All six normal release gates and both CodeQL refs pass on b5. Found existing PR3078/TASK-13410 relay liveness repair while retry remained runner-queued. Temporary /tmp test plugin injects60ms before authority staging without repo edits: release failing installer-handshake test reproduced exactly, committed.wait(5) false; observed push envelope apply_status pending. This establishes related relay deadline/predecessor path rather than test-timeout repair. Reuse and verify existing narrow relay fix/tests from PR3078 in release; no new infrastructure or timing increase. Controlled failing log /tmp/release046-slow-relay-repro.log.

Reuse of PR3078 minimal two-file relay repair verified: new fake-clock regression failed before source fix (stage-only vs record/ack/finalize); same60ms-controlled installer test passes afterward. Existing relay/recovery/activation suites207passed6fixture-reportedPostgresunavailable skips; initial custom /tmp basetemp rejected trusted database roots, rerun using pytest default native temp passed. Release/docs/licensing/helper80passed again. Production/test Ruff clean; scoped production Bandit0findings0errors, diffcheckclean. Changelog includes this exact CI root fix. Independent review requested via requesting-code-review skill before pushing.

Independent read-only relay review: no actionable findings; only successfully staged current-attempt row finishes after deadline, while lease/receipt/purge/current-row guards remain and next row/batch completion retain deadline. Canceled obsolete runner-queued old-head retry after concrete root repair; updated PR3074 body and changelog. Await normal CI on new committed candidate before main merge/tag/publication.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
