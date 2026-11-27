---
id: TASK-15647
title: Implement outbound privacy regression gate Increment 1
status: In Progress
labels:
- security
- privacy
- ci
dependencies:
- TASK-15670
references:
- Docs/Design/2026-11-24-outbound-destination-inventory-design.md
- Docs/superpowers/plans/2026-11-24-outbound-privacy-increment-1.md
- Docs/ADR/097-reviewed-outbound-destinations-and-privacy-modes.md
- https://github.com/rmusser01/tldw_server/pull/3612
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the reviewed product destination and asset inventory plus the required CI and local regression check for covered startup, setup, and application runtime flows. Scope this task to Increment 1.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A versioned inventory validates exact origins and real asset identities, distinguishes declared from transport-verified evidence, and guards new usage in its covered paths.
- [x] #2 The same local and required-CI runner exercises fresh server startup, first-run setup, selected runtime behavior, and opted-in synthetic download and provider paths with no real external traffic.
- [x] #3 Guarded production browser smoke covers the shipped setup wizard, Next.js including its server process, and the packaged Chrome worker and content script. Search results use a bundled icon.
- [x] #4 License-admitted required contexts fail on violations, missing evidence, or failed coverage. Relevant changes trigger the check, diagnostics are redacted, and coverage limits are documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: yes. Proposed ADR-097 covers the durable inventory and required CI rules for this increment. The linked plan documents the five test-first stages and the supported Linux validation profile.
Plan verification on 2026-11-24: existing required-gate classifier and direct HTTP guard baseline passed 99 tests in a plugin-disabled run. Offline planning probes exercised the real download SDK fixture redirect on 0.36.0 and 1.33.0 and the real installer and provider adapter paths with synthetic transports. Linux namespace and production browser execution remain implementation acceptance checks.
Requester-approved plan review amendments: use disposable Linux test containers with disabled external networking and no host service socket mounts, require stop-file ASGI shutdown and final evidence, specify runtime Next auth and full Chromium/cache settings, verify exact SDK repository identity and zero raw attempts in both opt-in children, and trigger required checks on both projected package source trees. ADR-097 remains the governing proposed decision. Verified document links, stage order, example syntax, and review-fix coverage. Container and browser preflights remain implementation acceptance checks. No runtime or workflow code changed.
Implementation started in the isolated outbound-egress-design worktree. Stage commits will record test-first verification and limitations.
Stage 1: observed missing-validator test failure, then 73 passed with the direct HTTP guard. Bandit on the validator and its unit tests and git diff --check passed. Package registry and downstream hosts are unverified where installer argv cannot identify them.
Stage 2: 39 harness and strict-guard tests passed after red runs, and the existing guard suite delivered 27 passes and one expected subprocess-only skip. The real Linux Docker controls and guarded default smoke passed, including startup, local setup state and notes, ASGI shutdown, and zero recorded attempts. Follow-up review regression tests require the preflight at internal entry points and reject local-only config edits before image preparation. The smoke uses the ASGI shutdown-complete message because Uvicorn 0.54 changed a private field name. Bandit and Ruff passed.
Stage 3: 77 tests passed in the isolated Linux backend image (Hugging Face Hub 1.33.0, zero skips). All 11 opt-in subprocess cases also passed with Hub 0.36.0 under macOS OS-level network denial in Python 3.12. The fixtures assert both the built-in and configured OpenAI destinations and leave HF_ENDPOINT unset on the successful Hub cases. Pip/uv and Git coverage is limited to exact argv dispatch. Ruff and Bandit passed.
Stage 4: all four browser scenarios ran together in disconnected Linux Docker and passed with zero violations. Wizard persistence, quickstart WebUI cookie and bundled search icon, the packaged copilot worker POST and rendered reply, and independent content-script activation are proved. Added the app early ONNX telemetry opt-out after a native DNS attempt was reproduced and verified absent. The missing-credentials check now precedes DNS on sample OpenAI endpoints, verified by 63 provider-readiness tests. The WebUI models catalog uses four exact unavailable local provider fixtures. Only Chromium’s verified failed UDP IPv6 route query is classified as platform behavior; actual network traffic remains subject to the exact fixture rules. 18 harness tests, 35 Add Sources tests, both TypeScript checks, Ruff, Bandit, and whitespace checks passed.
Stage 5: ran both image preparations and all seven scenarios from the final checkout in disconnected Linux Docker. Backend and browser independently passed serially in 93.26 and 86.29 seconds. The final harness and workflow suite passed 395 focused tests, and independent review reran 261 CI contract and runner tests. The workflow sentinel propagated a nonzero exit. Missing and stale preparation records failed closed. Report scans found no raw credentials or URLs. actionlint 1.7.12, Ruff on harness code, Bandit with no new findings, and git diff --check passed. GitHub-hosted timing awaits the first branch run.
Pre-push review follow-up authorized. Verify three stages test-first: dedicated license-admitted CI jobs and browser path detection, Chromium worker WebSocket detection, and safe diagnostics plus exception anchor validation. ADR required: no new ADR. The follow-up implements the existing Increment 1 decisions in proposed ADR-097 and the linked implementation plan.
Pre-push stage 1: auth, shared API and smoke inputs now trigger a dedicated browser job without selecting the frontend unit shards. Both dedicated runner jobs reclaim unused SDK space and record cold build time and lowest sampled Docker free disk. Required contexts fail for a selected job that fails, is cancelled or skips. Red run reproduced the missing flag and runner topology; 255 classifier and workflow contract tests passed with the fix, as did actionlint, Ruff and whitespace checks. Bandit found no new findings. Hosted x64 metrics await GitHub execution.
Pre-push stage 2: Chromium’s bounded startup NetLog detects WebSocket attempts in both extension scenarios. A separate worker profile proves a denied hostname is recorded and the negative control stays outside the normal profile. The focused suite passed 45 tests. In disconnected Linux Chromium, the content script and three clean worker repeats passed. A separate injected startup WebSocket was rejected while the Playwright flow itself passed. One earlier run had incomplete native trace output and will be checked again in the full run. Ruff, Bandit and whitespace checks passed.
Pre-push stage 3: the public report distinguishes missing, invalid and stale preparation, containment preflight, timeout and per-case failures using allowlisted diagnostic codes. GitHub displays the same redacted output. Reviewed Python exception anchors resolve by source and definition without imports. Negative tests reproduced each defect before the fix. The final focused run passed 425 tests. The exact checkout prepared both Linux arm64 images and passed all seven guarded scenarios in 243.79 seconds after preparation. Ruff, actionlint, Bandit baseline comparison and whitespace checks passed. Actual hosted x64 disk measurements and GitHub check results await the first branch run.
Hosted CI follow-up for PR #3612 authorized. Reproduce the extension worker trace failure, repair publication of ADR-097, and integrate current dev so the RAG workflow and its checked-out tests agree. ADR required: no new ADR. Proposed ADR-097 still governs the privacy boundary.
Renumbered during the PR #3612 CI follow-up because dev already uses the original ID.
Hosted CI stage 1: red runs reproduced the untracked published ADR and the strict MkDocs warning after refreshing the mirror. Published ADR-097 and used a repository-relative code reference for its unpublished design. All 79 docs and inventory tests passed, including strict MkDocs, plus Ruff, Backlog normalization and Bandit with only existing test assert findings. The design and implementation task IDs were updated because dev already held the originals.
Hosted CI stage 2: merged dev at dde6b02bfd to align the exact admitted head with the current RAG workflow. Preserved both the chat-executors and outbound-backend rollups and the exact checkout contract. All 59 RAG and 33 chat files listed by the merged workflow exist. The originally missing RAG test file passed alongside the classifier, license and Backlog contracts (332 passed). Actionlint, Ruff, the duplicate ID gate and the published ADR index comparison passed.
Hosted CI stage 3: a deliberately incomplete native trace for the calibration profile reproduced the public trace_incomplete failure on the old fixture. The calibration now uses its own Chromium NetLog in the same disconnected environment. The real worker still requires a complete native trace and successful fixture contact. The corrected Linux arm64 replay passed, recording the authenticated request and both NetLogs. An unfinished app trace and a synthetic unexpected startup WebSocket were still rejected. An earlier local replay failed to reach the worker request and an earlier Docker cleanup timed out. The isolated replay passed on its own. The TypeScript launch-options regression and all 80 focused Python cases passed. Ruff and Bandit passed, with no new findings. The original hosted x64 failure needs the rerun on the updated PR for confirmation.
PR #3612 follow-up authorized to address the two test-quality observations. Keep both existing interaction assertions and add independent checks on the isolated runner and installer behavior. ADR required: no new ADR. Proposed ADR-097 still governs the privacy boundary.
The full test-quality enforcement command reproduced the two GitHub findings before edits. Rejected container inspection now checks that a prior report was removed while preserving Docker call order. The OmniVoice test now checks real preparation of a missing checkout parent while preserving all reviewed Git and pip arguments. Both independent assertions caught their deliberate local failures. All 58 affected pytest cases passed. Test-quality enforcement passed over 5,543 files with both new offenses gone. Bandit found no new issues, Ruff had the same two existing findings, and whitespace checks passed. GitHub required verification is pending.
PR #3612 backend-gate follow-up authorized. Reproduced the 20 non-UTC order cases and both provider catalog ratchets before edits. Restore the non-UTC guard immediately after change detection, extract the existing catalog endpoint policy evaluation, and patch DNS resolution at its defining module. ADR required: no. These fixes preserve the existing guard semantics and ADR-026 and ADR-097 boundaries.
Backend gate verification: all 317 workflow contracts and 63 provider-readiness tests passed. The full CI contracts and code ratchets step passed with 1,719 cases and two skips in its first command, then 198 passing WebSocket auth cases. The catalog listing is 652 lines and both provider ceilings are unchanged. All 496 backend unit smoke cases passed. Startup liveness passed in a loopback-only sandbox, OpenAPI fingerprint and path checks matched, and test-quality enforcement passed over 5,543 files. Actionlint, whitespace checks, and Bandit and Ruff baseline comparisons passed. Hosted exact-head verification is pending.
GitHub change detection on 334c35806d stopped before backend tests because dev reused the design task ID. The design task now uses TASK-15670, and its dependency and documentation references follow it.
Dev synchronization for PR #3612 reconciled the Notes Save changes and immutable checkout assertions from both branches. The buttons use the verified-owner explanation from dev, and keyboard Save and autosave wait for the same owner check. The required frontend context still depends on the outbound browser check. Local verification passed 34 Notes tests, 42 Chat sidebar tests, and 718 CI contract tests with one skip. Actionlint, Backlog ID, whitespace, and Bandit and Ruff comparisons passed. ADR-097 still governs the outbound privacy checks.
PR #3612 Admin Usage follow-up. A deferred metadata response reproduced the token selection race at the exact CI base. The test now waits for Ops within the Token filter and checks token ID 12. The required frontend gate needs every case in the changed file to pass, so the Usage page restores the expanded screen-reader names for token columns and the test checks the current decorative divider. All 18 Usage and organization tests passed, along with ESLint, the TypeScript check and whitespace validation. Existing accessibility and CI rules apply.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Increment 1 ships the reviewed inventory, isolated backend and browser scenarios, and the existing required CI contexts. PR #3612 follow-up publishes ADR-097, incorporates current dev and its RAG tests, and keeps the worker WebSocket calibration out of application trace collection. Local regressions verified the required failure paths. Hosted x64 confirmation is pending.
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
