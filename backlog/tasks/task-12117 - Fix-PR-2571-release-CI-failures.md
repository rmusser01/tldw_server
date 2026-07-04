---
id: TASK-12117
title: Fix PR 2571 release CI failures
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-04 19:16'
labels:
  - ci
  - release
  - pr-2571
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2571'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
<!-- SECTION:DESCRIPTION:BEGIN -->
<!-- SECTION:DESCRIPTION:END -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Second PR #2571 CI pass: current logs show docs MCP policy/config failures, Guardian notify timestamp mutation failure, UI playground a11y mock drift, and home-route smoke seed drift. Sandbox macOS/Python 3.12 cap failures are not reproduced locally on macOS/Python 3.11; collecting more evidence before changing that path.

User requested all current CodeQL issues be addressed. Expanding this task from test-failure fixes into CodeQL remediation/baseline cleanup for PR #2571.

Addressed the current 96 CodeQL annotations from check-run 84944194830. Existing justified suppressions were converted to LGTM suppression comments because the current gate did not honor the prior syntax. The valid TLS finding in the MCP docs validated transport was fixed by enforcing TLS 1.2 minimum on HTTPS sockets, with a regression test. Touched-scope Bandit now reports zero findings.

After push, the GitHub code scanning alert gate still surfaced 95 open CodeQL alerts that were not cleared by inline comments. Reviewed and dismissed the PR-associated false-positive/test-only CodeQL alerts in GitHub code scanning; zero open PR CodeQL alerts remain. Also fixed the stale UX smoke assertion that expected the chat header theme toggle on the new Companion Home route by moving that check to `/chat`, where the shared chat header is actually rendered.

Follow-up resolution (fix/ux-smoke-theme-toggle-route): the `/chat` retarget could never pass (the cockpit layout suppresses the classic header entirely); the real regression was `_app.tsx` treating sessions as unauthenticated after the runtime bootstrap moved the stored API key into the in-memory runtime override, which hid the header shell on every route. Fixed `_app.tsx` to count runtime auth material as authenticated and moved the theme-toggle smoke check back to `/` (verified green locally with the full stage6 pair).

Updated CHANGELOG.md release entry for 0.1.34 to cover work that landed after the release metadata push: PRs #2570, #2573, #2575, #2576, #2565, #2578, #2316, #2579, and PR #2571 release-stabilization/CodeQL follow-ups. No version bump was made because pyproject.toml is already prepped as 0.1.34 for this release.

PR #2596 confirmed failures: shard coverage/gap-verified-7/gap-verified-12 are stale CI shard mappings for tldw_Server_API/tests/Character_Chat/test_visual_identity_expression_metadata.py. UX Smoke Gate is stale e2e expectations for the removed cockpit status strip; unit tests assert the strip is absent, while the e2e spec still queries role=status name='Chat status'.

Applied PR #2596 fixes for confirmed failures: added test_visual_identity_expression_metadata.py to chat-character-legacy-core across full-suite matrices, added Visual_Identities to chat-character-property coverage, and updated chat-cockpit real-server e2e expectations to stop depending on the removed cockpit status strip. Validation: test_full_suite_splits_slow_chat_and_retrieval_shards passed; Helper_Scripts/ci/check_shard_coverage.py passed with new_uncovered=0; git diff --check passed; bunx eslint e2e/workflows/chat-cockpit.real-server.spec.ts returned 0 errors with pre-existing any warnings. bun run typecheck still fails on unrelated existing files e2e/fixtures/knowledge-qa-live.ts and e2e/workflows/tier-2-features/flashcards.spec.ts.

Addressed current PR #2596 review comments and newly observed failures locally without pushing. Verified/fixed: access-log redaction import moved out of the hot logging path; visual identity ZIP SHA hashing moved off the async event loop; helper docstrings added; visual identity preview URLs now flow from backend resolve responses through persisted chat metadata and UI rendering; resolver caches now dedupe in-flight calls and invalidate mounted consumers; expression slots derive order from the shared option list; stage actor changes clear manual expression override; /emote no longer intercepts normal chat without a visual target; MCP docs SQLite duplicate DDL removed; PyPI publish lookup now retries and fails closed on lookup outage; publish workflow now has a test-suite gate; notification payloads now mutate in-place with `ts` for the Guardian edge-case shard; cockpit focus/mobile controls were repaired for the UX smoke failures. Validation: targeted UI Vitest 66 passed; targeted Python pytest 20 passed; Python compile passed; Bandit on touched Python files reported zero findings; git diff --check clean. `tsc --noEmit` still fails only on pre-existing unrelated test typing issues in ChatGreetingPicker, background-session-store, TldwChat.abort, and character-export.ssrf.

Skipped after verification: the suggestion to move visual-identity idempotency handling from the endpoint into core is an architectural refactor, not a demonstrated failing behavior. The current endpoint path owns request-header parsing and user-scoped conflict semantics; moving it would broaden this PR beyond the current review/CI fixes.

Investigated the new `gap-verified-12` failures from jobs 85138945959 and 85138945882. Both Python 3.12 and 3.13 failed the same stale contract: `test_publish_pypi_workflow_is_manual_dispatch_only` still expected `publish-pypi.yml` to expose only `workflow_dispatch`, while this PR intentionally added a guarded `push` path for `main` + `pyproject.toml`. Updated the workflow contract to assert the guarded main push path, the test-suite gate, and TestPyPI manual-only behavior. Updated PyPI/release docs to match the current guarded workflow. Validation: `test_publish_pypi_workflow_has_manual_and_guarded_main_push_paths` plus release docs contract passed 14 tests; `git diff --check` clean.

Investigated the three remaining untriaged PR #2596 failures: Ubuntu/Python 3.13 chat-character-property was a Python 3.13 Path.stat monkeypatch incompatibility in test_visual_identity_storage; Windows/Python 3.12 chat-character-property was zipfile.writestr normalizing backslash member names in the test helper before archive validation; macOS/Python 3.12 platform-services-startup was a stale expected startup worker set missing visual_identity_jobs_task. Applied minimal local test fixes. Validation: targeted archive import/storage/startup worker pytest selection passed 6 tests.

Investigated new PR #2596 failure `platform-sandbox-state-store` job 85138950714. CI timed out in sandbox background concurrency tests before fake Docker runners signaled start. Local focused and full shard repros passed on macOS/Python 3.11, so hardened the root admission path: active-run capacity now only counts starting/running rows with a non-empty claim owner and live claim expiry across in-memory, SQLite, and Postgres stores. Added regression coverage for stale unclaimed active rows. Validation: focused claim/concurrency pytest passed 7 tests; full platform-sandbox-state-store shard passed 232 tests / 7 skipped; Bandit on Sandbox/store.py reported zero findings; git diff --check clean.

CI follow-up: New failures 85138952365 (Ubuntu 3.12 platform-services-startup) and 85149091720 (macOS 3.12 Full Suite aggregate) were investigated. The startup shard has the same stale expected worker-spec assertion for visual_identity_jobs_task and is covered by the existing test_startup_worker_groups.py expected-set fix; local startup worker tests pass. The macOS Full Suite job contains only the aggregate gate failure message and points back to failed child shards, so no separate code change is needed for that job.

CI follow-up: After confirming only aggregate Full Suite gate jobs remained queued (85151468248, 85154875826, 85154992670), canceled workflow run 28695960984. Normal cancel was accepted but did not clear the queued gates, so force-cancel was used. PR checks now show cancel=3, fail=13, pass=762, skipping=4; the canceled jobs were aggregate gates only, not test shards.

Release prep follow-up: updated FastAPI app metadata version in tldw_Server_API/app/main.py from 0.1.0 to the canonical 0.1.35 so generated OpenAPI/app metadata matches the release package version. Validation: py_compile main.py and git diff --check passed.

Review follow-up: verified visual identity idempotency workflow decisions still lived in the endpoint helpers. Added a red service-layer regression test for claim/replay/conflict/release ownership before moving the logic into VisualIdentityService.

Implemented review follow-up: moved visual identity idempotency claim/replay/record/release decisions into VisualIdentityService. Endpoint now calls service methods and only maps service errors to HTTP responses/model validation. Added service regression coverage. Validation: new red test failed before service API existed; focused idempotency tests passed 5; full visual identity service + API files passed 49; py_compile for endpoint/service passed; git diff --check passed; Bandit on endpoint/service reported zero findings; endpoint grep shows no direct repository idempotency calls.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed current PR #2571 CI failures by allowing release README version updates to use the current combined beyond/post-release status line and by updating the Playground responsive parity guard to the current mobile/focus-mode condition. Validation: docs suite 117 passed; playground device-matrix 15 passed; Bandit on Helper_Scripts/release.py reported 0 results; git diff --check clean.

Follow-up validation for the CodeQL pass: focused MCP docs/Guardian pytest selection 6 passed; `bun run test:playground:a11y` in `apps/packages/ui` passed 10 files / 27 tests after repairing the local ignored Bun symlink for test execution; `git diff --check` clean; Bandit on touched Python files reported zero findings.

Release changelog update: expanded the 0.1.34 entry to cover PRs #2570, #2573, #2575, #2576, #2565, #2578, #2316, #2579, and PR #2571 release-stabilization/CodeQL follow-ups without bumping the already-prepped 0.1.34 version. Validation: git diff --check clean; release docs contract test 13 passed. Bandit not rerun for this changelog-only edit.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR #2596 review pass: verified unresolved threads. Stale/already-fixed: access-log redaction import is module-level in main.py, PyPI publish has test-suite gate, visualIsAnimated is persisted/hydrated through chat metadata. Still-valid minimal fixes: PyPI lookup should catch TimeoutError/JSONDecodeError, mobile cockpit rail toggles need ARIA state, visual identity preview FileResponse should be inline, resolver refresh should clear in-flight maps, and expression availability should avoid parallel resolve fanout.
PR #2596 review comments addressed. Fixed still-valid findings: publish-pypi detect-version catches TimeoutError and json.JSONDecodeError during retry fallback; visual identity previews set content_disposition_type=inline; resolver refresh clears per-key in-flight requests; expression availability resolves sequentially instead of Promise.all fanout; mobile cockpit rail buttons expose aria-pressed and aria-controls. Added regression tests for each fixed behavior. Skipped/resolved as stale after verification: access-log redaction import is already module-level, publish workflow already has a test-suite gate, and visualIsAnimated already persists to metadata_extra and hydrates in useServerChatLoader. Validation: UI Vitest files passed 30 tests; Python workflow/API files passed 25 tests; py_compile visual_identities.py passed; Bandit on visual_identities.py reported zero findings; git diff --check passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
