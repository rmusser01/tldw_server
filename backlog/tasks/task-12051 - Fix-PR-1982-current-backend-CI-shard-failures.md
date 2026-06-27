---
id: TASK-12051
title: Fix PR 1982 current backend CI shard failures
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-06-27 21:00'
labels:
  - ci
  - pr-1982
  - tests
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1982'
  - 'https://github.com/rmusser01/tldw_server/actions/runs/28282225659'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track the current PR #1982 CI failures after the full matrix appeared on head 93fb333a09. Known groups include workflow contract drift for the watchlists extension job, tokenizer metadata test monkeypatch drift, provider readiness tests affected by CI egress env, audio artifact invalid path handling on Windows, distributed lock residual file cleanup on Windows, workflow scheduler stats, and new llm-adapters/orchestrator/chat endpoint shard failures that need log triage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

2026-06-27 PR #1982 CI follow-up:
- Current live run checked before push: 28282225659 still shows 25 failed, 737 passed, 9 canceled, 4 skipped checks.
- Local focused regression set covering the failed shards passed: 23 passed, 8 warnings.
- Workflow YAML parse passed for ui-watchlists-extension-e2e.yml and ci.yml.
- git diff --check passed.
- Bandit on tldw_Server_API/app/core/Workflows/engine.py passed with 0 findings (/tmp/bandit_pr1982_workflows_engine.json).
- Remaining action: commit and push fixes so GitHub re-runs the failed matrix against the patched branch.

2026-06-27 post-push Watchlists E2E follow-up: live PR run 28293474837/job 83829303188 reached the strict Watchlists Playwright spec and timed out in the first test after 120s; this is no longer the Chromium install failure. Root cause: workflow target wait was 90s and each test only had a 120s budget, leaving too little room for extension target discovery, storage/React/connection waits, and backend startup/model warmup. Changed the workflow target wait back to 30s, preserved .watchlists-e2e-report.json into test-results even when the strict command fails, and raised the Watchlists spec timeout constant to 180s. Verification: workflow YAML parse passed; CI workflow contract test passed; apps/extension bun run compile passed; Watchlists Playwright --list parsed and listed all 14 tests; git diff --check passed. Vitest utility tests were not used as a gate because this worktree has no extension-local Vitest config and both Bun test and inherited monorepo Vitest discovery resolve the wrong runner/config for those files.

2026-06-27 PR #1982 head b3695d3a4f follow-up: after PR #2534 merged into dev, pre-commit failed on run 28294733263/job 83832583693 because Black reformatted tldw_Server_API/cli/wizard/cli.py around the new --api-key-env Typer option. Current check scan before push showed only this one failed check (pre-commit), with 11 pass, 33 pending, and 3 skipped. Applied Black to cli.py only. Verification: python -m black --check tldw_Server_API/cli/wizard/cli.py passed; python -m pre_commit run black --files tldw_Server_API/cli/wizard/cli.py passed; git diff --check passed.

2026-06-27 Watchlists headless launch follow-up: current head 6ccc7340ca failed UI Watchlists Extension E2E run 28294893523/job 83833000391 after all 14 tests skipped. The preserved JSON report showed every skip came from launchWithExtensionOrSkip catching browserType.launchPersistentContext Timeout 90000ms while the workflow forced TLDW_E2E_EXTENSION_HEADLESS=0 under xvfb. Setup, extension build, Playwright Chromium install, backend start, and health check all succeeded; failure was specifically headed persistent-context launch. Removed the workflow's headed override so the helper uses its CI-headless default, and added a workflow contract assertion that Watchlists E2E must not set TLDW_E2E_EXTENSION_HEADLESS. Verification: workflow YAML parse passed; test_required_workflow_contracts.py::test_watchlists_extension_e2e_uses_playwright_chromium passed; Watchlists Playwright --list parsed all 14 tests; git diff --check passed. Current check scan before push showed only this one failed check, with the full matrix expanded at 56 pass, 710 pending, 3 skipped.

2026-06-27 Watchlists E2E root-cause update: current head 59b4281962 failed because headless Chromium could start but could not load/open the extension page (no service worker and page.goto chrome-extension://.../options.html returned ERR_BLOCKED_BY_CLIENT in the saved report). Comparing the passing Extension Research Workspace Parity workflow on the same head showed it uses headed Chromium plus launchWithBuiltExtensionOrSkip, whose built-extension launcher seeds storage before page load and does not wait for backend connection during launch. Watchlists still uses launchWithExtensionOrSkip, which waits for connected/offline state inside launch and converts launch/connection timeouts into skips, causing the strict no-skip job to burn all 14 tests and fail with little detail. Next fix: move Watchlists to the built-extension launcher with allowOffline, restore headed CI mode, and contract-test the workflow/spec wiring.

2026-06-27 Watchlists built-launcher fix applied: changed the Watchlists spec from launchWithExtensionOrSkip plus explicit .output/chrome-mv3 path to launchWithBuiltExtensionOrSkip via a local allowOffline wrapper, and restored the workflow to headed Playwright Chromium with TLDW_E2E_EXTENSION_TARGET_WAIT_MS=5000. Added CI contract coverage for the headed env, target wait, and built-extension launcher usage. Verification before push: focused pytest contract checks passed (2 passed); workflow YAML parsed successfully; bun run compile passed in apps/extension; bun run test:e2e:watchlists -- --list found all 14 tests; git diff --check passed; Bandit with B101 excluded found no non-assert issues in the touched pytest contract file. Raw Bandit still reports existing pytest assert_used findings across the contract file. Live PR check immediately before push showed one failed check, UI Watchlists Extension E2E / Watchlists Extension E2E (No Skips), with 125 passed, 640 pending, and 4 skipped.

2026-06-27 tokenizer metadata CI follow-up: current PR #1982 run 28300003793/job 83847033571 failed Full Suite shard (Ubuntu / Python 3.13 / gap-verified-10) in test_llm_providers_tokenizer_metadata_mirrors_strict_fields because the metadata projection test faked tokenizer resolution but left provider readiness live. In CI, Ollama readiness could disable the provider before the fake resolver ran, producing count_accuracy=unavailable instead of exact. Added a readiness stub for the tokenizer metadata projection tests and removed the existing Bandit B108 hardcoded /tmp default from the helper. Verification: focused failed test passed with CI-like env flags; full test_llm_providers_tokenizer_metadata.py passed (6 tests); Bandit on the touched test file with B101 excluded passed; git diff --check passed. Live run check before staging still showed only this one failed current-head job.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented and locally verified the current PR #1982 CI shard fixes for run 28282225659. The patch covers the watchlists extension Chromium install contract, tokenizer metadata test isolation, egress-env leakage in readiness/model metadata tests, AuthNZ schema readiness for adapter endpoint tests, Windows path/lock/artifact range assumptions, deterministic circuit-breaker recovery tests, and workflow scheduler active-count cleanup. Pre-push verification: focused pytest set passed with 23 passed and 8 warnings; workflow YAML parse passed; git diff --check passed; Bandit on app/core/Workflows/engine.py reported 0 findings. Known pending item: GitHub Actions must rerun after the push; the 25 live failures observed before this push were from the previous head commit/run, not from the patched commit.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
