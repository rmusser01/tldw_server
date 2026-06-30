---
id: TASK-475
title: Run Watchlists remediation final verification gate
status: Done
labels:
- watchlists
- demo-readiness
- verification
priority: high
references:
- Docs/superpowers/plans/2026-05-20-watchlists-demo-remediation-implementation-plan.md
- Docs/Runbooks/watchlists_demo_readiness_2026_05_20.md
modified_files:
- Docs/Runbooks/watchlists_demo_readiness_2026_05_20.md
- Docs/superpowers/plans/2026-05-20-watchlists-demo-remediation-implementation-plan.md
- backlog/tasks/task-475 - Run-Watchlists-remediation-final-verification-gate.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run the final Watchlists remediation verification gate on current `origin/dev`, record exact automated evidence, and document any demo-environment proof that still cannot be claimed from the repo-only verification pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Run or explicitly document blockers for the full Watchlists frontend gates.
- [x] #2 Run or explicitly document blockers for the full Watchlists backend gate and Bandit touched-scope gate.
- [x] #3 Run or explicitly document blockers for WebUI and extension Watchlists smoke gates.
- [x] #4 Reconcile the remediation plan/runbook/backlog closeout state with current origin/dev evidence without removing existing Watchlists workflows.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Task 12 from Docs/superpowers/plans/2026-05-20-watchlists-demo-remediation-implementation-plan.md on a clean origin/dev worktree. Refresh current code evidence first, run the listed frontend/backend/browser/extension/security gates where available, record exact pass/fail/blocker evidence in the runbook and task, update stale plan checkboxes only when verified, and prepare a narrow docs/task PR if edits are needed.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Verification ran from clean worktree `codex/watchlists-final-verification`, based on `origin/dev` at `668ee4929dd2b27a786a1ca519cd22ed936486e4`.
- Frontend Watchlists gates passed from `apps/packages/ui`: `bun run test:watchlists:typecheck` reported 1 file and 3 tests passed; `bun run test:watchlists:scale` reported 7 files and 53 tests passed; `bun run test:watchlists:a11y` reported 12 files and 91 tests passed with expected mocked error-state stderr.
- Backend Watchlists gate passed: `.venv/bin/python -m pytest tldw_Server_API/tests/Watchlists -q` reported 498 passed, 9 skipped, 1 xpassed, and 147 warnings. Skips are environment-gated Watchlists integration/E2E cases.
- Bandit touched-scope gate passed: `.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/watchlists.py tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py tldw_Server_API/app/core/Watchlists -f json -o /tmp/bandit_watchlists_remediation_final.json` produced `results: []` and `errors: []`.
- WebUI Playwright smoke passed from `apps/tldw-frontend`: the first sandboxed attempt could not bind `0.0.0.0:8080` (`EPERM`), and the escalated rerun of `playwright test e2e/workflows/watchlists-demo-readiness.spec.ts --reporter=line` passed 3 tests.
- Extension Playwright smoke passed from `apps/extension`: the first headless CI-mode run skipped all 14 tests because Chromium could not keep the MV3 extension context alive in this environment; the escalated headed rerun with `TLDW_E2E_EXTENSION_HEADLESS=0` passed 14 tests, and `node scripts/assert-playwright-no-skips.mjs .watchlists-e2e-report.json` reported `passed=14 skipped=0 unexpected=0 flaky=0`.
- Manual live demo dry run was not completed in this repo-only pass. It still requires the actual demo API environment, reachable real source URLs, configured LLM/TTS providers, and chosen voices. Final playable podcast/audio claims remain gated by the runbook's Provider And Voice Preflight and Final playback gate.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Ran the Watchlists remediation final verification gate on current origin/dev and recorded the results in the runbook and implementation plan. All automated frontend, backend, Bandit, WebUI, and extension gates passed after using the required escalated runs for local server bind and headed MV3 extension launch. The only remaining demo-dependent item is the manual live dry run with real sources and provider/voice configuration before claiming final playable audio.
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
