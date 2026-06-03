---
id: TASK-599
title: Rebase PR 2223 onto latest dev and resolve conflicts
status: Done
labels:
- git
- rebase
- pr-review
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR #2223 branch codex/worker-lifecycle-bridge onto the latest dev branch, resolve merge conflicts, run focused verification, and push the rebased branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fetch latest `origin/dev` and rebase `codex/worker-lifecycle-bridge` onto it.
- [x] #2 Resolve merge conflicts without dropping the newer setup route landmark behavior from `dev`.
- [x] #3 Run focused verification for the conflicted setup route and worker lifecycle suite.
- [x] #4 Push the rebased PR branch.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Rebased `codex/worker-lifecycle-bridge` onto `origin/dev`; `origin/dev` is now an ancestor of the branch head.

Resolved conflicts in:
- `apps/packages/ui/src/routes/option-setup.tsx`
- `apps/packages/ui/src/routes/__tests__/option-setup-readiness.test.tsx`

The resolution kept the newer `dev` setup route structure: loader state uses the route-level `h1`, the wizard remains the sole visible `h1` when setup is required, and `SetupRequiredPanel` receives the appropriate heading level.

Removed untracked artifacts generated during verification:
- `0`
- `tldw_Server_API/Config_Files/templates/watchlists/cti_osint_report_markdown.md`
- `tldw_Server_API/Config_Files/templates/watchlists/news_briefing_markdown.md`

Verification passed:
- `bun run test:run ../packages/ui/src/routes/__tests__/option-setup-readiness.test.tsx`: 3 passed
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Services/test_lifecycle_worker_engine.py tldw_Server_API/tests/Services/test_lifecycle_worker_specs.py tldw_Server_API/tests/Services/test_lifecycle_worker_startup_adapters.py tldw_Server_API/tests/Services/test_lifecycle_worker_catalog.py tldw_Server_API/tests/Services/test_lifespan_shutdown_sequence.py tldw_Server_API/tests/Services/test_lifespan_startup_sequence.py tldw_Server_API/tests/Services/test_lifespan_worker_runtime_state.py tldw_Server_API/tests/Services/test_main_lifecycle_contract.py tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py tldw_Server_API/tests/Services/test_shutdown_job_poller_handoff.py tldw_Server_API/tests/Services/test_shutdown_owned_job_pollers.py tldw_Server_API/tests/Services/test_shutdown_post_worker_services.py tldw_Server_API/tests/Services/test_startup_claims_rebuild.py tldw_Server_API/tests/Services/test_startup_compactor_websub_workers.py tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py tldw_Server_API/tests/Services/test_startup_notifications_abtest_workers.py tldw_Server_API/tests/Services/test_startup_optional_workers.py tldw_Server_API/tests/Services/test_startup_primary_jobs_pollers.py tldw_Server_API/tests/Services/test_startup_sidecar_owned_jobs_pollers.py tldw_Server_API/tests/Services/test_startup_study_privilege_jobs_pollers.py tldw_Server_API/tests/Services/test_startup_worker_bootstrap.py tldw_Server_API/tests/Services/test_startup_worker_groups.py -v`: 297 passed
- `bun run e2e:pw e2e/smoke/stage4-responsive-landmarks.spec.ts --grep "/setup has one route heading" --project=chromium --reporter=line`: 1 passed
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r <PR-touched app service files> -f json -o /tmp/bandit_worker_lifecycle_pr_touched.json`: 0 findings
- `git diff --check`: passed

The first Playwright attempt failed because the sandbox blocked binding the local Next.js server to `0.0.0.0:8080`; the escalated rerun passed. A broader `tldw_Server_API/app/services` Bandit run reported existing findings in unrelated service modules; the scoped PR-touched service-file run passed with 0 findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2223 onto latest `origin/dev`, resolved the setup route/test conflict by preserving the newer route heading behavior from `dev`, verified the conflicted frontend path and lifecycle worker suite, and pushed the rebased branch.
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
