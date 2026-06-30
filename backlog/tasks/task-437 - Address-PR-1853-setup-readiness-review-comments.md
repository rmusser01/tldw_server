---
id: TASK-437
title: Address PR 1853 setup readiness review comments
status: Done
labels:
- implementation
- setup
- review-fix
documentation:
- https://github.com/rmusser01/tldw_server/pull/1853
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR #1853 onto latest dev, inspect live review threads and failing checks, then address actionable setup readiness comments with focused verification and thread resolution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR #1853 is rebased onto latest dev and pushed.
- [x] #2 All actionable review comments on PR #1853 are fixed or explicitly resolved as non-actionable.
- [x] #3 Focused backend/frontend verification for touched setup readiness files passes.
- [x] #4 PR review threads are resolved after verified fixes land.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Rebased branch onto origin/dev and inspected live PR #1853 review threads/check failures.

Implemented review fixes:
- Localized setup readiness screen copy and hook error keys.
- Preserved submitted hosted provider secrets only for immediate provisioning, while keeping stored previews and responses sanitized.
- Rejected preview-id provisioning when submitted secret values were no longer retained.
- Replaced the readiness profile radio `any` handler with AntD `RadioChangeEvent`.
- Removed raw provision status URL rendering from the UI.
- Kept /setup fallback to the connection onboarding wizard when readiness endpoints are unavailable.
- Updated the ingestion-first E2E entry point to the home onboarding route now that /setup is the readiness surface.
- Changed setup readiness status polling to use one in-flight refresh at a time and continue after transient failures.
- Checked prior task records, removed machine-local absolute command paths, and completed the original plan task DoD.
- Added fail-closed chat provider validation for unknown hosted/local providers before config or secret handling.
- Stopped exposing raw preview exceptions to API callers while logging server-side context.
- Moved async setup readiness handlers off direct store/config writes with thread offloading.
- Added setup readiness rate-limit dependencies to mutating first-run/admin endpoints.
- Hardened readiness store probing and load warnings, including symlink and load-error coverage.
- Made readiness store updates process-local atomic to prevent concurrent load/update/save races.
- Derived completed provision status from overlays/errors instead of always warning.
- Kept status polling lightweight by avoiding expensive recommendation calls.
- Mapped speech verification selection errors to a blocked lane with a stable blocker instead of returning 500/raw exception text.
- Converted audio installer lifecycle API tests to context-managed route-scoped TestClients.
- Preserved config file line endings while updating setup config keys.

Verification recorded before commit:
- `python -m pytest tldw_Server_API/tests/Setup/test_setup_readiness_preview.py tldw_Server_API/tests/Setup/test_setup_readiness_api.py -q` => 25 passed.
- `bunx vitest run src/components/Option/Setup/__tests__/ReadinessSetupScreen.test.tsx src/components/Option/Setup/hooks/__tests__/useSetupReadiness.test.tsx src/routes/__tests__/option-setup-readiness.test.tsx` => 17 passed.
- `bun run e2e:onboarding` against `127.0.0.1:18001` => 2 passed.
- Local setup-readiness status/profile/preview/verify calls against `127.0.0.1:18001` returned HTTP 200.
- `python -m bandit -r tldw_Server_API/app/api/v1/endpoints/setup.py tldw_Server_API/app/core/Setup/readiness_service.py -f json -o /tmp/bandit_pr1853_setup_readiness.json` => exit 0.
- `git diff --check` => exit 0.
- `bunx tsc --noEmit --pretty false` still fails on existing repo-wide TypeScript debt outside touched setup-readiness files.

Review-fix verification recorded after second review batch:
- `.venv/bin/python -m pytest tldw_Server_API/tests/Setup/test_setup_readiness_preview.py tldw_Server_API/tests/Setup/test_setup_readiness_store.py tldw_Server_API/tests/Setup/test_setup_readiness_api.py tldw_Server_API/tests/Setup/test_setup_audio_installer_lifecycle_api.py tldw_Server_API/tests/Setup/test_setup_manager_user_db_base_dir_validation.py -q --timeout=60` => 62 passed.
- `bunx vitest run src/components/Option/Setup/__tests__/ReadinessSetupScreen.test.tsx src/components/Option/Setup/hooks/__tests__/useSetupReadiness.test.tsx src/routes/__tests__/option-setup-readiness.test.tsx` => 19 passed.
- `bun run e2e:onboarding` against `127.0.0.1:18001` => 2 passed.
- Local setup-readiness status/profiles/preview/verify calls against `127.0.0.1:18001` returned HTTP 200.
- `.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/setup.py tldw_Server_API/app/core/Setup/readiness_service.py tldw_Server_API/app/core/Setup/readiness_store.py tldw_Server_API/app/core/Setup/setup_manager.py -f json -o /tmp/bandit_pr1853_setup_readiness_review.json` => exit 0.
- `git diff --check` => exit 0.
- `bunx tsc --noEmit --pretty false` still fails on existing repo-wide TypeScript debt outside touched setup-readiness files.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR #1853 was rebased onto latest `origin/dev`, review comments were addressed with focused setup readiness fixes, and the branch was verified with backend pytest coverage, frontend Vitest coverage, Bandit, diff whitespace checks, live onboarding E2E, and direct live setup-readiness API walkthroughs. TypeScript still reports inherited repo-wide baseline debt outside the touched setup readiness files.
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
