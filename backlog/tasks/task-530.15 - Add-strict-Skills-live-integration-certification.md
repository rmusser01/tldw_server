---
id: TASK-530.15
title: Add strict Skills live integration certification
status: Done
labels:
- skills
- webui
- extension
- e2e
- release-gate
priority: high
parent_task_id: TASK-530
documentation:
- Docs/superpowers/specs/2026-07-15-skills-live-integration-certification-design.md
- Docs/Development/Skills_Live_Integration_Certification.md
modified_files:
- Docs/Development/Skills_Live_Integration_Certification.md
- Docs/superpowers/specs/2026-07-15-skills-live-integration-certification-design.md
- apps/extension/playwright.skills-certification.config.ts
- apps/extension/tests/e2e/skills.live-certification.spec.ts
- apps/extension/tests/e2e/skills.parity.spec.ts
- apps/extension/tests/e2e/utils/extension-build.test.ts
- apps/extension/tests/e2e/utils/extension-build.ts
- apps/extension/tests/e2e/utils/extension-id.test.ts
- apps/extension/tests/e2e/utils/extension-id.ts
- apps/extension/tests/e2e/utils/skills-certification-relay.test.ts
- apps/extension/tests/e2e/utils/skills-certification-relay.ts
- apps/packages/ui/src/entries/__tests__/background.effective-auth.test.ts
- apps/packages/ui/src/entries/background.ts
- apps/packages/ui/src/services/tldw/__tests__/deployment-mode.test.ts
- apps/packages/ui/src/services/tldw/deployment-mode.ts
- apps/tldw-frontend/e2e/skills-certification/playwright.config.ts
- apps/tldw-frontend/e2e/skills-certification/skills.live.spec.ts
- apps/tldw-frontend/e2e/utils/skills-live-certification.ts
- apps/tldw-frontend/package.json
- apps/tldw-frontend/scripts/__tests__/onboarding-uat-runner.test.ts
- apps/tldw-frontend/scripts/__tests__/skills-certification-evidence.test.ts
- apps/tldw-frontend/scripts/__tests__/skills-certification-lifecycle.test.ts
- apps/tldw-frontend/scripts/__tests__/skills-certification-profile.test.ts
- apps/tldw-frontend/scripts/__tests__/skills-certification-runner.test.ts
- apps/tldw-frontend/scripts/onboarding-uat/artifacts.mjs
- apps/tldw-frontend/scripts/onboarding-uat/processes.mjs
- apps/tldw-frontend/scripts/skills-certification/evidence.mjs
- apps/tldw-frontend/scripts/skills-certification/lifecycle.mjs
- apps/tldw-frontend/scripts/skills-certification/profile.mjs
- apps/tldw-frontend/scripts/skills-certification/run.mjs
- backlog/tasks/task-530.15 - Add-strict-Skills-live-integration-certification.md
- tldw_Server_API/app/api/v1/endpoints/skills.py
- tldw_Server_API/app/api/v1/router_groups/content.py
- tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py
- tldw_Server_API/app/core/AuthNZ/initialize.py
- tldw_Server_API/app/core/config.py
- tldw_Server_API/app/services/startup_context_integrity.py
- tldw_Server_API/tests/AuthNZ/unit/test_initialize_mcp_secrets.py
- tldw_Server_API/tests/Config/test_env_file_selection.py
- tldw_Server_API/tests/Services/test_router_groups_contract.py
- tldw_Server_API/tests/Services/test_startup_context_integrity.py
references:
- https://github.com/rmusser01/tldw_server/pull/2746
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add an explicit release-gate command that certifies the /skills lifecycle against a runner-managed disposable backend in both the WebUI and the packaged browser extension. The extension path must prove that every Skills API request is owned by the exact MV3 background service worker, with no direct-page fallback. Keep existing mocked suites unchanged, keep the gate out of default PR CI, and avoid product-code changes unless the live gate reproduces a real defect.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A single explicit command creates an isolated single-user backend profile, starts required processes, runs both surfaces sequentially, and fails rather than skips on missing prerequisites.
- [x] #2 WebUI completes empty-state, create, exact search, dry render, reload persistence, Trash, restore, second Trash, purge, and direct API postcondition checks.
- [x] #3 The packaged MV3 extension completes the target-specific lifecycle and every /api/v1/skills request is owned by chrome-extension://<extension-id>/background.js with zero page-owned fallback requests.
- [x] #4 The extension relay ledger records sanitized request ownership and outcomes, with exactly one successful create, dry execute, restore, and purge plus two successful Trash mutations.
- [x] #5 The runner attempts both surfaces, records infrastructure failures explicitly, permits at most one same-port backend restart for second-surface evidence, and never converts a crashed run into a pass.
- [x] #6 Disposable runtime data and browser profiles are removed; retained evidence is sanitized and limited to logs, JSON results, relay ledger, and failure screenshots.
- [x] #7 Focused unit tests cover environment isolation, commands, aggregation, restart, teardown, artifact redaction, profile ownership, and relay-ledger rules.
- [x] #8 Existing mocked WebUI and extension Skills suites remain unchanged and pass as regression verification.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the explicit Skills live certification runner and dedicated WebUI/MV3 surfaces in an isolated worktree.

PR review remediation:
- Added the standard ingress rate-limit dependency to every Skills API route, with a route contract test.
- Made exclusive TLDW_ENV_FILE selection fail fast when the configured path is missing or not a file; AuthNZ no longer creates a typo path.
- Removed silent JSON parse catches from the certification runner; malformed response/result data now reaches the existing categorized failure handling.
- Added accepted unit markers to the changed config/AuthNZ test modules.
- Kept final result categories mandatory and verified that missing/unknown categories fail closed.
- Confirmed the reported extension readiness TDZ does not exist: timeout is initialized before attempt(), and the synchronous-callback regression test passes.
- Independent final review found no correctness/security issues; its one low-complexity finding was addressed by removing the behavior-free responseJson wrapper.

Verification:
- Frontend certification/helper Vitest: 136 passed; final runner rerun: 47 passed.
- Extension launcher/relay Vitest rerun: 29 passed; extension TypeScript compile passed.
- Backend isolation/router/AuthNZ pytest: 206 passed.
- Full Skills API integration pytest: 90 passed.
- Frontend scoped ESLint and Prettier: passed.
- Final strict live gate passed at apps/tldw-frontend/test-results/skills-certification/2026-07-17T05-19-42-369Z-blrzs4.
- Strict evidence: WebUI and extension each had 1 expected test, 0 skipped/flaky/unexpected; both postconditions passed; 23/23 extension Skills requests were worker-owned, successful, and 2xx; children closed; runtime deleted; artifact scan passed.
- Worktree Databases/system_logs.jsonl timestamp predates and was unchanged by the final live run.
- Bandit touched source and test scopes: 0 findings, 0 errors.
- git diff --check: passed.

Known skips or blockers: none. The explicit live gate remains intentionally outside default PR CI.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added one strict, explicit Skills release-gate command that owns a disposable single-user backend, certifies the complete lifecycle in both the WebUI and packaged Chrome MV3 extension, proves exact background-worker request ownership, checks direct API postconditions, and always performs bounded teardown plus sanitized evidence handling.

The final PR review remediation also rate-limits the newly exposed Skills API, fails fast on invalid exclusive runtime env paths, prevents AuthNZ from creating typo env files, and makes malformed certification JSON fail through explicit workflow/postcondition classification. Strict result categories remain mandatory by design.

All acceptance criteria and local verification gates pass.
PR: https://github.com/rmusser01/tldw_server/pull/2746
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
