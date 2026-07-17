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
- tldw_Server_API/app/api/v1/router_groups/content.py
- tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py
- tldw_Server_API/app/core/AuthNZ/initialize.py
- tldw_Server_API/app/core/config.py
- tldw_Server_API/app/services/startup_context_integrity.py
- tldw_Server_API/tests/AuthNZ/unit/test_initialize_mcp_secrets.py
- tldw_Server_API/tests/Config/test_env_file_selection.py
- tldw_Server_API/tests/Services/test_router_groups_contract.py
- tldw_Server_API/tests/Services/test_startup_context_integrity.py
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

Verification:
- Frontend certification/helper Vitest: 134 passed.
- Extension launcher/relay/fixture Vitest: 39 passed.
- Shared UI background/deployment Vitest: 5 passed.
- Backend isolation/router/AuthNZ pytest: 203 passed.
- Extension TypeScript compile: passed.
- Frontend scoped ESLint: passed.
- Existing mocked WebUI Skills gate: 13 expected, 0 skipped/unexpected/flaky.
- Existing extension parity gate: 6 expected, 0 skipped/unexpected/flaky.
- Strict live gate: passed at apps/tldw-frontend/test-results/skills-certification/2026-07-17T03-56-33-275Z-uw38dl.
- Strict evidence: both surface postconditions passed; 23/23 extension Skills requests worker-owned; all observed DB paths were inside the disposable runtime; children closed; runtime deleted; artifact scan passed.
- Worktree Databases/system_logs.jsonl was not modified by the final live run.
- Bandit touched source and test scopes: 0 findings, 0 errors.
- git diff --check: passed.
- Final correctness review: approved with no Critical, Important, or Minor findings.

Known skips or blockers: none. The explicit live gate remains intentionally outside default PR CI.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added one strict, explicit Skills release-gate command that owns a disposable single-user backend, certifies the complete lifecycle in both the WebUI and packaged Chrome MV3 extension, proves exact background-worker request ownership, checks direct API postconditions, and always performs bounded teardown plus sanitized evidence handling.

The implementation also closes isolation defects reproduced during certification: exclusive runtime env selection now applies to central config and AuthNZ loading, the chat request schema no longer performs an independent dotenv load, and all certification database and system-log paths stay under the disposable runtime. Product Skills behavior was not changed.

All acceptance criteria and verification gates pass.
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
