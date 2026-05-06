---
id: TASK-94
title: Show sandbox host-local runtime warnings in admin UI
status: Done
assignee: []
created_date: '2026-05-06 01:28'
updated_date: '2026-05-06 01:36'
labels:
  - sandbox
  - frontend
  - admin
dependencies: []
documentation:
  - Docs/Sandbox/sandbox-runtime-capability-inventory.md
  - Docs/API-related/Sandbox_API.md
  - tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py
  - Docs/superpowers/plans/2026-05-06-sandbox-host-local-warnings-ui.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved next sandbox roadmap slice by showing host-local sandbox runtime warnings in the existing admin monitoring UI. The UI should consume the current admin runtime diagnostics endpoint, keep the surface read-only, and make weaker host-local isolation for seatbelt/worktree visible to operators without creating a new dashboard route.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Admin UI surfaces sandbox runtime diagnostics without adding a new dashboard route.
- [x] #2 Host-local runtime warnings for seatbelt and worktree are visible to operators with clear weaker-isolation language.
- [x] #3 Focused frontend tests cover mocked diagnostics payloads for seatbelt and worktree.
- [x] #4 Verification includes targeted frontend tests, touched-file checks, and Bandit skip or run rationale.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect existing admin monitoring/server UI and API helper patterns. 2. Add a focused sandbox diagnostics data contract/helper if needed. 3. Write failing frontend tests for host-local warning rendering. 4. Implement the minimal admin UI section showing sandbox runtime status and warning badges. 5. Run targeted tests and touched-file checks; update docs/task notes.

Review fix: verify Qodo's 403 diagnostics finding, add a focused failing test that a 403 from sandbox diagnostics renders an access-denied card state rather than the generic unavailable state, then keep 404/optional endpoint behavior unchanged while reusing existing admin error classification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a read-only Sandbox Runtime Isolation card in MonitoringDashboardPage backed by tldwClient.getSandboxRuntimeDiagnostics(). The card summarizes readiness counts, lists runtime rows, and emits an explicit host-local warning when diagnostics reports seatbelt/worktree or other host-local warning runtimes. Added focused Vitest coverage with mocked seatbelt/worktree diagnostics. Verification: first focused Vitest run failed on the missing card as expected; after implementation the focused Vitest file passed; git diff --check passed; bun run verify:openapi passed after leaving the sandbox admin endpoint out of ClientPath because the current OSS OpenAPI verifier does not publish that admin route and AllowedPath remains intentionally wide. Bandit skipped because only frontend TypeScript, docs, and backlog files changed.

PR #1336 review fix: verified Qodo's sandbox diagnostics 403 finding. The root cause was that the sandbox diagnostics card stored only a string error and always rendered the fixed "Sandbox diagnostics unavailable" title. Added a RED Vitest case for a 403 diagnostics response, then changed the card to preserve a small error state with title, sanitized description, and severity. 403 now renders "Sandbox diagnostics access denied"; non-forbidden failures keep the unavailable path. Verification: focused Vitest file passed 8/8, git diff --check passed, and bun run verify:openapi passed with the existing reviewed exceptions.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added admin Monitoring UI propagation for sandbox host-local runtime warnings. Operators now see a Sandbox Runtime Isolation card with runtime readiness, boundary class, VM-grade status, untrusted eligibility, warning badges, and recommended action. Host-local runtimes from diagnostics, including seatbelt and worktree, show a warning that they are not VM-grade isolation and are not eligible for untrusted code. Added a focused mocked diagnostics test for seatbelt/worktree. Verification: red/green focused Vitest, final focused Vitest pass, git diff --check, and verify:openapi. Bandit skipped as non-Python/frontend-docs-only.

Review follow-up: sandbox diagnostics 403 responses are now labeled as access denied instead of generic unavailable diagnostics, while sanitized descriptions and optional-endpoint behavior remain scoped to the card. Added regression coverage for the 403 path.
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
