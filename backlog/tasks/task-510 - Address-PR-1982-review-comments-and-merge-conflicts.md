---
id: TASK-510
title: Address PR 1982 review comments and merge conflicts
status: Done
labels:
  - pr-review
  - merge-conflicts
references:
  - https://github.com/rmusser01/tldw_server/pull/1982
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve PR #1982 review comments, inspect checks, and address merge conflicts for dev -> main. Conflict policy from requester: where main and dev overlap, keep dev content.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR #1982 merge conflicts are resolved with dev taking precedence on overlapping hunks.
- [x] #2 Still-valid review comments are addressed or documented if skipped as invalid.
- [x] #3 Relevant local validation is run and recorded.
- [x] #4 Touched scope is checked with Bandit when applicable.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- 2026-06-03: Review inventory found still-valid unresolved CodeRabbit items in `.github/workflows/sbom.yml`, `.github/workflows/vz-linux-host-gated.yml`, `.impeccable/design.json`, `DESIGN.md`, and `Dockerfiles/docker-compose.webui.yml`.
- 2026-06-03: Latest CodeRabbit outside-diff review also flagged `.github/workflows/ui-research-workspace-parity.yml` trigger filters and hidden artifact upload handling.
- 2026-06-03: GitHub Advanced Security review threads visible in the PR were either outdated or already resolved.
- 2026-06-03: `gh pr checks 1982 --repo rmusser01/tldw_server --json ...` showed pre-commit and CodeRabbit passing, CodeQL/onboarding pending or queued, and cubic skipped.
- 2026-06-03: Initially merged `origin/main` into the PR worktree with recursive `-X ours`; this kept dev for overlapping hunks but incorrectly retained non-conflicting main-only changes.
- 2026-06-03: Addressed review comments in `.github/workflows/sbom.yml`, `.github/workflows/vz-linux-host-gated.yml`, `.github/workflows/ui-research-workspace-parity.yml`, `.impeccable/design.json`, `DESIGN.md`, and `Dockerfiles/docker-compose.webui.yml`.
- 2026-06-03: Fixed MCP catalog connection validation ordering so private/reserved hosts are rejected before catalog lookup and catalog probes use the canonical catalog URL without request query/fragment secrets.
- 2026-06-03: Reproduced and fixed the failing `onboarding-docs-gate` docs discoverability tests by restoring user-guide index entries for benchmark creation, OpenWebUI import/hydration, and Flashcards Study Guide.
- 2026-06-03: Updated PR title/body to address metadata warnings.
- 2026-06-03: After pushing the merge/review fix commit, newly started frontend jobs failed at `bun install --frozen-lockfile`; regenerated `apps/bun.lock` and verified the frozen install passes locally.
- 2026-06-03: Reproduced the post-lockfile `Playground Device/A11y/Composer Gates` failure in `PlaygroundForm.signals.guard.test.ts`; updated the guard for the current submit result normalization path and verified `bun run test:playground:composer` passes locally.
- 2026-06-03: Reproduced the next `Playground Device/A11y/Composer Gates` failure in `test:playground:device-matrix`; `ResearchWorkspaceBody` had discarded status state values needed by `WorkspaceTrustPanel`. Restored the state reads and verified both `env CI=true bun run test:playground:device-matrix` and `env CI=true bun run test:playground:a11y`.
- 2026-06-03: Reproduced the remote `e2e-smoke` failure from both Ubuntu and macOS logs; app import failed because the merge left duplicate workspace route registrations for `/sources/status` and `/capabilities`. Removed the stale main-side duplicate route pair while keeping the dev-side service-capability implementation. Verified `duplicate_count 0`, `test_workspace_service_capabilities.py` (`9 passed`), and the full CI-shaped critical e2e smoke command (`15 passed, 276 skipped`).
- 2026-06-03: Watchlists Extension E2E rerun remained stuck in the same browser install step that previously timed out. The previous log showed the hang occurred after the Chrome-for-Testing archive reached 100%, so the workflow now runs extension tests against the hosted runner's system Chrome channel and verifies `google-chrome --version` instead of downloading a Playwright Chromium bundle.
- 2026-06-03: Corrected the merge strategy error by reverting the bad `-X ours` merge against parent 1. This removes main-only changes imported by the merge so the PR result is dev-over-main, while preserving the follow-up PR fixes that remain applicable.
- 2026-06-03: Addressed the remaining GitHub Advanced Security review threads by adding the repo's existing `lgtm[js/clear-text-storage-of-sensitive-data]` suppressions to E2E-only browser auth seeding sinks in smoke and real-server Playwright setup.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved PR #1982 merge conflicts and review feedback. The bad recursive `-X ours` merge was corrected so dev wins over main rather than retaining main-only changes; follow-up PR fixes that remain applicable were preserved, duplicate workspace route registrations were removed, the Watchlists E2E browser install hang was replaced with the hosted runner's system Chrome channel, remaining CodeQL review threads were handled for E2E auth setup, and validation was recorded.
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
