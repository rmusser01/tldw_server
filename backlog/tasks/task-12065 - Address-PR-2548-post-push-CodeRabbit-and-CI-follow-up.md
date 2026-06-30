---
id: TASK-12065
title: Address PR 2548 post-push CodeRabbit and CI follow-up
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-30 02:31'
labels:
  - pr-2548
  - mcp
  - review
  - ci
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2548'
  - >-
    Docs/superpowers/plans/2026-06-28-mcp-unified-residual-ux-hardening-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track the second PR #2548 rebase pass: handle new CodeRabbit review threads, inspect post-push CI failures, update tests, verify, and push the branch on latest dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Branch is rebased onto the latest fetched origin/dev
- [x] #2 New unresolved PR review comments are verified and addressed or documented
- [x] #3 Fresh failing GitHub Actions checks are inspected and actionable failures are fixed or documented
- [x] #4 Relevant local tests, diff check, and Bandit are run
- [x] #5 Branch is pushed and final PR status is reported
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Used a temporary four-stage implementation plan for refresh, review fixes, CI follow-up, and verification; removed the temporary plan file before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Fetched latest `origin/dev` and rebased `codex/mcp-residual-ux-clean`; after `origin/dev` advanced during the first rebase, rebased again onto `5b61874981` and the local branch is `0 5` relative to `origin/dev`.
- Verified new CodeRabbit comments against current code and reproduced each behavior with RED tests:
  - package-local `/mcp/status` hard-coded `transport.mount_path` to `unknown`.
  - package-local readiness warnings exposed generic exception class names for profile/registry status failures.
  - HTTP permission details overwrote protocol-provided recovery metadata.
  - disabled high-risk module guidance hard-coded `Config_Files/mcp_modules.yaml`.
  - docs local-link validation treated root-relative API/site links as filesystem paths.
- Implemented minimal fixes:
  - Pass `Request` into the package-local status route and derive `transport.mount_path` from `request.url.path`.
  - Log generic profile/registry readiness exceptions server-side with `logger.opt(exception=True).warning(...)` and return fixed warning bodies.
  - Preserve protocol-supplied `hint`, `reason_code`, and `next_action` in `_mcp_permission_detail` before falling back to endpoint defaults.
  - Keep `write_tools_disabled` recovery metadata specific in `MCPProtocol._error_recovery_metadata` so the HTTP boundary can preserve the correct next action.
  - Use config-path-neutral disabled-module guidance (`your MCP modules config`).
  - Skip root-relative Markdown targets in the MCP docs local-link contract test.
- Inspected the four failed GitHub Actions aggregate checks. Their logs only report shard group result `cancelled`; the run conclusion is `cancelled`, so no actionable pytest/build failure was available from those aggregate logs. A new push should trigger a fresh check set.

Verification:
- RED run before implementation: 4 focused tests failed for expected reasons.
- GREEN focused rerun: 4 passed, 4 warnings.
- Affected MCP/docs/basic/http suite: 67 passed, 5 warnings.
- Full package gateway suite: 206 passed, 5 warnings.
- Docker packaging contract: 4 passed, 4 warnings.
- Docs contract rerun after root-relative link filter: 13 passed, 3 warnings.
- `git diff --check`: passed.
- Bandit touched production scope wrote `/tmp/bandit_pr2548_followup.json`: exit 0, no findings.
- Final post-rebase verification on `9cbb86936e`: docs/basic/http suite 67 passed, full package gateway suite 208 passed, Docker packaging contract 4 passed, `git diff --check` passed, and Bandit touched production scope wrote `/tmp/bandit_pr2548_rebase.json` with exit 0.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2548's branch onto latest `origin/dev` (`5b61874981`), addressed the new CodeRabbit review threads with tested fixes for package-local status warning redaction, request-derived mount path metadata, recovery metadata preservation, write-tools-disabled next-action specificity, config-neutral disabled-module guidance, and root-relative docs-link handling. Inspected the failed aggregate CI checks and found the run was cancelled with shard groups reporting `cancelled`, not an actionable pytest/build failure. Local verification passed for the focused RED/GREEN tests, docs/basic/http suite, full package-gateway suite, Docker contract, diff check, and Bandit touched production scope.
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
