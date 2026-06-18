---
id: TASK-2383
title: Add remaining Workspace membership adapters for ACP-adjacent resource domains
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-18 03:50'
labels:
  - workspace
  - acp
  - backend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/2378'
documentation:
  - Docs/superpowers/specs/2026-06-17-workspace-membership-adapters-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track implementation for GitHub issue #2378. Add or explicitly defer remaining Workspace membership resource adapters for prompt/workflow/watchlist and ACP/sandbox runtime-binding domains while preserving fail-closed unsupported types and the invariant that membership is not a trust source for ACP, Sandbox, MCP, or file access.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 prompt/workflow/watchlist/acp_session/sandbox_session are accepted resource types and resolve through domain-owned adapters
- [x] #2 note and acp_run remain fail-closed as deferred resource types
- [x] #3 prompt summaries do not expose prompt content, workflow summaries do not expose definitions, and watchlist summaries do not expose objectives
- [x] #4 runtime session memberships validate against workspace runtime binding descriptors and do not grant ACP/Sandbox/MCP/file trust
- [x] #5 forward and reverse API membership routes pass optional domain DB handles and workflow tenant/admin metadata
- [x] #6 focused tests and Bandit verification are recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-17-workspace-membership-adapters-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Workspace membership adapters for prompt, workflow, watchlist, acp_session, and sandbox_session.

Optional Prompts/Workflows/Watchlists DB handles now flow through forward and reverse membership API routes. Workflow resolution receives tenant/admin request metadata.

Runtime session adapters validate Workspace runtime binding kind/domain and expose only redacted descriptor metadata.

Deferred note and acp_run support remains fail-closed.

PR #2385 review pass:
- Rebased on origin/dev; branch was already up to date.
- Moved workflow admin/tenant metadata derivation into core Workspaces helper and normalized missing tenant_id to an empty string.
- Replaced per-request WorkflowsDatabase construction with a cached optional dependency and shutdown cleanup hook.
- Switched ACP/Sandbox session adapters from concrete inheritance to composition.
- Made unlink cleanup independent of live prompt/workflow/watchlist domain adapter availability, with best-effort prompt canonicalization when Prompts DB is available.
- Addressed Gemini status-code comments with raw 401/403 checks despite existing imports.

Review-pass verification:
- RED tests failed for the expected review issues before implementation.
- Focused suite: 125 passed, 6 warnings.
- compileall on touched modules passed.
- git diff --check passed.
- Bandit touched scope: 0 findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Supported ACP-adjacent Workspace membership adapters are implemented, documented, and updated for PR #2385 review feedback. Focused verification passed: 125 Workspace membership/context tests, compileall on touched modules, git diff --check, and Bandit on touched backend/API/service scope with 0 findings.
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
