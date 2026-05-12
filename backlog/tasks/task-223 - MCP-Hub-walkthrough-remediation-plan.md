---
id: TASK-223
title: MCP Hub walkthrough remediation plan
status: Done
assignee:
  - '@Codex'
created_date: '2026-05-10 06:13'
updated_date: '2026-05-10 19:20'
labels:
  - mcp
  - webui
  - ux
  - planning
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-10-mcp-hub-walkthrough-remediation-design.md
references:
  - https://github.com/rmusser01/tldw_server/pull/1514
  - https://github.com/rmusser01/tldw_server/pull/1531
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Plan and track the two-PR remediation program from the toy MCP server walkthrough. The work should make managed external server setup usable without backend restart, make chat MCP payloads honest and consistent, and then polish setup copy, catalog guidance, diagnostics, and setup isolation ergonomics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A reviewed design spec exists under Docs/superpowers/specs for the two-PR remediation program.
- [x] #2 The plan separates end-to-end blocker fixes from setup polish and diagnostics.
- [x] #3 The plan includes backend, frontend, chat, readiness, error handling, and verification coverage.
- [x] #4 Follow-up implementation tasks exist for both PR-sized phases.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created two PR-sized child implementation tasks: TASK-223.1 for live discovery/chat/readiness blockers and TASK-223.2 for setup polish/diagnostics. Drafted approved design spec at Docs/superpowers/specs/2026-05-10-mcp-hub-walkthrough-remediation-design.md.

Spec review loop passed on first review. Reviewer status: Approved. Advisory recommendations: resolve open implementation questions at planning start; make PR 2 setup isolation deliverable explicit as docs, tests, or both; keep E2E verification path relative to apps/tldw-frontend.

No blocking spec-review issues found. Human review is the remaining process gate before transitioning to implementation-plan writing.

After human-requested design review, clarified the spec around live MCP runtime resolution via get_mcp_server(), explicit ExternalServerManager.reconcile_servers(), normal chat/raw-preview scope, executable-tool data sources, and PR 2 setup isolation deliverables. Second spec review passed with no blocking issues. Advisory notes for implementation planning: resolve refresh endpoint path and external federation module-id fallback; include delete/disable runtime reconciliation flows; enumerate exact temp-path env/config values for walkthrough isolation.

2026-05-10: Both child implementation phases are merged to dev. PR #1514 landed live external discovery refresh, chat payload correctness, and degraded readiness handling. PR #1531 landed setup-state polish, Tool Catalog recovery guidance, deployment diagnostics, toy MCP walkthrough docs, and skip-safe E2E coverage.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Design spec created, split into two PR-sized implementation tasks, reviewed, implemented, and merged through PR #1514 and PR #1531. The remediation program now covers live MCP external discovery, honest chat MCP request construction, degraded readiness entry, no-auth setup clarity, Tool Catalog recovery states, deployment diagnostics, and toy MCP walkthrough isolation. Bandit was handled in the implementation tasks where backend Python was touched; this closeout only updates Backlog task state.
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
