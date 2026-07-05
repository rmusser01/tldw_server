---
id: TASK-12159
title: Address PR 2649 first-run MCP tools review comments
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-05 05:18'
labels:
  - review
  - mcp
  - setup
  - first-run
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR #2649 onto latest dev and address Qodo/CodeRabbit review comments for the first-run MCP tools setup flow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR branch is rebased onto latest dev.
- [x] #2 Valid Qodo backend/security/test issues are fixed or documented with technical rationale.
- [x] #3 Valid CodeRabbit UI/e2e/backlog comments are fixed or documented with technical rationale.
- [x] #4 Focused backend/frontend verification, Bandit, and diff checks are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Rebased codex/first-run-mcp-tools onto origin/dev and force-pushed the updated branch. Fixed valid Qodo/CodeRabbit issues: rate limits, docstrings, actor attribution, custom validation exceptions, sanitized logging, fail-closed baseline allowlist, stored catalog-version hashing, test markers, MCP Hub validation error feedback, stale drill-target handling, router-aware onboarding links, targeted Playwright fallback, sandbox diagnostics helper extraction, and duplicated task markers. Did not cache get_setup_mcp_tools_service because it wraps the existing per-request MCP Hub factory and dynamic registry snapshots; caching risks stale config/test overrides. Resolved the review threads and replied inline with the caching rationale.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #2649 review comments after rebasing onto latest dev. Focused verification passed: backend pytest selected 72 passed; frontend Vitest 29 passed; Bandit touched backend scope 0 findings; git diff --check clean. Focused Playwright grep with localhost bind escalation exited 0, but both selected real-backend tests were skipped by environment preconditions.
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
