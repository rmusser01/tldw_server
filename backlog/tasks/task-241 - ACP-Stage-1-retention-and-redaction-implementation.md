---
id: TASK-241
title: ACP Stage 1 retention and redaction implementation
status: Done
assignee: []
created_date: '2026-05-10 19:30'
updated_date: '2026-05-10 21:08'
labels:
  - ACP
  - retention
  - redaction
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1512'
  - 'https://github.com/rmusser01/tldw_server/issues/1513'
  - 'https://github.com/rmusser01/tldw_server/issues/1532'
  - 'https://github.com/rmusser01/tldw_server/pull/1548'
documentation:
  - Docs/Development/Agent_Client_Protocol.md
  - Docs/Development/ACP_Production_Readiness.md
  - Docs/Product/ACP_Agent_Orchestration_PRD.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first execution stage for ACP issue closure: enforce ACP retention cleanup (#1512) and add production-safe redacted transcript/event/artifact views (#1513) from an isolated worktree. This follows the approved staged plan on GitHub issue #1532 and should keep scope ACP-local: retention policy/enforcement, shared redaction helpers, redacted detail/events/artifacts modes, tests, and operator/readiness docs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Configured ACP retention windows are applied automatically in the app path for audit events and documented ACP session/message/artifact policy.
- [x] #2 Expired ACP sessions/messages/artifact references are purged or retained according to an explicit documented policy while non-expired drill-through remains available.
- [x] #3 Redacted ACP session detail/events/artifact views are available for authorized callers without breaking explicitly documented full-fidelity operator views.
- [x] #4 Redaction covers representative API keys/tokens/secrets, sensitive metadata keys, prompt/response payloads when policy requires it, tool args, environment-looking values, and path handling while preserving useful operational context.
- [x] #5 Focused ACP tests cover retention cutoff/preservation behavior and redacted-view secret/non-secret behavior.
- [x] #6 ACP operator docs and production readiness docs no longer carry stale caveats once verification passes.
- [x] #7 Verification includes focused ACP pytest coverage, Bandit on touched Python paths, and git diff whitespace checks.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Worktree created for implementation: .worktrees/acp-retention-redaction-stage1 on branch codex/acp-retention-redaction-stage1 from origin/dev. Main checkout is dirty and will not be edited for implementation.

Implemented in isolated worktree .worktrees/acp-retention-redaction-stage1 on branch codex/acp-retention-redaction-stage1.

Verification: focused final ACP pytest command passed with 78 passed and 5 warnings in 23.46s; targeted Stage 1 subset passed with 7 passed and 5 warnings; full ACP hardening controls passed with 10 passed and 5 warnings; ACP retention/session/config focused suite passed with 68 passed and 5 warnings.

Security: Bandit was run on touched backend Python paths and wrote a developer-local Bandit JSON report. It reported 0 high, 8 medium, 1 low findings, all at pre-existing ACP DB baseline locations outside the new retention/redaction lines; no new finding was introduced in the changed code.

Whitespace: git diff --check passed.

Post-rebase verification: branch rebased cleanly onto current origin/dev, focused ACP pytest passed with 78 passed and 5 warnings in 22.63s, git diff --check origin/dev..HEAD passed, and Bandit wrote a developer-local Bandit JSON report with the same pre-existing ACP DB baseline findings only.

Draft PR opened: https://github.com/rmusser01/tldw_server/pull/1548

Review-fix pass for PR #1548: actionable Gemini inline comments are UNC path redaction coverage and ACP audit DB singleton retention override behavior.

PR #1548 review fixes: added UNC path redaction coverage and fixed ACP audit DB singleton retention override handling. Verification: review-specific regression tests passed with 2 passed, 5 warnings; broader affected ACP tests passed with 28 passed, 5 warnings; focused Stage 1 ACP suite passed with 79 passed, 5 warnings; git diff --check passed; Bandit wrote a developer-local Bandit JSON report with 0 high, 1 medium, 0 low, the existing ACP_Audit_DB.py:170 B608 baseline finding.

PR #1548 second review-fix pass: addressing Qodo and CodeRabbit comments on managed SQLite purge transactions, audit retention cache filtering, non-negative audit retention normalization, redaction of relative/pathlike strings, plain string transcript event redaction, and portable task notes.

PR #1548 second review fixes completed: added managed transaction scopes for ACP retention purge deletes, normalized audit retention to non-negative values at config/singleton/service boundaries, pruned audit hot cache during purge, filtered endpoint in-memory audit events by retention, broadened redaction for relative and forward-slash paths, redacted plain string transcript event payloads, and sanitized task notes to remove workstation-specific paths.

Verification: review-specific regression tests passed with 6 passed and 5 warnings; affected ACP endpoint/session suites passed with 52 passed and 5 warnings; broader ACP Stage 1 suite passed with 121 passed and 5 warnings; git diff --check passed; Bandit wrote a developer-local JSON report with 0 high, 8 medium, 1 low pre-existing ACP DB baseline findings and no new changed-line findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented ACP Stage 1 retention and redaction, including PR #1548 review fixes. Added configurable ACP session retention, automatic retention maintenance for session and audit stores, support-safe redacted detail/events/artifact views, managed retention purge transactions, non-negative audit-retention normalization, audit cache retention filtering, expanded path redaction, focused regression coverage, config defaults, and updated ACP operator/readiness documentation. Verification passed for focused ACP pytest suites and git diff whitespace checks; Bandit reported only pre-existing ACP DB baseline findings outside the new changed lines.
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
