---
id: TASK-12784
title: Harden Governance core review findings
status: Done
assignee: []
created_date: 2026-06-23 14:39
updated_date: 2026-06-24 03:46
labels:
- governance
- review
- bugfix
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address validated current-code review findings in tldw_Server_API/app/core/Governance. Scope includes store-backed rule loading, strict candidate action validation, candidate updated_at preservation, gap scope dedupe hardening, and metadata category normalization.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 GovernanceService default runtime can load active candidates from GovernanceStore.
- [x] #2 Invalid candidate actions fail safely instead of propagating unknown actions.
- [x] #3 Mapping candidates preserve updated_at for resolver tie-breaking.
- [x] #4 Gap dedupe cannot collide null scope with invalid sentinel values.
- [x] #5 Null/non-string metadata category does not become category none.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused failing tests for each reviewed behavior.
2. Implement the narrow GovernanceStore and GovernanceService fixes needed to pass those tests.
3. Run focused Governance tests plus Bandit on the touched Governance source.
4. Update this task with verification notes and final summary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Manual Backlog task created because the Backlog MCP workflow was unavailable and the CLI create/list commands hung in this environment; user approved this temporary exception.

Verification before initial PR creation:
- Red run before implementation: `python -m pytest tldw_Server_API/tests/Governance/test_governance_service.py tldw_Server_API/tests/Governance/test_governance_gap_dedupe.py -q` failed on the expected 6 reviewed defects.
- Focused isolated unit run after fixes: `python -m pytest --confcutdir=tldw_Server_API/tests/Governance tldw_Server_API/tests/Governance/test_governance_service.py tldw_Server_API/tests/Governance/test_governance_gap_dedupe.py -q` passed, 12 passed.
- Full isolated Governance unit run: `python -m pytest --confcutdir=tldw_Server_API/tests/Governance tldw_Server_API/tests/Governance -q` passed, 26 passed.
- Direct MCP governance regression run: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_governance_module.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_governance_preflight.py -q` passed, 15 passed.
- Compile check passed for touched Governance source/tests with `python -m py_compile`.
- Whitespace check passed with `git diff --check` on touched files.
- Bandit completed on `tldw_Server_API/app/core/Governance` with 0 findings; report: `/tmp/bandit_governance_2415.json`.

PR #2456 review follow-up after rebase onto latest origin/dev:
- Addressed review comments by centralizing `InvalidGovernanceCandidateError`, catching integer/timestamp conversion edge cases, adding helper docstrings, using name-based PRAGMA column access, logging knowledge-query candidate load failures, treating malformed metadata scope IDs as global candidate lookups, and removing duplicate async test markers.
- Focused review regression run passed: `python -m pytest --confcutdir=tldw_Server_API/tests/Governance tldw_Server_API/tests/Governance/test_governance_service.py tldw_Server_API/tests/Governance/test_governance_gap_dedupe.py -q` passed, 15 passed.
- Full isolated Governance unit run passed: `python -m pytest --confcutdir=tldw_Server_API/tests/Governance tldw_Server_API/tests/Governance -q` passed, 29 passed.
- Direct MCP governance regression run passed: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_governance_module.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_governance_preflight.py -q` passed, 15 passed.
- Ruff check passed on touched Python files with `python -m ruff check`.
- Compile check passed for touched Governance source/tests and `exceptions.py` with `python -m py_compile`.
- Whitespace check passed with `git diff --check`.
- Bandit completed on `tldw_Server_API/app/core/Governance` plus `tldw_Server_API/app/core/exceptions.py` with 0 findings; report: `/tmp/bandit_governance_pr2456_rebase.json`.

Known verification note: one repository-global pytest invocation and one single-test invocation were interrupted during the initial fix pass after pytest cleanup/import hooks stalled; the same behaviors were verified using isolated `--confcutdir` unit invocations.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented store-backed governance rule candidates by default, added an additive `action` schema migration for existing governance databases, preserved rule timestamps for resolver tie-breaking, made malformed loaded candidate actions deny instead of propagating unknown actions, normalized blank text scopes and rejected invalid numeric scope sentinels, and fixed metadata category handling so null/non-string values do not become real categories. PR #2456 follow-up also centralized the candidate exception, hardened bool/overflow timestamp and integer coercion, made malformed caller metadata fail closed to global rule matching instead of warn fallback, added diagnostic logging for knowledge-query candidate load failures, removed duplicate async test markers, and added regression tests for the new review comments.
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
