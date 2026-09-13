---
id: TASK-13244
title: Harden Workspace Persona effective-default resolution for parity
status: Done
assignee: []
created_date: '2026-09-13 18:15'
updated_date: '2026-09-13 20:33'
labels:
  - persona
  - workspaces
  - parity
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2957'
documentation:
  - >-
    Docs/superpowers/plans/2026-09-13-persona-workspace-parity-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add privacy-safe degraded responses and logging, preserve malformed-storage diagnostics, and repair the obsolete v48 migration fixture identified during TASK-13243. First backend slice of issue 2950.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement the linked stage contract with focused regression coverage and recorded verification; do not claim completion from documentation alone.
- [x] #2 Permission-denied effective results redact identity; invalid payloads do not leak into logs; corrupt storage retains an invalid-default diagnostic; repaired migration tests retain meaningful upgrade coverage.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stage 1 started on codex/persona-workspace-resolver-hardening, stacked on planning PR #2952 after refreshing dev. Server baseline e157b6d1306a133e93595a8d457ecac76ac770fa; Chatbook baseline 392ce191fd28953550f85154ea1f8e4eda4ab7f3. Scoped diffs show no changes to Stage 1 server files or Chatbook Workspace default contracts since assessment. MCP overview read; task_view stalled and was terminated, using CLI fallback. Backend-only scope; remaining stages not started.

API TDD: 7 failed/11 passed before production changes; failures confirmed identity disclosure, validation-log content disclosure, mixed-key sorting crash, discarded corruption flag, and disabled-feature lookups. Initial API green: 18 passed. Bandit API file: zero findings. Added follow-up coverage for transient failure during deleted-profile lookup. Disabled non-null default saves return 503, while clearing remains supported. Safe logging records fixed category, known payload type, and workspace-id presence only, not arbitrary ids or input keys.

Broader API regression: 118 passed, 6 warnings across Workspace defaults API, full Workspace CRUD API, and chat conversations unit suite (70.47s). Ruff API/test scan has only the same four pre-existing BLE001 findings in unrelated endpoint handlers; baseline comparison verified. Black changed production range and API test module checked/formatted. DB/migration implementation and independent API review ongoing.

Independent API/spec review (Aquinas) returned no findings; DB half is still in progress. Review did not claim DB verification.

Full touched-production Bandit scan (workspaces.py and ChaChaNotes_DB.py): 0 findings, 0 errors; existing nosec suppressions retained unchanged. DB module and both focused test files pass Ruff. Additional combined run includes the Notes v59 migration guard suite to ensure fixture repair does not weaken migration safeguards.

Formatting caveat: Black --check --line-ranges on the large ChaChaNotes_DB.py reports only pre-existing whitespace/indentation outside the changed method (lines around 7732, 7774, 32373-34176). No broad formatting applied; changed method will be checked separately. Both focused test modules are Black-clean.

DB implementation complete: 41 focused tests passed. Mutation evidence: disabling assistant-default column creation fails the isolated v48->49 test; removing corruption flag fails both malformed DB->API cases while SQL NULL passes. Extra v59 suite produced 55 passes and 3 failures (Notes task v59 SQLite source catalog drifted); all 3 reproduced with the original HEAD DB normalizer restored in memory via /tmp/persona-stage1-v59-baseline-check.py, confirming they are outside this change. No production migration logic changed. Final focused159-case suite still running. Test Bandit scan excluding expected B101 test assertions also zero findings/errors. Current dev SHAs rechecked unchanged from implementation start.

Published draft PR #2957 stacked on #2952. Implementation commit d90d49ba55. Final parent verification:159 passed,6 warnings in86.78s; no skips. Whole-stage independent review found no actionable issues and reran41 DB tests. Human-written Change summary and planning PR merge remain required before implementation merge. Stages2-5 remain unstarted.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Stage 1 resolver hardening: privacy-safe effective defaults and validation logging, disabled-feature guards with clear still allowed, computed malformed-storage diagnostics, meaningful historical migration coverage. Final focused suite159 passed; mutation tests prove column migration and corruption projection coverage; both independent reviews found no actionable issues. Bandit zero new findings/errors, scoped formatting clean, only four pre-existing endpoint Ruff warnings. Expanded v59 suite has three independently reproduced existing source-catalog fixture failures, documented without suppression. No live PostgreSQL/browser/Chatbook execution or full-repo suite. Stage2 and later remain pending. Draft implementation PR is stacked on planning PR2952; human-written Change summary remains a merge gate.
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
