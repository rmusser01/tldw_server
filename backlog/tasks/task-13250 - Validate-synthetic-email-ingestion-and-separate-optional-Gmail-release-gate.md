---
id: TASK-13250
title: Validate synthetic email ingestion and separate optional Gmail release gate
status: Done
assignee: []
created_date: '2026-09-13 18:33'
updated_date: '2026-09-13 18:45'
labels: []
dependencies: []
documentation:
  - Docs/Operations/Email_Core_Validation_2026-09-13.md
  - Docs/Product/Email_Ingestion_Search_PRD.md
  - Docs/Operations/Email_Release_Checklist_and_Rollback.md
  - Docs/Operations/Email_Sync_Operations_Runbook.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow the September 13 email handoff: audit file parsing and persistence, prove a synthetic upload/search/detail path makes no model or outbound requests, keep Gmail tests mocked, and separate evidence-based core gates from deferred optional live Gmail validation. Never access personal Gmail or personal mail.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Trace supported upload formats, attachment behavior, persistence, indexing, and model call boundaries
- [x] #2 Validate synthetic parsing, attachments, dedupe, search and detail with model and network interception
- [x] #3 Run focused mocked Gmail regressions and document untested live behavior
- [x] #4 Update PRD, runbook and release checklist with separate core and optional Gmail gates and single-owner approval
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Audit current code and test seams. 2. Add synthetic offline integration coverage against temporary SQLite and execute focused tests. 3. Update scope and evidence docs, run lint and Bandit, self-review and commit only task files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Audit and synthetic validation complete. Current focused core suite: 75 passed, 8 warnings; mocked Gmail: 13 passed, 9 deselected, 4 warnings. A task-local socket/DNS guard covered collection/execution: zero outbound attempts in both runs. Six committed harness cases intercept summarization, claims, LLM chunk assistance, embedding jobs, background tasks, shared HTTP and sockets. Initial strict same-body probes failed EML/ZIP/MBOX; TASK-13251 records actual core dedupe blocker. Current collision characterization is not acceptance. No production behavior changes or live Gmail access. Metrics fixture checker passed but does not prove staging SLO. Ruff/format and Bandit passed (B101 omitted for intentional pytest assertions).

Final independent read-only review found no substantive issues; corrected abbreviated parser/DB paths in the audit. git diff --check passed after whitespace cleanup. Temporary implementation plan stages completed and plan removed per repository instructions; retained plan summary in this task. Validation completed, not feature/release certification.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Established synthetic offline EML/ZIP/MBOX ingestion/search/detail evidence with six new integration cases. Focused core tests: 75 passed; mocked Gmail: 13 passed; zero outbound attempts in guarded runs. Ruff/format and Bandit passed. PRD/runbook/checklist now separate core release criteria from deferred optional Gmail, with single-owner gates. Found and tracked TASK-13251: same-body distinct email identities collapse; current HTTP pagination is offset-only. No personal Gmail access or production code/config changes; no rollout readiness claim.
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
