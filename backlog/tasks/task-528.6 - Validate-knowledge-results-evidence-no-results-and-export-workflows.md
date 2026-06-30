---
id: TASK-528.6
title: Validate /knowledge results evidence no-results and export workflows
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-07 03:30'
labels:
  - webui
  - extension
  - knowledge
  - ux
  - testing
dependencies: []
documentation:
  - Docs/superpowers/plans/2026-06-07-knowledge-results-evidence-export-plan.md
parent_task_id: TASK-528
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Validate and harden /knowledge answer results, citations, evidence rail, Sources versus Details views, no-results recovery, follow-up search, and export behavior. Keep the workflow focused on grounded Knowledge QA.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Result answers expose visible citations that map to evidence rail sources.
- [x] #2 Sources and Details evidence views explain source choice, retrieval, reranking, web fallback, and verification when data is available.
- [x] #3 No-results recovery offers broaden scope, adjust query/settings, enable web fallback when available, or inspect nearest matches only when real candidates exist.
- [x] #4 Export supports expected formats and includes answer, query, sources, citations, and optional settings snapshot.
- [x] #5 Automated tests cover cited results, no-results, failed search, and export behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-06-07-knowledge-results-evidence-export-plan.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented TASK-528.6 results/evidence/export hardening. Export Markdown/PDF/Notes generation now includes citation mappings, safe settings snapshot context, preset, and search details when requested. Search details now explains absent telemetry instead of rendering N/A rows. No-results recovery gained focused component coverage. WebUI empty-recovery E2E selectors were hardened after Playwright strict-mode ambiguity.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
TASK-528.6 complete. Verification: focused ExportDialog/SearchDetails Vitest 30 tests passed; NoResultsRecovery Vitest 2 tests passed; broader Knowledge QA Vitest suite 97 tests passed; WebUI Knowledge QA route Playwright suite 4 tests passed after selector hardening; git diff --check passed; out-of-scope terminology guard found no matches in touched Knowledge QA source/test files; Bandit not applicable because no Python backend files changed. Known skip: extension Playwright run was attempted but WXT production build hung before tests executed and had to be terminated. Package-wide tsc remains blocked by existing baseline TypeScript errors outside this slice.
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
