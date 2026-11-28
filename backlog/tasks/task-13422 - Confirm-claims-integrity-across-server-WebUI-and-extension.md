---
id: TASK-13422
title: Confirm claims integrity across server WebUI and extension
status: Done
assignee: []
created_date: '2026-10-02 23:09'
updated_date: '2026-10-02 23:37'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/3090'
documentation:
  - Docs/Reviews/Claims_Integrity_Confirmation_2026_10_02.md
modified_files:
  - Docs/Reviews/Claims_Integrity_Confirmation_2026_10_02.md
  - Docs/Reviews/NotebookLM_Thread_Capability_Review_2026_10_02.md
  - Docs/Reviews/artifacts/claims-integrity-13422/claims-verdict-results.json
ordinal: 14969
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate the existing Claims subsystem and confirm suspected evidence-integrity gaps before feature design or implementation. Preserve server, WebUI, and extension findings plus reproducible evidence in review PR #3090. Scope is investigation and documentation; no product fixes or architectural approval are implied.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Server claims, scope, repair, and verdict findings are classified with exact source references and focused reproduction evidence where feasible.
- [x] #2 WebUI and extension request construction, shared-code reuse, defaults, evidence display, and persistence are traced with precise qualifications.
- [x] #3 Confirmed and unconfirmed findings, existing safeguards, test results and environment limitations are saved to PR #3090 without changing product code.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 1 Complete: trace server/client boundaries. Stage 2 Complete: reproduce verdict rules and run focused available tests. Stage 3 Complete: source-reviewed documentation verified, committed, pushed, and linked in draft PR 3090 against dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed independent server, WebUI/shared UI, and extension traces. Saved report: Docs/Reviews/Claims_Integrity_Confirmation_2026_10_02.md; JSON: Docs/Reviews/artifacts/claims-integrity-13422/claims-verdict-results.json; original capability review links to the confirmation. Executed product snapshot 8140e493f2d0a79e2039084930151eba6565df82; latest checked dev 9958110df2a9011e19f48b0eae821353e19d4af8 has no relevant source changes. Verification: 35 existing server tests passed, zero failures/errors/skips, 202 warnings, JUnit 27.278s; 8 imported-engine observations confirmed; 5 existing extension Bun tests and 4 temporary helper probes passed. Isolated server control-flow checks are qualified separately. Documentation verified 53 immutable references, 3 relative links, all 8 JSON observations, source SHA256, JUnit counts, current-dev equivalence, and staged whitespace/scope. Independent source review corrected wrapper/config references, refusal persistence wording, and streaming metadata name. Limits: full app import blocked by local FastAPI mismatch; WebUI Vitest did not collect due local dependency incompatibilities. No installs, product edits, live provider/browser certification, or dependency fixes. ADR required: no; Bandit inapplicable to documentation/data-only scope. Findings and evidence pushed as 17b67df7ca9493c4c2599e10ab9cf9cee04cce42; PR 3090 verified OPEN, draft, base dev, matching head. Human-written Change summary still required before merge; no design/implementation approval implied; partial supported answers vs whole-answer refusal unresolved. Preexisting duplicate Backlog IDs were not repaired.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Confirmed existing claims safeguards and concrete integrity gaps across server and both clients; saved a source-reviewed addendum plus deterministic evidence in draft PR 3090 against dev. Focused tests and probes are recorded with explicit static/isolated/live limits. No product behavior or dependencies changed. Investigation complete; evidence-integrity design can now proceed from the confirmed implementation.
<!-- SECTION:FINAL_SUMMARY:END -->
