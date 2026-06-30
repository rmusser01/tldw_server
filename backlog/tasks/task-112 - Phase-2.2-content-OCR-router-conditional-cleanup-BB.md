---
id: TASK-112
title: Phase 2.2 content OCR router conditional cleanup BB
status: Done
assignee: []
created_date: '2026-05-07 06:21'
updated_date: '2026-05-07 14:23'
labels:
  - phase-2.2
  - router-groups
  - content
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1361'
  - 'https://github.com/rmusser01/tldw_server/pull/1362'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move the remaining content-group OCR router factory onto the shared lazy optional-router registration path so it uses the same missing-target skip semantics and diagnostics as other optional content router specs while preserving the existing OCR route metadata.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Content OCR route metadata is represented through the shared optional router spec path while preserving prefix /api/v1, ocr tag, and route_key ocr.
- [x] #2 Focused contract coverage proves OCR module import and router attribute lookup stay deferred until router resolution.
- [x] #3 Focused contract coverage verifies missing OCR target skips while runtime import defects propagate.
- [x] #4 Existing router group, main router, and OpenAPI contracts still pass for the touched scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red/green evidence: focused selector tldw_Server_API/tests/Services/test_router_groups_contract.py -k 'content_router_specs and ocr' failed before production changes with 5 expected failures, then passed after moving OCR to ImportedRouterSpec with 5 passed, 141 deselected.

Validation: router group contracts passed with 146 passed; main router contracts passed with 6 passed; OpenAPI contracts passed with 69 passed; Bandit on tldw_Server_API/app/api/v1/router_groups/content.py reported 0 results and 0 errors; git diff --check was clean.

Docs: no user-facing documentation update was needed for this internal router registration cleanup. Known skips/blockers: none.

Review follow-up: Gemini requested explicit coverage for OptionalRouterMissingAttribute skip behavior when the OCR module imports successfully but lacks the router attribute. This is a test-only gap; production behavior already routes missing attrs through ImportedRouterSpec.

Review fix validation: added test_iter_content_router_specs_skips_ocr_missing_attribute_failures for the Gemini review thread. Focused OCR selector passed with 6 passed, 141 deselected; full router group contracts passed with 147 passed; git diff --check remained clean. No production files changed in this follow-up.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Moved the content OCR router from a hand-written lazy factory to the shared ImportedRouterSpec registration path. This preserves the existing /api/v1 prefix, ocr tag, and route key while using the shared missing-module/missing-attribute skip semantics and letting runtime import defects propagate.
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
