---
id: TASK-11000
title: Harden Third_Party provider review findings
status: Done
updated_date: 2026-06-24 20:18
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix current Third_Party module review findings: BioRxiv filtered pagination, PMC OA bounded PDF download, provider HTTP status handling, plain HTTP metadata URLs where supported, and viXra pagination semantics.

Reference: tldw_Server_API/app/core/Third_Party
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 BioRxiv filtered searches continue across full upstream batches and have regression coverage.
- [x] #2 PMC OA PDF downloads use bounded streaming validation instead of unbounded response.content.
- [x] #3 Provider adapters distinguish non-404 HTTP failures from successful empty records where practical.
- [x] #4 Plain HTTP scholarly metadata URLs are upgraded to HTTPS where provider support exists.
- [x] #5 viXra search pagination behavior is corrected or honestly represented with tests.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Verification:
- python -m pytest tldw_Server_API/tests/Research/test_biorxiv_sanitizers.py tldw_Server_API/tests/Research/test_pmc_oa_sanitizers.py tldw_Server_API/tests/Research/test_vixra_sanitizers.py tldw_Server_API/tests/Research/test_arxiv_sanitizers.py tldw_Server_API/tests/Research/test_repec_sanitizers.py tldw_Server_API/tests/http_client/test_third_party_adapters_http.py -q (67 passed)
- python -m pytest tldw_Server_API/tests/Research/test_semantic_scholar_sanitizers.py tldw_Server_API/tests/Research/test_ieee_sanitizers.py tldw_Server_API/tests/Research/test_springer_sanitizers.py tldw_Server_API/tests/Research/test_scopus_sanitizers.py tldw_Server_API/tests/Research/test_paper_search_endpoints.py tldw_Server_API/tests/Research/test_error_mapping_endpoints.py -q (52 passed)
- python -m bandit -r touched Third_Party files -f json -o /tmp/bandit_third_party_review_11000.json (0 findings)
- git diff --check on touched scope (clean)
- python -m compileall -q tldw_Server_API/app/core/Third_Party (clean)
Finalization update:
- Bandit report path aligned to /tmp/bandit_third_party_review_11000.json (0 findings).
- No known skips or blockers.
Review follow-up in progress:
- Rebase PR branch onto latest dev.
- Verify Qodo inline comments against current code.
- Address accepted review items with focused regression coverage and verification.
Review follow-up complete:
- Rebasing: PR branch rebased onto latest origin/dev on 2026-06-24.
- Centralized ThirdPartyHTTPStatusError in app/core/exceptions.py and updated adapters to import it from the shared module.
- Closed successful fetch responses in viXra lookup/search paths and Scopus DOI lookup.
- Added type annotations to the newly added tests/helpers and regression coverage for successful Scopus/viXra response closure.

Follow-up verification:
- python -m compileall -q tldw_Server_API/app/core/Third_Party tldw_Server_API/app/core/exceptions.py touched tests (clean)
- git diff --check (clean)
- python -m pytest focused Third_Party/Research suites -q (120 passed)
- python -m bandit -r tldw_Server_API/app/core/exceptions.py tldw_Server_API/app/core/Third_Party -f json -o /tmp/bandit_third_party_review_11000_followup.json (0 findings)
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the accepted Third_Party review findings: BioRxiv filtered pagination now advances based on upstream batch size, PMC OA PDF downloads use bounded temp-file streaming plus PMCID/PDF validation, selected JSON adapters surface HTTP status failures, arXiv/CitEc metadata URLs use HTTPS, and viXra search applies requested pagination before enrichment. Added focused regression coverage and ran security verification.
Review follow-up rebased the PR onto latest dev, moved the Third_Party HTTP status exception into core exceptions, closed successful Scopus/viXra responses, annotated the added tests/helpers, and added closure regression coverage. Focused tests, compile check, whitespace check, and Bandit all passed.
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
