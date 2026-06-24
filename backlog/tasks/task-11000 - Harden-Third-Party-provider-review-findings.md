---
id: TASK-11000
title: Harden Third_Party provider review findings
status: Done
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
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the accepted Third_Party review findings: BioRxiv filtered pagination now advances based on upstream batch size, PMC OA PDF downloads use bounded temp-file streaming plus PMCID/PDF validation, selected JSON adapters surface HTTP status failures, arXiv/CitEc metadata URLs use HTTPS, and viXra search applies requested pagination before enrichment. Added focused regression coverage and ran security verification.
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
