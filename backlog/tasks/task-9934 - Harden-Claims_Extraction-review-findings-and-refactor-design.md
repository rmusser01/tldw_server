---
id: TASK-9934
title: Harden Claims_Extraction review findings and refactor design
status: Done
assignee: []
created_date: 2026-06-23 21:39
updated_date: 2026-06-24 01:16
labels:
- claims
- review-hardening
- refactor
dependencies: []
references:
- tldw_Server_API/app/core/Claims_Extraction
priority: high
modified_files:
- Docs/superpowers/plans/2026-06-23-claims-extraction-hardening-plan.md
- Docs/superpowers/specs/2026-06-23-claims-extraction-hardening-refactor-design.md
- tldw_Server_API/app/core/Claims_Extraction/runtime_config.py
- tldw_Server_API/app/core/Claims_Extraction/claims_service.py
- tldw_Server_API/app/core/Claims_Extraction/claims_engine.py
- tldw_Server_API/app/core/Claims_Extraction/ingestion_claims.py
- tldw_Server_API/app/core/Claims_Extraction/claims_rebuild_service.py
- tldw_Server_API/app/core/Claims_Extraction/claims_notifications.py
- tldw_Server_API/app/core/Claims_Extraction/fva_pipeline.py
- tldw_Server_API/tests/Claims/test_claims_cancellation_and_timeout.py
- tldw_Server_API/tests/Claims/test_claims_rebuild_service_failure.py
- tldw_Server_API/tests/Claims/test_claims_runtime_config.py
- tldw_Server_API/tests/Claims/test_claims_review_notifications.py
- tldw_Server_API/tests/Claims/test_claims_dashboard_analytics.py
- tldw_Server_API/tests/Claims_Extraction/test_fva_pipeline.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and address validated Claims_Extraction review findings, then capture a focused refactor design for the module.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Validated review findings are covered by failing-first regression tests before production changes.
- [x] #2 Rebuild storage failure cannot soft-delete existing claims and report success.
- [x] #3 Claims cancellation propagates instead of being swallowed by noncritical exception handling.
- [x] #4 LLM extraction timeout returns promptly without waiting on stuck worker shutdown.
- [x] #5 Runtime limits, HTML escaping, analytics scoping, FVA metrics, and notification dispatch reliability are hardened.
- [x] #6 A focused Claims_Extraction refactor design spec is written for follow-up modularization.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Spec follow-up review tightened implementation constraints for rebuild strictness, timeout worker bounds, analytics owner-scope SQL, and notification dispatcher saturation behavior.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Verification Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented validated Claims_Extraction hardening findings with failing-first tests: runtime max clamps, review email HTML escaping, FVA adjudication metrics, atomic rebuild replacement rollback, cancellation propagation, prompt LLM timeout executor shutdown, owner-scoped analytics aggregates, and bounded notification dispatch. Verification: targeted pytest suite `python -m pytest -q tldw_Server_API/tests/Claims/test_claims_rebuild_service_failure.py tldw_Server_API/tests/Claims/test_claims_runtime_config.py tldw_Server_API/tests/Claims/test_claims_review_notifications.py tldw_Server_API/tests/Claims/test_claims_dashboard_analytics.py tldw_Server_API/tests/Claims/test_claims_cancellation_and_timeout.py tldw_Server_API/tests/Claims_Extraction/test_fva_pipeline.py` passed with 43 passed, 152 warnings. Bandit command on touched Claims_Extraction files wrote `/tmp/bandit_claims_extraction_9934.json` and exited 0.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed Claims_Extraction hardening for all validated review findings. Added failing-first regression coverage, fixed atomic rebuild replacement, cancellation propagation, prompt LLM timeout shutdown, runtime bounds, HTML escaping, analytics owner scoping, FVA adjudication metrics, and bounded notification dispatch. Verification passed: targeted Claims pytest suite reported 43 passed, 152 warnings; Bandit on touched Claims_Extraction files exited 0 with JSON output at /tmp/bandit_claims_extraction_9934.json.
<!-- SECTION:FINAL_SUMMARY:END -->
