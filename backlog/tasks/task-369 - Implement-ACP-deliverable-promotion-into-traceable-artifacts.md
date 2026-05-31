---
id: TASK-369
title: Implement ACP deliverable promotion into traceable artifacts
status: Done
assignee: []
created_date: '2026-05-15 03:51'
updated_date: '2026-05-15 22:11'
labels:
  - acp
  - artifacts
  - backend
dependencies:
  - TASK-357
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1706'
  - 'https://github.com/rmusser01/tldw_server/issues/1703'
  - 'https://github.com/rmusser01/tldw_server/issues/1707'
  - 'https://github.com/rmusser01/tldw_server/pull/1714'
documentation:
  - Docs/Product/Traceable_Work_Product_Artifact_Contract.md
  - Docs/Product/ACP_Agent_Orchestration_PRD.md
  - Docs/Development/Agent_Client_Protocol.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1706: promote one golden-path ACP run deliverable, such as a source-grounded workspace brief, into the traceable work-product artifact backend contract while preserving ACP execution evidence. This follows the merged storage/API foundation from #1703 and the Workspace UI detail surface from #1707/#1714. Keep raw low-level ACP session artifacts as execution evidence unless explicitly promoted into structured work products, and enforce redaction/review-state semantics from the traceable artifact contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ACP task completion can create or update a traceable work-product artifact through the backend contract.
- [x] #2 Promotion preserves ACP producer/session/run/review references.
- [x] #3 Rejected or needs_revision outputs do not masquerade as accepted artifacts.
- [x] #4 Tests cover accepted, retry, rejected, redacted, and malformed promotion paths.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Map the merged traceable artifact storage/API contract and current ACP execution surfaces. - Complete
2. Add focused failing backend tests for accepted, retry/needs_revision, rejected, redacted, and malformed ACP promotion payloads. - Complete
3. Implement the minimal backend promotion service/API seam that creates or updates traceable workspace artifacts while preserving ACP producer/session/run/review references. - Complete
4. Wire the golden-path ACP completion caller only where the existing architecture already exposes deliverable metadata. - Complete
5. Verify with focused pytest/Bandit and update #1706/TASK-369 evidence. - Complete
6. Address PR #1718 review feedback: avoid converting post-commit artifact promotion failures into failed dispatch responses, preserve promotion failure metadata in the response/audit path, replace repeated preview truncation literals with a named module constant, offload sync promotion I/O from the async endpoint, and harden malformed artifact metadata handling. - Complete
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ACP artifact promotion backend slice for issue #1706. Added a focused promotion service at tldw_Server_API/app/core/Agent_Orchestration/artifact_promotion.py and dispatch wiring in tldw_Server_API/app/api/v1/endpoints/agent_orchestration.py. Added regression coverage in tldw_Server_API/tests/Agent_Orchestration/test_artifact_promotion.py plus a dispatch-level golden path in test_orchestration_api.py. Verification: pytest tldw_Server_API/tests/Agent_Orchestration -q => 176 passed, 5 warnings. Ruff check on touched files => all checks passed. Bandit production touched files => exit 0; touched files with pytest B101/B105 excluded => exit 0.

PR #1718 review follow-up reopened this task for Gemini feedback on post-commit promotion error handling and preview truncation maintainability.

Review follow-up verification: `python -m pytest tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py -k promotion_failure_without_rolling_back_task -q` failed before the endpoint fix and passed after it; `python -m pytest tldw_Server_API/tests/Agent_Orchestration/test_artifact_promotion.py tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py -q` passed with 43 tests and 5 warnings; `python -m ruff check tldw_Server_API/app/api/v1/endpoints/agent_orchestration.py tldw_Server_API/app/core/Agent_Orchestration/artifact_promotion.py tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py` passed; `python -m bandit -r tldw_Server_API/app/api/v1/endpoints/agent_orchestration.py tldw_Server_API/app/core/Agent_Orchestration/artifact_promotion.py -f json -o /tmp/bandit_acp_artifact_promotion_1718.json` passed with 0 results/errors; `git diff --check` passed.

Second review follow-up added `_run_sync` offloading for promotion DB I/O, response-contract assertions, promotable artifact type validation with `promote_as` fallback, optional metadata/schema/export validation, and per-artifact promotion failure isolation. The CodeRabbit suggestion to skip redacted artifacts was evaluated and not implemented because TASK-369's accepted behavior is to preserve the redaction contract for promoted artifacts; the UI layer suppresses sensitive provenance/lineage when the redaction posture requires it. Verification after this patch: `python -m pytest tldw_Server_API/tests/Agent_Orchestration/test_artifact_promotion.py -k 'promote_as or promote_flag_without_allowed_artifact_type or malformed_optional_metadata' -q` passed with 7 tests and 5 warnings; `python -m pytest tldw_Server_API/tests/Agent_Orchestration/test_artifact_promotion.py tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py -q` passed with 50 tests and 5 warnings; Ruff passed on the touched endpoint/service/test files; Bandit passed with 0 results/errors; `git diff --check` passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented issue #1706 in PR #1718: https://github.com/rmusser01/tldw_server/pull/1718

Added ACP deliverable promotion into traceable workspace artifacts through a focused backend service and ACP dispatch wiring. The implementation promotes only structured work-product artifacts with source lineage, preserves ACP producer/session/run/review metadata, stores redaction/version/source-lineage contract fields, updates existing artifacts by version, and leaves retry/rejected/malformed payloads out of accepted artifact state.

Verification recorded: Agent Orchestration pytest suite passed (176 passed, 5 warnings); Ruff passed on touched files; Bandit passed on production touched files and on the touched set with pytest-only B101/B105 excluded. PR review follow-up also keeps dispatch responses successful when post-commit artifact promotion fails, includes a structured promotion failure result for audit/response metadata, replaces repeated preview truncation literals with a named module constant, offloads promotion DB work from the async endpoint, and validates malformed optional artifact metadata without aborting the whole promotion pass.
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
