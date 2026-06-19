---
id: TASK-2390
title: ACP release caveat artifact retention and transcript redaction policy
status: Done
labels:
- ACP
- release-caveat
- policy
- retention
- redaction
references:
- https://github.com/rmusser01/tldw_server/issues/2401
- https://github.com/rmusser01/tldw_server/issues/2398
- https://github.com/rmusser01/tldw_server/issues/2408
- https://github.com/rmusser01/tldw_server/pull/2409
- https://github.com/rmusser01/tldw_server/issues/2398#issuecomment-4752665089
- https://github.com/rmusser01/tldw_server/issues/2401#issuecomment-4752665267
modified_files:
- Docs/Development/Agent_Client_Protocol.md
- Docs/Development/ACP_Production_Readiness.md
- Docs/Development/ACP_Certification_Checklist.md
- Docs/Development/ACP_Compatibility_Matrix.md
- Docs/User_Guides/Integrations_Experiments/Getting_Started_with_ACP.md
- IMPLEMENTATION_PLAN_acp_artifact_retention_redaction_policy_2401.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track GitHub issue #2401: audit current ACP artifact/event/audit/diagnostic/transcript persistence, make the release retention and redaction policy explicit, and split any implementation gaps into follow-up issues.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Policy states retention boundaries for artifacts, events, diagnostics, audit records, and transcript previews.
- [x] #2 Policy states redaction guarantees and known non-guarantees.
- [x] #3 Any implementation gaps are split into follow-up issues instead of hidden in docs.
- [x] #4 Parent #2398 is updated with the evidence link and final status.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
[IMPLEMENTATION_PLAN_acp_artifact_retention_redaction_policy_2401.md](../../IMPLEMENTATION_PLAN_acp_artifact_retention_redaction_policy_2401.md)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed ACP issue #2401 in PR #2409. Audited ACP session/message persistence, event drill-through, diagnostics, audit DB retention, redacted support views, Agent Tasks run previews, and promoted workspace artifact boundaries. Documented the release retention/redaction policy in ACP_Production_Readiness.md and aligned Agent_Client_Protocol.md, the ACP setup guide, compatibility matrix, and certification checklist. Split the only implementation gap found into #2408 for support-safe Agent Tasks run summaries. Verification: git diff --check; targeted rg consistency checks for stale retention/redaction ownership wording and ACP retention/redaction terms. Bandit not applicable because only docs, plan, and Backlog files changed. Parent #2398 updated at https://github.com/rmusser01/tldw_server/issues/2398#issuecomment-4752665089; #2401 updated at https://github.com/rmusser01/tldw_server/issues/2401#issuecomment-4752665267.
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
