---
id: TASK-228
title: Document ACP retention and transcript redaction release policy
status: Done
assignee: []
created_date: '2026-05-10 07:17'
updated_date: '2026-05-10 15:40'
labels:
  - acp
  - release-signoff
  - docs
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1502'
  - 'https://github.com/rmusser01/tldw_server/issues/1512'
  - 'https://github.com/rmusser01/tldw_server/issues/1513'
documentation:
  - Docs/Development/ACP_Production_Readiness.md
  - Docs/Development/Agent_Client_Protocol.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define the #1502 release policy for ACP session/artifact retention and transcript redaction. Ground the policy in the current implementation, classify supported behavior versus caveats, and ensure release notes/operator docs avoid unsupported production retention claims.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Policy covers ACP session messages, artifacts, diagnostics, audit events, workspace paths, environment metadata, prompts, responses, and tool arguments.
- [x] #2 Current implementation behavior is classified as compliant, partial, or blocked with concrete code/doc references.
- [x] #3 Release notes and ACP operator docs state only supported retention/redaction behavior and call out accepted caveats.
- [x] #4 Any implementation mismatch is split into a focused GitHub issue or documented as an accepted release caveat before closing #1502.
- [x] #5 Verification includes formatting checks and a targeted search/read review for retention and redaction wording.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Audit current ACP session/audit/diagnostic/artifact persistence and redaction behavior. 2. Document release-safe retention/redaction policy in ACP operator docs and readiness matrix. 3. Split implementation gaps into focused GitHub follow-up issues. 4. Verify docs formatting and targeted retention/redaction wording.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Audit findings: ACP session TTL closes active sessions but does not hard-delete session rows/messages/artifact references; audit metadata and diagnostics have sanitizers; detail/events/artifacts remain full-fidelity authenticated drill-through surfaces; audit purge helper exists but automatic configured enforcement is not release-certified. Follow-up issues created: #1512 retention cleanup, #1513 redacted transcript/artifact views.

Verification: git diff --check passed. Targeted rg confirmed policy wording and #1512/#1513 links in CHANGELOG.md, Docs/Development/Agent_Client_Protocol.md, and Docs/Development/ACP_Production_Readiness.md. Bandit skipped because this slice changes docs/backlog only.

Review follow-up for PR #1515: added an explicit compliant/partial/blocked classification table for ACP session detail/events, artifacts, diagnostics, audit metadata, session TTL cleanup, audit retention, workspace environment metadata, and redacted transcript/artifact views. Replaced ambiguous `/events` shorthand with the full session-scoped `/api/v1/acp/sessions/{session_id}/events` route.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Documented ACP #1502 retention/redaction release policy and linked implementation follow-ups #1512 and #1513. The docs now state that session detail/events/artifacts are authenticated full-fidelity operator drill-through surfaces, while audit metadata and diagnostics are sanitized; automatic hard-delete retention and redacted transcript/artifact views remain follow-up implementation work. Review follow-up added an explicit compliant/partial/blocked status map and corrected the session events endpoint path. Verification: git diff --check and targeted rg review passed; Bandit skipped for docs-only changes.
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
