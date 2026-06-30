---
id: TASK-2311
title: Backfill Security outbound egress policy ADR
status: Done
assignee: []
created_date: '2026-06-07 17:46'
updated_date: '2026-06-07 17:49'
labels:
  - docs
  - process
  - adr
  - security
dependencies:
  - TASK-2247
references:
  - Docs/ADR/inventory/2026-06-04-security-confirmation-audit.md
  - tldw_Server_API/app/core/Security/README.md
  - tldw_Server_API/app/core/Security/egress.py
  - tldw_Server_API/app/core/Security/url_validation.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a bounded accepted ADR for the outbound egress/SSRF portion of INV-029. Scope the ADR to central egress/url-validation helper ownership and current policy defaults for scheme, host, port, allow/deny, environment profile, tenant webhook, DNS, and private/reserved-address checks. Do not claim universal historical coverage for every existing network call; describe the rule for outbound integrations going forward.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add a new immutable ADR under Docs/ADR/ with the next available ADR number and one bounded outbound egress/SSRF decision.
- [x] #2 Link the ADR from Docs/ADR/README.md and update Docs/ADR/inventory/2026-06-03-decision-inventory.md so INV-029 records this split ADR while preserving remaining secrets/serialization caveats.
- [x] #3 Update Docs/ADR/inventory/2026-06-04-security-confirmation-audit.md to record the outbound egress backfill result without collapsing INV-029 into one broad Security ADR.
- [x] #4 Keep caveats explicit: policy only protects callers that route outbound URLs through the central helpers, request-edge middleware remains ADR-019, and secrets/serialization still need a separate adoption audit before any ADR.
- [x] #5 Record verification and Bandit applicability in this task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-026 for the bounded outbound egress/SSRF portion of INV-029. Updated the ADR index, Security module README, INV-029 inventory row, provider/integration owner-review handoff, and the Security confirmation audit to point to ADR-026 while preserving ADR-019 request-edge ownership and the separate secrets/serialization caveat.

Verification recorded on 2026-06-07:
- git diff --cached --check: pass.
- ADR/link reference scan: ADR-026 is present in the ADR index, new ADR, Security README, inventory row, confirmation audit, and TASK-2311 record.
- Portability artifact scan: no developer-machine absolute paths or temporary Bandit report artifact names found in touched docs/task files.

Bandit: not run because this branch only touches Markdown ADR, inventory, audit, Security README, and Backlog task records; no Python files were changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added ADR-026 as the accepted bounded Security outbound egress/SSRF policy decision. Linked it from the ADR index and Security README, updated INV-029 and the Security confirmation audit to record the split backfill, and documented docs-only verification plus Bandit non-applicability.
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
