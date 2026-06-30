---
id: TASK-2313
title: Backfill Security crypto envelope ADR
status: Done
assignee: []
created_date: 2026-06-07 21:25
labels:
- docs
- process
- adr
- security
dependencies:
- TASK-2312
references:
- Docs/ADR/inventory/2026-06-07-security-secrets-serialization-adoption-audit.md
- tldw_Server_API/app/core/Security/crypto.py
- tldw_Server_API/app/core/Jobs/manager.py
- tldw_Server_API/app/core/External_Sources/connectors_service.py
- tldw_Server_API/app/core/AuthNZ/user_provider_secrets.py
- tldw_Server_API/app/core/AuthNZ/admin_webhook_secrets.py
modified_files:
- Docs/ADR/027-security-aes-gcm-json-envelope-helpers.md
- Docs/ADR/README.md
- Docs/ADR/inventory/2026-06-03-decision-inventory.md
- Docs/ADR/inventory/2026-06-04-security-confirmation-audit.md
- Docs/ADR/inventory/2026-06-07-security-secrets-serialization-adoption-audit.md
- tldw_Server_API/app/core/Security/README.md
- backlog/tasks/task-2313 - Backfill-Security-crypto-envelope-ADR.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a bounded accepted ADR for the Security AES-GCM JSON envelope primitive after TASK-2312 found helper-level evidence and known encrypted persistence consumers. Scope the decision to Security.crypto envelope format/key behavior and known Jobs/AuthNZ/External Sources/Workflows consumer patterns. Do not claim universal encryption for all sensitive JSON or universal SecretManager adoption.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add a new immutable ADR under Docs/ADR/ with the next available ADR number and one bounded Security crypto envelope decision.
- [x] #2 Link the ADR from Docs/ADR/README.md and update Docs/ADR/inventory/2026-06-03-decision-inventory.md so INV-029 records the crypto-envelope split while keeping SecretManager adoption and restricted pickle as separate caveats.
- [x] #3 Update Docs/ADR/inventory/2026-06-04-security-confirmation-audit.md and Docs/ADR/inventory/2026-06-07-security-secrets-serialization-adoption-audit.md to record the crypto-envelope backfill result without creating one broad Security ADR.
- [x] #4 Keep caveats explicit: encryption is configured/optional in some consumers, plaintext fallback exists where crypto is unavailable or not configured, SecretManager adoption is not covered, and restricted pickle remains separate.
- [x] #5 Record verification and Bandit applicability in this task.
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented ADR-027 for the bounded Security AES-GCM JSON envelope helper decision. Updated the ADR index, INV-029 inventory row, Security confirmation audit, secrets/serialization adoption audit, and Security README backlink.

Caveats preserved: caller-specific key/encryption boundaries, connector plaintext fallback where crypto is unavailable or not configured, no universal SecretManager adoption, and no restricted pickle decision.

Verification:
- `git diff --cached --check` exited 0.
- Staged ADR/reference scan for ADR-027, TASK-2313, and INV-029 exited 0.
- Staged portability scan for developer-machine absolute paths and temporary Bandit report artifact names exited 1 with no matches.
- Bandit not run: docs-only Markdown/Backlog changes; no Python files touched.

Known skips/blockers: no code tests or Bandit run were applicable for this docs-only slice; no blockers remain.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Backfilled the AES-GCM JSON envelope portion of INV-029 as ADR-027. Linked the ADR from the ADR index and Security README, updated the decision inventory and confirmation/adoption audits to record the split, and left SecretManager adoption plus restricted pickle compatibility as separate inventory-only or future slices. Verification is recorded with docs-only Bandit non-applicability.
<!-- SECTION:FINAL_SUMMARY:END -->
