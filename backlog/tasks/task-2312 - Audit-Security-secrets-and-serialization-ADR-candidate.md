---
id: TASK-2312
title: Audit Security secrets and serialization ADR candidate
status: Done
assignee: []
created_date: '2026-06-07 21:10'
updated_date: '2026-06-07 21:14'
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
  - tldw_Server_API/app/core/Security/secret_manager.py
  - tldw_Server_API/app/core/Security/crypto.py
  - tldw_Server_API/app/core/Security/safe_pickle.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a focused adoption audit for the remaining secrets/serialization portion of INV-029. Confirm whether the current repository behavior supports a bounded accepted ADR for SecretManager, AES-GCM JSON helpers, and restricted pickle compatibility, or document why the slice should remain inventory-only until more adoption work is done.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add a focused Security secrets/serialization confirmation audit under Docs/ADR/inventory/ with evidence for SecretManager, crypto, safe_pickle, tests, and known consumers.
- [x] #2 Update Docs/ADR/inventory/2026-06-03-decision-inventory.md so INV-029 records the audit result without creating a broad Security ADR.
- [x] #3 Update Docs/ADR/inventory/2026-06-04-security-confirmation-audit.md to point to the focused audit and preserve ADR-019/ADR-026 boundaries.
- [x] #4 Keep caveats explicit: do not claim universal SecretManager adoption unless evidence supports it, keep request-edge and egress covered by existing ADRs, and separate helper availability from caller adoption.
- [x] #5 Record verification and Bandit applicability in this task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a focused Security secrets/serialization adoption audit for INV-029. The audit records helper availability for SecretManager, Security crypto, and safe_pickle; bounded consumer adoption for AES-GCM JSON envelopes and restricted legacy pickle compatibility; and insufficient repository-wide SecretManager/serialization adoption for an accepted ADR. Updated the main decision inventory and Security confirmation audit to keep the slice inventory-only for now.

Verification recorded on 2026-06-07:
- git diff --cached --check: pass.
- Audit/link reference scan: TASK-2312, INV-029, the focused audit path, ADR-019, ADR-026, and inventory-only disposition references are present.
- Portability artifact scan: no developer-machine absolute paths or temporary Bandit report artifact names found in touched docs/task files.

Bandit: not run because this branch only touches Markdown ADR inventory/audit and Backlog task records; no Python files were changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the focused Security secrets/serialization adoption audit and updated INV-029 plus the Security confirmation audit. The result keeps this remaining Security slice inventory-only: helper-level evidence exists, but current caller adoption does not support an accepted ADR for centralized secret lookup or universal serialization policy.
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
