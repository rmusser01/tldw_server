---
id: TASK-2314
title: Backfill Security restricted pickle compatibility ADR
status: Done
dependencies:
- TASK-2312
labels:
- docs
- process
- adr
- security
references:
- Docs/ADR/inventory/2026-06-07-security-secrets-serialization-adoption-audit.md
- tldw_Server_API/app/core/Security/safe_pickle.py
- tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py
- tldw_Server_API/app/core/Scheduler/services/payload_service.py
- tldw_Server_API/tests/WebScraping/test_content_deduplicator_storage.py
- tldw_Server_API/app/core/Scheduler/tests/test_payload_service_security.py
modified_files:
- Docs/ADR/028-security-restricted-legacy-pickle-compatibility.md
- Docs/ADR/README.md
- Docs/ADR/inventory/2026-06-03-decision-inventory.md
- Docs/ADR/inventory/2026-06-04-security-confirmation-audit.md
- Docs/ADR/inventory/2026-06-07-security-secrets-serialization-adoption-audit.md
- tldw_Server_API/app/core/Security/README.md
- backlog/tasks/task-2314 - Backfill-Security-restricted-pickle-compatibility-ADR.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a bounded accepted ADR for the Security restricted legacy pickle compatibility helper after TASK-2312 found helper-level evidence and default-disabled consumers. Scope the decision to Security.safe_pickle and the known Web Scraping and Scheduler legacy compatibility paths. Do not claim universal pickle deserialization coverage or replace cache-local restrictive unpicklers unless implementation changes are made.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add a new immutable ADR under Docs/ADR/ with the next available ADR number and one bounded restricted legacy pickle compatibility decision.
- [x] #2 Link the ADR from Docs/ADR/README.md and update Docs/ADR/inventory/2026-06-03-decision-inventory.md so INV-029 records the restricted-pickle split while keeping SecretManager adoption as a separate caveat.
- [x] #3 Update Docs/ADR/inventory/2026-06-04-security-confirmation-audit.md and Docs/ADR/inventory/2026-06-07-security-secrets-serialization-adoption-audit.md to record the restricted-pickle backfill result without creating one broad Security ADR.
- [x] #4 Keep caveats explicit: compatibility paths are default-disabled or explicitly gated, allowed pickle globals are narrow, Embeddings cache has its own local unpickler, and SecretManager adoption is not covered.
- [x] #5 Record verification and Bandit applicability in this task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-read the TASK-2312 audit plus Security safe_pickle and known consumer evidence.
2. Draft ADR-028 as a bounded accepted decision for default-disabled restricted legacy pickle compatibility.
3. Update ADR index, INV-029, Security confirmation audit, adoption audit, and Security README backlink.
4. Verify staged docs for whitespace, references, portability, and Bandit applicability.
5. Finalize the Backlog task, commit, push, and open a PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Reviewed Security safe_pickle and the known Web Scraping/Scheduler compatibility consumers before writing the ADR.
- Added ADR-028 as a bounded accepted decision for Security restricted legacy pickle compatibility. Scope is limited to explicitly gated/default-disabled legacy compatibility paths and does not cover SecretManager adoption or Embeddings cache-local unpicklers.
- Updated ADR README, INV-029, Security confirmation audit, secrets/serialization adoption audit, and Security README backlink.
- Verification: staged whitespace check passed; staged changed-file list reviewed; targeted staged reference scan found ADR-028/TASK-2314/INV-029 links; targeted staged portability scan found no developer-machine absolute paths or temporary Bandit report artifact names.
- Bandit: not run because this task changes only Markdown/Backlog documentation and no Python/code paths.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Backfilled the restricted legacy pickle compatibility slice as ADR-028, linked it from the ADR index and Security README, and updated the INV-029 inventory/audit records to keep SecretManager adoption as a separate inventory-only caveat. Verification was docs-focused; Bandit was not applicable because no code paths changed.
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
