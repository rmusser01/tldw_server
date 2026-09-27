---
id: TASK-13251
title: Fix identical-body email uploads merging distinct message identities
status: Done
assignee: []
created_date: '2026-09-13 18:38'
updated_date: '2026-09-13 19:34'
labels: []
dependencies: []
references:
  - >-
    tldw_Server_API/tests/MediaIngestion_NEW/integration/test_email_offline_ingestion.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-13250 synthetic acceptance probes found distinct RFC Message-ID messages with identical bodies collapse to one Media row and one normalized email row for EML, ZIP and MBOX. media_repository.add_media_with_keywords matches content_hash before normalized email upsert; email_graph_persistence_ops falls back to media_id and replaces message metadata. The later message remains searchable while the first identity is lost. This blocks core FR-INGEST-001 / EMAIL-M0-002 correctness independently of optional Gmail.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Distinct email messages with identical bodies remain separate in legacy and normalized stores for EML ZIP and MBOX uploads
- [x] #2 Repeat imports remain idempotent within the documented source scope
- [x] #3 Replace TASK-13250 collision characterization with assertions for both original identities and consistent detail
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Follow-up: strict same-body EML/ZIP/MBOX tests passed after identity-scoped legacy dedupe and email safe-metadata preservation. Added legacy-row reuse, source/owner isolation, document-vs-email hash exclusion, ISO provider-date normalization and overflow regressions. Independent review also confirmed reimport overwrite=false graph inconsistency; fix in progress. TASK-13253 cursor and TASK-13254 evidence labeling implemented. Local 1000-message benchmark recorded; PostgreSQL/native PST checks skipped for unavailable prerequisites. No Gmail/model/network access.

Reimport follow-up complete: all three upload graph calls and Gmail use new DB-layer read_persisted_email_content helper (canonical Media body/latest version metadata, tenant-scoped native raw-metadata fallback for legacy stripped versions). Gmail retains mutable provider labels and empty-body semantics. No rejected incoming body/header/attachment can overwrite canonical native graph. Helper parse/read failures raise instead of substituting incoming payload; existing caller noncritical handling logs and skips graph mutation. TDD: EML real upload red native99 vs Media1; Gmail real SQLite/provider mock red incoming replacement vs original body; legacy helper regression added. Final focused combined suite 57 passed, 8 existing warnings, 0 outbound attempts (/tmp/email-reimport-final.log); covers 12 EML/ZIP/MBOX overwrite/backfill cases, legacy preservation+tenant scope, Gmail normal/empty body labels and existing sync/chunk tests. Root fixed full-overwrite safe_metadata and highlight deadlock exposed by these tests. New files Ruff clean; existing lint baseline unchanged: persistence25, connectors_worker3, policy test2, chunk test2. Prod Bandit zero findings; test touched new/updated files excluding existing large policy fixture zero findings (B101 skipped only); policy file retains20 pre-existing dummy credential findings. Reports /tmp/bandit_email_reimport_13251.json and /tmp/bandit_email_reimport_tests_new_13251.json. No staging or commit; root integrates.

Review follow-ups complete: accepted overwrites save safe_metadata; all upload/Gmail graph paths use persisted content, with tenant-scoped legacy metadata recovery and live Gmail labels preserved. Highlight updates reuse the Media transaction, avoiding lock contention and preserving rollback atomicity. All three acceptance criteria now covered; final integrated validation underway. Design: Docs/Design/email-core-correctness-13251.md.

Final verification: combined offline suite 218 passed/2 native-PST skips, zero outbound attempts; focused final mocked Gmail slice 25 passed; legacy helper 1 passed; extra Media/Collections compatibility 22 passed. Counts overlap. Production Bandit zero findings; expanded focused tests zero with B101 excluded; existing policy fixture20 unchanged. Ruff touched email modules/new tests pass; legacy files retain65 baseline diagnostics, no new ones. Independent reviewer reports no remaining blockers. Evidence and remaining deployment/PST/PostgreSQL/scale gates: Docs/Operations/Email_Core_Validation_2026-09-13.md. Implementation plan completed and removed per repo policy; design retained.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed email identity collisions, safe metadata loss, reimport Media/native content divergence, legacy metadata preservation, date normalization and overflow, and overwrite highlight transaction deadlocks. Added strict synthetic EML/ZIP/MBOX and mocked Gmail regressions. Related TASK-13253 cursor and TASK-13254 fixture evidence fixes completed; docs refreshed. Personal Gmail and external models untouched. Local release validation remains distinct from optional live Gmail.
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
