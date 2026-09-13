---
id: TASK-13251
title: Fix identical-body email uploads merging distinct message identities
status: To Do
assignee: []
created_date: '2026-09-13 18:38'
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
- [ ] #1 Distinct email messages with identical bodies remain separate in legacy and normalized stores for EML ZIP and MBOX uploads
- [ ] #2 Repeat imports remain idempotent within the documented source scope
- [ ] #3 Replace TASK-13250 collision characterization with assertions for both original identities and consistent detail
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
