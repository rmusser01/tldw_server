---
id: TASK-13350
title: >-
  Sync registry advertises notes.task domains that SYNC_V2_SUPPORTED_DOMAINS
  omits
status: To Do
assignee: []
created_date: '2026-09-23 00:51'
labels:
  - bug
  - sync
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
tests/Sync/test_sync_v2_domain_adapters.py::test_default_sync_v2_registry_advertises_personal_and_workspace_metadata_domains fails on:

  registry.supported_domains == sorted(SYNC_V2_SUPPORTED_DOMAINS)
  extra in registry: 'notes.task', 'notes.task_activity'

This is the test doing its job, not stale expectations. Two lists disagree:

- core/Sync/v2/models.py:189 SYNC_V2_SUPPORTED_DOMAINS is built from M1 + WORKSPACE + SOURCE_CACHE + MEDIA + NOTES_ORGANIZATION + NOTES_LINK + PERSONAL_CONTEXT. It DELIBERATELY excludes NOTES_TASK_SYNC_DOMAINS and NOTES_MOODBOARD_STUDIO_DOMAINS, which are instead named in SYNC_V2_KNOWN_DOMAINS (models.py:207) -- a distinction that exists precisely to separate 'known' from 'supported'.
- core/Sync/v2/factory.py:113-114 registers NotesTaskDomainAdapter() and NotesTaskActivityDomainAdapter() UNCONDITIONALLY, so the registry advertises them.

Note the contrast in the same factory: attachment.ref is gated behind SYNC_V2_ENABLE_NOTES_ATTACHMENT_SYNC (factory.py:96-99). The task adapters have no such gate.

SyncV2Service.supported_domains (service.py:858) defaults to SYNC_V2_SUPPORTED_DOMAINS, so the service advertises one set to clients while the registry serves another.

DECISION NEEDED, which is why this is filed rather than fixed: either notes.task sync is ready, in which case it belongs in SYNC_V2_SUPPORTED_DOMAINS; or it is not, in which case factory.py should gate the two adapters the way attachment.ref is gated. Changing the test to compare against SYNC_V2_KNOWN_DOMAINS would silence the inconsistency rather than resolve it.

Source: TASK-13344 triage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 notes.task and notes.task_activity are either in SYNC_V2_SUPPORTED_DOMAINS or gated out of the default registry
- [ ] #2 The registry and the service advertise the same domain set
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
