---
id: TASK-13158
title: Add Server-native Reading export re-import
status: To Do
assignee: []
created_date: '2026-09-03 02:33'
updated_date: '2026-09-03 02:34'
labels:
  - collections
  - reading-list
  - import-export
  - portability
dependencies:
  - TASK-13157
references:
  - 'tldw_chatbook:TASK-18919'
  - >-
    tldw_chatbook:Docs/superpowers/specs/2026-09-01-collections-followup-backlog-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Retain Pocket JSON and Instapaper CSV import, then admit versioned Server-native JSONL/ZIP artifacts produced by S5a. Portable fields are submitted URL, title, summary, freeform note, status, favorite, tags, published/read timestamps, optional sanitized text/clean HTML, and capture-owned highlights. Exclude database/user IDs, authoritative created/updated timestamps, Media/linked-Note identities, generated audio, offline/archive files, and internal metadata. Allocate new identities and canonicalize URLs. On canonical collision preserve existing scalar/state/content, union tags, and add only nonduplicate capture-owned highlights by stable fingerprint. Re-import is idempotent. Advertise exact `hasReadingNativeImportV1=true` only with this versioned field/collision contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Import accepts the versioned Server-native JSONL/ZIP artifacts produced by S5a while retaining Pocket JSON and Instapaper CSV compatibility.
- [ ] #2 Empty-authority import reproduces every declared portable field and allocates fresh local identities without importing user/database IDs, external links, private paths, or internal metadata.
- [ ] #3 Canonical-URL collisions preserve existing scalar/state/content fields, union tags, and add only nonduplicate capture-owned highlights using the documented stable fingerprint.
- [ ] #4 Re-importing the same artifact is idempotent and produces no duplicate capture or highlight; malformed, future-version, oversized, and partially corrupt input fails safely.
- [ ] #5 Docs-info advertises `hasReadingNativeImportV1=true` only with the field/collision contract and public portability documentation.
- [ ] #6 S5a is a same-repository dependency, and focused parser/job/database/API/security and multi-page round-trip tests pass with the applicable ADR recorded or amended.
<!-- AC:END -->
