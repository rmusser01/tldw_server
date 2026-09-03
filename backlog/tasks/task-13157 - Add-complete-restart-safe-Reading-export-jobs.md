---
id: TASK-13157
title: Add complete restart-safe Reading export jobs
status: To Do
assignee: []
created_date: '2026-09-03 02:32'
updated_date: '2026-09-03 02:33'
labels:
  - collections
  - reading-list
  - export
  - jobs
dependencies: []
references:
  - 'tldw_chatbook:TASK-18919'
  - >-
    tldw_chatbook:Docs/superpowers/specs/2026-09-01-collections-followup-backlog-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the existing page-scoped streaming export unchanged and add one cohesive Server-native export-job aggregate: authenticated create/history/detail/download/cleanup routes, coherent complete-scope evaluation, restart-truthful job state, a private managed artifact, idempotent request key, retention/cleanup, and the versioned portable-schema manifest needed to identify a complete artifact. These pieces are inseparable for a truthful complete job and form one reviewable PR. Exclude re-import parsing/merge, generic account backup, Chatbook UI, Local behavior, and additional export formats. Advertise exact `hasReadingExportJobsV1=true` only with the full lifecycle.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Existing page-scoped streaming export remains compatible, while additive authenticated job routes expose bounded create/history/detail/download/cleanup behavior.
- [ ] #2 Each export job reads one explicit filter scope coherently and writes every matching capture exactly once to a private managed artifact, including scopes larger than one API page.
- [ ] #3 A user-scoped request key makes identical creates idempotent and rejects a different normalized payload with a bounded conflict.
- [ ] #4 Interrupted or failed jobs resume or terminate truthfully and never publish a partial artifact as complete; retention and confirmed cleanup are restart-safe.
- [ ] #5 Every completed artifact includes the versioned Server-native schema identifier and manifest required by S5b, and unauthorized users cannot discover or download it.
- [ ] #6 Docs-info advertises `hasReadingExportJobsV1=true` only with the full contract; a Server ADR plus focused job/database/API/security tests are complete.
<!-- AC:END -->
