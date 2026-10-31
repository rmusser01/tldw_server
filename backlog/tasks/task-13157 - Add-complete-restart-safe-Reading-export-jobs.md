---
id: TASK-13157
title: Add complete restart-safe Reading export jobs
status: To Do
assignee: []
created_date: '2026-09-03 02:32'
updated_date: '2026-09-03 02:41'
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
Keep the current page-scoped streaming `/reading/export` route and response unchanged for existing
clients. Add asynchronous, user-scoped export-job routes that evaluate one explicit filter scope
coherently, write every matching item exactly once to a private managed artifact, expose bounded job
history/detail, and support authorized download and confirmed cleanup. Interruption is restart-safe
and never publishes a partial artifact as complete. A caller-generated export request key prevents
duplicate artifacts after a lost create response; reusing a key with a different normalized scope or
content payload returns a bounded conflict. Each artifact carries the versioned Server-native
portable-schema identifier and manifest consumed by S5b. Docs-info advertises exact
`hasReadingExportJobsV1=true` only when complete-scope export, job lifecycle, artifact retention,
and cleanup guarantees are active.

S5a is one reviewable Server PR boundary because its routes, job state, managed store,
request-key idempotency, private artifact lifecycle, and portable manifest form one new
Server-native export-job aggregate; none is truthful or independently useful without the others.
It excludes re-import, generic backup, UI work, Local behavior, and additional export formats.
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
