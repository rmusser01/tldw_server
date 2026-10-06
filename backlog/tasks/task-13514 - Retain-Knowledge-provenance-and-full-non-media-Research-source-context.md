---
id: TASK-13514
title: Retain Knowledge provenance and full non-media Research source context
status: In Progress
labels:
- knowledge
- research
- provenance
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow up PR3196 limitations: canonical note provenance currently depends on a content marker, and non-media Research handoffs carry retrieved excerpts. Trace existing Notes metadata and Research source APIs first, then reuse canonical persistence and owner-scoped content reads for the smallest compatible implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Preserve original Knowledge provenance independently of editable note content and across updates
- [x] #2 Use available canonical full non-media source content in Research while retaining identity, citations and evidence excerpts
- [x] #3 Preserve owner, deleted-source and stale-save fences with regression coverage
- [x] #4 Fresh server workspace imports wait for existing server confirmation and retain canonical note provenance without replaying completed imports
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented full owner-scoped canonical Notes snapshots through existing Notes GET and media ingestion. Original retrieved evidence remains separate; original source version survives provenance save/reopen. Denied, deleted, mismatched, invalid-revision and retired-workspace reads fail visibly before upload. Three focused suites pass: 80 tests. Backend provenance remains a coordinated Notes/Sync compatibility design, not an unsynchronized column.
Live browser follow-up exposed a canonical research-note save failure after sources attached. Investigating exact response and note link validation; all original excerpts and latest full-note v2 are retained in the pending draft.
Root cause confirmed: canonical import-note persistence was selected only by legacy migration tombstones. Fresh server workspaces can complete a local draft checkpoint without a canonical note or structured marker. Reuse the existing canonical Notes path once the existing server workspace identity is confirmed; preserve local fallback before that confirmation.
Final review replaced late local-to-server upgrades with the smaller readiness gate: fresh server-capable imports wait for existing workspace confirmation, then use canonical Notes persistence. Completed local imports retain retirement behavior. Streamed/restored media_db source_id/sourceId are reused as canonical media instead of duplicated excerpt uploads. Latest six affected frontend suites pass 150 tests; API regressions pass 91; final builds and both type checks pass.
Final real-browser check found two logical source groups can resolve to one canonical media ID when uploadMedia reuses an existing note snapshot already returned by retrieval. The later group replaces per-source evidence and loses the original Notes identity/version in structured provenance. Fix the shared attachment evidence calculation using all currently resolved payload sources for that media ID; add a mounted overlap regression before implementation.
Final browser proves complete canonical v2 note snapshot includes a post-retrieval appendix absent from original excerpts; canonical note revision 4 retains a portable structured marker and five source attachments after an actual UI Update (HTTP 200) and reopen. Fixed same-media-ID source collisions by retaining all resolved references and original note revisions; both source orders failed before and pass after. Final affected frontend set: 152 passed; both clients types/builds pass; independent review has no actionable findings. AC1 remains open for the concrete Notes/Sync sidecar proposal requiring approved design and ADR.
Published bounded follow-up fixes/evidence as draft PR3205 against latest dev: https://github.com/rmusser01/tldw_server/pull/3205 . The separate human Change summary, unresolved device/participant qualifications, independent provenance design and existing CI integration remain explicit in the PR/report. Disposable task runtimes/browser profiles are stopped; managed worktree retained for review.
2026-10-06: Requester approved the independent notes.provenance v1 capability proposed in Docs/Design/2026-10-06-knowledge-followup-source-context.md. Proceeding with a dedicated ADR, staged implementation plan, canonical Notes/Sync integration and compatibility validation on PR3205. AC1 remains open until structured provenance survives ordinary Markdown edits, replay and parent deletion with independent conflict fencing.
Approved capability implementation recorded in ADR065 and Docs/superpowers/plans/IMPLEMENTATION_PLAN_knowledge_provenance_20261006.md. Four stages cover strict storage, canonical Sync authority/atomic lifecycle, Notes/client compatibility and release verification. Inline execution follows the requester’s approval.
Independent provenance storage stage complete and independently approved (39eb8d7e19): 37 focused + 241 affected regression tests passed, including four live PostgreSQL checks; Bandit zero findings and normal pre-commit passed. SQLite75/PostgreSQL79 install owner-scoped independent records; shared NoteStore deletion tombstones evidence and ordinary restoration does not revive it. Continuing canonical Sync lifecycle/recovery integration.
Canonical Sync stage committed d2ff2c881f, independent task review pending. Strict notes.provenance capability, parent/child deletion and transactional projection, exact independent versions, bounded owner-verified enrollment and recovery are implemented. 66 final focused tests passed with live PostgreSQL; 806 affected regressions passed before only the final strict revision guard, covered by the focused run. Bandit zero findings; Ruff and normal pre-commit passed. API/client integration remains in progress; no global suite/device/participant qualification claimed.
Sync stage independently reviewed and complete after fixes344b24d4cc and93143bfdad: claimed conflict deletion now atomically accepts/projects child tombstone with retry and PostgreSQL connection guarantees; unrelated large plain Notes no longer block source-history enrollment, and required existing target retains exact owner/version source authority. Scoped re-review found both addressed and no new issues.133 final enrollment/profile/storage checks and493 conflict/store/replay regressions passed with PostgreSQL; zero touched Bandit findings. Notes API stage is now in progress; capability AC1 and final release verification remain open.
API stage db774657d8 implemented structured read/write and portable export, with 44 new API and225 legacy regressions passing. Independent review found inactive durable-retry and encryption read-policy gaps; both are being fixed before shared-client integration. Local receipt extension reuses the provenance store and canonical Notes transaction; final reviewed verification remains pending.
API stage complete and independently reviewed after db774657d8 plus9169853b38. Structured Notes reads/writes/restore and all portable exports preserve independent heads; durable local owner-scoped receipts recover sourced create/update/bulk/import/restore acknowledgments atomically even with Sync disabled. SQLite76/PostgreSQL80 migrations and forced receipt RLS verified with real restricted-role Postgres. Final150API/storage/organization tests and65affected Sync cases passed; zeroBandit and hooks passed, unchanged lint/deprecation diagnostics recorded. Read/export/replay policy gates reject unavailable encryption without leaking markers. Scoped re-review clean; shared clients now in progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed full owner-scoped Notes source snapshots, fresh-server canonical persistence, canonical media reuse and overlapping-source reference preservation using existing APIs. Portable marker survives aware-client edits but remains part of editable Markdown. Independent canonical Notes/Sync provenance is proposed, not implemented; keep In Progress for AC1.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
