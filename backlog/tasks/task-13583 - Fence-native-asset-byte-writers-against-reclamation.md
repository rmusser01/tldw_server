---
id: TASK-13583
title: Fence native asset byte writers against reclamation
status: Done
labels:
- chat
- ux
- h2
- storage
dependencies:
- TASK-13582
- TASK-13261.4
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue accepted H2/all-ten chat remediation with the private physical candidate adapter. Reuse LocalSyncBlobStore under the destination user chat_assets root, secure stable per-candidate guards, exact durable operation/descriptor and confirmed reservation admission. Durably fence reclaiming before waiting, drain writers and remove final bytes plus upload residue outside DB transactions. Do not expose public routes/capabilities or pretend to implement quota authority, asset adoption, MIME admission or full H2 in this increment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Only the original authenticated owner, live generation and exact reserved candidate with confirmed matching quota intent can write verified chunks or publish; malformed/shared storage identities fail before byte writes.
- [x] #2 Writers and cleanup use the same secure stable guard outside deletable namespaces; no DB transaction remains open while waiting for a guard or performing byte IO.
- [x] #3 Reclamation durably fences the candidate, drains existing chunk/final writers and removes exact final bytes and all upload residue; a waiting stale writer cannot recreate files.
- [x] #4 Unknown state, unsupported secure locking, unlink/verification failures and unconfirmed quota release remain conservative and retryable; discard is finalized only under the guard after confirmed release.
- [x] #5 Real filesystem and SQLite plus official PostgreSQL fixture where available verify paused chunk/final publication, generation replacement, process death, retry, isolation and failure controls.
- [x] #6 Two independent bounded source reviews and touched formatting/static/Bandit/normal hooks pass; ledger records quota authority, adoption/readers/routes/character/client/public usability gates still open.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no new ADR. Accepted Docs/ADR/049-chat-history-selection-ownership.md and Docs/ADR/050-native-chat-fork-storage-lifecycle.md govern; ADR059 canonical tracking applies. Accepted H2 design section8 specifies this fence. Fresh origin/dev39b53a876f2cfefcede02c92ae7dacf88f7a98ec is included; highest local/dev/open-PR13582 checked before explicit13583. Existing user permission for two independent reviewers applies. Own five-stage plan follows; no public capability and no primary checkout/browser mutation.
Initial causal RED and real SQLite/filesystem qualification recorded in /private/tmp/chat-native-assets-13583-{red,safety-red,descriptor-red,physical-green}.log. Physical suite: 44 passed, 44 official PostgreSQL skips, including actual spawned chunk/final drain and process death. Connected runner is in progress; source is frozen until its exit. Two independent reviewers identified immutable DB-owner/user-root binding, named enclosing user-directory replacement, case-insensitive upload identity aliases and post-admission-lock lease freshness. These must receive causal tests and repair before final qualification; public H2/quota authority/adoption/MIME remain open.
Final private physical qualification: 234 passed/187 skips/26 warnings across seven connected files, supported Python3.12, exit0, /private/tmp/chat-native-assets-13583-final-connected.log. 182 PostgreSQL fixture cases are unavailable; five SQLite skips are PostgreSQL-only controls (four inherited workspace driver cases and the new held-row-lock control). Review RED8, canonical-tenant RED1 and captured-owner RED3 prove repairs. Earlier interrupted pre-captured-owner run and formatter/fixture diagnostics are excluded. Two independently authorized final three-file source reviews CLEAR. New whole files and eight added functions match Black; original whole-file/selected old-scope formatting drift is retained without a full-file formatting claim. Ruff0; production Bandit0 findings/errors. AST proof preserves all13 pre-existing LocalSyncBlobStore methods exactly. Server-derived canonical tenant/user root, immutable captured owner, named user/root/lock rechecks, lowercase ASCII upload tokens and post-lock current lease are required. Native byte publication remains reserved pending real MIME/adoption authority; seeded confirmed intents are fixture controls, never real quota evidence. Public callers/routes/capabilities remain absent. Fresh dev advanced to2a935383ac1305fac3af26f2596c203b31a29909 after this source freeze; required integration is tracked by TASK13584. Original39b remains included. Normal owned hooks/commit pending.
Normal exact-owned eight-file hooks passed, /private/tmp/chat-native-assets-13583-hooks.log. All five stages complete and only this plan retires to /private/tmp/IMPLEMENTATION_PLAN_chat_native_asset_fence_13583_completed.md. Final metadata hooks and incremental commit follow; TASK13584 and all parent/public H2/usability gates remain open.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Qualified inactive native candidate byte fence with canonical captured owner, secure named-user/root/lock identity, exact layout/no-overwrite verification, post-lock lease freshness and conservative writer-drain/crash cleanup. Final connected234 passed/187 skips/26 warnings, supported Python3.12;182 unavailable official PG cases and5 SQLite PG-only controls. Two bounded source reviews CLEAR, Ruff/Bandit0, normal owned hooks pass, public Sync13 existing methods unchanged. No real quota authority, MIME/adoption, public H2 or all-ten acceptance claim; latest dev integration continues in TASK13584.
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
