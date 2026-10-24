---
id: TASK-13932
title: Design and implement secure Windows native asset storage
status: To Do
labels:
- windows
- security
- native-assets
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred by the human during Calendar PR3019 CI qualification: Windows remains unsupported and fail-closed for native asset physical I/O for now. Separately design and implement native Windows secure storage with equivalent anchored no-follow filesystem, tenant ownership, quota, and cleanup-fence guarantees before enabling successful native asset operations. Do not weaken the existing capability guard or use an ordinary pathname fallback. Evidence: PR3019 exact-eb workflow40087370412 Windows3.12 db-management-n/job120856685403; current scope tracked in TASK-13356.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Approved threat model and design map Windows handle-relative identity, reparse-point/junction/symlink refusal, private ACL enforcement, atomic publication, deletion and cross-process locking to existing guarantees; ADR assessment recorded before implementation.
- [ ] #2 Implementation preserves credential and tenant isolation, immutable payload integrity, durable reservation/adoption/reclamation and quota-release fences; unsupported or uncertain primitives remain fail-closed.
- [ ] #3 Real Windows Python3.12 and3.13 positive, adversarial path/ACL/identity, cancellation, crash and cross-process race tests pass; no stubs or simulated POSIX capability claims substitute for native qualification.
- [ ] #4 Supported POSIX behavior, all current negative security tests and Jobs authority remain unchanged, with scoped security validation, independent requesting-code-review and normal hooks before enabling Windows support.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Human explicitly chose option1 for now (retain unsupported/fail-closed Windows behavior with rejection coverage) and requested this separate option2 backlog task. This record is follow-up only, not permission to implement a new storage architecture within the CI watcher. ADR required: yes for the eventual native Windows storage security design; no new ADR is being authored by this tracking-only task creation. Existing native storage code is tldw_Server_API/app/core/Chat/native_chat_assets.py and core/Sync/v2/blob_store.py. Link PR https://github.com/rmusser01/tldw_server/pull/3019 and TASK-13356. Duplicate search checked Windows native, secure Windows and Windows tasks; no existing secure native asset Windows implementation task found. ID13932 exceeds max13931 across fresh origin/dev and all open PR file inventories.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
