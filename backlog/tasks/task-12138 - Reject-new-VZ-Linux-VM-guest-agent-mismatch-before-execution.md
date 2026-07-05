---
id: TASK-12138
title: Reject new VZ Linux VM guest-agent mismatch before execution
status: Done
labels:
- sandbox
- vz-linux
- guest-agent
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the VZ Linux guest-agent mismatch gap found in current code: newly-created VMs with explicit guest-agent contract mismatch should fail closed before guest execution, should not store reusable session control, and should be terminated/cleaned up.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add a focused failing runner test for newly-created VM guest-agent mismatch.
- [x] #2 Reject explicit mismatch after create_vm and before exec_guest.
- [x] #3 Do not store reusable session-control rows for mismatched newly-created VMs.
- [x] #4 Terminate/cleanup the mismatched VM through the existing failure path.
- [x] #5 Run focused pytest, py_compile or equivalent, diff check, and Bandit on touched production code.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
- Write RED test in VZ Linux runner tests.
- Implement the smallest runner guard using the existing guest-agent classifier and failure cleanup path.
- Run focused verification and update task final summary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Verified current code before editing: existing session reuse already rejected explicit guest-agent mismatch, but the newly-created VM path executed before checking create_vm metadata.
- Added a focused regression for a newly-created session VM with mismatched guest metadata. The RED run failed because the command reached exec_guest and completed.
- Added a create_vm gate that rejects explicit guest-agent mismatch before image-store clone prep, session-control persistence, or exec_guest. Unknown metadata remains allowed to preserve existing metadata-light guest behavior.
- The guard sets should_terminate_vm before raising so the existing finally cleanup terminates the rejected VM.
- Review follow-up: added marker, fixture type annotations, and docstrings to the new regression test/local helpers.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Newly-created VZ Linux VMs with explicit guest-agent mismatch now fail closed before execution.
- Mismatched newly-created session VMs are not stored for reuse and are terminated through the existing cleanup path.
- Verification passed: focused regression pytest, related runner pytest selection, guest-agent helper pytest, py_compile, git diff --check, and Bandit on the touched runner file.
- PR review hygiene comments were addressed in the regression test.
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
