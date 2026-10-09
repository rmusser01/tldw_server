---
id: TASK-13569
title: Validate the native Edit-resend target atomically during admission
status: To Do
labels:
- chat
- webui
- extension
- backend
- ux-audit
dependencies:
- TASK-13554
- TASK-13565
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the target-only TOCTOU after final client Edit and resend capture. The target opaque revision remains private client state; before_message admission excludes the edited user target from retained ancestry, allowing a concurrent target-only content/role/metadata change to admit an obsolete edit proposal. Add an explicit qualified target admission precondition through ordinary and character paths under the existing owner transaction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both admission paths validate the captured edited user target ID, live role, opaque revision and direct parent atomically before append or provider dispatch.
- [ ] #2 Target-only edit, metadata/image/role drift and edit-Undo refuse with no new rows/provider call while preserving original graph and edited input.
- [ ] #3 Unchanged target admits the intended edited sibling once; generic Branch, ordinary Send and protected Regenerate remain valid.
- [ ] #4 Acknowledged existing-ID replay validates original immutable operation intent without retroactively rejecting a later target edit; conflicting intent is refused.
- [ ] #5 Schema/docs/client capability qualify the added precondition; older servers refuse protected Edit and resend before mutation without an unguarded fallback.
- [ ] #6 Real database and authenticated ordinary/character HTTP regressions, clients, touched security/static and current native evidence meet the declared contract.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Two independent source reviews confirmed current gap. Do not reopen13554 client freshness,13556 plain Save numeric PUT CAS,13558/59 atomic Regenerate or P1/P6 branch/freshness; these are different completed operations. D7 section6.2/ADR049 selected-source ownership and ADR050 immutable recovery govern;059 tracking. Assess the explicit new optional public precondition/schema/capability during design, without untyped metadata. Preserve ancestry-tolerant generic H1 validation. Meaningful red: real update_message after capture leaves retained ancestors/context unchanged, current selected append admits; expected protected operation409/no insert/provider. No empirical red claimed yet.
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
