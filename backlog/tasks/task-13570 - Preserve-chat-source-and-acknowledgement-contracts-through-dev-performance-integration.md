---
id: TASK-13570
title: Preserve chat source and acknowledgement contracts through dev performance
  integration
status: In Progress
labels:
- chat
- ux
- integration
- performance
dependencies:
- TASK-13458
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Integrate refreshed dev4d6a85c4c693c4c83480d0f7c8046cfda0634c55 into the existing chat remediation branch, preserving reviewed edit/delete acknowledgement, rendered source and H1 ownership, explicit Stop focus, research callback authority, scoped streaming publication and exact model preference acknowledgements. Reuse incoming safe performance improvements and existing guards; repair demonstrated integration regressions before the merge is committed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Latest dev is an ancestor of the final chat remediation head and unrelated primary checkout/runtime artifacts remain untouched.
- [ ] #2 Actual full and compact row actions retain acknowledgements, original editor/source/target leases and explicit Stop focus; stale confirmations and callbacks cannot adopt replacement chats.
- [ ] #3 Research row policy and rendered callback changes invalidate cached actions and memoized callbacks.
- [ ] #4 Deferred streaming flushes and draft-only source changes respect original ownership, while current normal/native/model flows remain usable.
- [ ] #5 Affected causal and existing tests, both client types, scoped static/security checks and two independent source reviews are recorded truthfully.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Duplicate search finds TASK13552 Done for the earlier dev97ea9cd5fa integration; this is the new performance PR3210 overlap unit. ID allocation verified highestlocal13569/dev13533/openPR13534 across51 open PRs. Before source edits, plan:Docs/Design/IMPLEMENTATION_PLAN_chat_dev_performance_integration_13570.md. ADR required:no; preserving existing ADR049 owner/publication and private D7 contracts plus ADR059 tracking, with no new authority, durable protocol, queue or dependency. Independent read-only source traces identify integration hazards in latest-ref row caches, discarded Promise/lease arguments, Stop focus bypass, research builder/callback cache criteria, delayed legacy-streaming flushes and omitted draft address/revision subscriptions. Current bounded actor/default repair is committed as3b27865375; affected286 checks/types/lint/reviews pass, with unchanged mocked Current49 timer failures excluded and retained in TASK13549. All-ten/actor namespace/first-bind/projection/native/packaged/AT/participant gates remain open.
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
