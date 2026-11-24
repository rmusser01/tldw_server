---
id: TASK-15624
title: 'PDF search: stop index rebuild loop, build lazily, reset on document change'
status: Done
labels:
- webui
- performance
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
usePdfSearch rebuilt its index in an unbounded loop after a failed build (isIndexing in effect deps), built eagerly for every opened PDF, and kept a stale index after the document changed. Failed builds are now recorded and retried only when search reopens; build starts when searchOpen goes true; index and in-flight builds are keyed to the document proxy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A failing build is attempted once, retried only when search is reopened
- [x] #2 No getPage calls until search opens
- [x] #3 Index resets on document change and a stale in-flight build cannot overwrite it
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: bug fix inside one hook, no architectural decision; ADR not required (checked Docs/ADR/README.md scope).
Round 2: ref assigned without re-render noticed via pdf-text-layer-rendered; tab switch hides the previous proxy until the ref changes; entry cleared on proxy change. Pre-fix loop count 51 (mock capped at 50).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
usePdfSearch builds lazily on searchOpen, keyed to the document proxy, records failures and retries only on reopen.
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
