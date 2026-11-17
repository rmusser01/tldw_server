---
id: TASK-13542
title: Keep acknowledged chat deletion and owned device discovery consistent
status: Done
labels:
- chat-ux
- webui
- extension
- data-integrity
dependencies:
- TASK-13541
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Saved chat Trash and permanent deletion acknowledge the server but compact bulk and full-page single/bulk paths leave their owned local mirrors searchable. Compact search/recent cached results also lack the existing Dexie mutation observer and can reoffer deleted rows even after cleanup. Reuse the owned mirror-removal primitive and existing change observer at the real deletion/discovery roots, preserving current account, namespace and pending-write safeguards.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Mirror cleanup respects global/workspace namespace, account/view retirement and pending local settings writes; cleanup failure does not misreport server acknowledgement.
- [x] #2 Compact search and recent discovery react to actual local history changes without changing query or tabs, retire held old reads and clean up observers.
- [x] #3 Causal deployed component/service checks, client types, authored static checks and two independent reviews are recorded; native and all-ten limits remain explicit.
- [x] #4 Every user saved-chat Trash/permanent-delete path cleans only the acknowledged owned cache; declined tab closure preserves live text/draft but does not leave a deleted read cache. Failed server operations preserve copies.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Read-only full caller trace and two independent reviewers confirm the root. Compact single cleanup is the positive control; four full-page single/bulk soft/hard paths and compact bulk omit it. KnowledgeQA deletes use their separate provider-owned search-history cache; abandoned native promotion/branch cleanup has no published mirror and stays transport-only. The deployed chat-rag domain overrides the older API class method; no local deletion belongs in generic transport. Existing LocalChatList mutation observation is reused for compact discovery. Highest IDs just verified local13541/dev13530/openPR13526; fresh origin/dev2c5f19d032 is included. ADR required:no; ADR-049/050 ownership/lifetime and ADR-059 tracking govern the accepted D7 device cache/Trash policy, no new architecture/dependency/storage authority. Plan:Docs/Design/IMPLEMENTATION_PLAN_chat_deleted_mirror_discovery_13542.md.
Independent caller review also finds QA normal user-history deletion can leave a mirror after that conversation was opened in /chat; include that user path, keeping unpublished rollback cleanup separate. Cache cleanup follows authoritative server ACK independently of compact view closure; failed device cleanup stays a separate warning and must not classify acknowledged Trash as backend failure or replay it. Existing observer invalidates readers synchronously through a local read revision before rerender/effect cleanup. Exact namespace includes only truly unqualified legacy global rows without a workspace ID. Accept/reject source fixtures remain explicit.
Reviewer continuation checks reproduce seven additional real-provider failures: same-thread typed input and edit/Undo are lost during held server ACK, device cleanup and Retry; a retained rendered Delete callback adopts a later same-account topic revision. Capture the rendered view revision and track admitted SET_QUERY intent at the existing owned dispatch seam, guarding only query reset while still removing acknowledged history and deleted active results. Evidence: /private/tmp/chat-mirror-discovery-13542-input-red.log (7 fail/41 positive controls). Two Temporary wording oracle failures reproduce against byte-identical HEAD production/test paths; update only the stale expected accepted TASK13537 wording, retaining all behavioral assertions. Evidence: /private/tmp/chat-mirror-discovery-13542-copy-baseline.log.
Stages1-3 Complete. Causal deployed component/service baseline:22 failures/174 controls in6 suites, /private/tmp/chat-mirror-discovery-13542-causal-red.log; six additional held cleanup retirement controls; actual QA new-topic and Retry completion2 failures; final query/retained-callback7 failures/41 controls in -input-red.log. All failure-first runs retain declared storage/network/observer I/O and exclude fixture setup/wrong-cwd attempts. Final reviewed matrix:768/768 checks across22 suites with no skips, /private/tmp/chat-mirror-discovery-13542-reviewed-matrix.log. Both final clients typecheck (reviewed-web-types.log, reviewed-extension-types.log). Effective unchanged-rule lint covers14 authored TS/TSX,17baseline/17current/0added;19 owned formatting regions,3 exact English mirror values with no unrelated locale changes and diff check pass (reviewed-static.log and -eslint.json). Bandit N/A: authored TS/TSX/JSON/docs only, no repository Python source. Fresh origin/dev2c5f19d0328360d45bce99b8bae75ce252cc312f fetched and included. State reviewer clears final query/view continuation; second final source review pending. Native IndexedDB commit/locks, packaged extension, physical/AT/participants and parent all-ten journeys are not inferred from declared I/O fixtures.
Stage4 Complete. Both independently authorized reviewers clear the final source, all actual user deletion callers, query/view retirement, truthful partial outcomes and observer/storage boundaries; neither review is a native/backend validation claim. All4 plan stages completed and recorded in the parent design checkpoint; only this owned completed temporary plan is retired. Parent TASK13458 and TASK13540 positive native acceptance remain In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reused acknowledged mirror cleanup across full-page/compact/QA user deletion and the existing Dexie change observer for current device discovery. Preserved exact ownership/namespace, pending settings and transaction lifetimes; device failure is a separate warning. Shared QA completion and query/view revisions preserve newer topics/drafts including edit/Undo and Retry. Final768/768 in22 suites, both client types, effective covered lint14files17/17/0,19 owned format regions,3 exact English mirrors and diff checks pass; two source reviews clear. Bandit N/A for TS/JSON/docs-only changes. Latest dev2c5f19d032 included. Native/packaged/physical/AT/participants and all-ten parent limits stay explicit.
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
