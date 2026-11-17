---
id: TASK-13543
title: Locate uncertain chat creation by its durable client UUID
status: Done
labels:
- chat
- ux
- bug
- frontend
dependencies:
- TASK-13458
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Adopt the existing latest-dev P3 client chat ID in the shared first-create journal and both ordinary/character callers. Let an owned explicit recovery action find that exact chat after a lost creation response without another create or provider turn. Preserve retained input, H1 ownership, canonical aliases, concurrent initial-send exclusion, and legacy uncertainty.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Persist a strict client UUID before POST, send it in both ordinary and character create bodies, validate the matching ACK, and preserve one initial POST with legacy uncertainty blocked.
- [x] #2 Explicit Check server uses exact scoped owned GET and current H1 acceptance; successful lookup acknowledges the same canonical operation without claiming attempted text settlement or replaying create/provider requests.
- [x] #3 Late ACK, canonical alias, record replacement, storage failure, account/view/input retirement, failed lookup and differing draft boundaries retain the correct recovery state and input.
- [x] #4 Behavioral regression checks, both client type checks, authored lint/format/locales and independent source reviews pass with limits documented.
- [x] #5 Update the accepted remediation evidence and parent tracking; keep native confirmation and broader Stage 5 acceptance open.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Fresh origin/dev 2c5f19d0328360d45bce99b8bae75ce252cc312f is already included in HEAD bc5e1e392d; highest task ID checked across local, origin/dev and all open PR files was 13542. Authorized continuation of accepted D7/P3 and five-stage remediation. ADR assessment: no new ADR required; ADR-049 governs owner-validated H1 binding, ADR-050 governs native lifecycle fencing, and ADR-059 governs tracking. Reuses existing journal/CAS and server-supported ChatSessionCreate.id; no new public protocol, namespace or replay mechanism. Plan: Docs/Design/IMPLEMENTATION_PLAN_chat_client_uuid_recovery_13543.md.
Causal baseline: 16 direct failures / 1 legacy control in four suites, /private/tmp/chat-client-uuid-13543-causal-red.log. Excluded initial held-lookup baseline setup because the old implementation cannot enter GET; no causal claim from /private/tmp/chat-client-uuid-13543-red.log. New checks green 30. First complete six-suite matrix 369 passed / 2 fixture failures: prebuilt fake H1 ready receipt held the obsolete fixed ID; moved only declared receipt construction after actual request ID is known, retaining ownership assertions. SDK fixtures echo request IDs. Two independent design reviewers confirmed existing P3, canonical-operation ACK and lookup/settlement distinction.
Reviewed 376/6 full affected checks passing. Both independent implementation reviews found a retained Check start boundary. Actual composer + H1 causal baseline: 3 failures for same-address H1 epoch change, changed input and edit/Undo before old callback invocation; all had incorrectly opened the chat. Evidence /private/tmp/chat-client-uuid-13543-rendered-red.log. Added render-captured H1 fence and input revision at entry, preserving the existing owned-load acceptance flow.
Final: 520/520 behavior checks in13 actual affected/shared suites, no skips; 35/35 existing P3 FastAPI/database integration checks, no skips, using standard SQLite/PostgreSQL fixtures on plain non-Sync-v2 route. Both final client compilers pass. Effective authored lint10TS/TSX:113baseline/113current/0added;15owned formatter regions;2exact English nested/Chrome locale mirrors; diff check. Logs /private/tmp/chat-client-uuid-13543-reviewed-matrix.log, -backend-contract.log, -web-types.log, -extension-types.log, -static.log and -eslint.json. Both independent final source reviewers clear the rendered fence/input, canonical alias/late ACK and namespace boundaries. Bandit N/A because no repository Python code authored. Declared frontend I/O fixtures do not prove native IndexedDB/Web Locks, live Check or packaged worker acceptance. Initial setup/held-baseline attempts excluded, final WebUI narrowing repaired. Fresh origin/dev 2c5f19d0328360d45bce99b8bae75ce252cc312f included. Four owned stages Complete; completed temporary plan retired after evidence captured in master design. Native TASK13540 AC4 and Stage5 remain open; existing inaccessible dialog awaits user dismissal.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Persisted one client UUID before ordinary/character creation and added explicit exact owned Check server recovery without replay. Shared canonical ACK preserves newer bound/settled facts and input semantics. Rendered operation, H1 epoch and input revision stop retained callbacks adopting new intent. Verified520frontend +35backend checks, both types, no added authored lint, formatting/locales/diff and two independent source reviews. Native/package/complete-program acceptance remains open.
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
