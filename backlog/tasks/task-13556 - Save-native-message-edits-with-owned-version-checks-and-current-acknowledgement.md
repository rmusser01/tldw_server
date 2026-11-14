---
id: TASK-13556
title: Save native message edits with owned version checks and current acknowledgement
status: Done
labels:
- frontend
- chat
- ux
- bug
dependencies:
- TASK-13555
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Loaded native ordinary and character chats still refuse plain Save via captureLocalMutationOwner. Use the existing server message version-CAS endpoint through owner-scoped reads/writes, qualify a canonical text target, preserve graph/metadata and edited drafts on conflict, failure or retired editor/view/account, and refresh only the owning current H1 selection after acknowledgement. This is the next all-ten remediation unit, not program completion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Native canonical plaintext user and assistant Save performs one owner-scoped version-CAS write and preserves graph/metadata without generation or local-history writes.
- [x] #2 Conflicts, partial/unknown writes and late source/account/editor/selection changes retain edited text and do not overwrite a replacement view; only verified current acknowledgement closes the editor.
- [x] #3 Shared full/compact handlers, public SDK transport and real backend CAS are covered by red-first tests, client compilers/static checks, independent review and truthful native/program limits.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new durable rule. Existing ADR049 chat history owner authority and ADR050 storage lifecycle plus accepted D7 CM01 apply. Reuse PUT /api/v1/messages/{id}?expected_version= server CAS; private owner scope and H1 source/installation guards. Search message edit/native edit found no overlapping unfinished unit. Latest fetched origin/dev97ea9cd remains ancestorHEAD6c4c1ff28d. Highest ID13555 confirmed across dev, all open PR paths and both local backlogs. User explicitly requests all identified fixes and enhancements; accepted staged design remains authorized.
Plan: Docs/Design/IMPLEMENTATION_PLAN_chat_native_message_save_13556.md. Scope includes shared handler, owner service, H1 guarded canonical refresh, public/deployed SDK and actual full/compact callers. Backend PUT substitutes roleplay placeholders; a subsequent scoped raw canonical capture must qualify acknowledgement. Preserve content-only metadata/graph, original cursor and edited drafts; no generation or local native writes. Independent state and transport reviewers are investigating the contract.
Red-first: /private/tmp/chat-native-save-13556-red.log has 14 existing behavior failures (eight actual full/compact plain Save refusals, four SDK owner-scope/pre-abort controls and two exact scoped route admission controls) plus two new service/H1 method absence failures; five policy negative controls passed, 615 filtered skips. Native Save now uses captured server message ID/owner, verifies original target version/revision/metadata before one scoped content-only CAS, then guarded H1 refresh checks exact raw content and ACK revision before atomic installation while preserving cursor/siblings/metadata. Extended caller tests cover held read/write/refresh with editor, New, same-ID epoch and account retirement. Web compiler caught a capture-result narrowing issue; corrected with an explicit never-return guard, verification pending.
Real authenticated controls expose a root expected-account gap in character_messages: read/list/search/edit/delete ignore X-TLDW-Expected-User-ID (five causal red failures); add already returns412. Extend this unit to reuse require_expected_user at the module router and remove its duplicate add-route declaration. Absent-header compatibility remains. The initial positive fixture used unprotected sibling posts and correctly required legacy review; qualify a real admitted user with two settled replies instead. Opaque target digest and final scoped raw numeric-version GET remain distinct from CAS. Final client36-suite matrix1753 runner-passes includes one inherited Regenerate expected failure, excluded from acceptance; both compilers pass and touched lint has zero introduced findings.
Final:36 client suites1753 runner-passes =1752 positive +1 inherited Regenerate expected failure excluded from acceptance; both type gates;28 touched TS/TSX lint28 baseline/current warnings, zero errors/introduced; AST/comment authored formatting; backend50/50 plus3 compatibility/route-binding; Python syntax/Bandit three touched files with B101 excluded for assertions and zero findings. State/transport independent bounded source reviews CLEAR, including shared router correction. Fresh dev97ea9cd remains included. All4 owned stages complete, plan retired with completed /private/tmp/chat-native-save-13556-completed-plan.md; master design records evidence and pending all-ten work. Current native browser/packaged/physical/AT/participant/whole-program acceptance remains unqualified.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Native full/compact ordinary and character plaintext Save uses scoped numeric CAS, fresh opaque target verification, raw read-back and current origin/editor acknowledgement without replacing newer input or cursor/siblings/metadata. Shared message-router expected-account guard closes a reproduced stale-account read/write/delete gap while preserving absent/matching clients. Verified client/API/static/security gates and two independent source reviews; native and all-program limits remain explicit.
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
