---
id: TASK-13530
title: Require explicit review for unqualified chat copies
status: Done
labels:
- chat
- ux
- extension
- handoff
dependencies:
- TASK-13457
- TASK-13457.11
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair the existing same-origin Copy draft & page references flow in the approved all-ten chat UX program. Opaque packages have no source-owner proof, but an empty verified destination currently imports and consumes them without review. Copy also opens a character route whose unqualified ID can resolve against another account. Require the existing explicit review for literal text/context, preview bounded page references and shortening before choosing, and open canonical /chat without transferring character authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every valid same-origin copy requires an explicit current Insert/Replace or Cancel choice, including empty, whitespace-only and context-only destination workflows; no automatic draft/context write or package consumption occurs.
- [x] #2 Choice-time rereads, expiration/read-failure recovery, destination owner/H1/draft/route fences, held-read retirement and existing WebUI settings consent remain intact.
- [x] #3 Actual full-page composer review exposes exact copied draft and complete bounded page title/URL/snippets plus shortening before acceptance; user text remains escaped and source ownership is described truthfully.
- [x] #4 Copy opens canonical /chat and carries no unqualified character choice; explicit Open/Expand still opens its intended character/history destination and source draft stays preserved.
- [x] #5 Failure-first behavior checks, affected regressions, both client type checks, no-added diagnostics, locale/task checks and two independent reviews are recorded before the local checkpoint.
- [x] #6 Legacy or mixed copy URLs cannot automatically apply bare character or character-session route references before review or after acceptance, Cancel or invalid-copy cleanup; unrelated parameters, hidden second copies and qualified history selections remain intact.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Parent TASK-13457 and the all-ten program remain In Progress. Latest origin/dev 26ae4fd679ddd43f9f904cd3cca1b1cf8af51e48 is fetched and included in HEAD35d72c2aa5; the primary checkout is untouched. Task ID13527 exceeds the highest13526 found across origin/dev and all50 open PR file lists. Duplicate search found no active equivalent unit. Plan: Docs/Design/IMPLEMENTATION_PLAN_chat_handoff_review_13527.md; governing approved design: Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md. ADR assessment: no new ADR; apply ADR-049 history/source authority and ADR-050 owner/lifetime fencing, with ADR-059 backlog-py tracking. This unit removes automatic authority from an existing literal-copy flow and adds no protocol, owner registry or persistence contract. Follow-up: unresolved destination owner currently waits silently; expose existing verification status/retry in a separate bounded unit. Native extension, backend, full roleplay and assistive-technology journeys remain parent acceptance gates.
Pre-checkpoint legacy route trace widened the same copy boundary: previously issued mode=character/characterId plus handoff URLs still reached independent Playground route effects, and token cleanup could expose them after review. Sender canonicalization alone is insufficient. Add failure-first actual Playground route controls; suppress character route intent while any copy token is present, then strip only the associated unqualified character-mode pointers during owned copy cleanup. Keep explicit character links and qualified history expansion. No source owner schema/protocol/registry is introduced; ADR049/050 still govern. TASK13527 remains In Progress until this destination bypass is verified.
Final affected matrix369/369 in12 suites; /private/tmp/chat-handoff-review-13527-final-matrix.log. Both WebUI typecheck and extension compile pass; logs chat-handoff-review-13527-{web,extension}-types.log. Added lint0 across8 TS/TSX files (lint-report.json); authored Form/locale and legacy-root formatting reports pass. Receiver77, source33, cockpit62 and actual Form50 pass; causal legacy-root red10fail52positive against exact HEAD with corrected actual QueryClient fixture, distinct from reproduced old fixture6fail44pass. Receiver red4, receipt red1, source red2, Form red2, cleanup red6 with1positive retained in /private/tmp logs. Official locale sync conversion matches9 review keys and2 descriptions; no unrelated locale churn. Both authorized independent state/transport cross-reviews clear legacy masking/cleanup, actual content review and source routing. Master design records roots, ADR049/050/059, corrected baseline fixture and evidence limitations; own completed temporary plan removed per repo staging rule. Bandit N/A: only TS/TSX/JSON/docs authored. Native worker, browser HashRouter, backend and full first-save/load/roleplay/assistive-tech proofs remain parent gates. Fresh dev26ae4fd unchanged and included.
On2026-10-07 the private chat record originally numberedTASK13527 was reidentified asTASK13530 before latest-dev integration: incoming Admin readiness PR3209 owns remoteTASK13527. Original chat implementation checkpoint03a902f8e2 and13527-named evidence artifacts remain immutable history. This canonical CLI-created replacement retains every description, criterion, completion state, implementation note and final summary; related current dependencies now useTASK13530. No Admin record is changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
All valid literal chat copies require review before writing or consumption. Exact bounded draft/page references and shortening are visible; context-only Add preserves text and reports references imported. Generic Copy cannot authorize character IDs, including old mixed URLs and stale ambient commands; current explicit links and qualified H1 expansion remain supported. Reuses existing owned choices/native details with no new registry or protocol. Verified369 checks, both client type gates and0added lint diagnostics; two independent reviewers clear bounded evidence. Parent remains In Progress for owner verification feedback/Retry and native/backend journeys.
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
