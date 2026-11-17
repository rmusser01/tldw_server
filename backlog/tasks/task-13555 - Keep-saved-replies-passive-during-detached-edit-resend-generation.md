---
id: TASK-13555
title: Keep saved replies passive during detached edit-resend generation
status: Done
labels:
- frontend
- chat
- ux
- bug
dependencies:
- TASK-13554
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The native Edit & resend walkthrough retains the original timeline while generating a detached sibling reply, but chat-wide busy flags currently label the last old assistant as Generating and Stop, suppress its rendered content and announce Response complete on that unchanged row. Track the existing detached request lifetime as private presentation state; retain global mutation/queue/Stop ownership guards and original editor/draft integrity. Qualify full and compact native ordinary and character callers and shared reply rendering.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An unchanged saved assistant never receives generating/loading/Stop/aria-busy or a new completion announcement from a detached edited reply; original content and saved status remain readable.
- [x] #2 The actual full and compact dispatch lifetimes install and retire private detached presentation identity without a late action clearing a replacement request; ordinary and Compare behavior and global mutation locks remain intact.
- [x] #3 Red-first renderer and held mounted dispatch controls, relevant matrix, both type/static checks and two independent reviews are recorded with native preview and program limits explicit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. ADR049/050 and accepted D7 CM01 govern owner-bound graph actions; this unit changes only ephemeral request presentation and reuses current controller identity. No public wire, durable authority, backend mutation or new dependency. Latest fetched origin/dev97ea9cd remains ancestorHEAD4e6735d5ff. Search generation identity found no overlapping repair; highestID13554 confirmed across dev, all open PR paths and both local backlogs. User requests fixing all identified root causes and enhancements; accepted scope persists.
Scope/design review confirms chat-wide flags reach every shared row; detached native action deliberately has no visible placeholder. Use one ephemeral shared request token because composer and timeline instantiate hooks separately and can remount. A shared dispatch wrapper covers all full/compact early exits and exceptions, sets presentation identity before busy publication, normal dispatch clears it and finally retires only its own token. Shared Message derives response activity for loading, text/Markdown/Mermaid/dynamic UI, aria-busy, inline Stop, completion and settled-empty/reasoning-only notices; global action/composer/TTS/skill mutation gates remain intact. No full active-generation-ID migration is required for existing concurrent Compare producers.
Completed owned plan: /private/tmp/chat-detached-response-13555-completed-plan.md (retired tracked plan). Master: Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md, TASK13555 entry. Final source: shared State/core/account reset; chat-action-utils wrapper and full/compact submit entry; shared Message presentation/recovery/automatic source signature; connected fixtures and regressions. Two reviewers independently found pre-admission marker clobber/recovery mutation seam; causal red9 reproduced them plus automatic copy replay, then corrected and both reviewers CLEAR.
Evidence: /private/tmp/chat-detached-response-13555-red.log16 failures; review-red.log9 failures; final matrix.log32 suites /1413 runner passes=1412 positive+1 existing Regenerate it.fails counterexample excluded, no ordinary skips; web-types-verified.log and extension-types-verified.log exit0; static.log18 files,0errors,0introduced warnings (18 current=18 baseline); format.log authored AST/comment equivalence; git diff checks. No authored Python, Bandit N/A. Fresh dev97ea9cd5fa7e3a61ee4e7c56f3d9643b7311d511 remains HEAD ancestor.
Limits: mounted/declared HTTP-provider-storage seams, remount while held; no later unmarked remount/reload autoplay claim. Native preview ChunkLoadError/Failed to fetch after Save remains unproven (host HTTP positives do not qualify native). No packaged MV3, external provider, physical keyboard/touch/200%/VoiceOver, participant UAT or all-ten acceptance. Program TASK13458 remains In Progress; native Save/Regenerate/D7/scoped settings/roleplay/import work continues.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Saved source replies now stay passive during detached Edit & resend. Admitted attempt identity preserves replacement/source fences, global recovery locks and automatic actions on actual new replies. Reviewed and verified as a complete bounded unit; all-ten program and native/browser/packaged acceptance remain open.
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
