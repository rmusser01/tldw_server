---
id: TASK-13529
title: Keep shared chat textarea sizing current across layout changes
status: Done
labels:
- chat
- ux
- extension
- responsive
dependencies:
- TASK-13457
- TASK-13528
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the shared native sizing root found during the approved all-ten chat UX walkthrough: narrowing a mounted composer without changing text leaves wrapped lines clipped behind hidden overflow. Trace all four existing hook callers, reuse its height/cap calculation and coalesced native measurement, and cover current-node lifetime so layout or lazy remounts cannot leave stale measurements or writes. Preserve each caller display policy and scrolling caps; keep parent stage4 In Progress.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Failure-first actual shared hook evidence reproduces unchanged-text narrow/wide reflow and distinguishes capped scrolling, coalesced measurement and height-only observer feedback.
- [x] #2 All four existing callers route width-driven sizing through the same shared calculation; WebUI collapsed-display sizing, Sidepanel raw text and existing edit/model caps are preserved without dependency or observer registry.
- [x] #3 Sizing follows the mounted textarea and ignores cancelled or retired work; delayed attachment or replacement cannot write to a departed node or remain unsized within the supported caller lifecycle.
- [x] #4 Actual worktree browser verification narrows and widens a typed composer without further input, preserves its draft and demonstrates unclipped content or reachable capped scrolling; proof limitations are recorded.
- [x] #5 Affected behavior tests, both client types, no-added lint, authored formatting, canonical task/diff checks and two independent reviews pass; Bandit assessment is recorded.
- [x] #6 Master design and task retain ADR assessment, final evidence and open program gates, and the bounded repair is locally committed with parent In Progress.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Fresh origin/dev26ae4fd unchanged and ancestor of14cfa07039. ID13529 exceeds remote13515, all50 open PR13526 and local13528; textarea duplicate search found related completed layout work but no shared width-measurement repair. All4 consumers use useDynamicTextareaSize: WebUI adapter, Sidepanel Form, ModelPlaygroundChat and EditMessageForm. Shared apply function already resets/shrinks/clamps correctly; hook schedules only on text/ref identity/max. Native320 proof:44px clientHeight64px scrollHeight overflow hidden after unchanged draft width shrink. Lazy shell ref detach/attach is a source-derived adjacent lifetime boundary requiring direct coverage before claiming remounts. ADR assessment: existing ADR049/050 authority/lifetime and ADR059 tracking apply; no new durable architecture decision is proposed. Native browser uses current worktree frontend with older existing backend; full latest-dev backend/save/load/roleplay, packaged extension and AT gates remain open.
Plan Docs/Design/IMPLEMENTATION_PLAN_chat_textarea_layout_13529.md: reproduce shared hook width/lazy-node behavior, implement stable returned callback with React node state and effect-local native resources, verify current frontend reflow and checkpoint. Both independent read-only reviewers clear this minimal approach. Effect dependencies include node/external object/text/cap; width comparison retains fractional changes; original object stays available to focus/caret/mentions. No operation registry or MutationObserver is needed.
Final source: one stable native callback in existing shared hook, original ref synchronized plus React node state; node/ref/text/cap layout effect owns exact-width RO and coalesced guarded RAF, teardown/fallback. All4 production callers attach without changing display policies/caps; real WebUI ref forwarding and actual lazyV1/V3/V5 are covered. Exact14cfa hook red11/15controls; component red1/2; adapter red1/7excluded. Currentfocused26/26 and affected549/549 in26suites; both client types pass; no-added ESLint10 authored TS/TSX and8 owned format checks pass. Both independent reviews clear source/evidence. Bandit N/A TS/TSX/docs. Native unchanged draft1280→320→390 sizes44→64→40 with matching content;320no document overflow.208content/120cap scrollable, keyboard scrollTop88 and typing retains position. Screenshot /private/tmp/chat-textarea-layout-13529-320.jpg. Controlled text cleared, viewport reset, no Chrome windows opened. Current frontend/older existing backend only; native packaged extension/font-only/touch/zoom/themes/VoiceOver and full latest-dev backend journeys remain open. ADR assessment and retired plan/evidence retained in master; parent13457 In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed width clipping and missed lazy/remounted textareas once in the existing shared sizing hook. All4 callers retain their focus refs, display policies and height caps. Passed549affected checks, both client types, no-added diagnostics and two independent reviews; native narrow/wide/capped keyboard checks verified. Local checkpoint includes master/task evidence; parent all-ten program remains In Progress.
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
