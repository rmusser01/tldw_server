---
id: TASK-13279
title: Keep copy and regenerate visible on the last assistant message
status: In Progress
assignee: []
created_date: 2026-09-19 14:47
updated_date: 2026-09-19 14:47
labels:
- chat
- webui
- recognition
- ux-audit
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Copy, Edit, Regenerate and Generation Info measure 0x0 and are absent from the accessibility tree until the message is hovered, then become 32x32. The two most frequent actions in any chat client are hidden while a rarely-used overflow menu is permanent. There are two overflow menus on every message; the always-visible one measures 37x22, below the accessibility target floor. Touch users have no hover.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Copy and Regenerate are persistently visible on the last assistant message.
- [ ] #2 Only one overflow menu renders per message.
- [ ] #3 Every persistent message action meets the minimum target size.
- [ ] #4 Hover-reveal remains acceptable for older turns.
- [ ] #5 Shared Casual user-message controls expose a real touch/keyboard action entry instead of an inert More chip, preserving original Copy/Edit/Delete authority and current busy/edit guards.
- [ ] #6 The existing shared assistant feedback controls also retain at least 44px pointer targets and keyboard access without changing feedback persistence or conversation authority.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Continuing the approved all-ten queue alongside independent TASK-13549 transport fixture qualification. Bounded plan: Docs/Design/IMPLEMENTATION_PLAN_chat_last_reply_actions_13279.md. Shared Message hides the entire Casual action row; both full and compact use it. MessageActionsBar has an inert overflow chip in addition to the actual trigger, and its small breakpoint removes minimum pointer geometry. Reuse isLastMessage to make the latest assistant actions visible, retain real older-row overflow and current busy/edit/source action guards, and preserve 44px control targets. Actual renderer and real Popover controls must demonstrate RED before implementation; no forced flex fixture alone qualifies visibility. ADR required: no; this changes presentation of existing shared controls without new protocol/dependency, with ADR-059 governing task tracking. Native generated CSS geometry and physical keyboard/touch evidence remain separate from DOM class tests.
Source review also identified the exact shared PlaygroundUserMessage sibling: inert More actions chip and hidden-only Copy/Edit/Delete. The user explicitly included all identified enhancements, so the same presentation unit includes that shared renderer and its causal touch/keyboard access controls; no new action framework or mutation authority.
Independent final source review identified the existing shared FeedbackButtons sibling at 20px/24px targets. Include only its button geometry and focused actual-render checks in the same shared accessibility unit; preserve compact icons, wrapping, feedback semantics and existing owner-scoped persistence. No new feedback feature or framework.
2026-10-10 real portal focus qualification: actual shared assistant/user renderer with ConfigProvider tldw-portal-root and later composer controls had 12 causal keyboard focus failures in the 68-case suite. Existing menu handlers now schedule first enabled popup control through the house helper, guard scheduled focus by popup identity/open state, and restore the original trigger on Escape or an action only while focus remains inside the popup. Explicit editor focus is respected. Fresh /private/tmp/chat-menu-focus-local-settlement-reasoning-qualified-green.log: renderer68, full compact206, accumulator7 =281 pass. Independent connected source review of focus is CLEAR; native browser geometry/animation and final type/static checks remain open.
2026-10-10 connected checkpoint: visible latest Copy/Regenerate, working user/assistant overflow,44px primary/feedback/menu targets and preserved processing controls are verified in the earlier connected component suites. Both current WebUI and extension compilers pass after the broader consumer repair. Owned static/format/hooks and current native geometry/touch/keyboard checks remain pending. Keep In Progress and distinguish component utility proof from packaged extension, physical AT and participant acceptance. Existing presentation,049 and059 govern; no new ADR.
Final original-command consumer checkpoint: 20 connected suites / 1436 tests pass after fixture typing and production cleanup; both WebUI and extension compilers pass; all 112 authored TypeScript scopes introduce zero ESLint findings; owned formatter 21 is runtime-AST equivalent; git diff --check passes. Final independent source-only reviewers A and B CLEAR for the production cleanup, excluding their own earlier SDK/fixture authorship. Evidence: /private/tmp/chat-final-consumer-static-repair-connected.log, /private/tmp/chat-final-consumer-{web,extension}-types-qualified.log, /private/tmp/chat-final-consumer-static-qualified.jsonl, /private/tmp/chat-final-consumer-format.json. Latest fetched origin/dev 0c53973f17eede4074f4aa7894d0b42322340614 remains included by 2e47e3aa01c6bfc7cb61dea5f1cf47c15b7c35f0. Normal incremental commit/hooks are next. The current native bundle is stale; packaged/physical AT/uncoached participant and all-ten acceptance remain open. Three scene components contain four remaining unqualified operations (two selection preflights and Drawer hydration/Apply), to migrate next with original qualified receipts and CAS. No task/stage/namespace closure; ADR049/050/059 and approved D7 govern, no new durable rule. Bandit N/A for authored TS/Markdown-only scope.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
