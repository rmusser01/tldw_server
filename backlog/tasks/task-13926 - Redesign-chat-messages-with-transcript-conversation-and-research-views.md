---
id: TASK-13926
title: Redesign chat messages with transcript conversation and research views
status: Done
labels:
- chat
- webui
- design
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reimagine the /chat message area as a readable transcript by default, with explicit Conversation and Research presentation choices. User requested Conversation for roleplay and Research for source-heavy workflows, while preserving existing message behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Transcript is the default; Conversation and Research are explicit saved presentation preferences.
- [x] #2 Each view has a clear speaker/content/action hierarchy and works with character identity, attachments, citations, tools, reasoning and generated content.
- [x] #3 Switching views preserves drafts, message identity, edits, history ownership and virtualized scrolling.
- [x] #4 All existing action capabilities and privacy restrictions remain enforced; keyboard and mobile access are retained.
- [x] #5 Verify representative populated light/dark desktop/mobile states and focused behavior, typing, lint and security applicability checks; document limits.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Separate message-area branch codex/chat-message-views begins at the header-only dev reconciliation commit 2d6e2618ca (PR #3311). Composer PR #3309 and rails PR #3310 already exist and are preserved. User approved readable transcript plus Conversation and Research as selectable alternatives, with Transcript default. Read-only search found no duplicate active task. Highest task id across origin/dev and open PR files was 13924; local header task is 13925. ADR assessment pending source review.
Design and scope: Docs/Design/chat-message-views.md. Three-stage plan: IMPLEMENTATION_PLAN_chat_message_views.md. ADR required: no; this adds a local presentation preference through the existing registry and preserves shared UI conventions, APIs and ADR-049 history ownership. Approved views share renderers and retain existing action/privacy gates, editing and virtualization identity.
Implemented Transcript default plus Conversation/Research through one saved normalized UI preference, selector in conversation toolbar and Chat appearance settings, shared normal/compare forwarding, existing rendering/action/capability gates retained. Fresh code review found one saved-persona avatar case; fixed with two regression cases and reviewer marked resolved. First focused cohort: 98 pass, 2 old cockpit assertions fail identically on clean header base (31 pass / 2 fail there), documented rather than weakened. Additional recovery/memoization/compare checks: 109 pass; 6 virtualization tests initially blocked by duplicate installed React runtimes, rerun with temporary test-only runtime aliases underway. Installed TypeScript check passes; scoped ESLint comparison has no new findings; Bandit scans no Python in this UI scope. First browser matrix: 7 pass, six mobile axe failures identify an unfocusable existing table scroll region. Added accessible focus/region labeling to TableBlock; final browser matrix pending. One detector run flags only preserved legacy colored stripe and existing streaming bounce; new message views remove the stripe.
All six virtualization window/lifecycle cases pass with temporary runtime alias/dedupe and react-virtual inlining, confirming the prior failures were duplicate local React installations. The temporary config will be removed before commit. Code reviewer reinspection scored the persona-avatar fix resolved. Final UI changes include a focusable, labeled horizontal table scroll region to resolve the browser axe finding.
Final browser matrix: 13/13 pass (three layouts × desktop/mobile × light/dark, plus data/draft/preference/manual source disclosure preservation). Axe passes in all twelve message regions; no viewport overflow; source feedback targets >=44px. Captures and manifest: .impeccable/review/messages/final/. With one React runtime, focused cohort: 418 pass / 3 fail across22 files. All three failures reproduce on unchanged header base: cockpit-shell unapplied-character-route expected status, and two coordinator Temporary chat tests expecting the now-menu action as a directly visible button. The earlier two cockpit-controls failures and six virtualization failures disappear with correct runtime. No assertions weakened or disabled. TypeScript final pass and final table/e2e ESLint baseline comparison pass. Visual review in progress.
Impeccable finish review disposition fix: approved the three-mode hierarchy and incumbent design fit, requested legacy omitted-view spacing preservation and alignment of wide table grid/frame. Fixed both: pb-4 scoped to explicit views, and Markdown table override now supplies the semantic table element TableBlock already expects/styles. Browser regression failed first for missing native table; final matrix includes native table and equal desktop grid/frame width assertions. Additional Markdown/table/user-message regression cohort: 32/32 pass. Same-viewports recapture and reviewer verdict pending.
Final reviewer disposition ship for the bounded verdict: both requested fixes scored resolved, all16 recaptures valid. Documenter confirms existing Grounded Research Workbench system fit and accurate surface brief; no global product/design files changed. Pre-existing document-format drift (headings and sidecar completeness) recorded and left untouched. All comparison sheets regenerated from final captures. Final13 browser tests pass including native table/frame equality, axe, mobile overflow and retained preference/drafts/source disclosure. Additional32 Markdown/table/user regressions pass; installed TypeScript and scoped no-new ESLint checks pass. Original Next configuration restored byte-for-byte; temporary test aliases removed. No live provider testing. Three unchanged-base focused test failures remain documented; no tests disabled. Completed task-specific implementation plan removed as instructed. No message-area PR created; header PR3311, composer3309 and rails3310 preserved.
User approved the completed message design and requested a PR against dev. Preparing codex/chat-message-views-pr-dev directly from origin/dev, replaying only message commit0ed5b64f08 so pending header PR3311 is excluded. Original codex/chat-message-views branch preserved. Resolve locale-tail and fixture-tail overlap by keeping only message-view entries/tests; no header redesign changes enter this PR. Re-run focused verification on the dev-based result before publishing.
Dev-only reconciliation review reports no material findings and confirms header-specific locales/tests excluded, with dev theme/timeline controls retained. Fresh dev-based focused cohort421/421 across22 files passes; installed TypeScript and no-new scoped ESLint pass; Bandit UI applicability has no Python/findings/errors. First parallel browser run12/13 passes; the remaining failure is transient null boundingBox during view-switch layout. Test now waits for the select popup to close and polls an atomic DOM measurement of native table/frame width; product code unchanged. Full browser rerun pending. Preserve the pre-existing duplicate locale entry rather than changing unrelated JSON content.
Final dev-only browser rerun13/13 passes, including all layouts/themes/viewports, axe, source controls, native table/frame alignment and retained preferences/drafts/data. Scoped E2E ESLint0 errors/0 warnings; locale mirror3/3 passes. The dev-only421-test cohort is fully green, so earlier three failures on the header-based prototype are not active failures for this PR. All temporary runtime/Next configuration restored/removed; no header/composer/rail redesign files included. User-authorized draft PR publication in progress; repository requires requester-authored Change summary before merge.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Approved /chat message redesign prepared as an independent dev-based PR: Transcript default, saved Conversation/Research alternatives, existing renderers and ownership retained, semantic keyboard-accessible tables. Reconciliation review found no material issues. Fresh dev-based focused tests421/421, browser13/13, locale mirror3/3, scoped ESLint no-new findings and Bandit UI applicability pass; TypeScript initial dev-based check passes. Live providers, full repository suite and extension browser runtime were not exercised. Header3311, composer3309 and rails3310 remain separate.
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
