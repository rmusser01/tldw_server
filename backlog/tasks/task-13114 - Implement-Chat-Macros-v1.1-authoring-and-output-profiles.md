---
id: TASK-13114
title: Implement Chat Macros v1.1 authoring and output profiles
status: Done
assignee: []
created_date: '2026-08-24 04:15'
updated_date: '2026-08-24 08:13'
labels:
  - chat-macros
  - frontend
  - backend
dependencies:
  - TASK-12126
documentation:
  - Docs/superpowers/specs/2026-07-03-chat-macros-design.md
  - Docs/superpowers/plans/IMPLEMENTATION_PLAN_chat_macros_v1_1_authoring.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the first Chat Macros expansion slice: a complete user-facing authoring workflow for custom macros and a richer configurable output-profile editor, building on the merged v1 execution and persistence architecture.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Users can create, edit, delete, import, and export custom macro definitions through the settings UI without manually editing server files.
- [x] #2 The authoring workflow validates definitions before persistence and presents actionable schema, command-collision, permission, and execution-cap errors.
- [x] #3 Users can configure single-response and structured-section output profiles, including ordering, headings, branch-result inclusion, and synthesis behavior supported by the runtime.
- [x] #4 Existing enable, disable, clone, run, cancel, retry, and built-in /wrapup behavior remains compatible.
- [x] #5 Backend and frontend tests cover successful authoring, validation failures, ownership boundaries, output-profile round trips, and destructive-action confirmation.
- [x] #6 Documentation describes the authoring workflow, stored definition format, compatibility constraints, and operational/security considerations.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Detailed TDD plan: Docs/superpowers/plans/IMPLEMENTATION_PLAN_chat_macros_v1_1_authoring.md. Five stages cover backend identity/heading contracts, typed YAML helpers, macro editor UI, output-profile editor UI, and manager integration with security and visual verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-08-23 baseline: backend Chat_Macros suite passed 134 tests with 2 warnings. ChatMacrosSettings frontend component suite passed 4 tests. The frontend service suite could not collect in this isolated worktree because wxt/browser was unresolved across the monorepo dependency roots; stop-after-three-attempts rule applied and the plan requires a complete workspace dependency layout before Task 2.

Current tracked files: Docs/superpowers/plans/IMPLEMENTATION_PLAN_chat_macros_v1_1_authoring.md; backlog task metadata. Isolated branch: codex/chat-macros-v1-1 from merge commit 5c268daa7a.

2026-08-24 implementation complete on codex/chat-macros-v1-1 after rebase onto current origin/dev. Added backend resource/YAML identity enforcement and bounded output-profile section titles; typed frontend contracts and YAML helpers; guided/raw authoring with validate-before-save, import/export, clone, immutable built-ins, and accessible delete confirmation; structured/single-response output-profile editing; manager integration; Next.js settings route; docs and locale updates.

Verification: backend Chat_Macros plus Jobs startup suite 145 passed (2 warnings); frontend focused suites 94 passed (60 settings/service/helper, 33 workspace run surfaces, 1 route); Bandit 0 findings and 0 errors across 3,562 backend LOC; desktop QA 1440x1000 and mobile QA 390x844 with no horizontal overflow; live create/save/reload, import-as-draft, output-profile persistence, clone, built-in immutability, and cancel-focused delete confirmation exercised. Package-wide UI TypeScript remains nonzero on 304 lines of existing diagnostics outside the touched Chat Macros files; scoped diagnostic search returned none. Screenshots: /tmp/chat-macros-v1-1-visual-qa/desktop-macros.png, mobile-macros.png, mobile-output-profiles.png.

Known deferred minor improvements from scoped reviews: complete WAI-ARIA tab keyboard/tabpanel behavior; clarify README wording around standalone validation versus create/update identity checks; strengthen create ordering, delete-cancel/failure, stale-copy-error, and semantic save-label tests. None blocks the implemented contracts or verified workflows.
2026-08-24 final-review fix round started at f7253b9b26. Scope is limited to the six verified findings: source-mode canonical-field affordances, dirty-draft/selection preservation across catalog refresh and toggles, one-shot import consumption, unknown settings preservation, stale copy-error clearing, and README validation wording. ARIA tabs remain deferred.
Final-review TDD evidence: RED frontend run failed 4 targeted cases (source-mode controls, import replay, selection change/dirty draft loss, stale copy error) with 29 unrelated cases passing; RED backend run failed the unit and API unknown-key cases with 22 unrelated cases passing. GREEN: 63/63 focused frontend authoring/profile/service tests and 140/140 full Chat_Macros backend tests passed. Bandit scanned 44 LOC in settings.py with 0 findings and 0 errors. git diff --check passed. ARIA tabs remains deferred as requested.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Final review closeout: the independent whole-branch review identified four blocking authoring/persistence defects. Commit 58eaf7ccbfa fixed source-mode canonical-field affordances, stable catalog refresh/selection, import consumption, unknown top-level settings preservation, stale copy errors, and README validation wording. Scoped re-review confirmed three blockers resolved but correctly rejected losing the edited import on tab unmount. Commit cbb3a52657 then kept both panels mounted/hidden, added tabpanel relationships, and changed the regression to require the edited import to survive a full tab round trip. Live browser verification returned the exact edited YAML after Macros -> Output profiles -> Macros.

Final evidence: frontend 97 passed (63 authoring/profile/service, 33 workspace run/cancel/retry, 1 WebUI route); backend Chat_Macros plus Jobs startup 146 passed with 2 existing warnings; Bandit 0 findings/0 errors across 3,564 LOC; git diff --check clean; branch current with origin/dev; desktop and mobile document widths equal viewports. Package TypeScript remains exit 2 on the unchanged 304-line repository baseline, with no diagnostics naming touched Chat Macros files. Final screenshots: /tmp/chat-macros-v1-1-visual-qa/desktop-macros-final.png and mobile-macros-final.png.

Remaining non-blocking follow-ups: add roving Arrow/Home/End behavior to the tabs (tabpanel relationships are now present); strengthen deferred validation-order and delete failure/cancel cases; replace the existing global fixed N control behavior that overlaps lower mobile content in the wider settings shell. None is introduced as a functional blocker by this branch.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Delivered Chat Macros v1.1 authoring and configurable output profiles on codex/chat-macros-v1-1. Users can create, validate, edit, import, export, clone, enable/disable, and delete user macros while built-ins remain immutable; advanced YAML stays canonical, dirty drafts survive refreshes and tab changes, and imports remain one-shot without losing edits. Output profiles support structured or single responses, ordered sections, custom headings, and branch-output inclusion while preserving unknown future settings. Backend identity, ownership, validation, path, and size protections remain authoritative.

Independent whole-branch review and scoped re-review findings were resolved. Final verification: 97 frontend tests, 146 backend/Jobs tests, Bandit with zero findings across 3,564 LOC, clean diff checks, responsive live browser QA, and no Chat Macros diagnostics within the known package-wide TypeScript baseline.
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
