---
id: TASK-13114
title: Implement Chat Macros v1.1 authoring and output profiles
status: In Progress
assignee: []
created_date: '2026-08-24 04:15'
updated_date: '2026-09-26 03:48'
labels:
  - chat-macros
  - frontend
  - backend
dependencies:
  - TASK-12126
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2951'
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

Final review closeout: the independent whole-branch review identified four blocking authoring/persistence defects. Commit 58eaf7ccbfa fixed source-mode canonical-field affordances, stable catalog refresh/selection, import consumption, unknown top-level settings preservation, stale copy errors, and README validation wording. Scoped re-review confirmed three blockers resolved but correctly rejected losing the edited import on tab unmount. Commit cbb3a52657 then kept both panels mounted/hidden, added tabpanel relationships, and changed the regression to require the edited import to survive a full tab round trip. Live browser verification returned the exact edited YAML after Macros -> Output profiles -> Macros.

Final evidence: frontend 97 passed (63 authoring/profile/service, 33 workspace run/cancel/retry, 1 WebUI route); backend Chat_Macros plus Jobs startup 146 passed with 2 existing warnings; Bandit 0 findings/0 errors across 3,564 LOC; git diff --check clean; branch current with origin/dev; desktop and mobile document widths equal viewports. Package TypeScript remains exit 2 on the unchanged 304-line repository baseline, with no diagnostics naming touched Chat Macros files. Final screenshots: /tmp/chat-macros-v1-1-visual-qa/desktop-macros-final.png and mobile-macros-final.png.

Remaining non-blocking follow-ups: add roving Arrow/Home/End behavior to the tabs (tabpanel relationships are now present); strengthen deferred validation-order and delete failure/cancel cases; replace the existing global fixed N control behavior that overlaps lower mobile content in the wider settings shell. None is introduced as a functional blocker by this branch.

Latest-dev closeout: rebased conflict-free onto origin/dev 21aed4cc0d after it advanced 66 commits. Rewritten review-fix commits are 330b940346 (final-review findings) and f45a82165a (preserve drafts across tabs); earlier SHA references in the chronological notes are their pre-rebase identities. Post-rebase verification repeated successfully: 96 UI-package tests plus 1 WebUI route test, 146 backend/Jobs tests with 2 existing warnings, Bandit 0 findings across 3,564 LOC, clean diff check, TypeScript baseline still 304 lines with no Chat Macros diagnostics, and branch 0 behind origin/dev.
2026-09-13: Resumed publication at user request. Rebased all 19 existing commits onto current origin/dev without conflicts. Next: verify rebased backend/frontend and Bandit, inspect final diff, push branch and create PR against dev. Prior verification results remain historical until rerun.
2026-09-13 publication verification: rebased onto origin/dev c70387f496 without conflicts; range-diff retained all patches except the identical WebUI route already upstream. Backend/Jobs 146 passed (2 warnings); frontend 97 passed, then editor/manager 33 passed after final fixes. Bandit zero findings/errors across 3564 LOC. Fixed editor translation callback hook dependency using a ref so translation changes do not reload drafts. Browser QA exposed cramped desktop numeric labels inside the settings shell; stacked execution and branch sections and verified labels fit with no horizontal overflow at 1440x1000 and 390x844. Tab round-trip preserves draft. Screenshots: /tmp/chat-macros-v11-1440-final-20260913.png and /tmp/chat-macros-v11-390-final-20260913.png. Scoped ESLint passes with no source diagnostics (Next plugin root/pages discovery notice). TypeScript with 8GB heap exits 2, 192 diagnostics outside changed macro files; default heap initially exhausted. Browser authenticated save/reload not repeated: no API credential or backend listening on 8000; API/component coverage passed. Full repo E2E/build not run. PR prepared for publication; human-written v1.1 Change summary still needed before eventual merge.
2026-09-13: Published codex/chat-macros-v1-1 and opened PR #2951 against dev: https://github.com/rmusser01/tldw_server/pull/2951 . Verification/fix commit abcb5d44f9. Task remains In Progress while remote CI/review and the human-authored v1.1 Change summary are outstanding.
2026-09-25: User authorized latest-dev rebase, all Qodo review fixes, and merge after checks. Rebased 21 commits conflict-free. Qodo posted 11 findings: stale profile settings overwrite; empty profiles; stale export; dirty profile refresh; blank headings; missing docstrings; mixed API test flows; stale save/delete callbacks; legacy profile names; API tests coupled to filesystem. Plan: regression-first editor/profile fixes in disjoint files, atomic output-profiles backend update and validation tests, regenerate OpenAPI fingerprint/types to fix backend-required drift, run focused suites/security/static checks, push with lease, respond to review threads, verify CI/review and merge when human v1.1 Change summary is supplied.
2026-09-25 review fixes: rebased all 21 commits cleanly onto origin/dev a2f5e1b816cfe189db7f553a1ccf8d481dc2edbe. Addressed all 11 Qodo findings: profiles-only atomic settings mutation with concurrent-writer/rollback/user-isolation coverage; min-one-section and trimmed nonblank heading validation; legacy profile name editing; dirty profile preservation; current-draft YAML export with exact untouched source; stale async macro mutation/import/clone guards; test docstrings and independent API-observable identity checks. Regenerated OpenAPI types/fingerprint; drift check passes. Verification so far: 38 focused backend +19 repository +6 Jobs startup tests; 95 frontend tests; package-wide TypeScript, scoped Ruff and ESLint pass; Bandit zero findings/errors across 3632 backend lines. Full macro suite and independent review pending. PR2951 remains blocked for merge by missing human-written v1.1 Change summary; requested from user.
Independent review found two additional issues, now fixed with regression-first coverage: previously valid empty stored profiles use read-only normalization (strict new-input validation retained), and profile editor remains mounted through refresh/failure/retry. Backend full macro suite: 158 passed; Jobs startup: 6 passed; final service suite: 26 passed including two added compatibility tests. Frontend: 95 passed plus two added refresh regressions (41 editor/profile tests passed after fix). TypeScript package-wide now passes. Final Bandit: zero findings/errors, 3643 lines. Independent reviewer found no remaining backend issues. Awaiting PR CI/Qodo rerun and human v1.1 Change summary.
2026-09-25 follow-up Qodo reviewer guide (comment5835803913) flags multiline/control-character headings and empty-string legacy compatibility. Empty strings were rejected before the review-fix commit, so that compatibility claim is not a previously-valid-data regression. Tightening new heading inputs to single-line/control-free text, repairing previously accepted stored multiline/control headings on read, and checking downstream Markdown safety. CI remains queued; human summary still pending.
Heading follow-up verified: 16 new validation/legacy-read regressions failed before fix. After fix, service+executor 83 passed, both settings API validation paths 6 passed, chat-rich-text sanitization 6 passed. Ruff/diff checks clean; touched-scope Bandit zero findings/errors across210 lines. Standard ReactMarkdown path has no raw-HTML plugin; ST compatibility path calls DOMPurify via sanitizeChatRichHtml. Empty-string title compatibility claim is not applicable: pre-fix23c6b756 output_profiles.py already rejects not title. New /agentic_review command is preferred by Qodo over deprecated /review.
2026-09-26 CI blocker investigation: backend-required run36162910088 failed only OpenAPI contract drift. Checked-in812c35ad has2098 paths/3211 schemas; CI7a9fc914 has2098 paths/3210 schemas. CI installs Pydantic2.13.5/pydantic-core2.46.5/pydantic-settings2.15.0/Starlette1.7.0; local shared venv uses Pydantic2.11.7/Starlette1.2.1. Regenerating in isolated temp dependency overlay matching CI and inspecting schema differences before snapshot update. Human Change summary received and published verbatim; merge gate satisfied on that requirement.
Root cause confirmed: isolated CI dependency overlay reproduces CI fingerprint7a9fc91443c4cfca4e929fafb9c54cc5daab78d00cea3b1085011a39bc60e83e exactly. Schema diff has no changed paths or Chat Macros schemas; Pydantic2.13.5 combines equivalent OscePatientContext-Input/-Output into OscePatientContext, updating references in three OSCE models. Updating only generated fingerprint (types regenerated locally, gitignored). No backend behavior change or shared-venv modification.
Fingerprint correction verification: regenerated OpenAPI JSON+TypeScript using CI-matched dependency overlay; a separate fresh exporter --check passes with expected7a9fc914 fingerprint. Package-wide bun run typecheck exits0; git diff --check passes. Only tracked changes are fingerprint and task record, no application code; prior security scan remains applicable.
2026-09-26 rebased onto dev3f909e133b (ADR inventory documentation update), cleanly. git range-diff shows all24 PR commits patch-equivalent; application/frontend/helper trees are identical to prior da6c9dfa9e. New AGENTS ADR assessment requirement: ADR required:no new ADR; governed by Docs/ADR/003-jobs-vs-scheduler-default.md. This v1.1 authoring/profile UI and validation follow-up retains v1 per-user YAML storage, database records and Jobs ownership; no durable architecture decision changes. Recording same assessment in implementation plan and PR. Rechecking CI-matched OpenAPI contract before publication.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

2026-09-26 follow-up rebase: dev advanced to 59bd584503 (PR2996 MCP sanitizer changes, no Chat Macros overlap). Rebased all25 PR commits cleanly; git range-diff confirms every commit patch-equivalent. Fresh CI-matched OpenAPI contract check passes using /tmp/pr2951-openapi-ci-deps. No new application edits or security findings introduced by this rebase. Publishing with an exact lease against2445e67792; required CI and any new review feedback remain merge gates.
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
