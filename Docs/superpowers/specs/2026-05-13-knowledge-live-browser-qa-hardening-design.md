# Knowledge Live-Browser QA Hardening Design

Date: 2026-05-13
Status: Approved for planning
Owner: Codex brainstorming session
Related PR: https://github.com/rmusser01/tldw_server/pull/1617
Backlog: TASK-297.6

## Summary

Add a verification-first QA hardening stage to PR #1617 for the `/knowledge` QA page. This stage broadens live-browser coverage across viewport sizes and data profiles, records keyboard and power-user friction from actual use, and permits only narrow same-PR fixes for clear `/knowledge` regressions or low-risk usability defects.

Saved-view sharing, profile sharing, and advanced organization remain out of scope for PR #1617. Those ideas should become a follow-up product-expansion issue only after QA produces evidence that the new source picker needs them.

## Goals

- Verify `/knowledge` with live browser evidence across WebUI and extension surfaces.
- Cover desktop, tablet, mobile, and extension-sized viewports.
- Test both repeatable seeded fixture data and one local real-data database pass.
- Identify keyboard, focus, repeat-action, and power-user friction in the new source picker and QA workflow.
- Keep PR #1617 focused by fixing only clear regressions or low-risk affordance gaps found during QA.
- Produce a follow-up product-expansion issue after QA synthesis for saved-view sharing or advanced organization, using QA evidence rather than guesses to define its scope.

## Non-Goals

- Turn `/knowledge` into the canonical knowledge CRUD, import, or management hub.
- Add saved-view sharing, profile export/import, multi-user sharing, or advanced source organization in PR #1617.
- Replace source owner pages or Quick Ingest.
- Add automatic web fallback recommendations.
- Include generated, test, or workspace artifacts in normal source selection by default.
- Perform repo-wide frontend cleanup or unrelated design-system/routing refactors.

## Approved Scope

PR #1617 receives a QA hardening stage, not a product expansion stage.

Allowed in PR #1617:

- QA checklist/spec artifacts.
- Backlog task updates.
- Playwright or Vitest coverage for deterministic `/knowledge` regressions found during browser QA.
- Small UI fixes for broken focus, unreachable controls, viewport overlap, misleading copy, source-picker bugs, or route parity problems.

Not allowed in PR #1617:

- New saved-view sharing/export/import UX.
- New advanced organization model.
- New backend sharing APIs.
- General knowledge CRUD/import hub behavior.
- Large design-system, routing, or data-model refactors unrelated to observed `/knowledge` QA failures.

## Surfaces

The QA matrix should cover only `/knowledge` and flows directly reachable from it:

- WebUI `/knowledge`.
- Extension options `#/knowledge`.
- Extension options `#/knowledge/thread/:threadId`.
- Extension options `#/knowledge/shared/:shareToken`.
- `/knowledge` Add Sources entry into Quick Ingest, only to verify the handoff remains discoverable.
- Answer-to-workspace handoff only to verify the existing action is reachable and does not break route state.

Sidepanel KnowledgePanel remains a chat context search-and-insert surface, not the full `/knowledge` QA workspace. It should be checked only for copy/route parity if touched by current PR changes.

## Stage 1: QA Matrix Definition

Define the exact browser QA matrix before running tests.

Viewport groups:

- Desktop wide: around 1440 x 900.
- Desktop constrained: around 1280 x 720.
- Tablet: around 1024 x 768.
- Mobile: around 390 x 844.
- Extension-sized: route-specific extension viewport already used by the extension E2E harness.

Data profiles:

- Empty or first-run state.
- Seeded realistic library with media, notes, chats, characters, task boards, prompts, world books, and dictionaries.
- Sources with unavailable or empty status.
- Weak or no-result retrieval.
- Workspace-scoped artifacts that should stay hidden globally and appear only with explicit workspace scope.
- Web fallback disabled and enabled, using the server default provider disclosure.
- One local real-data pass against an existing or sanitized user database.

Task groups:

- First arrival and ready state comprehension.
- Add Sources discovery.
- Source category selection.
- Specific source picker filtering by query, status, recent imports, and workspace.
- Bulk select visible, clear visible, and select recent imports.
- Saved profile save/load for local QA scopes.
- Simple/Detailed toggle.
- Run a QA query with local-only sources.
- Run weak or no-result query and recover.
- Inspect citations/source cards.
- Continue in editor handoff.
- Extension route parity for main, thread, and shared routes.

## Stage 2: Seeded Browser QA

Use seeded fixture data for repeatable coverage. Seeded data should exercise the canonical source set without relying on private user content.

Prefer existing test fixtures, route helpers, seed scripts, or lightweight test-only mocks already used by the WebUI/extension suites. If covering every canonical source would require building a broad new fixture framework, keep the PR focused: document the missing fixture coverage, run the closest available seeded coverage, and create a follow-up testing-infrastructure task instead of expanding PR #1617.

Evidence should record:

- Surface: WebUI or extension.
- Route.
- Viewport.
- Data profile.
- Task.
- Result: pass, fail, blocked, or observation.
- Console or network errors.
- Screenshot path when useful.
- Fix decision: current PR, follow-up issue, or no action.

The seeded pass is the preferred source for committed screenshots and automated regression tests because it can be repeated without exposing private content.

## Stage 3: Local Real-Data QA

Run one local real-data pass after seeded QA. This pass is for realism and edge-case discovery, not committed fixture creation.

Safety rules:

- Prefer a copied or sanitized database profile over live mutable user databases.
- If live local databases must be used, avoid destructive flows and do not intentionally create, edit, delete, reindex, export, or share private content during QA.
- Keep local real-data evidence outside git unless it is fully synthetic or redacted.
- Record only the workflow, source type, viewport, and symptom needed to reproduce with seeded data later.

Privacy rules:

- Do not commit screenshots that reveal user titles, prompts, chats, notes, document text, or source excerpts.
- Do not quote private content in task notes, PR comments, or issue bodies.
- Summarize issues by behavior, source type, viewport, and workflow.
- If a real-data issue needs a regression test, reproduce it with synthetic seeded data before committing a test.

The local real-data pass may justify follow-up product work, but it should not by itself expand PR #1617.

## Stage 4: Keyboard And Power-User Friction Pass

Exercise repeated-use workflows explicitly:

- Tab order through search, source menus, filters, bulk actions, saved profiles, settings, answer actions, and source cards.
- Enter behavior for search submission and source filtering.
- Escape behavior for menus/modals.
- Focus return after closing source picker, settings, source viewer, and existing export/share dialogs if present.
- Keyboard access to Simple/Detailed mode.
- Repeat query with modified source scope.
- Save profile, load profile, and update profile-like workflow using existing local profile behavior.
- Bulk source actions after filtering.
- Citation navigation and source card opening.

Current-PR fixes are allowed only when the issue is a clear defect or low-risk affordance gap, such as unreachable controls, lost focus, broken Escape behavior, overlapping UI, misleading copy, or a deterministic source-picker bug.

## Stage 5: Triage And Fix Gate

Every finding should be classified before implementation.

Fix in PR #1617:

- Broken `/knowledge` route or extension route.
- Inaccessible control or broken keyboard path.
- Viewport overlap or unreachable primary action.
- Lost focus after modal/menu close.
- Incorrect or misleading copy introduced by this PR.
- Source picker filtering, selection, profile, or source-status regression.
- Testable low-risk bug with a focused regression test.

Follow-up issue:

- Saved-view or profile sharing.
- Profile export/import.
- Advanced source grouping or organization.
- Cross-page source management.
- New backend sharing APIs.
- Command palette or large keyboard workflow redesign.
- Any feature that needs product decisions beyond the QA-only page boundary.

Document only:

- Existing repo-wide typecheck/design-state limitations outside `/knowledge`.
- Environment-specific browser or extension harness limitations.
- Real-data-only observations that cannot be reproduced safely with synthetic data yet.

Severity gates:

- P0/P1: fix in PR #1617 or explicitly block merge.
- P2: fix in PR #1617 when the fix is small and local; otherwise document with a follow-up issue and rationale.
- P3: document in the QA summary unless the fix is trivial and clearly risk-free.
- Product expansion: do not implement in PR #1617 even if high value; create the product-expansion issue with evidence.

## Stage 6: Product Expansion Issue

Create the product-expansion GitHub issue after QA synthesis, so it cites evidence instead of guesses. If QA finds no meaningful saved-view, sharing, or advanced-organization friction, create a low-priority tracking issue that records the low-severity evidence and keeps implementation deferred rather than inventing feature work.

Issue scope:

- Saved views beyond local single-user profiles.
- Profile sharing/export/import if users need to reuse source scopes across devices, users, or installations.
- Advanced source organization if large libraries are hard to navigate with the current filters.
- Workspace/source grouping improvements only if QA shows the current filters are insufficient.
- Keyboard command palette or saved-view shortcuts only if repeated-task friction is observed.

Issue non-goals:

- Making `/knowledge` a canonical CRUD/import/management hub.
- Replacing source owner pages.
- Automatic web fallback recommendation.
- Showing generated/test/workspace artifacts by default.

The issue acceptance criteria should include:

- Evidence summary from PR #1617 QA.
- Workflows that need saved/share/organization support.
- Decision on whether each change belongs in `/knowledge`, Quick Ingest, owner pages, or shared source-management components.
- Accessibility requirements.
- Privacy and data ownership requirements for any sharing/export behavior.

## Evidence Table Template

The `TBD` values below are template placeholders to be filled during QA execution, not unresolved design requirements.

| Surface | Route | Viewport | Data profile | Task | Result | Evidence | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WebUI | `/knowledge` | 1440 x 900 | Seeded | Source picker filters by status | TBD | TBD | TBD |
| WebUI | `/knowledge` | 390 x 844 | Seeded | Add Sources is reachable | TBD | TBD | TBD |
| Extension | `#/knowledge` | Extension harness | Seeded | Thread/share route parity | TBD | TBD | TBD |
| WebUI | `/knowledge` | 1280 x 720 | Local real data | Repeated source profile workflow | TBD | No private content in committed notes | TBD |

## Verification Expectations

Required for the QA hardening stage:

- Live-browser QA across the defined seeded matrix.
- Local real-data QA pass with privacy-safe notes.
- Focused Playwright or Vitest tests for deterministic bugs fixed in PR #1617.
- `git diff --check`.
- Bandit only if Python production files change.
- PR review-thread and check refresh after pushing.
- PR comment summarizing tested surfaces, fixes, deferrals, and product-expansion issue URL if created.

Exit criteria:

- The evidence table is complete for the planned matrix, or any skipped rows have a concrete blocked reason.
- No unresolved P0/P1 `/knowledge` findings remain.
- P2 findings are either fixed in PR #1617 or linked to a follow-up with rationale.
- Local real-data findings are summarized without private content.
- The product-expansion issue is created after QA synthesis, even if it is only a low-priority tracker recording that no immediate expansion is justified.
- Remote CI and PR review state are refreshed after the final push.

## Open Questions For Execution

- Which seeded fixture mechanism should be used for the browser QA pass if existing fixtures do not already cover every canonical source?
- Which local database should be used for the private real-data pass, and should it be copied/sanitized first?
- Should screenshots from seeded QA be committed, attached to the PR comment, or kept as local evidence only?
- Should the product-expansion issue be created immediately after QA synthesis even if no severe friction is found, or only when evidence shows a clear need?
