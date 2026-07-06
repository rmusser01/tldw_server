---
id: TASK-12904
title: Full UAT for Chatbooks Flashcards and Quizzes WebUI pages
status: Done
labels:
- uat
- webui
- chatbooks
- flashcards
- quizzes
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run live-backend UAT for the Chatbooks, Flashcards, and Quizzes WebUI pages, patch root causes of user-facing issues found, and prepare a PR against dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Live backend UAT completed for Chatbooks, Flashcards, and Quizzes without mock backends.
- [x] llama.cpp-compatible server at `127.0.0.1:9099` was verified and used in the UAT environment.
- [x] Reproducible user-facing issues were patched at the root cause.
- [x] Focused regression coverage was added or updated for each patchable issue.
- [x] PR prepared against `dev`.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-06-chatbooks-flashcards-quizzes-uat-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Live UAT setup:
- Backend: FastAPI on `127.0.0.1:8000` using the UAT environment configured with the live llama.cpp-compatible server at `127.0.0.1:9099`.
- Egress: `WORKFLOWS_EGRESS_ALLOWED_PORTS=80,443,9099`.
- WebUI: Next dev server on `http://localhost:8080`, advanced mode with live API URL/key.
- No mock backend used.

Confirmed root causes and fixes:
- Chatbooks import page object used stale dropzone copy; updated it to the rendered `.zip or .chatbook archive` text.
- SQLite FTS search treated punctuation-heavy user text as operators/columns; added SQLite token normalization/quoting for literal user terms while preserving valid operators.
- ResourceGovernor mapped Flashcards and Quizzes traffic into `core.default`; added scoped `health.default`, `flashcards.default`, and `quizzes.default` policies plus route-map coverage.
- Flashcards Manage first-entry state hid primary search/deck controls; kept those controls visible while still hiding advanced filter chrome until useful.
- Flashcards create drawer ignored the active Manage deck filter; it now preselects that deck.
- Flashcards `Test with Quiz` only honored quiz handoff context; it now enables when a review deck is selected.
- Flashcards edit E2E targeted stale dialog text; updated the page object to the current `Edit Card` drawer and hardened the edit trigger.
- PR review follow-up: SQLite FTS normalization now keeps explicit quoted column phrases intact, treats unquoted `field:value` text as a literal to avoid invalid-table-column 500s in Flashcards, and converts mixed negative terms to valid FTS5 `NOT` operands.
- PR review follow-up: Flashcards edit pointerdown now only stops row propagation; click remains the only edit action trigger, preventing double `onEdit()` calls.
- PR review follow-up: CodeRabbit items addressed by extracting Flashcards Manage expert filters into `ManageExpertFilters`, preventing an already-selected Manage deck dropdown from staying open in E2E helpers, and making Flashcards/Quizzes ResourceGovernor policies use `fallback_memory`.
- PR review follow-up: CodeRabbit's FTS hyphen comment was verified against `9368cde2` and was already addressed by the prior negative-token fix and regression coverage.

Verification:
- Live tier-2 UAT passed 32/32 for Chatbooks, Flashcards, and Quizzes against the live backend.
- Screenshot smoke evidence saved outside the repo at `/private/tmp/tldw-study-pages-uat-shots-20260706` for desktop and mobile Chatbooks, Flashcards, and Quiz pages.
- Focused UI regression suite passed 61/61.
- Focused backend regression suite passed 4/4.
- `git diff --check` passed.
- Bandit found no issues in the touched implementation file. A full touched Python scan only reported `B101` pytest asserts in test files; the non-assert scan passed with `-s B101`.
- PR review follow-up focused checks passed: backend FTS/Flashcards/ResourceGovernor regressions 6/6, Flashcards UI Vitest 54/54, `git diff --check`, and Bandit with pytest asserts excluded.
- CodeRabbit follow-up focused checks passed: Flashcards UI Vitest 54/54, backend FTS/Flashcards/ResourceGovernor regressions 6/6, and frontend `bun run typecheck`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed live WebUI UAT for Chatbooks, Flashcards, and Quizzes using the real backend and llama.cpp server, then patched the confirmed user-facing issues in import locators, SQLite FTS search, ResourceGovernor route policy, Flashcards first-entry controls, deck-scoped creation, quiz handoff enablement, and edit-flow test targeting.
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
