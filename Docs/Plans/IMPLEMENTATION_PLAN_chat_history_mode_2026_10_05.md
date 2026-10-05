# Chat history mode wording — TASK-13454.2

**Baseline:** HEAD `85bf26502a`; latest `origin/dev` `c2260945ee`, integrated in `10d6abff30`.
**Approved scope:** all ten finding groups in five stages, equal novice/power/roleplay weight. This bounded correction removes false saving claims and false navigation warnings; it does not close Stage 1 AC1.
**Architecture:** reuse `usePersistenceMode` for both composers and mode-change notifications. Mode describes whether future messages are eligible for saved history. Connection readiness, a server ID, and a submission result do not prove a latest write. Add no status store/index/API. ADR-049/050 still govern ownership; no new ADR is required for this semantic correction.
**Accepted D7 sequencing:** native-owner create/admission/settlement prerequisites precede P9's confirmed-write status. P9 needs qualified complete recovery reads, exact displayed content versus owned capture, and an owned preparation fence. Existing H1 empty recovery arrays/completed or dismissed markers are not last-write evidence. Temporary Q3 retention on every surface remains a later P7 delivery: do not promise transcript survival or deletion on window close in this unit. Genuine write/promotion failures keep their existing notices.
**Tracking:** official backlog-py CLI in this isolated worktree; MCP workflow resources read but task search did not respond. Completed submit-recovery plan is archived in `85bf26502a` and removed here. No changes to the primary checkout.

## Stage 1: Reproduce misleading mode and navigation feedback
**Goal**: meaningful regressions using the actual shared hook, translated catalogs, actual toolbar and compact mode/warning callbacks.
**Success Criteria**: connected fresh and bare-ID states fail a mode-only expectation; server-to-New/local transitions reproduce the false warning; Temporary labels expose no unshipped lifecycle promises.
**Tests**: English-catalog and missing-key fallback hook rendering; actual toolbar Casual/Pro/mobile labels; actual compact mode/warning AST harness; existing accepted/cancelled mode-toggle tests.
**Status**: Complete

## Stage 2: Share truthful mode wording
**Goal**: History on / Temporary and clear mode help on both composers and accepted mode-change notices.
**Success Criteria**: no confirmed-save claim derived from connection/ID; no warning from ID disappearance; unused SaveStatusIcon removed with only its obsolete assertion; genuine failures remain; English asset/public catalogs agree.
**Tests**: Stage 1 regressions plus existing promotion/toggle/toolbar/badge suites.
**Status**: Complete

## Stage 3: Verify and review
**Goal**: independently reviewed, tested incremental commit.
**Success Criteria**: affected suites/typecheck/no added lint/diff/task normalization pass; both authorized source reviewers clear; plan and task record limitations; only own completed prior plan removed; local dependency symlinks excluded.
**Tests**: bounded affected Vitest suites, WebUI typecheck, baseline-relative touched ESLint, locale parity, diff and Backlog normalization. Bandit N/A for TS/docs-only scope.
**Status**: Complete

Red evidence: /private/tmp/chat-history-mode-red.log — 14 meaningful failures and 34 existing passes across two suites. Both ID-disappearance cases called the false warning once; actual English labels were Saved Locally+Server / Saved locally and toolbar Saved. No application runtime walkthrough claimed.

Independent review reproduced an accessibility polarity gap: a Temporary-named switch was off during Temporary and the History-on button was unpressed. Added actual compact AntD Switch JSX and mounted Casual/Pro/mobile keyboard tests; nine assertions failed before the correction (/private/tmp/chat-history-mode-a11y-red.log). Both controls now use the stable semantic prefix Chat history plus their visible state, with checked/pressed = !temporaryChat; this preserves visible label in the accessible name.

Final verification: 6 affected suites / 91 tests pass (/private/tmp/chat-history-mode-final-suites.log); WebUI typecheck exit0 (/private/tmp/chat-history-mode-types.log); baseline-relative touched ESLint adds no diagnostics (/private/tmp/chat-history-mode-lint.log); all five English asset/public keys match; git diff --check passes. Both authorized independent source reviewers report no remaining Critical/Important finding after the polarity correction. Bandit N/A TypeScript/docs-only. No live browser or screen-reader acceptance claimed. Full confirmed latest-write status, Stage1 AC1 and accepted D7 Temporary retention stay open.
