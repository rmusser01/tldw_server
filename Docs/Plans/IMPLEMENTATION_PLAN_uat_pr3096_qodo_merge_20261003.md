# PR3096 rebase, review, and merge

Owner: TASK13260.281. PR: https://github.com/rmusser01/tldw_server/pull/3096

## Stage 1: Rebase on current dev
**Goal**: Publish the existing repairs on the latest dev without the integration merge.
**Success Criteria**: Rebase preserves the reviewed source; publish with an exact force-with-lease.
**Tests**: Compare pre/post trees and run git diff --check; qualify any actual source changes.
**Status**: Complete

## Stage 2: Address Qodo review
**Goal**: Verify and resolve actionable Qodo findings against the actual PR source.
**Success Criteria**: Findings have fixes or source-supported replies; changed behavior has causal regressions and independent review.
**Tests**: Affected suites, touched lint, and Bandit for changed Python production scope.
**Status**: In Progress

## Stage 3: Verify gates and merge
**Goal**: Merge the reviewed final head normally.
**Success Criteria**: Seven required contexts pass on the final head; human-written Change summary is present; exact-head merge is confirmed on dev.
**Tests**: Current PR review/check readback and merged commit ancestry.
**Status**: In Progress

Live UAT remains held. Existing bug notes and runtime data remain; no evidence bundles or failed-case replay.

Server Qodo source corrections have independent review and affected regressions complete; see the concise resumed-UAT review for results. The related Chatbook compatibility commit is reviewed locally, with companion publication approval pending after automatic approval review rejected its push. Final-head checks/Qodo acceptance and the human-written PR3096 Change summary remain open.

Latest base: dev25f11ffe (PR3077). Conflict-free rebase preserves all sixteen prior commits and the complete owned patch exactly. Independent rebase review is clear: only the fifteen upstream paths differ, with no direct repair-path overlap or changed own source/tests. Documentation/drawer unit checks pass 10 cases across two files; pure smoke classifiers pass 11 cases without browser/page/API or webserver use. Next config syntax and scoped frontend/UI lint exit 0; React-detection/Next-pages configuration warnings remain. The frontend linter initially ignored two UI paths outside its base; explicit UI-base lint then checked both. No Docker, browser, native, PostgreSQL or live-UAT qualification. Bandit is N/A for this integration with no new Python change; previous own-source results remain unchanged. Fresh final-head hosted gates and the human/companion gates remain open. ADR required: no; this rebase preserves existing architecture decisions.

CI docs correction: the full-suite gap-verified-9 job failed two refresh checks because the canonical Chatbook guide changed without its Published copy. The supported refresh adds only that missing six-line paragraph. Both failures reproduced before regeneration; all 39 refresh/adjacent Chatbook docs checks then passed (4 warnings, 19.76s, exit 0). No code or test changes; Bandit N/A for prose. Current-head hosted checks and the existing human/companion gates remain open.

CI ingestion assertion correction (TASK13260.281.1): three gap-verified-4 failures reproduce before edits. Four downstream assertions trim the final LF from Markdown fixtures despite the reviewed exact-text contract. Require the complete source strings in local-directory create/change, detached-note reattachment and archive-snapshot sync; retain all existing identity/status/conflict checks. The full affected suites plus plaintext conversion pass 22 cases (4 warnings, 20.85s, exit 0, no skips). Syntax and Ruff checks pass; the two format flags and 104 LOW Bandit B101 notices match HEAD exactly, with no new findings or scan errors. Independent assertion review is clear. An initial aggregate invocation stopped before collection due to plugin autoload; only operator setup changed. No production or live-UAT change. Supported Backlog clearing removes the duplicate active summary section; orphan END comments remain a CLI formatting limit, with task criteria/status/notes unchanged.
