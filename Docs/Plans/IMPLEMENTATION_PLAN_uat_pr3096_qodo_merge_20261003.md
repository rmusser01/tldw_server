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

Server Qodo corrections are independently reviewed and published with affected regressions passing. The human approved companion publication and browser recovery. The reviewed consumer fix is now draft [Chatbook PR2978](https://github.com/rmusser01/tldw_chatbook/pull/2978); its production/tests are unchanged from reviewed commit 2094809343 and current main f3aeb32fb3. Browser recovery is authorized but unexecuted; live UAT stays held until repair PR completion and corrected-runtime readiness. The requester supplied the PR3096 Change summary directly; it is published verbatim and read back exactly. That gate is satisfied. Final-head checks and normal exact-head merge remain open.

Latest base: dev ea1eda99 (PR3098). Conflict-free rebase preserves all 17 prior commits and the complete owned patch exactly; only 36 upstream paths differ, with no direct repair-path overlap. Independent actual-source review is clear. Quotas remain off by default, with explicit switch precedence, the billing-repository gate, counters and separate request-rate controls preserved. The root fixture enables existing quota tests; default-off tests clear both switches. Affected Usage/Billing/audio/RAG checks pass 100 cases (4 warnings, 1.25s, exit 0, no skips). All 26 changed Python files compile; Bandit on 13 incoming production files has 0 findings/0 errors. Ruff reports 124 inherited diagnostics in source identical to current dev; the undefined FastAPI and duplicate asyncio diagnostics also occur on previous dev. Operator Ruff module/cache setup failures required only existing-tool/no-cache correction. No PostgreSQL, native, browser or live-UAT qualification. Existing own-source checks remain unchanged. ADR required: no; existing architecture decisions are preserved.

CI docs correction: the full-suite gap-verified-9 job failed two refresh checks because the canonical Chatbook guide changed without its Published copy. The supported refresh adds only that missing six-line paragraph. Both failures reproduced before regeneration; all 39 refresh/adjacent Chatbook docs checks then passed (4 warnings, 19.76s, exit 0). No code or test changes; Bandit N/A for prose. Fresh current-head hosted checks and normal merge remain open; the human summary and approved companion publication are complete.

CI ingestion assertion correction (TASK13260.281.1): three gap-verified-4 failures reproduce before edits. Four downstream assertions trim the final LF from Markdown fixtures despite the reviewed exact-text contract. Require the complete source strings in local-directory create/change, detached-note reattachment and archive-snapshot sync; retain all existing identity/status/conflict checks. The full affected suites plus plaintext conversion pass 22 cases (4 warnings, 20.85s, exit 0, no skips). Syntax and Ruff checks pass; the two format flags and 104 LOW Bandit B101 notices match HEAD exactly, with no new findings or scan errors. Independent assertion review is clear. An initial aggregate invocation stopped before collection due to plugin autoload; only operator setup changed. No production or live-UAT change. Supported Backlog clearing removes the duplicate active summary section; orphan END comments remain a CLI formatting limit, with task criteria/status/notes unchanged.

CI hosted email-header correction: media-ingestion-new-integration job111225338658 fails the org-scoped upload assertion because its test lacks the hosted billing-repository fixture required by current dev. The unchanged case reproduces exit1. Request the existing billing_repo_wired fixture for that case only; retain every test body, auth/storage/search assertion and offline tripwire. The full affected upload file plus billing-gate contracts pass18cases (10warnings,26.88s,natural0,no skips). Syntax/Ruff lint/format pass; touched-test Bandit has39unchanged findings/0errors. Independent actual-source review is clear. Repository presence is simulated for this regression; no deployed hosted repository, production change or live-UAT acceptance. ADR required:no. Fresh final-head hosted checks remain required.
