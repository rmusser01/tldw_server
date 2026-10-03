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

Latest base: dev29d8dbcc (PR2819 web-scraping integration). Conflict-free rebase preserves all thirteen prior commits exactly, including the docs synchronization correction. Affected configuration/extraction suites collected 37 tests and completed exit 0 with normal logging. The initial operator LOGURU_LEVEL=ERROR suppressed an asserted INFO logging contract; no source/test change was needed. Prior 41 Jobs/Chatbook checks and 39 docs checks qualify their unchanged selected source; no PostgreSQL execution is claimed. Existing repair review/Bandit remains unchanged; current-head hosted gates are pending.

CI docs correction: the full-suite gap-verified-9 job failed two refresh checks because the canonical Chatbook guide changed without its Published copy. The supported refresh adds only that missing six-line paragraph. Both failures reproduced before regeneration; all 39 refresh/adjacent Chatbook docs checks then passed (4 warnings, 19.76s, exit 0). No code or test changes; Bandit N/A for prose. Current-head hosted checks and the existing human/companion gates remain open.
