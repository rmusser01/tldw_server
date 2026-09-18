# PR2967 integration and UAT continuation

Tracking: TASK13260.219. User requested merge of the accumulated repair PR into `dev`, then UAT resumption starting with TestBot261. This changes the execution order, not the exact-output criterion or the full SQLite/PostgreSQL single-user/multi-user workflow scope.261 remains open.

## Stage 1: establish the merge candidate and failing checks
**Goal:** Include the reviewed local repairs and identify actual CI/review blockers.
**Success Criteria:** Current origin/dev and PR revisions recorded; failed job logs retained; independent integration review started; findings separated by cause.
**Tests:** Inspect current GitHub Actions job conclusions/logs, source diff and retained regression evidence. No skipped job is a test pass.
**Status:** Complete.

Initial base59049e094e0845a4611ea725ae19b7c1754ea709 is an ancestor of local91be80160ec9a8962daba5a4b5fcbd0688dfcd03. DraftPR2967 currently points to3e57d8887ed08e911808d56648731cec71953e27. Eighteen failed job conclusions include aggregate jobs; determine distinct causes before assigning repairs. Current local branch includes native276277 acceptance.

User requested removing excessive generated Playwright evidence from the PR. The 11,712 branch-added files under `output/playwright/` account for 3,254,093 added lines. Preserve every local file, remove only Git index entries, and use the existing `/output` ignore rule. Keep the tracker, matrix and concise verification results; historical evidence references become local archive references. No branch history rewrite.

## Stage 2: repair and verify merge blockers
**Goal:** Fix confirmed failures with causal evidence and minimal changes.
**Success Criteria:** Required CI passes on the reviewed candidate; substantive review findings resolved; associated tasks and running tracker current; no bypass of hooks, tests or branch protection.
**Tests:** Reproduce failing tests, retain baseline comparisons where needed, run relevant package/backend suites and actual official-fixture PostgreSQL controls for database changes. Run scoped lint/security checks and final candidate CI.
**Status:** In Progress.

Generated evidence removal is committed and pushed at `6bc0b1c13368d241461c35257a2a322b67f89b12`. Confirmed integration defects are World Book PostgreSQL ownership (278), rejected Chat Retry connection cleanup (279), and World Book secondary operations/LIKE placeholder conversion (280). Retry has independent review and 110 passing focused tests; World Book verification is expanding to RAG and conversation consumers found during review. CI repairs include realistic frontend fixtures, stable OpenAPI dependency/schema generation, ingestion opt-out, backend shard assignment, onboarding and extension persistence fixtures. Final candidate CI and the human-authored Change summary remain required.

World Book ownership/consumer/transaction repairs now have independent review and official PostgreSQL/SQLite verification. Notes remediation passes32 tests, published docs pass52 tests plus three macOS controls, and the final World Book attachment fixture passes10 tests. Additional frontend readiness changes preserve all original assertions; exact CI Node20.20.2 local verification passes46 cases and times out in four under unrelated worktree CPU saturation. At pushed1c4ff92226, hosted CI confirms all repaired areas except two Settings assertions and the original extension fixture. The user's approved single trace-enabled extension replay passes with zero skips/retries; source review then moves temporary-chat setup before the first history seed. The final refinement requires hosted CI. Its Axe report also reveals a named generic content container, tracked as UAT281/TASK13260.219.9 for a minimal semantic repair and regression. These are merge gates, not completed UAT acceptance.

## Stage 3: complete PR requirements and merge
**Goal:** Integrate the verified repair branch into dev.
**Success Criteria:** PR description represents final scope and261 limitation; human-requester Change summary required by repository policy is present; PR ready and required checks/review clear; normal merge succeeds and remote merge commit verified.
**Tests:** Fresh PR status/head/base/check review immediately before merge; inspect remote merged state afterward.
**Status:** Not Started.

## Stage 4: resume UAT from261, then the full workflow scope
**Goal:** Continue the original UAT → review → fix loop on the integrated code.
**Success Criteria:** First resumed scenario is original TestBot character flow, all outcomes retained; subsequent fresh four-configuration workflow matrix proceeds with261 still open until genuinely resolved. No green-only retries or acceptance weakening.
**Tests:** Characters → TestBot → Chat, exact public question and instruction, real completion/canonical reload; then the retained twelve-row protocol across SQLite/PostgreSQL and single/multi-user setups, including actual image attachment.
**Status:** Not Started.
