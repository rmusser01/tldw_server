# H1 chat history integration with current dev

**Tracking:** TASK-13261.5. **PR:** #2968 (draft). **Pinned source check:** server `91e8bbf84c25d3afbba2bb53ed06280d44c35307`, Chatbook `4d3e2d380e6ebb7a8e465d2d90a7564a82e7f6ef` on 2026-09-23. Recheck before publishing.

## Stage 1: Audit source drift and preserve the accepted H1 contract

**Goal:** Record every material changed chat behavior and merge seam against the accepted H1 parity matrix and H2 interface.
**Success Criteria:** SQLite/PG migration meanings, strict image reads, saved-turn retry, RLS/operation scope, agent-created chat routing, unread attention and inspector are classified with owner and priority. No `tldw-agent` chat-runtime dependency is introduced.
**Tests:** Source line/commit audit, `git diff` of both pinned dev ranges, migration and UI call-site inventory.
**Status:** Complete

## Stage 2: Integrate H1 on server dev

**Goal:** Preserve accepted H1 history-selection and fork ownership while incorporating all newer server changes.
**Success Criteria:** Merge current `origin/dev` into this isolated PR branch without force-pushing or rewriting prior H1 commits. Retain both schema-v68 lineages: migrate H1 history structures at fresh SQLite/PG versions, detect previously initialized H1-v68 databases, and apply all missing dev migrations before marking the new versions. Preserve new strict image, saved-turn reuse, and PostgreSQL owner/RLS behavior.
**Tests:** New SQLite/PG clean init, dev-v68/PG-v72 upgrade, H1-v68 upgrade, repeated init and rollback fixtures; H1 history/ownership tests; strict multi-image and saved-turn retry regressions.
**Status:** Complete

## Stage 3: Qualify and review the integrated H1 PR

**Goal:** Re-establish H1 correctness in both full-page clients and the compact sidepanel on the new backend.
**Success Criteria:** Focused backend/API/frontend tests and browser flows pass; lint/build/type baseline and touched Python Bandit evidence are recorded. Independent review finds no open P1/P2, including migration and owner-disclosure paths.
**Tests:** Relevant H1 SQLite/PG suites, API integration, shared UI and both-shell tests, browser selected-history and copy failure/recovery, formatter/linter/compile/Bandit, review package and fix loop.
**Status:** Complete

## Stage 4: Refresh PR and H2 contract

**Goal:** Publish the reviewed H1 merge commit to the existing draft PR and make H2 Task 1.2 consume the integrated source.
**Success Criteria:** PR #2968 stays draft and reflects the verified branch; description has current evidence and still identifies the human-written merge summary gate. H2 branch/plan is updated to the correct migration baseline, strict-image reopen, saved-turn and PG operation/RLS constraints; its Task 1.2 brief is regenerated before implementation.
**Tests:** Remote PR head/base verification, H2 source diff and design/plan link check, no new P1/P2 in scoped integration review.
**Status:** In Progress
