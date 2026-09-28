# PR2979 latest-dev integration — 2026-09-28

Task: TASK-13260.278.18. Resume PR-first repairs; full UAT remains paused.
ADR required: no new ADR for integration. Preserve ADR020 storage and ADR049
Chat ownership and ADR050 native/local fork lifecycle; reassess if a resolution
changes a durable architecture rule.

## Stage 1: Preserve and rebase
**Goal:** Retain b71c04cde8 and replay UAT work onto verified newest dev.
**Success Criteria:** Recovery ref exists; conflicts retain both intended behaviors.
**Tests:** Clean worktree, ancestry, range-diff and overlap review.
**Status:** Complete

Recovery ref `codex/pr2979-pre-dev-38c1455-20260928` retains the pre-rebase
work. All62 commits replayed onto dev38c1455, then the newly landed dev3d102e0d31.
Rebased head23417aebb1 has dev3d102 as its exact merge base. Stashfa97b77e
preserves the first verified repair batch and was applied without conflicts.
The second rebase retains both automatic lease validation and cancellation in
Playground, plus both isolation-ratchet and journey-provider CI contracts.

## Stage 2: Verify overlapping behavior
**Goal:** Check changed Chat, AuthNZ, Notes, providers and CI boundaries.
**Success Criteria:** Focused controls pass, including required official PostgreSQL.
**Tests:** Existing relevant suites, frontend types/lint, touched Python Bandit.
**Status:** Complete

Initial 12-file Chat integration run: 312 passed, 10 failed. TASK13260.278.18.1
tracks account-stamped local history admission and ordinary Character Retry
receipt preservation (UAT473). TASK13260.278.18.2 tracks the real PostgreSQL
Jobs diagnostic parameter and two causal test-oracle repairs (UAT474). Three
existing Jobs failures reproduce on the official PostgreSQL fixture, zero skips.
UAT474 now passes79 real-PG and30 companion checks after also correcting index
sort/null catalog validation. UAT473 Character/source-auth controls pass111;
shared account/revision review fixes continue. Historical migration modules64
pass, deployment84 pass, backend overlap182 pass (including required PostgreSQL).
After the second rebase 25 CI contracts pass. TASK13260.278.18.3 tracks API drift
(UAT475). Declared Pydantic 2.13.5 and checkout-local package sources reproduce
the upstream fingerprint; the earlier extra component came from the stale local
environment. Exactly three retained response/request components change. The
existing generation pipeline and drift gate pass, with full schemas/types kept
ignored; 79 contract checks pass, including three official PostgreSQL cases.
Final WebUI and extension types pass. Expanded UAT473 controls pass 281 across
nine shared suites and 186 across seven Character/mirror suites; these overlap.
The profile-only public-loader follow-up passes 130 across five affected suites,
zero skips. Scoped lint adds no findings; independent review is clear. Preserve
the distinction between transactional fixtures and native IndexedDB acceptance.

## Stage 3: Publish and handle review
**Goal:** Update PR2979 and resolve current CI/Qodo findings before full UAT.
**Success Criteria:** Current-base reviewable head; honest tracker and PR gate status.
**Tests:** Hosted checks and exact PR head/base; generated-artifact exclusion.
**Status:** In Progress

Source checkpoint `0bc05d3f35` is published, and GitHub confirms dev `3d102e0d31`
as its base. Final pre-push fetch showed zero missing dev commits. The PR body
records all settled repairs, bounded evidence and excluded generated artifacts.
Fresh hosted CI/Qodo and the requester-owned Change summary remain required;
the prior 14 review threads are resolved, not a new-head review receipt.
