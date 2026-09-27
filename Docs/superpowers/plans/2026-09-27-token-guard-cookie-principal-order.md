# Cookie-principal token-guard ordering implementation plan (TASK-13376.13)

> Execute the approved bounded correction inline with TDD; request a read-only
> reviewer before the exact-source qualification build.

**Goal:** A verified single-user cookie admin can load Knowledge history when
its scope guard precedes the user dependency.
**Architecture:** Retain existing bearer resolution and add missing-principal
resolution only when both explicit credential headers are absent. Use the
existing canonical resolver and unchanged explicit role/permission claim check
within the admin branch. No credential, permission, scope, quota,
CSRF, dependency version or product-gate relaxation.
**Tech Stack:** Existing FastAPI AuthNZ/session integration fixture, pytest,
Bandit, existing signed paired-candidate helper and ordinary Playwright CLI.
**Spec:** Approved review checkpoint in
`Docs/superpowers/reviews/2026-09-26-complete-app-provider-document-qualification.md`.

## Stage 1: Reproduce and correct the guard branch
**Goal:** Real session integration fails first, then passes after the minimal fix.
**Success Criteria:** Minted admin and wildcard permission claims pass; absent,
invalid, expired, revoked, wrong-type and multi-user cookies, explicit invalid
headers, non-admin/legacy boolean claims and bypass-disabled requests reject.
**Tests:** Extend `tests/AuthNZ/integration/test_single_user_cookie_session.py`
with real conversations and alias paths plus guard claim/bypass cases. Reuse
real mint/revocation/session validation; only alter claims for policy cases.
Run new positive cases red before modifying `API_Deps/auth_deps.py`.
**Status**: Complete

- [x] Extend real HTTP cookie coverage and observe expected 401 failures.
- [x] Resolve a missing principal in the existing admin branch.
- [x] Run cookie integration plus existing scoped JWT/API-key/quota regressions.

Red evidence: the unchanged guard returned `401 Authentication required` for
the real minted-cookie canonical history route and both explicit admin claim
cases. The first alias test used an incorrect URL and returned 404; after
correcting it to `/api/v1/chats/conversations`, its separate red run returned
the expected 401. Those results are retained privately. Production changes
only the existing principal-resolution condition and its explanation. Review
caught an initial overly broad condition that would let an admin-owned
`X-API-KEY` bypass its constraints. Five added red regressions cover scope,
endpoint, method, path and quota; the predicate now adds only headerless cookie
eligibility and retains the original bearer behavior.

## Stage 2: Verify and review the approved scope
**Goal:** Commit a reviewed source suitable for qualification.
**Success Criteria:** Scoped formatting/lint and Bandit introduce no findings;
read-only review finds no unresolved important issues. Record baseline findings
separately without broad cleanup. Update this plan/task and commit working code.
**Tests:** Project-venv pytest, changed-scope formatter/linter checks,
`git diff --check`, Bandit on touched Python paths and independent review.
**Status**: Complete

The hashed final scope passed all 82 unique cases: 40 scoped JWT/API-key,
service-token and route-chain cases, plus 42 real-cookie integration cases.
The eight fresh pytest processes each exited zero, validated their JUnit XML,
and reported no failure, error, skip or duplicate. The owned Docker wrapper
exited zero after 804 seconds and removed its container; baseline containers,
images and volumes were preserved and both existing PostgreSQL services were
still running. Private logs, aggregate proof and seven captured XML reports
are retained under `/private/tmp/task13376-cookie-docker-grouped-v2`.

Read-only re-review found no unresolved important findings. Ruff and scoped
Black/diff checks pass; production Bandit and test non-assert Bandit have zero
findings. Whole-file formatting differences and test assertion B101 findings
are recorded separately. Earlier host cleanup/diagnostic and Docker runner
setup failures are not counted as successful runs. Commit this verified scope
with its task and review record before qualification.

## Stage 3: Qualify the exact signed source and record limits
**Goal:** Actual signed candidate and ordinary Knowledge history/provider/document
restart checks prove the correction.
**Success Criteria:** Exact source/signatures/file hashes verified; Knowledge
history succeeds before/after signed recreation; provider setup, Markdown,
lexical search and a distinct new chat pass without manual master-key wiring.
Native evidence remains source-specific; full native/core-format/G12 gates
remain open unless independently satisfied. Owned cleanup preserves baseline
resources and recovery data. Preserve completed own plan privately/history,
then remove only this plan under AGENTS; retain the broader incomplete plan.
**Tests:** Existing `qualify_app_bundle_candidate.sh` on local arm64 with the
reviewed 15-minute quiet bound; ordinary browser workflow on extracted signed
archive outside checkout. Existing native CI and artifact verification.
**Status**: In Progress
