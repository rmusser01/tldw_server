# Persona Stage 1 September dev rebase

September 26 continuation: the requester now explicitly authorizes merge after another refresh onto latest `dev`, current-head Qodo remediation, and passing required checks. Fetched `origin/dev` is `a2826f103f02a67f57adb40ed048dbfa2ecfc6e5`; published PR head before this refresh is `e7a577c9245374a815120f08aa70463d9295348e`. Preserve the dirty root checkout and both recovery snapshots. The human-owned Change summary must explain implementation reasoning before merge; the current body describes the feature but does not yet satisfy that rationale requirement.

## Stage 1: Review the current PR and integration surface
**Goal**: Identify the exact PR commit range, current `origin/dev`, Qodo findings, and overlapping Persona files.
**Success Criteria**: The original branch tip is recorded, every Qodo finding is classified, and a recovery ref exists before rewriting history.
**Tests**: `git status --porcelain=v1`; `git merge-base HEAD origin/dev`; inspect PR review and comments.
**Status**: Complete

## Stage 2: Rebase the scoped Persona series
**Goal**: Replay the Persona Stage 1 net change on current `origin/dev`, preserving newer Persona behavior. The original commit-by-commit replay was aborted when its historic migration conflicted with the current schema; a recovery branch retains the original series.
**Success Criteria**: The integration branch has no unrelated commits or unresolved conflicts; current migration numbering and APIs remain coherent.
**Tests**: `git diff --check`; compare changed-file list with the original PR; migration tests.
**Status**: Complete

## Stage 3: Repair and verify integration regressions
**Goal**: Resolve confirmed behavior or test failures caused by the rebase using failing tests before code fixes.
**Success Criteria**: Affected backend and frontend tests, type/lint checks, and Bandit pass or have documented, scoped failures.
**Tests**: Persona API/DB and migration pytest suites; Persona Buddy frontend tests; frontend type/lint; Bandit on touched Python files.
**Status**: Complete

### Latest-dev continuation, September 26

- [x] Rebase onto `a2826f103f02a67f57adb40ed048dbfa2ecfc6e5` with no conflicts. The original implementation and CI-fix patches are unchanged according to `git range-diff`. Recovery ref: `codex/persona-ambient-stage1-pre-20260926`.
- [x] Verify 292 focused UI tests in 22 files under Node 20 and CI's existing 15-second timeout.
- [x] Verify the OpenAPI fingerprint using the temporary CI-version dependency overlay; no drift.
- [x] Run Bandit over the touched backend scope; zero findings and zero errors. Report: relative `bandit_persona_latestdev.json` (untracked/ignored).
- [x] Reproduce two inherited published-document parity failures and synchronize only the tokenizer admin-only clarification in the generated character-chat and environment-variable guides. Keep canonical sources and existing tests unchanged.
- [x] Re-run all 212 Docs tests, including strict MkDocs; all pass after the exact generated-document synchronization.
- [x] Re-run the scoped Persona backend matrix: 382 pass and 3 PostgreSQL integration cases skip only when the existing fixture reports no available database. Use the native macOS temporary root; a `/private/tmp` basetemp is intentionally rejected by database-path enforcement.
- [x] Verify scoped compilation, public/private documentation boundary checks, and ESLint (zero errors, one existing `any` warning).
- [x] Perform post-commit documentation verification: tracked published-file parity and strict MkDocs both pass (2 tests). The final amended tracking commit changes no runtime or generated-document bytes.
- [ ] Publish with a fresh remote lease; confirm current-head Qodo and required CI before merge.

**ADR check**: ADR required: no. This refresh preserves the approved Persona runtime contracts and introduces no durable architectural decision. Governing accepted policies: ADR-004 (human-written Change summary), ADR-006 (portable Bandit report paths), and ADR-020 (per-user database ownership). Record the same assessment in TASK-12125.11 and the PR description.

### CI follow-up authorized September 25

- [x] Reproduce the asset tests under CI's Node 20. Node 26 passes; Node 20 rejects the jsdom `ArrayBuffer` at `crypto.subtle.digest`.
- [x] In `apps/packages/ui/src/services/persona-visual-assets.ts`, pass `new Uint8Array(bytes)` to WebCrypto without weakening checksum validation. Both asset test files pass under Node 20 and Node 26, using the package-owned UI config.
- [x] Reproduce `test_refresh_output_matches_tracked_published_files`: the generated Buddy documentation is missing from Git's tracked published tree.
- [x] Run `bash Helper_Scripts/refresh_docs_published.sh`, review the generated diff, and track the new Buddy document plus its updated Persona Visual Packs link. The full 212-test Docs suite, including strict MkDocs, passes. Exclude two unrelated generated tokenizer-doc updates already stale on dev. Recheck strict MkDocs after committing to verify the new published document has Git history; do not relax strictness or timestamp checks.
- [x] Reproduce the OpenAPI drift gate: the new Persona endpoints add four paths. The shared local venv has older schema dependencies than CI, so its fingerprint must not be committed.
- [x] Regenerate frontend API types and fingerprint using CI's FastAPI 0.136.3 / Pydantic 2.13.5 / Starlette 1.7.0, verify the 2101-path/3217-schema CI fingerprint (`9a598ad3e5aa527e64f94882b0a6a60e3d366ae9ba975781a9e23d6884cec33a`), and re-run the drift check.
- [x] Verify 289 focused Buddy UI tests under Node 20 with CI's 15-second timeout, both asset suites under Node 26, `git diff --check`, and ESLint (no errors; existing `any` warning). Bandit reports zero findings on the companion behavior, visual fingerprint, and visual-service scope. TASK-12125.11 records results.

Verification environment note: the first full Docs run exhausted local disk while creating disposable test copies; it was stopped, and only its identified temporary outputs were removed. The retry used an isolated task-specific basetemp with successful-fixture cleanup and passed all 212 tests. An initial broader UI run used the default 5-second timeout and timed out during imports; the retry with CI's existing 15-second timeout passed without code or timeout-policy changes.

## Stage 4: Update PR and review closure
**Goal**: Push the rebased branch safely, address fresh Qodo feedback, and merge only after the requester's human-owned rationale and repository gates are satisfied.
**Success Criteria**: Remote head matches the tested local head, every actionable Qodo comment is resolved or answered in-thread, required current-head CI passes, and merge preserves repository governance without bypasses.
**Tests**: GitHub PR metadata, review comments, exact-head checks and license status after push; verify merge result via API.
**Status**: In Progress

Closure verified September 26 at 08:48 UTC on published head `e7a577c9245374a815120f08aa70463d9295348e`: all seven required gates pass, all eight frontend test shards and four full-suite platform/Python variants pass, and Qodo's summary explicitly identifies this head with zero open findings. No failed or timed-out checks remain. GitHub reports `BEHIND`; this is ready for human review, not a merge-readiness or governance approval claim. The follow-up is stopped without merging. This tracking-only closure stays local so the reviewed remote head remains unchanged.
