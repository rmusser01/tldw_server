# PR #2633 Review Rebase Design

Task: TASK-12148
Date: 2026-07-12
Status: Approved

## Goal

Rebase PR #2633 onto the current `origin/dev`, preserve the Research Workspace artifact-verification behavior that is still additive to `dev`, address every review finding that remains valid after the rebase, and update the existing PR branch without disturbing unrelated local work.

## Current State

- PR head before repair: `07292d91aa046f60902d0a81cd0ab354ed991871`.
- Approved base: `origin/dev` at `5a309be86b043f5a67b65324a81819f59aa860fc`.
- Divergence at audit time: 832 base-only commits and 10 PR-only commits.
- PR state: open, conflicting, and 58 files changed.
- Review inventory: 20 unresolved inline threads plus one CodeRabbit outside-diff finding.
- GitHub Actions inventory: no failing Actions checks; the only reported checks are completed external reviewers.
- Clean pre-rebase baseline: 101 focused frontend tests and 19 focused backend tests pass.

## Chosen Approach

Use a history-preserving rebase in an isolated repair worktree:

1. Rebase the 10 PR commits onto the audited `origin/dev` commit.
2. Resolve each conflict against the current `dev` architecture, retaining PR behavior only when it is not already present or superseded.
3. Re-audit every review finding against the rebased result.
4. Add a failing regression test before each behavior fix, then implement the smallest change that makes it pass.
5. Apply mechanical documentation/test strictness fixes directly when no runtime behavior changes.
6. Run touched-scope verification and security checks.
7. Push the repair branch to the PR head ref with `--force-with-lease` pinned to the audited remote head.
8. Reply in each GitHub review thread, resolve addressed threads, and recheck PR state.

The repair worktree uses local branch `codex/pr-2633-review-rebase`. The existing checkout of `codex/issue-2605-research-workspace-uat` is left untouched because it contains unrelated untracked files.

## Conflict Resolution Policy

- Treat current `origin/dev` behavior and interfaces as the default source of truth.
- Preserve PR additions for Claims-based artifact verification, Research Workspace artifact generation, and source-grounded output validation when they remain absent from `dev`.
- Do not restore code paths that `dev` has replaced, especially authentication/bootstrap and frontend generation flows.
- Prefer already-landed helpers and tests over duplicate PR implementations.
- Resolve conflicts one commit at a time and inspect the semantic diff after the rebase.
- Do not broaden the PR with unrelated cleanup discovered during conflict resolution.

## Review Disposition Rules

Every finding receives one of three explicit dispositions:

- **Fixed:** the issue remains present after rebase and is covered by a focused regression test or a strict mechanical check.
- **Satisfied by dev:** current `dev` already contains the requested behavior; preserve it during conflict resolution and cite the resulting code/test in the thread reply.
- **Rejected with rationale:** the suggestion conflicts with approved behavior, is based on a stale implementation, or would add unnecessary abstraction. Reply with concrete code/test evidence before resolving the thread.

No reviewer suggestion is applied blindly. Outside-diff feedback is tracked with the same standard as inline feedback.

## ACP Authentication Decision

For `authMode: "single-user"`:

1. Normalize the configured API key.
2. Use a valid, non-placeholder configured key when present.
3. Otherwise use the normalized runtime single-user override.
4. Return no single-user key for multi-user mode.

This preserves the current `dev` contract introduced with the API-key persistence work. It satisfies the valid stored-key precedence comment and the runtime fallback requirement while preventing a single-user runtime override from leaking into multi-user requests. The conflicting request for unconditional runtime precedence will be answered with this approved contract and focused tests.

## Expected Review Fixes

The post-rebase audit is expected to cover these areas:

- ACP key normalization, fallback, type contract, precedence, and multi-user isolation.
- E2E click fallback error filtering and auth-evidence assertions.
- Research Workspace slides API/fallback result discrimination.
- Quiz question/source-media association after normalization filters invalid entries.
- Empty flashcard generation before claim verification.
- Research Workspace `media_ids` request bounds.
- Claims verification metadata preservation and truncation verdict downgrades.
- Claims verifier environment overrides.
- Missing status mapping coverage and strict monkeypatch targets.
- Duplicate Backlog section markers.
- Repeated frontend claim-verification fixtures, only if a small existing shared-test pattern can be reused without adding speculative infrastructure.

## Testing Strategy

### Rebase verification

- Confirm `git merge-base HEAD origin/dev` equals the audited `origin/dev` commit.
- Inspect `git range-diff` between the original 10-commit series and the rebased series.
- Run `git diff --check origin/dev...HEAD`.

### Frontend

- Focused Vitest for ACP connection behavior.
- Focused Research Workspace StudioPane Stage 1/2/3 suites.
- Focused tests added for slide fallback and quiz source-media association.
- TypeScript typecheck for the frontend workspace when the rebased dependency graph permits it.
- Parse/list the real-backend Playwright spec; run the live workflow only if its required services are available.

### Backend

- Focused Claims verification unit/property tests.
- Focused Research Workspace artifact generation tests.
- Focused flashcards, quizzes, and slides endpoint/service tests touched by the review fixes.
- Add configuration reload coverage for verifier environment overrides.

### Security and hygiene

- Run Bandit from the project virtual environment over every touched Python implementation path.
- Run relevant linters/format checks for touched frontend files.
- Confirm no conflict markers, secrets, generated databases, or unrelated files are included.

## GitHub Update

The remote update must use force-with-lease against original PR head `07292d91aa046f60902d0a81cd0ab354ed991871` so concurrent remote work cannot be overwritten. After the push:

- Re-fetch PR metadata and checks.
- Reply to inline comments in their existing threads.
- Resolve threads only after the response and corresponding code evidence are present.
- Add one concise top-level status comment only if needed to summarize non-inline outside-diff feedback.
- Leave the repository's human-authored `Change summary` merge gate for the human requester; do not fabricate that summary.

## Non-Goals

- Do not squash or rewrite the PR into a new feature design.
- Do not modify unrelated dirty worktrees or local `dev` changes.
- Do not fix pre-existing `dev` issues that are not required by a conflict or review finding.
- Do not run external-reviewer autofix services.
- Do not claim the PR is merge-ready while the human-authored change summary remains missing.
