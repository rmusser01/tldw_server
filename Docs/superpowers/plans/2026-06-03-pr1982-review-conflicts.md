# PR 1982 Review Comments and Merge Conflicts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve PR #1982 review comments and merge conflicts for `dev` into `main`.

**Architecture:** Keep the PR branch as the source of truth for overlapping conflict hunks, per requester instruction. Apply only still-valid review fixes, keeping workflow and documentation edits minimal.

**Tech Stack:** Git, GitHub Actions YAML, Docker Compose YAML, JSON design tokens, Markdown documentation.

---

## Stage 1: Review Inventory

**Goal:** Identify all still-valid unresolved review comments and current check/conflict state.
**Success Criteria:** Unresolved comments and pending/failing checks are documented in Backlog notes.
**Tests:** `gh pr checks 1982 --repo rmusser01/tldw_server --json ...`
**Status:** Complete

- [x] Query unresolved, non-outdated review threads.
- [x] Query current PR check rollup.
- [x] Record findings in Backlog/task notes.

## Stage 2: Merge Conflict Resolution

**Goal:** Merge current `origin/main` into the PR worktree with `dev` winning overlapping conflict hunks.
**Success Criteria:** `git status` shows no unmerged paths and no conflict markers remain.
**Tests:** `git diff --check`; `rg '^(<<<<<<<|=======|>>>>>>>)'`
**Status:** Complete

- [x] Merge `origin/main` into the PR branch.
- [x] Resolve conflict hunks in favor of `dev` where both sides overlap.
- [x] Keep non-overlapping `main` changes that merge cleanly.

## Stage 3: Review Comment Fixes

**Goal:** Address still-valid CodeRabbit comments.
**Success Criteria:** Workflow/comment guidance issues are fixed in the touched files.
**Tests:** YAML parse checks where practical; JSON parse check for `.impeccable/design.json`.
**Status:** Complete

- [x] Prefer `pyproject.toml` before requirements in `.github/workflows/sbom.yml`.
- [x] Set `persist-credentials: false` on the self-hosted checkout in `.github/workflows/vz-linux-host-gated.yml`.
- [x] Align `.impeccable/design.json` status badge CSS with `badge-status` tokens.
- [x] Fix `DESIGN.md` hyphenation.
- [x] Clarify `NEXT_PUBLIC_X_API_KEY` quickstart-only guidance in `Dockerfiles/docker-compose.webui.yml`.
- [x] Widen parity workflow trigger paths and include hidden artifact uploads.

## Stage 4: Validation and PR Metadata

**Goal:** Verify the touched scope and update PR metadata warnings.
**Success Criteria:** Validation results are recorded; PR title/body no longer have obvious template failures.
**Tests:** `python -m json.tool .impeccable/design.json`; targeted YAML parsing; Bandit skip documented for non-Python-only touched scope or run if Python files are touched.
**Status:** Complete

- [x] Run local validation on touched files.
- [x] Run or document Bandit per touched scope.
- [x] Update PR title and body if GitHub permissions allow.
- [x] Update Backlog final notes.

Validation completed:

- `git diff --name-only --diff-filter=U`: no unmerged paths.
- `rg -n "^(<<<<<<<|>>>>>>>|=======$)"`: no exact conflict markers.
- `git diff --check` and `git diff --cached --check`: passed.
- `python -m json.tool .impeccable/design.json`: passed.
- PyYAML parse for `.github/workflows/sbom.yml`, `.github/workflows/vz-linux-host-gated.yml`, `.github/workflows/ui-research-workspace-parity.yml`, and `Dockerfiles/docker-compose.webui.yml`: passed.
- Bandit production touched scope: passed.
- Bandit all touched Python files with pytest `assert` warning `B101` suppressed: zero findings.
- Focused pytest slice for OCR runtime/backend tests and MCP catalog endpoint tests: `51 passed`.
- Onboarding docs-gate reproduction for benchmark/OpenWebUI/flashcards discoverability tests: `9 passed`.
- Post-push frontend install failure reproduction: `bun install --frozen-lockfile` initially failed because `apps/bun.lock` was stale after the merge; regenerated `apps/bun.lock` and reran successfully with no changes.
- Post-push playground composer gate reproduction: `PlaygroundForm.signals.guard.test.ts` expected the old unused `_data` callback signature; updated the guard to assert the current `data` normalization path and reran `bun run test:playground:composer`: `6 passed`, `38 passed`.
- Post-push playground device-matrix reproduction: `ResearchWorkspaceBody` discarded `workspaceSourceStatus` and `workspaceStatusProjectionLoading` state while rendering `WorkspaceTrustPanel`, causing the workspace error boundary. Restored the state reads and reran `env CI=true bun run test:playground:device-matrix`: `6 passed`, `35 passed`; `env CI=true bun run test:playground:a11y`: `10 passed`, `24 passed`.
