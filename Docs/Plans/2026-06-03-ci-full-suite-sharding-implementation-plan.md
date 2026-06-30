## Stage 1: CI Scope
**Goal**: Replace monolithic full-suite PR jobs with jobs whose trigger scope matches the intended signal.
**Success Criteria**: PRs run full Linux coverage for Python 3.12 and 3.13, Python 3.11 runs compatibility smoke coverage, and macOS/Windows Python 3.12 PR checks run full shard coverage.
**Tests**: Inspect the workflow event/job `if` expressions and validate GitHub Actions syntax.
**Status**: Complete

## Stage 2: Parallel Shards
**Goal**: Split the slow backend full suite into independent shard jobs instead of serial module steps in one runner.
**Success Criteria**: Full Linux jobs use a shard matrix and no longer enumerate every module as a separate serial step.
**Tests**: Validate the matrix definitions and run a local YAML parse/action syntax check where available.
**Status**: Complete

## Stage 3: Release Coverage
**Goal**: Run expanded macOS and Windows Python 3.13 coverage only for release/main/manual contexts.
**Success Criteria**: Expanded macOS/Windows Python 3.13 shard jobs are skipped for pull requests and run for `push` to `main`, `release`, and `workflow_dispatch`.
**Tests**: Inspect job conditions and validate the workflow.
**Status**: Complete

## Stage 4: Verification and PR
**Goal**: Commit, push, and open a PR with evidence recorded in Backlog.
**Success Criteria**: Workflow validation has been run, non-code Bandit skip is documented, and a PR exists against `dev`.
**Tests**: `git diff --check`, YAML parse, `actionlint` if available, and `gh pr view`.
**Status**: Complete
