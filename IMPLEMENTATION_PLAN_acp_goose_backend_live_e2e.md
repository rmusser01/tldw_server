# ACP Goose Backend Live E2E Implementation Plan

## Stage 1: Reproduce Concrete Failures

**Goal**: Capture the runner-cwd and live-E2E wrapper failures as focused tests.
**Success Criteria**: Tests fail for the shipped ACP runner cwd and bare Python manifest argv.
**Tests**: Focused pytest on ACP config cwd and certification smoke manifest tests.
**Status**: Complete

## Stage 2: Fix ACP Runner Defaults

**Goal**: Point the shipped ACP runner cwd at the in-repo `tools/tldw-agent` source.
**Success Criteria**: Default ACP config validation no longer warns about a missing runner cwd.
**Tests**: ACP config cwd tests.
**Status**: Complete

## Stage 3: Harden Live-E2E Manifest Execution

**Goal**: Make Python-backed certification manifest commands use the current interpreter.
**Success Criteria**: Manifest execution no longer depends on a bare `python` executable being on PATH.
**Tests**: ACP certification smoke manifest tests.
**Status**: Complete

## Stage 4: Run Goose Backend Live-E2E

**Goal**: Re-run Goose through the backend live-E2E path and update support metadata only if it passes.
**Success Criteria**: Goose evidence is either recorded as passing support metadata or documented as a remaining blocker.
**Tests**: Live-E2E helper against local backend plus focused regression tests.
**Status**: Complete

## Stage 5: Verify and Finalize

**Goal**: Run focused tests, Bandit on touched Python, and update task evidence.
**Success Criteria**: Verification output is recorded and the Backlog task reflects final status.
**Tests**: Focused pytest, Bandit, git diff checks.
**Status**: Complete
