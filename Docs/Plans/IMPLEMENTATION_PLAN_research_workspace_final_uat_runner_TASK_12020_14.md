## Stage 1: Define Runner Contract
**Goal**: Capture the final UAT runner behavior without requiring a browser launch during unit tests.
**Success Criteria**: Tests define localhost binding defaults, browser-launch failure classification, product failure classification, skipped-test handling, and evidence artifact shape.
**Tests**: `research-workspace-uat-runner.test.ts`.
**Status**: Complete

## Stage 2: Implement Runner
**Goal**: Add a repeatable Research Workspace UAT command that runs focused Playwright specs and writes a pass/product-failure/environment-blocked evidence JSON.
**Success Criteria**: The runner defaults to localhost-safe WebUI startup and keeps environment failures distinct from product failures.
**Tests**: Focused runner unit tests.
**Status**: Complete

## Stage 3: Document Operating Procedure
**Goal**: Document required permissions, env vars, command usage, fallback path, and how to interpret blocked evidence.
**Success Criteria**: UAT matrix and development docs explain the runner and do not count skipped/blocked runs as product passes.
**Tests**: Documentation review plus `git diff --check`.
**Status**: Complete

## Stage 4: Verify and Update Backlog
**Goal**: Run focused tests/checks and update TASK-12020.14 with exact evidence.
**Success Criteria**: Backlog records verification, touched files, Bandit applicability, and any remaining limits.
**Tests**: Focused Vitest, `git diff --check`.
**Status**: Complete
