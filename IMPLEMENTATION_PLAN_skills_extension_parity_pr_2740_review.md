# Skills Extension Parity PR 2740 Review Implementation Plan

## Stage 1: Add focused regressions
**Goal**: Capture the theme-bootstrap and fixture-validator review findings before production edits.
**Success Criteria**: Focused tests fail for blocked storage handling, unexpected bootstrap errors, method normalization, and malformed URL diagnostics.
**Tests**: `bunx vitest run tests/unit/options-theme-bootstrap.test.ts tests/unit/skills-fixture-request-contract.test.ts`
**Status**: Complete

## Stage 2: Apply minimal review fixes
**Goal**: Apply the smallest production and parity-harness changes needed to satisfy confirmed review findings.
**Success Criteria**: Expected storage denial falls back safely, unexpected bootstrap errors surface, request methods normalize, malformed URLs produce bounded contract errors, null download streams fail clearly, failed extension setup closes its context without masking the primary error, fixture routes do not overlap, and compact dialog focus remains stable through a details-to-test transition.
**Tests**: Focused Vitest, extension TypeScript compile, and strict packaged-extension Skills parity.
**Status**: Complete

## Stage 3: Rebase, verify, and integrate
**Goal**: Resolve review threads, rebase onto latest `origin/dev`, rerun release gates, and merge when repository policy permits.
**Success Criteria**: No unresolved actionable review threads, clean rebase, required verification passes, and PR merge gates are satisfied.
**Tests**: Focused extension/shared UI tests, strict parity, production extension build, diff hygiene, PR checks, and merge-state verification.
**Status**: In Progress
