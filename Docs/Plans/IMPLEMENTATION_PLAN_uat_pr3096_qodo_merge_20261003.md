# PR3096 rebase, review, and merge

Owner: TASK13260.281. PR: https://github.com/rmusser01/tldw_server/pull/3096

## Stage 1: Rebase on current dev
**Goal**: Publish the existing repairs on the latest dev without the integration merge.
**Success Criteria**: Rebase preserves the reviewed source; publish with an exact force-with-lease.
**Tests**: Compare pre/post trees and run git diff --check; qualify any actual source changes.
**Status**: Complete

## Stage 2: Address Qodo review
**Goal**: Verify and resolve actionable Qodo findings against the actual PR source.
**Success Criteria**: Findings have fixes or source-supported replies; changed behavior has causal regressions and independent review.
**Tests**: Affected suites, touched lint, and Bandit for changed Python production scope.
**Status**: In Progress

## Stage 3: Verify gates and merge
**Goal**: Merge the reviewed final head normally.
**Success Criteria**: Seven required contexts pass on the final head; human-written Change summary is present; exact-head merge is confirmed on dev.
**Tests**: Current PR review/check readback and merged commit ancestry.
**Status**: Not Started

Live UAT remains held. Existing bug notes and runtime data remain; no evidence bundles or failed-case replay.
