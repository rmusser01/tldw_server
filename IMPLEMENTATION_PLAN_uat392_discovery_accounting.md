# UAT392: explicit discovery and exact case accounting

Task13260.278.5; approved engineering sweep. Integration checkpoint c2b7ab1fe8.

## Stage 1: Prove discovery and accounting gaps
**Goal**: Preserve complete-profile collection failures and count-preserving false passes.
**Success Criteria**: Wrong, duplicate, retried or unbound required cases fail meaningful report regressions.
**Tests**: Existing real collection receipts; new exact-case controls against current project accounting.
**Status**: Complete

## Stage 2: Reuse existing discovery and validators
**Goal**: Explicit Playwright spec boundary, a fixed list-only environment, full workflow project selection, and identity checks extending the current reporter.
**Success Criteria**: Collection cannot execute browsers or inherit application credentials; exact required identities and first attempts are checked alongside existing no-skips/project checks.
**Tests**: New regressions plus existing live-tier runner tests and real full/workflow collection.
**Status**: Complete

## Stage 3: Verify scope and preserve remaining release gates
**Goal**: Record actual collection and validation results without confusing registrations with execution.
**Success Criteria**: No new diagnostics, independent review, generated receipts private. Runtime artifact verification and a fully mapped43-family executable manifest remain required before AC3 or freeze.
**Tests**: Source-hashed registration inventory and targeted tests/lint/Bandit applicability.
**Status**: In Progress

## Verification and remaining gate

Exact-accounting controls first failed20 cases with5 positive controls; stable project/file/full-title identity controls then failed6 with24 controls. Final accounting/collection/existing-runner suite passes64 tests; the combined new harness suite passes126 across9files. Final full collection:1525 registrations/9projects/0errors/0execution; workflow collection:671. Firefox13 cases have stable identities across project selection despite13 changed raw Playwright IDs. Frontend app TypeScript and seven-file scoped lint are clean. These JavaScript/TypeScript-only changes are outside Bandit analysis; the existing eight touched production Python files already have a clean scan.

Independent review identified raw Playwright ID instability; the correction and actual collection comparison are complete. A delegated rereview was interrupted by the account usage limit, so it is not a completed review. Root reviewed the final visitor/validator/CLI and causal regressions. AC1/AC2 are satisfied within discovery and receipt validation. AC3 remains open: the runner must verify actual runtime artifacts and the complete43-family executable manifest before recording metadata. No frozen release or native result is implied.


## PR-first follow-up — 2026-09-22

Independent review confirmed that execution still used project totals, while the runner's certification label only checked three selected projects. Six causal regressions expose substituted/duplicated/retried identities, false certification, and list-only application startup. Execution now reads exact registration JSON, shares the same identity/first-attempt validator as release receipts, and preserves individual attempts in Markdown. List-only uses the existing credential-free collection profile and starts no services. Every current dev-server run is explicitly diagnostic; the commit is captured at start rather than guessed afterward.

A further review found that cancellation was checked before reading a flushed partial result. The runner now reads the outcome after teardown, records any partial report, and preserves the original abort reason. The combined runner/accounting/collection suite passes 73 tests. Actual tier1 list-only collection finds 34 registrations, 0 errors and 0 executions, without an application profile or reserved ports. Scoped lint passes.

Catalog follow-up adds an explicit versioned 43-family requirement inventory and complete scope/variant/context/mode reconciliation against exact list-only registrations. The public --catalog command validates a plan and labels its output certifiesRelease:false. Exclusions require reasons; unmapped cases remain gaps; human review remains separate. This does not claim that existing tests satisfy their declared assertions or that candidate runtime artifacts are verified. AC3 stays open until actual production execution is bound to independently verified artifacts.

Final diagnostic-runner review is clean after the cancellation repair. Whole frontend app type checking and scoped ESLint both pass with empty output. The separate catalog change is still under review; these runner checks do not close AC3.
