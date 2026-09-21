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
