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


Catalog verification completes with 43 families, 426 named variants and 6 explicit advertised-instance inventories. Independent review identified that generic format/connector rows could not represent two advertised options; 12 causal failures now pass after instance identity and expansion were added. All 140 combined checks pass (67 catalog plus 73 runner/accounting/collection); ESLint, Node syntax and frontend TypeScript pass. Independent scoped rereview finds no remaining material issue. Bandit is inapplicable to the JS/TS-only delta.

### Remaining runtime binding work

Reuse the existing isolated profile, initialization and official PostgreSQL fixture lifecycle; do not add a second scheduler. Promote only reviewed reusable source/build verification from the private diagnostic harness, stripping run-specific paths and historical recovery claims. A retained BUILD_ID is not proof of a successful build, copied dependencies are reuse rather than clean installation, source inventory must reject added executable files, and symlink targets/referenced bytes need verification.

The serving process must attest actual imported application module origins after startup, active application database identities and restricted PostgreSQL role state. WebUI HTTP/browser asset bytes must match the sealed production artifacts; extension verification must seal the actual loaded tree after any declared manifest transformation and check packaged resource bytes. Bind a fresh owned process/run identity to those observations, recheck after execution, then populate candidate metadata. Mere matching declared hashes, health response, listening port or a separate interpreter/DB connection does not meet this gate. Existing private runtime helpers are diagnostic evidence, not a completed portable release runner. Unit rejection cases and integration wiring remain engineering work; actual native acceptance remains paused.


### Quiescent artifact integrity — 2026-09-22

Added a Node-stdlib complete tree inventory and two modes in the existing receipt CLI. No ignore patterns: added/removed/modified files, permissions, hidden entries, directories and symlink bytes/ultimate referents are checked. Unsafe/cyclic/broken links and special files fail. External receipts use exclusive writes and cannot replace prior evidence. Independent review identified an initial root-mode digest gap; its causal regression now passes with the full canonical-entry digest. All130 artifact/catalog/accounting tests pass (27 artifact controls); lint/syntax are clean and independent rereview has no actionable findings. JS/TS-only delta, Bandit inapplicable. No application process or UAT launched.

This is a reviewed integrity prerequisite, not completed AC3. Actual build receipts, serving-process imports and active database identity, browser/extension loaded assets, before/after runtime checks and audited workflow mappings remain required. A trusted receipt and quiescent tree are explicit preconditions; the helper is neither an atomic filesystem snapshot nor a signature.
