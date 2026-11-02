# UserProfiles Stage 2 Verification Follow-Ups

Tracking: TASK-15413. Base: dev `3ab1ce5d3a49d5a7fd130e8939d55d7639e5a373`.
WP2 runtime is merged; these follow-ups do not start WP3 or alter merge policy.
The requester authorized these outstanding repairs on November 2.

## Stage 1: Reproduce Recorded Failures
**Goal**: Establish the boundary and cause of each recorded failure.
**Success Criteria**: Deterministic delayed-layout reproduction for the Companion
Home notice; fresh focused controls for recorded UI assertions and Python/ACP
environment failures. Prior isolated passes do not qualify a full suite.
**Tests**: CompanionHomePage, useSpeechHistory, useComposerVariantPreference,
workspace.persist-diff, Studio.model-targets, StandaloneHtmlWorkspace, Workspaces
collection, and the previously failing ACP/WebSocket tests.
**Status**: Complete

## Stage 2: Narrow Repairs
**Goal**: Correct established test synchronization, fixture, import or environment
defects without weakening product behavior or verification.
**Success Criteria**: Original assertions and security boundaries remain effective;
no timeout inflation, blanket skips, baseline changes, test exclusions, required
gate dispatch or workflow/policy changes. Each change has RED/GREEN evidence.
**Tests**: Affected files and adjacent scopes, lint, compile, diff, and applicable
touched-scope Bandit. Use official fixture-managed PostgreSQL if needed, never
concurrent suites sharing fixed databases.
**Status**: Complete

## Stage 3: Independent Review and Publication
**Goal**: Publish a coherent, independently reviewed follow-up patch.
**Success Criteria**: Validate spec and code quality, record accurate separate test
runs and remaining environmental constraints, and require all current protected
checks plus applicable human-summary policy before merge.
**Tests**: Canonical task formatting, review findings, exact-head CI and merge
readbacks. No unchanged whole-PR review or unnecessary broad-suite repetition.
**Status**: In Progress

## ADR Assessment
No new ADR is required for behavior-preserving test synchronization or local
verification environment repair. ADR085's membership transaction/version decision
is unchanged. Any proposed runtime behavior change must be assessed separately
before implementation, not folded into a test-only correction.

## Scope Boundaries
WP3-WP5 retain their approved Stage 2 plan and sequential package review gates.
WP2 tracking closeout is a separate two-file PR with the requester's explicit
human-summary waiver; that waiver does not apply to runtime changes here or later.

## Verification Checkpoint (November 2)

The patch changes tests and their environment only. Companion Home waits for
the independently persisted layout; shared UI uses the real jsdom Storage
object; Studio waits for actual modal visibility; the composer quota spy is
restored even on failure. Workspace collection uses declared httpx, the auth
matrix isolates only downstream runner startup, and backup fixtures align both
real settings consumers after collection reloads. Original behavior and safety
assertions remain; no product, dependency, workflow, timeout or skip changes.

Separate final controls: six affected UI files pass 65/65 with v8 on Node 25;
the unchanged frontend StandaloneHtmlWorkspace file passes 111/111 on Node 22;
the four Python files pass 366/366; complete root collection succeeds with
85,935 tests. The backup controls additionally pass seven cases after full
root collection. Focused RED/GREEN and scoped independent review evidence is
linked in TASK-15413. Python Bandit adds only three expected test assertions,
with no new non-assert findings; backup Bandit remains empty. Full-tree test
quality enforcement and the 869-shard guard pass without baseline changes.

Non-green evidence is retained: bare root execution was interrupted (exit 130)
after 1,000 passes and two backup failures, subsequently repaired and verified
in the root-collection context. Bare shared-UI execution was interrupted
(exit 130), without a complete suite result. An initial six-file UI run had
63 passes and two Studio 15-second timeouts while other UI jobs ran; after
stopping only owned competing jobs, the unchanged command passed 65/65. That
does not establish harmlessness or full-suite correctness. CI uses Node 20,
which was not executed locally. No compiled Go runner/RPC integration was
claimed. Exact-head remote CI and the human-summary gate still govern merge.
