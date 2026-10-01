# Chat Workspace Latest-Dev Acceptance

Tracking: TASK-13398, TASK-13398.12 and its implementation children.
Epic: https://github.com/rmusser01/tldw_server/issues/1239.
Baseline: dev `3763d4b013ad31ad48cc7f9d895d2ae096223da8`, rechecked before PR packaging.

## Current Gate

Native account/backend-target acceptance and final Browse/mobile continuation
are still running. This record does not claim complete UAT. The PR must remain
draft/unmerged until these gates are recorded. A requester-authored Change
summary explaining what changed and why is separately required before merge.

## Real Acceptance Recorded

All browser acceptance below used real Chrome raw CDP, native API authentication,
SQLite, browser IndexedDB/localStorage, the actual local Gemma provider and real
embeddings. No request interception, mocked responses, injected application
state, fabricated receipts, focus emulation or plugin installation was used.
Unit-test doubles are separate from browser acceptance.

| Latest-dev check | Result | Evidence |
| --- | --- | --- |
| Grounded selected-history send | PASS: one retrieval with generation disabled, one actual Gemma dispatch, canonical input/result receipts and full source metadata | `native-grounded-keyboard-roundtrip.json` |
| Citation expansion and protected fresh-document restore | PASS: captured memo excerpt, real H1 read, native IndexedDB draft restored; no automatic retrieval/inference on reload | Same grounded evidence |
| Stop, reload, Verify and explicit Reprepare | PASS: two real dispatches, one canonical input; earlier unknown observation retained | `native-stop-latest.json` |
| Deep link, hash, Back/Forward, reload, malformed route, mobile | PASS: six checks, zero completion dispatches, rejected route retains checkpoint, no horizontal overflow | `native-route-latest-complete.json` |
| Workspace A/B/A, shared Research handoff, New Chat precedence | PASS: eleven checks, zero completion dispatches, inactive/outgoing checkpoints preserved | `native-workspace-latest.json` |
| Original data/browser preservation | PASS: original conversation projections retain ten/eight rows and exact hashes; both original targets remain open | `final-preservation/native-after-dev-refresh.json` |
| Actual served OpenAPI | PASS: unchanged production capability predicates and twelve negative controls | `final-preservation/served-openapi-gate.json` |

Private evidence root: `/private/tmp/chat-workspace-task12-v1-20261001`.
Original failed runners are retained, not relabeled as passing: date-format and
Ant Design selector assumptions were corrected; native keyboard scrolling was
used for offscreen citation controls. The final preview runner hit a readiness/
compositor timeout before any preview request or send; its failure and the
subsequent real readiness/migration observations remain recorded separately.

This is not an exactly-once inference claim. Unknown outcomes remain unknown;
explicit re-preparation can produce another provider answer.

## Final Code Verification

- All selected unit files ran directly from their owning package configs in the
  actual checkout: 80 shared files and three frontend files, both commands exit 0.
  Exact commands, results and logs: `final-owned-{shared,frontend}*`.
- Independent immutable broad review: 240 selected paths, zero hash mismatches.
  Initial 2,266 owned assertions passed; the final three-file replacement was
  independently verified with 54 preview/URL assertions and 23 IndexedDB checks.
- Independent strict SourceV1 review passed 166 frontend and 159 Python checks,
  capability/boundary controls and all eight metadata digest/receipt mutations.
- Official SQLite/PostgreSQL scoped regression: 213 passed, one inherited
  SQLite-only connection-lifetime skip. Actual PostgreSQL arms ran; earlier
  sandbox reachability skips are not the acceptance gate.
- Additional changed backend scope: 403 passed, one inherited heartbeat skip.
  Static migration-test SQL cleanup: 20 passed, no skips. Counts overlap and are
  not a unique repository-wide total.
- Final Next production webpack build and full frontend TypeScript both exit 0.
  Build uses the preserved quickstart environment and an isolated output folder.
- Bandit: twelve touched production modules, zero findings/errors. Five setup
  test-fixture B105 warnings exactly match HEAD; assert checks are excluded for
  pytest files. No new findings in the changed scope.
- Final preview and loader ESLint: zero errors/warnings. Capability scope has
  nineteen inherited warnings, verified unchanged; no blanket lint-clean claim.

The real-source JSON fixture is ignored by the repository-wide JSON rule and
must be explicitly included in the PR tree with SHA256
`d5bfed3bb97db124bcc990f13427dfd880b17af2b32984755211a1d06a83771d`.
The IndexedDB unit loader now resolves the existing dependency from its actual
frontend package rather than the runner's working directory; no install.

## Known Limits

The broader neighboring aggregate is not fully green. Nine untouched
SourceViewControls/Kitten assertions reproduce on HEAD; separate untouched
Studio export/lazy-modal assertions are timing-sensitive or not reproduced in
isolation. They remain enabled, with immutable controls in `pr-review/triage.md`.
This PR does not silently change unrelated Studio behavior or claim the full
repository suite passes. Auth/account/target acceptance remains the explicit
open native gate above, not a unit-test substitute.

All 68 stashes were retained. Runtime builds, databases, logs, credentials,
screenshots and unrelated historical task records are excluded from the PR.
