# Scene record integrity and truthful device acknowledgement — TASK13549

The approved all-issue remediation continues on latest dev4d6a85c4c693c4c83480d0f7c8046cfda0634c55, included by merge742cb26271. The shared scene getter currently collapses present malformed/unsupported records into absence or neutral settings. The scene writer validates neither the whole payload nor the durable backend after completion. Repair those trust boundaries without changing storage identity in this bounded unit; qualified ownership/read receipts/applied projection/first-bind remain required next.

ADR assessment: no new ADR; preserve ADR049/private D7 validation and ADR059 tracking. No new authority, key schema, public protocol, background replay or dependency. Existing neutral defaults remain valid for verified undefined absence. Present bad records remain untouched and propagate read failure ahead of profile fallback. Supported valid records retain existing aspect-key normalization.

## Stage 1: Trace shared consumers
**Goal**: Identify all scene readers/writers and their recovery boundaries.
**Success Criteria**: Shared service, three scene editors, startup and full/compact/native send paths are inventoried; no UI reset or fallback can certify a broken record as neutral.
**Tests**: Read-only call-site and existing recovery-fixture inventory.
**Status**: Complete

## Stage 2: Qualify causal failures
**Goal**: Demonstrate malformed/version failures and false durable Save acknowledgement.
**Success Criteria**: Actual service tests retain raw records, block profile fallback and reject invalid saves; a held write followed by backend loss returns false. Valid supported reads/writes and undefined absence remain positive controls.
**Tests**: Malformed/falsy/legacy/future scene matrix through both read APIs; no storage mutation/profile read; invalid Save and held backend loss.
**Status**: Complete

## Stage 3: Repair shared integrity checks
**Goal**: Fail explicitly at the actual service boundary.
**Success Criteria**: Only undefined is absence, complete supported shape/version is required before migration or Save, and Save ACK requires persistent backend after completion. Reuse existing validators and default normalization.
**Tests**: New causal tests plus existing actual service and scene editor recovery cases; no disabling tests or silent cleanup.
**Status**: Complete

## Stage 4: Verify and continue ownership repairs
**Goal**: Commit a tested bounded repair while retaining the whole master scope.
**Success Criteria**: Affected service/editor checks, both client types, authored lint/format/security, canonical task and two independent source reviews pass. Commit through normal hooks; continue qualified namespaces/read receipts/projection/first-bind. Bandit N/A for TS/Markdown only.
**Tests**: Scoped causal/actual recovery gates, typecheck/extension compile, lint delta, formatter runtime equivalence, diff and normal commit. No native/packaged/physical AT/participant/all-master claim.
**Status**: In Progress
