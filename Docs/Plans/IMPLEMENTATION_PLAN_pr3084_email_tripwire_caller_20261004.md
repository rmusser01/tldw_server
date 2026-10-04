# PR3084 email tripwire caller repair implementation plan

Associated task: TASK-13260.278.18.83.45. Human explicitly requested fixing the remaining failures after the monitoring hold.

## Constraints
- Start from published e6a2; exclude all local closure/tracking commits. Integrate reviewed dev 502da5bf task-only changes while preserving all nine published commits and the diagnostic patch.
- Preserve all existing offline guards, assertions, production source and CI until causal evidence supports a repair.
- Earlier three diagnostic attempts remain exhausted. At most three new declared, distinct causal experiments; no unchanged full heavy shard replay, model download or old CI log redownload.
- Existing Python 3.12 environment only; no dependency/cache/model/environment workaround. Match actual bound CI inputs only to reproduce a stated difference.
- No proof files or bundles. Native/UAT/installer limits and separate pending API replacement approval remain intact.

## Stage 1: Identify the forbidden caller
**Goal**: Trace the actual failing path with content-free function metadata.
**Success Criteria**: Caller and causal trigger reproduced, or stronger content-free tripwire diagnostic published with unchanged rejection behavior to identify the hosted caller; no causal fix claim before evidence.
**Tests**: First declared experiment selects only the failing email upload while collecting the actual integration directory, using the bound CI Redis inputs and in-memory call tracing. Outer 600 seconds and per-case 300 seconds. No other test executes. Experiment 1 passed (1 selected/239 deselected/46 warnings/24.23s pytest/32.756s outer), no caller. Add a behavioral red/green test for boundary/caller diagnostics without argument retention, validate all existing email attachment and offline cases, then independently review and publish this diagnostic to obtain the hosted caller. Do not replay experiment 1 unchanged.
**Status**: In Progress

## Stage 2: Repair the demonstrated cause
**Goal**: Minimal source or fixture correction with the offline contract intact.
**Success Criteria**: Causal red on unchanged source, green after correction; original assertions retained; affected tests pass naturally.
**Tests**: Determined by the Stage 1 caller. No broad native, frontend, export or model controls.
**Status**: Not Started

## Stage 3: Review and publish
**Goal**: Review and publish the diagnostic, then any independently qualified causal fix to existing PR3084.
**Success Criteria**: Scoped baseline/current Bandit, lint/compile, independent actual-source and staged review, exact-head/dev/human-summary guards and normal hooks/exact-lease publication; no merge acceptance.
**Tests**: Affected checks and supported owned-task format check. Complete incoming dev review verifies 2,271 task-only paths: 2,269 exact supported normalizations preserve frontmatter/content and are idempotent; separate TASK-13441 completion and TASK-13443 follow-up reviewed. Verify nine-commit range-diff and complete owned binary patch after integration. No unchanged application/native/build controls for this prose-only base. ADR assessment: no new architecture decision; ADR-059 task tooling unchanged.
**Status**: In Progress
