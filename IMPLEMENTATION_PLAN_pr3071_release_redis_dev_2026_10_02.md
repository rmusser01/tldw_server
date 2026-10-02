# PR3071 Release And Redis Dev Qualification

Tracking: TASK-13421. Existing draft PR3071; no merge authorization.
Frozen dev: 413c2c9123509f17d96d514a7722e076598ea28f.
Preserve live services, Chrome tabs/profiles, rows, drafts, stashes and unrelated files.

## Stage 1: Preserve And Integrate
**Goal**: Archive the completed own TASK-13418 intact before merging incoming
unrelated same-ID records; retain all histories without changing upstream tasks.
**Success Criteria**: Original tracker hash matches its archive; clean frozen-dev
integration and understood source delta.
**Tests**: Archive hash, merge preview and incoming-record/source comparison.
**Status**: Complete

## Stage 2: Qualify Shared Runtime Changes
**Goal**: Review and verify incoming release, Redis governor, sync, startup-secret
and smoke changes that can affect admission or workspace behavior.
**Success Criteria**: Relevant owning/adjacent tests and scoped Bandit have no
new actionable failures; no unrelated cleanup or disabled guards.
**Tests**: Resource-governance, relay recovery, initialize-secret and smoke
regressions; focused review and touched production security checks.
**Status**: Complete

## Stage 3: Qualify Actual Workspace And Contract
**Goal**: Verify the intended current OpenAPI contract and actual latest-source
Chat Workspace without mocks or fabricated state, while preserving real data.
**Success Criteria**: Actual Chrome loaded/authenticated state, bounded live
provider flow where required, citations/draft/reload and no automatic resend;
source-bound reuse is labeled and real retained data/tabs/stashes stay intact.
**Tests**: Existing schema export/drift/type checks if changed, owning-source
bindings, native Chrome/CDP acceptance and read-only preservation.
**Status**: In Progress

Desktop/live-provider/reload and actual latest-API preservation pass. Mobile
viewport sizing was rejected by auto-review; explicit user approval requested.
Historical mobile evidence remains labeled, not promoted to fresh413 evidence.

## Stage 4: Verify And Publish
**Goal**: Normally publish one verified integration to the existing draft PR,
with requester Change summary unchanged and no merge.
**Success Criteria**: Scoped hooks/checks pass; exact head/body/state readback
matches; queued, skipped and completed CI results are reported distinctly.
**Tests**: Final source/security/test evidence, normal commit/push and PR readback.
**Status**: Complete

Normal source publication93780ae and exact PR/body/draft readback pass. Its CI
snapshot57success/28skipped/22running/7queued has no failures, not a full CI pass.
All configured applicable pre-commit checks pass; summary bytes are unchanged.
TASK-13421 remains In Progress only for Stage3's viewport-only mobile approval.
