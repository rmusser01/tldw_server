# TASK-13629: Current remote dev integration

ADR required: no new rule; preserve existing H1 ownership/admission, ADR067 provider accounting, ADR059 task records and proposed ADR069 API contract ownership while integrating remote quota enforcement.

## Stage 1: Map current dev and conflicts
**Goal**: Pin current remote dev and identify why required PR workflows are absent.
**Success Criteria**: Verify conflicting PR and remote79aa6f00; enumerate semantic overlap.
**Tests**: GitHub mergeability, merge-tree/index inspection and remote diff.
**Status**: Complete

## Stage 2: Resolve and qualify
**Goal**: Preserve both remote LLM quota checks and local source/lease/recovery safety.
**Success Criteria**: Relevant backend suites, client types and API contract qualify; security scope has no new findings.
**Tests**: Focused quota/chat/profile suites, compiler/API generator and scoped Bandit.
**Status**: Complete

## Stage 3: Review and restore CI
**Goal**: Review resolved overlap, run normal hooks and publish an integrated head.
**Success Criteria**: PR is mergeable and required workflows can launch; remaining work explicit.
**Tests**: Independent bounded reviews, hooks and exact-head GitHub state.
**Status**: In Progress
