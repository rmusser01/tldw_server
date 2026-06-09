# MCP Filesystem Lock Leases Implementation Plan

## Stage 1: Lock Manager And Tool Contract

**Goal**: Add process-local lock lease primitives and expose `fs.lock_acquire` / `fs.lock_release` in the filesystem tool catalog.

**Success Criteria**: Tool descriptors exist with strict schemas, `lock` path-action metadata, bounded TTL/owner fields, and no absolute path leakage.

**Tests**: Add failing filesystem module tests for descriptor presence, argument validation, acquire success, active conflict, renewal, expiry, release, wrong-token conflict, and path escape rejection.

**Status**: Not Started

## Stage 2: Mutation Validation

**Goal**: Add optional lock validation to `fs.edit`, `fs.patch`, and `fs.write` without replacing existing hash/read-receipt checks.

**Success Criteria**: Mutations accept `lock_lease_id`; when the module setting requires locks, missing or mismatched leases fail with stable reason codes before writing; valid leases allow the existing mutation paths to proceed.

**Tests**: Add failing tests for `fs.write replace` and `fs.patch` requiring active matching leases, including multi-file patch paths derived from the diff.

**Status**: Not Started

## Stage 3: Documentation, Backlog, And Verification

**Goal**: Document process-local limits and record verification.

**Success Criteria**: The package user guide mentions advisory lock usage and limitations; TASK-2300 records implementation notes, test output, Bandit output, and known future shared-store follow-up.

**Tests**: Run focused filesystem MCP tests, compile touched Python files, Bandit on touched Python implementation scope, and `git diff --check`.

**Status**: Not Started
