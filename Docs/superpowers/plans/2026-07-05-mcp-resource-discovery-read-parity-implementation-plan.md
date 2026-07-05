# MCP Resource Discovery and Read Parity Implementation Plan

## Stage 1: External Runtime Resource Tests
**Goal**: Prove external resources list/read through the runtime manager and adapter.
**Success Criteria**: Tests cover redacted virtual resource descriptors, read routing, and stopped/missing errors.
**Tests**: `test_gateway_external_runtime.py`, `test_gateway_fastapi_package.py`
**Status**: Complete

## Stage 2: Transport Resource Methods
**Goal**: Add minimal `list_resources` and `read_resource` support to fake and stdio transports.
**Success Criteria**: Stdio transport sends MCP `resources/list` and `resources/read` and normalizes malformed upstream data safely.
**Tests**: Existing stdio smoke fixture or focused unit tests.
**Status**: Complete

## Stage 3: Runtime Integration
**Goal**: Merge base and external resources in the gateway adapter.
**Success Criteria**: External resources use virtual URIs, raw upstream URIs stay private, and reads route by virtual URI.
**Tests**: Adapter and manager tests pass.
**Status**: Complete

## Stage 4: Bounded Readiness
**Goal**: Add a small wait helper over existing runtime status.
**Success Criteria**: Wait returns ready/unavailable server ids without spawning a monitor.
**Tests**: Focused runtime test.
**Status**: Complete

## Stage 5: Verification
**Goal**: Verify the touched scope and record results.
**Success Criteria**: Focused pytest, Bandit on touched code, and `git diff --check` are clean or documented.
**Tests**: Focused commands only.
**Status**: Complete
