## Stage 1: Contract Tests
**Goal**: Add failing tests for structured runtime discovery reason details.
**Success Criteria**: Focused tests fail because current runtime discovery lacks `normalized_reason_details` and metadata helpers.
**Tests**: `pytest` targets for runtime inventory, runtime capability gate, and sandbox public docs contract.
**Status**: Complete

## Stage 2: Metadata Catalog
**Goal**: Centralize `RuntimeReasonDetails` metadata for every `RuntimeReasonCode`.
**Success Criteria**: Import-time validation catches missing, extra, or mismatched runtime reason metadata.
**Tests**: Runtime inventory contract tests cover catalog completeness and unknown-code fallback.
**Status**: Complete

## Stage 3: API Projection
**Goal**: Expose additive reason details on runtime discovery and admin diagnostics.
**Success Criteria**: Existing raw `reasons` and `normalized_reasons` remain unchanged while details are populated from the shared catalog.
**Tests**: Feature discovery and admin diagnostics schema/projection tests pass.
**Status**: Complete

## Stage 4: Documentation And Verification
**Goal**: Update the runtime capability inventory and run focused validation.
**Success Criteria**: Docs describe the new reason-details contract and no longer list it as a current gap; focused tests, compile check, Bandit, and diff whitespace checks pass.
**Tests**: Runtime capability docs contract test plus verification commands in TASK-123.
**Status**: Complete
