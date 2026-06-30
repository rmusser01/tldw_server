## Stage 1: Archive Preview Diagnostics
**Goal**: Extend Persona Visual archive preview so Manifest V2 renderer metadata is routed through the existing renderer import-preview validator instead of failing as a malformed V1 visual manifest.
**Success Criteria**: Disabled known renderers and unknown renderers return structured renderer diagnostics, blockers, normalized role categories, and non-activation/non-commit eligibility.
**Tests**: Add focused previewer tests for disabled `live2d`, unknown renderer, missing required role category, and existing V1 regression coverage.
**Status**: Complete

## Stage 2: Preview Status Wiring
**Goal**: Preserve V1 completed preview behavior while storing V2 renderer-blocked previews as non-committable review results.
**Success Criteria**: Background import-preview jobs complete validation, but blocked preview rows are not accepted by the import-commit path.
**Tests**: Focused unit coverage plus existing Persona visual portability tests.
**Status**: Complete

## Stage 3: Documentation And Tracker Closeout
**Goal**: Document the V2 preview behavior and explicitly keep this slice out of runtime activation, MCP provider execution, frontend changes, and VN/CYOA code.
**Success Criteria**: Docs and Backlog task note the backend-only preview boundary and verification results.
**Tests**: Focused pytest and Bandit on touched Python scope.
**Status**: Complete
