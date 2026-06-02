## Stage 1: Shared Model Normalization Contract
**Goal**: Define a single helper contract for model metadata normalization and provider availability enrichment.
**Success Criteria**: A focused unit test exercises the shared helper with model metadata and provider listing payloads.
**Tests**: `apps/packages/ui/src/services/tldw/__tests__/model-normalization.test.ts`
**Status**: Complete

## Stage 2: Shared Chat Image Attachment Policy
**Goal**: Move chat attachment image MIME inference and data URL normalization into shared image utilities.
**Success Criteria**: Image utility tests cover extension-based inference and data URL MIME correction.
**Tests**: `apps/packages/ui/src/utils/__tests__/image-utils.test.ts`
**Status**: Complete

## Stage 3: Wire Call Sites
**Goal**: Update both Tldw client model paths and the composer attachment hook to use the shared helpers.
**Success Criteria**: The previous duplicated logic is removed from the call sites.
**Tests**: Existing `TldwApiClient getModels` and `useComposerAttachments` tests continue to pass.
**Status**: Complete

## Stage 4: Verification
**Goal**: Run targeted tests and touched-scope security checks, then record results in Backlog.
**Success Criteria**: Targeted frontend tests pass and Bandit is either run on Python touched scope or explicitly skipped as non-Python.
**Tests**: `bunx vitest run` for touched frontend tests.
**Status**: Complete
