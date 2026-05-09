## Stage 1: Frontend Contract
**Goal**: Add typed WebUI support for `GET /api/v1/vn-play/setup-options`.
**Success Criteria**: `listVNPlaySetupOptions()` serializes query parameters and returns the backend setup response shape.
**Tests**: `apps/tldw-frontend/__tests__/vn-play/vnPlayApi.test.ts`.
**Status**: Complete

## Stage 2: Dialog Integration
**Goal**: Move `NewSessionDialog` from client-side character, pack, and readiness fan-out to the backend setup-options contract.
**Success Criteria**: The dialog renders backend-derived defaults, labels, warnings, compatibility, trust, and empty states while preserving the existing create-session payload.
**Tests**: `apps/tldw-frontend/__tests__/vn-play/VNPlayWorkspace.test.tsx`.
**Status**: Complete

## Stage 3: Smoke Coverage
**Goal**: Keep the VN Play smoke route aligned with the setup-options API.
**Success Criteria**: The smoke test mocks `/vn-play/setup-options` and no longer needs setup-only character/pack/readiness routes.
**Tests**: `apps/tldw-frontend/e2e/smoke/vn-play.spec.ts`.
**Status**: Complete

## Stage 4: Verification
**Goal**: Validate the focused WebUI slice and record results.
**Success Criteria**: Focused unit tests, lint/diff checks, and applicable security checks complete or have documented skips.
**Tests**: Focused Vitest, ESLint, `git diff --check`, and Bandit only if backend Python files are touched.
**Status**: Complete
