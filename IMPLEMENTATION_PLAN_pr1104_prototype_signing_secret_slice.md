## Stage 1: Verify Signing Secret Review Finding
**Goal**: Confirm preview/access token services currently fall back to a per-process random signing secret.
**Success Criteria**: Focused tests fail because services instantiate without a configured stable signing secret.
**Tests**: Focused pytest tests for `PrototypePreviewBroker` and `PrototypeAccessService`.
**Status**: Complete

Notes:
- Focused red run failed both new tests because the services did not raise when all configured signing-secret sources were absent.

## Stage 2: Require Stable Signing Secrets
**Goal**: Remove process-local random fallback signing secrets while preserving explicit constructor injection and existing JWT/API-key configuration fallbacks.
**Success Criteria**: Preview and access services resolve only explicit/configured secrets and raise a clear error when none are available.
**Tests**: Focused pytest and full `PrototypeWorkspaces` regression.
**Status**: Complete

Notes:
- `PrototypePreviewBroker` now resolves only an explicit secret, `PROTOTYPE_PREVIEW_SIGNING_SECRET`, `JWT_SECRET_KEY`, or `SINGLE_USER_API_KEY`.
- `PrototypeAccessService` now resolves only an explicit secret, `JWT_SECRET_KEY`, or `SINGLE_USER_API_KEY`.
- Both services raise a clear `RuntimeError` instead of creating a per-process random fallback secret when no stable source exists.

## Stage 3: Verify and Publish
**Goal**: Run focused/regression tests, Bandit on touched backend code, diff checks, then push/reply to relevant PR threads.
**Success Criteria**: Local verification passes and stable-signing-secret review threads have replies.
**Tests**: Focused pytest, full PrototypeWorkspaces pytest, Bandit, `git diff --check`.
**Status**: Complete

Notes:
- Focused signing-secret tests passed.
- Full `tldw_Server_API/tests/PrototypeWorkspaces` regression passed with 65 tests.
- Ruff passed for touched signing/test files.
- Bandit passed with 0 results on `preview_broker.py` and `access.py`.
- `git diff --check` passed.
