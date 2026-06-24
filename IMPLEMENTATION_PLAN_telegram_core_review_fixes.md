## Stage 1: Regression Tests
**Goal**: Capture the reviewed Telegram core risks before production edits.
**Success Criteria**: Tests fail for cross-scope pairing-code consumption, plaintext code storage, and unsafe session mapper inputs.
**Tests**: Targeted Telegram runtime/session mapper pytest tests.
**Status**: Complete

## Stage 2: Runtime Repository Fixes
**Goal**: Scope-bind pairing-code consumption and hash newly stored pairing codes.
**Success Criteria**: Pairing codes are consumed atomically by scope and no new raw code is persisted.
**Tests**: Runtime repo tests pass for SQLite path and endpoint link flow uses scoped consumption.
**Status**: Complete

## Stage 3: Session Mapper Hardening
**Goal**: Make session/conversation identity derivation canonical and reject unsafe component values.
**Success Criteria**: Public helpers accept scalar IDs, reject non-scalars, and preserve existing stable outputs for current supported Telegram IDs where required.
**Tests**: Session mapper tests cover accepted scalars, rejected containers/objects, and delimiter-safe UUID derivation.
**Status**: Complete

## Stage 4: Verification
**Goal**: Verify targeted behavior and security checks for touched code.
**Success Criteria**: Targeted tests and Bandit complete with no new findings in touched scope.
**Tests**: `python -m pytest ...` and `python -m bandit ...`.
**Status**: Complete
