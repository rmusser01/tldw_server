# TASK-13144 — Encrypted Server Personal Context Repository

## Stage 1: Contract and RED evidence

**Goal**: Pin the shared contract and express key, crypto, schema, repository,
and durable-owner behavior as failing tests.

**Success Criteria**: Tests fail only because the new server modules and schema
do not exist.

**Tests**: `test_personal_context_contract.py`,
`test_personal_context_crypto.py`, `test_personal_context_key_custody.py`,
`test_personal_context_repository.py`, and
`test_personal_context_plaintext_canary.py`.

**Status**: Complete

## Stage 2: Existing database and cryptographic boundary

**Goal**: Add canonical tables and `BEGIN IMMEDIATE` transaction support to
`PersonalizationDB`, then implement explicit master-key custody and per-version
envelope encryption.

**Success Criteria**: Schema, key-locking, nonce, AAD, integrity, and rollback
tests pass without changing existing Personalization behavior.

**Tests**: New crypto/key tests plus existing
`test_personalization_endpoints.py`.

**Status**: Complete

## Stage 3: Canonical repository

**Goal**: Implement encrypted immutable versions, compare-and-set heads,
profile isolation, content-free tombstones/receipts, and bounded reads.

**Success Criteria**: Repository lifecycle and contract-conformance tests pass.

**Tests**: New contract/repository tests.

**Status**: Complete

## Stage 4: Privacy, regression, security, and review

**Goal**: Prove the no-plaintext boundary and complete repository quality gates.

**Success Criteria**: New and existing targeted tests, Ruff/format, compilation,
Bandit, diff checks, and independent review pass; TASK-13144 is complete.

**Tests**: New plaintext-canary suite and all touched Personalization tests.

**Status**: Complete
