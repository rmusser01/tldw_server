# TASK-13013.7.4 — PyJWT migration

Approved scope: preserve existing JWT consumers and token contracts while removing
python-jose and ecdsa. No Chroma changes, vulnerability exceptions, or shared
environment changes. Tracking: TASK-13013.7.4; PR #2869.

## Stage 1: Characterize the existing contract
**Goal**: Protect legacy tokens, claim errors, and OIDC dictionary keys.
**Success Criteria**: Regression tests fail before the migration and pass after it.
**Tests**: Legacy signed token, invalid claims, expiry, algorithm allowlist, RSA/EC JWKs.
**Status**: Complete

## Stage 2: Migrate consumers and locked runtime evidence
**Goal**: Use the existing PyJWT dependency across all six consumers.
**Success Criteria**: Resolver removes python-jose/ecdsa; runtime probe uses PyJWT.
**Tests**: Consumer suites, locked graph regression, crypto runtime integration.
**Status**: Complete

## Stage 3: Verify and publish
**Goal**: Validate the candidate without weakening release gates.
**Success Criteria**: Locked-version regressions, Bandit, source SBOM, independent review recorded.
**Tests**: Focused authentication suites, supply-chain tests, security scan.
**Status**: In Progress
