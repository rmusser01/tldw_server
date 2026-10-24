# X01: API key and provider-secret initialization under the serving role

Task: TASK-13260.278.18.83.45. Candidate ec3 on dev4e; original session protected.

Root: actual profile GET500 invokes lazy APIKeyManager initialization, which executes PostgreSQL DDL under a nonowner. Keep the failure visible and retain all failed results. No new model run merely for diagnosis.

ADR assessment: no new ADR. Reuse the current AuthNZ owner-migration/read-only serving-admission pattern. No credential, persistence or deployment policy change. SQLite and explicit owner bootstrap remain unchanged.

## Stage 1: Confirm the real role failure
**Goal**: Add APIKeyManager initialization to the official migrated nonowner regression.
**Success Criteria**: Real PostgreSQL causal red with the demonstrated privilege failure; ownership and schema/data snapshots preserved.
**Tests**: Only the new surface, normal fixtures/plugins and600s/300s budgets.
**Status**: Complete

## Stage 2: Minimal complete admission repair
**Goal**: Nonowners validate the API key and audit schema without DDL; owners retain normal provisioning.
**Success Criteria**: Valid schema serves; missing/drifted columns/types/keys/indexes refuse without mutation; single-slot pool remains safe.
**Tests**: Actual nonowner positive/drift controls, connected API key repository/manager/owner controls, compile/lint/Bandit and independent review.
**Status**: Complete

## Stage 3: Package and finish the live X01 checks
**Goal**: Qualified isolated candidate and actual profile/Settings recovery, followed by remaining X01 cells/variants.
**Success Criteria**: Package bindings preserved; live profile and Settings work; original session restored;15s cleanup; concise task outcomes; remove only this owned plan after completion.
**Tests**: Reviewed no-generation diagnosis first; only genuinely affected live validation afterward. No publication or X02 before X01 closure.
**Status**: In Progress

## Current evidence

Real nonowner causal privilege failure confirmed; initialization/read positive passed. Twelve malformed-schema refusals passed without mutation. Two reviewer-found type edges reproduced red, then both passed with supported JSONB lists and actual fractional virtual-key write. Three-file compile/Ruff and scoped baseline/current Bandit clear. Connected owner/SQLite/rotation/audit checks and final review remain in progress. No new live or package claim.

Connected ten modules:88passed3208warnings23.78s pytest/28.139s outer. Constrained async pool max5/hold4 real connections:1passed5warnings9.17s pytest/13.103s outer,15s inner bound; initial synchronous max1 setting was not async-pool evidence. One edit-guard mistake caused an unchanged positive replay, not additional qualification. Source review CLEAR. Nine package guards pass;3816inputs admitted for fresh isolated api-keys site. Packaging and actual live profile/Settings/lifecycle check pending; no frontend change/build needed.

## Downstream provider-secret admission

Installed f4db API-key repair is qualified3816files/7.873s. Actual no-generation Settings check now passes that initialization, then fails at user_provider_secrets request-time owner DDL. Extend the same read-only serving boundary to this repository; existing profile-overrides readiness already read-only. Real causal regression before product edits, finite schema/refusal/write checks and normal owner/SQLite behavior retained. API0.246s/frontend0.006s actual cleanup within15s, original fullcanonical/Chat/draft/owners preserved. Whole live Settings still failed; no X01 closure.

Provider-secret causal/green:1failed then1passed real restricted role, onefree asyncslot and actual synthetic-ciphertext write/isolation/revocation; eleven drift refusals pass. Reviewer found coexisting deferrable conflict arbiter (including reversed key order):2red then2green; minimal explicit refusal added. Connected4modules49passed255warnings32.87s pytest/37.086s outer. Compile/Ruff and scoped baseline/current Bandit clear; source review CLEAR. This qualifies repository persistence, not encryption or liveprofile. All8runtime pins and fresh packager target align;3816inputs admitted. Stage3 remains open until actual package/Settings/release acceptance.
