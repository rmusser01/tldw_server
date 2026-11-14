# Load reviewed UAT client fixes

Task: TASK-13260.278.18.83.45, declared before build/runtime changes.
ADR required: no. Use existing normal production build, token/bundle checks and existing owned client/runtime/profile. No security or configuration changes.
Scope: new reviewed evaluation permission guidance, shared tooltip forwarding, truthful Media count and zero-visible multi-page continuation. Current source/dependencies/auth/data remain bound; no earlier UAT acceptance is inferred.

## Stage 1: Freeze reviewed source
**Goal**: Independent connected-source reviews and configured types pass.
**Success Criteria**: Corrected source reviews clear; post-change UI/Web types0errors and hashes match.
**Tests**: Completed normal affected tests, syntax/lint and normal UI/Web tsconfig.
**Status**: Complete

## Stage 2: Build one fresh client artifact
**Goal**: Normal production-profile webpack build with checks on the same actual directory.
**Success Criteria**: Natural0 within600outer, source inputs unchanged, token/bundle checks pass, API rewrites18903.
**Tests**: Official build-with-profile, verify-shared-token-sync and check-bundle-budget helpers; installed tools only.
**Status**: Complete

## Stage 3: Guarded load and live acceptance
**Goal**: Later replace only the exact existing owned multi-user frontend after a qualified build.
**Success Criteria**: Fresh owner/cwd/listener guards,15s graceful stop/no force,ONE120s startup; current build verified and existing tabs reused.
**Tests**: Root declares exact runtime/live actions before execution. No API restart, ACP allowlist change, permissions grant or extra model/action retry is implied.
**Status**: In Progress

Qualified fresh build TkYU_JmF65WnU2F2eimMA: natural0/114.589s,156pages;6888inputs unchanged, routes-reference-only next-env restored; official same-directory token/bundle checks passed (591.4KiB shared,791.5KiB heaviest),API18903rewrites. Exact owned replacement/live actions declared in associated task; no API/security/private-config changes.
