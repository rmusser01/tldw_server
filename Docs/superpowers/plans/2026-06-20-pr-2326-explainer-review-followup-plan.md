## Stage 1: Rebase And Route Integration
**Goal**: Rebase the Explainer PR branch onto latest `origin/dev` and adapt route wiring to the dev router-groups architecture.
**Success Criteria**: Branch is based on `origin/dev`; Explainer route is registered through router groups; stale direct `main.py` route wiring is absent.
**Tests**: OpenAPI verification and Explainer endpoint tests.
**Status**: Complete

## Stage 2: Backend Review Fixes
**Goal**: Resolve valid backend review findings around Explainer DB access, helper typing, job status offloading, grounding coercion, and Chatbook export hardening.
**Success Criteria**: Valid unresolved backend comments are addressed or have a recorded technical reason to skip.
**Tests**: Explainer/Chatbook pytest coverage plus Bandit on touched backend scope.
**Status**: Complete

## Stage 3: Frontend Review Fixes
**Goal**: Resolve valid frontend review findings around session hydration, query side effects, detail-panel option access, tests, and E2E request mocks.
**Success Criteria**: Valid unresolved frontend comments are addressed and targeted Vitest/Playwright coverage passes.
**Tests**: Explainer Vitest suite and Explainer Playwright E2E.
**Status**: Complete

## Stage 4: Backlog Metadata And PR Hygiene
**Goal**: Clean inconsistent Backlog task metadata/DoD markers and update PR tracking after rebase.
**Success Criteria**: Review comments on Backlog task records are addressed; TASK-2393 records final summary and verification.
**Tests**: Backlog file inspection and `git diff --check`.
**Status**: Complete

## Stage 5: Final Verification And Push
**Goal**: Run focused verification, update PR base to `dev`, force-push the rebased branch safely, and summarize resolved/skipped threads.
**Success Criteria**: Required checks pass locally; PR is based on `dev`; branch is pushed; unresolved comments are either fixed or documented.
**Tests**: Targeted pytest, Vitest, OpenAPI verification, Bandit, Playwright E2E, and GitHub PR checks as available.
**Status**: In Progress
