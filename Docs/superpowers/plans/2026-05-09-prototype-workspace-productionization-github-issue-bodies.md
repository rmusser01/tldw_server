# Prototype Workspace Productionization GitHub Issue Bodies

Source spec: Docs/superpowers/specs/2026-05-09-prototype-workspace-productionization-issue-tree-design.md
Parent tracker: https://github.com/rmusser01/tldw_server/issues/1440

These issue bodies are drafts. Do not create GitHub issues until this file is reviewed.

[Risk Gate 1][Split] Prototype collaboration threat model and authorization invariants

## Parent

Tracks part of #1440.

## Risk Being Burned Down

Private links, external actors, preview grants, promotion authority, and resume/session tokens can become confused or over-permissive. This risk must be reduced before persistence, runtime, API, or frontend work can safely build on the MVP.

## Owner Lane

Owner lane: Split

Backend/Core responsibilities: threat model, authorization invariants, token/session requirements, audit/quota policy, backend tests

Frontend/Product responsibilities: contract fixture schema, mock states, route-state audit checklist, and contract feedback deadline

Reviewer lane: Frontend/Product reviews contract usability; Backend/Core reviews frontend prep for contract adherence

## Dependencies

- Depends on: none
- Blocks: Risk Gates 2, 3, 4, and 5

## Scope

- [ ] Define actor identities: owner, internal collaborator, designated promoter, and external shared actor.
- [ ] Define authorization invariants for token creation, public exchange, collaborator session creation, preview grant renewal, and promotion review.
- [ ] Define non-enumerating error behavior for invalid, expired, revoked, exhausted, archived, and mismatched links.
- [ ] Define token and session security requirements: token storage/hash rules, TTLs, replay handling, cookie flags, referrer leakage controls, password-protected link behavior, signing secret rotation, and revocation propagation. Each requirement must be marked `enforce now`, `document existing behavior`, or `defer to Risk Gate N`.
- [ ] Define audit event taxonomy for token creation/exchange, actor creation, preview grant issuance/renewal/revocation, promotion review, and authorization failures.
- [ ] Define rate/quota policy for link exchange, collaborator session creation, preview renewal, and runtime job creation.
- [ ] Produce a draft contract matrix for frontend use: state, backend condition, HTTP status, stable error category, user-facing state bucket, and retryability.
- [ ] Frontend/Product creates preparatory contract fixtures, mock states, a route-state audit checklist, and a contract feedback deadline for Risk Gate 4.
- [ ] Add focused backend tests for the highest-risk invariants that already exist in the MVP code.

## Non-Goals

- Implementing all audit dashboards or operational visibility.
- Implementing all quota enforcement paths if those depend on later runtime/job changes.
- Freezing every OpenAPI schema. Risk Gate 4 owns the final contract freeze.

## Acceptance Criteria

- [ ] Actor identities and authorization invariants are documented.
- [ ] Token/session handling requirements are documented, and each item has an explicit disposition: `enforce now`, `document existing behavior`, or `defer to Risk Gate N`.
- [ ] Revocation, expiration, ownership, and cross-workspace isolation have explicit expected behavior.
- [ ] A draft frontend-consumable contract matrix exists.
- [ ] Frontend/Product has fixture and mock-state preparation work ready to unblock parallel implementation.
- [ ] Focused backend tests cover core ownership, revocation, expiration, and cross-workspace isolation behavior.
- [ ] Any audit/quota behavior deferred to later gates is explicitly listed.

## Verification

- Backend tests: prototype workspace endpoint, sharing exchange, access service, and service authorization tests relevant to auth invariants.
- Frontend tests: none required, but Frontend/Product reviews the draft contract matrix.
- Security checks: Bandit only if backend code changes occur.
- Manual/browser checks: not required.

[Risk Gate 2][Backend/Core] Prototype workspace persistence and transaction hardening

## Parent

Tracks part of #1440.

## Risk Being Burned Down

Partial writes, stale pointers, cleanup gaps, or SQLite/PostgreSQL drift could corrupt collaboration state or make promotion decisions unsafe.

## Owner Lane

Owner lane: Backend/Core

Reviewer lane: Backend/Core peer or maintainer

## Dependencies

- Depends on: Risk Gate 1
- Blocks: Risk Gates 3, 4, and 8

## Scope

- [ ] Review repository invariants for workspaces, snapshots, sessions, shared actors, promotion requests, and preview handles.
- [ ] Strengthen transaction and compensation behavior for multi-step service operations.
- [ ] Verify SQLite and PostgreSQL behavior for migrations, table discovery, placeholder conversion, constraints, and indexes.
- [ ] Verify query plans or index coverage for workspace detail, session lookup, active actor lookup, promotion listings, and preview-handle lookup.
- [ ] Define cleanup and retention behavior for archived workspaces, revoked actors, expired sessions, stale promotion requests, and preview handles.
- [ ] Add tests for failed intermediate writes and compensation behavior.

## Non-Goals

- Moving all prototype persistence into a new DB_Management abstraction unless the review proves the existing AuthNZ repo pattern is insufficient.
- Redesigning the full workspace persistence model.
- Implementing runtime job durability, which belongs to Risk Gate 3.

## Acceptance Criteria

- [ ] Repo/service tests cover failed intermediate writes and compensation.
- [ ] SQLite and PostgreSQL behavior is documented and covered where the project supports both.
- [ ] Cleanup and retention rules are documented and tested for the prototype tables.
- [ ] Index/query-plan review is recorded with any necessary changes.

## Verification

- Backend tests: prototype repo, service compensation, migrations, and persistence integration tests.
- Frontend tests: none required.
- Security checks: Bandit only if backend code changes occur.
- Manual/browser checks: not required.

[Risk Gate 3][Backend/Core] Runtime jobs and preview lifecycle durability

## Parent

Tracks part of #1440.

## Risk Being Burned Down

Runtime bootstrap, preview handles, and promotion validation may fail under retries, restarts, cancellation, or multi-worker deployment. These failures can expose stale previews, advance the wrong canonical pointer, or leave collaborator sessions stuck.

## Owner Lane

Owner lane: Backend/Core

Reviewer lane: Frontend/Product for user-visible runtime state needs

## Dependencies

- Depends on: Risk Gates 1 and 2
- Blocks: Risk Gates 4, 6, 7, and 8

## Scope

- [ ] Move prototype runtime orchestration toward the shared JobManager pattern where appropriate.
- [ ] Apply the Jobs/Scheduler decisions from this spec: branch bootstrap, preview boot/restart/replacement, snapshot save, publish validation, and canonical promotion should be Jobs-backed unless an issue records a specific exception.
- [ ] Define retry, cancellation, timeout, idempotency, and cleanup behavior for branch session bootstrap, preview boot, snapshot save, and publish-validate-promote jobs.
- [ ] Harden preview handle lifecycle across persistent lookup, active-handle replacement, rollback, revocation, cache refresh, and renewal.
- [ ] Define and implement the runtime bootstrap status, preview health, and promotion validation failure surfaces needed by frontend and operators.
- [ ] Confirm failed publish validation never advances the canonical pointer.

## Non-Goals

- Providing stable long-lived production hosting.
- Building a full deployment platform for prototype previews.
- Solving future merge-heavy collaboration or continuous hosting workflows.
- Exposing owner secrets or production-like environment parity to external collaborators.

## Acceptance Criteria

- [ ] Prototype jobs are idempotent and safe to retry.
- [ ] Cancellation, timeout, and cleanup semantics are documented and tested.
- [ ] Preview revocation and renewal behavior is covered by backend tests.
- [ ] Failed publish validation never advances the canonical pointer.
- [ ] Runtime and preview status fields needed by later frontend work are implemented or explicitly documented for the contract freeze.

## Verification

- Backend tests: runtime jobs, preview broker, promotion service, and service authorization tests.
- Frontend tests: none required, but Frontend/Product reviews runtime status needs.
- Security checks: Bandit on touched backend paths.
- Manual/browser checks: not required.

[Risk Gate 4][Backend/Core] Backend API contract and error semantics freeze

## Parent

Tracks part of #1440.

## Risk Being Burned Down

Frontend and backend can drift around response shapes, error semantics, retryability, and lifecycle assumptions. This gate converts the draft matrix from Risk Gate 1 into a frozen contract.

## Owner Lane

Owner lane: Backend/Core

Reviewer lane: Frontend/Product

## Dependencies

- Depends on: Risk Gates 1, 2, and 3
- Blocks: Risk Gates 5, 6, 7, and 8

## Scope

- [ ] Freeze owner and collaborator endpoint response models.
- [ ] Freeze stable error categories for expired, revoked, archived, exhausted, stale, conflict, unauthorized, missing, bootstrap-failed, preview-unavailable, and failed-promotion states.
- [ ] Finalize the contract matrix: state, backend condition, HTTP status, stable error category, frontend state bucket, retryability, and suggested user-facing handling.
- [ ] Document lifecycle examples: create workspace, create share link, exchange public link, create collaborator session, renew preview grant, submit promotion, review promotion.
- [ ] Document configuration requirements for signing secrets, runtime policy profiles, quotas, and jobs.
- [ ] Add migration and rollback notes for deployments enabling prototype workspace collaboration.

## Non-Goals

- Implementing all frontend states. Risk Gates 5 and 6 own frontend consumption.
- Adding broad unrelated OpenAPI cleanup.
- Reworking feature semantics already settled by Risk Gates 1 through 3 unless a contract review finds a blocking issue.

## Acceptance Criteria

- [ ] OpenAPI and API docs match implemented behavior.
- [ ] The contract matrix is complete enough for frontend implementation without guessing.
- [ ] Frontend/Product signs off that the contract covers collaborator entry and owner promotion UX needs.
- [ ] Migration and rollback notes exist.

## Verification

- Backend tests: endpoint response model and error semantics tests.
- Frontend tests: contract fixture validation where practical.
- Security checks: Bandit if backend endpoint/service code changes occur.
- Manual/browser checks: not required.

[Risk Gate 5][Frontend/Product] Collaborator entry and route-state safety

## Parent

Tracks part of #1440.

## Risk Being Burned Down

Public-share entry can accidentally reuse stale owner workspace state or present an unsafe/incorrect collaborator context. This is the highest-risk frontend path because it crosses unauthenticated/private-link entry, external actor identity, and runtime session creation.

## Owner Lane

Owner lane: Frontend/Product

Reviewer lane: Backend/Core for contract adherence

## Dependencies

- Depends on: Risk Gates 1 and 4
- Blocks: Risk Gates 6 and 8

## Scope

- [ ] Harden public-share password handoff into prototype collaborator session exchange.
- [ ] Ensure collaborator entry never falls back to stale owner workspace state.
- [ ] Tighten query keys, disabled queries, missing workspace/session handling, and local state reset behavior.
- [ ] Add UI states for revoked, expired, archived, exhausted, mismatched-token, bootstrap-failed, and preview-unavailable flows.
- [ ] Build against contract fixtures from Risk Gate 4.

## Non-Goals

- Owner promotion review UX, which belongs to Risk Gate 6.
- Changing backend error semantics outside the frozen contract. Missing needs should be fed back as contract issues.
- Polishing all product copy beyond what is needed for correct collaborator entry behavior.

## Acceptance Criteria

- [ ] Frontend tests cover public link exchange and route-state isolation.
- [ ] Collaborator session context is explicit and is not inferred from stale local owner state.
- [ ] User-facing states map to backend error categories from the contract matrix.
- [ ] The public-share handoff is covered with at least one browser-observed or E2E path by the final release gate.

## Verification

- Backend tests: none required unless contract gaps require backend changes.
- Frontend tests: route, hook, client, and component tests for collaborator entry.
- Security checks: not required unless backend code changes occur.
- Manual/browser checks: targeted public-share entry check if practical before Risk Gate 8.

[Risk Gate 6][Frontend/Product] Owner review and promotion UX hardening

## Parent

Tracks part of #1440.

## Risk Being Burned Down

Owners may make unsafe promotion decisions if branch, preview, validation, and review states are unclear. Promotion UX must distinguish review state from runtime state and must not offer actions that backend authority or validation would reject.

## Owner Lane

Owner lane: Frontend/Product

Reviewer lane: Backend/Core for authority and validation semantics

## Dependencies

- Depends on: Risk Gates 3, 4, and 5
- Blocks: Risk Gate 8

## Scope

- [ ] Harden pending, approved, rejected, promoted, stale, conflict, validation-running, validation-failed, and promotion-failed states.
- [ ] Make branch/session inventory readable enough for owner review decisions.
- [ ] Align promotion actions with backend authority and validation semantics.
- [ ] Add owner and collaborator copy for promotion request and review outcomes.
- [ ] Ensure stale/conflict states are not hidden behind generic failure UI.

## Non-Goals

- Full git-style merge/rebase UI.
- Live multi-user canonical filesystem editing.
- Redesigning the entire prototype workspace product surface.

## Acceptance Criteria

- [ ] Owner UI distinguishes review state from runtime state.
- [ ] Promotion actions are unavailable when backend authority or validation rules would reject them.
- [ ] Frontend tests cover the main review-state transitions.
- [ ] Backend/Core confirms the UI state mapping matches contract semantics.

## Verification

- Backend tests: none required unless contract gaps require backend changes.
- Frontend tests: owner view, promotion flow, branch/session inventory, and action availability tests.
- Security checks: not required unless backend code changes occur.
- Manual/browser checks: targeted owner review flow check if practical before Risk Gate 8.

[Risk Gate 7][Split] Operational visibility and documentation

## Parent

Tracks part of #1440.

## Risk Being Burned Down

Operators and users cannot diagnose runtime, preview, sharing, or promotion failures. Production readiness requires enough visibility and documentation to support failures without reading implementation internals.

## Owner Lane

Owner lane: Split

Backend/Core responsibilities: operational status, audit/config/runtime docs

Frontend/Product responsibilities: owner/collaborator workflow docs and user-facing examples

## Dependencies

- Depends on: Risk Gates 3 and 4
- Blocks: Risk Gate 8

## Scope

- [ ] Document runtime bootstrap, preview health, signing secrets, quotas, and job behavior.
- [ ] Document and lightly wire the operational status surfaces defined or implemented in Risk Gate 3.
- [ ] Document owner and collaborator workflows end to end.
- [ ] Add examples for password-protected links, single-use links, resume cookies, revoked links, archived workspaces, exhausted links, and promotion conflicts.
- [ ] Identify audit events and status fields available for support.

## Non-Goals

- Building a full admin dashboard unless required status surfaces already exist and only need wiring.
- Rewriting broad sharing, jobs, or workspace documentation.
- Replacing Risk Gate 8 release evidence.

## Acceptance Criteria

- [ ] Operator docs explain setup, configuration, and failure diagnosis.
- [ ] Product docs explain owner and collaborator lifecycle without relying on implementation internals.
- [ ] Observability fields/events are documented and covered by tests where practical.
- [ ] Split ownership is reflected in the issue checklist so each implementer has clear deliverables.

## Verification

- Backend tests: status/audit/config tests where code changes occur.
- Frontend tests: docs-only changes do not require tests; UI status wiring does.
- Security checks: Bandit if backend code changes occur.
- Manual/browser checks: docs review and any new status UI check.

[Risk Gate 8][Split] End-to-end release gate and production readiness review

## Parent

Tracks part of #1440.

## Risk Being Burned Down

The feature can look complete in isolated tests while failing as a full owner-to-collaborator-to-promotion workflow. The final gate must collect evidence across backend, frontend, security, and browser-observed behavior.

## Owner Lane

Owner lane: Split

Backend/Core responsibilities: backend test matrix, security checks, runtime/API evidence

Frontend/Product responsibilities: frontend test matrix, browser-observed UX evidence, workflow evidence

## Dependencies

- Depends on: Risk Gates 1 through 7
- Blocks: production-ready claim for #1440

## Scope

- [ ] Create focused backend and frontend test matrices.
- [ ] Add CI-friendly smoke coverage for the full prototype path without requiring external runtime services.
- [ ] The minimum smoke path is: owner creates workspace, owner creates share link, collaborator exchanges link, collaborator creates branch session against a runtime/preview stub, collaborator submits promotion request, owner reviews request, publish validation failure is handled without advancing canonical state, publish validation success advances canonical state.
- [ ] The minimum negative security smoke path is: an expired or revoked prototype share link fails exchange without confirming whether the token, workspace, or actor exists.
- [ ] Run Bandit/security checks on touched backend paths.
- [ ] Complete a final browser-observed UX pass.
- [ ] Complete a final security review of private-link/session-token flows.
- [ ] Record remaining known risks and either close them explicitly or file follow-up issues.

## Non-Goals

- Adding new feature scope not already covered by Risk Gates 1 through 7.
- Reopening major product direction decisions unless the evidence shows a blocker.
- Claiming production readiness without recorded verification evidence.

## Acceptance Criteria

- [ ] Backend, frontend, smoke, and security verification results are recorded.
- [ ] Browser-observed owner and collaborator flows are recorded.
- [ ] The production-readiness decision is evidence-backed.
- [ ] Remaining work, if any, is explicitly triaged rather than folded into done.

## Verification

- Backend tests: focused backend test matrix and CI-friendly smoke coverage for the full prototype path without requiring external runtime services.
- Frontend tests: focused frontend test matrix and CI-friendly smoke coverage for the full prototype path without requiring external runtime services.
- Security checks: Bandit/security checks on touched backend paths and final security review of private-link/session-token flows.
- Manual/browser checks: final browser-observed UX pass covering owner and collaborator flows.

## Creation Checklist

- [x] Reviewed issue-body draft file.
- [x] Created Risk Gate 1 issue: https://github.com/rmusser01/tldw_server/issues/1453
- [x] Created Risk Gate 2 issue: https://github.com/rmusser01/tldw_server/issues/1454
- [x] Created Risk Gate 3 issue: https://github.com/rmusser01/tldw_server/issues/1455
- [x] Created Risk Gate 4 issue: https://github.com/rmusser01/tldw_server/issues/1456
- [x] Created Risk Gate 5 issue: https://github.com/rmusser01/tldw_server/issues/1457
- [x] Created Risk Gate 6 issue: https://github.com/rmusser01/tldw_server/issues/1458
- [x] Created Risk Gate 7 issue: https://github.com/rmusser01/tldw_server/issues/1460
- [x] Created Risk Gate 8 issue: https://github.com/rmusser01/tldw_server/issues/1461
- [x] Posted summary comment on #1440 with all child issue URLs: https://github.com/rmusser01/tldw_server/issues/1440#issuecomment-4413757329
