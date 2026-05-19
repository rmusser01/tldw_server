# Prototype Workspace Productionization Issue Tree Design

## Summary

PR #1104 shipped the prototype workspace collaboration MVP. GitHub issue #1440 now tracks the follow-up work needed to turn that MVP into a production-grade collaboration feature.

This design defines a staged sub-issue tree for #1440. The tree is optimized for risk burn-down rather than simple workstream grouping. It is intended for two implementers:

- Backend/Core: AuthNZ, persistence, services, runtime jobs, preview lifecycle, API contracts, backend verification, and security checks.
- Frontend/Product: owner/collaborator UX, public share handoff, route-state safety, frontend contract tests, browser-observed UX verification, and user-facing documentation.

No GitHub sub-issues should be created until this issue-tree design is reviewed and accepted.

## Context

The merged MVP established the first vertical slice:

- prototype workspace metadata
- branch sessions
- preview handles
- private-link sharing
- external collaborator access
- promotion/review flow
- owner/collaborator API endpoints
- frontend route and screens
- initial backend and frontend regression coverage

The post-MVP tracker must keep the feature moving toward production readiness without allowing the two implementation lanes to drift. Security, runtime durability, and data invariants can invalidate later UX and documentation work, so they come first.

## Design Goals

- Break #1440 into reviewable sub-issues with clear dependency gates.
- Burn down the highest-risk assumptions first: authorization, persistence, runtime durability, and API semantics.
- Give Backend/Core and Frontend/Product enough independent work to proceed in parallel once contracts are available.
- Prevent frontend work from relying on unstated backend behavior.
- Prevent backend hardening from expanding into long-lived production hosting.
- Ensure the final release gate records evidence instead of relying on vague readiness claims.

## Non-Goals

- Creating GitHub sub-issues in this step.
- Designing future merge-heavy collaboration beyond the MVP promotion model.
- Promising stable long-lived hosting for prototype previews.
- Replacing ACP, Sandbox, or Jobs as infrastructure primitives.
- Refactoring unrelated workspace, sharing, or jobs systems outside what the prototype collaboration feature needs.

## Coordination Model For Two Implementers

The staged tracker should be risk-gated, but the two implementers should not be forced into strict serial work.

Backend/Core owns the early invariants and contract-producing work. Frontend/Product can begin mock-driven and contract-driven work as soon as the draft contract matrix exists, then revise against the frozen contract in Risk Gate 4.

The handoff rules are:

- Backend/Core publishes a draft contract matrix in Risk Gate 1 and freezes it in Risk Gate 4.
- Frontend/Product may build against the draft matrix with test fixtures, but must not invent error states, response fields, or lifecycle assumptions outside that matrix.
- Frontend/Product records any missing contract needs as explicit feedback on Risk Gate 4 rather than silently filling gaps.
- Frontend/Product starts preparatory work in Risk Gate 1 once the draft contract matrix exists: fixture schema, mock states, route-state audit, and test harness setup.
- Backend/Core records test names, touched backend surfaces, and known semantics for each handoff.
- Frontend/Product records route, hook, component, and browser-observed surfaces for each handoff.
- Cross-lane issues must name both the primary owner lane and the required reviewer lane.
- Each sub-issue should link back to #1440 and name its Risk Gate number.

## Tracker Decisions

Use these decisions when creating GitHub sub-issues after this spec is accepted:

- Sub-issue titles should always use prefixes: `[Risk Gate N][Backend/Core]`, `[Risk Gate N][Frontend/Product]`, or `[Risk Gate N][Split]`.
- GitHub labels can duplicate the risk gate and lane when available, but labels should not replace title prefixes.
- Lane and risk gate should also be present in the issue body so the issue remains readable if labels change.
- The contract matrix should be a standalone document artifact at `Docs/API-related/Prototype_Workspaces_Contract_Matrix.md`, with summaries linked from the Risk Gate 1 and Risk Gate 4 issues.
- Risk Gate 8 should produce a final release-readiness document in the repo and post a summary checklist comment on #1440.
- No GitHub sub-issues should be created while any Risk Gate source issue text still has unresolved process questions.

## Jobs And Scheduler Decisions

Prototype workspace productionization should use **Jobs** for user-visible, retryable, quota-sensitive, or admin-observable work. Scheduler should only be used for internal orchestration where dependency handling is the central concern and the result does not need first-class user/admin visibility.

For the proposed sub-issues, the default routing is:

- Branch session bootstrap: Jobs.
- Preview boot, restart, and replacement: Jobs.
- Preview grant renewal: synchronous preview-broker path unless it needs a preview boot/restart, in which case it enqueues Jobs work.
- Snapshot save: Jobs.
- Publish validation and canonical promotion: Jobs.
- Promotion review record write: synchronous API write; approval enqueues publish-validation/promote Jobs work when promotion needs validation.
- Cleanup and retention: APScheduler or startup maintenance may trigger the cadence, but cleanup should enqueue Jobs-backed work when users/operators need visibility, retries, or quotas.
- Quota/rate-limit enforcement: endpoint and Jobs policy, not Scheduler.

Any future Scheduler use must be called out as an exception in the relevant sub-issue with the reason it is not Jobs-backed.

## Proposed Issue Format

Each proposed sub-issue should be written as a ready-to-create GitHub issue using this structure:

```markdown
## Parent

Tracks part of #1440.

## Risk Being Burned Down

[Concrete risk this issue reduces.]

## Owner Lane

Backend/Core or Frontend/Product.

## Dependencies

- Depends on: [Risk Gate numbers or issue titles]
- Blocks: [Risk Gate numbers or issue titles]

## Scope

- [ ] Concrete work item
- [ ] Concrete work item

## Non-Goals

- Explicitly excluded work
- Work deferred to another risk gate

## Acceptance Criteria

- [ ] Observable outcome
- [ ] Security/data/runtime invariant where relevant
- [ ] Documentation or contract update where relevant

## Verification

- Backend tests:
- Frontend tests:
- Security checks:
- Manual/browser checks:
```

## Risk Gate 1: Prototype Collaboration Threat Model And Authorization Invariants

**Owner lane:** Split
**Backend/Core responsibilities:** threat model, authorization invariants, token/session requirements, audit/quota policy, backend tests
**Frontend/Product responsibilities:** contract fixture schema, mock states, route-state audit checklist, and contract feedback deadline
**Reviewer lane:** Frontend/Product reviews contract usability; Backend/Core reviews frontend prep for contract adherence
**Depends on:** none
**Blocks:** Risk Gates 2, 3, 4, and 5

### Risk Being Burned Down

Private links, external actors, preview grants, promotion authority, and resume/session tokens can become confused or over-permissive. This risk must be reduced before persistence, runtime, API, or frontend work can safely build on the MVP.

### Scope

- Define actor identities: owner, internal collaborator, designated promoter, and external shared actor.
- Define authorization invariants for token creation, public exchange, collaborator session creation, preview grant renewal, and promotion review.
- Define non-enumerating error behavior for invalid, expired, revoked, exhausted, archived, and mismatched links.
- Define token and session security requirements: token storage/hash rules, TTLs, replay handling, cookie flags, referrer leakage controls, password-protected link behavior, signing secret rotation, and revocation propagation. Each requirement must be marked `enforce now`, `document existing behavior`, or `defer to Risk Gate N`.
- Define audit event taxonomy for token creation/exchange, actor creation, preview grant issuance/renewal/revocation, promotion review, and authorization failures.
- Define rate/quota policy for link exchange, collaborator session creation, preview renewal, and runtime job creation.
- Produce a draft contract matrix for frontend use: state, backend condition, HTTP status, stable error category, user-facing state bucket, and retryability.
- Frontend/Product creates preparatory contract fixtures, mock states, a route-state audit checklist, and a contract feedback deadline for Risk Gate 4.
- Add focused backend tests for the highest-risk invariants that already exist in the MVP code.

### Non-Goals

- Implementing all audit dashboards or operational visibility.
- Implementing all quota enforcement paths if those depend on later runtime/job changes.
- Freezing every OpenAPI schema. Risk Gate 4 owns the final contract freeze.

### Acceptance Criteria

- Actor identities and authorization invariants are documented.
- Token/session handling requirements are documented, and each item has an explicit disposition: `enforce now`, `document existing behavior`, or `defer to Risk Gate N`.
- Revocation, expiration, ownership, and cross-workspace isolation have explicit expected behavior.
- A draft frontend-consumable contract matrix exists.
- Frontend/Product has fixture and mock-state preparation work ready to unblock parallel implementation.
- Focused backend tests cover core ownership, revocation, expiration, and cross-workspace isolation behavior.
- Any audit/quota behavior deferred to later gates is explicitly listed.

### Verification

- Backend tests: prototype workspace endpoint, sharing exchange, access service, and service authorization tests relevant to auth invariants.
- Frontend tests: none required, but Frontend/Product reviews the draft contract matrix.
- Security checks: Bandit only if backend code changes occur.
- Manual/browser checks: not required.

## Risk Gate 2: Prototype Workspace Persistence And Transaction Hardening

**Owner lane:** Backend/Core
**Reviewer lane:** Backend/Core peer or maintainer
**Depends on:** Risk Gate 1
**Blocks:** Risk Gates 3, 4, and 8

### Risk Being Burned Down

Partial writes, stale pointers, cleanup gaps, or SQLite/PostgreSQL drift could corrupt collaboration state or make promotion decisions unsafe.

### Scope

- Review repository invariants for workspaces, snapshots, sessions, shared actors, promotion requests, and preview handles.
- Strengthen transaction and compensation behavior for multi-step service operations.
- Verify SQLite and PostgreSQL behavior for migrations, table discovery, placeholder conversion, constraints, and indexes.
- Verify query plans or index coverage for workspace detail, session lookup, active actor lookup, promotion listings, and preview-handle lookup.
- Define cleanup and retention behavior for archived workspaces, revoked actors, expired sessions, stale promotion requests, and preview handles.
- Add tests for failed intermediate writes and compensation behavior.

### Non-Goals

- Moving all prototype persistence into a new DB_Management abstraction unless the review proves the existing AuthNZ repo pattern is insufficient.
- Redesigning the full workspace persistence model.
- Implementing runtime job durability, which belongs to Risk Gate 3.

### Acceptance Criteria

- Repo/service tests cover failed intermediate writes and compensation.
- SQLite and PostgreSQL behavior is documented and covered where the project supports both.
- Cleanup and retention rules are documented and tested for the prototype tables.
- Index/query-plan review is recorded with any necessary changes.

### Verification

- Backend tests: prototype repo, service compensation, migrations, and persistence integration tests.
- Frontend tests: none required.
- Security checks: Bandit only if backend code changes occur.
- Manual/browser checks: not required.

## Risk Gate 3: Runtime Jobs And Preview Lifecycle Durability

**Owner lane:** Backend/Core
**Reviewer lane:** Frontend/Product for user-visible runtime state needs
**Depends on:** Risk Gates 1 and 2
**Blocks:** Risk Gates 4, 6, 7, and 8

### Risk Being Burned Down

Runtime bootstrap, preview handles, and promotion validation may fail under retries, restarts, cancellation, or multi-worker deployment. These failures can expose stale previews, advance the wrong canonical pointer, or leave collaborator sessions stuck.

### Scope

- Move prototype runtime orchestration toward the shared JobManager pattern where appropriate.
- Apply the Jobs/Scheduler decisions from this spec: branch bootstrap, preview boot/restart/replacement, snapshot save, publish validation, and canonical promotion should be Jobs-backed unless an issue records a specific exception.
- Define retry, cancellation, timeout, idempotency, and cleanup behavior for branch session bootstrap, preview boot, snapshot save, and publish-validate-promote jobs.
- Harden preview handle lifecycle across persistent lookup, active-handle replacement, rollback, revocation, cache refresh, and renewal.
- Define and implement the runtime bootstrap status, preview health, and promotion validation failure surfaces needed by frontend and operators.
- Confirm failed publish validation never advances the canonical pointer.

### Non-Goals

- Providing stable long-lived production hosting.
- Building a full deployment platform for prototype previews.
- Solving future merge-heavy collaboration or continuous hosting workflows.
- Exposing owner secrets or production-like environment parity to external collaborators.

### Acceptance Criteria

- Prototype jobs are idempotent and safe to retry.
- Cancellation, timeout, and cleanup semantics are documented and tested.
- Preview revocation and renewal behavior is covered by backend tests.
- Failed publish validation never advances the canonical pointer.
- Runtime and preview status fields needed by later frontend work are implemented or explicitly documented for the contract freeze.

### Verification

- Backend tests: runtime jobs, preview broker, promotion service, and service authorization tests.
- Frontend tests: none required, but Frontend/Product reviews runtime status needs.
- Security checks: Bandit on touched backend paths.
- Manual/browser checks: not required.

## Risk Gate 4: Backend API Contract And Error Semantics Freeze

**Owner lane:** Backend/Core
**Reviewer lane:** Frontend/Product
**Depends on:** Risk Gates 1, 2, and 3
**Blocks:** Risk Gates 5, 6, 7, and 8

### Risk Being Burned Down

Frontend and backend can drift around response shapes, error semantics, retryability, and lifecycle assumptions. This gate converts the draft matrix from Risk Gate 1 into a frozen contract.

### Scope

- Freeze owner and collaborator endpoint response models.
- Freeze stable error categories for expired, revoked, archived, exhausted, stale, conflict, unauthorized, missing, bootstrap-failed, preview-unavailable, and failed-promotion states.
- Finalize the contract matrix: state, backend condition, HTTP status, stable error category, frontend state bucket, retryability, and suggested user-facing handling.
- Document lifecycle examples: create workspace, create share link, exchange public link, create collaborator session, renew preview grant, submit promotion, review promotion.
- Document configuration requirements for signing secrets, runtime policy profiles, quotas, and jobs.
- Add migration and rollback notes for deployments enabling prototype workspace collaboration.

### Non-Goals

- Implementing all frontend states. Risk Gates 5 and 6 own frontend consumption.
- Adding broad unrelated OpenAPI cleanup.
- Reworking feature semantics already settled by Risk Gates 1 through 3 unless a contract review finds a blocking issue.

### Acceptance Criteria

- OpenAPI and API docs match implemented behavior.
- The contract matrix is complete enough for frontend implementation without guessing.
- Frontend/Product signs off that the contract covers collaborator entry and owner promotion UX needs.
- Migration and rollback notes exist.

### Verification

- Backend tests: endpoint response model and error semantics tests.
- Frontend tests: contract fixture validation where practical.
- Security checks: Bandit if backend endpoint/service code changes occur.
- Manual/browser checks: not required.

## Risk Gate 5: Collaborator Entry And Route-State Safety

**Owner lane:** Frontend/Product
**Reviewer lane:** Backend/Core for contract adherence
**Depends on:** Risk Gates 1 and 4
**Blocks:** Risk Gates 6 and 8

### Risk Being Burned Down

Public-share entry can accidentally reuse stale owner workspace state or present an unsafe/incorrect collaborator context. This is the highest-risk frontend path because it crosses unauthenticated/private-link entry, external actor identity, and runtime session creation.

### Scope

- Harden public-share password handoff into prototype collaborator session exchange.
- Ensure collaborator entry never falls back to stale owner workspace state.
- Tighten query keys, disabled queries, missing workspace/session handling, and local state reset behavior.
- Add UI states for revoked, expired, archived, exhausted, mismatched-token, bootstrap-failed, and preview-unavailable flows.
- Build against contract fixtures from Risk Gate 4.

### Non-Goals

- Owner promotion review UX, which belongs to Risk Gate 6.
- Changing backend error semantics outside the frozen contract. Missing needs should be fed back as contract issues.
- Polishing all product copy beyond what is needed for correct collaborator entry behavior.

### Acceptance Criteria

- Frontend tests cover public link exchange and route-state isolation.
- Collaborator session context is explicit and is not inferred from stale local owner state.
- User-facing states map to backend error categories from the contract matrix.
- The public-share handoff is covered with at least one browser-observed or E2E path by the final release gate.

### Verification

- Backend tests: none required unless contract gaps require backend changes.
- Frontend tests: route, hook, client, and component tests for collaborator entry.
- Security checks: not required unless backend code changes occur.
- Manual/browser checks: targeted public-share entry check if practical before Risk Gate 8.

## Risk Gate 6: Owner Review And Promotion UX Hardening

**Owner lane:** Frontend/Product
**Reviewer lane:** Backend/Core for authority and validation semantics
**Depends on:** Risk Gates 3, 4, and 5
**Blocks:** Risk Gate 8

### Risk Being Burned Down

Owners may make unsafe promotion decisions if branch, preview, validation, and review states are unclear. Promotion UX must distinguish review state from runtime state and must not offer actions that backend authority or validation would reject.

### Scope

- Harden pending, approved, rejected, promoted, stale, conflict, validation-running, validation-failed, and promotion-failed states.
- Make branch/session inventory readable enough for owner review decisions.
- Align promotion actions with backend authority and validation semantics.
- Add owner and collaborator copy for promotion request and review outcomes.
- Ensure stale/conflict states are not hidden behind generic failure UI.

### Non-Goals

- Full git-style merge/rebase UI.
- Live multi-user canonical filesystem editing.
- Redesigning the entire prototype workspace product surface.

### Acceptance Criteria

- Owner UI distinguishes review state from runtime state.
- Promotion actions are unavailable when backend authority or validation rules would reject them.
- Frontend tests cover the main review-state transitions.
- Backend/Core confirms the UI state mapping matches contract semantics.

### Verification

- Backend tests: none required unless contract gaps require backend changes.
- Frontend tests: owner view, promotion flow, branch/session inventory, and action availability tests.
- Security checks: not required unless backend code changes occur.
- Manual/browser checks: targeted owner review flow check if practical before Risk Gate 8.

## Risk Gate 7: Operational Visibility And Documentation

**Owner lane:** Split
**Backend/Core responsibilities:** operational status, audit/config/runtime docs
**Frontend/Product responsibilities:** owner/collaborator workflow docs and user-facing examples
**Depends on:** Risk Gates 3 and 4
**Blocks:** Risk Gate 8

### Risk Being Burned Down

Operators and users cannot diagnose runtime, preview, sharing, or promotion failures. Production readiness requires enough visibility and documentation to support failures without reading implementation internals.

### Scope

- Document runtime bootstrap, preview health, signing secrets, quotas, and job behavior.
- Document and lightly wire the operational status surfaces defined or implemented in Risk Gate 3.
- Document owner and collaborator workflows end to end.
- Add examples for password-protected links, single-use links, resume cookies, revoked links, archived workspaces, exhausted links, and promotion conflicts.
- Identify audit events and status fields available for support.

### Non-Goals

- Building a full admin dashboard unless required status surfaces already exist and only need wiring.
- Rewriting broad sharing, jobs, or workspace documentation.
- Replacing Risk Gate 8 release evidence.

### Acceptance Criteria

- Operator docs explain setup, configuration, and failure diagnosis.
- Product docs explain owner and collaborator lifecycle without relying on implementation internals.
- Observability fields/events are documented and covered by tests where practical.
- Split ownership is reflected in the issue checklist so each implementer has clear deliverables.

### Verification

- Backend tests: status/audit/config tests where code changes occur.
- Frontend tests: docs-only changes do not require tests; UI status wiring does.
- Security checks: Bandit if backend code changes occur.
- Manual/browser checks: docs review and any new status UI check.

## Risk Gate 8: End-To-End Release Gate And Production Readiness Review

**Owner lane:** Split
**Backend/Core responsibilities:** backend test matrix, security checks, runtime/API evidence
**Frontend/Product responsibilities:** frontend test matrix, browser-observed UX evidence, workflow evidence
**Depends on:** Risk Gates 1 through 7
**Blocks:** production-ready claim for #1440

### Risk Being Burned Down

The feature can look complete in isolated tests while failing as a full owner-to-collaborator-to-promotion workflow. The final gate must collect evidence across backend, frontend, security, and browser-observed behavior.

### Scope

- Create focused backend and frontend test matrices.
- Add CI-friendly smoke coverage for the full prototype path without requiring external runtime services.
- The minimum smoke path is: owner creates workspace, owner creates share link, collaborator exchanges link, collaborator creates branch session against a runtime/preview stub, collaborator submits promotion request, owner reviews request, publish validation failure is handled without advancing canonical state, publish validation success advances canonical state.
- The minimum negative security smoke path is: an expired or revoked prototype share link fails exchange without confirming whether the token, workspace, or actor exists.
- Run Bandit/security checks on touched backend paths.
- Complete a final browser-observed UX pass.
- Complete a final security review of private-link/session-token flows.
- Record remaining known risks and either close them explicitly or file follow-up issues.

### Non-Goals

- Adding new feature scope not already covered by Risk Gates 1 through 7.
- Reopening major product direction decisions unless the evidence shows a blocker.
- Claiming production readiness without recorded verification evidence.

### Acceptance Criteria

- Backend, frontend, smoke, and security verification results are recorded.
- Browser-observed owner and collaborator flows are recorded.
- The production-readiness decision is evidence-backed.
- Remaining work, if any, is explicitly triaged rather than folded into done.

### Verification Evidence Template

```markdown
## Backend Verification

- Command:
- Result:
- Notes:

## Frontend Verification

- Command:
- Result:
- Notes:

## Security Verification

- Bandit command:
- Bandit output path:
- Result:
- Notes:

## Browser-Observed UX Pass

- Owner flow observed:
- Collaborator link/password flow observed:
- Promotion review flow observed:
- Screenshots or run notes:

## Contract Review

- Contract matrix version/path:
- Backend/Core reviewer:
- Frontend/Product reviewer:
- Open gaps:

## Remaining Risks

- Risk:
- Disposition:
```

## Dependency Map

```text
Risk Gate 1: Threat model and auth invariants
  -> Risk Gate 2: Persistence and transactions
  -> Risk Gate 3: Runtime jobs and preview lifecycle
  -> Risk Gate 4: API contract freeze
       -> Risk Gate 5: Collaborator entry and route-state safety
       -> Risk Gate 6: Owner review and promotion UX
       -> Risk Gate 7: Operational visibility and documentation
            -> Risk Gate 8: End-to-end release gate

Risk Gate 5 also blocks Risk Gate 6 because owner promotion UX should not assume collaborator entry state is safe.
Risk Gate 8 depends on every prior gate.
```

## Recommended GitHub Tracker Flow

After this spec is accepted:

1. Create eight GitHub sub-issues under #1440 using the issue contents above.
2. Use title prefixes for `Risk Gate 1` through `Risk Gate 8`.
3. Add labels for `Backend/Core`, `Frontend/Product`, or `Split` when available, but keep the title prefixes as the durable tracker signal.
4. Link dependencies in each issue body.
5. Start Backend/Core on Risk Gate 1.
6. Start Frontend/Product on mock/fixture preparation after the Risk Gate 1 draft contract matrix exists.
7. Do not mark #1440 production-ready until Risk Gate 8 records evidence.

## Decisions For User Review

- GitHub sub-issues should use title prefixes like `[Risk Gate 1][Backend/Core]`; labels are optional duplicates, not replacements.
- The contract matrix should be a standalone doc artifact at `Docs/API-related/Prototype_Workspaces_Contract_Matrix.md` and summarized in Risk Gate 1 and Risk Gate 4.
- Risk Gate 8 should create a final release-readiness document in the repo and post a summary checklist comment on #1440.
