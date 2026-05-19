# Writing Full-Stack Review Design

Date: 2026-04-09
Topic: Full-stack review of the Writing module in `tldw_server`

## Goal

Produce an evidence-based full-stack review of the Writing module that identifies:

- correctness bugs and edge-case failures
- cross-surface parity issues between shared UI, web route, extension route, and backend contracts
- data integrity and optimistic-locking/versioning risks
- security and unsafe-input handling issues
- maintainability and drift risks in the large Writing code surface
- performance concerns that can plausibly create user-facing defects
- missing or misleading automated tests

The review is intended to prioritize concrete, user-relevant findings over stylistic commentary.

## Scope

This review covers the current full Writing module in the workspace, centered on:

- backend Writing endpoints and schemas:
  - `tldw_Server_API/app/api/v1/endpoints/writing.py`
  - `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`
  - `tldw_Server_API/app/api/v1/schemas/writing_schemas.py`
  - `tldw_Server_API/app/api/v1/schemas/writing_manuscript_schemas.py`
  - `tldw_Server_API/app/core/Writing/manuscript_analysis.py`
  - direct persistence and helper boundaries used by those routes, especially `ChaChaNotes_DB.py` and `ManuscriptDB.py`
- backend Writing tests under `tldw_Server_API/tests/Writing/`
- shared Writing Playground UI under `apps/packages/ui/src/components/Option/WritingPlayground/`
- Writing route, store, and service layers:
  - `apps/packages/ui/src/routes/option-writing-playground.tsx`
  - `apps/packages/ui/src/store/writing-playground.tsx`
  - `apps/packages/ui/src/services/writing-playground.ts`
- web and extension route wrappers:
  - `apps/tldw-frontend/pages/writing-playground.tsx`
  - `apps/tldw-frontend/extension/routes/option-writing-playground.tsx`
- Writing-specific frontend unit, integration, parity, and targeted e2e coverage

The review includes route parity, API shape compatibility, capability handshake behavior, stateful session/template/theme flows, manuscript CRUD and reorder behavior, analysis and wordcloud behavior, editor integration, import/export paths, and the test coverage that defends those behaviors.

Because the full Writing surface is large, execution should prioritize the highest-risk paths first rather than attempting a file-by-file sweep of every Writing utility. The highest-priority surfaces are:

- shared Writing entrypoints, shell, store, and service layers
- backend stateful Writing and manuscript routes plus their direct persistence boundaries
- cross-surface parity guards and contract-heavy tests
- auxiliary utility modules only when they participate in a concrete workflow, a failing guard, or a candidate finding

## Non-Goals

This review does not cover:

- unrelated Notes, Chat, or broader app surfaces unless a Writing path directly depends on them
- broad visual or styling feedback that does not affect correctness, parity, or maintainability
- implementing fixes during the review phase
- unrelated refactoring outside the Writing module
- blanket repo-wide test or security sweeps outside Writing-related paths

## Approaches Considered

### 1. Full-stack boundary-first audit

Start with user-visible Writing flows, then trace each flow through route, shared UI, store, service, backend, and persistence boundaries.

Strengths:

- strongest fit for a full-stack module with shared UI and multiple route shells
- good at catching parity drift and contract mismatches
- keeps findings tied to real workflows instead of isolated files

Weaknesses:

- slower than a backend-only or pure coverage-first review

### 2. UI-first workflow audit

Drive from end-user workflows first, then inspect code only where the workflow appears fragile or inconsistent.

Strengths:

- good at surfacing defects users will actually feel
- efficient for navigation, editing, and interaction regressions

Weaknesses:

- easier to miss backend integrity issues that are not obvious from the UI

### 3. Coverage-first audit

Map the existing tests first, then inspect under-tested or untested branches and use targeted execution to validate risky paths.

Strengths:

- efficient for identifying false confidence and high-value test gaps
- useful in a module with many guard tests and parity checks

Weaknesses:

- weaker when current tests already encode flawed assumptions
- can underweight design-level contract problems

## Recommended Approach

Use a full-stack boundary-first audit with risk-first ordering.

Execution order:

1. inspect the shared UI surface, route wrappers, service/store contracts, and backend contracts together
2. trace state-changing workflows end to end
3. inspect editor, auxiliary tooling, and security-relevant client behavior
4. compare findings against current tests and run only narrow verification needed to confirm or weaken specific claims

This gives the best balance between bug-finding speed, parity checking, and practical full-stack scope control.

## Review Method

### Pass 1: Surface and parity pass

Inspect:

- web route wrapper, extension route wrapper, and shared Writing Playground entrypoints
- shared UI component boundaries and state ownership
- Writing store and service layers
- backend endpoint and schema contracts
- auth, dependency, and rate-limit boundaries on the Writing backend routes
- parity guards and route-level tests

Primary questions:

- do the web and extension surfaces rely on the shared Writing module consistently?
- do service and store assumptions match backend request and response shapes?
- are capability, mode, and route assumptions consistent across surfaces?
- are auth, dependency, and rate-limit protections consistent with the exposed Writing behavior?

### Pass 2: Stateful workflow pass

Inspect:

- sessions, templates, themes, and defaults
- snapshot import/export and clone flows
- manuscript CRUD, reorder, soft-delete, and search flows
- analysis persistence and stale-marking behavior
- versioning and optimistic-locking assumptions across client and server

Primary questions:

- can data be lost, duplicated, corrupted, or silently reshaped?
- can stale state or ordering drift produce user-visible defects?
- do state transitions preserve invariants across client and server boundaries?

### Pass 3: Editor and auxiliary tools pass

Inspect:

- TipTap/editor integration and plain-text bridges
- workspace mode logic and inspector panels
- analysis modals and feedback-related helpers
- tokenizer, logprob, response-inspector, and wordcloud utilities
- import/export helpers and any unsafe rendering or CSS-handling assumptions

Primary questions:

- can editor-state or helper-state drift break workflows or mislead users?
- are failure modes explicit and bounded?
- are there client-side security or sanitization assumptions that are too weak?

### Pass 4: Targeted verification pass

Inspect and run only the highest-value tests needed to answer concrete questions:

- backend pytest slices under `tldw_Server_API/tests/Writing/`
- Writing-specific shared UI tests under `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/`
- route/store/service tests in the shared UI package
- Writing parity guards and route-level tests before any heavier end-to-end execution
- the smallest meaningful Writing e2e tests only when a candidate finding cannot be settled by local code reading plus narrower tests, and only when the relevant harness is already runnable
- Bandit on the Writing backend scope if code changes are later made as follow-up remediation

Primary questions:

- which risky paths are already covered versus weakly defended?
- do executed tests confirm the suspected behavior or reveal blind spots?
- which remaining unknowns need to stay labeled as open questions?
- if a heavier verification path is unavailable or too expensive for this review, is that blind spot stated explicitly instead of silently skipped?

## Review Criteria

Each potential issue is evaluated against these categories:

- correctness and edge-case safety
- cross-surface parity and contract consistency
- data integrity and version/concurrency behavior
- security and unsafe-input handling rigor
- maintainability and drift risk
- performance and unnecessary work on hot paths
- test coverage and test quality

## Evidence Standard

The review should avoid speculative claims. A finding should be backed by at least one of:

- a concrete code path that produces incorrect or risky behavior
- an invariant mismatch between client, service, backend, or persistence layers
- an unguarded failure mode
- a meaningful missing or weak test around a critical branch or user-visible contract

Ambiguous items should be labeled as open questions or assumptions rather than overstated as defects.

## Deliverable Format

The final review output should be organized as:

1. findings first, ordered by severity
2. open questions or assumptions
3. lower-priority improvements
4. verification performed and remaining blind spots

Each finding should include:

- severity (`High`, `Medium`, or `Low`)
- confidence (`Confirmed` or `Probable`)
- type (`correctness`, `security`, `performance`, `maintainability`, `parity`, or `test gap`)
- impact
- concrete reasoning
- file reference(s)

## Severity Model

- `High`: likely bug, data loss/corruption risk, security issue, or major cross-surface regression risk
- `Medium`: correctness edge case, contract inconsistency, meaningful maintainability problem, or performance issue with user-visible impact
- `Low`: smaller cleanup, localized code smell, or lower-priority missing-test issue

## Constraints and Assumptions

- This phase is analysis-first; fixes are out of scope unless explicitly requested afterward.
- The review targets the current workspace state and should call out if any finding depends on uncommitted Writing changes. At the time of scoping, there were no uncommitted Writing-module changes.
- Large Writing files and duplicated logic are not findings by themselves unless they create drift, defects, or material maintenance risk.
- Tests and code both matter, but passing tests do not override contradictory code evidence.
- Targeted verification is required, but broad blanket suites are out of scope.
- The review should not devolve into a blanket sweep of every Writing helper or test file. Lower-level utilities and heavier e2e suites are only pulled in when they support a concrete workflow, guard contract, or candidate finding.

## Success Criteria

This design is successful if it produces a full-stack Writing review that:

- stays focused on the actual Writing module surface
- finds issues across UI, API, and persistence boundaries rather than treating layers in isolation
- ranks findings in a defensible, evidence-backed way
- identifies where the module is solid, where it is risky, and where coverage is misleading or insufficient
