# Audit Module Review Design

Date: 2026-04-07
Status: Approved for planning
Owner: Codex brainstorming session

## Summary

Run a deep review of the Audit module in `tldw_server`, optimized for reliability and operational risk rather than a broad style pass.

The review should cover:

- the core Audit service
- dependency injection and lifecycle management
- export and count endpoints
- storage mode and tenant scoping behavior
- migration and fallback durability paths
- cross-module audit emitters and adapters
- dedicated Audit tests plus selected integration tests that exercise audit behavior

The output should prioritize actionable findings first, then separate non-bug improvement suggestions.

## Problem

The Audit subsystem is a central reliability and compliance surface. It is responsible for collecting security, authentication, authorization, API, and operational events across multiple backend modules. If the module has flaws, those flaws are likely to show up as:

- silent event loss
- flush or shutdown races
- fallback queue corruption or replay ambiguity
- cross-tenant leakage in shared storage mode
- expensive or unsafe export behavior
- inconsistent caller context from integrations
- configuration-dependent drift between expected and actual behavior

The module is also large and spread across multiple layers:

- core audit service logic in `tldw_Server_API/app/core/Audit/`
- FastAPI dependency management in `tldw_Server_API/app/api/v1/API_Deps/`
- REST endpoints in `tldw_Server_API/app/api/v1/endpoints/`
- integrations across AuthNZ, Chat, Embeddings, Evaluations, Jobs, Sharing, MCP, and related backend surfaces

That size and spread means an ad hoc review would be easy to dilute. The review needs explicit slices, evidence thresholds, and output rules so the results are useful for remediation planning.

## Goals

- Produce a severity-ordered findings report for the Audit subsystem.
- Optimize the review for reliability and operational risk.
- Include deep cross-module inspection rather than limiting the pass to the Audit core package.
- Use both static inspection and targeted test execution as evidence.
- Distinguish clearly between:
  - confirmed issues
  - likely risks with weak or missing coverage
  - improvement suggestions
- Identify meaningful testing gaps where current coverage does not protect critical behavior.
- Keep the output actionable enough to turn directly into an implementation plan.

## Non-Goals

- Perform a general code-quality or stylistic review of the entire backend.
- Refactor the Audit module during the review pass.
- Expand the review into unrelated product or frontend audit features.
- Treat passing tests as proof that reliability concerns do not exist.
- Produce a speculative architecture rewrite unless it is needed to explain an issue or improvement.

## Scope Confirmed With User

The user confirmed the following scope decisions:

- review depth: deep review including cross-module Audit integrations
- optimization target: reliability and operational risks first
- output style: actionable findings plus improvements

## Current Context

The repository already has a dedicated Audit subsystem and supporting documentation:

- core implementation in `tldw_Server_API/app/core/Audit/unified_audit_service.py`
- migration helper in `tldw_Server_API/app/core/Audit/audit_shared_migration.py`
- DI and lifecycle layer in `tldw_Server_API/app/api/v1/API_Deps/Audit_DB_Deps.py`
- admin export/count endpoints in `tldw_Server_API/app/api/v1/endpoints/audit.py`
- dedicated guide in `Docs/Code_Documentation/Guides/Audit_Module_Code_Guide.md`
- dedicated tests in `tldw_Server_API/tests/Audit/`

The module also has a broad integration footprint in backend callers, including AuthNZ, Chat, Embeddings, Evaluations, Sharing, Jobs, MCP-related components, RAG-adjacent surfaces, and Workflow paths.

Because the current workspace may contain uncommitted changes in Audit-related files, the execution plan must declare its review baseline explicitly:

- review the current working tree by default
- note when a finding depends on uncommitted local changes
- avoid presenting worktree-only behavior as repository-wide history without saying so

## Review Architecture

Use a risk-led layered review as the primary approach.

Primary review order:

1. core service failure modes
2. DI, lifecycle, and tenant-boundary behavior
3. endpoint and export behavior
4. cross-module integrations and adapter consistency
5. coverage gaps and contract drift

This approach is preferred because it is most likely to surface production-impacting risks such as data loss, stuck shutdowns, scoping errors, and backpressure behavior. Test-led and integration-led inspection are still useful, but only as supporting lenses inside the main risk-led pass.

## Inspection Slices

### 1. Core Service and Migration Reliability

Inspect the audit core and migration path for:

- buffer and flush semantics
- high-risk immediate flush logic
- background task behavior
- shutdown guarantees
- DB initialization and schema management
- retention cleanup
- fallback queue append behavior
- shared/per-user storage mode handling
- query and export helper safety assumptions
- shared-db migration correctness, resumability, and partial-failure behavior

Primary files:

- `tldw_Server_API/app/core/Audit/unified_audit_service.py`
- `tldw_Server_API/app/core/Audit/audit_shared_migration.py`

### 2. Service Lifecycle and Tenancy

Inspect the dependency layer and DB path resolution for:

- per-user instance caching
- eviction and reuse semantics
- owner-loop shutdown behavior
- cross-loop stopping logic
- request user binding
- shared storage enablement rules
- rollback precedence
- potential tenant scoping mistakes
- audit-related configuration resolution and precedence

Primary files:

- `tldw_Server_API/app/api/v1/API_Deps/Audit_DB_Deps.py`
- `tldw_Server_API/app/core/DB_Management/db_path_utils.py`
- `tldw_Server_API/app/core/config.py`

### 3. Export, Filtering, and Admin Access

Inspect API behavior for:

- filter correctness
- timestamp parsing
- enum mapping and validation
- stream vs non-stream behavior
- large export memory risks
- filename sanitization
- access-control assumptions
- shared-storage tenant visibility rules

Primary file:

- `tldw_Server_API/app/api/v1/endpoints/audit.py`

### 4. Cross-Module Emitters and Adapter Consistency

Sample high-value audit producers and adapters across backend modules to find:

- missing or inconsistent `AuditContext`
- incorrect or missing user scoping
- fire-and-forget or unawaited writes in unsafe paths
- swallowed audit failures
- inconsistent event naming or category mapping
- writes performed in brittle shutdown or error paths

Priority integration areas:

- AuthNZ
- Chat
- Embeddings
- Evaluations
- Jobs
- Sharing
- MCP
- RAG
- Workflows

### 5. Coverage and Drift

Use the dedicated Audit tests and selected integration tests to establish what behavior is actually protected. Then identify where reliability-sensitive paths appear untested or under-tested.

Priority gap areas:

- shutdown races
- fallback queue locking and durability
- shared-storage edge cases
- large export behavior
- cross-tenant filtering rules
- migration recovery and resumability

## Method and Evidence Standard

The review should use a mixed static-plus-targeted-execution method.

### Static Analysis

Use code inspection to reason about:

- control flow
- locking
- task ownership
- error handling
- persistence semantics
- tenant scoping
- config precedence

### Targeted Execution

Run the dedicated Audit test suite and selected related integration tests that exercise audit behavior. Test execution is evidence of covered behavior, not proof of overall correctness.

### Finding Classification

Each review item should be tagged as one of:

- `Confirmed issue`
- `Likely risk`
- `Improvement suggestion`

A confirmed issue requires at least one of:

- a concrete failure mode visible in code
- behavior that contradicts intended contract or documented behavior
- a reproducible issue from test execution

A likely risk is appropriate when the concern is credible but not fully proven from available evidence, usually because coverage is weak or the critical path is configuration-dependent.

Improvement suggestions should remain separate from bug findings and should focus on reducing operational fragility rather than general cleanup.

## Test Execution Targets

The review should prioritize:

- `tldw_Server_API/tests/Audit/`

Then sample related audit-focused tests in:

- `tldw_Server_API/tests/AuthNZ/`
- `tldw_Server_API/tests/AuthNZ_SQLite/`
- `tldw_Server_API/tests/AuthNZ_Postgres/`
- `tldw_Server_API/tests/Chat/`
- `tldw_Server_API/tests/Chat_NEW/`
- `tldw_Server_API/tests/Embeddings/`
- `tldw_Server_API/tests/Evaluations/`
- `tldw_Server_API/tests/Jobs/`
- `tldw_Server_API/tests/Sharing/`
- `tldw_Server_API/tests/MCP_unified/`
- `tldw_Server_API/tests/Admin/`
- `tldw_Server_API/tests/UserProfile/`

Selection should be driven by audit integration coverage rather than blanket execution of unrelated suites.

Selection rule:

- identify candidate tests by searching for direct Audit module references or explicit audit-side-effect assertions
- choose at least one representative audit-focused test module for each priority integration area that appears to have meaningful coverage
- expand beyond representative tests only when a finding requires deeper verification

## Deliverable Format

The final review output should present:

1. findings first, ordered by severity
2. concise explanation of each failure mode or risk
3. affected file references
4. brief remediation direction for each finding
5. open questions or assumptions
6. improvement suggestions separated from bug findings
7. a short verification note listing what tests were run and what remains unverified
8. a baseline note stating whether each major finding is based on the current worktree, committed behavior, static inspection, test execution, or a combination

The report should be readable as a review document first, not a changelog.

## Planning Readiness

This design is ready to hand off to implementation planning if the resulting plan covers:

- repository inspection order
- concrete code areas to review
- targeted pytest commands
- a method for recording evidence against each finding
- a structure for separating findings from improvements
- a declared review baseline for handling dirty-worktree conditions
- explicit config and migration inspection steps
- a final verification step before presenting conclusions
