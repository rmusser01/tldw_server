# Audit Module Review Design

Date: 2026-04-08
Status: Approved for planning
Owner: Codex brainstorming session

## Summary

Run a deep, reliability-first review of the Audit subsystem in `tldw_server`.

The review covers:

- the core Audit service
- dependency injection and lifecycle management
- export and count endpoints
- storage mode and tenant scoping behavior
- migration and fallback durability paths
- representative cross-module audit emitters and adapters
- dedicated Audit tests plus selected integration tests that exercise audit behavior

The output must prioritize actionable findings over commentary, with bugs and risks clearly separated from hardening suggestions.

## Problem

The Audit subsystem is a compliance and operational reliability surface. If it is wrong, the likely failures are not cosmetic. They are event loss, incomplete shutdown flushes, fallback queue corruption, tenant-boundary mistakes in shared mode, or export behavior that becomes unsafe under load.

The module is also spread across several layers:

- core logic in `tldw_Server_API/app/core/Audit/`
- DI and lifecycle code in `tldw_Server_API/app/api/v1/API_Deps/`
- REST endpoints in `tldw_Server_API/app/api/v1/endpoints/`
- emitters and adapters across AuthNZ, Chat, Embeddings, Evaluations, Jobs, Sharing, MCP, and related backend surfaces

Because the behavior is distributed, a useful review needs explicit slices, an evidence bar, and a narrow definition of what counts as a finding.

## Goals

- Produce a severity-ordered Audit review with actionable findings.
- Optimize for reliability and operational risk rather than style.
- Inspect both the Audit core and representative integrations.
- Use static inspection and targeted test execution as evidence.
- Separate `Confirmed issue`, `Likely risk`, and `Improvement suggestion`.
- Identify important testing gaps where reliability-sensitive behavior is weakly protected.

## Non-Goals

- Refactor the Audit module during the review.
- Expand the pass into a general backend style review.
- Treat passing tests as proof that a code path is safe.
- Propose broad architecture rewrites unless needed to explain a concrete issue.

## Scope Confirmed With User

The user approved the broader review scope rather than narrowing to only the Audit core package.

The confirmed scope includes:

- deep review depth
- reliability and operational risk as the main optimization target
- findings-first output with separate improvements

## Review Baseline

The review will inspect the current working tree by default.

If a finding depends on uncommitted local changes, the review must say so explicitly instead of presenting the behavior as if it were clean `HEAD`.

## Recommended Approach

Use a risk-led layered review.

Three candidate approaches were considered:

1. Risk-led layered review
2. Test-led review
3. Integration-led review

The recommended choice is the risk-led layered review because it is the most reliable way to surface silent-loss, shutdown, tenant-boundary, and export-scaling failures before test coverage biases the outcome.

## Inspection Slices

### 1. Core Service and Migration Reliability

Inspect:

- buffering and flush semantics
- high-risk immediate flush logic
- background-task behavior
- shutdown guarantees
- schema creation and initialization
- retention cleanup
- fallback queue append behavior
- shared/per-user storage handling
- migration resumability and partial-failure behavior

Primary files:

- `tldw_Server_API/app/core/Audit/unified_audit_service.py`
- `tldw_Server_API/app/core/Audit/audit_shared_migration.py`

### 2. DI, Lifecycle, and Tenancy

Inspect:

- per-user instance caching
- eviction and reuse semantics
- owner-loop and cross-loop shutdown behavior
- request user binding
- shared storage enablement and rollback precedence
- DB path resolution
- tenant scoping assumptions

Primary files:

- `tldw_Server_API/app/api/v1/API_Deps/Audit_DB_Deps.py`
- `tldw_Server_API/app/core/DB_Management/db_path_utils.py`
- `tldw_Server_API/app/core/config.py`

### 3. Export, Filtering, and Admin Access

Inspect:

- timestamp parsing
- enum and filter mapping
- stream versus non-stream behavior
- row-limit and truncation logic
- filename sanitization
- access-control assumptions
- tenant visibility rules in shared mode

Primary file:

- `tldw_Server_API/app/api/v1/endpoints/audit.py`

### 4. Cross-Module Emitters and Adapter Consistency

Sample high-value producers and adapters to find:

- missing or inconsistent `AuditContext`
- missing or incorrect user scoping
- unsafe fire-and-forget audit writes
- swallowed audit failures
- inconsistent event naming or category mapping
- brittle behavior in shutdown or error paths

Priority integration areas:

- AuthNZ
- Chat
- Embeddings
- Evaluations
- Jobs
- Sharing
- MCP
- RAG-adjacent or governance surfaces when they materially rely on unified audit behavior

### 5. Coverage and Drift

Use dedicated Audit tests and selected integration tests to establish what behavior is actually protected, then identify where reliability-sensitive paths appear weakly covered or untested.

Priority gap areas:

- shutdown races
- fallback durability and locking
- shared-storage tenant edges
- large export behavior
- migration recovery and resumability
- cross-module contract drift

## Evidence Model

Every review item must be labeled as one of:

- `Confirmed issue`
- `Likely risk`
- `Improvement suggestion`

Classification rules:

- `Confirmed issue`: concrete failure mode in code, contract mismatch, or reproducible behavior from tests
- `Likely risk`: credible concern with weak coverage or configuration-dependent behavior, but not fully proven
- `Improvement suggestion`: non-bug hardening work that reduces fragility or operator pain

Each finding must include:

- exact file references
- whether evidence came from static inspection, targeted test execution, or both
- whether the behavior depends on the dirty worktree rather than only committed code

## Deliverable Shape

The final review should be concise and findings-first, using this structure:

## Findings

- severity-ordered issues and likely risks with file references and evidence notes

## Open Questions / Assumptions

- only unresolved items that materially affect confidence

## Improvements

- non-bug hardening suggestions kept separate from bug findings

## Verification

- baseline reviewed
- tests run
- important files inspected
- anything left unverified

## Verification Strategy

Run the dedicated Audit tests plus selected integration tests that directly exercise audit behavior. Test execution is evidence of covered behavior, not proof of global correctness.

The review should prefer representative test slices over blanket unrelated suite execution.
