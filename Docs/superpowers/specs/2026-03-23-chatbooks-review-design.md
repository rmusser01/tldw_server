# Chatbooks Review Design

Date: 2026-03-23
Topic: Deep backend and contract-surface review of the Chatbooks module in `tldw_server`
Status: Approved design

## Goal

Produce an evidence-based review of the Chatbooks module that identifies:

- correctness bugs and edge-case failures
- security and archive-handling weaknesses
- job lifecycle and state consistency problems
- API, schema, doc, and PRD drift that can mislead callers or maintainers
- maintainability risks that materially increase defect likelihood
- missing or misleading tests around risky behavior

The review is intended to prioritize concrete findings over style commentary.

## Scope

This review covers the Chatbooks backend and its contract surface, centered on:

- `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
- `tldw_Server_API/app/core/Chatbooks/chatbook_validators.py`
- `tldw_Server_API/app/core/Chatbooks/quota_manager.py`
- `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`
- `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`
- supporting docs and contracts in:
  - `Docs/Code_Documentation/Guides/Chatbooks_Code_Guide.md`
  - `Docs/Product/Chatbooks_PRD.md`
  - nearby Chatbooks API/schema docs when they materially affect contract interpretation
- backend tests under:
  - `tldw_Server_API/tests/Chatbooks/`
  - related integration and end-to-end tests that exercise Chatbooks behavior

The review includes export, import, preview, download, cancellation, cleanup, quota enforcement, job tracking, archive validation, and the consistency of documented behavior versus implementation.

## Non-Goals

This review does not cover:

- frontend Chatbooks pages or WebUI workflows
- visual UX concerns except where backend contracts create downstream defects
- implementing fixes during the review phase
- broad refactoring unrelated to clear defect risk

## Approaches Considered

### 1. Code-first audit

Start with the service and endpoint code, then validate tests and docs against what the code actually does.

Strengths:

- strong for finding real bugs quickly
- efficient when the main risk lives in runtime behavior

Weaknesses:

- easier to defer or miss contract drift until later

### 2. Contract-first audit

Start from schemas, docs, and the PRD, then verify whether the implementation and tests satisfy those contracts.

Strengths:

- strong for detecting stale guarantees and interface drift
- useful when external callers depend on stable behavior

Weaknesses:

- can over-index on planned gaps instead of implementation defects

### 3. Layered audit

Review the module in ordered passes: backend correctness and security first, then endpoint and job behavior, then schema/doc/PRD alignment, then test adequacy and maintainability risk.

Strengths:

- best fit for a deep review without losing severity ordering
- balances runtime bug-finding with contract validation
- keeps findings organized by impact rather than by file

Weaknesses:

- slightly more process overhead than a pure code-first pass

## Recommended Approach

Use the layered audit.

Execution order:

1. inspect core service, validator, and quota behavior
2. trace endpoint flows and job-state transitions end to end
3. compare schemas, docs, and PRD claims against actual behavior
4. review tests to identify risky untested or weakly tested paths
5. synthesize findings by severity and type

This sequence front-loads correctness and security while still treating contract drift and coverage gaps as first-class review outputs.

## Review Method

### Pass 1: Core backend correctness and security

Inspect:

- archive validation and path-safety logic
- export/import/preview behavior in the service
- quota and retention logic
- temp-file and storage-path handling

Primary questions:

- can malformed input or archive contents bypass validation?
- can exports or imports fail in ways that leave inconsistent state behind?
- are limits, retention, and safety checks actually enforced where claimed?

### Pass 2: Endpoint and job lifecycle behavior

Inspect:

- request parsing and schema coercion
- sync versus async behavior
- job creation, lookup, cancellation, download, and cleanup paths
- error handling and status transitions

Primary questions:

- do endpoint contracts match runtime behavior?
- are status changes and terminal states consistent across flows?
- can callers observe misleading success, progress, or failure states?

### Pass 3: Contract-surface alignment

Inspect:

- request/response schemas
- Chatbooks code guide and PRD statements
- API-level guarantees around supported types, limits, signed URLs, cleanup, and job semantics

Primary questions:

- do docs and schemas promise behavior the code does not provide?
- are known limitations clearly represented, or is drift likely to mislead consumers?
- are there ambiguous contracts that would cause a planner or caller to build against the wrong behavior?

### Pass 4: Test adequacy and maintainability risk

Inspect:

- Chatbooks unit, integration, and end-to-end tests
- high-risk branches and state transitions in the backend code
- concentration of responsibilities in large files where defects can hide

Primary questions:

- which risky paths are untested or weakly tested?
- do tests cover the failure modes that matter, not just the happy path?
- are there structural hotspots that materially increase regression risk?

## Review Criteria

Each potential issue is evaluated against these categories:

- correctness
- security
- state and job lifecycle consistency
- API/schema/doc/PRD drift
- maintainability risk with likely behavioral impact
- test gap around risky behavior

## Evidence Standard

The review should avoid speculative claims. A finding should be backed by at least one of:

- a concrete code path that produces incorrect or risky behavior
- an inconsistency between endpoint behavior, schema, and documented contract
- an unguarded failure mode or unsafe state transition
- a missing test around a critical branch, invariant, or security-sensitive flow

Ambiguous items should be labeled as open questions or assumptions rather than overstated as defects.

## Deliverable Format

The final review output should be organized as:

1. findings first, ordered by severity
2. open questions or assumptions
3. short coverage and residual-risk summary
4. secondary improvements, only after confirmed findings

Each finding should include:

- severity (`High`, `Medium`, or `Low`)
- type (`correctness`, `security`, `contract drift`, `maintainability`, or `test gap`)
- impact
- concrete reasoning
- file reference(s)

## Severity Model

- `High`: likely bug, security issue, broken job lifecycle/state behavior, or contract defect with meaningful operational impact
- `Medium`: correctness edge case, misleading contract/documentation mismatch, or maintainability issue that materially raises regression risk
- `Low`: localized cleanup, smaller mismatch, or narrower missing-test issue

## Constraints and Assumptions

- This phase is analysis-only; no code changes are part of the review deliverable.
- Primary evidence comes from code and tests; docs and PRD are used to evaluate contract alignment, not to override implementation reality.
- Existing PRD TODOs and deferred features should be treated as planned gaps unless the current code or docs present them as already supported.

## Success Criteria

This design is successful if it produces a Chatbooks review that:

- stays scoped to the backend and contract surface without drifting into frontend audit work
- yields ranked, defensible findings tied to concrete evidence
- separates confirmed bugs from open questions and from lower-priority improvements
- surfaces contract drift and test gaps clearly enough to guide follow-up remediation or deeper investigation
