# Evals Module Review Design

- Date: 2026-04-07
- Project: `tldw_server`
- Topic: Deep sequential review of the Evals module for issues, bugs, and practical improvements
- Review mode: Findings-first review with one cumulative repo-local review document

## 1. Objective

Run a deep backend review of the Evals module in sequential slices instead of a single broad pass.

The goal is to produce an evidence-backed review that identifies:

- concrete bugs
- correctness and contract risks
- security and authorization problems
- operational and persistence issues
- maintainability issues with clear failure potential
- meaningful test gaps

The review should not stop at findings. Each finding should also carry:

- severity
- practical impact
- recommended fix direction
- recommended tests
- priority for remediation

## 2. Scope

### In Scope

The review starts with the Evals backend and its direct edges, beginning from the unified entrypoint and then moving inward and outward in controlled slices.

Review baseline:

- the review targets the current working tree at execution start, not `HEAD` alone
- before slice 1 begins, the cumulative review document must record:
  - the current `HEAD` commit
  - the list of modified and untracked Evals-related files present at review start
- when a finding is materially affected by an in-progress local edit, the slice notes should say so explicitly instead of implying the issue necessarily exists in the last committed state

Planned slice order:

1. Unified API and auth surface
   - `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py`
   - `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_auth.py`
   - directly related route wiring, request guards, and rate-limit behavior
2. Core orchestration and execution
   - `tldw_Server_API/app/core/Evaluations/eval_runner.py`
   - directly related service wiring and evaluator dispatch paths
3. Persistence and state management
   - `tldw_Server_API/app/core/Evaluations/evaluation_manager.py`
   - directly related DB adapters, factories, and migrations used by Evals
4. CRUD and run lifecycle endpoints
   - evaluation creation, retrieval, history, exports, and run management paths
5. Retrieval and recipe-driven evaluation surfaces
   - RAG pipeline endpoints, response-quality surfaces, and recipe execution or tuning paths
6. Benchmark and dataset generation surfaces
   - benchmarks, datasets, and synthetic evaluation flows
7. Embeddings A/B and webhook surfaces
   - embeddings A/B evaluation endpoints, webhook registration and delivery, and related security checks
8. Cross-slice contract synthesis
   - shared schemas
   - evaluation config coupling
   - residual coverage gaps and contract mismatches discovered across prior slices

### Out of Scope

- frontend review of evaluation UI flows except where tests or endpoint contracts expose backend expectations
- implementing fixes during the review phase
- broad unrelated refactors
- a repo-wide audit of non-Evals modules unless a direct dependency materially affects the current slice

## 3. Approaches Considered

### 1. Entry-point first

Start at the public API surface, then follow calls inward.

Strengths:

- catches externally visible contract and permission problems early
- gives fast signal on user-facing breakage

Weaknesses:

- deeper systemic issues may only become clear later

### 2. Core first

Start at storage and orchestration internals, then move outward to API surfaces.

Strengths:

- good for uncovering systemic correctness and persistence issues

Weaknesses:

- slower to surface externally visible failures or guardrail defects

### 3. Risk-layered hybrid

Review slices in the order: unified API and auth, orchestration, persistence, discrete feature surfaces, then cross-slice synthesis of shared schemas and residual gaps.

Strengths:

- aligns with failure propagation
- catches high-blast-radius problems early while still tracing root causes
- fits a cumulative review document that will be updated slice by slice

Weaknesses:

- requires tighter discipline around overlaps between slices

## 4. Recommended Approach

Use the risk-layered hybrid review.

This is the best fit because the Evals module spans API contracts, authorization, asynchronous execution, persistence, specialized feature endpoints, and a large test surface. A simple file-by-file read would be less useful than reviewing in the order that production failures would actually manifest.

Execution order:

1. Unified API and auth surface
2. Core orchestration and execution
3. Persistence and state management
4. CRUD and run lifecycle endpoints
5. Retrieval and recipe-driven evaluation surfaces
6. Benchmark and dataset generation surfaces
7. Embeddings A/B and webhook surfaces
8. Cross-slice contract synthesis

## 5. Review Method

Each slice review will follow the same workflow:

1. Read the slice entry files and immediate dependencies.
2. Map control flow, persistence touchpoints, auth and rate-limit behavior, and external integrations.
3. Inspect the most relevant tests for that slice and identify missing or weak coverage.
4. Record only actionable findings:
   - bug
   - correctness risk
   - security or privacy issue
   - behavioral regression risk
   - maintainability issue with a clear failure mode
5. Add a recommended fix, priority, and test for every finding.
6. Append the slice section to the cumulative review document before moving to the next slice.

Test review is part of every slice. The final cross-slice synthesis pass should only capture shared schema or config drift and residual coverage gaps that were not already exhausted within the earlier slice reviews.

If a suspected issue cannot be confirmed from static review alone, it should be marked `needs verification` instead of being overstated as a confirmed defect.

If a later slice changes the understanding of an earlier finding, the cumulative review document should be updated in place rather than duplicating the issue.

## 6. Cumulative Review Artifact

The review output will be maintained in a single cumulative repo-local document:

- `Docs/superpowers/reviews/evals-module/README.md`

That file will be the source of truth for the entire review. It should be updated sequentially as each slice is completed.

Planned structure:

- baseline snapshot: `HEAD` commit plus dirty-file inventory at review start
- review scope and slice order
- review methodology and severity model
- one section per reviewed slice
- cross-slice systemic issues
- priority summary
- recommended remediation order
- coverage gaps and follow-up verification items

Each slice section should include:

- files reviewed
- baseline note when in-progress local edits materially affect interpretation
- short control-flow and data-flow notes
- findings ordered by severity
- each finding must include:
  - severity
  - confidence
  - priority
  - why it matters
  - exact file references
  - recommended fix
  - recommended tests
  - verification note or reproduction condition when needed
- open questions or assumptions
- slice status: `reviewed`, `needs follow-up`, or `blocked`

No per-slice standalone review files should be created unless the scope changes later.

## 7. Severity and Priority Model

### Severity

- `Critical`: auth bypass, cross-user leakage, data corruption, destructive persistence errors, or comparable security failures
- `High`: externally visible incorrect behavior, broken contracts, persistent inconsistency, or major operational failure
- `Medium`: edge-case reliability problems, misleading errors, significant coverage gaps, or fragile control-flow assumptions
- `Low`: maintainability or clarity issues that are likely to cause future defects

### Priority

- `Immediate`: should be fixed before building more on the reviewed surface
- `Near-term`: important but not release-blocking
- `Later`: worth tracking after higher-value defects are addressed

Every recorded finding should include both severity and remediation priority.

## 8. Evidence Standard

The review should avoid speculative claims.

A finding should be backed by at least one of:

- a concrete code path that can produce incorrect or unsafe behavior
- a contract mismatch between implementation, schema, and tests
- a guardrail or state assumption that is missing or inconsistently applied
- a meaningful test gap around an important invariant

When certainty is limited, the item should be written as an open question or `needs verification` risk rather than a confirmed bug.

## 9. Completion Criteria

A slice is complete when:

- the slice entry files and direct dependencies have been inspected
- relevant tests for that slice have been checked
- actionable findings have been appended to the cumulative review document
- overlaps with earlier slices have been reconciled in the same cumulative document
- the slice has a final status marker
- any baseline-sensitive findings are labeled as working-tree-specific when appropriate

The overall review is complete when:

- the initial baseline snapshot is recorded
- all planned slices have been reviewed
- cross-slice issues have been synthesized
- remediation priorities have been summarized
- remaining verification items are explicitly listed

## 10. Success Criteria

This design is successful when the eventual review:

- stays deep rather than broad and shallow
- progresses in an explicit sequence of slices
- leaves one stable cumulative review document in the repo
- produces findings that are actionable and evidence-backed
- includes recommended fixes, priorities, and tests for each meaningful issue
