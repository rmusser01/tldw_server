# Chunking Module Review Design

Date: 2026-03-23
Topic: Full backend Chunking module review in `tldw_server`
Status: Approved design

## Goal

Produce an evidence-based review of the backend Chunking module that identifies:

- correctness bugs and regression risks
- metadata and offset integrity issues
- strategy-contract inconsistencies and surprising fallback behavior
- security, regex-safety, and concurrency weaknesses
- maintainability hazards that materially increase defect likelihood
- missing or misleading tests around risky behavior

The review should prioritize concrete findings over style commentary and avoid broad rewrite suggestions unless they directly reduce defect risk in the Chunking package.

## Scope

This review covers the full backend Chunking package centered on:

- `tldw_Server_API/app/core/Chunking/chunker.py`
- `tldw_Server_API/app/core/Chunking/async_chunker.py`
- `tldw_Server_API/app/core/Chunking/base.py`
- `tldw_Server_API/app/core/Chunking/constants.py`
- `tldw_Server_API/app/core/Chunking/exceptions.py`
- `tldw_Server_API/app/core/Chunking/multilingual.py`
- `tldw_Server_API/app/core/Chunking/regex_safety.py`
- `tldw_Server_API/app/core/Chunking/security_logger.py`
- `tldw_Server_API/app/core/Chunking/template_initialization.py`
- `tldw_Server_API/app/core/Chunking/templates.py`
- all splitters under `tldw_Server_API/app/core/Chunking/splitters/`
- all strategies under `tldw_Server_API/app/core/Chunking/strategies/`
- chunking utility modules under `tldw_Server_API/app/core/Chunking/utils/`

The review also includes the immediate integration surface that defines or materially exercises Chunking behavior:

- `tldw_Server_API/app/api/v1/endpoints/chunking.py`
- `tldw_Server_API/app/api/v1/endpoints/chunking_templates.py`
- template application and initialization call paths where Chunking behavior is selected or modified
- dedicated backend tests under `tldw_Server_API/tests/Chunking/`
- nearby integration tests that rely materially on Chunking contracts
- Chunking package documentation when it affects contract interpretation:
  - `Docs/Design/Chunking.md`
  - `tldw_Server_API/app/core/Chunking/README.md`
  - `tldw_Server_API/app/core/Chunking/SECURITY.md`

The review includes:

- orchestration and strategy selection
- option normalization and alias handling
- chunk boundary and overlap behavior
- metadata/span integrity and cross-strategy invariants
- async and batch behavior
- template-driven flows
- fallback behavior under parsing or dependency failure
- regex and parser safety controls

## Non-Goals

This review does not cover:

- frontend Chunking Playground behavior except where it exposes a direct backend contract issue
- a broad RAG, embedding, or retrieval audit outside direct Chunking interfaces
- unrelated ingestion, storage, or vectorization design except where Chunking assumptions directly affect them
- implementing fixes during the review phase
- unrelated refactors or cleanup not tied to clear backend Chunking risk

## Approaches Considered

### 1. Surface audit

Perform a quick read across the package and sample a subset of tests.

Strengths:

- fast
- likely to catch obvious defects and drift

Weaknesses:

- weak at finding cross-strategy contract mismatches
- likely to miss metadata, overlap, and fallback inconsistencies
- not ideal for a package with many strategies and helper layers

### 2. Risk-based full audit

Review the package as a system, then verify the highest-risk behaviors with the dedicated Chunking test suite and targeted follow-up runs.

Strengths:

- best fit for full-package review
- strong for finding concrete bugs and systemic inconsistencies
- produces both findings and a practical improvement backlog

Weaknesses:

- slower than a surface audit
- requires discipline to avoid drifting into unrelated adjacent modules

### 3. Behavior-first audit

Start from test execution, then inspect implementation deeply only where failures or weak coverage suggest risk.

Strengths:

- efficient for finding regressions
- strong for contract drift already represented in tests

Weaknesses:

- can miss latent problems in lightly tested strategies
- overweights existing test shape, which may not match production risk

## Recommended Approach

Use the risk-based full audit.

Execution order:

1. map entrypoints, orchestration, shared helpers, and strategy families
2. inspect option handling, metadata/span contracts, and fallback paths
3. inspect strategy-specific behavior for correctness, consistency, and edge cases
4. inspect async, template, and safety-oriented paths
5. run the dedicated Chunking tests
6. run targeted follow-up tests or subsets for suspicious or failing areas
7. synthesize findings by severity, evidence, and type

This approach keeps the review grounded in runtime behavior while still treating architecture and maintainability risks as first-class outputs when they materially affect correctness or future defect probability.

## Review Method

### Pass 1: Static system audit

Inspect:

- public entrypoints and orchestration flow
- strategy registration and selection behavior
- option normalization and alias resolution
- metadata and offset/span contracts
- overlap clamping and chunk-boundary logic
- fallback and exception-handling paths
- template classification/application flow
- shared state and async/thread-safety concerns
- regex-safety and security-oriented helpers

Primary questions:

- do all strategies honor the same effective contract, or do callers need strategy-specific knowledge?
- can offsets, counts, or chunk text drift from source reality?
- do fallback paths silently change semantics in unsafe or undocumented ways?
- are shared helpers or template flows introducing hidden coupling or race risk?

### Pass 2: Behavior verification

Inspect:

- the dedicated Chunking test suite under `tldw_Server_API/tests/Chunking/`
- nearby integration tests where Chunking behavior is materially relied on
- targeted reruns for failing or suspicious areas

Primary questions:

- which suspected issues are directly reproducible or already failing?
- which behaviors appear relied upon by tests but weakly asserted?
- where do tests imply a contract that the implementation does not consistently honor?

## Verification Targets

The review should explicitly verify:

- correctness of chunk boundaries, overlap handling, and off-by-one behavior
- metadata integrity, especially `start_char`, `end_char`, chunk indices, counts, and source-span fidelity
- strategy consistency across common operations and return contracts
- fallback behavior when optional dependencies, parsing, or semantic/code logic fails
- async and thread-safety concerns in batch flows and template initialization/use
- security and robustness issues in regex-heavy or parser-facing paths
- test quality gaps in risky branches and invariants

## Findings Model

Potential issues should be categorized as one of:

- `confirmed defect`
- `high-risk likely bug`
- `coverage gap`
- `improvement/refactor opportunity`

This distinction is important because a large strategy-heavy module can produce many plausible concerns that are not equally supported by evidence. The final review should make confidence and evidence explicit rather than flattening everything into the same class of claim.

## Evidence Standard

The review should avoid speculative claims. A finding should be backed by at least one of:

- a concrete code path that can produce incorrect or risky behavior
- an observed failing or suspicious test result
- a clear mismatch between implementation and tested/documented contract
- a missing or weak test around a critical branch, invariant, or state transition

Ambiguous items should be labeled as open questions or assumptions rather than overstated as defects.

## Deliverable Format

The final review output should be organized as:

1. findings first, ordered by severity
2. open questions or assumptions
3. coverage and residual-risk summary
4. secondary improvements and refactor opportunities

Each finding should include:

- severity (`High`, `Medium`, or `Low`)
- type (`correctness`, `security`, `reliability`, `maintainability`, or `test gap`)
- confidence/evidence basis
- impact
- concise reasoning
- file reference(s)

## Severity Model

- `High`: likely correctness defect, unsafe fallback, metadata corruption, broken contract, or concurrency/safety issue with meaningful operational impact
- `Medium`: important edge case, reliability weakness, or maintainability problem that materially raises regression risk
- `Low`: localized cleanup, narrower mismatch, or missing-test issue with limited immediate blast radius
