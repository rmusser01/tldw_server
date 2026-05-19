# Workflows Backend Review Design

- Date: 2026-04-07
- Project: tldw_server
- Topic: Backend Workflows module review for issues, bugs, and potential problems/improvements
- Review mode: Findings-first review, no implementation in this phase

## 1. Objective

Run a backend-only review of the Workflows module to identify correctness bugs, security risks, performance problems, maintainability issues, and operational weaknesses.

The deliverable is a prioritized findings report with evidence, not a patch set.

## 2. Scope

### In Scope

- Backend Workflows API surfaces:
  - `tldw_Server_API/app/api/v1/endpoints/workflows.py`
  - `tldw_Server_API/app/api/v1/endpoints/scheduler_workflows.py`
  - `tldw_Server_API/app/api/v1/schemas/workflows.py`
- Core execution and orchestration:
  - `tldw_Server_API/app/core/Workflows/engine.py`
  - `tldw_Server_API/app/core/Workflows/registry.py`
  - `tldw_Server_API/app/core/Workflows/capabilities.py`
  - `tldw_Server_API/app/core/Workflows/investigation.py`
  - `tldw_Server_API/app/core/Workflows/daily_ledger.py`
  - `tldw_Server_API/app/core/Workflows/subprocess_utils.py`
  - high-risk adapter boundaries under `tldw_Server_API/app/core/Workflows/adapters/`
- Persistence and scheduling layers:
  - `tldw_Server_API/app/core/DB_Management/Workflows_DB.py`
  - `tldw_Server_API/app/core/DB_Management/Workflows_Scheduler_DB.py`
  - `tldw_Server_API/app/core/Scheduler/handlers/workflows.py`
  - `tldw_Server_API/app/services/workflows_scheduler.py`
  - related support services for artifact GC, DB maintenance, and webhook DLQ handling
- Tests under `tldw_Server_API/tests/Workflows/` and closely related backend workflow tests when they define or imply core contracts, especially AuthNZ, Resource_Governance, and DB-management tests tied to workflow permissions, quota, and persistence behavior

### Out of Scope

- Frontend workflow editor, chat-workflows UI, and extension workflow surfaces
- Style-only feedback with no behavioral, safety, or maintenance consequence
- Implementation changes during this phase

## 3. Review Method

The review will be contract-driven instead of file-by-file only.

The audit starts from the backend guarantees the module appears to make:

- authorization for creating, viewing, and controlling runs
- legality of run and step state transitions
- idempotency and retry behavior
- consistency of runs, steps, events, and artifacts
- scheduler behavior around duplicate, missed, or overlapping work
- safety at adapter boundaries involving network, filesystem, subprocesses, or external services

These guarantees will be checked in two directions:

1. Implementation inward
- Inspect core API, engine, DB, scheduler, and service entry points
- Follow risky call paths into adapters and helper modules
- Validate that guardrails are enforced where assumptions actually enter the system

2. Tests outward
- Use the existing tests to infer claimed behavior and supported invariants
- Identify weak assertions, missing edge cases, and uncovered control-flow branches
- Treat gaps in test coverage as risk signals, not defects by themselves

The audit will run in four focused passes:

1. Contract and authorization pass
- Trace API request/response contracts, schema normalization, permission checks, ownership checks, and scheduler handoff expectations

2. Lifecycle and persistence pass
- Trace run creation, step execution, status transitions, retry/idempotency behavior, event sequencing, artifact persistence, and quota/ledger side effects

3. Boundary and operations pass
- Inspect adapters and helpers that touch network egress, subprocesses, filesystem paths, webhook delivery, or long-running scheduler behavior

4. Evidence and verification pass
- Convert suspected issues into evidence-backed findings
- Use targeted, minimal test execution only where static trace review is insufficient to confirm or reject a suspected invariant break

## 4. Review Areas

The review will explicitly cover:

1. Correctness
- invalid or unsafe state transitions
- race conditions and duplicate execution risks
- idempotency failures
- pagination/event sequencing errors
- mismatch between API contract and persisted state

2. Security
- RBAC and ownership enforcement
- tenant-isolation boundaries
- egress, webhook, and MCP tool safety
- artifact/path handling and secret exposure risks

3. Performance and Operations
- unbounded scans, caches, or payload sizes
- scheduler rescan and enqueue behavior
- hot-path DB usage and indexing assumptions
- failure recovery, retry storms, and operator visibility gaps

4. Maintainability
- unclear module boundaries
- duplicated validation logic
- behavior hidden behind broad exception handling
- code paths that are difficult to reason about or safely extend

## 5. Evidence Standard

To keep the report reliable:

- Findings should be tied to exact code paths and file references
- Suspected issues should be verified against tests or surrounding implementation before being stated as defects
- If certainty remains limited, the item should be recorded as an open question or risk, not overstated as a bug

## 6. Output Format

The final review output will be findings-first and ordered by severity.

Each finding should include:

- issue statement
- affected file reference(s)
- why it matters in runtime, safety, or operational terms
- issue class: correctness, security, performance, or maintainability
- concrete fix direction when clear

After findings, include:

- open questions or assumptions if behavior is ambiguous
- a short secondary section for lower-priority improvements

## 7. Constraints

- No implementation or patching in this phase
- No expansion into frontend workflow surfaces
- No unrelated refactoring recommendations
- Keep adapter review selective: prioritize adapters and helpers that materially affect safety, control flow, or operational risk rather than attempting an exhaustive read of every adapter file
- Keep the review focused on issues that would matter to operators, maintainers, or users of the backend workflows system

## 8. Success Criteria

The design is successful when:

- the review stays backend-only
- findings are evidence-backed and actionable
- the output prioritizes real defects and risks over broad commentary
- maintainability, performance, and security improvements are included when they materially affect the module
