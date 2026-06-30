# Research Workspace Capability Readiness Hardening Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the Research_Workspace capability review findings without changing unrelated workspace behavior.

**Architecture:** Keep the API endpoint thin and keep capability derivation in `tldw_Server_API/app/core/Research_Workspace/capabilities.py`. Replace endpoint-function health imports with core-owned, injectable health probes that run concurrently with bounded timeouts. Preserve the existing sanitized capability response shape while tightening typing and docs.

**Tech Stack:** FastAPI-adjacent Python core code, Pydantic schemas, pytest/pytest-asyncio, Bandit.

---

## Stage 1: Regression Coverage
**Goal**: Prove the review findings with focused tests before code changes.
**Success Criteria**: Tests fail for real Slides DB unavailability, API endpoint collector imports, timeout/concurrency behavior, sync placeholder top-level status, schema key typing, and README function naming.
**Tests**: `tldw_Server_API/tests/Research_Workspace/test_capability_derivation.py` and `tldw_Server_API/tests/Research_Workspace/test_capability_endpoint.py`.
**Status**: Complete

- [x] Add a collector test showing `try_get_slides_db_for_user` returning `None` makes `slides_generation` block.
- [x] Add a collector test showing slow probes are bounded and independent probes run concurrently.
- [x] Add a source-level assertion that core capability collection no longer imports API endpoint health functions.
- [x] Add a derivation test showing implemented ready capabilities are not top-level degraded solely because `sync_share` is still unknown.
- [x] Add a schema test showing capability response keys use the `ResearchWorkspaceCapabilityId` literal.
- [x] Add a README check for the actual `collect_research_workspace_capabilities()` function name.

## Stage 2: Core Collector Fixes
**Goal**: Make capability collection reliable without depending on API endpoint functions.
**Success Criteria**: Collectors are core-owned/injectable, probes have bounded timeouts, and synchronous Slides work does not block the event loop.
**Tests**: Focused Research Workspace tests pass.
**Status**: Complete

- [x] Add a small `ResearchWorkspaceHealthCollectors` dataclass to carry collector callables.
- [x] Run aggregate, RAG, LLM, Slides, and TTS probes with `asyncio.gather` and per-probe timeout handling.
- [x] Move aggregate, RAG, and LLM checks to core-owned best-effort probes instead of importing endpoint callables.
- [x] Run the Slides DB probe through `asyncio.to_thread`.
- [x] Return sanitized `unknown` payloads on timeout or unexpected collector exceptions.

## Stage 3: Capability Semantics And Contract Cleanup
**Goal**: Fix fail-open and noisy-status behavior while preserving the frontend contract.
**Success Criteria**: Broken Slides DB blocks Slides generation, `sync_share` remains advertised but does not make all-ready implemented capabilities top-level degraded, schema keys are typed, and README naming is accurate.
**Tests**: Focused Research Workspace tests pass.
**Status**: Complete

- [x] Change real Slides DB lookup failure to `unavailable` so composed `slides_generation` blocks.
- [x] Exclude the placeholder `sync_share` capability from overall status until a concrete sync collector exists.
- [x] Tighten the Pydantic capabilities mapping to `dict[ResearchWorkspaceCapabilityId, ResearchWorkspaceCapability]`.
- [x] Correct the README architecture note to use `collect_research_workspace_capabilities()`.

## Stage 4: Verification And Backlog Closeout
**Goal**: Verify the focused behavior and record task status.
**Success Criteria**: Tests, Bandit, and diff checks complete with outcomes recorded in TASK-2423.
**Tests**: `python -m pytest tldw_Server_API/tests/Research_Workspace -q`; Bandit on touched backend files.
**Status**: Complete

- [x] Run focused Research Workspace tests.
- [x] Run Bandit on touched backend source.
- [x] Run `git diff --check`.
- [x] Update TASK-2423 implementation notes/final summary with verification results.
