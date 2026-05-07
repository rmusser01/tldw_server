---
id: TASK-96.8
title: Implement real Auto Chunking boundary assistant adapter
status: Done
assignee:
  - Codex
created_date: '2026-05-06 17:53'
updated_date: '2026-05-07 04:28'
labels:
  - backend
  - chunking
  - auto-chunking
  - llm
dependencies:
  - TASK-96.7
documentation:
  - Docs/superpowers/specs/2026-05-06-auto-chunking-design.md
  - Docs/superpowers/plans/2026-05-06-auto-chunking-implementation-plan.md
  - Docs/superpowers/specs/2026-05-07-auto-chunk-boundary-assistant-design.md
  - >-
    Docs/superpowers/plans/2026-05-07-auto-chunk-boundary-assistant-implementation-plan.md
parent_task_id: TASK-96
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up for Auto Chunking V1. Add a real LLM-backed boundary assistant only after defining an explicit adapter interface and availability checks. The adapter should refine boundaries or labels from deterministic candidate plans and must fall back deterministically on provider, timeout, config, or runtime errors.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Define a narrow AutoChunkBoundaryAssistant interface and result type before adding provider calls.
- [x] #2 Availability checks are explicit and do not rely only on provider keys being configured.
- [x] #3 Adapter is invoked only when auto_chunking_use_llm=true.
- [x] #4 Timeout, provider error, and invalid response paths preserve deterministic Auto plans with fallback metadata.
- [x] #5 Tests cover default no-call behavior, explicit opt-in success, timeout/error fallback, and metadata used_llm semantics.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Document the approved TASK-96.8 design and implementation plan in the worktree, scoped to backend-only Auto Chunking LLM assistance with no UI changes. 2. Add tests first for a narrow AutoChunkBoundaryAssistant interface/result, explicit availability, no-call default behavior, opt-in success, timeout/provider/invalid-response fallback, and used_llm metadata semantics. 3. Implement a focused boundary assistant module that wraps perform_chat_api_call_async with bounded prompt/input/output, explicit provider/model/adapter/key availability checks, timeout handling, strict JSON parsing, and deterministic validation. 4. Add an async resolver alongside the existing sync deterministic resolver so legacy/manual behavior and existing sync call sites remain stable, while async ingestion paths can refine Auto plans only when auto_chunking_use_llm=true. 5. Wire async media add, ingest jobs, process endpoints, and web/article ingestion paths to the async resolver; preserve template application only when no Auto plan is returned. 6. Run focused backend tests, Bandit on touched production code, and git diff --check; update Backlog acceptance criteria and final notes with verification results.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started in isolated worktree /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/auto-chunk-boundary-assistant on branch codex/auto-chunk-boundary-assistant from origin/dev@204212ed7d1029b47e20f1a3a9866b46af032f62. User approved backend-only bounded JSON LLM boundary assistant scope: no UI changes, explicit opt-in only, deterministic fallback on unavailable/error paths.

Added TASK-96.8-specific spec and implementation plan. External spec/plan review via subagent was not used because this session's developer constraints only allow subagents when the user explicitly asks for them; proceeding with inline self-review and TDD.

Implemented backend-only LLM boundary assistant behind explicit auto_chunking_use_llm opt-in. Added bounded strict-JSON chat adapter with explicit provider/model/adapter/API-key availability checks, timeout/provider/invalid-response deterministic fallbacks, canonical provider alias handling, async resolver wiring for media add/jobs/process/web ingestion paths, and focused tests for no-call default, success, fallback, metadata, and worker wiring. Verification: focused pytest suite passed 59 tests; py_compile passed for touched production modules; Bandit JSON scan of touched production files reported zero results; git diff --check passed. No known skips or blockers.

PR #1354 review-fix pass started. Actionable review items verified from GitHub review threads and review bodies: avoid sync config loading on async assistant path, wrap overlong availability line, keep pre-assistant Auto plan deterministic, avoid TypeError retry masking in API-key resolver, reuse one LLM boundary resolution per process_* batch, reject ebook_chapters outside ebook media, preserve api_name-encoded model when api_provider is already present, propagate asyncio cancellation in persistence resolver awaits, and propagate web request LLM selection into synthetic chunking forms.

PR #1354 review-fix pass completed. Implemented all actionable review findings: availability checks now run off the event loop; API-key resolver signatures are inspected instead of retrying broad TypeError; pre-assistant plans remain deterministic; ebook_chapters is ebook-only; api_name provider/model parsing preserves encoded model values; process_* endpoints reuse one LLM boundary resolution per batch; persistence re-raises asyncio.CancelledError; and web chunking forms carry api_name/provider/model fields. Verification: focused pytest suite passed 66 tests; py_compile passed for touched production modules; Bandit JSON scan of touched production files reported zero results; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the real Auto Chunking boundary assistant adapter as a backend-only opt-in refinement path, then addressed PR #1354 review findings. The assistant now runs availability checks off the event loop, validates provider/model/key availability without masking resolver TypeError retries, canonicalizes provider aliases, rejects ebook-only methods outside ebook media, and preserves deterministic Auto plans until an assistant result is actually applied. Async ingestion paths use the resolved chunk options for media add/jobs/process/web flows, with process_* endpoints reusing one LLM boundary resolution per batch and persistence preserving asyncio cancellation semantics. Web chunking forms now propagate request LLM selection fields so api_name-encoded models are honored. Verification: focused pytest suite passed 66 tests; py_compile passed for touched production modules; Bandit scan of touched production files reported zero findings; git diff --check passed. No known skips or blockers.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
