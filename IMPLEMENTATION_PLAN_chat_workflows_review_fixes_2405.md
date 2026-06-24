## Stage 1: Tracking And Scope
**Goal**: Track the Chat Workflows review fixes under TASK-2405 and keep the change set scoped to the reviewed module plus required tests and persistence support.
**Success Criteria**: Backlog task exists, plan is task-specific, and no unrelated module behavior is changed.
**Tests**: N/A.
**Status**: Complete

## Stage 2: Regression Tests
**Goal**: Add failing tests for dialogue round idempotent replay/conflict behavior, renderer fallback semantics, prompt content handling, prompt bounds, and sanitized renderer error metadata.
**Success Criteria**: Focused tests fail on the current implementation for the expected reasons.
**Tests**: `python -m pytest tldw_Server_API/tests/Chat_Workflows/test_chat_workflows_service.py tldw_Server_API/tests/Chat_Workflows/test_chat_workflows_dialogue_orchestrator.py -q`
**Status**: Complete

## Stage 3: Implementation
**Goal**: Update Chat Workflow service/orchestrator/renderer behavior to satisfy the review findings with minimal changes.
**Success Criteria**: Idempotent dialogue retries replay correctly, mismatched retries conflict, no-op LLM phrasing is reported as an explicit fallback, prompt payloads are bounded and separated from fixed system instructions, and fallback metadata is sanitized.
**Tests**: Focused Chat Workflows pytest suite.
**Status**: Complete

## Stage 4: Verification And Closeout
**Goal**: Run focused tests and security scan, then update TASK-2405 with final verification.
**Success Criteria**: Relevant tests pass, Bandit completes on touched code, and Backlog task records summary and verification evidence.
**Tests**: Focused pytest and Bandit touched-scope command.
**Status**: Complete
