# Chat Workflows

Chat_Workflows implements template-driven multi-step chat workflows and moderated dialogue runs. It stores workflow templates and run state in a per-user database, renders or drafts questions, records answers, drives Socratic dialogue rounds through the shared chat orchestrator, and exposes the workflow lifecycle through the chat workflows API.

## Start Here

- `service.py` contains `ChatWorkflowService`, the main service for templates, draft generation, run start, current step lookup, answer recording, dialogue rounds, and cancellation.
- `dialogue_orchestrator.py` coordinates debate and moderator LLM turns for dialogue workflows.
- `question_renderer.py` renders the current step prompt and generation metadata.
- Related API surface: `tldw_Server_API/app/api/v1/endpoints/chat_workflows.py`, declared with prefix `/api/v1/chat-workflows`.
- Related dependencies: `tldw_Server_API/app/api/v1/API_Deps/chat_workflows_deps.py`.
- Related schemas: `tldw_Server_API/app/api/v1/schemas/chat_workflows.py`.
- Related tests: `tldw_Server_API/tests/Chat_Workflows/`.

## Responsibilities

- Create, list, retrieve, update, and delete workflow templates.
- Generate draft templates from user goals, an optional base question, and the desired step count.
- Start workflow runs and track active, completed, or canceled state.
- Render current workflow questions and record answers.
- Run moderated dialogue rounds with debate and moderator model selections.
- Build run transcripts and continue chat after structured workflow steps.
- Enforce idempotency for repeated workflow-start requests.

## Module Map

- `service.py`: workflow template and run service, idempotency handling, state transitions, answer recording, and transcript behavior.
- `dialogue_orchestrator.py`: debate/moderator LLM orchestration and moderator decision parsing.
- `question_renderer.py`: question rendering helper for workflow steps.
- `__init__.py`: package marker.

## How It Connects

- The endpoint uses AuthNZ permissions for chat workflow read, run, and write operations.
- `chat_workflows_deps.py` resolves the per-user chat workflows database path through `DatabasePaths` and caches database instances.
- Persistence is handled by `ChatWorkflowsDatabase` in the DB management layer.
- LLM calls route through the shared Chat module, including `chat_orchestrator.chat_api_call_async`.
- Context reference fields are preserved in schemas, templates, and runs, but current draft generation does not resolve content sources or pass `context_refs` into the service; run start stores `selected_context_refs` with an empty `resolved_context_snapshot`.
- API schemas define LLM selection, dialogue config, template steps, draft requests, run requests, answers, rounds, transcripts, and continue-chat responses.

## Architecture Notes

### Core Flow

- Template and run requests enter through `chat_workflows.py`, resolve the per-user `ChatWorkflowsDatabase` in `chat_workflows_deps.py`, then call `ChatWorkflowService`.
- `service.py` owns run lifecycle: create or reuse an idempotent run, render the current step through `question_renderer.py`, record answers, advance steps, and build transcripts.
- Dialogue workflows call `dialogue_orchestrator.py`, which runs debate and moderator turns through the shared Chat orchestrator instead of direct provider adapters.
- Dialogue rounds use `ChatWorkflowsDatabase.begin_dialogue_round` and `complete_dialogue_round` to distinguish fresh, replayed, stale, failed, and conflicting idempotent submissions.

### State And Data

- `DB_Management/ChatWorkflows_DB.py` owns templates, template steps, runs, answers, run events, and dialogue rounds. Idempotency keys are enforced in the database and mirrored by service-level conflict handling.
- `selected_context_refs` are stored with runs and passed into dialogue orchestration, while `resolved_context_snapshot` remains empty unless a caller adds source resolution.
- Database instances are cached per user by dependency helpers, so tests that override paths should use the existing dependency override fixtures.

### Security And Operations

- Permission boundaries live at the endpoint and dependency layer. Service code assumes it is called with a user-scoped database handle and should not open a global workflow DB.
- Moderator decisions are parsed from model output. Treat parser changes as behavior changes and cover malformed, continue, and finish cases.
- Repeated answer or dialogue requests must stay idempotent for matching keys and reject key reuse with different content.

### Extension Checklist

- New template or run field: update `schemas/chat_workflows.py`, `ChatWorkflows_DB.py`, `service.py`, and schema/database tests.
- New dialogue policy: update `dialogue_orchestrator.py`, moderator parsing tests, and API tests that exercise dialogue rounds.
- New context resolution: update endpoint request handling, `service.py`, dependency wiring, and tests for both stored `selected_context_refs` and resolved snapshots.

## Extension Points

- Add template fields or validation in `schemas/chat_workflows.py`, then update `service.py` and schema tests.
- Add a new dialogue policy in `dialogue_orchestrator.py`.
- Change step rendering in `question_renderer.py`.
- Extend run lifecycle behavior in `service.py` before changing endpoint handlers.
- If adding context source resolution, start in the service and dependency path around `service.py`, `chat_workflows_deps.py`, and the endpoint draft/run handlers.

## Testing

- Direct tests live under `tldw_Server_API/tests/Chat_Workflows/`.
- Use `test_chat_workflows_service.py` for service behavior.
- Use `test_chat_workflows_api.py` for endpoint behavior.
- Use `test_chat_workflows_dialogue_orchestrator.py` for moderated dialogue behavior.
- Use `test_chat_workflows_schemas.py` and `test_chat_workflows_db.py` for schemas and persistence.

## Gotchas

- The endpoint file declares a prefix that already includes `/api/v1`; check router mounting before adding or moving routes.
- Moderator responses are expected to parse into the configured decision shape, including continue or finish decisions.
- Database instances are cached per user, so tests that change paths should use the existing dependency fixtures.
