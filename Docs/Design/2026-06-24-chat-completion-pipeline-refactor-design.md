# Chat Completion Pipeline Refactor Design

Backlog task: `TASK-12013`

## Purpose

Refactor the Chat completion execution path so safety, persistence, streaming, tool execution, command authorization, and response shaping are owned by focused modules instead of one large `chat_service.py` flow.

The implementation must preserve the public `/api/v1/chat/completions` API, exported `chat_service.py` entry points, response metadata, persistence behavior, and SSE event shapes except where the validated fixes intentionally reject or sanitize unsafe behavior.

## Validated Findings

The first implementation phase must fix these current-code findings:

1. Non-streaming provider responses can include multiple choices, but current post-processing only handles `choices[0]` while request schema allows `n` up to 128.
2. Chat paths log raw user messages and full prompt/system content.
3. Document prompt versioning uses a table-level `UNIQUE(document_type, is_active)` constraint that fails after repeated prompt saves.
4. Slash commands advertise required permissions but skip enforcement unless a separate flag is enabled.
5. The exported legacy chat-history replacement helper can soft-delete existing messages before replacement can safely complete. It currently appears to have no application callers in this checkout, but it remains exported and documented as a legacy utility.

## Scope

In scope:

- Non-streaming Chat completion response processing.
- Shared output moderation and redaction decisions.
- Streaming pipeline orchestration boundaries without rewriting `StreamingResponseHandler`.
- Tool auto-execution and continuation handoff boundaries.
- Slash-command authorization.
- Document prompt active-version storage.
- Legacy history replacement safety.
- Chat logging hygiene.
- Chat module architecture documentation.

Out of scope:

- Frontend changes.
- Chatbooks, Character Chat, Chat Workflows, Workflows, and MCP changes unless a direct compatibility call site requires a narrow adapter.
- Changing provider adapters except where needed for the Chat pipeline contract.
- Changing public Chat response shapes, except for intentional safety failures.
- Replacing `StreamingResponseHandler` internals.

## Architecture

`chat_service.py` remains the import-facing compatibility facade. Existing endpoint imports and helper call sites continue to work. The implementation introduces a thin internal orchestration object, tentatively `ChatCompletionPipeline`, that delegates to focused modules.

New modules under `tldw_Server_API/app/core/Chat/`:

- `response_processor.py`: walks provider response choices, normalizes assistant content, applies structured validation, asks moderation for decisions, mutates redacted returned payloads, injects assistant names where currently supported, and returns explicit processing metadata.
- `moderation_pipeline.py`: owns policy lookup, output moderation decisions, redaction decisions, self-monitoring, topic monitoring, audit writes, and moderation review capture. It does not traverse provider payloads.
- `persistence_service.py`: owns assistant, system, user, and legacy history persistence transaction boundaries. It keeps current persistence behavior unless a validated fix requires safer ordering.
- `streaming_pipeline.py`: owns stream orchestration glue around provider streams, `StreamingResponseHandler`, moderation holdback, save callback setup, audit handling, and stream error mapping. It should not rewrite `StreamingResponseHandler` internals in the first pass.
- `tool_execution_service.py`: owns legacy local tool auto-execution, tool continuation, result payload assembly, and early rejection of unsupported multi-choice tool-auto-exec requests.
- `command_authorization.py`: owns command authorization decisions for dispatch and listing so those paths cannot drift.
- `chat_logging.py`: owns content-free logging helpers for user messages, prompts, choices, and provider metadata.

`document_generator.py` may either delegate prompt storage to a small `document_prompt_store.py` helper or keep a focused private helper if adding a module would create unnecessary coupling.

Import rule: new modules must not import `chat_service.py`. `chat_service.py` may import them as the facade. Shared dataclasses should live in focused modules or a small neutral module to avoid circular imports.

## Non-Streaming Data Flow

1. The FastAPI endpoint continues to validate the request and build `cleaned_args`.
2. `chat_service.execute_non_stream_call` remains callable with the same signature.
3. `chat_service.py` performs provider selection, queue/fallback setup, prompt cost guardrails, and the provider call as it does today.
4. Unsupported modes are rejected before the provider call when possible. In particular, local tool auto-exec with `n > 1` fails fast with HTTP 400 because executing one of several assistant choices is ambiguous.
5. Raw provider response is passed to `ChatCompletionPipeline`, which calls `response_processor.process_non_stream_response`.
6. `response_processor` walks every returned choice and delegates policy decisions to `moderation_pipeline`.
7. The processor returns:
   - the mutated response payload,
   - blocked status,
   - redacted choice indexes,
   - structured-validation result,
   - persisted choice index,
   - tool-execution choice index,
   - usage-estimation metadata.
8. `persistence_service` persists the first assistant choice only, matching current behavior. Multi-choice persistence is explicitly not supported in this pass.
9. `tool_execution_service` handles the supported first-choice tool-call path.
10. `chat_service.py` appends existing `tldw_*` response metadata and returns the same JSON shape.

## Multi-Choice Rules

Plain non-streaming responses:

- Return all choices.
- Moderate, redact, and self-monitor every choice.
- If any choice is blocked, block the whole response using the current non-streaming moderation error behavior.
- If choices are redacted, only the affected choice content is mutated.

Structured output:

- Validate every returned choice.
- If any choice fails validation, fail the whole response before persistence.
- Error details include failed choice indexes and concise validation context, not raw response content.

Tool auto-exec:

- Reject `n > 1` before provider call when local tool auto-exec is enabled.
- Keep first-choice-only behavior for `n == 1`.

Persistence:

- Persist the first assistant choice only, as today.
- Do not add multi-choice transcript storage in this pass.

Usage estimation:

- When provider usage exists, keep the current normalizer path and pass the full choices list.
- When provider usage is missing, estimate completion tokens from all returned assistant choices, not only the persisted first choice.

## Streaming Data Flow

1. `chat_service.py` still creates the FastAPI streaming response facade.
2. `streaming_pipeline.py` wires provider stream, `StreamingResponseHandler`, moderation holdback, audit behavior, topic monitoring, and save callback setup.
3. `moderation_pipeline.py` is shared with non-streaming, but streaming decisions operate on chunk windows plus final-save validation.
4. Existing SSE event shapes remain stable. Existing unsafe output may now be blocked or redacted through the shared decision layer.
5. The first implementation pass should move orchestration and moderation wiring only. `StreamingResponseHandler` internals remain mostly unchanged unless a failing test proves a targeted change is required.

## Command Authorization

Command dispatch must fail closed when a command declares `required_permission`.

Rules:

- In authenticated multi-user mode, users must have the declared permission.
- In single-user API-key mode, the owner/admin context may satisfy declared command permissions through the existing AuthNZ owner/admin model.
- Anonymous, malformed, or incomplete `CommandContext` values do not silently grant permission.
- Command listing and command dispatch use the same authorization decision object.
- Authenticated user-facing listings may filter unauthorized commands when permission enforcement is active. Admin/internal metadata paths may still expose `required_permission` and `rbac_required` for discoverability.
- Dispatch preserves the existing `CommandResult(ok=False, metadata={"error": "permission_denied"})` shape on denial.

## Document Prompt Versioning

The prompt storage fix must support repeated prompt saves for a single document type while preserving exactly one active prompt.

Preferred SQLite shape:

- Remove the table-level `UNIQUE(document_type, is_active)` constraint.
- Add a partial unique index enforcing only one active prompt:
  `CREATE UNIQUE INDEX ... ON user_prompts(document_type) WHERE is_active = 1`.

Migration notes:

- SQLite cannot drop the existing table-level unique constraint in place.
- The implementation should rebuild the table or otherwise migrate to a compatible shape without losing existing prompt rows.
- Tests must use a temporary SQLite database and prove three or more consecutive saves succeed with exactly one active prompt.
- Prompt content must not be logged during migration or save failures.

## Legacy History Replacement

Keep `save_chat_history_to_db_wrapper` exported with the same signature.

Target behavior:

- Validate replacement input before deleting existing messages.
- If `CharactersRAGDB.transaction()` covers the relevant operations, soft-delete old messages and insert replacement messages in one transaction.
- If DB helper internals commit independently, avoid claiming full atomicity. In that case, improve the ordering, document the remaining DB limitation in code comments, and cover the safest achievable behavior with tests.
- If replacement fails, old messages should remain intact whenever transaction semantics support it.

## Logging

Chat logs must not include:

- Raw user messages.
- Full system prompts.
- Custom prompts.
- Tool arguments.
- API keys.
- Generated assistant content.

Preferred metadata:

- request id,
- conversation id,
- provider,
- model,
- content length,
- choice count,
- redaction/block flags,
- image/attachment presence.

Stable content hashes are allowed only behind explicit debug settings and must not become default log output.

## Error Handling

Use existing exception surfaces where callers already expect them:

- `HTTPException` for request and moderation failures.
- `ChatProviderError` for provider failures.
- `MandatoryAuditWriteError` for mandatory audit persistence failures.
- Existing structured-output exception mapping for structured generation failures.

Do not introduce a new broad exception hierarchy in this pass.

Specific behavior:

- Non-streaming moderation block keeps the current HTTP 400 behavior.
- Streaming moderation block keeps the current SSE error-frame behavior and terminal frame behavior.
- Redaction mutates only affected returned content and records content-free internal metadata.
- Document prompt migration/save failures keep the current endpoint error behavior while logging sanitized context.
- Legacy history wrapper keeps its `(conversation_id, status_message)` return shape.

## Implementation Stages

Stage 1: Design and planning

- Write this design.
- Link it from `TASK-12013`.
- Self-review the spec for ambiguity, contradictions, placeholders, and scope.
- Ask for user review before implementation planning.

Stage 2: Behavior fixes with tests first

- Add failing tests for the validated issues.
- Fix multi-choice response processing.
- Fix sensitive logging.
- Fix command authorization.
- Fix document prompt versioning.
- Fix or improve legacy history replacement safety.

Stage 3: Extract non-streaming pipeline services

- Add `ChatCompletionPipeline`.
- Extract response processing, moderation decisions, persistence, tool execution, and safe logging behind stable `chat_service.py` calls.
- Keep public signatures stable.

Stage 4: Extract streaming orchestration glue

- Add `streaming_pipeline.py`.
- Move setup/wiring out of `chat_service.py`.
- Preserve `StreamingResponseHandler` behavior and SSE shapes.

Stage 5: Documentation and cleanup

- Update `tldw_Server_API/app/core/Chat/README.md`.
- Update `tldw_Server_API/app/core/Chat/REFACTORING_PLAN.md`.
- Record final verification in `TASK-12013`.

## Testing Strategy

All behavior fixes must follow test-first implementation. Each bug test must be observed failing for the expected reason before production code changes.

Required targeted tests:

- Non-streaming `n > 1` returns all choices and processes every choice.
- Every choice is moderated/redacted.
- A block in any choice blocks the whole non-streaming response.
- Missing provider usage estimates completion tokens from all assistant choices.
- Structured output validates every choice.
- Tool auto-exec rejects `n > 1` before provider call.
- Raw user input and full prompt/system content are not emitted in Chat logs.
- Commands with `required_permission` deny by default without permission.
- Granted users can execute command paths.
- Single-user owner/admin context can execute declared-permission commands.
- Command listing and dispatch use the same authorization decision.
- Three or more consecutive document prompt saves succeed in a real SQLite test.
- Exactly one active document prompt remains per document type.
- Legacy history replacement failure leaves existing messages intact where DB transaction semantics support it.
- Existing streaming response frames remain compatible after extraction.

Regression checks:

- Targeted Chat unit tests touched by the refactor.
- Chat command unit/integration tests.
- Document generation endpoint tests.
- Relevant `tests/Chat` integration tests after targeted checks pass.
- Bandit over touched Chat files before completion.

## Compatibility Guarantees

- Keep `/api/v1/chat/completions` request and response shapes stable.
- Keep exported `chat_service.py` functions importable.
- Keep `tldw_*` metadata fields stable.
- Keep SSE frame shapes stable.
- Keep first-choice persistence behavior for non-streaming responses.
- Keep command denial result shape stable.
- Keep legacy history wrapper signature stable.

Intentional behavior changes:

- Unsafe `n > 1` modes reject before provider calls.
- All non-streaming choices are moderated/redacted/validated.
- Commands with declared permissions fail closed.
- Logs no longer include sensitive Chat content.
- Document prompt versioning supports repeated saves.

## Risks And Mitigations

Risk: circular imports between new services and `chat_service.py`.

Mitigation: new modules must not import `chat_service.py`; the facade imports services.

Risk: broad extraction changes response behavior.

Mitigation: behavior fixes land first, then extraction stages must preserve tests and response fixtures.

Risk: streaming regressions.

Mitigation: move only orchestration wiring first; keep `StreamingResponseHandler` internals stable.

Risk: command authorization breaks single-user local setups.

Mitigation: explicitly support single-user owner/admin permission resolution while denying anonymous malformed contexts.

Risk: document prompt migration loses data.

Mitigation: use real SQLite migration tests and avoid logging prompt content.

Risk: structured-output multi-choice failures expose content.

Mitigation: report failed choice indexes and concise validation context only.

## Spec Self-Review

- Placeholder scan: no placeholders, TODOs, or open-ended requirements remain.
- Internal consistency: module boundaries distinguish traversal, policy decisions, persistence, and orchestration.
- Scope check: the design is broad but limited to Chat completion execution paths and direct compatibility call sites.
- Ambiguity check: multi-choice, command authorization, document prompt migration, logging, and streaming boundaries are explicit.
