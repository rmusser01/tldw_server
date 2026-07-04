# Chat Macros Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the v1 Chat Macros system with `/wrapup` as a built-in macro that can run against normal chat/workspace chat, fan out branch prompts, merge results, and post a structured final result back to the originating conversation.

**Architecture:** Add a dedicated `Chat_Macros` backend package that owns macro definitions, validation, storage, settings, runs, execution, and Jobs integration. Reuse ChaChaNotes for durable macro records, Skills-style filesystem safety for user macro files, the existing chat command router as the slash-command entry point, and the existing WebUI chat message path for final post-back. Ship the first frontend slice as status/final rendering plus run detail/cancel controls, not a full macro authoring IDE.

**Tech Stack:** FastAPI, Pydantic v2, SQLite via `CharactersRAGDB`/ChaChaNotes, existing `JobManager`/`WorkerSDK`, pytest, Vitest/React Testing Library, `apiSend`, existing chat/workspace React components.

---

## Source Documents

- Spec: `Docs/superpowers/specs/2026-07-03-chat-macros-design.md`
- Backlog: `TASK-12125`
- Related design Backlog task: `TASK-12124`. `TASK-12125` intentionally tracks only this implementation-plan artifact.
- Implementation Backlog task: `TASK-12126`. Use that task for all code/frontend/docs implementation work before editing implementation files.
- Existing patterns to follow:
  - `tldw_Server_API/app/core/Chat/command_router.py`
  - `tldw_Server_API/app/core/Skills/skills_service.py`
  - `tldw_Server_API/app/core/Jobs/worker_sdk.py`
  - `tldw_Server_API/app/services/study_pack_jobs_worker.py`
  - `tldw_Server_API/app/api/v1/endpoints/skills.py`
  - `apps/packages/ui/src/services/api-send.ts`
  - `apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceChatPanel.tsx`

## File Structure

Backend core:

- Create `tldw_Server_API/app/core/Chat_Macros/__init__.py`: public package exports.
- Create `tldw_Server_API/app/core/Chat_Macros/exceptions.py`: narrow exception classes and safe public error codes.
- Create `tldw_Server_API/app/core/Chat_Macros/models.py`: Pydantic/domain models for macro definitions, steps, settings, runs, branch records, output profiles, and invocation args.
- Create `tldw_Server_API/app/core/Chat_Macros/parser.py`: YAML loading, shell-style slash arg parsing, command validation, alias normalization, and schema validation.
- Create `tldw_Server_API/app/core/Chat_Macros/storage.py`: file-backed macro definition IO using Skills-style path safety.
- Create `tldw_Server_API/app/core/Chat_Macros/repository.py`: ChaChaNotes-backed registry/run/settings DAO.
- Create `tldw_Server_API/app/core/Chat_Macros/settings.py`: default macro settings and output profile merging.
- Create `tldw_Server_API/app/core/Chat_Macros/output_profiles.py`: final output rendering and failed branch formatting.
- Create `tldw_Server_API/app/core/Chat_Macros/context_snapshot.py`: bounded chat/workspace/ACP context snapshot construction.
- Create `tldw_Server_API/app/core/Chat_Macros/branch_runner.py`: branch prompt interface and test fake seams.
- Create `tldw_Server_API/app/core/Chat_Macros/executor.py`: run lifecycle, branch fan-out, retries, merge, final output persistence, idempotent post-back.
- Create `tldw_Server_API/app/core/Chat_Macros/jobs.py`: job enqueue helpers and worker handler.
- Create `tldw_Server_API/app/core/Chat_Macros/acp_adapter.py`: v1 fallback stubs and ACP branch metadata hooks.
- Create `tldw_Server_API/app/core/Chat_Macros/builtin/wrapup/MACRO.yaml`: immutable bundled `/wrapup`.
- Create `tldw_Server_API/app/core/Chat_Macros/README.md`: module boundary and v1 limitations.

Backend integration:

- Modify `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`: add macro registry/settings/run/branch tables, migration, and DAO helpers or thin table ensure methods.
- Create `tldw_Server_API/app/api/v1/schemas/chat_macros.py`: request/response schemas.
- Create `tldw_Server_API/app/api/v1/API_Deps/Chat_Macros_Deps.py`: `ChatMacrosService` dependency using current user, user base dir, ChaChaNotes DB, and optional Jobs manager.
- Create `tldw_Server_API/app/api/v1/endpoints/chat_macros.py`: CRUD/settings/run/cancel endpoints.
- Modify `tldw_Server_API/app/api/v1/router_groups/core.py`: include the chat macros router under `/api/v1/chat/macros`.
- Modify `tldw_Server_API/app/core/Chat/command_router.py`: expose core-command precedence and slash candidate parsing without forcing every macro into `_registry`.
- Modify `tldw_Server_API/app/api/v1/endpoints/chat.py`: short-circuit macro invocations before ordinary LLM completion; do not inject macro output as command context.
- Create `tldw_Server_API/app/services/chat_macros_jobs_worker.py`: macro Jobs worker loop.
- Modify `tldw_Server_API/app/services/startup_content_jobs_pollers.py`: register optional macro worker, guarded by `CHAT_MACROS_JOBS_WORKER_ENABLED`.

Backend tests:

- Create `tldw_Server_API/tests/Chat_Macros/unit/test_macro_parser.py`.
- Create `tldw_Server_API/tests/Chat_Macros/unit/test_macro_repository.py`.
- Create `tldw_Server_API/tests/Chat_Macros/unit/test_macro_storage.py`.
- Create `tldw_Server_API/tests/Chat_Macros/unit/test_macro_service.py`.
- Create `tldw_Server_API/tests/Chat_Macros/unit/test_macro_executor.py`.
- Create `tldw_Server_API/tests/Chat_Macros/unit/test_macro_jobs.py`.
- Create `tldw_Server_API/tests/Chat_Macros/integration/test_chat_macros_api.py`.
- Add focused cases to `tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py`.
- Add focused cases to `tldw_Server_API/tests/Chat_NEW/integration/test_chat_completions_api.py`.
- Add startup/shutdown cases near `tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py` or a new `tldw_Server_API/tests/Services/test_chat_macros_jobs_worker_startup.py`.

Frontend:

- Create `apps/packages/ui/src/services/chat-macros.ts`: typed client for list/settings/run/get/cancel.
- Create `apps/packages/ui/src/services/__tests__/chat-macros.test.ts`.
- Create `apps/packages/ui/src/components/Option/ChatWorkspace/MacroStatusCard.tsx`: status/final metadata renderer for chat-workspace messages.
- Create `apps/packages/ui/src/components/Option/ChatWorkspace/MacroRunDetailDrawer.tsx`: run detail and cancel action.
- Modify `apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceChatPanel.tsx`: render macro metadata status/final states when present.
- Create `apps/packages/ui/src/components/Option/Settings/ChatMacrosSettings.tsx`: minimal macro manager/settings surface.
- Create `apps/packages/ui/src/components/Option/Settings/__tests__/ChatMacrosSettings.test.tsx`.
- Modify `apps/packages/ui/src/routes/option-settings-route-registry.tsx`: register `/settings/chat-macros`.
- Modify `apps/packages/ui/src/routes/route-registry.tsx`: register the same settings route if this registry remains active for the current app shell.
- Create `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/MacroStatusCard.test.tsx`.
- Create `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/MacroRunDetailDrawer.test.tsx`.
- Add focused cases to `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.test.tsx`.

## Task 1: Core Macro Models, Parser, And Built-In `/wrapup`

**Files:**
- Create: `tldw_Server_API/app/core/Chat_Macros/__init__.py`
- Create: `tldw_Server_API/app/core/Chat_Macros/exceptions.py`
- Create: `tldw_Server_API/app/core/Chat_Macros/models.py`
- Create: `tldw_Server_API/app/core/Chat_Macros/parser.py`
- Create: `tldw_Server_API/app/core/Chat_Macros/builtin/wrapup/MACRO.yaml`
- Create: `tldw_Server_API/app/core/Chat_Macros/README.md`
- Test: `tldw_Server_API/tests/Chat_Macros/unit/test_macro_parser.py`

- [x] **Step 1: Write failing parser/model tests**

Add tests for:

```python
def test_builtin_wrapup_loads_and_validates():
    macro = load_macro_definition(BUILTIN_WRAPUP_PATH.read_text())
    assert macro.command == "wrapup"
    assert [step.output for step in macro.steps if step.type == "branch_prompt"] == [
        "summary", "decisions", "action_items", "open_questions"
    ]

def test_non_empty_tool_or_skill_permissions_rejected():
    raw = "schema_version: 1\nname: bad\ncommand: bad\npermissions:\n  tool_calls: [shell]\nsteps: []\n"
    with pytest.raises(MacroValidationError, match="tool"):
        load_macro_definition(raw)

def test_parse_slash_args_normalizes_aliases_and_repeated_questions():
    spec = WrapupArgsSpec()
    args = parse_macro_args(
        '--preset dev_handoff --keep-forks --output-profile compact '
        '--question "What changed?" --question "What is next?"',
        spec,
    )
    assert args["keep_forks"] is True
    assert args["output_profile"] == "compact"
    assert args["question"] == ["What changed?", "What is next?"]
```

- [x] **Step 2: Run the new tests and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chat_Macros/unit/test_macro_parser.py -v
```

Expected: import errors for the new `Chat_Macros` modules.

- [x] **Step 3: Implement minimal models and parser**

Implement:

- `MacroValidationError`, `MacroStorageError`, `MacroNotFoundError`, `MacroExecutionError` in `exceptions.py`.
- Pydantic models in `models.py`: `MacroDefinition`, `MacroArgSpec`, `MacroStep`, `MacroPermissions`, `MacroExecution`, `MacroContext`, `OutputProfile`, `MacroRunRecord`, `MacroBranchRecord`.
- `load_macro_definition(raw: str) -> MacroDefinition` in `parser.py`.
- `parse_macro_args(raw: str | None, arg_specs: Mapping[str, MacroArgSpec], *, max_questions: int) -> dict[str, Any]` using `shlex.split`.
- Command validation pattern `^[a-z][a-z0-9_]{0,63}$`.
- Permission validation that rejects non-empty `tool_calls` and `skills`.
- Step validation that every `merge.consumes` and `post_result.consumes` target exists as a previous `output`.

- [x] **Step 4: Add the built-in `MACRO.yaml`**

Use the spec's `/wrapup` definition, with:

```yaml
schema_version: 1
name: wrapup
command: wrapup
description: Close out the current conversation or workspace.
enabled: true
builtin_version: 1
```

Keep `execution.branch_strategy: auto`, `max_branches: 6`, `retries_per_branch: 1`, and the four default branch prompts from the spec.

- [x] **Step 5: Run parser tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chat_Macros/unit/test_macro_parser.py -v
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Chat_Macros tldw_Server_API/tests/Chat_Macros/unit/test_macro_parser.py
git commit -m "feat: add chat macro definition parser"
```

## Task 2: ChaChaNotes Macro Tables And Repository

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Create: `tldw_Server_API/app/core/Chat_Macros/repository.py`
- Test: `tldw_Server_API/tests/Chat_Macros/unit/test_macro_repository.py`

- [x] **Step 1: Write failing repository tests**

Add tests that create a temporary `CharactersRAGDB` and assert:

```python
repo = ChatMacroRepository(db)
run = repo.create_run(user_id="1", macro_name="wrapup", macro_command="wrapup", normalized_args={})
assert repo.get_run(run.run_id).macro_name == "wrapup"

repo.store_final_output(run.run_id, final_output="Done", final_output_format="markdown")
repo.mark_final_posted(run.run_id, final_message_id="msg-1", post_idempotency_key="macro:run:post")
assert repo.get_run(run.run_id).final_message_id == "msg-1"

repo.upsert_branch(run.run_id, step_id="summary", label="Summary", status="completed", output_text="S")
assert repo.list_branches(run.run_id)[0].output_text == "S"
```

Also test that `post_idempotency_key` is unique per run/post target and that cancellation stores `cancel_requested_at`.

- [x] **Step 2: Run repository tests and verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chat_Macros/unit/test_macro_repository.py -v
```

Expected: import errors or missing table errors.

- [x] **Step 3: Add schema migration**

In `ChaChaNotes_DB.py`:

- Increment `_CURRENT_SCHEMA_VERSION` by one.
- Add a migration from the previous version to the new version.
- Create tables:
  - `chat_macro_registry`
  - `chat_macro_settings`
  - `chat_macro_runs`
  - `chat_macro_run_branches`
- Add indexes on `(user_id, command)`, `(user_id, status, created_at)`, `(run_id, step_id)`, and unique `(run_id, post_idempotency_key)` where possible.
- Keep fields JSON-compatible as `TEXT` columns containing canonical JSON where existing DB conventions do that.

- [x] **Step 4: Implement `ChatMacroRepository`**

Implement methods:

- `ensure_ready()`
- `upsert_registry_entry(...)`
- `list_registry_entries(user_id: str)`
- `get_settings(user_id: str)`
- `save_settings(user_id: str, settings: dict[str, Any])`
- `create_run(...) -> MacroRunRecord`
- `get_run(run_id: str) -> MacroRunRecord | None`
- `update_run_status(...)`
- `request_cancel(run_id: str)`
- `upsert_branch(...)`
- `list_branches(run_id: str)`
- `store_final_output(...)`
- `mark_final_posted(...)`

Use parameterized SQL only. Do not expose raw connection usage outside this repository.

- [x] **Step 5: Run repository tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chat_Macros/unit/test_macro_repository.py -v
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/Chat_Macros/repository.py tldw_Server_API/tests/Chat_Macros/unit/test_macro_repository.py
git commit -m "feat: store chat macro runs in chacha notes"
```

## Task 3: File-Backed Macro Storage, Registry Sync, And Settings

**Files:**
- Create: `tldw_Server_API/app/core/Chat_Macros/storage.py`
- Create: `tldw_Server_API/app/core/Chat_Macros/settings.py`
- Create: `tldw_Server_API/app/core/Chat_Macros/output_profiles.py`
- Create: `tldw_Server_API/app/core/Chat_Macros/service.py`
- Test: `tldw_Server_API/tests/Chat_Macros/unit/test_macro_storage.py`
- Test: `tldw_Server_API/tests/Chat_Macros/unit/test_macro_service.py`

- [x] **Step 1: Write failing storage and settings tests**

Cover:

- User macros live under `Databases/user_databases/<user_id>/macros/<macro_name>/MACRO.yaml`.
- Path traversal and symlinks are rejected.
- Built-in `/wrapup` is listed and immutable.
- A built-in can be disabled in the registry.
- Cloning a built-in creates a user-owned macro with a non-conflicting command.
- Default output profile renders summary/decisions/action_items/open_questions/failed_branches in order.

- [x] **Step 2: Run tests and verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chat_Macros/unit/test_macro_storage.py \
  tldw_Server_API/tests/Chat_Macros/unit/test_macro_service.py \
  -v
```

Expected: import errors.

- [x] **Step 3: Implement storage**

Use the Skills service as a reference, but keep the domain separate:

- Macro name pattern: `^[a-z][a-z0-9_]{0,63}$`.
- Supporting file names: conservative basename-only names.
- Reject symlinked macro directories and supporting files.
- Bound `MACRO.yaml` bytes and total supporting file bytes.
- Compute a digest from canonical macro content plus supporting file metadata.

- [x] **Step 4: Implement service and settings**

`ChatMacrosService` should:

- List built-in and user macros with core-command collision checks.
- Validate a macro without saving it.
- Create/update/delete user macros.
- Clone built-ins.
- Disable built-ins per user.
- Resolve output profiles from settings, with macro-local overrides bounded by global caps.
- Reject non-empty future permissions.

- [x] **Step 5: Run tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chat_Macros/unit/test_macro_storage.py tldw_Server_API/tests/Chat_Macros/unit/test_macro_service.py -v
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Chat_Macros tldw_Server_API/tests/Chat_Macros/unit/test_macro_storage.py tldw_Server_API/tests/Chat_Macros/unit/test_macro_service.py
git commit -m "feat: add chat macro storage service"
```

Completed in commits `80eeb8c9e1`, `a2f7ca82ee`, `8f48bd6bc8`, `c826582720`, and `9972f2b768`.
Verification:
- Storage/service tests: 14 passed, 3 warnings.
- Parser/repository regressions: 26 passed, 3 warnings.
- Bandit on `tldw_Server_API/app/core/Chat_Macros`: JSON results empty.
- `git diff --check`: clean.
Reviews: spec and code-quality re-reviews found no findings at `9972f2b768`.

## Task 4: Chat Macros API

**Files:**
- Create: `tldw_Server_API/app/api/v1/schemas/chat_macros.py`
- Create: `tldw_Server_API/app/api/v1/API_Deps/Chat_Macros_Deps.py`
- Create: `tldw_Server_API/app/api/v1/endpoints/chat_macros.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/core.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/minimal.py`
- Test: `tldw_Server_API/tests/Chat_Macros/integration/test_chat_macros_api.py`

- [x] **Step 1: Write failing API tests**

Use the existing FastAPI test client fixtures. Cover:

- `GET /api/v1/chat/macros` returns built-in `/wrapup`.
- `GET /api/v1/chat/macros/{name}` returns macro detail.
- `POST /api/v1/chat/macros` creates a user macro.
- `PUT /api/v1/chat/macros/{name}` updates a user macro.
- `DELETE /api/v1/chat/macros/{name}` soft-deletes or disables a user macro according to service behavior.
- `POST /api/v1/chat/macros/validate` rejects tool permissions.
- `GET/PUT /api/v1/chat/macros/settings` round-trips output profiles.
- `POST /api/v1/chat/macros/{name}/clone` creates a user macro.
- `POST /api/v1/chat/macros/run` creates a run in background mode and returns a run ID.
- `GET /api/v1/chat/macros/runs/{run_id}` returns run detail with branch summaries and redacted-safe errors.
- `POST /api/v1/chat/macros/runs/{run_id}/cancel` records cancellation.

- [x] **Step 2: Run API tests and verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chat_Macros/integration/test_chat_macros_api.py -v
```

Expected: 404 or import errors.

- [x] **Step 3: Implement schemas**

Include:

- `ChatMacroSummary`
- `ChatMacroDetail`
- `ChatMacroValidateRequest/Response`
- `ChatMacroSettingsRequest/Response`
- `ChatMacroRunRequest/Response`
- `ChatMacroRunDetailResponse`
- `ChatMacroCancelResponse`

Keep response payloads stable and frontend-friendly: status strings, run IDs, branch summaries, output profile names, and safe errors.

- [x] **Step 4: Implement dependency and endpoints**

`get_chat_macros_service(...)` should use:

- `CurrentPrincipal`
- `get_chacha_db_for_user`
- `DatabasePaths.get_user_base_directory(user_id)`
- `try_get_job_manager` when available

Endpoints should map domain exceptions to `400`, `404`, `409`, `413`, or `500` without leaking raw provider errors.

- [x] **Step 5: Register router**

In `router_groups/core.py`, append a `RouterSpec` for:

```python
import_path="tldw_Server_API.app.api.v1.endpoints.chat_macros"
prefix=f"{API_V1_PREFIX}/chat/macros"
tags=("chat-macros",)
route_key="chat-macros"
```

- [x] **Step 6: Run API tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chat_Macros/integration/test_chat_macros_api.py -v
```

Expected: PASS.

Verification:

- `python -m pytest tldw_Server_API/tests/Chat_Macros/integration/test_chat_macros_api.py -v` -> 3 passed.
- `python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k "minimal_test_router_specs or minimal_required_router_specs" -v` -> 10 passed, 166 deselected.
- `python -m pytest tldw_Server_API/tests/Chat_Macros -v` -> 43 passed.
- `python -m py_compile tldw_Server_API/app/api/v1/endpoints/chat_macros.py tldw_Server_API/app/api/v1/schemas/chat_macros.py tldw_Server_API/app/api/v1/API_Deps/Chat_Macros_Deps.py` -> passed.
- Review fixes: broadened run/branch error redaction for bearer/header/JSON-style secrets and persisted the resolved output profile name after fallback.
- `python -m bandit -r tldw_Server_API/app/api/v1/endpoints/chat_macros.py tldw_Server_API/app/api/v1/API_Deps/Chat_Macros_Deps.py tldw_Server_API/app/api/v1/schemas/chat_macros.py tldw_Server_API/app/api/v1/router_groups/minimal.py tldw_Server_API/app/api/v1/router_groups/core.py -f json -o /tmp/bandit_chat_macros_task4_api_reviewfix.json` -> no findings.
- `git diff --check` -> passed.

- [x] **Step 7: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/chat_macros.py tldw_Server_API/app/api/v1/API_Deps/Chat_Macros_Deps.py tldw_Server_API/app/api/v1/endpoints/chat_macros.py tldw_Server_API/app/api/v1/router_groups/core.py tldw_Server_API/tests/Chat_Macros/integration/test_chat_macros_api.py
git commit -m "feat: expose chat macros api"
```

Committed as `874ef6112d`.

## Task 5: Slash Command Entry Point And Chat Completion Short-Circuit

**Files:**
- Modify: `tldw_Server_API/app/core/Chat/command_router.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat.py`
- Test: `tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py`
- Test: `tldw_Server_API/tests/Chat_NEW/integration/test_chat_completions_api.py`

- [x] **Step 1: Write failing command-router tests**

Add tests for:

```python
def test_extract_slash_candidate_returns_unknown_macro_candidate():
    assert command_router.extract_slash_candidate("/wrapup --preset dev") == ("wrapup", "--preset dev")

def test_core_parse_still_only_returns_registered_core_commands():
    assert command_router.parse_slash_command("/time") == ("time", None)
    assert command_router.parse_slash_command("/wrapup") is None
```

- [x] **Step 2: Write failing chat endpoint macro tests**

Add integration tests with a fake `ChatMacrosService` seam asserting:

- `/time` still uses existing injection behavior.
- `/wrapup` invokes macro service, returns a status assistant message, and does not call the ordinary provider completion path.
- Unknown `/not_a_macro` preserves current behavior.
- Macro invalid args return a chat-visible error without creating a run.

- [x] **Step 3: Run tests and verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py::test_extract_slash_candidate_returns_unknown_macro_candidate \
  tldw_Server_API/tests/Chat_NEW/integration/test_chat_completions_api.py \
  -v
```

Expected: missing `extract_slash_candidate` and missing macro short-circuit.

- [x] **Step 4: Add slash candidate parsing**

In `command_router.py`:

- Add `extract_slash_candidate(message: str) -> tuple[str, str | None] | None` using the existing `SLASH_RE`.
- Keep `parse_slash_command` unchanged for registered core commands.
- Add a small helper exposing reserved core command names for collision checks.

- [x] **Step 5: Add chat endpoint macro short-circuit**

In `chat.py`, after finding the latest user message:

1. Check `parse_slash_command(last_text)` first and keep existing core command flow.
2. If no core command, call `extract_slash_candidate(last_text)`.
3. Ask `ChatMacrosService` whether the command resolves to an enabled macro.
4. If yes, invoke macro run creation and return a non-streaming chat completion shape containing a status/final message plus macro metadata.
5. Do not pass macro output into `build_injection_text`.
6. If the incoming request is streaming, return a safe non-streaming macro response or a small SSE completion compatible with existing client expectations; cover whichever behavior is chosen in tests.

- [x] **Step 6: Run command/chat tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py \
  tldw_Server_API/tests/Chat_NEW/integration/test_chat_completions_api.py \
  -v
```

Expected: PASS for touched command and chat completion tests.

- [x] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/Chat/command_router.py tldw_Server_API/app/api/v1/endpoints/chat.py tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py tldw_Server_API/tests/Chat_NEW/integration/test_chat_completions_api.py
git commit -m "feat: route chat macro slash commands"
```

## Task 6: Context Snapshot, Branch Execution, Merge, And Output Profiles

**Files:**
- Create: `tldw_Server_API/app/core/Chat_Macros/context_snapshot.py`
- Create: `tldw_Server_API/app/core/Chat_Macros/acp_adapter.py`
- Create: `tldw_Server_API/app/core/Chat_Macros/branch_runner.py`
- Create: `tldw_Server_API/app/core/Chat_Macros/executor.py`
- Modify: `tldw_Server_API/app/core/Chat_Macros/models.py`
- Modify: `tldw_Server_API/app/core/Chat_Macros/output_profiles.py`
- Modify: `tldw_Server_API/app/core/Chat_Macros/builtin/wrapup/MACRO.yaml`
- Test: `tldw_Server_API/tests/Chat_Macros/unit/test_macro_executor.py`
- Test: `tldw_Server_API/tests/Chat_Macros/unit/test_acp_adapter.py`

- [x] **Step 1: Write failing executor tests**

Use fake branch and merge callables. Cover:

- Snapshot captures conversation ID, workspace ID, selected message IDs, bounded excerpts, model selection, and output profile.
- Model/provider availability and token/cost caps fail early before branches start.
- `/wrapup --preset dev_handoff` selects the dev handoff preset branch set.
- Repeated `/wrapup --question ...` appends custom branches with generated IDs and labels.
- `/wrapup --include-branches` includes branch appendices only when the selected profile allows them.
- `/wrapup --sync` forces sync mode only under configured sync thresholds and fails early when the run is too large.
- Branches run up to `max_concurrency`.
- A failed branch retries once.
- Partial failure renders `failed_branches`.
- All branches failed renders a failure report, not a synthetic wrapup.
- Merge failure preserves successful branch outputs in the run detail.
- Final output is stored before post-back.
- ACP adapter records forkability metadata, falls back to chat-native execution when ACP is unavailable, and marks `acp_fork` required branches as failed according to policy.

- [x] **Step 2: Run executor tests and verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chat_Macros/unit/test_macro_executor.py -v
```

Expected: import errors.

- [x] **Step 3: Implement context snapshot builder**

Implement a bounded builder that accepts:

- `chat_db`
- `conversation_id`
- `workspace_id`
- `acp_session_id`
- latest request messages
- model/provider selection
- selected RAG/media IDs from request metadata when present

Return JSON-safe snapshot data. Never include secrets or raw provider keys.

- [x] **Step 4: Implement ACP adapter metadata seam**

Create `acp_adapter.py` with:

- `resolve_acp_branch_capability(snapshot: MacroContextSnapshot) -> AcpBranchCapability`
- `select_branch_strategy(step_strategy, macro_strategy, capability) -> BranchStrategyDecision`
- Chat-native fallback metadata when no resumable ACP session is present.
- Required-ACP failure metadata when `branch_strategy: acp_fork` cannot be satisfied.

V1 does not need full retained-fork UI before chat-native execution works, but it must preserve explicit ACP capability/fallback metadata in run and branch records.

- [x] **Step 5: Implement branch runner seams**

Define a protocol-like callable:

```python
class BranchPromptRunner(Protocol):
    async def run_branch(self, *, prompt: str, snapshot: MacroContextSnapshot, model_selection: dict[str, Any]) -> BranchPromptResult: ...
```

The production runner can initially call the existing chat completion/orchestrator seam with fakeable dependencies. Tests should use a fake runner and not call external providers.

- [x] **Step 6: Implement executor**

Executor responsibilities:

- Load run and macro definition by digest.
- Resolve `/wrapup` presets before branch planning.
- Apply repeated custom questions as additional bounded `branch_prompt` steps.
- Enforce `--sync`, `--include-branches`, and `--keep-forks` options against macro/settings caps.
- Check model/provider availability, token estimates, and cost caps before launching any branch.
- Call the ACP adapter before branch execution and store branch strategy/fallback metadata.
- Mark run running/completed/failed/cancelled.
- Execute `branch_prompt` steps concurrently under caps.
- Normalize every branch output into `{step_id, label, status, text, citations, usage}`.
- Apply retry policy per branch.
- Render merge prompt input from named outputs.
- Store final output before post-back.
- Respect `cancel_requested_at` before starting new branches.

- [x] **Step 7: Run executor tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chat_Macros/unit/test_macro_executor.py \
  tldw_Server_API/tests/Chat_Macros/unit/test_acp_adapter.py \
  -v
```

Expected: PASS.

- [x] **Step 8: Commit**

```bash
git add tldw_Server_API/app/core/Chat_Macros/context_snapshot.py tldw_Server_API/app/core/Chat_Macros/acp_adapter.py tldw_Server_API/app/core/Chat_Macros/branch_runner.py tldw_Server_API/app/core/Chat_Macros/executor.py tldw_Server_API/app/core/Chat_Macros/output_profiles.py tldw_Server_API/tests/Chat_Macros/unit/test_macro_executor.py tldw_Server_API/tests/Chat_Macros/unit/test_acp_adapter.py
git commit -m "feat: execute chat macro branches"
```

**Task 6 verification notes (2026-07-03):**
- Reviewer re-review: no findings. Residual risk remains for real Jobs/LLM/post-back integration in later slices.
- Focused executor/ACP suite: `29 passed, 3 warnings`.
- Full Chat_Macros suite: `72 passed, 4 warnings`.
- Command/chat regression slice: `50 passed, 9 skipped, 5 warnings`.
- Static/security: `compileall` exit 0, `git diff --check` exit 0, Bandit report `/tmp/bandit_chat_macros_task6_final.json` had empty `errors` and `results`.

## Task 7: Jobs Worker, Cancellation, And Idempotent Post-Back

**Files:**
- Create: `tldw_Server_API/app/core/Chat_Macros/jobs.py`
- Create: `tldw_Server_API/app/services/chat_macros_jobs_worker.py`
- Modify: `tldw_Server_API/app/services/startup_content_jobs_pollers.py`
- Test: `tldw_Server_API/tests/Chat_Macros/unit/test_macro_jobs.py`
- Test: `tldw_Server_API/tests/Services/test_chat_macros_jobs_worker_startup.py`

- [ ] **Step 1: Write failing Jobs tests**

Cover:

- `enqueue_chat_macro_run_job` creates a `JobManager` job with domain `chat_macros`, type `chat_macro_run`, and minimal payload `{macro_run_id, user_id, macro_digest, normalized_args}`.
- Job handler rejects wrong domain/type.
- Job handler loads run by ID and executes through `MacroExecutor`.
- Cancellation finalizes queued/running jobs and marks run cancelled.
- Successful post-back persists a visible assistant message through the normal chat message path with macro metadata.
- Duplicate post-back uses the existing visible assistant message for the same `post_idempotency_key` instead of creating a second message.

- [ ] **Step 2: Run Jobs tests and verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chat_Macros/unit/test_macro_jobs.py \
  tldw_Server_API/tests/Services/test_chat_macros_jobs_worker_startup.py \
  -v
```

Expected: import errors.

- [ ] **Step 3: Implement enqueue and handler**

In `jobs.py`:

- Constants: `CHAT_MACROS_DOMAIN = "chat_macros"`, `CHAT_MACROS_JOB_TYPE = "chat_macro_run"`.
- `chat_macro_jobs_queue()` reads `CHAT_MACROS_JOBS_QUEUE`, default `default`.
- `enqueue_chat_macro_run_job(...)`.
- `handle_chat_macro_job(job: dict[str, Any])`.
- `should_cancel_chat_macro_job(...)`.

- [ ] **Step 4: Implement worker**

Follow `study_pack_jobs_worker.py`:

- Use `WorkerConfig(domain=CHAT_MACROS_DOMAIN, queue=chat_macro_jobs_queue(), worker_id=...)`.
- Use `WorkerSDK.run(handler=handle_chat_macro_job, cancel_check=...)`.
- Fetch per-user ChaChaNotes DB with `get_chacha_db_for_user_id`.
- Close DB handles in `finally`.

- [ ] **Step 5: Register startup worker**

In `startup_content_jobs_pollers.py`:

- Add a `chat_macros_jobs_task` handle.
- Add a `stop_event_worker_spec` gated by `CHAT_MACROS_JOBS_WORKER_ENABLED` and route key `chat-macros`.
- Add `_run_chat_macros_jobs_worker_service`.

- [ ] **Step 6: Run Jobs tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chat_Macros/unit/test_macro_jobs.py \
  tldw_Server_API/tests/Services/test_chat_macros_jobs_worker_startup.py \
  -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/Chat_Macros/jobs.py tldw_Server_API/app/services/chat_macros_jobs_worker.py tldw_Server_API/app/services/startup_content_jobs_pollers.py tldw_Server_API/tests/Chat_Macros/unit/test_macro_jobs.py tldw_Server_API/tests/Services/test_chat_macros_jobs_worker_startup.py
git commit -m "feat: run chat macros through jobs"
```

## Task 8: Minimal Frontend Client, Settings Surface, And Workspace Chat Rendering

**Files:**
- Create: `apps/packages/ui/src/services/chat-macros.ts`
- Create: `apps/packages/ui/src/services/__tests__/chat-macros.test.ts`
- Create: `apps/packages/ui/src/components/Option/ChatWorkspace/MacroStatusCard.tsx`
- Create: `apps/packages/ui/src/components/Option/ChatWorkspace/MacroRunDetailDrawer.tsx`
- Modify: `apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceChatPanel.tsx`
- Create: `apps/packages/ui/src/components/Option/Settings/ChatMacrosSettings.tsx`
- Create: `apps/packages/ui/src/components/Option/Settings/__tests__/ChatMacrosSettings.test.tsx`
- Modify: `apps/packages/ui/src/routes/option-settings-route-registry.tsx`
- Modify: `apps/packages/ui/src/routes/route-registry.tsx`
- Test: `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/MacroStatusCard.test.tsx`
- Test: `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/MacroRunDetailDrawer.test.tsx`
- Test: `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.test.tsx`

- [ ] **Step 1: Write failing service tests**

Mock `apiSend` and verify:

```ts
await listChatMacros()
expect(apiSend).toHaveBeenCalledWith({ path: "/api/v1/chat/macros", method: "GET" })

await cancelChatMacroRun("run-1")
expect(apiSend).toHaveBeenCalledWith({ path: "/api/v1/chat/macros/runs/run-1/cancel", method: "POST" })

await setChatMacroEnabled("wrapup", false)
expect(apiSend).toHaveBeenCalledWith({ path: "/api/v1/chat/macros/wrapup", method: "PUT", body: expect.objectContaining({ enabled: false }) })
```

- [ ] **Step 2: Write failing component tests**

Cover:

- Running status renders macro name, branch count, output profile, and cancel button.
- Failed branch summary renders without raw provider exception details.
- Final macro output renders as normal assistant content plus metadata affordance.
- `WorkspaceChatPanel` chooses `MacroStatusCard` when `message.metadataExtra.chat_macro` is present.
- `ChatMacrosSettings` lists macros, toggles built-ins enabled/disabled, clones `/wrapup`, edits default output profile settings, and shows validation errors from the backend.

- [ ] **Step 3: Run frontend tests and verify failure**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/services/__tests__/chat-macros.test.ts \
  src/components/Option/Settings/__tests__/ChatMacrosSettings.test.tsx \
  src/components/Option/ChatWorkspace/__tests__/MacroStatusCard.test.tsx \
  src/components/Option/ChatWorkspace/__tests__/MacroRunDetailDrawer.test.tsx \
  src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.test.tsx
```

Expected: missing module/component failures.

- [ ] **Step 4: Implement `chat-macros.ts`**

Export:

- `listChatMacros`
- `getChatMacro`
- `createChatMacro`
- `updateChatMacro`
- `deleteChatMacro`
- `setChatMacroEnabled`
- `getChatMacroSettings`
- `updateChatMacroSettings`
- `runChatMacro`
- `getChatMacroRun`
- `cancelChatMacroRun`
- `cloneChatMacro`
- `validateChatMacro`

Use `apiSend` and keep payload types local until generated OpenAPI types are refreshed.

- [ ] **Step 5: Implement settings surface**

`ChatMacrosSettings` should provide the minimal required manager from the spec:

- List macros with source, command, enabled state, and validation state.
- Enable/disable built-ins and user macros through existing API helpers.
- Clone `/wrapup` into a user macro.
- Edit macro settings/output profiles through a simple JSON/YAML-like text area or structured fields, whichever is simpler in existing settings UI patterns.
- Show backend validation errors inline.
- Link to run history only if the API already exposes it in this slice; otherwise leave run detail available from status messages.

This first settings surface intentionally defers full user macro authoring/editing polish beyond clone, enable/disable, validation, and settings/output-profile editing. The advanced editor can land after the backend run path and minimal manager are stable.

- [ ] **Step 6: Register settings route**

Add `/settings/chat-macros` to `option-settings-route-registry.tsx`, and mirror it in `route-registry.tsx` if that registry still defines the active options route list. Include route tests if existing route metadata tests require coverage.

- [ ] **Step 7: Implement workspace chat components**

Use existing design language in `WorkspaceChatPanel`:

- No nested cards.
- Dense status layout.
- Cancel action only for running/queued runs.
- Run detail drawer fetches run detail lazily.
- Do not expose raw branch transcripts if backend redacts them.

- [ ] **Step 8: Wire `WorkspaceChatPanel`**

When a message has macro metadata:

- Render `MacroStatusCard` for queued/running/failed status messages.
- Render normal `PlaygroundMessage` for final assistant content, with a compact macro metadata control.
- Preserve existing non-macro message behavior.

- [ ] **Step 9: Run frontend tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/services/__tests__/chat-macros.test.ts \
  src/components/Option/Settings/__tests__/ChatMacrosSettings.test.tsx \
  src/components/Option/ChatWorkspace/__tests__/MacroStatusCard.test.tsx \
  src/components/Option/ChatWorkspace/__tests__/MacroRunDetailDrawer.test.tsx \
  src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.test.tsx
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add apps/packages/ui/src/services/chat-macros.ts apps/packages/ui/src/services/__tests__/chat-macros.test.ts apps/packages/ui/src/components/Option/Settings/ChatMacrosSettings.tsx apps/packages/ui/src/components/Option/Settings/__tests__/ChatMacrosSettings.test.tsx apps/packages/ui/src/routes/option-settings-route-registry.tsx apps/packages/ui/src/routes/route-registry.tsx apps/packages/ui/src/components/Option/ChatWorkspace/MacroStatusCard.tsx apps/packages/ui/src/components/Option/ChatWorkspace/MacroRunDetailDrawer.tsx apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceChatPanel.tsx apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/MacroStatusCard.test.tsx apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/MacroRunDetailDrawer.test.tsx apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.test.tsx
git commit -m "feat: add chat macro frontend controls"
```

## Task 9: End-To-End Verification, Security Sweep, And Docs

**Files:**
- Modify: `tldw_Server_API/app/core/Chat_Macros/README.md`
- Optionally modify: `Docs/Development/` or relevant chat docs if a chat command doc already exists.
- Backlog: update implementation task with commands and results.

- [ ] **Step 1: Run focused backend suite**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chat_Macros \
  tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py \
  tldw_Server_API/tests/Chat_NEW/integration/test_chat_completions_api.py \
  tldw_Server_API/tests/Services/test_chat_macros_jobs_worker_startup.py \
  -v
```

Expected: PASS.

- [ ] **Step 2: Run focused frontend suite**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/services/__tests__/chat-macros.test.ts \
  src/components/Option/Settings/__tests__/ChatMacrosSettings.test.tsx \
  src/components/Option/ChatWorkspace/__tests__/MacroStatusCard.test.tsx \
  src/components/Option/ChatWorkspace/__tests__/MacroRunDetailDrawer.test.tsx \
  src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.test.tsx
```

Expected: PASS.

- [ ] **Step 3: Run Bandit on touched backend scope**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/Chat_Macros \
  tldw_Server_API/app/api/v1/endpoints/chat_macros.py \
  tldw_Server_API/app/api/v1/schemas/chat_macros.py \
  tldw_Server_API/app/api/v1/API_Deps/Chat_Macros_Deps.py \
  tldw_Server_API/app/services/chat_macros_jobs_worker.py \
  -f json -o /tmp/bandit_chat_macros.json
```

Expected: no new findings in touched code. Fix new findings before continuing.

- [ ] **Step 4: Run OpenAPI/router smoke if backend routes changed**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Config/test_openapi_config_jobs.py -v
```

Expected: PASS or no unrelated regression.

- [ ] **Step 5: Manual smoke with local server**

Start server:

```bash
source .venv/bin/activate
python -m uvicorn tldw_Server_API.app.main:app --reload
```

In another terminal, send a normal chat command and a macro command against a local test conversation. Verify:

- `/time` still behaves as before.
- `/wrapup` returns macro status/final metadata and creates a run record.
- Cancelling a running run transitions run and job status.
- Re-running post-back for the same run does not create duplicate final messages.

- [ ] **Step 6: Update README/docs**

Document:

- v1 macro command names use word/underscore identifiers.
- `/wrapup` options: `--preset`, repeated `--question`, `--output-profile`, `--keep-forks`, `--sync`, `--include-branches`.
- v1 rejects tools/skills permissions.
- Background mode requires macro Jobs worker for async processing.

- [ ] **Step 7: Final commit**

```bash
git add tldw_Server_API/app/core/Chat_Macros/README.md Docs/Development
git commit -m "docs: document chat macros v1"
```

Only include docs paths that actually changed.

## Implementation Notes

- Keep `/wrapup` as the proving slice. Do not build the full macro editor before the backend, command path, and minimal status UI work.
- Keep ACP fork execution behind `branch_strategy` and metadata until chat-native execution is passing.
- Never call external providers in unit tests. Use fake branch/merge runners.
- Do not manually edit Backlog task files unless MCP/CLI tooling is unavailable.
- Do not stage unrelated dirty worktree files.
- Commit after each task so partial progress remains reviewable.

## Final Verification Checklist

- [ ] `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat_Macros -v`
- [ ] `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py -v`
- [ ] `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat_NEW/integration/test_chat_completions_api.py -v`
- [ ] `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_chat_macros_jobs_worker_startup.py -v`
- [ ] `cd apps/packages/ui && bunx vitest run src/services/__tests__/chat-macros.test.ts src/components/Option/Settings/__tests__/ChatMacrosSettings.test.tsx src/components/Option/ChatWorkspace/__tests__/MacroStatusCard.test.tsx src/components/Option/ChatWorkspace/__tests__/MacroRunDetailDrawer.test.tsx src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.test.tsx`
- [ ] `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Chat_Macros tldw_Server_API/app/api/v1/endpoints/chat_macros.py tldw_Server_API/app/api/v1/schemas/chat_macros.py tldw_Server_API/app/api/v1/API_Deps/Chat_Macros_Deps.py tldw_Server_API/app/services/chat_macros_jobs_worker.py -f json -o /tmp/bandit_chat_macros.json`
