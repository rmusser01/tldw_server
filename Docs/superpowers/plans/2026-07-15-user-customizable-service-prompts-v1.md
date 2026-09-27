# User-Customizable Service Prompts v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an authenticated user inspect, preview, save, and reset four curated workflow prompts from `/settings/prompt`, with the same owner-scoped values used by the backend, WebUI, and browser extension.

**Architecture:** Add one immutable backend registry, one v6 table in each existing per-user Prompts database, four authenticated API operations, and one shared TypeScript client/runtime. Server-side Translation resolves directly; Chat wrappers load one immutable snapshot before their existing pipeline preflight and pass it through the pipeline, while Compare resolves before its shared message fan-out. The Settings page uses the same API but keeps its React Query cache separate from fresh per-invocation runtime reads.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic, SQLite, pytest, TypeScript, React 18, TanStack Query, Ant Design, Vitest, Playwright, Bun.

---

## Source of truth and execution prerequisites

- Approved specification: `Docs/superpowers/specs/2026-07-12-user-customizable-service-prompts-design.md`.
- Planning/reconciliation task: `TASK-13142` (legacy ID: `TASK-13013`).
- Approved source commit: `1a038599753e780f32f62243871026ca9b6d2c06`.
- Reconciled current-`dev` planning commit: `e6665ddf89`.
- Before runtime edits, search Backlog.md and create one implementation task referencing `TASK-13142` and this plan. Export its allocated ID as `TASK_ID`, record the exact generated task-record path as `TASK_FILE`, and use `TASK_ID` in every implementation commit. Stage `TASK_FILE` in Task 1 so the task exists in Git, and stage every later metadata change before the commit that claims that progress.
- Work in an isolated worktree based on current `origin/dev`. If the project-local worktree has no `.venv`, create an ignored symlink to the repository environment with `ln -s ../../.venv .venv`, then run `source .venv/bin/activate`.
- Install the existing frontend workspace dependencies once with `cd apps && bun install`. Do not add a dependency for parsing, rendering, state, or navigation.
- CI shard changes are explicitly out of scope for this slice. Keep Python tests under already-covered `Prompt_Management`, `Translation`, and `Services` directories.

## Scope guard

Implement exactly the four definitions below. Do not add an approval service, history/restore, signed manifests, keyring integration, deployment policy states, a preview endpoint, a second database, portable import/export, prompt-specific Jobs/Scheduler machinery, or a broad inventory migration. Do not expose `webSearchFollowUpPrompt`: it has no runtime consumer.

The only intentional adjacent hardening is required by the approved design:

- render authored templates once and concatenate runtime values literally;
- append `systemPromptAppendix` only after template rendering;
- force every RAG question-rewrite call to `toolChoice: "none"`, `tools: []`, and `saveToDb: false`.

## Locked v1 contract

### Definitions

| Definition ID | English label | Part contract | Required variables | Stable affected-workflow IDs |
| --- | --- | --- | --- | --- |
| `chat.rag.answer` | RAG answer | `template` / template | `context`, `question` | `chat.main.rag`, `chat.tab.rag`, `chat.document.rag`, `chat.sidepanel.rag` |
| `chat.rag.question_rewrite` | RAG follow-up rewrite | `template` / template | `chat_history`, `question` | `chat.main.rag`, `chat.document.rag`, `chat.sidepanel.rag` |
| `chat.web_search.answer` | Web-search answer | `template` / template | `current_date_time`, `search_results` | `chat.main.web_search`, `chat.compare.web_search` |
| `media.text.translation` | Text translation | `system` / literal and `user_template` / template | `target_language`, `text` on `user_template` | `media.text.translation` |

Use these English descriptions:

- `chat.rag.answer`: “Controls how retrieved context and the current question are presented to the model.”
- `chat.rag.question_rewrite`: “Controls how a conversational follow-up is rewritten into a standalone retrieval query.”
- `chat.web_search.answer`: “Controls how normalized web-search results are presented for the final answer.”
- `media.text.translation`: “Controls the visible instructions used by synchronous text translation.”

Part labels are `Template`, `System instructions`, and `User template`. Workflow labels are `Main chat RAG`, `Tab chat RAG`, `Document chat RAG`, `Sidepanel RAG`, `Main chat web search`, `Compare web search`, and `Text translation`.

### Canonical defaults

- Copy the three Chat defaults byte-for-byte from `apps/packages/ui/src/services/tldw-server.ts`.
- Copy the Translation system and user-template defaults byte-for-byte from `tldw_Server_API/app/api/v1/endpoints/translate.py`.
- Move canonical ownership to the backend registry.
- Retain clearly named TypeScript `LEGACY_*` copies only for catalog-404 compatibility with an older server.
- Put all five exact default strings and the cross-language render cases in one test fixture: `apps/packages/ui/src/utils/__fixtures__/service-prompt-rendering.json`. Python and TypeScript tests must both consume it.

### API

| Method | Path | Scope | Result |
| --- | --- | --- | --- |
| GET | `/api/v1/service-prompts` | read | Metadata summaries only; never prompt bodies |
| GET | `/api/v1/service-prompts/{definition_id}` | read | Default, saved, and effective parts |
| PUT | `/api/v1/service-prompts/{definition_id}` | write | Validate, compare-and-swap, activate immediately |
| DELETE | `/api/v1/service-prompts/{definition_id}?expected_revision=...` | write | Conditional reset and packaged-default detail |

The detail contract is:

```json
{
  "id": "chat.rag.answer",
  "label": "RAG answer",
  "description": "Controls how retrieved context and the current question are presented to the model.",
  "parts": [
    {
      "key": "template",
      "label": "Template",
      "mode": "template",
      "required_variables": ["context", "question"]
    }
  ],
  "affected_workflows": [
    {"id": "chat.main.rag", "label": "Main chat RAG"}
  ],
  "default_parts": {"template": "..."},
  "saved_parts": null,
  "effective_parts": {"template": "..."},
  "source": "packaged",
  "revision": null
}
```

PUT accepts:

```json
{
  "parts": {"template": "..."},
  "expected_revision": null
}
```

Use `ConfigDict(extra="forbid")`. `expected_revision` and the DELETE query parameter are `UUID | None`. Domain errors use FastAPI's outer `{"detail": ...}` envelope:

- 404 `service_prompt_unknown_definition`;
- 409 `service_prompt_revision_conflict` with `current_revision: UUID | null`;
- 422 `service_prompt_validation_failed` with `field_errors: dict[str, str]`;
- 500 `service_prompt_corrupt_override` with `revision` and `can_reset: true`;
- 500 `service_prompt_store_failed` with no prompt content.

Malformed Pydantic bodies/queries, authentication failures, scope failures, and Prompts-database dependency failures keep their existing framework envelopes. A corrupt row cannot be repaired with PUT in v1; the user must conditionally DELETE it using the separately stored revision.

### Rendering and validation

- Parts must have exactly the registered keys.
- Every part is a non-blank string of at most 20,000 Unicode code points.
- Literal parts do not parse braces.
- Template fields are simple ASCII identifiers matching `^[A-Za-z_][A-Za-z0-9_]*$`.
- Attribute/index access, numeric fields, conversions, format specifications, unknown fields, missing fields, and repeated fields are errors.
- Every declared variable appears exactly once.
- `{{` and `}}` produce literal braces.
- Parse only the authored template. Inserted values containing braces, `$&`, `$'`, backslashes, newlines, Unicode, or placeholder-looking text remain byte-literal.
- Never log prompt bodies or include them in catalog/error payloads.

## Minimal file map

### Create

- `tldw_Server_API/app/core/Prompt_Management/service_prompts.py`
- `tldw_Server_API/app/api/v1/schemas/service_prompt_schemas.py`
- `tldw_Server_API/app/api/v1/endpoints/service_prompts.py`
- `tldw_Server_API/tests/Prompt_Management/test_service_prompts.py`
- `tldw_Server_API/tests/Prompt_Management/test_service_prompts_api.py`
- `tldw_Server_API/tests/Translation/test_translate_service_prompt.py`
- `apps/packages/ui/src/utils/__fixtures__/service-prompt-rendering.json`
- `apps/packages/ui/src/services/tldw/domains/service-prompts.ts`
- `apps/packages/ui/src/services/tldw/domains/__tests__/service-prompts.test.ts`
- `apps/packages/ui/src/services/service-prompts.ts`
- `apps/packages/ui/src/services/__tests__/service-prompts.test.ts`
- `apps/packages/ui/src/components/Option/Settings/ServicePromptsSettings.tsx`
- `apps/packages/ui/src/components/Option/Settings/__tests__/ServicePromptsSettings.test.tsx`
- `apps/packages/ui/src/hooks/chat-modes/__tests__/service-prompts.test.ts`
- `apps/packages/ui/src/hooks/chat/__tests__/useChatActions.service-prompts.test.tsx`
- `apps/packages/ui/src/hooks/__tests__/useMessage.service-prompts.test.tsx`
- `apps/tldw-frontend/__tests__/pages/settings-prompt-route.test.tsx`
- `apps/extension/tests/e2e/service-prompts.spec.ts`

### Modify

- `tldw_Server_API/app/core/DB_Management/Prompts_DB.py`
- `tldw_Server_API/tests/Prompt_Management/test_prompts_db_v2.py`
- `tldw_Server_API/app/api/v1/endpoints/translate.py`
- `tldw_Server_API/tests/Translation/test_translate_endpoint_error_mapping.py`
- `tldw_Server_API/app/api/v1/router_groups/content.py`
- `tldw_Server_API/app/api/v1/router_groups/minimal.py`
- `tldw_Server_API/tests/Services/test_router_groups_contract.py`
- `apps/packages/ui/src/services/tldw/openapi-guard.ts`
- `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- `apps/packages/ui/src/services/tldw/domains/index.ts`
- `apps/packages/ui/src/services/__tests__/background-proxy.test.ts`
- `apps/packages/ui/src/services/tldw-server.ts`
- `apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts`
- `apps/packages/ui/src/hooks/chat-modes/ragMode.ts`
- `apps/packages/ui/src/hooks/chat-modes/tabChatMode.ts`
- `apps/packages/ui/src/hooks/chat-modes/documentChatMode.ts`
- `apps/packages/ui/src/hooks/chat-modes/normalChatMode.ts`
- `apps/packages/ui/src/hooks/chat/useChatActions.ts`
- `apps/packages/ui/src/hooks/useMessage.tsx`
- `apps/packages/ui/src/routes/option-settings-route-registry.tsx`
- `apps/packages/ui/src/routes/route-registry.tsx`
- `apps/tldw-frontend/extension/routes/route-registry.tsx`
- `apps/tldw-frontend/pages/settings/prompt.tsx`
- `apps/packages/ui/src/components/Layouts/settings-nav-config.ts`
- `apps/packages/ui/src/data/settings-index.ts`
- `apps/packages/ui/src/hooks/useOmniSearchDeps.tsx`
- `apps/packages/ui/src/components/Common/PromptSearch.tsx`
- `apps/packages/ui/src/routes/__tests__/option-settings-route-split.test.tsx`
- `apps/packages/ui/src/components/Layouts/__tests__/settings-layout-labels.test.tsx`
- `apps/packages/ui/src/components/Layouts/__tests__/settings-nav.guardian.test.ts`
- `apps/packages/ui/src/data/__tests__/settings-index.test.ts`
- `apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx`
- `apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.backup-all.test.tsx`
- `apps/packages/ui/src/assets/locale/en/settings.json`
- generated `apps/packages/ui/src/public/_locales/en/settings.json`
- `apps/tldw-frontend/e2e/smoke/page-inventory.ts`
- `apps/extension/tests/e2e/page-inventory.ts`
- `apps/tldw-frontend/lib/api/openapi.fingerprint.json`

### Delete

- `apps/packages/ui/src/components/Option/Settings/prompt.tsx` after all three registries and the Next page use `ServicePromptsSettings`. This file is already unreachable and must not be revived.

---

### Task 1: Add the static registry, validator, resolver, and single-pass renderer

**Files:**

- Create: `apps/packages/ui/src/utils/__fixtures__/service-prompt-rendering.json`
- Create: `tldw_Server_API/tests/Prompt_Management/test_service_prompts.py`
- Create: `tldw_Server_API/app/core/Prompt_Management/service_prompts.py`

- [x] **Step 1: Write the shared fixture**

Add:

- `defaults` containing the exact parts for all four definitions;
- `render_cases` covering ordinary substitution, escaped braces, Unicode, newlines, backslashes, `$&`, `$'`, and inserted values containing `{question}` or another registered name;
- exact expected rendered strings.

The fixture is test data only. Runtime registry initialization must not read a frontend file.

- [x] **Step 2: Write failing Python tests**

Cover:

- exactly the four locked IDs, part keys, variables, metadata, and workflows;
- fixture defaults equal registry defaults byte-for-byte;
- every packaged default validates at import;
- unknown definition;
- missing/extra/non-string/blank/over-20,000-code-point parts;
- traversal, indexing, numeric fields, conversions, non-empty format specs, explicitly empty format specs such as `{question:}`, malformed braces, unknown, missing, and repeated variables;
- escaped braces and every shared render case;
- packaged fallback, valid-user precedence, and corrupt-row failure with revision;
- error strings never contain prompt text.

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Prompt_Management/test_service_prompts.py -v
```

Expected red result: import fails because `Prompt_Management.service_prompts` does not exist.

- [x] **Step 3: Implement the minimal registry module**

Use frozen dataclasses and `MappingProxyType`; do not add a registry class or plugin mechanism.

```python
@dataclass(frozen=True)
class ServicePromptPart:
    key: str
    label: str
    mode: Literal["literal", "template"]
    required_variables: tuple[str, ...]


@dataclass(frozen=True)
class ServicePromptDefinition:
    id: str
    label: str
    description: str
    parts: tuple[ServicePromptPart, ...]
    default_parts: Mapping[str, str]
    affected_workflows: tuple[ServicePromptWorkflow, ...]


@dataclass(frozen=True)
class ResolvedServicePrompt:
    definition: ServicePromptDefinition
    parts: Mapping[str, str]
    source: Literal["user", "packaged"]
    revision: str | None


class UnknownServicePromptDefinition(ValueError): ...


class ServicePromptValidationError(ValueError):
    field_errors: Mapping[str, str]


class ServicePromptCorruptOverride(RuntimeError):
    revision: str
```

Exception messages and attributes carry definition/part identifiers, safe field errors, and revision only—never authored text. The endpoint maps these three domain exceptions plus `ServicePromptRevisionConflict`/`DatabaseError`; do not add a generic catch-all around authentication or dependency resolution.

Export only the small surface the callers need:

```python
def get_service_prompt_definition(definition_id: str) -> ServicePromptDefinition: ...
def list_service_prompt_definitions() -> tuple[ServicePromptDefinition, ...]: ...
def validate_service_prompt_parts(
    definition: ServicePromptDefinition,
    parts: Mapping[str, object],
) -> dict[str, str]: ...
def render_service_prompt_part(
    definition: ServicePromptDefinition,
    part_key: str,
    authored_text: str,
    values: Mapping[str, str],
) -> str: ...
def resolve_service_prompt(
    db: PromptsDatabase,
    definition_id: str,
) -> ResolvedServicePrompt: ...
```

Look up the part metadata by `part_key`, return `authored_text` unchanged for a literal part, and reject an unknown part key. Before `string.Formatter.parse`, make one small brace-aware lexical pass over authored text: skip `{{`/`}}`, reject unmatched braces, and reject any `:` or `!` inside a real field token. This is required because `Formatter.parse` reports both `{question}` and the forbidden `{question:}` with an empty `format_spec`. Then parse and render by concatenating literals and values once:

```python
rendered: list[str] = []
for literal, field, format_spec, conversion in Formatter().parse(template):
    rendered.append(literal)
    if field is not None:
        rendered.append(values[field])
return "".join(rendered)
```

Use `Counter(fields) == Counter(part.required_variables)` after enforcing simple identifiers, no conversion, and an empty format spec. Python `len(str)` supplies the required Unicode-code-point count.

Resolver order is exactly saved valid override, then packaged default. Parse `parts_json`, validate it, and raise a corruption exception carrying only `revision` if parsing or semantic validation fails. Never fall back silently.

- [x] **Step 4: Run the focused tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Prompt_Management/test_service_prompts.py -v
```

Expected green result: all registry, validation, fixture, renderer, and resolver tests pass.

- [x] **Step 5: Commit**

```bash
git add "$TASK_FILE" apps/packages/ui/src/utils/__fixtures__/service-prompt-rendering.json tldw_Server_API/app/core/Prompt_Management/service_prompts.py tldw_Server_API/tests/Prompt_Management/test_service_prompts.py
git commit -m "feat: add service prompt registry and renderer ($TASK_ID)"
```

---

### Task 2: Add the v6 per-user override table and atomic store methods

**Files:**

- Modify: `tldw_Server_API/tests/Prompt_Management/test_prompts_db_v2.py`
- Modify: `tldw_Server_API/app/core/DB_Management/Prompts_DB.py`

- [x] **Step 1: Write failing schema tests**

Add tests that:

- a fresh database reaches schema v6 and has `ServicePromptOverrides`;
- an actual v5 database migrates to v6 without losing an existing ordinary Prompt.

Build the v5 fixture by temporarily monkeypatching `PromptsDatabase._CURRENT_SCHEMA_VERSION` to `5`, creating/inserting/closing the database, restoring it to `6`, and reopening it. Do not modify historical v1 SQL merely to make fresh databases contain the v6 table.

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Prompt_Management/test_prompts_db_v2.py -k "schema or service_prompt" -v
```

Expected red result: current schema is 5 and the table is absent.

- [x] **Step 2: Write failing store tests**

Cover:

- raw read of an absent/present row;
- first insert with `expected_revision=None`;
- same parts are a no-op;
- retrying the same parts with a stale expected revision returns the current row unchanged;
- CAS update generates a new UUID revision;
- stale update returns conflict with current revision;
- insert race maps to identical success or conflict after refetch;
- absent reset with omitted expected revision is idempotent;
- matching reset deletes;
- stale reset and “row was already reset” with a supplied UUID conflict;
- corrupt `parts_json` retains a readable revision and can be reset;
- failed writes roll back.

Expected red result: store methods do not exist.

- [x] **Step 3: Implement v6 and the store**

Set `_CURRENT_SCHEMA_VERSION = 6`, add:

```sql
CREATE TABLE IF NOT EXISTS ServicePromptOverrides (
    definition_id TEXT PRIMARY KEY,
    parts_json TEXT NOT NULL,
    revision TEXT NOT NULL
);
```

Add `_SCHEMA_UPDATE_VERSION_SQL_V6`, `_apply_schema_v6`, and the `current_db_version == 5` dispatch. Use individual `conn.execute` calls inside the existing `transaction()`; do not use `executescript` inside the migration.

Keep the storage interface narrow:

```python
@dataclass(frozen=True)
class ServicePromptOverrideRow:
    definition_id: str
    parts_json: str
    revision: str


class ServicePromptRevisionConflict(ConflictError):
    def __init__(self, current_revision: str | None):
        super().__init__("Service Prompt override changed concurrently.")
        self.current_revision = current_revision


def get_service_prompt_override(
    self,
    definition_id: str,
) -> ServicePromptOverrideRow | None: ...

def save_service_prompt_override(
    self,
    definition_id: str,
    parts: Mapping[str, str],
    expected_revision: str | None,
) -> ServicePromptOverrideRow: ...

def reset_service_prompt_override(
    self,
    definition_id: str,
    expected_revision: str | None,
) -> ServicePromptOverrideRow | None: ...
```

Inside one `BEGIN IMMEDIATE` save transaction:

1. select the current row;
2. parse current JSON only to compare it with the already-validated requested mapping;
3. return the current row before checking the revision when parts are equal;
4. otherwise require the expected revision;
5. update with `WHERE definition_id = ? AND revision = ?`;
6. on a new row, require `expected_revision is None`, insert, and narrowly catch `sqlite3.IntegrityError` to refetch and classify the race;
7. generate `str(uuid.uuid4())` only for content changes.

DELETE selects only `definition_id, revision`; it must not parse `parts_json`.
Raise `ServicePromptRevisionConflict` for every CAS mismatch so the endpoint can return `current_revision` without rereading outside the transaction. Wrap other SQLite failures in the existing `DatabaseError` family using content-free messages, and never interpolate `parts`, `parts_json`, or authored text into logs or exceptions.

- [x] **Step 4: Run the DB tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Prompt_Management/test_prompts_db_v2.py -k "schema or service_prompt" -v
```

Expected green result: v5→v6, fresh v6, CAS, retries, races, reset, and rollback pass.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/Prompts_DB.py tldw_Server_API/tests/Prompt_Management/test_prompts_db_v2.py
git commit -m "feat: persist service prompt overrides ($TASK_ID)"
```

---

### Task 3: Expose the API and make Translation the server-side canary

**Files:**

- Create: `tldw_Server_API/app/api/v1/schemas/service_prompt_schemas.py`
- Create: `tldw_Server_API/app/api/v1/endpoints/service_prompts.py`
- Create: `tldw_Server_API/tests/Prompt_Management/test_service_prompts_api.py`
- Create: `tldw_Server_API/tests/Translation/test_translate_service_prompt.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/translate.py`
- Modify: `tldw_Server_API/tests/Translation/test_translate_endpoint_error_mapping.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/content.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/minimal.py`
- Modify: `tldw_Server_API/tests/Services/test_router_groups_contract.py`
- Modify: `apps/tldw-frontend/lib/api/openapi.fingerprint.json`

- [x] **Step 1: Write failing API tests**

Use the existing Prompt Management app/client fixtures and `get_prompts_db_for_user` dependency override. Cover:

- exact four-item catalog metadata and absence of `default_parts`, `saved_parts`, and `effective_parts`;
- detail packaged state plus `Cache-Control: no-store`;
- PUT activates immediately and returns the new revision;
- identical retry preserves revision;
- DELETE packaged state, idempotence, stale conflict, and corrupt-row reset;
- exact 404/409/422/500 domain envelopes and safe `dict[str, str]` field errors;
- structural body/query 422 remains FastAPI-native;
- dependency/auth errors retain existing shapes;
- read-scoped API key can GET but receives 403 on PUT/DELETE;
- write-scoped API key and ordinary JWT principal can mutate;
- two user-specific DB overrides cannot cross-read or mutate;
- no request schema includes user ID or database path;
- no prompt body appears in catalog, error text, or captured logs.

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Prompt_Management/test_service_prompts_api.py -v
```

Expected red result: endpoint module/path is missing.

- [x] **Step 2: Implement schemas and endpoint mapping**

Use normal current-user ownership:

```python
class ServicePromptUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parts: dict[str, object]
    expected_revision: UUID | None


db: PromptsDatabase = Depends(get_prompts_db_for_user)
```

Keep `parts` as `dict[str, object]` at the wire boundary so wrong part keys/types reach the shared semantic validator and use the locked `service_prompt_validation_failed` envelope. Missing request fields, a malformed UUID, a non-object body, or an extra top-level field remain FastAPI-native structural 422 responses. Declare response schemas for catalog/detail so prompt bodies cannot accidentally enter the summary model.

Decorate GET operations with `Depends(require_api_key_scope("read"))` and PUT/DELETE with `Depends(require_api_key_scope("write"))`. JWT users retain the dependency's existing bypass behavior.

Set `Cache-Control: no-store` on catalog and detail/mutation responses. Catch only post-dependency Service Prompt domain/store exceptions. Convert each to the locked envelope without prompt content.

On PUT:

1. resolve the definition;
2. validate the full parts object;
3. resolve the existing row first so a semantic corruption returns 500 instead of being overwritten;
4. call the atomic store method;
5. resolve and return current detail.

On DELETE, call the raw conditional reset path so malformed JSON cannot block reset, then return packaged detail.

- [x] **Step 3: Write failing router contract tests**

Add a fake `service_prompts` module to `test_router_groups_contract.py` and assert:

```python
ImportedRouterSpec(
    import_path="tldw_Server_API.app.api.v1.endpoints.service_prompts",
    log_name="service_prompts",
    prefix="/api/v1",
    tags=("service-prompts",),
    route_key="",
)
```

Also assert `"service_prompts"` is in `MINIMAL_REQUIRED_ROUTER_NAMES`.

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k "content_router_specs_populates_expected_specs or minimal" -v
```

Expected red result: the canonical and minimal selections omit `service_prompts`.

- [x] **Step 4: Register the always-on router**

Add the imported router spec to `iter_content_router_specs()` and add `service_prompts` to `MINIMAL_REQUIRED_ROUTER_NAMES`. A blank `route_key` is intentional: catalog 404 must mean an older server, not a disabled feature.

- [x] **Step 5: Write failing Translation tests**

Capture `analyze()` kwargs and prove:

- no override produces byte-identical `input_data` and `system_message`;
- a saved override affects the next call;
- Translation uses both parts atomically;
- `api_name`, `model_override`, `custom_prompt_arg=None`, `temp=0.3`, and `streaming=False` stay unchanged;
- provider/model/source-language response behavior stays unchanged;
- corrupt override never falls back;
- resolver/store failures occur outside the provider catch;
- existing provider errors remain sanitized.

Update direct calls in `test_translate_endpoint_error_mapping.py` to pass a DB/test double explicitly; FastAPI does not resolve a `Depends` default during a direct function call.

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Translation/test_translate_service_prompt.py tldw_Server_API/tests/Translation/test_translate_endpoint_error_mapping.py -v
```

Expected red result: `translate_text` still uses module constants and `str.format`.

- [x] **Step 6: Switch Translation to the resolver**

Resolve `media.text.translation` before the existing provider `try` block:

```python
definition = get_service_prompt_definition("media.text.translation")
resolved = resolve_service_prompt(db, definition.id)
prompt = render_service_prompt_part(
    definition,
    "user_template",
    resolved.parts["user_template"],
    {"target_language": request.target_language, "text": request.text},
)
```

Pass `resolved.parts["system"]` as `system_message`. Remove the duplicate canonical constants from `translate.py`; the registry now owns them.

- [x] **Step 7: Regenerate and verify the OpenAPI fingerprint**

```bash
cd apps/tldw-frontend
PYTHON=../../.venv/bin/python bun run generate:api-types
cd ../..
source .venv/bin/activate
python Helper_Scripts/export_openapi_schema.py --check apps/tldw-frontend/lib/api/openapi.fingerprint.json
```

Expected result: generated client contract and committed fingerprint match the new routes.

- [x] **Step 8: Run the focused backend slice**

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Prompt_Management/test_service_prompts.py \
  tldw_Server_API/tests/Prompt_Management/test_prompts_db_v2.py \
  tldw_Server_API/tests/Prompt_Management/test_service_prompts_api.py \
  tldw_Server_API/tests/Translation/test_translate_service_prompt.py \
  tldw_Server_API/tests/Translation/test_translate_endpoint_error_mapping.py \
  tldw_Server_API/tests/Services/test_router_groups_contract.py -v
```

Expected green result: registry, DB, API, router, and Translation tests pass.

- [x] **Step 9: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/service_prompt_schemas.py tldw_Server_API/app/api/v1/endpoints/service_prompts.py tldw_Server_API/tests/Prompt_Management/test_service_prompts_api.py tldw_Server_API/app/api/v1/endpoints/translate.py tldw_Server_API/tests/Translation/test_translate_service_prompt.py tldw_Server_API/tests/Translation/test_translate_endpoint_error_mapping.py tldw_Server_API/app/api/v1/router_groups/content.py tldw_Server_API/app/api/v1/router_groups/minimal.py tldw_Server_API/tests/Services/test_router_groups_contract.py apps/tldw-frontend/lib/api/openapi.fingerprint.json
git commit -m "feat: expose service prompts and translation ($TASK_ID)"
```

---

### Task 4: Add the shared TypeScript API, renderer, migration probe, and request snapshot loader

**Files:**

- Create: `apps/packages/ui/src/services/tldw/domains/service-prompts.ts`
- Create: `apps/packages/ui/src/services/tldw/domains/__tests__/service-prompts.test.ts`
- Create: `apps/packages/ui/src/services/service-prompts.ts`
- Create: `apps/packages/ui/src/services/__tests__/service-prompts.test.ts`
- Modify: `apps/packages/ui/src/services/tldw/openapi-guard.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/index.ts`
- Modify: `apps/packages/ui/src/services/__tests__/background-proxy.test.ts`
- Modify: `apps/packages/ui/src/services/tldw-server.ts`

- [x] **Step 1: Write failing typed-client and transport tests**

Keep wire IDs forward-compatible while locking the three Chat runtime IDs and Translation ID locally:

```typescript
export type KnownServicePromptId =
  | "chat.rag.answer"
  | "chat.rag.question_rewrite"
  | "chat.web_search.answer"
  | "media.text.translation"

export type ServicePromptSource = "user" | "packaged"

export class ServicePromptApiError extends Error {
  status: number
  code?: string
  fieldErrors?: Record<string, string>
  currentRevision?: string | null
  revision?: string
  canReset?: boolean
}
```

Catalog/detail DTOs use `id: string`; `loadServicePromptSnapshot` and the migration map accept `KnownServicePromptId`. This lets an older Settings client render a newer server definition using safe English metadata instead of pretending unknown IDs are impossible.

Test list/detail/PUT/DELETE paths and payloads, `expectedStatuses`, public `options.signal` forwarding to `bgRequest.abortSignal`, and normalization from both direct and extension-proxy errors. The normalizer must check `error.details?.detail ?? error.detail`. Add a transport regression proving `revision`, `current_revision`, and `can_reset` survive proxy redaction.

Add `/api/v1/service-prompts` and `/api/v1/service-prompts/{definition_id}` to `ClientPath`.

Run:

```bash
cd apps/packages/ui
bunx vitest run src/services/tldw/domains/__tests__/service-prompts.test.ts src/services/__tests__/background-proxy.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected red result: paths and methods do not exist.

- [x] **Step 2: Implement the domain mixin**

Add four methods to `servicePromptMethods`:

```typescript
listServicePrompts(options?: { signal?: AbortSignal })
getServicePrompt(id, options?)
saveServicePrompt(id, request, options?)
resetServicePrompt(id, expectedRevision, options?)
```

Use `bgRequest`, exact locked types, and the existing domain-mixin pattern. Add the mixin import, declaration merge, and `Object.assign` in `TldwApiClient.ts`; export it from `domains/index.ts`.
Encode `definition_id` as one path segment with `encodeURIComponent`; never concatenate an unescaped catalog ID into a URL.

- [x] **Step 3: Write failing renderer/default tests**

Consume the shared JSON fixture and cover:

- exact legacy Chat defaults;
- the same valid/invalid rules as Python, including explicit `{question:}` and conversion rejection;
- `[...value].length` for Unicode code points;
- escaped braces;
- literal parts;
- every single-pass fixture, especially `$&`, `$'`, backslashes, and inserted `{question}`;
- no sequential `.replace` or replacement-string behavior.

Expected red result: shared validator/renderer does not exist.

- [x] **Step 4: Implement one small shared service module**

Keep types/API transport in the domain file and put the remaining v1 behavior in one `services/service-prompts.ts`; do not create separate `types`, `template`, `runtime`, or `legacy-storage` modules.

Expose only these validation/rendering functions from that service module:

```typescript
validateServicePromptParts(definition, parts): Record<string, string>
renderServicePromptPart(definition, partKey, authoredText, values): string
```

The renderer looks up the registered part, returns `authoredText` unchanged for literal mode, and tokenizes only template-mode authored text. Callers pass the effective saved/default text explicitly; the renderer must never reach for a local default.

The tokenizer may use a small character loop:

```typescript
type TemplateToken =
  | { kind: "literal"; value: string }
  | { kind: "field"; name: string }
```

It parses only authored text, converts `{{`/`}}` to literal brace tokens, validates simple names, then renders with:

```typescript
return tokens
  .map((token) =>
    token.kind === "literal" ? token.value : variables[token.name]
  )
  .join("")
```

Export the legacy defaults from `tldw-server.ts` as clearly named `LEGACY_SERVICE_PROMPT_DEFAULTS`; existing getter functions remain for older-server fallback only.

- [x] **Step 5: Write failing raw-migration and runtime-loader tests**

Cover:

- raw local RAG value wins over raw sync;
- raw sync RAG value is used only when local is absent;
- web-search predecessor reads local only;
- no default-producing/auto-moving getter is called by migration detection;
- `webSearchFollowUpPrompt` is ignored;
- catalog 404 alone returns a legacy snapshot;
- catalog 401/403/5xx/network/protocol failures do not fall back;
- supported server plus unresolved mapped value throws an actionable migration-required error before detail reads;
- supported server loads requested details concurrently and freshly for each invocation;
- a detail failure is never converted to legacy;
- snapshot and nested parts are immutable;
- hosted/multi-user scope waits for `tldwAuth.getCurrentUser()` and never uses `user:anonymous`;
- config/account change aborts old reads;
- partial import cleanup removes only a value whose PUT succeeded, from both local and sync areas.

Use the existing `createSafeStorage({area: "local"|"sync"})`, `tldwClient.getConfig()`, `tldwAuth.getCurrentUser()`, and `buildChatSurfaceScopeKeyFromConfig()`. Do not call `readTldwSetting()` during supported-server detection because it mutates legacy sync evidence.

Use this fixed migration map—no key discovery or generic migration framework:

| Raw browser key | Definition/part | Areas inspected |
| --- | --- | --- |
| `systemPromptForRag` | `chat.rag.answer.template` | local first, then legacy sync |
| `questionPromptForRag` | `chat.rag.question_rewrite.template` | local first, then legacy sync |
| `webSearchPrompt` | `chat.web_search.answer.template` | local only |

`media.text.translation` has no browser-local predecessor. `webSearchFollowUpPrompt` is deliberately neither read nor cleared.

Lock the snapshot. Each entry carries only the deeply frozen render schema the
shared renderer needs. Supported entries clone it from the validated detail;
catalog-404 compatibility uses three fixed internal Chat schemas. Do not put
defaults, labels, workflows, a cache, or a second client registry in the
snapshot, and do not create a Translation compatibility schema:

```typescript
export type ServicePromptSnapshot = Readonly<{
  scopeKey: string
  capability: "supported" | "legacy-404"
  definitions: Readonly<
    Partial<Record<KnownServicePromptId, Readonly<{
      definition: Readonly<{
        id: string
        parts: readonly Readonly<{
          key: string
          mode: "literal" | "template"
          required_variables: readonly string[]
        }>[]
      }>
      parts: Readonly<Record<string, string>>
      source: ServicePromptSource
      revision: string | null
    }>>>
  >
}>
```

Run:

```bash
cd apps/packages/ui
bunx vitest run src/services/__tests__/service-prompts.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected red result: raw migration and snapshot functions are missing.

- [x] **Step 6: Implement capability, scope, migration, and snapshot loading**

Export:

```typescript
readLegacyServicePromptCandidates()
clearLegacyServicePromptCandidate(id)
resolveServicePromptScope()
loadServicePromptSnapshot(ids, { signal })
importLegacyServicePromptCandidate(candidate, detail, { signal })
```

Runtime order:

1. resolve exact server/account scope;
2. GET catalog;
3. on catalog 404 only, build legacy parts with existing compatibility getters;
4. on a supported catalog, raw-probe mapped values and stop affected workflows if unresolved;
5. GET only requested details concurrently with the invocation signal;
6. freeze and return one snapshot.

Do not use the Settings React Query cache for runtime reads.

- [x] **Step 7: Run the shared-client tests and OpenAPI guard**

```bash
cd apps/packages/ui
bunx vitest run src/services/tldw/domains/__tests__/service-prompts.test.ts src/services/__tests__/service-prompts.test.ts src/services/__tests__/background-proxy.test.ts --maxWorkers=1 --no-file-parallelism
bun run verify:openapi
```

Expected green result: API, proxy, renderer, scope, migration, and snapshot tests pass.

- [x] **Step 8: Commit**

```bash
git add apps/packages/ui/src/services/tldw/domains/service-prompts.ts apps/packages/ui/src/services/tldw/domains/__tests__/service-prompts.test.ts apps/packages/ui/src/services/service-prompts.ts apps/packages/ui/src/services/__tests__/service-prompts.test.ts apps/packages/ui/src/services/tldw/openapi-guard.ts apps/packages/ui/src/services/tldw/TldwApiClient.ts apps/packages/ui/src/services/tldw/domains/index.ts apps/packages/ui/src/services/__tests__/background-proxy.test.ts apps/packages/ui/src/services/tldw-server.ts
git commit -m "feat: add shared service prompt client ($TASK_ID)"
```

---

### Task 5: Replace the Workflow prompts Settings page and correct navigation/backup copy

**Files:**

- Create: `apps/packages/ui/src/components/Option/Settings/ServicePromptsSettings.tsx`
- Create: `apps/packages/ui/src/components/Option/Settings/__tests__/ServicePromptsSettings.test.tsx`
- Create: `apps/tldw-frontend/__tests__/pages/settings-prompt-route.test.tsx`
- Modify: `apps/packages/ui/src/routes/option-settings-route-registry.tsx`
- Modify: `apps/packages/ui/src/routes/route-registry.tsx`
- Modify: `apps/tldw-frontend/extension/routes/route-registry.tsx`
- Modify: `apps/tldw-frontend/pages/settings/prompt.tsx`
- Delete: `apps/packages/ui/src/components/Option/Settings/prompt.tsx`
- Modify: `apps/packages/ui/src/components/Layouts/settings-nav-config.ts`
- Modify: `apps/packages/ui/src/data/settings-index.ts`
- Modify: `apps/packages/ui/src/hooks/useOmniSearchDeps.tsx`
- Modify: `apps/packages/ui/src/components/Common/PromptSearch.tsx`
- Modify: `apps/packages/ui/src/routes/__tests__/option-settings-route-split.test.tsx`
- Modify: `apps/packages/ui/src/components/Layouts/__tests__/settings-layout-labels.test.tsx`
- Modify: `apps/packages/ui/src/components/Layouts/__tests__/settings-nav.guardian.test.ts`
- Modify: `apps/packages/ui/src/data/__tests__/settings-index.test.ts`
- Modify: `apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.backup-all.test.tsx`
- Modify: `apps/packages/ui/src/assets/locale/en/settings.json`
- Regenerate: `apps/packages/ui/src/public/_locales/en/settings.json`

- [x] **Step 1: Write failing component tests**

Test the shared component with mocked domain/runtime functions:

- loading and retryable disconnected/error states;
- plain four-definition list and `?prompt=<id>` detail selection;
- localized copy for known IDs and escaped server-English fallback for an unknown catalog ID;
- packaged/customized status;
- exact affected workflows and server/account scope;
- one editor per part, literal parts without variable chips, template parts with exact chips;
- local preview calls the production renderer with visible `[variable_name]` marker values, keeps literal parts unchanged, displays parts in registry order, and makes no API/LLM request;
- complete-parts PUT with current revision;
- atomic Translation edit;
- reset confirmation names the definition and permanent removal;
- validation field errors;
- conflict preserves the entire draft and offers reload;
- corrupt state exposes conditional reset with returned revision;
- catalog 404 older-server notice, while non-404 failures stay errors;
- raw migration import/discard, replacement confirmation, invalid-value repair, and partial cleanup;
- scope changes cancel queries, clear migration state, and disable an old draft;
- prompt bodies render only in text controls/escaped text;
- selected prompt remains usable on narrow layouts.

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Settings/__tests__/ServicePromptsSettings.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected red result: component does not exist.

- [x] **Step 2: Implement the smallest shared Settings component**

Use React Query only for Settings state:

```typescript
["service-prompts", scopeKey, "catalog"]
["service-prompts", scopeKey, "detail", definitionId]
```

Disable queries until the exact scope resolves. On config update, abort in-flight reads, cancel/invalidate the old scope keys, clear selection/migration state, and mark any old draft unsaveable.

Keep the layout simple:

- list on top/left depending on available width;
- detail editor;
- status, workflow text, variable chips;
- Preview, Save changes, Reset to default;
- secondary “Open reusable Prompts workspace” link to `/prompts`.

Preview needs no sample-input form: build the values directly from each registered variable name as `[${name}]`, render the current complete draft locally, and show the ordered per-part output as escaped plain text.

Do not add a diff viewer, history UI, bulk actions, or live LLM testing.

Implement unsaved protection locally in this component rather than changing the global router shim:

- confirm before changing `?prompt=`;
- capture only unmodified primary-button same-origin anchor clicks while dirty; ignore `target`, `download`, external, modified, and already-prevented clicks, and cancel only when the user declines;
- listen for `popstate`; on a declined browser back/forward, return to the edited history entry with a one-shot recursion guard;
- register `beforeunload` in both hosts;
- remove every listener on cleanup.

Tests must exercise query selection, Settings nav links, browser history, and `beforeunload`. Do not rely on the current no-op Next `useBlocker`.

- [x] **Step 3: Write failing route and navigation tests**

Assert:

- all three route registries and the Next page render `ServicePromptsSettings` at `/settings/prompt`;
- Settings nav label is “Workflow prompts” and remains in Preferences & Workflow;
- Settings index and Prompt Search send reusable Prompt Library navigation to `/prompts`;
- Omni Search keeps its existing reusable-prompts result at `/prompts` and describes `/settings/prompt` only as Workflow prompts, avoiding duplicate library results;
- the Workflow prompts page retains its secondary `/prompts` link;
- route inventories identify the page as Workflow prompts.

Run:

```bash
cd apps/packages/ui
bunx vitest run src/routes/__tests__/option-settings-route-split.test.tsx src/components/Layouts/__tests__/settings-layout-labels.test.tsx src/components/Layouts/__tests__/settings-nav.guardian.test.ts src/data/__tests__/settings-index.test.ts --maxWorkers=1 --no-file-parallelism
cd ../../tldw-frontend
bunx vitest run __tests__/pages/settings-prompt-route.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected red result: route still renders the reusable-workspace empty state and links still point to `/settings/prompt`.

- [x] **Step 4: Rewire routes and remove the dead editor**

Point all route registries and the Next wrapper to `ServicePromptsSettings`. Delete the unreachable old local editor. Keep the legacy getter/setter functions in `tldw-server.ts` because catalog-404 runtime compatibility still needs them.

Retarget only links that mean reusable Prompt Library to `/prompts`; keep Settings navigation at `/settings/prompt` and rename it Workflow prompts.
Use a new `settings:servicePrompts.title` route/nav token and keep the existing `managePrompts.*` copy owned by the reusable Prompt Library; do not globally relabel that older namespace. Keep `/settings/prompt` in the existing Preferences & Workflow settings group in both hosts.

- [x] **Step 5: Correct backup disclosure and English fallback keys**

Change the UI claim from “Backup all account data” to “Backup supported account data” and explicitly list Service Prompt overrides among portable-backup exclusions. The migration panel repeats the same exclusion.

Add new English Settings keys for:

- the `servicePrompts` page title/description without changing reusable Prompt Library keys;
- the four known definition labels/descriptions;
- seven workflow labels;
- part labels;
- states/actions/errors/migration/unsaved confirmations;
- truthful portable-backup wording.

Do not hand-copy English into every source locale. Use the existing i18next English/default fallback for untranslated new keys and server English metadata for unknown IDs. Run locale sync so the generated English extension catalog is current:

```bash
cd apps/extension
bun run locales:sync settings.json
bun run check:i18n:dupes
```

- [x] **Step 6: Run Settings and route tests**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Settings/__tests__/ServicePromptsSettings.test.tsx src/routes/__tests__/option-settings-route-split.test.tsx src/components/Layouts/__tests__/settings-layout-labels.test.tsx src/components/Layouts/__tests__/settings-nav.guardian.test.ts src/data/__tests__/settings-index.test.ts src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.backup-all.test.tsx --maxWorkers=1 --no-file-parallelism
cd ../../tldw-frontend
bunx vitest run __tests__/pages/settings-prompt-route.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected green result: shared editor, migration, guards, navigation, and backup disclosure pass.

- [x] **Step 7: Commit**

```bash
git add apps/packages/ui/src/components/Option/Settings apps/packages/ui/src/routes/option-settings-route-registry.tsx apps/packages/ui/src/routes/route-registry.tsx apps/tldw-frontend/extension/routes/route-registry.tsx apps/tldw-frontend/pages/settings/prompt.tsx apps/tldw-frontend/__tests__/pages/settings-prompt-route.test.tsx apps/packages/ui/src/components/Layouts/settings-nav-config.ts apps/packages/ui/src/data/settings-index.ts apps/packages/ui/src/hooks/useOmniSearchDeps.tsx apps/packages/ui/src/components/Common/PromptSearch.tsx apps/packages/ui/src/components/Layouts/__tests__ apps/packages/ui/src/routes/__tests__/option-settings-route-split.test.tsx apps/packages/ui/src/data/__tests__/settings-index.test.ts apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.backup-all.test.tsx apps/packages/ui/src/assets/locale/en/settings.json apps/packages/ui/src/public/_locales/en/settings.json
git commit -m "feat: replace workflow prompt settings ($TASK_ID)"
```

---

### Task 6: Resolve one immutable snapshot in every Chat consumer

**Files:**

- Create: `apps/packages/ui/src/hooks/chat-modes/__tests__/service-prompts.test.ts`
- Create: `apps/packages/ui/src/hooks/chat/__tests__/useChatActions.service-prompts.test.tsx`
- Create: `apps/packages/ui/src/hooks/__tests__/useMessage.service-prompts.test.tsx`
- Modify: `apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts`
- Modify: `apps/packages/ui/src/hooks/chat-modes/ragMode.ts`
- Modify: `apps/packages/ui/src/hooks/chat-modes/tabChatMode.ts`
- Modify: `apps/packages/ui/src/hooks/chat-modes/documentChatMode.ts`
- Modify: `apps/packages/ui/src/hooks/chat-modes/normalChatMode.ts`
- Modify: `apps/packages/ui/src/hooks/chat/useChatActions.ts`
- Modify: `apps/packages/ui/src/hooks/useMessage.tsx`

- [x] **Step 1: Write failing mode-wrapper and golden tests**

For each top-level wrapper, assert prompt loading occurs before `runChatPipeline` creates message stubs:

- Main RAG loads answer + rewrite once;
- Tab Chat loads answer once and never rewrites;
- Document Chat loads answer + rewrite once;
- normal Chat loads web-search answer only when web search is enabled;
- a caller-supplied snapshot prevents another load.

Add provider-message goldens:

- Main RAG, Tab, and Document final answer render `question=original current user message`;
- legacy Sidepanel final answer renders `question=rewritten retrieval query`;
- every rewrite receives that path's current serialized history;
- main/Compare web search preserve current ISO datetime and normalized result formatting;
- no-override ordinary output is byte-identical;
- placeholder-looking runtime values and JavaScript replacement metasequences prove the intentional single-pass correction;
- render first, then append literal `systemPromptAppendix`.

Expected red result: modes still read local settings inside preflight/preparePrompt and use `.replace`.

- [x] **Step 2: Thread the optional snapshot through existing params**

Add the same optional field to `ChatModeParamsBase` **and** to each independently declared mode parameter type: `RagModeParams`, `TabChatModeParams`, `DocumentChatModeParams`, and `NormalChatModeParams`:

```typescript
servicePromptSnapshot?: ServicePromptSnapshot
```

Do not assume the four mode types extend `ChatModeParamsBase`; they currently do not. Each mode wrapper loads the exact IDs it needs unless its own parameter object already carries this field, then passes it into `runChatPipeline`. Mode definitions only read the snapshot; they never fetch. The focused TypeScript test must compile calls through all four wrapper signatures so a missing carrier field cannot hide behind a cast.

This placement matters: `runChatPipeline` creates optimistic user/assistant messages before its current `preflight`, so resolution inside `preflight` is too late.

- [x] **Step 3: Replace mode-local getters and replacement loops**

- Remove `promptForRag()` and `getWebSearchPrompt()` calls from mode definitions.
- Use `renderServicePromptPart()` from the snapshot.
- In RAG/Tab/Document, render the answer template first, then call `appendSystemPromptSuffix(rendered, appendix)`.
- Move web-search prompt availability outside the best-effort provider/search catch. A provider search failure may still fall back; a supported-server prompt failure aborts preflight.
- Keep selected-source retrieval fallback behavior, but do not let its catch convert a prompt-resolution failure into a grounding fallback.

- [x] **Step 4: Harden all question-rewrite calls**

In Main RAG, Document Chat, and legacy `useMessage`, construct the rewrite model with:

```typescript
{
  toolChoice: "none",
  tools: [],
  saveToDb: false
}
```

The customized rewrite output is used only as the retrieval query. Tests also prove authenticated sources, media IDs, retrieval options, provider/model selection, and tool configuration are unchanged.

- [x] **Step 5: Write failing Compare tests**

In `useChatActions.service-prompts.test.tsx`, assert:

- when Compare uses web search, snapshot loading happens before `setMessages`, `setHistory`, title/history creation, or `saveMessage`;
- one snapshot is loaded before `models.map`;
- every branch receives the same object identity and revision;
- load failure produces no shared user-message/history side effect;
- Compare without web search performs no Service Prompt read.

`useCompareSubmit` handles a later per-model reply as its own top-level invocation, so its ordinary `normalChatMode` wrapper may load a new snapshot once. Verify that path through the wrapper test; no `useCompareSubmit.ts` code change is expected.

- [x] **Step 6: Implement Compare snapshot ownership**

In `useChatActions.ts`, resolve the web-search definition before creating/persisting the Compare user message and pass the same snapshot in `compareEnhancedParams` to every branch. Do not resolve inside `models.map`.

- [x] **Step 7: Write failing legacy Sidepanel tests**

In `useMessage.service-prompts.test.tsx`, assert:

- one immutable RAG snapshot resolves before the first `setMessages`;
- unresolved supported-server legacy migration blocks the send with the Workflow prompts link;
- catalog 404 preserves current local behavior;
- supported-server failures do not use local values;
- rewrite is tool-disabled/non-persistent;
- final answer uses rewritten query while Main/Tab/Document keep original-question semantics.

- [x] **Step 8: Implement the legacy snapshot path**

Load the applicable snapshot once into a local constant at the beginning of the top-level legacy send path, before UI mutation. Reuse it for rewrite and final answer. Do not retrofit the whole legacy hook onto `ChatModeContext`.

- [x] **Step 9: Run focused Chat tests**

```bash
cd apps/packages/ui
bunx vitest run src/hooks/chat-modes/__tests__/service-prompts.test.ts src/hooks/chat/__tests__/useChatActions.service-prompts.test.tsx src/hooks/__tests__/useMessage.service-prompts.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected green result: all consumers resolve once at the correct boundary and preserve the locked semantics.

- [x] **Step 10: Commit**

```bash
git add apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts apps/packages/ui/src/hooks/chat-modes/ragMode.ts apps/packages/ui/src/hooks/chat-modes/tabChatMode.ts apps/packages/ui/src/hooks/chat-modes/documentChatMode.ts apps/packages/ui/src/hooks/chat-modes/normalChatMode.ts apps/packages/ui/src/hooks/chat/useChatActions.ts apps/packages/ui/src/hooks/useMessage.tsx apps/packages/ui/src/hooks/chat-modes/__tests__/service-prompts.test.ts apps/packages/ui/src/hooks/chat/__tests__/useChatActions.service-prompts.test.tsx apps/packages/ui/src/hooks/__tests__/useMessage.service-prompts.test.tsx
git commit -m "feat: resolve service prompts in chat workflows ($TASK_ID)"
```

---

### Task 7: Prove cross-host behavior and run the completion gates

**Files:**

- Create: `apps/extension/tests/e2e/service-prompts.spec.ts`
- Create: `apps/tldw-frontend/e2e/workflows/service-prompts-runtime.spec.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/page-inventory.ts`
- Modify: `apps/extension/tests/e2e/page-inventory.ts`
- Modify implementation task metadata and final summary

- [x] **Step 1: Add the cross-host and corrupt-transport E2E cases**

Launch the production build with the existing `launchWithExtension` helper—not an `OrSkip` wrapper—so this completion gate fails on missing browser/build/server prerequisites. Seed the extension's `tldwConfig` through the helper, then create a second page in the same persistent context and seed the same config in WebUI `localStorage` with `page.addInitScript` before navigation. Keep two cases in this one file:

**Live cross-host case, against one real server/account:**

1. open WebUI `/settings/prompt?prompt=chat.rag.answer`;
2. save a unique valid template;
3. open `${optionsUrl}#/settings/prompt?prompt=chat.rag.answer`;
4. assert the same revision/template is shown;
5. switch the extension to a tiny in-test Node HTTP server with a distinct normalized server scope and assert the first value/draft is absent and cannot be saved while that scope is unresolved;
6. switch back, reset in the extension, reload the WebUI detail, and assert it returns to packaged state;
7. run Main RAG once with deterministic retrieval/model routes and assert the outgoing provider message uses the packaged `chat.rag.answer` text rather than the deleted marker, proving reset changed a consumer as well as both editors;
8. in `afterEach`, GET the current detail and DELETE with its returned revision when present, so cleanup remains conditional and deterministic.

**Corrupt transport case, against the in-test Node server:**

1. return the exact 500 `service_prompt_corrupt_override` detail with a known revision from GET detail;
2. assert the built extension presents the corrupt state and keeps that revision;
3. confirm reset and assert the proxied DELETE carries the same `expected_revision`;
4. return packaged detail and assert recovery.

This proves the real built-extension transport without adding a production test endpoint or reaching into a live server's SQLite files. Task 2/3 integration tests prove the actual corrupt-row deletion transaction.

Also assert Preview makes no API/LLM call and the page has no serious axe violation at desktop and narrow widths.

Update only the two page inventories needed to name `/settings/prompt` “Workflow prompts”. Do not add CI shard configuration.

- [x] **Step 2: Add the four-definition runtime propagation matrix**

In `apps/tldw-frontend/e2e/workflows/service-prompts-runtime.spec.ts`, use the real disposable Service Prompt API and real WebUI workflow components. Stub only nondeterministic retrieval, web-search, and model responses, and capture the actual provider payload emitted by each workflow. Do not satisfy this matrix by calling `renderServicePromptPart()` directly.

Exercise every named consumer from the approved registry:

| Definition | Runtime paths that must emit the unique saved marker |
| --- | --- |
| `chat.rag.answer` | Main RAG, Tab Chat, Document Chat, and legacy Sidepanel RAG final-answer calls |
| `chat.rag.question_rewrite` | Main RAG, Document Chat, and legacy Sidepanel rewrite calls |
| `chat.web_search.answer` | normal Chat with web search and every Compare web-search branch |
| `media.text.translation` | real `POST /api/v1/translate`, with both the customized system part and rendered user template present at the configured OpenAI-compatible upstream |

For the three browser-owned definitions, PUT a unique marker through the real owner-scoped API, drive the named UI path, and inspect the intercepted `/api/v1/chat/completions` request. Retrieval/search stubs return fixed data so assertions target only prompt propagation. For Translation, start a tiny OpenAI-compatible recording server inside the spec on the fixed `TLDW_E2E_CAPTURE_URL`; launch the disposable tldw server with `CUSTOM_OPENAI_API_URL` pointing at that URL and a fake `CUSTOM_OPENAI_API_KEY`, call `/translate` with `provider: "custom-openai-api"`, and assert the recording server received both customized parts. The recording server returns a valid deterministic completion and binds only to loopback.

After each definition's custom-marker assertions, reset it through a real client, reload the other client when applicable, rerun the same consumer matrix, and assert the exact packaged message plus absence of the marker. Use the WebUI for the remaining three definitions; the cross-host case separately proves extension reset for `chat.rag.answer`. This is the E2E proof that reset from both client types restores workflows, while Task 3 and Task 6 retain the faster per-layer goldens. Conditional `afterEach` cleanup still GETs the current revision before DELETE.

Expected red result: editors can persist overrides, but current workflows continue reading local/default prompt text and reset is not reflected in a new invocation.

- [x] **Step 3: Run the backend gate**

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Prompt_Management/test_service_prompts.py \
  tldw_Server_API/tests/Prompt_Management/test_prompts_db_v2.py \
  tldw_Server_API/tests/Prompt_Management/test_service_prompts_api.py \
  tldw_Server_API/tests/Translation/test_translate_service_prompt.py \
  tldw_Server_API/tests/Translation/test_translate_endpoint_error_mapping.py \
  tldw_Server_API/tests/Services/test_router_groups_contract.py -v
python -m compileall -q tldw_Server_API/app
python -m ruff check \
  tldw_Server_API/app/core/Prompt_Management/service_prompts.py \
  tldw_Server_API/app/core/DB_Management/Prompts_DB.py \
  tldw_Server_API/app/api/v1/schemas/service_prompt_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/service_prompts.py \
  tldw_Server_API/app/api/v1/endpoints/translate.py \
  tldw_Server_API/app/api/v1/router_groups/content.py \
  tldw_Server_API/app/api/v1/router_groups/minimal.py \
  tldw_Server_API/tests/Prompt_Management/test_service_prompts.py \
  tldw_Server_API/tests/Prompt_Management/test_prompts_db_v2.py \
  tldw_Server_API/tests/Prompt_Management/test_service_prompts_api.py \
  tldw_Server_API/tests/Translation/test_translate_service_prompt.py \
  tldw_Server_API/tests/Translation/test_translate_endpoint_error_mapping.py \
  tldw_Server_API/tests/Services/test_router_groups_contract.py
```

Expected result: all focused backend tests pass, compileall is silent, and Ruff reports no errors.

- [x] **Step 4: Run Bandit on every touched Python path**

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/Prompt_Management/service_prompts.py \
  tldw_Server_API/app/core/DB_Management/Prompts_DB.py \
  tldw_Server_API/app/api/v1/schemas/service_prompt_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/service_prompts.py \
  tldw_Server_API/app/api/v1/endpoints/translate.py \
  tldw_Server_API/app/api/v1/router_groups/content.py \
  tldw_Server_API/app/api/v1/router_groups/minimal.py \
  -f json -o /tmp/bandit_service_prompts_v1.json
```

Expected result: exit 0 and no new finding in changed code.

- [x] **Step 5: Run frontend unit/integration gates**

```bash
cd apps/packages/ui
bunx vitest run \
  src/services/tldw/domains/__tests__/service-prompts.test.ts \
  src/services/__tests__/service-prompts.test.ts \
  src/services/__tests__/background-proxy.test.ts \
  src/components/Option/Settings/__tests__/ServicePromptsSettings.test.tsx \
  src/hooks/chat-modes/__tests__/service-prompts.test.ts \
  src/hooks/chat/__tests__/useChatActions.service-prompts.test.tsx \
  src/hooks/__tests__/useMessage.service-prompts.test.tsx \
  src/routes/__tests__/option-settings-route-split.test.tsx \
  src/components/Layouts/__tests__/settings-layout-labels.test.tsx \
  src/components/Layouts/__tests__/settings-nav.guardian.test.ts \
  src/data/__tests__/settings-index.test.ts \
  src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.backup-all.test.tsx \
  --maxWorkers=1 --no-file-parallelism
bun run verify:openapi

cd ../../tldw-frontend
bunx vitest run __tests__/pages/settings-prompt-route.test.tsx --maxWorkers=1 --no-file-parallelism
bun run typecheck
bun run lint

cd ../extension
bun run locales:sync settings.json
bun run check:i18n:dupes
bun run compile
bun run build:chrome:prod
```

Expected result: focused tests, OpenAPI guard, typecheck, lint, locale check, compile, and Chrome production build pass.

- [x] **Step 6: Run the real runtime and cross-host E2E gates**

Run these only against a disposable local single-user test server using the fake keys below, with the WebUI already running at `TLDW_WEB_URL`. Start that server with `CUSTOM_OPENAI_API_URL=http://127.0.0.1:18112/v1` and `CUSTOM_OPENAI_API_KEY=sk-e2e-fake`; the runtime spec owns the loopback capture listener at that URL. Never point the mutating save/reset cases at a personal or shared account. Both specs must health-check their required URLs before mutation and fail rather than skip when a prerequisite is unavailable.

```bash
cd apps/tldw-frontend
TLDW_WEB_URL=http://127.0.0.1:8080 \
TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 \
TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY \
TLDW_E2E_CAPTURE_URL=http://127.0.0.1:18112/v1 \
bunx playwright test e2e/workflows/service-prompts-runtime.spec.ts --reporter=line --workers=1

cd ../extension
TLDW_WEB_URL=http://127.0.0.1:8080 \
TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 \
TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY \
bunx playwright test tests/e2e/service-prompts.spec.ts --reporter=line --workers=1
```

Expected result: all named consumers use the saved marker, reset restores their packaged messages, and cross-host save visibility, scope isolation, corrupt reset, preview, and responsive accessibility pass without skips.

- [x] **Step 7: Run repository hygiene**

```bash
git diff --check
git status --short
```

Review every changed file against this plan. Confirm there is no:

- extra definition;
- prompt body in catalog/log/error;
- user ID or DB path in the API;
- silent supported-server fallback;
- second local source of truth;
- history/approval/Jobs/deployment machinery;
- `webSearchFollowUpPrompt` migration;
- CI shard edit.

- [x] **Step 8: Commit E2E/inventory closeout**

```bash
git add apps/extension/tests/e2e/service-prompts.spec.ts apps/tldw-frontend/e2e/workflows/service-prompts-runtime.spec.ts apps/tldw-frontend/e2e/smoke/page-inventory.ts apps/extension/tests/e2e/page-inventory.ts
git commit -m "test: verify service prompts v1 ($TASK_ID)"
```

- [x] **Step 9: Finalize and commit the Backlog task**

- Record focused test, typecheck, lint, build, E2E, OpenAPI, Bandit, and `git diff --check` evidence.
- Mark every acceptance criterion complete only after the evidence exists.
- Record the deliberate CI-shard skip as user-directed and note that tests stayed in existing covered directories.
- Add the final summary and mark the implementation task ready for review or done according to the Backlog workflow.

Stage and commit that tracked state; do not leave completion metadata only in the working tree:

```bash
git add "$TASK_FILE"
git commit -m "chore: finalize service prompts v1 task ($TASK_ID)"
```

- [x] **Step 10: Prepare the PR and commit its task link**

- Use the branch-finishing workflow to push and create the PR only after every gate above passes.
- Add the resulting PR URL to the implementation task through the official Backlog workflow.
- If that changes `TASK_FILE`, stage, commit, and push the link before handoff; verify `git status --short` is clean.
- Before merge, the human requester must write the required human-owned `Change summary` explaining what changed and why this static-registry/one-table/request-snapshot design was chosen.
