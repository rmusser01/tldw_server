# MCP Prompt Catalog Support Design

Date: 2026-06-22
Status: Implemented in TASK-2344
Backlog: TASK-2342
Related follow-up: TASK-2341
MCP reference: https://modelcontextprotocol.io/specification/2025-06-18/server/prompts

## Summary

Expose tldw prompt libraries through MCP protocol-level prompts so MCP clients can discover and invoke prompts with `prompts/list` and `prompts/get`.

The v1 scope is deliberately narrow but useful:

- all readable, non-deleted records from the authenticated user's regular Prompts library
- explicitly allowlisted config-file prompts from `tldw_Server_API/Config_Files/Prompts`
- no Prompt Studio prompts
- no live `notifications/prompts/list_changed` support

The implementation should use a Prompt Catalog Adapter Layer behind the existing `PromptsModule`. It should not build the broader shared prompt registry service in this slice; that larger Approach C follow-up is tracked by TASK-2341.

## Goals

- Make regular user prompt-library records available as MCP prompts.
- Make selected config-file prompts available as MCP prompts only when explicitly allowlisted by server config.
- Keep prompt protocol names stable across prompt renames by using prompt UUIDs.
- Preserve tldw prompt semantics while returning MCP-compliant prompt messages.
- Keep existing MCP prompt tools, especially tool-level `prompts.search` and `prompts.get`, compatible.
- Add enough protocol support to handle dynamic per-user prompts safely.
- Add cursor pagination from the first implementation.

## Non-Goals

- No Prompt Studio prompt exposure in v1.
- No frontend changes in v1.
- No per-config-entry RBAC in v1.
- No live prompt list-change notifications in v1.
- No broad shared prompt registry service in v1.
- No arbitrary execution of template logic. Rendering remains strict placeholder substitution through existing prompt-schema and legacy extraction behavior.
- No new HTTP convenience route for fetching one prompt in v1.

## Current State

MCP Unified already has protocol handlers for:

- `prompts/list`
- `prompts/get`
- `GET /api/v1/mcp/prompts` as a list-only HTTP convenience route

The existing `PromptsModule` exposes prompt access as MCP tools:

- tool `prompts.search`
- tool `prompts.get`

It does not yet expose actual MCP prompt definitions through the module prompt hooks.

The current module registry caches prompt names globally at module initialization through context-free `get_prompts()`. That is not safe for user-library prompts because they are dynamic and per-user. The protocol currently advertises prompts as `{"available": bool(modules)}`, while the MCP prompt spec expects a `prompts` capability with `listChanged`.

The regular Prompts DB already supports:

- legacy prompts with `system_prompt` and `user_prompt`
- structured prompts with `prompt_definition_json`
- UUIDs
- soft deletes
- search/list/get operations

The structured prompt foundation already assembles structured prompts into role-based messages and legacy snapshots. This should be reused for MCP rendering.

## Approved Approach

Use Approach B: a Prompt Catalog Adapter Layer behind `PromptsModule`.

`PromptsModule` remains the MCP boundary and delegates to focused helpers:

- `UserPromptCatalogSource`: lists and renders prompt records from the authenticated user's regular Prompts DB.
- `ConfigPromptCatalogSource`: lists and renders config prompts that are explicitly published in the MCP module settings.
- `MCPPromptFormatter`: converts tldw prompt records, structured prompt definitions, legacy prompt fields, and config prompt entries into MCP prompt definitions and messages.

This keeps the protocol integration focused while avoiding a large mixed-responsibility `PromptsModule`.

## Protocol Changes

### Capability Declaration

During `initialize`, the server should advertise:

```json
{
  "prompts": {
    "listChanged": false
  }
}
```

`listChanged: false` means the server supports fresh `prompts/list` calls but will not emit `notifications/prompts/list_changed` in v1.

### Context-Aware Prompt Hooks

Add backward-compatible context-aware prompt hooks to `BaseModule`:

```python
async def get_prompts_for_context(
    self,
    context: RequestContext,
    params: dict[str, Any],
) -> dict[str, Any]:
    return {"prompts": await self.get_prompts()}

async def get_prompt_for_context(
    self,
    name: str,
    arguments: dict[str, Any],
    context: RequestContext,
) -> dict[str, Any]:
    return await self.get_prompt(name, arguments)
```

Existing static modules can keep overriding `get_prompts()` and `get_prompt()`. `PromptsModule` should override the context-aware hooks.

The protocol handler should use the context-aware hooks when available. Dynamic user-library prompt names must not be inserted into the registry's global `_prompt_registry`, because doing so risks stale names and cross-user leakage.

### Prompt Routing

Protocol prompt names are routed by stable namespace prefix:

- `library:<uuid>`: prompt from the authenticated user's regular Prompts DB
- `config:<module>.<key>`: single allowlisted config prompt
- `config:<module>.<group>`: grouped allowlisted config prompt

`PromptsModule` owns these namespaces. Other future static prompt modules can still use the existing static registry path.

For `prompts/get`, the protocol handler must dispatch `library:` and `config:` names directly to `PromptsModule` before consulting the global prompt registry. The global prompt registry is only appropriate for static, context-free prompt providers.

### Listing Prompts

`prompts/list` supports cursor pagination.

Ordering is deterministic:

1. user-library prompts sorted by `name COLLATE NOCASE ASC`, then UUID ascending
2. config prompts in allowlist order

The server uses a module setting `prompt_list_page_size`, default `50`, clamped to `1..100`. MCP clients provide only `cursor`; they do not control page size in v1.

The cursor is opaque to clients. Use base64url JSON with this internal shape:

```json
{
  "v": 1,
  "library_after_name": "last prompt name",
  "library_after_uuid": "last-library-prompt-uuid",
  "library_done": false,
  "config_index": 0
}
```

Malformed, unsupported, or tampered cursors return JSON-RPC invalid params. `library_after_name` and `library_after_uuid` must be both present or both absent; partial keyset cursors are invalid. `library_done: true` means the cursor has moved into the config-prompt segment and the next list call must skip user-library rows. Cursors with `library_done: true` must not also contain library keyset fields. A nonzero `config_index` is valid only when `library_done: true`.

Library pagination uses keyset semantics over `(name COLLATE NOCASE, uuid)` rather than raw offsets, so prompt inserts/deletes/renames between pages are best-effort but do not depend on unstable row offsets. Config pagination uses the allowlist index because config ordering is fixed by server config.

When the user-library page exactly fills `prompt_list_page_size` but the library source has no additional rows, the server must still emit a cursor into the config segment when allowlisted config entries remain. Otherwise config prompts can be hidden forever behind an exactly full library page.

Each prompt definition includes:

- `name`: stable MCP name, for example `library:2f...` or `config:rag.retrieval_guidance`
- `title`: user-facing name
- `description`: short human-readable summary without the full prompt body
- `arguments`: customization arguments
- `_meta`: tldw metadata for tldw-aware clients

`_meta.tldw` may include:

- source type: `library` or `config`
- prompt UUID
- prompt version
- tags
- config module/key/group
- original role summary

### Getting A Prompt

`prompts/get` resolves by stable prompt name and renders with supplied arguments.

Library prompt rendering:

- Structured prompts use `assemble_prompt_definition`.
- Legacy prompts use existing legacy placeholder extraction and rendering behavior, either by temporary conversion into a structured definition or by an equivalent shared renderer.
- Missing required variables return JSON-RPC invalid params with sanitized variable metadata.

Config prompt rendering:

- Single entries load one template through prompt-loader-compatible resolution.
- Grouped entries load multiple templates as role-labeled parts.
- Environment file overrides supported by `load_prompt()` are respected.
- Override file paths are never returned or logged.

Rendered output uses MCP text content only in v1.

### Role Mapping

MCP prompt messages are spec-compliant:

- tldw `system` and `developer` content folds into labeled `user` text
- tldw `user` content remains `user` text
- tldw `assistant` blocks remain `assistant` messages
- legacy `system_prompt + user_prompt` becomes one labeled `user` message
- grouped config entries with system/user parts follow the same folding rule

Example:

```json
{
  "description": "Summary prompt from user library",
  "messages": [
    {
      "role": "user",
      "content": {
        "type": "text",
        "text": "System instructions:\nYou are concise.\n\nUser prompt:\nSummarize: ..."
      }
    }
  ],
  "_meta": {
    "tldw": {
      "source": "library",
      "prompt_uuid": "..."
    }
  }
}
```

## Prompt Arguments

Structured library prompts use declared `variables`.

Legacy library prompts and config prompts use the existing legacy variable extraction style, including placeholders such as:

- `{{topic}}`
- `{context}`
- `$query`
- `<input>`

Arguments in `prompts/list` are marked required when the source declares them required. Legacy/config extracted placeholders are treated as required in v1 because no default-value contract exists for them.

MCP prompt argument values must be strings in v1. Non-string argument values return invalid params instead of being implicitly coerced. Unknown extra arguments are ignored unless the structured prompt assembler already rejects them. Missing required arguments return invalid params.

## Permissions And Scope

Any authenticated MCP caller with `prompts.read` may call `prompts/list` and `prompts/get`.

Protocol-level prompt access must not also require `modules.read`. The `library:` and `config:` namespaces exposed by `PromptsModule` are gated by `Resource.PROMPT` read permission, API-key read allowance, and prompt/resource scopes. Existing module-level permission checks can remain for module management APIs and for unrelated static prompt providers, but they must not block `PromptsModule` prompt catalog access when `prompts.read` is granted.

The protocol layer should use a dedicated namespaced prompt permission path for `library:` and `config:` names. That path must require `Resource.PROMPT` read and must not fall back to module permission. The legacy `_has_prompt_permission()` module fallback can remain only for static context-free prompt providers.

User-library prompts are restricted to the authenticated user's Prompts DB through `RequestContext.db_paths["prompts"]`. The caller must not be able to provide or override DB paths through MCP prompt arguments.

Existing persona/path-scope checks still apply to user-library prompts. When context scopes contain prompt IDs, list and get behavior filters by those IDs after UUID lookup. Config prompts are visible to any authenticated MCP caller with `prompts.read` once allowlisted by server config.

Prompt Studio prompts are not a source in v1.

## Configuration

Add the `prompts` module to `tldw_Server_API/Config_Files/mcp_modules.yaml`:

```yaml
- id: prompts
  class: tldw_Server_API.app.core.MCP_unified.modules.implementations.prompts_module:PromptsModule
  enabled: true
  name: Prompts
  version: "1.0.0"
  department: knowledge
  max_concurrent: 10
  settings:
    prompt_list_page_size: 50
    max_rendered_prompt_chars: 100000
    config_prompts:
      enabled: true
      entries: []
      # Example entries; replace entries: [] with entries like these to publish
      # config prompts through MCP.
      # entries:
      #   - id: rag.retrieval_guidance
      #     module: rag
      #     key: retrieval_guidance
      #     title: Retrieval Guidance
      #   - id: chat.summary
      #     title: Conversation Summary
      #     messages:
      #       - role: system
      #         module: chat
      #         key: summary_system
      #       - role: user
      #         module: chat
      #         key: summary_user
```

Config entries are explicit. The server must not publish every prompt-loader file or every key in a file by default.

The shipped default configuration should keep `config_prompts.entries` empty. Config prompt publication is opt-in by replacing the empty list with explicit entries.

Single config entries use:

- `id`
- `module`
- `key`
- optional `title`
- optional `description`

Grouped config entries use:

- `id`
- `title`
- optional `description`
- `messages`, each with `role`, `module`, and `key`

Allowed config roles are `system`, `developer`, `user`, and `assistant`; they are mapped to MCP-safe roles during rendering.

## Error Handling

- Missing or invalid `name`: invalid params.
- Unknown prompt: invalid params.
- Malformed cursor: invalid params.
- Missing required argument: invalid params.
- Permission denied: MCP permission error.
- Config allowlist entry points to a missing file/key: omit from list; direct get returns invalid params.
- User Prompts DB unavailable: direct library get returns internal error with sanitized message.
- User Prompts DB unavailable during list: return allowlisted config prompts when available, and include a sanitized `_meta.tldw.warnings` entry such as `{"source": "library", "code": "prompt_db_unavailable"}`.
- Rendered output exceeds `max_rendered_prompt_chars`: invalid params if caused by supplied arguments; otherwise internal error with sanitized message.

Error messages must not include prompt bodies, rendered prompt text, argument values, or override file paths.

## Security Guardrails

- Validate prompt name prefixes and UUID format before DB access.
- Never log prompt bodies or rendered arguments.
- Never expose prompt override file paths.
- Strip/ignore user-provided fields that could override ownership or DB paths.
- Cap list page size and rendered output size.
- Treat config prompt publication as an admin/server configuration decision.
- Do not fail open if prompt DB access, scope filtering, or config prompt resolution errors occur.

## Testing

Unit tests for `MCPPromptFormatter`:

- stable names
- variable extraction
- role folding
- assistant message preservation
- metadata
- missing-variable errors
- rendered output size limits

Unit tests for `UserPromptCatalogSource`:

- non-deleted filtering
- UUID-based names
- mixed pagination state
- structured assembly
- legacy rendering
- prompt ID scope filtering
- DB unavailable behavior

Unit tests for `ConfigPromptCatalogSource`:

- explicit allowlist only
- single entries
- grouped entries
- missing key omission
- environment override behavior
- argument extraction across grouped templates
- allowlist ordering
- mixed pagination with user-library results

Protocol tests:

- `initialize` returns `{"prompts": {"listChanged": false}}`
- `prompts/list`
- `prompts/get`
- direct namespace dispatch for `library:` and `config:` before global prompt registry lookup
- malformed cursor handling
- cursor resume across library and config sources
- keyset cursor behavior when prompts are inserted or deleted between pages
- permission denied
- `prompts.read` grants prompt access without requiring `modules.read`
- unknown prompt
- missing argument mapping
- non-string argument value mapping
- output-size error mapping
- partial list warning metadata when the library source fails but config prompts are returned

Integration tests:

- use `/api/v1/mcp/request` for both protocol-level `prompts/list` and `prompts/get`
- keep `GET /api/v1/mcp/prompts` list-only in v1 and add a `cursor` query parameter that maps to protocol-level `prompts/list`
- cover context DB-path isolation at unit/protocol level
- check the existing multi-user fixtures during implementation planning; if they support isolated user Prompt DB setup, add an integration test proving user A cannot list or get user B's library prompt
- if the existing fixtures do not support that setup, record the limitation in the test module and in the Backlog task verification notes

Security tests:

- targeted tests for catalog-source and formatter error paths to ensure prompt bodies, rendered arguments, and override file paths are not logged
- avoid brittle global assertions over all logs

## Rollout And Compatibility

- Add the `prompts` module to `mcp_modules.yaml`.
- Ship `config_prompts.entries` empty by default so config prompt exposure remains opt-in.
- Keep tool-level `prompts.search` and `prompts.get` unchanged.
- Document the distinction between protocol-level `prompts/get` and tool-level `prompts.get`.
- Add MCP prompt support behind normal module enablement config, not a new global flag.
- No prompt list-change notifications in v1.
- No frontend work in v1.
- Add focused docs at `Docs/MCP/mcp_prompts.md`.
- Link the new docs from `Docs/MCP/mcp_tool_catalogs.md`.

Existing `tools/list` and `tools/call` clients should see no behavior change.

Existing `GET /api/v1/mcp/prompts` starts returning real prompt definitions once the module is enabled and permissions allow it.

Protocol-level `prompts/get` uses `/api/v1/mcp/request` in v1.

Static context-free prompt modules, if added later, can keep using existing `get_prompts()` and `get_prompt()` methods.

User prompt protocol names use UUIDs, so prompt renames do not break MCP clients.

## Acceptance Criteria

- `initialize` advertises `prompts.listChanged: false`.
- `prompts/list` returns paginated, permission-filtered prompt definitions from the user prompt library and allowlisted config prompt entries.
- `prompts/get` renders `library:<uuid>` prompts and `config:<module>.<key-or-group>` prompts with strict argument validation.
- `library:` and `config:` prompt names route directly to `PromptsModule` before global prompt registry lookup.
- `prompts.read` is sufficient for protocol-level prompt catalog access and does not require `modules.read`.
- User prompt-library entries include all readable, non-deleted prompts and exclude deleted prompts.
- Prompt Studio prompts are not listed or retrievable.
- Config prompts are excluded unless explicitly allowlisted in the `prompts` module settings, and the shipped default allowlist is empty.
- Prompt names are stable and namespace-prefixed.
- MCP messages use only spec-compliant `user` and `assistant` roles.
- MCP prompt arguments reject non-string values with invalid params.
- `GET /api/v1/mcp/prompts` remains list-only but supports cursor pagination.
- Partial list results include sanitized warning metadata when a source fails.
- Existing tool-level `prompts.search` and `prompts.get` behavior remains compatible.
- Prompt bodies, rendered arguments, and override file paths are not exposed in logs or errors.
