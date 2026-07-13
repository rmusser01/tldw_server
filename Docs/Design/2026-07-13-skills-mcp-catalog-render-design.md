# Skills MCP Catalog and Safe Render Design

**Backlog task:** TASK-2294.1

**Parent task:** TASK-2294

**Status:** Approved for implementation planning

**Date:** 2026-07-13

## Summary

Add a read-only-from-the-caller's-perspective Skills module to MCP Unified with three tools:

- `skills.list` discovers model-visible skills using bounded search and pagination.
- `skills.get` retrieves the same metadata shape for one skill.
- `skills.render` substitutes bounded arguments into one authorized skill and returns the rendered prompt without calling a model or tool.

This is the first stability-focused slice of the broader skill and workflow runner work. It reuses the existing Skills registry, integrity resolver, dry-run executor, MCP authorization gateway, approval leases, hooks, reporting, timeout, and circuit-breaker paths. It does not introduce another execution engine or policy evaluator.

## Problem

Skills are available through the REST API and chat integration, but MCP clients do not have a supported way to discover Skills metadata or safely render a Skill for later use. A direct implementation has several failure risks:

- exposing raw `SKILL.md`, supporting files, or filesystem paths during discovery;
- advertising hidden or model-disabled skills to model-facing clients;
- authorizing the MCP tool while failing to authorize the selected `Skill(name)` subject;
- presenting declared tool patterns as effective tool permissions;
- accidentally invoking fork-mode models or tools during a render request;
- bypassing MCP hooks, reporting, caller identity, timeouts, or approval leases;
- duplicating workflows or Jobs behavior in a new runner.

## Goals

1. Provide bounded, user-scoped Skills discovery through MCP Unified.
2. Return metadata without reading or returning full Skill content for list/get operations.
3. Render one visible, integrity-approved, policy-approved Skill with forced dry-run behavior.
4. Preserve existing MCP `deny`, `ask`, approval-lease, and `allow` semantics.
5. Make declarations and effective authorization clearly distinct.
6. Register and test the module through the same production paths as existing MCP modules.

## Non-Goals

- Calling an LLM, including for a Skill whose context is `fork`.
- Executing any declared tool.
- Running scripts, subagents, hooks owned by Skills, workflows, or Jobs.
- Creating or mutating authored Skills, workflows, jobs, profiles, or approval grants. Existing Skills registry synchronization may update derived index rows to match files on disk.
- Returning raw frontmatter, raw `SKILL.md`, supporting-file content, directory paths, or content hashes.
- Adding frontend or browser-extension UI.
- Adding a second permission parser, confirmation flow, telemetry system, database table, or execution abstraction.
- Solving output consolidation, workflow orchestration, evaluations, or durable execution from the parent TASK-2294.

## Existing Components Reused

- `SkillsService` for registry synchronization, metadata, integrity filtering, and full Skill loading.
- `SkillExecutor.execute(..., dry_run=True)` for argument substitution and side-effect-free rendering.
- `build_skill_runtime_metadata` for runtime declaration fields.
- MCP `BaseModule` and `create_tool_definition` for module behavior and schemas.
- MCP tool authorization and gateway profile permission rules.
- Existing policy grant store and TTL approval leases for `ask` outcomes.
- MCP coordinator hooks, tool-use reporting, identity propagation, rate limiting, timeouts, and module circuit breakers.
- `CharactersRAGDB` and the authenticated request context's `db_paths["chacha"]` for the user-scoped Skill registry.

## Architecture

`SkillsModule` is a thin MCP adapter. It validates inputs, opens a request-scoped user database, constructs `SkillsService`, delegates discovery or rendering, formats a bounded response, and closes the database connection. Existing `SkillsService` initialization and reads may create the user's Skills directory, ensure the registry table, or synchronize derived registry rows. These are internal index-maintenance effects, not caller-authored Skill mutations.

The module must fail closed when any of these are absent or invalid:

- authenticated `context.user_id` convertible to a positive integer;
- `context.db_paths["chacha"]`;
- a user-scoped ChaChaNotes database that can initialize the Skill registry.

The Skills base directory is the parent directory of the authenticated context's ChaChaNotes database path. The module must not accept `user_id`, `db_path`, `db_paths`, or a Skill base path from tool arguments.

All three tools run through normal MCP tool execution. The module must not call another module directly or bypass coordinator policy, hooks, reporting, timeout, or circuit-breaker behavior.

## Visibility Model

A Skill is model-visible only when all conditions are true:

- it belongs to the authenticated user's registry;
- it is not deleted;
- `user_invocable` is true;
- `disable_model_invocation` is false;
- the existing integrity resolver allows it for Skill discovery or execution.

`skills.list`, `skills.get`, and `skills.render` use this same model-visible definition. An exact lookup of an unknown, deleted, hidden, or model-disabled Skill returns the same `skill_not_found` error. The response must not reveal which condition failed.

Catalog visibility is not an execution grant. `skills.list` may show metadata for a model-visible Skill that the active profile cannot render. `skills.render` performs the subject-specific permission check. This avoids per-row policy evaluation, policy-derived pagination errors, and a misleading `can_render` field.

## Tool Contracts

### `skills.list`

Purpose: search and page through model-visible Skill metadata.

Input:

```json
{
  "q": "optional case-insensitive name/description query, 0..200 characters",
  "limit": "integer, 1..100, default 50",
  "offset": "integer, >= 0, default 0"
}
```

Whitespace-only `q` is normalized to no query. Unknown input properties are rejected by module validation even if the generic MCP schema layer permits them.

Output:

```json
{
  "skills": [
    {
      "name": "summarize-paper",
      "description": "Summarize a paper for a chosen audience.",
      "argument_hint": "[audience]",
      "user_invocable": true,
      "disable_model_invocation": false,
      "declared_tools": ["rag.search"],
      "model": null,
      "context": "inline",
      "runtime": {
        "execution_mode": "inline",
        "test_run_may_call_model": false,
        "declares_tools": true,
        "declared_tool_count": 1,
        "model_override": null,
        "auto_invocation_enabled": true
      },
      "version": 3
    }
  ],
  "count": 1,
  "total": 1,
  "limit": 50,
  "offset": 0,
  "next_offset": null
}
```

`total` is the integrity-filtered count for the same query. `next_offset` is `offset + count` only when that value is less than `total`; otherwise it is null. Sorting is fixed to name ascending in this first slice so pagination is deterministic.

### `skills.get`

Purpose: retrieve metadata for one exact model-visible Skill without returning its instructions.

Input:

```json
{
  "name": "summarize-paper"
}
```

`name` uses the existing Skill name normalization and validation rules. The output is one metadata object with the same fields as a `skills.list` item. It does not include content, raw content, supporting files, directory path, content hash, IDs, or timestamps.

`skills.get` is a discovery operation and does not emit a `Skill(name)` permission subject. Its output is no more sensitive than the same row returned by `skills.list`.

### `skills.render`

Purpose: substitute arguments into one authorized Skill without executing it.

Input:

```json
{
  "skill_name": "summarize-paper",
  "arguments": "expert"
}
```

- `skill_name` is required and uses existing Skill name validation.
- `arguments` is optional, defaults to an empty string, and is limited to 10,000 characters.
- The caller cannot provide `dry_run`, model, provider, tools, user identity, paths, or execution settings.

Output:

```json
{
  "skill_name": "summarize-paper",
  "rendered_prompt": "Summarize this paper for an expert audience...",
  "declared_tools": ["rag.search"],
  "model_override": null,
  "execution_mode": "inline",
  "dry_run": true,
  "version": 3
}
```

`declared_tools` contains Skill declarations only. It does not assert tool availability or permission. Every later tool call remains subject to MCP catalog, profile, RBAC, hook, and argument-sensitive permission checks.

## Authorization

### Tool-level authorization

The existing gateway must first allow the caller to execute `skills.render`. Normal MCP module and tool RBAC checks remain authoritative.

### Skill subject authorization

The shared MCP permission-subject extractor currently parses `Skill(pattern)` rules but does not emit Skill subjects from tool arguments. Add a bounded `skill_name` argument convention:

- a non-empty string under the exact key `skill_name` emits one normalized `skill` subject;
- Skill subjects and `Skill(...)` rule patterns are normalized to lowercase, matching canonical Skill name normalization and approval-grant lookup;
- the existing maximum subject count and value-length limits apply;
- generic keys such as `name` do not emit Skill subjects;
- no module evaluates profile documents directly.

Only `skills.render` uses the `skill_name` field in this slice. Therefore its gateway evaluation combines the tool subject and `Skill(skill_name)` subject before the module can load full content.

Outcomes use existing behavior:

- `deny`: return the existing gateway policy-denied response;
- `ask` with no active lease: return the existing `approval_required` response;
- `ask` with a valid scoped lease: continue and attach the existing redacted approval marker;
- `allow` or no matching Skill rule: continue according to existing gateway defaults.

The module does not create grants, prompt for approval, or weaken a decision.

## Render Sequence

1. MCP coordinator validates module and tool access.
2. Gateway extracts `tool=skills.render` and `skill=<skill_name>` subjects.
3. Gateway evaluates profile rules and any existing approval lease.
4. Module validates the exact argument shape and hard limits.
5. Module resolves authenticated user context and opens the user-scoped registry database.
6. Module loads metadata only and applies model-visible and integrity checks.
7. Module loads the full Skill through the existing verified service path.
8. Module calls `SkillExecutor.execute` with `context=None` and `dry_run=True` forced by code.
9. Module rejects rendered output above the configured character limit.
10. Module returns the sanitized result and closes the request-scoped database.

Using `context=None` is intentional: dry render reports declarations and cannot accidentally interpret the current MCP catalog as effective authorization. The executor must never enter inline execution or fork execution in this tool.

## Bounds and Configuration

The module configuration is:

```yaml
settings:
  list_page_size: 50
  max_rendered_skill_chars: 100000
```

- Tool callers may request `limit` from 1 through 100.
- `list_page_size` supplies the default and is clamped to 1 through 100.
- `q` is at most 200 characters.
- `arguments` is at most 10,000 characters.
- `max_rendered_skill_chars` defaults to 100,000 and is clamped to 1 through 100,000. Configuration cannot raise the hard ceiling.
- An oversized rendered prompt fails atomically. It is never truncated or partially returned.
- Full Skill loading retains all existing Skills parser and storage bounds.

## Error Contract

Module-owned failures use the existing MCP JSON-RPC error classes and bounded reason tokens. Gateway-owned authorization failures retain the existing gateway response format. This slice does not add a new exception transport or error-data schema.

| Condition | JSON-RPC mapping and message | Disclosure rule |
| --- | --- | --- |
| Missing or invalid authenticated user context | Authorization error: `skills_user_context_required` | No fallback user or path |
| Unknown, deleted, hidden, or model-disabled Skill | Invalid params: `skill_not_found` | Same response for every case |
| Invalid name, query, pagination, or arguments | Invalid params with field and bound | No submitted content |
| Integrity resolver blocks the Skill | Authorization error: `context_integrity_blocked` | No path, digest, or content |
| Render exceeds output limit | Invalid params: `rendered_skill_too_large`; include limit | No partial rendered prompt |
| Registry/storage unavailable | Existing sanitized internal error | Detailed exception type only in bounded logs |
| Profile rule denies | Existing gateway denial | Existing sanitized provenance rules |
| Profile rule asks without a lease | Existing `approval_required` | Existing grant availability metadata |

Logs may include a bounded Skill name, user ID, operation, and exception type. Logs and tool responses must not include rendered content, raw Skill content, supporting-file content, API keys, approval tokens, database paths, or filesystem paths.

## Metadata Semantics

The MCP response renames the registry field `allowed_tools` to `declared_tools`. Existing REST schemas remain unchanged.

`runtime` remains declaration metadata:

- `execution_mode` is the Skill's declared `inline` or `fork` mode;
- `test_run_may_call_model` describes a separate non-dry REST test run, not this MCP render;
- `declares_tools` and `declared_tool_count` describe declarations only;
- `model_override` is declarative and is not used by render;
- `auto_invocation_enabled` is true for every returned row because model-disabled Skills are excluded.

## Registration and Surface

Add an enabled `skills` module to `tldw_Server_API/Config_Files/mcp_modules.yaml` immediately after `prompts`:

```yaml
- id: skills
  class: tldw_Server_API.app.core.MCP_unified.modules.implementations.skills_module:SkillsModule
  enabled: true
  name: Skills
  version: "0.1.0"
  department: knowledge
  max_concurrent: 10
  settings:
    list_page_size: 50
    max_rendered_skill_chars: 100000
```

Classify `skills` as read-only in `module_surface.py`. Each tool definition uses `readOnlyHint: true` because no tool intentionally changes caller-authored state. Existing Skills registry synchronization remains permitted as derived index maintenance.

Do not add Skills to gateway presets or persona archetypes in this slice unless an existing test proves enabled modules require an explicit preset entry. Module/tool authorization remains the rollout control.

## Testing Strategy

### Skills service

- After registry synchronization, metadata-only lookup does not call the verified full-directory parser, load supporting files, or return path/content fields.
- Exact lookup excludes hidden, model-disabled, deleted, and integrity-blocked Skills.
- List count and pagination remain correct after integrity filtering.

### Permission subject extraction

- `skill_name` emits one normalized Skill subject.
- Skill subjects, rule patterns, and approval-grant values use the same lowercase canonical form.
- `name`, blank values, non-string values, and nested values do not emit Skill subjects.
- Subject length and count limits still fail closed.
- `deny`, `ask`, active approval lease, and `allow` behavior is exercised through gateway runtime tests.

### MCP module

- Tool definitions, annotations, defaults, bounds, and unknown-property rejection.
- Missing user context and missing ChaChaNotes path fail closed.
- List/get response shapes expose metadata only.
- Hidden and model-disabled Skills cannot be listed, fetched, or rendered.
- Render substitutes full and indexed arguments using existing executor behavior.
- Fork-mode render remains dry and does not call a model or tool.
- Response uses `declared_tools` and never claims effective authorization.
- Oversized output is rejected without a partial prompt.
- Database connections close on success and failure.
- Errors and logs do not expose content or paths.

### Registration and integration

- Default module configuration declares and enables the Skills module with bounded settings.
- Dynamic module loading imports and registers `SkillsModule`.
- Module surface reports Skills as read-only.
- Package-boundary tests allow the intended dependency direction without importing FastAPI endpoint dependencies.
- An authenticated MCP integration test lists, gets, and renders a user-owned Skill and cannot access another user's Skill.

## Rollout and Compatibility

- No database migration is required.
- No REST API response is changed.
- Existing Skills execution behavior is unchanged.
- Existing profile documents remain valid. `Skill(...)` rules begin applying to tools that use the explicit `skill_name` argument convention; currently this slice introduces the first such tool.
- The module is enabled by default because it does not intentionally mutate caller-authored state, is user-scoped and integrity-checked, and remains subject to MCP module/tool/profile authorization.
- If production validation finds unacceptable discovery overhead, disable the module through existing MCP module configuration rather than adding a feature flag.

## Deferred Parent-Task Work

Later TASK-2294 child tasks may add execution, but only after separate designs define ownership and failure semantics for:

- model invocation and fork-mode execution;
- tool catalog resolution and effective allowed-tool binding;
- subagent orchestration;
- multi-step scripts and output consolidation;
- durable Jobs or Workflows handoff;
- execution cancellation, retries, quotas, and progress;
- execution telemetry and evaluations;
- frontend controls and approval UX.

Those tasks must build on this catalog/render contract rather than expanding `skills.render` into an execution endpoint.

## Decision Record

1. Use three tools rather than one overloaded tool for clear schemas and policy surfaces.
2. Keep get metadata-only; raw Skill instructions are available only after render authorization.
3. Exclude both hidden and model-disabled Skills from every model-facing operation.
4. Keep catalog discovery independent from per-Skill render permission to preserve deterministic pagination.
5. Extend shared permission-subject extraction instead of parsing profile rules in the module.
6. Reuse existing approval leases for `ask`; add no second confirmation system.
7. Force dry-run rendering and omit executor context to prevent execution and avoid permission claims.
8. Use `declared_tools` in MCP responses to distinguish declarations from effective authorization.
9. Reject oversized output rather than truncating executable instructions.
10. Defer all execution and workflow behavior to separately reviewable TASK-2294 child tasks.
11. Treat existing registry synchronization as derived index maintenance rather than claiming every read is physically mutation-free.
12. Canonicalize Skill permission subjects, patterns, and grants to lowercase before matching.
