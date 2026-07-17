# Skills Render Tool Resolution Design

**Backlog task:** TASK-2294.2

**Parent task:** TASK-2294

**Status:** Ready for review

**Date:** 2026-07-13

## Summary

Extend MCP `skills.render` with one additive field:

```json
"catalog_matches": ["rag.search"]
```

The field contains unique base tool names from the Skill's existing
`declared_tools` that exactly match the embedded MCP catalog with
`canExecute: true`. It is `[]` when matching completes with no result and
`null` when the whole catalog lookup is unavailable or exceeds its bounded
deadline.

This is advisory catalog matching, not an authorization verdict. It does not
execute a model or tool, predict whether a future call will succeed, or add a
second preflight workflow.

## Goal And Code Fit

TASK-2294 covers reusable agentic Skills and workflows. TASK-2294.1 delivered
MCP `skills.list`, `skills.get`, and side-effect-free `skills.render`, then
deferred tool catalog resolution and effective allowed-tool binding. This
child delivers only the catalog-resolution part on the existing render path.

The implementation reuses existing code:

- `SkillsModule._render_skill()` owns the authorized, verified Skill load and
  rendered-output limit.
- `SkillExecutor.substitute_arguments()` owns the argument-substitution
  behavior needed by render.
- `MCPProtocol._handle_tools_list()` builds the embedded request catalog and
  attaches `canExecute`.
- `MCPServer` owns the active protocol and injects its bound list handler into
  module configuration. `SkillsModule` therefore has no server singleton or
  protocol-construction dependency.

`SkillExecutor.execute()` is not needed for dry render. Calling substitution
directly and copying the existing metadata keeps render structurally outside
the inline and fork execution branches. The request-scoped Skills database is
closed before catalog discovery begins.

## Scope

In scope:

- one nullable `catalog_matches` response field;
- exact matching against one embedded `tools/list` result;
- preserving every existing render field and failure boundary;
- focused module, integration, documentation, and security verification.

Out of scope:

- REST `/skills`, WebUI, browser-extension, or `SkillPreview` changes;
- a new public preflight, runner, workflow, job, or subagent tool;
- model or tool execution;
- effective-profile, command, path, approval, credential, quota, or backend
  readiness decisions;
- direct/deferred routes, schemas, denial reasons, or installation guidance;
- gateway snapshots, caches, persistence, execution tokens, or generic
  request-context changes;
- shared Skill parser or executor changes.

## Response Contract

Input remains unchanged:

```json
{
  "skill_name": "review-paper",
  "arguments": "focus on methods"
}
```

Example response:

```json
{
  "skill_name": "review-paper",
  "rendered_prompt": "Review the paper and focus on methods.",
  "declared_tools": ["rag.search", "browser.snapshot"],
  "model_override": null,
  "execution_mode": "inline",
  "supporting_files_omitted": false,
  "dry_run": true,
  "version": 3,
  "catalog_matches": ["rag.search"]
}
```

Every existing field retains its current meaning. `declared_tools` remains
normalized declaration metadata and is neither replaced by `catalog_matches`
nor converted into an execution grant.

`catalog_matches` has three response forms:

| Value | Meaning |
| --- | --- |
| `["rag.search"]` | Matching completed and these declared base names appeared with `canExecute is True`. |
| `[]` | Matching completed with no matches, or the Skill had no valid declaration strings and no catalog read was needed. |
| `null` | The catalog handler was unavailable, catalog lookup failed, or the top-level catalog response was malformed. Matching was not computed. |

An absent field identifies an older server. A returned list is best effort,
not proof that the catalog was complete: the reused protocol handler logs and
suppresses individual module-listing failures. A missing name therefore does
not distinguish absent, disabled, denied, filtered, deferred, or temporarily
unavailable tools.

The name `catalog_matches` is deliberate. `SkillExecutor` already uses
`resolved_tools` internally for a different operation: normalizing declared
specifications against an optional name list. Reusing that name for catalog
base-name matches would create two incompatible meanings.

## Matching Rules

1. Start with non-empty string entries in the verified Skill's parsed
   `allowed_tools` list and trim outer whitespace. These normalized values are
   returned as `declared_tools`, preserving current behavior for valid Skills.
2. Ignore blank and non-string parsed entries. They neither appear in the
   response nor reach the current `.strip()` crash in
   `SkillExecutor.resolve_allowed_tools()`.
3. A declaration without `(` uses the whole normalized string as its base
   name.
4. A command restriction uses the trimmed text before the first `(` as its
   base only when it ends with `)` and both the base and restriction are
   non-empty. For example, `Bash(git *)` has base `Bash`.
5. A malformed command wrapper remains in `declared_tools` but has no catalog
   match.
6. A descriptor matches only when it is a dictionary with a non-empty string
   `name`, its `canExecute` value is exactly boolean `true`, and its name
   exactly and case-sensitively equals a declaration base.
7. Return matching catalog names in first-declaration order and deduplicate by
   exact name.

Aliases, display names, prefixes, capabilities, and tool IDs are not matched.
Malformed descriptors are ignored. A malformed top-level response or non-list
`tools` value produces `null` rather than a computed empty result.

## Request Flow

1. Existing MCP tool authorization and `Skill(skill_name)` authorization run
   unchanged.
2. The module performs existing input, visibility, integrity, and verified
   Skill-load checks.
3. The module normalizes declarations, substitutes arguments directly, builds
   the existing response fields, and applies the rendered-output limit.
4. The request-scoped service lifecycle closes the Skills database.
5. No valid declaration strings returns `catalog_matches: []` immediately.
6. Otherwise, call the active server protocol handler injected through
   `ModuleConfig` once with the same `RequestContext`, bounded by the smaller
   of the Skills module timeout and two seconds. Do not locate a global server
   or construct a protocol per render.
7. Validate the top-level result, match eligible descriptors, and append
   `catalog_matches`.
8. A whole-lookup exception returns `catalog_matches: null` without changing
   the rendered prompt or existing metadata.
9. Cancellation is never converted into `null`. After the handler returns, the
   module also propagates any outstanding task cancellation that the handler's
   existing per-module noncritical exception loop suppressed.

No model, tool, workflow, job, script, or subagent executes in this flow.

## Authorization Boundary

`canExecute` reflects the current embedded listing's module, RBAC, MCP-scope,
tool, and API-key checks. The listing handler does not evaluate future tool
arguments, context `allowed_tools`, the MCP Hub effective policy, approval
leases, path scopes, credentials, quotas, backend health, or later state.

Resolution is internal to authorized `skills.render`; it does not require a
separate grant to invoke the public `mcp.tools.list` tool. The response reveals
only the intersection with names already declared by the authorized Skill,
not unrelated catalog entries or schemas. Normal tool-call enforcement remains
authoritative.

The standalone profile gateway is package infrastructure rather than the
supported in-repository server path. This task does not add interception or
routing to it. A gateway placed in front of the embedded server remains
authoritative when a tool is eventually called.

## Parser And Failure Boundaries

This task does not claim to validate raw YAML declaration types. The current
Skill parser converts some scalar values before `SkillsModule` receives them.
Changing that behavior requires a shared parser and execution-semantics
decision and remains separate work.

The module must preserve existing input, authorization, not-found, integrity,
storage, render-size, database-cleanup, and cancellation behavior. Newly
emitted warnings use the request-bound logger with operation, component, tool,
exception-class, and bounded traceback-frame metadata. They contain no prompt
content, declaration values, catalog payloads, absolute paths, source lines,
policies, credentials, or raw exception text.

## Implementation Scope

The full task's production/documentation changes are limited to:

- one optional catalog-handler dependency on `ModuleConfig`;
- `MCPServer` composition-root wiring for the active protocol handler;
- `tldw_Server_API/app/core/MCP_unified/modules/implementations/skills_module.py`;
- `Docs/MCP/Unified/Modules.md`.

The dependency-injection review follow-up changes only the first three
production paths above; the operator document already describes the public
`catalog_matches` behavior and requires no wording change for handler ownership.

Tests remain in the existing Skills module and integration suites. No frontend,
REST, gateway, profile, protocol implementation, parser, executor, database, or
response-model production file is required.

## Verification

- Existing successful field values remain equivalent for valid Skills.
- Exact and command-restricted declarations match by exact base name.
- Duplicate declarations produce one catalog name in first-declaration order.
- False, missing, non-boolean, and malformed descriptors do not match.
- Blank and non-string parsed declarations do not crash or leak into output.
- No declarations return `[]` without protocol construction or catalog access.
- No matches return `[]` after exactly one catalog read.
- Whole lookup and malformed-envelope failures return `null` while preserving
  the rendered prompt.
- A catalog timeout cancels and drains the lookup task, returns `null`, and
  preserves the rendered prompt.
- A simulated partial catalog is treated as best effort and does not claim
  completeness.
- Internal matching does not require permission to call `mcp.tools.list` and
  never returns an undeclared catalog name.
- Catalog discovery starts after database close.
- Inline and fork Skills remain dry and cannot enter execution branches;
  cancellation is propagated even when the catalog handler suppresses it.
- Existing list/get, visibility, integrity, authorization, render-size,
  cancellation, cleanup, gateway, and package-boundary tests remain passing.
- Focused lint, compile, and Bandit checks pass for touched Python code.

## Acceptance Criteria

1. MCP `skills.render` preserves every existing successful field and adds
   nullable `catalog_matches`.
2. Valid declarations are normalized once and compared with at most one
   embedded catalog result using the same request context after the Skills
   database closes.
3. Matches are unique exact catalog names with `canExecute is True`, ordered by
   first declaration; command restrictions match only by valid base name.
4. No match or no valid declaration produces `[]`; whole lookup failure,
   timeout, or malformed top-level response produces `null`; partial module
   results are documented as best effort.
5. Render uses argument substitution directly, preserves its output limit, and
   cannot enter model or tool execution.
6. No REST/UI change, new public tool, gateway route/snapshot, generic policy
   change, cache, persistence, token, shared parser change, or executor change
   is introduced.
7. Focused regression, integration, lint, compile, package-boundary, and Bandit
   checks pass for the touched scope.
