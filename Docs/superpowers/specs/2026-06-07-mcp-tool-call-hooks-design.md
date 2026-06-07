# MCP Tool-Call Hooks Design

## Status

Draft specification for TASK-2278.

This design adds a hook layer around MCP tool calls so operators can run
policy, logging, validation, and automation before, during, and after tool
execution. Claude Code's hook system is the reference model, but this MCP
server should adapt it to profile policy, path scope, tool-use reporting, and
standalone-library embedding.

## Reference Notes

Claude Code's tools reference treats tool names as policy and hook match
targets. It also uses separate Read, Edit, and Write semantics: Write is a
full-file operation, Edit is targeted replacement, and existing-file mutation
requires read-before-edit/write. Its hooks guide and hooks reference define
tool-call lifecycle events, matcher groups, handler types, decision output,
timeouts, and the rule that hooks can tighten permissions but should not bypass
denies.

Useful behaviors to copy:

- Tool names are stable policy/hook identifiers.
- Tool-event matchers can target exact tools or patterns.
- Pre-tool hooks can deny, ask, or add context before execution.
- Post-tool hooks can observe success/failure but cannot undo side effects.
- Hook input is structured JSON with session, cwd, tool name, and tool input.
- Multiple matching hooks are merged with the most restrictive decision winning.
- Hook output must be explicit and machine-readable for policy decisions.
- Hook results and tool-use telemetry must avoid persisting secrets or raw file
  contents.

Intentional differences:

- MCP should use deterministic ordering for input-rewriting hooks instead of
  nondeterministic "last finisher wins" behavior.
- Shell/command hooks are high risk in a server context and should be disabled
  unless an operator explicitly enables them for a trusted local deployment.
- Hook allow decisions never bypass RBAC, profile policy, deny rules,
  path-scope enforcement, credential grants, or external-server grants.

References:

- Claude Code tools reference:
  <https://code.claude.com/docs/en/tools-reference#edit-tool-behavior>
- Claude Code hooks guide:
  <https://code.claude.com/docs/en/hooks-guide>
- Claude Code hooks reference:
  <https://code.claude.com/docs/en/hooks>

## Goals

- Add operator-configurable hooks for tool-call lifecycle events.
- Support before, during, after-success, after-failure, and batch-level
  observability.
- Let hooks deny, request approval, add model-visible context, or rewrite safe
  arguments before execution.
- Ensure hooks can tighten policy but cannot loosen configured denies or
  profile/path/credential boundaries.
- Keep hook inputs and outputs structured, redacted, bounded, and auditable.
- Support standalone package embedders without requiring tldw_server-specific
  imports.
- Make hook results available to tool-use metrics/evaluations without storing
  sensitive payloads.

## Non-Goals

- No UI implementation in this slice.
- No prompt or agent hooks in the first implementation slice.
- No command/shell hook execution by default in server deployments.
- No hooks that can execute arbitrary MCP tools recursively without loop guards.
- No post-tool rollback guarantee. Post hooks run after effects already
  happened.
- No replacement for profile policy, RBAC, approval, path scope, or credential
  grants.

## Existing Context

`MCPProtocol.prepare_tool_call()` already centralizes tool lookup,
sanitization, input-schema validation, write classification, RBAC/tool
permission checks, path-scope evaluation, runtime approval, governance
preflight, idempotency, and prepared-call integrity.

`execute_prepared_tool_call()` owns execution and tool-use reporting. That gives
the hook system two natural integration points:

- pre-execution hooks inside `prepare_tool_call()` after sanitization and
  candidate extraction, before runtime approval and execution
- post-execution hooks inside `execute_prepared_tool_call()` after success or
  failure classification, before/alongside tool-use reporting

Filesystem hooks must respect the new safe file tools design:

- `fs.read`, `fs.patch`, and `fs.write` are canonical.
- `fs.patch` uses module-derived path candidates.
- `fs.patch` and `fs.write replace` require read receipts or expected hashes in
  strict/default mode.
- legacy `fs.read_text` and `fs.write_text` remain compatibility surfaces.

## Hook Lifecycle

First-slice events:

- `PreToolUse`: fires after tool lookup, argument hardening/sanitization,
  schema validation, write classification, external-source discovery, and
  path-candidate extraction. It fires before approval and before execution.
  Blocking decisions apply here.
- `ToolUseStarted`: fires after approval and prepared-call integrity are
  established, immediately before calling the module. Observation only.
- `ToolUseProgress`: optional during execution for modules that emit structured
  progress. Observation only in the first slice.
- `PostToolUse`: fires after a tool call succeeds. It can add context or
  telemetry but cannot undo the tool's effects.
- `PostToolUseFailure`: fires after a tool call fails or is denied. It can add
  model-visible feedback or telemetry.
- `PostToolBatch`: future event after parallel/batched tool calls resolve. The
  first slice should reserve the event schema but does not need full batching
  support.

Lifecycle ordering:

1. Resolve tool and sanitize arguments.
2. Validate schema.
3. Classify read/write risk.
4. Extract path candidates when needed.
5. Run `PreToolUse` hooks.
6. Re-run validation, write classification, path extraction, and path/policy
   evaluation if any hook rewrites input.
7. Run RBAC/profile/path/credential/approval checks. A hook allow cannot bypass
   these checks.
8. Run governance preflight and build prepared-call integrity tag.
9. Emit `ToolUseStarted`.
10. Execute the module.
11. Emit `PostToolUse` or `PostToolUseFailure`.
12. Record tool-use event with hook outcome metadata.

Security ordering rule:

- A `deny` from `PreToolUse` is final unless a higher-trust managed policy hook
  explicitly returns a documented override. User/project hooks should not
  override deny.
- A hook `allow` means "no objection from this hook"; it does not authorize the
  call by itself.
- Hook rewrites must never skip schema, path-scope, approval, or prepared-call
  integrity checks.

## Hook Configuration

Canonical shape:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "fs.patch|fs.write",
        "handlers": [
          {
            "id": "block-generated-lockfiles",
            "type": "http",
            "url": "http://127.0.0.1:9801/hooks/files",
            "timeout_seconds": 10,
            "mode": "blocking"
          }
        ]
      }
    ]
  }
}
```

Configuration sources, in decreasing trust:

- managed/admin policy hooks
- package/plugin-provided hooks
- workspace/project hooks
- user-local hooks
- profile-scoped hook grants

First-slice storage should be package-neutral:

- `HookConfigStore` interface in `mcp_unified.interfaces`
- tldw_server adapter may store configs in the existing MCP Hub policy/admin
  storage
- standalone package can accept in-memory or file-backed config

Managed hooks can be marked `immutable: true` and `required: true`. A server
setting should support "managed hooks only" for locked-down deployments.

## Matchers

Matcher fields:

- `matcher`: tool-name matcher. Empty or `*` means all tools.
- `if`: optional rule expression that can match tool name plus selected,
  redacted argument fields.
- `risk_classes`: optional list that matches write/destructive/network risk.
- `capabilities`: optional list that matches tool metadata capabilities.

Tool-name matcher syntax:

- plain tool name: exact match, for example `fs.read`
- pipe-separated names: exact any match, for example `fs.patch|fs.write`
- regex when prefixed with `re:`, for example `re:^mcp__.*__write`
- external MCP tool names should be matched after gateway normalization, for
  example `mcp__filesystem__read_file`

Avoid implicit regex based only on punctuation. A required `re:` prefix keeps
configuration predictable and avoids surprising matches.

## Handler Types

First-slice handler types:

- `http`: POST hook input JSON to an allowlisted HTTP endpoint. Recommended for
  server deployments.
- `mcp_tool`: call a hook tool exposed by an already-connected, explicitly
  granted MCP server. Must include recursion guards.
- `command`: run a local executable with hook input on stdin. Disabled by
  default and allowed only for trusted local deployments.

Deferred handler types:

- `prompt`: single-turn model decision hook.
- `agent`: multi-turn verifier with tool access.

Command hook constraints:

- disabled unless `hooks.command.enabled=true`
- command path must be absolute or workspace-relative under an allowed hook
  directory
- no shell-form commands in first slice; use exec-form argv only
- explicit environment allowlist
- fixed cwd from hook config or request workspace
- stdout/stderr size caps
- timeout required with server maximum

HTTP hook constraints:

- URL scheme allowlist, default `http://127.0.0.1` and `https`
- optional host allowlist
- explicit environment interpolation allowlist for headers
- response body size cap
- timeout required with server maximum

MCP tool hook constraints:

- hook target server/tool must be granted by profile or admin policy
- hook tool cannot call back into the same hook event without a recursion token
- hook tool result is sanitized through the same output contract as other
  handlers

## Hook Input

Common input fields:

```json
{
  "hook_event_name": "PreToolUse",
  "hook_invocation_id": "uuid",
  "session_id": "session-1",
  "request_id": "request-1",
  "profile_id": "frontend-engineer",
  "workspace_id": "workspace-1",
  "cwd": "/workspace/project",
  "tool_name": "fs.patch",
  "tool_input": {},
  "tool_metadata": {},
  "risk": {
    "is_write": true,
    "risk_classes": ["filesystem_edit"],
    "capabilities": ["filesystem.edit"]
  },
  "path_scope": {
    "candidate_count": 1,
    "display_paths": ["src/app.py"],
    "actions": ["edit"]
  }
}
```

Redaction rules:

- `tool_input` is included only after the tool-specific argument sanitizer runs.
- Content-bearing fields are replaced with summaries by default:
  - `diff`: byte count, file count, hunk count, display paths
  - `content`: byte count and hash-present boolean
  - `fs.read.content`: never included
  - secrets and credential slot values: never included
- Hook configs can request full arguments only for specific low-risk tools, and
  never for known secret/content fields.
- Absolute paths are replaced with workspace-relative display paths where
  possible.

Event-specific fields:

- `PreToolUse`: sanitized input, derived path candidates, approval context,
  current policy mode
- `ToolUseStarted`: prepared-call id, sanitized summaries, no ability to block
- `ToolUseProgress`: module-emitted progress summary
- `PostToolUse`: sanitized result allowlist, duration, status
- `PostToolUseFailure`: sanitized error class, reason code, governance/path
  summary, duration
- `PostToolBatch`: aggregate statuses and per-tool safe summaries

## Hook Output

Preferred JSON output:

```json
{
  "decision": "deny",
  "reason": "Writes to lockfiles require manual approval",
  "additional_context": "Use fs.patch on package.json only.",
  "updated_input": null,
  "metadata": {
    "policy_id": "lockfile-guard"
  }
}
```

Decision values:

- `no_decision`: default when output is empty or decision absent
- `allow`: no objection; does not bypass ordinary policy
- `ask`: force runtime approval with reason
- `deny`: block execution
- `rewrite`: replace input with `updated_input`, then re-run validation and all
  policy checks

Merging rules:

- all matching blocking hooks run to completion within timeout
- most restrictive decision wins: `deny` > `ask` > `rewrite` > `allow` >
  `no_decision`
- only one `rewrite` can apply; if multiple hooks return rewrite, deny with
  `multiple_hook_rewrites`
- managed hooks run before lower-trust hooks; observer hooks may run in
  parallel
- `additional_context` is concatenated after size limits and sent to the model
  as hook feedback only when safe

Exit-code compatibility for command hooks:

- exit `0` with no JSON means `no_decision`
- exit `2` with stderr means `deny` with stderr as reason
- any other nonzero exit is a hook error and does not block unless the hook is
  configured `fail_closed: true`

## Failure And Timeout Behavior

Per-handler settings:

- `timeout_seconds`
- `fail_closed`
- `mode`: `blocking` or `async`
- `max_stdout_bytes`
- `max_stderr_bytes`
- `max_response_bytes`

Defaults:

- blocking HTTP/MCP/command hooks: 10 seconds
- prompt hooks when added later: 30 seconds
- agent hooks when added later: 60 seconds
- async hooks cannot block, rewrite, or ask

When a blocking hook times out:

- if `fail_closed=true`, deny with `hook_timeout`
- otherwise continue with a warning metadata entry

Post hooks cannot undo tool effects. If a post hook denies or fails closed, the
result should be surfaced as hook feedback and tool-use telemetry, not as a
rollback.

## During-Execution Hooks

The first slice should define but keep during-execution behavior conservative.

`ToolUseStarted` and `ToolUseProgress` are observation events:

- they cannot block, ask, or rewrite
- they can emit telemetry, notifications, or async follow-up work
- they receive only summarized inputs/progress
- modules opt into progress events through a `HookProgressEmitter` interface

Cancellation can be a later slice. A future `ToolUseProgress` hook may request
cooperative cancellation for long-running tools, but only for tools that expose
safe cancellation points.

## Integration Points

New package interfaces:

- `HookConfigStore`: resolves effective hook config for context/profile.
- `HookEvaluator`: runs hook lifecycle events and returns merged decisions.
- `HookHandlerRunner`: runs HTTP, MCP tool, or command handlers.
- `HookInputSanitizer`: builds safe event payloads.
- `HookDecisionMerger`: deterministic merge and precedence rules.

Protocol integration:

- Inject hooks through `MCPRuntimeDependencies`.
- Call `HookEvaluator.evaluate("PreToolUse", ...)` from
  `MCPProtocol.prepare_tool_call()` after schema validation and path candidate
  extraction.
- If `updated_input` is returned, restart validation/path/policy evaluation
  once. Reject repeated rewrites with `hook_rewrite_loop`.
- Call `ToolUseStarted` and post events from `execute_prepared_tool_call()`.
- Include hook decision summaries in `ToolUseEvent` metadata.

Gateway/admin integration:

- Add admin CRUD APIs for hook configs after the core package interface is
  stable.
- Import/export snapshots should include hook configs with secrets redacted.
- Hook configs that reference credentials should use credential grants, not raw
  secret values.

Profile integration:

- Profiles may allow, deny, or require hook groups.
- A profile cannot disable managed required hooks.
- Profiles may grant `mcp_tool` hook targets separately from model-visible
  tools to avoid accidental recursive capability expansion.

## Security Model

Hooks are privileged automation. They can observe tool intent and sometimes
alter or block execution, so they need a stricter model than ordinary tools.

Rules:

- Hooks can tighten permissions but cannot loosen policy.
- Hook config mutation is admin-only.
- Command hooks are local-trusted and disabled by default.
- HTTP hooks require URL allowlists and no implicit secret interpolation.
- MCP tool hooks require explicit hook-target grants and recursion guards.
- All hook inputs and outputs are size capped.
- Hook output shown to the model is sanitized and length capped.
- Hook errors are logged with class/reason, not raw sensitive payloads.
- Managed hooks may be required for compliance and cannot be bypassed by user
  profile selection.

## Observability And Evaluation

Hook telemetry should be metadata-only:

- event name
- matcher id
- handler id/type
- decision
- reason code
- duration bucket
- timeout flag
- fail-open/fail-closed flag
- rewrite occurred boolean
- denied/asked boolean
- additional-context byte count

Do not persist:

- raw hook input
- raw hook output
- file contents
- raw diffs
- secrets
- absolute paths when a workspace-relative path is available

Tool-use evaluation should be able to compare:

- model success with and without hooks
- hook false positives and false negatives
- frequency of denials, asks, rewrites, and timeouts
- per-profile hook impact
- per-tool hook impact

## Testing Strategy

Unit tests:

- matcher exact, pipe, wildcard, and `re:` behavior
- handler config validation by type
- hook input redaction for `fs.read`, `fs.patch`, `fs.write`, command tools,
  and credential-backed external tools
- decision merge precedence
- multiple rewrite denial
- timeout fail-open and fail-closed behavior
- command hook disabled by default
- HTTP header env allowlist behavior
- MCP tool hook recursion guard

Protocol tests:

- `PreToolUse deny` blocks before execution
- `PreToolUse allow` does not bypass profile/path denial
- `PreToolUse ask` forces approval
- `PreToolUse rewrite` re-runs schema/path/policy checks
- hook additional context is surfaced safely
- `ToolUseStarted`, `PostToolUse`, and `PostToolUseFailure` fire in order
- hook metadata appears in tool-use reporting without raw payloads

Integration tests:

- filesystem hook blocks `fs.write` outside operator policy
- hook rewrite cannot move an `fs.patch` diff outside path grants
- post hook failure does not roll back a successful write
- managed required hook cannot be disabled by profile
- snapshot import/export redacts hook secrets

## Rollout

Recommended implementation slices:

1. Package interfaces and typed hook config models.
2. Hook input sanitizer and decision merger.
3. `PreToolUse` blocking hooks with HTTP handler support.
4. Protocol integration and tool-use metadata.
5. `PostToolUse` and `PostToolUseFailure` observer hooks.
6. MCP tool hook handler with recursion guards.
7. Command hook handler behind explicit local-trusted config.
8. Admin routes, CLI, import/export, and docs.
9. Optional prompt/agent hooks and cooperative cancellation.

## Open Questions

- Should project/workspace hook configs be committed files, database records, or
  both?
- Should managed hooks be loaded only by admin config, or can package plugins
  provide required hooks?
- Should prompt/agent hooks use local model providers, configured remote models,
  or only externally supplied hook MCP tools?
- What is the first UI surface: admin settings, profile editor, or CLI only?
