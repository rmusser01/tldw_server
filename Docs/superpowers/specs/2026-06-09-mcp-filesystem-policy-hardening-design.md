# MCP Filesystem Policy Hardening Design

## Context

The current `origin/dev` branch already includes the broad filesystem primitives:
`fs.read`, `fs.write`, `fs.edit`, `fs.patch`, `fs.stat`, `fs.glob`, and
`fs.grep`, plus legacy `fs.read_text` and `fs.write_text`. The newer tools carry
file-policy metadata, hashes, read receipts, bounded reads, and module-derived
path-scope candidates for unified diffs.

The remaining risk is not missing primitives. It is making sure the policy
boundary is action-aware before those primitives become the default surface:

- `read` grants must not imply `edit` or `write`.
- `edit` grants must not imply whole-file `write` unless explicitly granted.
- `fs.patch` must enforce each changed file with the candidate action derived
  from the diff: `edit` for modifying existing files and `write` for creating
  files.
- The host path-enforcement service already accepts module-derived candidates,
  but the default `TldwPathScopeEnforcer` adapter must also accept and forward
  them. Otherwise `fs.patch` can fail closed through the real runtime path while
  passing isolated protocol tests.
- The virtual CLI runtime still routes `cat` and `write` through legacy
  `fs.read_text` and `fs.write_text`, so it does not yet prefer the structured
  primitives where safe.

## Goals

1. Lock down action-specific filesystem policy behavior with focused regression
   tests for `fs.read`, `fs.edit`, `fs.write`, and `fs.patch`.
2. Prefer structured filesystem primitives in the virtual CLI runtime where the
   command grammar can preserve semantics safely.
3. Keep legacy tools available for compatibility, but avoid expanding their use
   as the default path for new command/runtime behavior.
4. Keep this as a small hardening slice. Larger shell alias parsing, locks,
   temporary grants, and richer unified-diff compatibility remain separate
   follow-up items.

## Non-Goals

- Do not add raw shell execution or the bash/shell alias parser in this branch.
- Do not implement delete, rename, move, share/export, chmod/admin, or lock
  actions.
- Do not replace the existing path-enforcement service with a new policy engine.
- Do not make a bare `write <path> <content>` command silently choose between
  create and replace if the runtime cannot represent that choice explicitly.
- Do not add a structured replace command that bypasses, hides, or weakens
  `fs.write` preimage requirements.

## Design

### Policy Regression Coverage

Add tests at the enforcement boundary rather than only checking tool metadata.
The tests should exercise the real `McpHubPathEnforcementService` decision path
with synthetic workspace scope data and effective policies that include
`path_grants`.

Required cases:

- A profile with `read` on `docs` can run `fs.read` for `docs/a.txt`.
- The same profile is denied for `fs.edit`, `fs.write`, and an `fs.patch` modify
  candidate under `docs`.
- A profile with `edit` on `docs` can run `fs.edit` and `fs.patch` modify
  candidates under `docs`, but is denied for `fs.write` create/replace unless
  `write` is separately granted.
- A profile with `write` on `docs` can run `fs.write` and an `fs.patch` create
  candidate under `docs`.
- A `write` grant permits whole-file create/replace operations, but does not
  imply `read` or bounded `edit` visibility. If a profile needs all three, it
  must grant all three explicitly.
- Internal preimage reads performed by `fs.write` replace mode are part of the
  guarded `write` operation. They do not require a separate `read` grant, but
  the caller still must satisfy `fs.write` preimage authorization with an
  expected hash or read receipt.
- A deny grant on a more specific prefix, such as `docs/private`, wins over a
  broader allow grant on `docs`.
- Patch bundles fail closed if any candidate path/action is not granted, even
  when other files in the same diff are allowed.

This coverage is the acceptance gate for making structured tools more visible.

### Candidate Forwarding Boundary

Update `TldwPathScopeEnforcer.evaluate_tool_call` to accept the same optional
`path_scope_candidates` argument as the shared policy protocol and forward it to
`McpHubPathEnforcementService.evaluate_tool_call`.

This is intentionally a seam fix, not a new policy behavior. The service already
knows how to evaluate candidate paths/actions; the adapter should not drop that
context or force the protocol into the compatibility fallback.

### Filesystem Tool Metadata

Keep the existing metadata contract:

- Direct path tools use `path_scope_action`.
- `fs.patch` uses `path_scope_candidate_source: "module"` because one diff may
  contain multiple files and mixed actions.
- Legacy tools stay marked with `legacy_tool` and replacement metadata.

Only adjust metadata if tests reveal a mismatch. The preferred outcome is to
preserve the current contract and strengthen tests around it.

### Virtual CLI Runtime

Move command runtime defaults away from legacy primitives only where semantics
are unambiguous:

- Change `cat <path>` from `fs.read_text` to `fs.read`. The renderer already
  accepts either `text` or `content`, so this should preserve CLI output while
  adding hash/read-receipt behavior behind the scenes.
- Preserve compatibility for profiles that still expose only `fs.read_text` if
  the runtime can do so without making `cat` visible when neither read tool is
  executable. The command registry and adapter must agree on the selected
  backing tool: prefer `fs.read` when visible, fall back to `fs.read_text` only
  when `fs.read` is unavailable and `fs.read_text` is visible.
- Keep legacy `write <path> <content>` available for compatibility only if it
  still maps to `fs.write_text`. Do not silently remap it to `fs.write`, because
  `fs.write` requires an explicit `create` or `replace` mode and has stricter
  preimage semantics.
- Add explicit structured write commands:
  `write-create <path> <content>` maps to `fs.write` with `mode: "create"`.
  A replacement command may be added only if its syntax carries preimage
  authorization, for example
  `write-replace --expected-sha256 <hash> <path> <content>` or
  `write-replace --read-receipt <receipt> <path> <content>`. If that parser
  addition would broaden this slice too much, defer `write-replace` and keep
  only `write-create`.
- Update `RunCommandModule.is_write_tool_call` and its tests so structured
  write commands backed by `fs.write` are classified as write-capable for
  approval, audit, and prompt-surface purposes.

This gives models a safer structured route without breaking callers that still
depend on the old `write` command.

### Diff-First Editing

`fs.patch` remains the preferred edit primitive for generated changes when a
unified diff is available. `fs.edit` remains a bounded exact-string fallback for
small literal substitutions. This branch should not attempt full git-apply
parity unless a current test proves the existing parser blocks normal use.

If a minimal parser gap is found during implementation, it may be fixed in this
branch only when it is directly required for the policy/runtime tests. Larger
diff compatibility improvements stay in a follow-up task.

## Error Handling

Path-policy failures should keep the existing safe troubleshooting shape:

- `within_scope: false`
- `reason` or path-decision `reason_code`
- redacted normalized relative paths and grant metadata
- no raw file content
- no absolute paths in user-facing denial payloads

Patch enforcement should fail the whole request when any candidate is denied.
Partial application must not occur.

## Testing

Focused test targets:

- `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_path_scope_candidates.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_command_runtime_registry.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_command_runtime_execution.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_run_command_module.py`
- `tldw_Server_API/app/core/MCP_unified/adapters/tldw_policy.py`
- `tldw_Server_API/tests/MCP_unified/test_mcp_protocol_path_scope.py` for
  service/protocol path-scope enforcement coverage

Validation commands should include the touched MCP test files and Bandit over
the touched Python implementation files. Docs-only spec work can skip Bandit
until code is touched.

## Open Follow-Ups

- Full shell alias parsing with command splitting, wrapper handling, and
  structured tool dispatch.
- Fuller unified-diff compatibility, including common git diff headers and
  no-newline markers, if real workloads need it.
- File lock leases, temporary session grants, and permission-change governance.
- Observability/evaluation coverage for all tool families beyond this small
  filesystem hardening pass.
