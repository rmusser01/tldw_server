# MCP fs.read, fs.patch, And fs.write Safe File Tools Design

## Status

Draft specification for TASK-2277.

This is the next MCP filesystem safety slice after metadata-only tool-use
reporting. It adds a canonical bounded read primitive, a diff-first edit
primitive, and a deliberate whole-file write primitive while tightening path
policy so profiles can grant different read, edit, and write access by
workspace subpath.

## Summary

Add three filesystem primitives to the workspace-bounded MCP filesystem module:

- `fs.read`: canonical bounded UTF-8 text read primitive. It returns file
  content with hash and truncation metadata so later `fs.patch` and `fs.write`
  calls can be guarded against stale preimages.
- `fs.patch`: preferred edit primitive. It accepts unified diffs, parses them
  in process, derives touched paths before execution, applies hunks with
  conflict detection, and never invokes a host shell or platform patch tool.
- `fs.write`: deliberate whole-file create or replace primitive. It is for
  creating a complete file or intentionally replacing a complete file, not for
  incremental edits. Existing files require an expected content hash.

`fs.read_text` and `fs.write_text` remain as compatibility surfaces for
existing callers, but new role profiles, docs, and model guidance should prefer
`fs.read` for file inspection, `fs.patch` for modifying existing files, and
`fs.write` for intentional whole-file writes.
`fs.write` is intentionally higher impact than `fs.patch`: it creates or
replaces a whole file and therefore requires the explicit `write` path action,
including for replacements.

This slice also makes path constraints action-aware. A profile can read a path
without being allowed to edit it, edit one folder without writing another, or
write multiple workspace roots only when explicitly granted.

## Goals

- Prefer unified-diff editing over raw text replacement whenever possible.
- Provide a canonical read primitive that pairs with hash-guarded patch/write
  flows.
- Keep all file mutation workspace-bounded and profile/path-scope governed.
- Support granular path grants such as:
  - Profile A can read/edit/write `documents/` but cannot access `downloads/`.
  - Profile B can read/edit/write `documents/` and `downloads/`.
  - Profile C can read `downloads/` but cannot edit or write there.
- Parse patch paths during protocol preflight so policy checks use paths
  derived from the diff, not caller-supplied path hints.
- Make every write/edit operation conflict-aware, bounded, UTF-8 only, and
  observable through the existing MCP tool-use metadata contract.
- Preserve the current no-shell, no-subprocess filesystem safety model.

## Non-Goals

- No raw shell execution, `patch`, `git apply`, PowerShell, platform diff tools,
  or subprocess delegation.
- No arbitrary binary patching.
- No chmod, chown, symlink creation, rename, copy, delete, or mode-only patch
  support in the first slice.
- No fuzzy hunk matching, three-way merge, or best-effort context relocation.
- No AST-aware code refactoring.
- No collaborative merge algorithm.
- No UI implementation.
- No change to the existing `fs.read_text` or `fs.write_text` behavior beyond
  metadata/path-scope participation needed for policy consistency.

## Existing Context

`FilesystemModule` currently exposes:

- `fs.list`
- `fs.read_text`
- `fs.write_text`
- `fs.stat`
- `fs.glob`
- `fs.grep`

The module already resolves a trusted workspace root, rejects path escapes,
offloads blocking file I/O with `asyncio.to_thread`, rejects binary text reads,
and uses metadata such as `uses_filesystem`, `path_boundable`, and
`path_argument_hints` for path-scope policy.

`MCPProtocol.prepare_tool_call()` already validates tool schemas, determines
write status, asks the injected `PathScopeEnforcer` to evaluate path scope, and
then runs approval checks before execution. That is the correct boundary for
`fs.read`, `fs.patch`, and `fs.write`.

The current `McpHubPathEnforcementService` supports:

- workspace-root and cwd-descendant path scope modes
- multi-root workspace bundles
- path-only `path_allowlist_prefixes`
- metadata-driven extraction of explicit path arguments

The missing pieces are action-aware grants and derived path extraction for
diff-based tools.

## Design Decisions

### 1. Diff-first edits

Models should use `fs.patch` for existing-file edits. Unified diffs carry
context, changed lines, target paths, and enough structure to detect conflicts.
They are safer than raw search/replace operations because the tool can reject a
patch when the file no longer matches the expected context.

Rejected alternatives:

- Raw exact-text edit primitive first: easier to implement, but it encourages
  brittle unstructured edits and does not match the user's preferred diff-first
  direction.
- Host `patch` or `git apply`: familiar, but violates the no-shell/no-process
  filesystem model and is harder to make portable, bounded, and policy-aware.
- Only improving `fs.write_text`: insufficient because full-file writes are too
  broad for routine edits.

### 2. `fs.read` as the canonical read-side primitive

`fs.read` should become the preferred file-inspection primitive for text files.
It pairs with `fs.patch` and `fs.write` by returning the file hash, newline
style, size, and truncation state needed for conflict-aware follow-up calls.

It must not become an unbounded content exfiltration path:

- Reads require the explicit `read` path action.
- Returned content is UTF-8 text only and subject to byte/line limits.
- Large reads must report truncation clearly rather than silently returning a
  partial file that looks complete.
- The tool response may include file content for the active model call, but
  tool-use reporting and evaluation traces must not persist that raw content.

### 3. `fs.write` as the write-side whole-file primitive

`fs.write` is useful when the model is creating a new file or deliberately
replacing an entire file, such as writing a generated Markdown story, a new
configuration example, or a complete new test file.

It must not become a silent overwrite escape hatch:

- Callers must choose `mode: "create"` or `mode: "replace"`.
- `mode: "create"` fails if the file already exists.
- `mode: "replace"` fails if the file is missing and requires
  `expected_sha256`.
- No default upsert mode in the first slice.
- `mode: "replace"` uses the `write` action rather than `edit`, because it
  replaces the whole file and does not carry hunk-level context.

### 4. Action-aware path grants

Path grants must include action semantics. Existing path-only allowlists cannot
express "read downloads, edit documents" without overgranting.

The new policy shape should be:

```json
{
  "path_scope_mode": "workspace_root",
  "path_scope_enforcement": "approval_required_when_unenforceable",
  "path_grants": [
    { "prefix": "documents", "actions": ["read", "edit", "write"] },
    { "prefix": "downloads", "actions": ["read"] }
  ]
}
```

First-slice grant rules:

- `prefix` is workspace-relative and uses `/` separators.
- Absolute paths, drive-qualified paths, UNC roots, and parent traversal are
  rejected.
- `actions` is an explicit list from `read`, `edit`, `write`.
- Actions do not imply one another. A grant with `write` alone does not allow
  reads or contextual patch edits unless `read` or `edit` is also listed.
- `read` authorizes read tools such as `fs.read`, `fs.read_text`, `fs.stat`,
  `fs.glob`, and `fs.grep`.
- `edit` authorizes `fs.patch` modifications to existing files. The tool may
  read existing file content internally for conflict checks, but it must not
  return raw current content unless the caller also invokes a read tool.
- `write` authorizes `fs.write` create/replace operations and `fs.patch`
  creations.
- A candidate path must match at least one grant prefix that includes the
  required action.
- Prefix matching is path-segment aware. A grant for `documents` matches
  `documents/spec.md` and `documents/nested/file.md`, but not
  `documents-private/file.md`.
- If `path_grants` is present, it is authoritative.
- Existing `path_allowlist_prefixes` remains a compatibility fallback only when
  `path_grants` is absent. It behaves as a path-only scope for whatever tools
  the policy otherwise grants.
- A policy with neither `path_grants` nor `path_allowlist_prefixes` keeps the
  existing path-scope behavior for backward compatibility; strict bundled
  profiles should prefer explicit `path_grants`.

## Tool Contracts

### `fs.read`

Purpose: read a workspace-scoped UTF-8 text file with bounded output and
preimage metadata for later `fs.patch` or `fs.write` calls.

Arguments:

- `path: string` required.
- `start_line: integer` optional, one-based, default `1`.
- `max_lines: integer` optional, default `read_default_max_lines`, bounded by
  `read_max_lines`.
- `max_bytes: integer` optional, default `read_default_max_bytes`, bounded by
  `read_max_bytes`.
- `include_line_numbers: boolean` optional, default `false`.

Response:

- `path`
- `content`
- `start_line`
- `end_line`
- `line_count_total` when cheap to compute within limits
- `bytes_read`
- `bytes_total`
- `sha256`
- `newline_style`: `lf`, `crlf`, `cr`, or `mixed`
- `truncated`: boolean
- `truncation_reason`: `max_bytes`, `max_lines`, or absent

Failure response or exception reason codes should be stable:

- `path_outside_workspace_scope`
- `path_outside_allowlist_scope`
- `path_action_not_granted`
- `file_missing`
- `file_not_regular`
- `binary_content_not_supported`
- `file_too_large`
- `read_too_large`
- `invalid_line_range`

Behavior:

- Requires the `read` path action.
- Resolves and validates the path with the same workspace/path-scope rules as
  other filesystem tools.
- Reads regular files only; symlink targets are rejected in the first slice.
- Rejects binary/NUL content and non-UTF-8 files with
  `binary_content_not_supported`.
- Applies byte and line limits before returning content.
- Returns `sha256` for the complete file, not only the returned slice, when the
  file is within `read_hash_max_file_bytes`.
- If the file is too large to hash under configured limits, returns a stable
  reason such as `hash_omitted_file_too_large` and keeps `sha256` absent.
- Does not persist returned `content` in tool-use reporting, audit summaries, or
  evaluation traces.

### `fs.patch`

Purpose: apply one unified diff to workspace-scoped UTF-8 text files.

Arguments:

- `diff: string` required. Unified diff text.
- `expected_sha256_by_path: object[string,string]` optional. Workspace-relative
  paths mapped to expected preimage hashes.
- `create_parent_directories: boolean` optional, default `false`.
- `allow_create: boolean` optional, default `false`.
- `dry_run: boolean` optional, default `false`.

Response:

- `applied`: boolean.
- `dry_run`: boolean.
- `files`: list of per-file results:
  - `path`
  - `action`: `edit` or `write`
  - `created`
  - `hunks_applied`
  - `additions`
  - `deletions`
  - `bytes_written` when applied
  - `sha256_before` when the file existed
  - `sha256_after`
- `summary`: aggregate file, hunk, addition, and deletion counts.
- `reason_code` on failure.

Failure response or exception reason codes should be stable:

- `invalid_unified_diff`
- `unsupported_diff`
- `path_outside_workspace_scope`
- `path_outside_allowlist_scope`
- `path_action_not_granted`
- `patch_conflict`
- `expected_hash_mismatch`
- `file_missing`
- `file_already_exists`
- `binary_content_not_supported`
- `file_too_large`
- `patch_too_large`
- `too_many_files`
- `too_many_hunks`
- `partial_write_rollback_attempted`

Supported unified diff subset:

- `diff --git a/path b/path` headers are accepted.
- `--- a/path` and `+++ b/path` headers are accepted.
- Standard `@@ -old_start,old_count +new_start,new_count @@` hunks are
  required.
- Context, removal, and addition lines use standard leading ` `, `-`, and `+`.
- `a/` and `b/` prefixes are stripped only when they are conventional diff
  prefixes.
- `/dev/null` is accepted only for file creation when `allow_create=true`.
- `\ No newline at end of file` is supported and reflected in the in-memory
  result.
- Multiple file sections targeting the same normalized path are rejected in the
  first slice. A single file section may still contain multiple hunks.
- Diff paths use POSIX `/` separators regardless of host OS. Backslash paths are
  rejected in the first slice to avoid Windows escape and drive-prefix
  ambiguity.

Rejected in the first slice:

- binary patches
- renames
- copies
- deletes
- mode-only changes
- symlink path writes
- absolute paths
- Windows drive-qualified paths
- UNC paths
- parent traversal
- malformed hunk counts
- mixed file headers that disagree about the target path

Patch application behavior:

- Parse the entire diff before any file I/O.
- Derive every touched path and required action from the parsed diff.
- Run path/action policy before reading or writing target files.
- Normalize parsed paths to a workspace-relative display path and a resolved
  filesystem path before policy checks. Only workspace-relative display paths
  may appear in responses, approval payloads, or telemetry.
- Read existing target files as UTF-8 bytes with binary/NUL detection.
- Reject files over `patch_max_file_bytes`.
- Apply all hunks in memory before writing any file.
- Match hunk context against the current file; conflicts fail before writes.
- Preserve existing dominant newline style for edited files.
- Default new files to LF.
- Use temporary files plus replace for each write.
- For multi-file writes, dry-run all files before commit. If a write fails
  during commit, attempt rollback from in-memory backups and return or raise
  `partial_write_rollback_attempted`. The tool must not claim crash-proof
  filesystem transactions.

### `fs.write`

Purpose: create or deliberately replace a complete UTF-8 text file.

Arguments:

- `path: string` required.
- `content: string` required.
- `mode: "create" | "replace"` required.
- `expected_sha256: string` required for `mode="replace"`.
- `create_parent_directories: boolean` optional, default `false`.
- `dry_run: boolean` optional, default `false`.

Response:

- `path`
- `mode`
- `applied`
- `dry_run`
- `created`
- `bytes_written`
- `sha256_before` when replacing
- `sha256_after`

Behavior:

- `create` requires the path to be absent.
- `replace` requires the path to exist, be a regular file, be UTF-8 text, and
  match `expected_sha256`.
- No upsert mode in the first slice.
- Parent directories are created only for `mode="create"` when
  `create_parent_directories=true` and all parent paths remain inside the
  allowed workspace/path grant. `mode="replace"` requires the file and parent
  directory to already exist.
- Symlink targets are rejected in the first slice.
- Content is UTF-8 only and subject to `write_max_bytes`.
- Writes use temporary files and atomic replace where supported by the OS.

### `fs.read_text`

`fs.read_text` remains for compatibility. It should keep its current request
shape and behavior unless a later migration explicitly deprecates it.

For policy consistency:

- It must remain path-boundable.
- It requires the `read` path action when action-aware grants are enabled.
- New profile recommendations should prefer `fs.read`.
- Strict new profiles may omit `fs.read_text` while keeping `fs.read`.

### `fs.write_text`

`fs.write_text` remains for compatibility. It should keep its current request
shape and behavior unless a later migration explicitly deprecates it.

For policy consistency:

- It must remain path-boundable.
- It requires the `write` path action when action-aware grants are enabled.
- New profile recommendations should prefer `fs.write`.
- Strict new profiles may omit `fs.write_text` while keeping `fs.write` and
  `fs.patch`.

## Path Scope Preflight

`fs.patch` cannot rely on `path_argument_hints` because the paths are embedded
inside `diff`. The protocol and module need a derived path-candidate seam.

Add an optional module hook:

```python
def extract_path_scope_candidates(
    self,
    tool_name: str,
    arguments: dict[str, Any],
) -> list[dict[str, str]]:
    ...
```

Candidate shape:

```json
{
  "path": "documents/spec.md",
  "action": "edit",
  "source": "unified_diff"
}
```

Protocol behavior:

1. Sanitize and validate tool arguments.
2. If the module implements `extract_path_scope_candidates`, call it before
   path enforcement.
3. Treat extraction failures as invalid params.
4. Pass the derived candidates to `PathScopeEnforcer`.
5. Ensure approval and governance payloads include only bounded relative paths
   or path-scope summaries, not raw diff text.

Compatibility requirement:

- Extend `PathScopeEnforcer.evaluate_tool_call` in a backward-compatible way,
  for example by adding an optional `derived_candidates` argument or a small
  companion method. Existing enforcers that only understand metadata hints
  should continue to work for ordinary tools.
- `fs.patch` must fail closed when the protocol cannot pass derived candidates
  to the active enforcer, because metadata hints cannot prove which files the
  diff touches.
- `fs.read` and `fs.write` may continue to use ordinary path hints because each
  tool path is an explicit schema field.

Path enforcer behavior:

- Use derived candidates when present.
- Fall back to metadata path hints for ordinary tools.
- Check every candidate path against workspace root, cwd scope, multi-root
  workspace bundle, and action-aware `path_grants`.
- Deny the whole request when any candidate is outside scope or lacks its
  required action.

This is required for security. The caller must not provide a separate `paths`
field for `fs.patch`, because that lets the request lie about the diff.

Candidate extraction should parse only enough of the diff to identify file
sections and intended actions during preflight. Full hunk application still
belongs to the execution planner, and both phases should share the same parser
library so they cannot disagree about target paths.

## Tool Metadata

`fs.read` metadata:

```json
{
  "category": "management",
  "uses_filesystem": true,
  "path_boundable": true,
  "capabilities": ["filesystem.read"],
  "path_argument_hints": ["path"],
  "path_scope_action": "read",
  "eval": {
    "action_family": "filesystem_read",
    "expected_result_kind": "structured_filesystem_read"
  }
}
```

`fs.patch` metadata:

```json
{
  "category": "management",
  "uses_filesystem": true,
  "path_boundable": true,
  "capabilities": ["filesystem.edit"],
  "path_scope_candidate_source": "module",
  "eval": {
    "action_family": "filesystem_patch",
    "expected_result_kind": "structured_filesystem_edit"
  }
}
```

`fs.write` metadata:

```json
{
  "category": "management",
  "uses_filesystem": true,
  "path_boundable": true,
  "capabilities": ["filesystem.write"],
  "path_argument_hints": ["path"],
  "path_scope_action": "write",
  "eval": {
    "action_family": "filesystem_write",
    "expected_result_kind": "structured_filesystem_write"
  }
}
```

Read tools should declare `path_scope_action: "read"`. Existing
`fs.read_text` should declare `path_scope_action: "read"`, and existing
`fs.write_text` should declare `path_scope_action: "write"`.

For `fs.patch`, the module-derived candidates decide each file action:
existing-file modifications require `edit`, and file creations require `write`.
For `fs.write`, both `create` and `replace` require `write`.

## Profile And Preset Guidance

Package defaults should keep current compatibility but move toward:

- `_FILES_READ_TOOLS`: `fs.list`, `fs.read`, `fs.stat`, `fs.glob`, `fs.grep`
- `_FILES_EDIT_TOOLS`: `_FILES_READ_TOOLS`, `fs.patch`
- `_FILES_WRITE_TOOLS`: `_FILES_EDIT_TOOLS`, `fs.write`
- `_LEGACY_FILES_READ_TOOLS`: `fs.read_text`
- `_LEGACY_FILES_WRITE_TOOLS`: `fs.write_text`

Profiles that inspect files should receive `fs.read`. Profiles that need
routine source/document edits should receive `fs.patch`. Profiles that generate
complete files should receive `fs.write`. Existing profiles that currently have
`fs.read_text` or `fs.write_text` can keep them until a compatibility review
decides whether to replace them.

The default preset update should avoid silently broadening existing roles. If a
role currently has no file read capability, adding `fs.read` must be a
deliberate preset change backed by an explicit `read` path grant. If a role
currently has no file mutation capability, adding `fs.patch` or `fs.write` must
be a deliberate preset change backed by an explicit path grant. If a role
already has `fs.read_text` or `fs.write_text`, the migration may add `fs.read`,
`fs.patch`, and `fs.write` while keeping the legacy tool temporarily for
compatibility.

Example policy snippets:

Profile A:

```json
{
  "allowed_tools": ["fs.read", "fs.patch", "fs.write"],
  "policy_document": {
    "path_scope_mode": "workspace_root",
    "path_grants": [
      { "prefix": "documents", "actions": ["read", "edit", "write"] }
    ]
  }
}
```

Profile B:

```json
{
  "allowed_tools": ["fs.read", "fs.patch", "fs.write"],
  "policy_document": {
    "path_scope_mode": "workspace_root",
    "path_grants": [
      { "prefix": "documents", "actions": ["read", "edit", "write"] },
      { "prefix": "downloads", "actions": ["read", "edit", "write"] }
    ]
  }
}
```

Profile C:

```json
{
  "allowed_tools": ["fs.read", "fs.stat", "fs.glob", "fs.grep"],
  "policy_document": {
    "path_scope_mode": "workspace_root",
    "path_grants": [
      { "prefix": "downloads", "actions": ["read"] }
    ]
  }
}
```

## Command Runtime Follow-Up

The governed `run`/`bash`/`shell` runtime may later add commands such as:

- `cat <path>` or `sed -n ...` -> `fs.read`
- `patch < diff.patch` -> `fs.patch`
- `write-file <path> ...` -> `fs.write`

That should be a separate follow-up. This slice should ship typed MCP tools and
policy first. The command runtime must continue to call typed MCP tools through
`prepare_tool_call` so path grants, approval, and tool-use reporting remain in
force.

## Observability And Evaluation

Tool-use events and metrics must not persist raw diffs, raw file contents,
absolute paths, or raw errors.

Safe metadata:

- requested tool
- effective tool
- action family
- bytes read/written
- line range when applicable
- file count
- hunk count
- additions/deletions counts
- read/created/edited/write booleans
- dry-run flag
- status
- reason code
- conflict flag
- path-scope status
- truncation/limit flags
- duration bucket
- profile or mode id after existing sanitization

Evaluation traces for all filesystem tools should use the same redaction
contract. `fs.read`, `fs.read_text`, `fs.patch`, `fs.write`, `fs.write_text`,
and other read tools may report structured outcomes and scoped path summaries,
but they must not persist content-bearing arguments or returned file text in
tool-use evaluation records.
For `fs.read`, this means traces can record byte counts, truncation state,
hash-present/hash-omitted status, and scoped path summaries, but not `content`.

## Limits

Recommended settings:

- `read_max_bytes`
- `read_max_lines`
- `read_default_max_bytes`
- `read_default_max_lines`
- `read_hash_max_file_bytes`
- `patch_max_bytes`
- `patch_max_files`
- `patch_max_hunks`
- `patch_max_changed_lines`
- `patch_max_file_bytes`
- `write_max_bytes`
- `write_create_parent_directories_default`

All limits must fail closed with stable reason codes.

Limits should be evaluated before expensive work where possible. Read byte and
line caps should be applied before returning content. The diff byte limit can be
checked before parsing, file and hunk count limits during parsing, and file-size
limits before loading each target file.

## Testing Strategy

Filesystem module tests:

- `fs.read` descriptor includes metadata, strict schema, and eval metadata.
- `fs.read` returns UTF-8 content, full-file hash, byte counts, newline style,
  and truncation metadata.
- `fs.read` enforces byte/line limits and clearly marks truncated responses.
- `fs.read` rejects path escapes, symlinks, binary/NUL content, non-UTF-8
  content, oversized hash requests, and unsupported arguments.
- `fs.patch` descriptor includes metadata, strict schema, and eval metadata.
- `fs.write` descriptor includes metadata, strict schema, and eval metadata.
- Unified diff parser accepts standard single-file and multi-file hunks.
- Parser rejects binary, rename, copy, delete, mode-only, malformed count,
  absolute, drive-qualified, UNC, and parent traversal patches.
- Parser rejects backslash paths, duplicate target file sections, and segment
  prefix collisions that would bypass grants such as `documents` versus
  `documents-private`.
- Patch applies cleanly to an existing UTF-8 file.
- Patch creates a file only with `allow_create=true` and write action.
- Patch detects context conflict without writing.
- Patch detects expected hash mismatch without writing.
- Multi-file patch writes nothing when one file conflicts.
- Multi-file patch writes nothing when one file is outside path grant.
- Symlink patch targets fail closed.
- UTF-8, NUL/binary, file-size, patch-size, hunk-count, and changed-line limits.
- No-newline marker behavior.
- Dry run returns planned results without writes.
- `fs.write create` writes a new file and fails if it exists.
- `fs.write replace` requires and validates `expected_sha256`.
- `fs.write` rejects path escapes, symlinks, binary/NUL content, oversized
  content, missing parents unless enabled, and unsupported mode.

Policy/path-scope tests:

- Profile A can read/edit/write `documents/` and cannot access `downloads/`.
- Profile B can read/edit/write both `documents/` and `downloads/`.
- Profile C can read `downloads/` and is denied for `fs.patch` and `fs.write`.
- Action inheritance is absent: `write` without `read` denies read tools, and
  `read` without `edit` or `write` denies mutation tools.
- `fs.patch` path checks use derived diff paths, not caller-supplied hints.
- Any denied path/action denies the entire multi-file patch.
- Existing `path_allowlist_prefixes` behavior remains compatible when
  `path_grants` is absent.
- Multi-root workspace bundle enforcement still maps each path to an allowed
  workspace root before action grant checks.

Protocol tests:

- `prepare_tool_call` calls module path-candidate extraction before path
  enforcement.
- Extraction errors map to invalid params.
- Path-scope denials happen before filesystem reads/writes.
- Approval payloads include bounded path-scope summaries.
- Tool-use reporting records status/reason/action metadata without raw diffs.
- Tool-use reporting records `fs.read` metadata without returned file content.

Command runtime tests are deferred until command aliases are added.

## Rollout

Recommended implementation slices:

1. Action-aware path grants in `McpHubPathEnforcementService`, with compatibility
   fallback for `path_allowlist_prefixes`.
2. Protocol/module derived path-candidate seam.
3. `fs.read` tool execution and tests.
4. Unified diff parser and in-memory patch planner.
5. `fs.patch` tool execution and tests.
6. `fs.write` tool execution and tests.
7. Profile metadata, docs, and package-boundary verification.

Each slice should use TDD and focused security scans on touched Python code.

## Design Review Findings Incorporated

- `path_argument_hints` alone cannot secure `fs.patch`; a derived path-candidate
  seam is required.
- Path grants must be action-aware to support read-only, edit-only, and
  write-capable folders.
- `fs.read` should be a first-class bounded primitive rather than relying only
  on legacy `fs.read_text`, because safe patch/write flows need hashes,
  truncation state, and consistent read-action policy.
- `fs.write` must be treated as a separate whole-file primitive with explicit
  create/replace modes, required replace hashes, and `write` action semantics.
- Unified diff support must be intentionally narrow and reject unsupported
  patch features.
- Atomicity must be described as dry-run plus best-effort rollback, not
  crash-proof multi-file transactions.
- Conflict detection requires hunk context and optional expected hashes.
- Encoding and newline behavior must be specified.
- Symlinks should fail closed in the first slice.
- Tool-use reporting must avoid raw diffs and contents.
- Tests must include a Profile A/B/C permission matrix and denial no-write
  assertions.
- Cross-platform path handling must reject ambiguous diff paths and perform
  path-segment-aware grants so policy behavior is consistent across macOS,
  Linux, and Windows hosts.
