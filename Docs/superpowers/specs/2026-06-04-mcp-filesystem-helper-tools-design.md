# MCP Filesystem Helper Tools Design

## Summary

Add the next native/default-included MCP filesystem helper slice to the existing
workspace-bounded `FilesystemModule`: `fs.stat`, `fs.glob`, and `fs.grep`.

This slice intentionally stays read-only. It expands discovery/search workflows
for default role profiles without introducing edit/patch semantics, arbitrary
process execution, shell globbing, or OS-specific command behavior. `fs.edit`
and `fs.patch` remain separate follow-up work because write conflict handling,
encoding preservation, and approval semantics need their own review.

The existing governed `run(command=...)` virtual CLI is the right place for
shell-shaped ergonomics. This filesystem slice should expose typed MCP tools
first; command aliases such as `stat`, `find`/`glob`, `rg`, and optional
`bash`/`shell` facade names should be handled as a follow-up over the governed
command runtime, not by invoking a host shell.

## Goals

- Provide portable filesystem inspection and text-search helpers for MCP/ACP
  workspaces.
- Reuse the current workspace root resolver and path-scope enforcement.
- Make behavior deterministic across macOS, Linux, Windows, containers, and
  network-mounted workspaces.
- Keep all new tools read-only and governed by the existing
  `filesystem.read` capability metadata.
- Update bundled profile metadata so read-capable profiles can discover the new
  helpers.

## Non-Goals

- No free-form shell, subprocess grep, `find`, PowerShell, or platform command
  delegation.
- No new `bash`/`shell` tool alias in this implementation slice. A follow-up
  may expose those names only as aliases for the governed `run` runtime.
- No writes, patches, file deletion, renames, chmod/chown, or symlink creation.
- No binary search, archive traversal, or repository indexing service.
- No attempt to normalize filesystem case rules globally. The tools provide a
  deterministic application-level `case_sensitive` option instead.
- No external MCP server installation or browser/CDP integration in this slice.

## Existing Context

`tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`
already provides:

- `fs.list`
- `fs.read_text`
- `fs.write_text`

The module resolves a trusted workspace root through
`McpHubWorkspaceRootResolver`, rejects path escapes, offloads blocking file I/O
with `asyncio.to_thread`, rejects binary text reads, caps large reads, and emits
tool metadata used by policy/path-scope layers.

This design extends that module rather than creating a second filesystem
module.

The codebase also already includes
`tldw_Server_API/app/core/MCP_unified/modules/implementations/run_command_module.py`
and `tldw_Server_API/app/core/MCP_unified/command_runtime/`. That runtime
accepts CLI-shaped strings, parses pipes and chain operators, and routes
commands through policy-checked MCP tool calls. It is intentionally not a raw
host shell. New filesystem helpers should be designed so that a later command
runtime task can safely map familiar command names to these typed tools.

## Tool Contracts

### `fs.stat`

Purpose: return metadata for one workspace-scoped path.

Arguments:

- `path: string` required.
- `follow_symlinks: boolean` optional, default `false`.

Response:

- `path`: workspace-relative path using `/` separators.
- `name`: final path segment.
- `type`: `file`, `directory`, `symlink`, or `other`.
- `size`: byte size when available.
- `modified_at`: UTC ISO timestamp when available.
- `mode`: permission bits as an integer when available.
- `is_symlink`: boolean.
- `target_within_workspace`: boolean only when `follow_symlinks=true` or when
  the platform exposes enough symlink metadata without leaking the target.

Security behavior:

- Does not return a symlink target path.
- If `follow_symlinks=true`, the resolved target must remain inside the
  workspace root or the call fails with `PermissionError`.
- If `follow_symlinks=false`, metadata is gathered with non-following stat
  behavior where the platform supports it.

### `fs.glob`

Purpose: enumerate workspace-relative paths matching a portable pattern.

Arguments:

- `pattern: string` required. Portable patterns use `/` separators. Backslashes
  are accepted as separators and normalized before matching.
- `base_path: string` optional, default `"."`.
- `include_hidden: boolean` optional, default `false`.
- `include_files: boolean` optional, default `true`.
- `include_directories: boolean` optional, default `true`.
- `follow_symlinks: boolean` optional, default `false`.
- `case_sensitive: boolean` optional, default `true`.
- `limit: integer` optional, default from module setting `glob_result_limit`
  or `500`.

Response:

- `base_path`: normalized workspace-relative base path.
- `pattern`: normalized portable pattern.
- `matches`: sorted list of `{path, type, size?}` records.
- `truncated`: boolean.
- `remaining_count`: count of additional matches skipped after the limit.

Matching behavior:

- Implemented in Python using `os.walk`, `fnmatch.fnmatchcase`, and normalized
  POSIX-style relative paths. It must not invoke a shell or rely on
  platform-native glob behavior.
- Matching is deterministic by default: `case_sensitive=true` means exact
  string matching on every OS, including Windows and default macOS filesystems.
- `case_sensitive=false` lowercases both the candidate path and pattern before
  matching.
- `**` is supported through a small wrapper around `fnmatch`. Patterns such as
  `**/*.py` must match both `app.py` and `pkg/app.py`, so the implementation
  cannot rely on raw `fnmatch.fnmatchcase` alone.
- Absolute patterns, drive-qualified patterns, UNC-style roots, and parent
  traversal segments are rejected.
- Directory traversal stops at a configurable `glob_walk_entry_limit` even
  when the result limit has not been reached. This prevents a broad pattern
  such as `**/*` from walking an unexpectedly large workspace forever.

### `fs.grep`

Purpose: search UTF-8 text files under a workspace-scoped base path.

Arguments:

- `pattern: string` required.
- `base_path: string` optional, default `"."`.
- `include: list[string]` optional, default `["*", "**/*"]`.
- `exclude: list[string]` optional, default `[]`.
- `regex: boolean` optional, default `false`; accepted only when the module
  setting `grep_allow_regex` is enabled.
- `case_sensitive: boolean` optional, default `true`.
- `include_hidden: boolean` optional, default `false`.
- `follow_symlinks: boolean` optional, default `false`.
- `limit: integer` optional, default from module setting `grep_result_limit`
  or `200`.
- `max_file_bytes: integer` optional, capped by module setting
  `grep_max_file_bytes` or the current `max_read_bytes`.
- Global scan limits come from module settings `grep_max_total_bytes`,
  `grep_max_files`, and `grep_walk_entry_limit`.

Response:

- `base_path`: normalized workspace-relative base path.
- `matches`: sorted list of match records:
  - `path`
  - `line_number` (1-based)
  - `line`
  - `match_text`
- `truncated`: boolean.
- `remaining_count`: count of additional matches skipped after the limit.
- `remaining_count_known`: false when the scan stops early because a walk-entry,
  file, or byte budget was exhausted.
- `truncation_reasons`: zero or more of `walk_entry_limit`, `io_budget`,
  `file_budget`, and `match_limit`.
- `skipped`: counts for `binary`, `decode_error`, `too_large`,
  `permission_error`, and `unsupported_type`.

Search behavior:

- Implemented in Python. No shell, subprocess, ripgrep, GNU grep, BSD grep, or
  PowerShell dependency.
- Reads files as bytes, skips files containing NUL bytes, and decodes UTF-8.
- Non-UTF-8 files are skipped with `decode_error`.
- Literal search is the default. Regex search uses Python `re` with normal
  module limits, bounded pattern length, bounded file size, and explicit regex
  compile-error responses.
- Line numbers use universal newline handling via `splitlines()`, so CRLF,
  LF, and CR are reported consistently.
- Results are sorted by normalized path and line number.
- Include/exclude matching uses the same portable pattern helper as `fs.glob`,
  including `**/` matching zero or more directories.
- Directory traversal stops at a configurable `grep_walk_entry_limit` even
  when the match limit has not been reached.

## Cross-Platform Policy

All external MCP clients should see stable behavior independent of the host OS:

- Public paths use `/` separators in responses.
- Inputs accept `/` on every OS; backslashes are normalized as separators for
  compatibility.
- Windows drive prefixes, UNC roots, and POSIX absolute patterns are rejected
  for pattern arguments.
- Path resolution continues to use the trusted workspace root and rejects
  escapes after normalization.
- Case sensitivity is explicit. Defaults do not depend on NTFS, APFS, ext4, or
  network share behavior.
- Hidden path handling is application-defined: a path is hidden when any
  workspace-relative segment starts with `.`. The tools do not attempt to read
  Windows hidden attributes or platform-specific metadata.
- Symlink tests must handle platforms where symlink creation requires elevated
  privileges by using existing pytest skip patterns when necessary.
- Symlink handling must cover symlinked files, symlinked directories, targets
  outside the workspace, and loop avoidance. `follow_symlinks=true` is allowed
  only after resolved targets are proven to stay within the workspace root.
- File mode metadata is best-effort. Tests should assert stable fields such as
  path, type, and size, not exact platform-specific permission masks.

## Governed Shell Facade Follow-Up

The backend-lead note argues for a single CLI-shaped surface because models are
good at composing terminal-like commands. This project already has that shape
through `run(command=...)`: a parser/executor/presentation runtime that keeps
pipe execution raw, applies truncation and binary guards only at presentation
time, and routes backend actions through approved MCP tools.

The safe follow-up is to improve that governed runtime rather than add real
shell execution:

- Keep `run` as the canonical tool name.
- Optionally expose `bash` and/or `shell` as compatibility aliases that call
  the same `RunCommandModule` implementation and clearly describe themselves
  as governed, shell-like facades rather than host shells.
- Add command aliases only after the typed tools exist, for example
  `stat <path>` -> `fs.stat`, `glob <pattern> [base]` or
  `find <pattern> [base]` -> `fs.glob`, and `rg <pattern> [base]` or
  `grep-files <pattern> [base]` -> `fs.grep`.
- Do not overload the existing pure `grep` command in this slice. It currently
  filters stdin in pipelines. File-backed grep needs either a distinct command
  name such as `rg`/`grep-files` or a later hybrid-command refactor that can
  distinguish stdin transforms from backend filesystem searches during
  preflight.
- Keep command visibility policy-driven: aliases appear only when their backing
  MCP tools are executable for the active profile.
- Keep `--help`, usage errors, stderr, exit-code footer, spill handling, and
  binary/truncation presentation behavior consistent with the existing command
  runtime.

## Profile Metadata Changes

The default profile metadata should treat `fs.stat`, `fs.glob`, and `fs.grep`
as read-only filesystem tools:

- Add them to `_FILES_READ_TOOLS`.
- Existing profiles that already include `_FILES_READ_TOOLS` inherit them.
- Tool descriptors use `filesystem.read`, `uses_filesystem`,
  `path_boundable`, and path argument hints.
- `fs.grep` and `fs.glob` belong to the `files` or `retrieval` category
  consistently with existing filesystem tools.

## Safety And Limits

- Every new operation resolves the workspace root before touching the
  filesystem.
- Every candidate path must remain under the resolved workspace root.
- Directory walks must have configurable caps to avoid unbounded traversal.
- Regex mode must cap pattern length and return actionable compile errors.
- Grep must not read files larger than its configured limit.
- Binary and undecodable files are skipped, not returned partially.
- Unknown arguments are rejected by both module validation and protocol
  schema validation.

## Testing Requirements

- Tool descriptor tests for all new tools, metadata, required arguments, and
  `additionalProperties: false`.
- Unit tests for `fs.stat` on files, directories, missing paths, and symlinks.
- Unit tests for `fs.glob` deterministic sorting, limits, hidden handling,
  case sensitivity, pattern validation, traversal caps, and workspace escape
  rejection.
- Unit tests for `fs.grep` literal search, regex search, UTF-8 handling,
  binary/decode skips, file-size skips, regex pattern guards, result limits,
  traversal caps, and line numbering with mixed newline styles.
- Symlink regression tests for outside-workspace targets and symlinked
  directory traversal behavior.
- Protocol validation tests for unknown arguments.
- Preset metadata test proving `_FILES_READ_TOOLS` exposes the new helpers.
- Follow-up command-runtime tests should cover policy-filtered visibility and
  help text for `stat`, `glob`/`find`, `rg`/`grep-files`, and any `bash`/`shell`
  alias once that follow-up is implemented.
- Cross-platform tests should avoid exact permission masks and skip symlink
  creation only when the OS denies it.

## Rollout

The implementation should land as one reviewable branch with small commits:

1. Tool schema and validation tests.
2. `fs.stat`.
3. `fs.glob`.
4. `fs.grep`.
5. Preset metadata and docs.
6. Final verification and security scan.
