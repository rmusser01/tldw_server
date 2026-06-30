# MCP Filesystem Diff Parser V2 Design

## Goal

Improve the existing MCP filesystem patch path so `fs.patch` can be safely preferred over raw edit/write for common agentic file edits. This slice focuses on parser fidelity and deterministic patch application inside the already-governed filesystem module.

## Current State

`fs.read`, `fs.edit`, `fs.patch`, and `fs.write` already exist. `fs.patch` already:

- parses unified diffs into file and hunk plans;
- extracts path scope candidates before policy enforcement;
- requires hashes or read receipts for existing-file edits;
- supports dry runs, create gating, rollback-on-partial-write, and non-content result metadata.

The remaining gap is that the parser is too narrow for real model-produced diffs. It rejects or mishandles cases that should be deterministic and safe, especially paths with spaces and `\ No newline at end of file` markers.

## Scope

This slice will:

1. Preserve final-line no-newline markers during parse and apply.
2. Parse file header paths with spaces when unambiguous.
3. Accept git-style preamble lines before `---` / `+++` headers without treating them as patch content.
4. Keep unsafe paths rejected after normalization, including absolute paths, drive-qualified paths, empty segments, `.` segments, and `..` traversal.
5. Keep delete and rename application unsupported until separate path-policy verbs exist.
6. Add focused tests for parser behavior and end-to-end `fs.patch` application.

## Non-Goals

- No `fs.delete`, `fs.rename`, or `fs.move` execution in this slice.
- No fuzzy patch matching.
- No binary patch support.
- No raw shell or raw edit routing changes.
- No frontend changes.

## Design

`filesystem_diff.py` remains the parser and in-memory patch engine. It will extend `PatchHunkLine` with an EOF-newline flag so the parser can represent `\ No newline at end of file` without leaking file contents into result metadata.

Header path parsing will prefer tab-separated metadata when present. When no tab exists, it will preserve the full remaining path text instead of truncating at the first space. This supports common model output like `--- a/docs/my note.txt` and `+++ b/docs/my note.txt` while retaining existing normalization checks.

Unsupported operations remain explicit:

- delete patches return `delete_not_supported`;
- rename patches return `rename_not_supported`;
- mode-only patches without hunks return `invalid_diff`.

`FilesystemModule` continues to govern `fs.patch` through `extract_path_scope_candidates()`, `_resolve_workspace_path_no_follow()`, preimage checks, read receipts, hash checks, and atomic writes. This slice should not add a side path around those checks.

## Testing

Focused tests should cover:

- parsing paths with spaces in file headers;
- applying a patch that preserves a missing final newline;
- rejecting malformed orphan no-newline markers;
- `fs.patch` end-to-end write preserving no-final-newline content;
- existing parser/module tests continuing to pass.

Verification before PR closeout:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_patch_parser.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py \
  -q

source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_diff.py \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py \
  -f json -o /tmp/bandit_mcp_fs_diff_parser_v2.json

git diff --check
```
