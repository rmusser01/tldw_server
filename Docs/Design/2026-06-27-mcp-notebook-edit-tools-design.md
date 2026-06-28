# MCP Notebook Edit Tools Design

## Context

`TASK-2282` adds notebook-safe MCP tools modeled after Claude Code `NotebookEdit`.
The current MCP filesystem module already provides the trust boundary this work
needs: workspace-root resolution, path-scope metadata, file-policy action
grants, preimage checks, read receipts, advisory lock leases, atomic writes, and
metadata-only tool reporting. Notebook tools should reuse that boundary instead
of introducing a parallel filesystem policy path.

Baseline note: before this design was written, the existing filesystem module
test file passed 103 tests and failed
`test_filesystem_glob_marks_file_size_unavailable` with
`OSError("metadata unavailable")`. The notebook work will avoid touching the
glob path and will record this as pre-existing baseline risk unless it becomes
directly relevant.

## Goals

- Let MCP clients inspect Jupyter notebook structure without reading or writing
  the whole notebook body by default.
- Let MCP clients replace, insert, and delete notebook cells by stable Jupyter
  cell `id`.
- Require normal MCP filesystem path grants and preimage protection before cell
  mutation.
- Preserve JSON validity and minimize avoidable notebook churn.
- Return bounded, audit-friendly summaries without leaking raw notebook content
  into telemetry, errors, or eval metadata.

## Non-Goals

- No full notebook overwrite tool.
- No notebook execution, kernel management, or output generation.
- No automatic repair or normalization of notebooks with missing or duplicate
  cell ids.
- No broad dependency addition such as `nbformat`; this implementation will use
  the Python standard library and validate the invariants it depends on.

## Architecture

Add focused notebook helpers under
`tldw_Server_API/app/core/MCP_unified/modules/implementations/notebook_files.py`.
The existing `FilesystemModule` will own MCP tool registration and workspace
integration, then delegate parsing, validation, cell summarization, and JSON
mutation to that helper. This avoids expanding the already-large filesystem
module with notebook-specific mechanics while preserving a single policy and
execution surface for filesystem-backed MCP tools.

The helper should expose small pure functions or simple dataclasses for:

- parsing a notebook byte payload into validated notebook state;
- summarizing cell structure with bounded source previews;
- applying one cell operation in memory;
- serializing the edited notebook while preserving practical formatting traits
  such as trailing newline, indentation style, key order, and cell source
  string/list representation where possible.

## Tool Surface

### `notebook.read`

Reads one `.ipynb` file under the active trusted workspace root.

Arguments:

- `path` string, required.
- `include_source` boolean, default `false`.
- `cell_ids` string array, optional filter for source previews.
- `max_source_chars` positive integer, optional bounded source preview limit.
- `max_total_source_chars` positive integer, optional total preview budget.
- `include_receipt` boolean, default `true`.

Returns:

- workspace-relative `path`;
- `cell_count`;
- notebook `nbformat` and `nbformat_minor`;
- `sha256`, byte count, truncation state, and optional `read_receipt`;
- per-cell summaries: `id`, `cell_type`, index, source line count,
  source character count, execution/output counts for code cells, and source
  preview only when explicitly requested.

Default behavior is structure-first and source-sparing. Clients that need cell
content must request it intentionally, can scope the request to specific cell
ids, and remain bounded by both per-cell and total source preview budgets.

### `notebook.edit_cell`

Mutates one `.ipynb` cell by Jupyter cell `id`.

Arguments:

- `path` string, required.
- `cell_id` string, required for `replace`, `insert`, and `delete`.
- `mode` enum: `replace`, `insert`, `delete`.
- `insert_position` enum: `before`, `after`; required only for `insert`.
- `new_cell_id` string, optional for inserted cells and ignored otherwise.
- `cell_type` enum: `code`, `markdown`, `raw`; required for inserted cells and
  optional for replace.
- `source` string, required for replace and insert.
- `expected_sha256` string or `read_receipt` string, one required.
- `lock_lease_id` string, optional.
- `dry_run` boolean, default `false`.

Returns a bounded diff summary:

- workspace-relative `path`;
- `mode`, `edited`, and `dry_run`;
- affected cell ids and indexes;
- cell count before/after;
- source line and character counts before/after;
- output count before/after for code cells;
- `sha256_before` and `sha256_after`;
- `bytes_before`, `bytes_after`, and `bytes_written` when committed.

Cell deletion is a file-policy `edit`, not filesystem `delete`, because the
notebook file remains and the mutation is bounded to notebook structure.

## Policy And Permissions

Both tools use existing MCP file-policy metadata:

- `notebook.read`: `uses_filesystem=true`, `path_boundable=true`,
  `path_scope_action=read`, `file_policy_action=read`, `readOnlyHint=true`.
- `notebook.edit_cell`: `uses_filesystem=true`, `path_boundable=true`,
  `path_scope_action=edit`, `file_policy_action=edit`, `write_capable=true`,
  `readOnlyHint=false`.

Profiles still need the tool allowed and a path grant that covers the requested
path/action. `NotebookEdit(...)` already exists in permission-rule parsing as a
path-oriented rule family, so documentation should connect the new tool names to
that mental model without treating path rules as tool grants.

Preset updates should be conservative:

- add `notebook.read` only to file-read-oriented presets;
- add `notebook.edit_cell` only where bounded file edit tools such as
  `fs.patch` are already available;
- leave explicit allow-list profiles unchanged unless operators opt in.

## Validation And Error Recovery

The tools reject:

- non-`.ipynb` paths;
- symlinks that cannot be resolved inside the workspace;
- files exceeding configured notebook read/edit limits;
- invalid UTF-8 or invalid JSON;
- notebook JSON without a top-level `cells` list;
- cells without valid string `id` values;
- duplicate cell ids;
- missing target `cell_id`;
- unsupported `cell_type`, invalid source shape, or invalid mode arguments;
- stale `expected_sha256` or `read_receipt`;
- missing required lock lease when the filesystem module requires locks.

Do not synthesize ids for existing notebooks that lack ids. `notebook.read`
should report missing or duplicate id diagnostics, and `notebook.edit_cell`
should fail closed. Inserted cells may use caller-provided `new_cell_id` when
unique and valid; otherwise the tool generates a valid unique id.

Replacing a code cell source clears `outputs` and sets `execution_count` to
`null` by default. This avoids presenting stale execution results as if they
belong to the edited source. A future explicit `preserve_outputs` option can be
considered separately if there is a concrete trusted workflow.

Errors should use stable reason-code-style messages such as
`notebook_invalid_json`, `notebook_cell_id_not_found`,
`notebook_duplicate_cell_id`, `notebook_preimage_mismatch`, and
`notebook_write_too_large`. Error messages must not include raw cell source,
absolute host paths, or full notebook content.

## Serialization

The helper should use `json.loads`/`json.dumps` with insertion-order preservation
and no `sort_keys`. It should preserve a trailing newline when the input had one.
It should keep source representation compatible with the edited cell's original
shape where practical: if a replaced cell stored `source` as a list of lines,
write a list; if it stored a string, write a string. Inserted cells can use a
string source unless the surrounding notebook strongly indicates list-style
sources.

Whole-file byte output is acceptable internally because `.ipynb` is JSON, but
the public edit API must remain cell-scoped and must not accept a full notebook
body.

## Telemetry And Reporting

Tool results can include bounded source previews only when the user explicitly
requests them from `notebook.read`. Eval metadata, tool-use reporting, policy
explanations, logs, and exception messages must remain metadata-only. They may
include tool names, action family, result kind, truncation flags, reason codes,
workspace-relative paths after existing sanitization, and redacted path-policy
decisions. They must not include raw cell source, outputs, full diffs, read
receipt values, or absolute local paths.

## Tests

Use TDD. Add tests before implementation for:

- tool definitions and path-scope metadata;
- `notebook.read` default structure-only response;
- explicit bounded source previews and truncation;
- read receipts on full hashed reads;
- replace, insert before/after, and delete by cell id;
- dry-run mutation summaries without writes;
- preimage mismatch and receipt mismatch;
- lock-required behavior through existing lock settings;
- invalid JSON, non-notebook paths, missing cells list, missing ids, duplicate
  ids, missing target id, and unsupported cell types;
- code-cell output clearing after source replacement;
- path-scope extraction/enforcement metadata;
- preset inclusion for file read/edit profiles;
- docs examples that show both tool allow-list and path-grant requirements.

Run targeted notebook/filesystem tests, relevant profile preset tests, and
Bandit on touched Python files before completion.

## Acceptance Criteria

- `notebook.read` and `notebook.edit_cell` are visible through the MCP tool
  catalog with correct schemas, metadata, and read/write classification.
- `notebook.read` returns notebook structure by default and only returns bounded
  source previews when explicitly requested.
- `notebook.edit_cell` supports replace, insert, and delete by cell id with
  preimage checks, optional lock leases, dry-run behavior, and atomic writes.
- Notebook mutations preserve valid JSON and avoid raw whole-notebook overwrite
  inputs.
- Invalid notebooks and unsafe/stale edits fail closed with stable reason-code
  errors that do not leak raw content.
- File-policy path grants control notebook read/edit access using existing
  `read` and `edit` actions.
- Presets, docs, and tests reflect the conservative permission model.
- Verification results, baseline skips or failures, Bandit output, and final
  summary are recorded on `TASK-2282`.
