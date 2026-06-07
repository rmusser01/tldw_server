# MCP Safe File Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Implement workspace-scoped `fs.read`, `fs.patch`, and `fs.write` with action-aware path grants, read receipts, module-derived path enforcement, redacted observability metadata, and legacy compatibility.

**Architecture:** Extend the existing filesystem module instead of replacing it, but split patch parsing and read-receipt logic into focused helpers so `filesystem_module.py` does not absorb all of the complexity. The protocol asks modules for derived path candidates only when a tool declares that requirement, and the hub path enforcer evaluates those candidates against action-aware `path_grants` while preserving the existing `path_allowlist_prefixes` fallback.

**Tech Stack:** Python 3.10+, FastAPI-era MCP Unified runtime, Pydantic-compatible dict contracts, stdlib `pathlib`/`hashlib`/`hmac`, pytest, Bandit.

---

## Context

Primary spec: `Docs/superpowers/specs/2026-06-07-mcp-fs-patch-write-safe-edit-tools-design.md`

Backlog task: `TASK-2297`

Existing filesystem module: `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`

Existing path enforcement service: `tldw_Server_API/app/services/mcp_hub_path_enforcement_service.py`

Existing protocol path-scope call site: `tldw_Server_API/app/core/MCP_unified/protocol.py`

## File Structure

- Create `mcp_unified/interfaces/path_scope.py`
  - Owns `PathScopeAction`, `PathScopeCandidate`, and helper normalization for module-derived path candidates.
- Modify `mcp_unified/interfaces/policy.py`
  - Adds the optional `path_scope_candidates` parameter to the `PathScopeEnforcer` protocol.
- Modify `tldw_Server_API/app/core/MCP_unified/modules/base.py`
  - Adds optional `extract_path_scope_candidates(...)` with a fail-closed default.
- Modify `tldw_Server_API/app/core/MCP_unified/protocol.py`
  - Calls module candidate extraction for tools declaring `metadata.path_scope_candidate_source == "module"` and passes candidates to the path enforcer.
- Modify `tldw_Server_API/app/services/mcp_hub_path_enforcement_service.py`
  - Supports action-aware `policy_document.path_grants` and derived candidates while retaining existing allowlist behavior.
- Create `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_diff.py`
  - Parses unified diffs and plans in-memory patch application. No filesystem IO.
- Create `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_receipts.py`
  - Creates and validates short-lived HMAC read receipts. No file content in receipts.
- Modify `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`
  - Adds descriptors, validation, candidate extraction, and execution for `fs.read`, `fs.patch`, and `fs.write`.
- Modify `mcp_unified/profiles/presets.py`
  - Adds canonical file-tool buckets while keeping legacy tools explicit.
- Modify `mcp_unified/USER_GUIDE.md`
  - Documents canonical file tools, read-before-mutate flow, and path grants.
- Test files:
  - `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py`
  - `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_patch_parser.py`
  - `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_path_scope_candidates.py`
  - `tldw_Server_API/tests/MCP_unified/test_mcp_hub_path_enforcement_service.py`
  - `tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py`

## Behavioral Contracts

Canonical tools:

- `fs.read` reads bounded UTF-8 text and returns content, byte counts, line counts, truncation state, newline style, SHA-256 when available, and a short-lived read receipt for complete hashed reads.
- `fs.patch` is the preferred existing-file edit primitive. It accepts unified diff text, derives affected paths from the diff, applies patches in memory first, then writes only after path scope and preimage checks pass.
- `fs.write` creates or replaces a whole file. `mode="create"` fails if the file exists. `mode="replace"` requires `expected_sha256` or a valid read receipt for the same path and preimage.
- `fs.read_text` and `fs.write_text` keep their current behavior but declare `path_scope_action` and legacy metadata.

Path grants:

```json
{
  "path_scope_mode": "enforce",
  "path_grants": [
    {"path": "documents", "actions": ["read", "edit", "write"]},
    {"path": "documents/private", "actions": ["edit", "write"], "effect": "deny"},
    {"path": "downloads", "actions": ["read"]}
  ]
}
```

`effect` is optional and defaults to `allow`. Deny grants take precedence over allow grants for the same path/action. The runtime enforcer should consume a flat effective grant list; future hierarchical authoring can compile into that flat list.

Reserved future path actions are `delete`, `rename`, `move`, `share`, `export`, `chmod`, `admin`, and `lock`. The first implementation enforces only `read`, `edit`, and `write`, but these names should be rejected as first-slice executable actions instead of being folded into `write`.

Candidate shape:

```python
from dataclasses import dataclass
from typing import Literal

PathScopeAction = Literal["read", "edit", "write"]

@dataclass(frozen=True)
class PathScopeCandidate:
    path: str
    action: PathScopeAction
    source: str
    display_path: str | None = None
    requires_existing_file: bool = False
    creates_file: bool = False
    workspace_id: str | None = None
```

`fs.patch` must declare:

```python
metadata={
    "category": "filesystem",
    "readOnlyHint": False,
    "write_capable": True,
    "path_scope_candidate_source": "module",
}
```

If a tool declares `path_scope_candidate_source: "module"` and candidates cannot be extracted, the request fails before execution with reason code `path_scope_candidates_unavailable`.

## Local Plan Review Pass

The writing-plans skill normally asks for a separate plan-review subagent. This thread does not have explicit user approval to spawn subagents, so this pass records the local review items incorporated before implementation:

- The protocol seam must remain compatible with old test fakes and host enforcers. It may fall back to the old enforcer call shape only when no derived candidates are required; if derived candidates exist and the enforcer cannot accept them, fail closed.
- If `path_grants` is present, malformed or non-matching grants must not fall through to broader legacy allowlists. Legacy `path_allowlist_prefixes` is a fallback only when `path_grants` is absent.
- Deny grants should be supported in the first implementation so private subtrees can override broader writable parent grants.
- Path-enforcement results should expose safe structured decision metadata now, because permission previews, simulations, audit events, and troubleshooting can build on the same contract later.
- Denial payloads should be actionable but redacted: reason code, requested action, normalized workspace-relative path, required grant class, matched grant effect/source when available, and no absolute host path.
- Read receipts must bind to normalized workspace-relative paths, not raw user input or absolute filesystem paths.
- Patch and write flows must check configured byte limits before loading or writing content, including existing-file preimages.
- Tests should use explicit receipt secrets so receipt assertions are deterministic and do not depend on process-global random state.

## Task 1: Add The Derived Path-Candidate Interface

**Files:**
- Create: `mcp_unified/interfaces/path_scope.py`
- Modify: `mcp_unified/interfaces/policy.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/base.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_path_scope_candidates.py`

- [x] **Step 1: Write failing interface tests**

Add protocol-focused tests that assert a module marked with `path_scope_candidate_source: "module"` is asked for candidates before path enforcement, and that missing candidates fail closed.

```python
async def test_protocol_passes_module_path_candidates_to_enforcer() -> None:
    module = _CandidateModule(
        candidates=[PathScopeCandidate(path="src/app.py", action="edit", source="module")]
    )
    enforcer = _RecordingPathEnforcer()
    protocol = _protocol_with(module=module, path_scope_enforcer=enforcer)

    await protocol.prepare_tool_call("fs.patch", {"diff": _PATCH}, _context())

    assert enforcer.received_candidates == [
        PathScopeCandidate(path="src/app.py", action="edit", source="module")
    ]


async def test_protocol_fails_closed_when_module_candidates_unavailable() -> None:
    module = _NoCandidateModule()
    protocol = _protocol_with(module=module, path_scope_enforcer=_RecordingPathEnforcer())

    with pytest.raises(PermissionError, match="path_scope_candidates_unavailable"):
        await protocol.prepare_tool_call("fs.patch", {"diff": _PATCH}, _context())
```

- [x] **Step 2: Run tests to verify failure**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_protocol_path_scope_candidates.py -q`

Expected: FAIL because `PathScopeCandidate` and the protocol extraction seam do not exist.

- [x] **Step 3: Add the interface model**

Create `mcp_unified/interfaces/path_scope.py` with `PathScopeAction`, `PathScopeCandidate`, `normalize_path_scope_candidate(raw)`, and `normalize_path_scope_candidates(raw_items)`.

Implementation requirements:

- Accept already-instantiated `PathScopeCandidate` objects.
- Accept dicts with `path`, `action`, `source`, `display_path`, `requires_existing_file`, `creates_file`, and `workspace_id`.
- Reject blank paths and actions outside `read`, `edit`, `write`.
- Keep paths as caller-provided strings; path resolution remains inside the enforcer.

- [x] **Step 4: Extend the protocols and base module**

Modify `mcp_unified/interfaces/policy.py` so `PathScopeEnforcer.evaluate_tool_call(...)` accepts:

```python
path_scope_candidates: list[PathScopeCandidate] | None = None
```

Modify `BaseModule` with:

```python
async def extract_path_scope_candidates(
    self,
    tool_name: str,
    arguments: dict[str, Any],
    context: Optional[Any] = None,
) -> list[PathScopeCandidate]:
    raise NotImplementedError(f"Path scope candidate extraction not implemented for {tool_name}")
```

- [x] **Step 5: Wire protocol extraction**

In `MCPProtocol.prepare_tool_call(...)`, after schema validation and before `_evaluate_path_scope(...)`, detect `tool_def["metadata"]["path_scope_candidate_source"] == "module"`. Call `module.extract_path_scope_candidates(...)`, normalize results, and pass the candidates into `_evaluate_path_scope(...)`.

Keep the seam inert for existing tools that do not declare this metadata.

For compatibility with old enforcer test doubles and host adapters:

- If no derived candidates are required and the enforcer rejects the new keyword shape with an unexpected-keyword `TypeError`, call the old signature.
- If derived candidates are required and the enforcer cannot accept them, raise `PermissionError("path_scope_candidates_unsupported")`.
- Do not swallow `TypeError` raised from inside the enforcer body; only fallback on an unexpected `path_scope_candidates` keyword error.

- [x] **Step 6: Run focused tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_protocol_path_scope_candidates.py -q`

Expected: PASS.

- [x] **Step 7: Commit**

Run:

```bash
git add mcp_unified/interfaces/path_scope.py mcp_unified/interfaces/policy.py tldw_Server_API/app/core/MCP_unified/modules/base.py tldw_Server_API/app/core/MCP_unified/protocol.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_path_scope_candidates.py
git commit -m "feat: add mcp path scope candidate seam"
```

## Task 2: Implement Action-Aware Path Grants

**Files:**
- Modify: `tldw_Server_API/app/services/mcp_hub_path_enforcement_service.py`
- Test: `tldw_Server_API/tests/MCP_unified/test_mcp_hub_path_enforcement_service.py`

- [x] **Step 1: Write failing grant tests**

Add tests for three profiles over one temporary workspace:

- Profile A can `read`, `edit`, and `write` under `documents/` but is denied under `downloads/`.
- Profile B can `read`, `edit`, and `write` under both directories.
- Profile C can `read` under `downloads/` but is denied for `edit` and `write`.

Also add regression coverage that `path_allowlist_prefixes` still authorizes legacy tools without `path_grants`.

Add deny precedence coverage:

- `documents/**` is writable through an allow grant.
- `documents/private/**` has a deny grant for `edit` and `write`.
- `fs.read` for `documents/private/notes.md` remains allowed when read is not denied.
- `fs.patch` or `fs.write` for `documents/private/notes.md` is denied with a safe reason payload.

Add decision-metadata coverage:

```python
result = await service.evaluate_tool_call(...)
assert result["allowed"] is False
assert result["reason_code"] == "path_action_denied"
assert result["path_decisions"][0] == {
    "requested_action": "write",
    "normalized_path": "documents/private/notes.md",
    "grant_outcome": "denied",
    "grant_source": "path_grants",
    "matched_grant_prefix": "documents/private",
    "matched_grant_effect": "deny",
    "redacted": True,
}
assert "/Users/" not in repr(result)
```

- [x] **Step 2: Run tests to verify failure**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/MCP_unified/test_mcp_hub_path_enforcement_service.py -q`

Expected: FAIL on missing `path_grants` behavior.

- [x] **Step 3: Add grant normalization helpers**

Implement helpers near `_policy_allowlist_prefixes(...)`:

```python
_PATH_GRANT_ACTIONS = {"read", "edit", "write"}
_RESERVED_PATH_GRANT_ACTIONS = {"delete", "rename", "move", "share", "export", "chmod", "admin", "lock"}

def _policy_path_grants(policy_document: Mapping[str, Any]) -> list[dict[str, Any]]:
    ...

def _candidate_action(tool_def: Mapping[str, Any], candidate: PathScopeCandidate | None) -> str:
    ...

def _path_permission_decision(...) -> dict[str, Any]:
    ...
```

Accepted grant forms:

- `{"path": "documents", "actions": ["read", "edit"]}`
- `{"prefix": "documents", "actions": ["read"]}` for compatibility with naming variants
- `{"prefix": "documents/private", "actions": ["edit", "write"], "effect": "deny"}`

Reject invalid grant entries by ignoring them, not by broadening access. If a grant uses a reserved future action, ignore that action for first-slice enforcement and return a safe invalid-policy decision only when the request depends on that unsupported action.

- [x] **Step 4: Evaluate derived candidates first**

If `path_scope_candidates` is non-empty, evaluate only those candidates instead of extracting paths from `path_argument_hints`. Each candidate must match a grant for its own action.

For non-derived paths, infer the action from `tool_def.metadata.path_scope_action`, defaulting to `write` for write-capable tools and `read` for read-only tools.

Decision behavior:

- Normalize all checked paths to workspace-relative display paths before returning metadata.
- Never return absolute host paths.
- If any matching deny grant applies, deny with `reason_code="path_action_denied"`.
- If no allow grant applies and no deny grant applies, deny with `reason_code="path_action_not_granted"`.
- Include `path_decisions` in the service result for both allowed and denied requests.

- [x] **Step 5: Preserve current fallback behavior**

When `path_grants` is absent, keep the current `path_allowlist_prefixes` and multi-root behavior unchanged. This is the compatibility path for existing profiles and tests.

When `path_grants` is present, do not fall back to `path_allowlist_prefixes` for candidates that fail to match an action grant. This prevents a profile from accidentally broadening action-specific policy by carrying an older allowlist field.

- [x] **Step 6: Run focused tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/MCP_unified/test_mcp_hub_path_enforcement_service.py -q`

Expected: PASS.

- [x] **Step 7: Commit**

Run:

```bash
git add tldw_Server_API/app/services/mcp_hub_path_enforcement_service.py tldw_Server_API/tests/MCP_unified/test_mcp_hub_path_enforcement_service.py
git commit -m "feat: enforce mcp action aware path grants"
```

## Task 3: Add Read Receipts And `fs.read`

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_receipts.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py`

- [x] **Step 1: Write failing `fs.read` tests**

Cover:

- Tool catalog contains `fs.read` with `readOnlyHint=True`, `path_scope_action="read"`, eval metadata, and strict schema.
- Complete text read returns `content`, `sha256`, `read_receipt`, `bytes_read`, `bytes_total`, `line_count`, and `newline_style`.
- Truncated reads set `truncated=True`, include a truncation reason, and omit `read_receipt`.
- Binary/NUL, non-UTF-8, symlink, directory, and outside-workspace paths are rejected.

- [x] **Step 2: Run tests to verify failure**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py -q`

Expected: FAIL because `fs.read` is not registered.

- [x] **Step 3: Implement `filesystem_receipts.py`**

Use stateless HMAC receipts:

- Payload fields: `v`, `path`, `sha256`, `size`, `issued_at`, `expires_at`, optional `workspace_id`, optional safe `session_id`.
- Signature: `hmac.new(secret, canonical_json, hashlib.sha256).hexdigest()`.
- Encoding: URL-safe base64 of `canonical_json.signature`.
- Default TTL: 1,800 seconds.
- Secret source: module config key `read_receipt_secret`; if absent, use a per-process random secret and document that receipts do not survive restart.
- Receipt `path` must be the normalized workspace-relative path returned by `_to_workspace_relative_path(...)`. Do not store raw user input or an absolute host path in the receipt payload.
- Unit tests should configure a fixed `read_receipt_secret` through the module config to avoid nondeterminism.

Expose:

```python
class ReadReceiptError(ValueError):
    reason_code: str


class ReadReceiptManager:
    def issue(...)
    def validate(...)
```

- [x] **Step 4: Implement `fs.read`**

In `filesystem_module.py`:

- Add `_TOOL_NAME_READ = "fs.read"` if tool constants exist, otherwise keep local string constants.
- Add descriptor in `get_tools()`.
- Add validation for `path`, optional `start_line`, optional `end_line`, optional `max_bytes`, and optional `include_receipt`.
- Use existing workspace-root resolution helpers.
- Read bytes with a hard cap, reject NUL and decode failures, return bounded UTF-8 text.
- Compute full-file SHA-256 only when the full file is within the configured hash limit.
- Return a receipt only when the full hash is available and the read is not truncated.
- Use the normalized workspace-relative path in the response and receipt, and keep absolute paths out of all returned metadata.

- [x] **Step 5: Add eval metadata to results**

Return an `eval` map using `build_execution_eval_metadata(...)`:

```python
"eval": {
    "tool_name": "fs.read",
    "action_family": "read",
    "result_kind": "bounded_text_file",
    "path_filter_used": True,
    "truncated": result["truncated"],
}
```

Do not include file content, absolute paths, or receipt contents in eval metadata.

- [x] **Step 6: Run focused tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py -q`

Expected: PASS.

- [x] **Step 7: Commit**

Run:

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_receipts.py tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py
git commit -m "feat: add mcp fs read tool"
```

## Task 4: Add Unified Diff Parser And In-Memory Planner

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_diff.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_patch_parser.py`

- [x] **Step 1: Write parser tests first**

Cover:

- Parses `--- a/path` / `+++ b/path` headers and hunk ranges.
- Handles modify, create (`--- /dev/null`), and delete (`+++ /dev/null`) headers, but delete is rejected for this first implementation.
- Rejects absolute paths, drive-qualified paths, `..` traversal, empty paths, and diffs exceeding configured file/hunk limits.
- Applies a simple modify patch in memory and returns the exact expected text.
- Detects context mismatch with a stable reason code.

- [x] **Step 2: Run tests to verify failure**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_patch_parser.py -q`

Expected: FAIL because the parser file does not exist.

- [x] **Step 3: Implement parser dataclasses**

Create:

```python
@dataclass(frozen=True)
class PatchHunkLine:
    kind: Literal["context", "add", "remove"]
    text: str

@dataclass(frozen=True)
class PatchHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: tuple[PatchHunkLine, ...]

@dataclass(frozen=True)
class PatchFile:
    old_path: str | None
    new_path: str | None
    action: Literal["modify", "create"]
    hunks: tuple[PatchHunk, ...]
```

- [x] **Step 4: Implement parser and planner**

Expose:

```python
def parse_unified_diff(diff_text: str, *, max_files: int, max_hunks: int, max_bytes: int) -> tuple[PatchFile, ...]:
    ...

def apply_patch_to_text(original: str, patch_file: PatchFile) -> str:
    ...
```

Keep all path normalization lexical and portable; workspace resolution still belongs to `FilesystemModule`.

- [x] **Step 5: Run parser tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_patch_parser.py -q`

Expected: PASS.

- [x] **Step 6: Commit**

Run:

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_diff.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_patch_parser.py
git commit -m "feat: add mcp filesystem unified diff parser"
```

## Task 5: Add `fs.patch`

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_diff.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_path_scope_candidates.py`

- [x] **Step 1: Write failing `fs.patch` tests**

Cover:

- Catalog descriptor includes `path_scope_candidate_source="module"` and write metadata.
- Candidate extraction returns `edit` for existing-file modification and `write` for create.
- Candidate paths are normalized to workspace-relative paths before read-receipt validation and permission-decision metadata.
- Existing-file patch in strict mode requires `expected_sha256_by_path` or `read_receipt_by_path`.
- Valid expected hash applies the patch and returns metadata without content.
- Valid read receipt applies the patch.
- Stale hash, stale receipt, mismatched receipt path, context mismatch, symlink, and outside-workspace paths are rejected.
- Dry-run returns the planned changes and does not write.

- [x] **Step 2: Run tests to verify failure**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_path_scope_candidates.py -q`

Expected: FAIL because `fs.patch` is not implemented.

- [x] **Step 3: Implement candidate extraction**

Override `FilesystemModule.extract_path_scope_candidates(...)`.

For `fs.patch`:

- Parse the diff.
- Return one candidate per affected file.
- `modify`: `action="edit"`, `requires_existing_file=True`, `creates_file=False`.
- `create`: `action="write"`, `requires_existing_file=False`, `creates_file=True`.

For other tools, delegate to the base default only if they declare module-derived candidates.

- [x] **Step 4: Implement execution**

Implementation order inside `execute_tool("fs.patch", ...)`:

1. Resolve workspace root.
2. Parse diff.
3. Resolve each patch file path with `_resolve_workspace_path_no_follow(...)`.
4. Reject symlinks and unsupported file types.
5. Check existing-file size against the configured patch preimage byte limit before reading.
6. Read existing files and validate preimage hash or receipt before mutation.
7. Apply hunks in memory.
8. Check resulting content size against the configured write byte limit.
9. If `dry_run=True`, return plan metadata only.
10. Write via temp file in the target directory, then atomic replace.
11. Return per-file metadata: path, action, created, bytes_before, bytes_after, sha256_before, sha256_after.

- [x] **Step 5: Add rollback handling**

Track parent directories created for new files. If a later file write fails, remove only empty directories created by this call and never remove pre-existing directories.

- [x] **Step 6: Run focused tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_path_scope_candidates.py -q`

Expected: PASS.

- [x] **Step 7: Commit**

Run:

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_diff.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_path_scope_candidates.py
git commit -m "feat: add mcp fs patch tool"
```

## Task 6: Add `fs.write`

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py`

- [x] **Step 1: Write failing `fs.write` tests**

Cover:

- Catalog descriptor includes `path_scope_action="write"` and write metadata.
- `mode="create"` creates a new UTF-8 text file and fails when the file already exists.
- `mode="replace"` requires `expected_sha256` or `read_receipt`.
- `mode="replace"` rejects stale hashes and mismatched receipts.
- Path escape, symlink, directory, binary/NUL content, and oversized content are rejected.
- Dry-run returns metadata and does not write.

- [x] **Step 2: Run tests to verify failure**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py -q`

Expected: FAIL because `fs.write` is not implemented.

- [x] **Step 3: Implement descriptor and validation**

Add input schema fields:

- `path` string, required.
- `content` string, required.
- `mode` enum `["create", "replace"]`, required.
- `expected_sha256` string, optional.
- `read_receipt` string, optional.
- `dry_run` boolean, optional default `False`.

- [x] **Step 4: Implement execution**

Use the same workspace and no-follow path helpers as `fs.patch`.

For replace:

- Reject missing target.
- Reject symlinks.
- Compute current SHA-256.
- Accept either matching `expected_sha256` or valid receipt for the same path/hash.

For create:

- Reject if target exists or is a symlink.
- Create parents only after path checks pass.

For both:

- Write through a temp file in the target directory and atomic replace.
- Return metadata only, including `sha256_after`.

- [x] **Step 5: Run focused tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py -q`

Expected: PASS.

- [x] **Step 6: Commit**

Run:

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py
git commit -m "feat: add mcp fs write tool"
```

## Task 7: Update Profiles, Legacy Metadata, And Reporting Safety

**Files:**
- Modify: `mcp_unified/profiles/presets.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py`

- [x] **Step 1: Write failing metadata/profile tests**

Cover:

- `_FILES_READ_TOOLS` includes `fs.read` and no longer relies only on `fs.read_text`.
- `_FILES_EDIT_TOOLS` includes `fs.patch`.
- `_FILES_WRITE_TOOLS` includes `fs.write`.
- Legacy buckets keep `fs.read_text` and `fs.write_text` explicit.
- `fs.read_text` declares `path_scope_action="read"` and legacy metadata.
- `fs.write_text` declares `path_scope_action="write"` and legacy metadata.
- Tool-use reporting for filesystem tools records metadata fields only and never records returned file content, raw diffs, absolute paths, or receipts.
- Permission decision metadata is allowed in tool-use reporting only through safe scalar fields or redacted relative path summaries.

- [x] **Step 2: Run tests to verify failure**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py -q`

Expected: FAIL on missing profile buckets or metadata.

- [x] **Step 3: Update profile buckets conservatively**

Implement the spec buckets:

```python
_FILES_READ_TOOLS = ["fs.list", "fs.read", "fs.stat", "fs.glob", "fs.grep"]
_FILES_EDIT_TOOLS = [*_FILES_READ_TOOLS, "fs.patch"]
_FILES_WRITE_TOOLS = [*_FILES_EDIT_TOOLS, "fs.write"]
_LEGACY_FILES_READ_TOOLS = ["fs.read_text"]
_LEGACY_FILES_WRITE_TOOLS = ["fs.write_text"]
```

Do not grant file tools to any profile that currently lacks equivalent file access.

- [x] **Step 4: Add legacy metadata**

`fs.read_text` metadata should include:

```python
{"legacy_tool": True, "replacement_tool": "fs.read", "path_scope_action": "read"}
```

`fs.write_text` metadata should include:

```python
{"legacy_tool": True, "replacement_tools": ["fs.patch", "fs.write"], "path_scope_action": "write"}
```

- [x] **Step 5: Verify reporting remains metadata-only**

If the current `ToolUseEvent` model already prevents content capture, add regression tests rather than expanding the model. If filesystem result metadata needs a nested `mcp_tool_use` envelope for gateway propagation, include only allowlisted scalar fields from the spec.

- [x] **Step 6: Run focused tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py -q`

Expected: PASS.

- [x] **Step 7: Commit**

Run:

```bash
git add mcp_unified/profiles/presets.py tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py
git commit -m "feat: wire mcp safe file tools into profiles"
```

## Task 8: Update Documentation And Run Full Validation

**Files:**
- Modify: `mcp_unified/USER_GUIDE.md`
- Modify: `backlog/tasks/task-2297 - Implement-MCP-safe-file-tools.md`

- [x] **Step 1: Document the user workflow**

Add a section covering:

- `fs.read` before `fs.patch` or `fs.write replace`.
- How `expected_sha256` and read receipts protect against stale edits.
- Why `fs.patch` is preferred over raw whole-file writes.
- Example `path_grants` for read-only, edit-only, and write-enabled profiles.
- Example deny override for a private subtree under a broader writable parent.
- Safe permission decision payloads and denial reason codes.
- Compatibility status of `fs.read_text` and `fs.write_text`.

- [x] **Step 2: Run targeted tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_patch_parser.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_path_scope_candidates.py \
  tldw_Server_API/tests/MCP_unified/test_mcp_hub_path_enforcement_service.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py \
  -q
```

Expected: PASS.

- [x] **Step 3: Run broader MCP smoke tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_preexec_validation.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_write_tools_validators.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_tool_observability.py \
  -q
```

Expected: PASS.

- [x] **Step 4: Run Bandit on touched code**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_diff.py \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_receipts.py \
  tldw_Server_API/app/core/MCP_unified/protocol.py \
  tldw_Server_API/app/core/MCP_unified/modules/base.py \
  tldw_Server_API/app/services/mcp_hub_path_enforcement_service.py \
  mcp_unified/interfaces/path_scope.py \
  mcp_unified/interfaces/policy.py \
  mcp_unified/profiles/presets.py \
  -f json -o /tmp/bandit_mcp_safe_file_tools.json
```

Expected: PASS or only baseline findings outside changed lines. Fix any new touched-code findings before continuing.

- [x] **Step 5: Run diff hygiene**

Run: `git diff --check`

Expected: no whitespace errors.

- [x] **Step 6: Update Backlog task**

Record:

- Implementation summary.
- Test commands and outcomes.
- Bandit output path and outcome.
- Any known skips with reasons.

- [x] **Step 7: Final commit**

Run:

```bash
git add mcp_unified/USER_GUIDE.md "backlog/tasks/task-2297 - Implement-MCP-safe-file-tools.md"
git commit -m "docs: document mcp safe file tools"
```

## Risk Review

- **Path-scope bypass through patch paths:** `fs.patch` must never use caller-provided path hints. It must parse diff paths and pass module-derived candidates before execution.
- **Deny override bypass:** When `path_grants` is present, deny effects must take precedence over allow effects and must not fall through to `path_allowlist_prefixes`.
- **Symlink escapes:** Use existing no-follow resolution and reject symlink targets for mutating tools. Do not follow symlinks during write or patch.
- **Stale edits:** Strict mode requires a matching hash or read receipt before mutating existing files.
- **Receipt secrecy:** Receipts must not embed file content, absolute paths, or secrets. Sign canonical JSON with HMAC and compare signatures with `hmac.compare_digest`.
- **Trace leakage:** Tool-use reporting must stay metadata-only. Do not add content, raw diff, receipts, or absolute paths to eval metadata.
- **Profile broadening:** Add canonical tools only where equivalent legacy file authority already existed, unless the profile is explicitly intended to gain that capability.
- **Large-file behavior:** Avoid hashing or loading unbounded data. `fs.patch` and `fs.write` must enforce configured byte limits before mutation.

## Completion Criteria

- `fs.read`, `fs.patch`, and `fs.write` are in the filesystem catalog with schemas, metadata, and eval metadata.
- `fs.patch` uses module-derived path candidates and fails closed when candidate extraction is unavailable.
- `path_grants` enforce `read`, `edit`, and `write` separately while preserving legacy allowlist behavior.
- Deny path grants override broader allow grants and return safe denial metadata.
- Existing-file `fs.patch` and `fs.write replace` require `expected_sha256` or a valid read receipt.
- Legacy tools still work and are marked as compatibility tools.
- Profile presets prefer canonical file tools without accidentally expanding roles that had no file access.
- Targeted tests, MCP smoke tests, Bandit touched-scope scan, and `git diff --check` are recorded in `TASK-2297`.
