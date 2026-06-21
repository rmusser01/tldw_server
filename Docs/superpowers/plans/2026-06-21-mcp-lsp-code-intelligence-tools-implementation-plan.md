# MCP LSP Code Intelligence Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Python-first LSP-backed MCP code intelligence tools using Ruff for diagnostics/edit previews and pylsp for semantic navigation.

**Architecture:** Put host-neutral LSP runtime, models, process management, and standalone gateway glue in `mcp_unified/lsp`. Add a thin tldw-hosted MCP `LSPModule` that registers `lsp.*` tools, delegates to the package runtime, and relies on existing profile/path-scope governance. Keep all edit-producing LSP operations preview-only; actual mutation remains behind `fs.patch`/`fs.write`.

**Tech Stack:** Python 3.10+, asyncio subprocesses, JSON-RPC over LSP `Content-Length` framing, Ruff `ruff server`, python-lsp-server `pylsp`, pytest/pytest-asyncio, Bandit.

---

## Source Spec

- `Docs/superpowers/specs/2026-06-20-mcp-lsp-code-intelligence-tools-design.md`
- Backlog task: `TASK-2281`

## File Structure

Create host-neutral package files:

- `mcp_unified/lsp/__init__.py`: public exports for embedders.
- `mcp_unified/lsp/errors.py`: structured exceptions and reason codes.
- `mcp_unified/lsp/models.py`: request/result dataclasses or Pydantic models for positions, ranges, diagnostics, symbols, locations, preview edits, backend status, and tool schemas.
- `mcp_unified/lsp/config.py`: `LspRuntimeConfig` limits, timeouts, idle TTL, backend command paths, and env/settings parsing helpers.
- `mcp_unified/lsp/executables.py`: safe executable discovery from explicit config, project virtualenv, then PATH; returns argv arrays and provenance.
- `mcp_unified/lsp/jsonrpc.py`: async LSP Content-Length JSON-RPC client over stdin/stdout, request ids, notifications, timeout handling, and bounded stderr metadata.
- `mcp_unified/lsp/backends.py`: `LspBackend` protocol, capability constants, backend status contract, and fake backend support for tests.
- `mcp_unified/lsp/sessions.py`: per-workspace `LspSessionManager`, session cache, idle shutdown, degraded status, and stop-all behavior.
- `mcp_unified/lsp/router.py`: capability router mapping tools to Ruff/pylsp and enforcing stable errors.
- `mcp_unified/lsp/filtering.py`: profile/request-scoped result filtering helpers for locations, diagnostics, symbols, and preview affected paths.
- `mcp_unified/lsp/ruff.py`: Ruff backend for diagnostics, format preview, and code-action preview.
- `mcp_unified/lsp/pylsp.py`: pylsp backend for symbols, definition, references, hover, and signature help.
- `mcp_unified/lsp/service.py`: high-level `LspCodeIntelligenceService` consumed by both tldw module and standalone gateway runtime.
- `mcp_unified/lsp/gateway_runtime.py`: minimal standalone `GatewayRuntime` exposing only `lsp.*` tools for smoke/UAT.

Modify package metadata:

- `mcp_unified/pyproject.toml`: add package `mcp_unified.lsp` and optional extra `lsp = ["ruff>=0.13", "python-lsp-server>=1.14"]`. If pinned minimums conflict with repository dependency policy, use the lowest versions proven by local/CI tests.

Create tldw-hosted module files:

- `tldw_Server_API/app/core/MCP_unified/modules/implementations/lsp_module.py`: `BaseModule` adapter, tool definitions, argument validation, path-scope candidate extraction, result normalization passthrough.

Modify tldw registration:

- `tldw_Server_API/app/core/MCP_unified/server.py`: add optional `MCP_ENABLE_LSP_MODULE` registration block, disabled by default.

Modify profiles and docs:

- `mcp_unified/profiles/presets.py`: add `_LSP_TOOLS`, code-intelligence capability metadata, and include `lsp.*` tooling for code-oriented presets.
- `tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py`: add preset assertions for LSP-capable roles.
- `mcp_unified/README.md`: add optional LSP extra summary.
- `mcp_unified/USER_GUIDE.md`: add install, enable, and safety notes.
- `Docs/MCP/Unified/Smoke_Client.md`: document LSP smoke scenario flags.

Create tests:

- `tldw_Server_API/app/core/MCP_unified/tests/test_lsp_models.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_lsp_router.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_lsp_filtering.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_lsp_jsonrpc.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_lsp_sessions.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_lsp_backends_fake.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_lsp_real_backends.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_lsp_module.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_lsp_gateway_runtime.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_lsp_smoke_scenario.py`
- `tldw_Server_API/app/core/MCP_unified/tests/fixtures/fake_lsp_stdio_server.py`

## Implementation Rules

- Use TDD: write the failing test first for each behavior, run the targeted test, then implement.
- Do not add direct file mutation to LSP tools.
- Do not accept executable command strings from model/tool input.
- LSP protocol framing is `Content-Length` JSON-RPC, not line-delimited JSON.
- Do not expose raw absolute paths, raw stderr, raw environment values, or file contents in errors/metrics.
- Every file-scoped tldw module tool must produce `PathScopeCandidate(action="read")`.
- `lsp.workspace_symbols` must require a workspace-root read grant in the first slice.
- Returned LSP paths must be filtered through the current request/profile path policy. In tldw-hosted MCP this is enforced by protocol path-scope preflight plus module-level post-result filtering. In standalone runtime, callers may provide an optional path-allow predicate; the default standalone LSP-only runtime allows all paths under its configured workspace and must document that embedders with profile policy should wrap it with their own profile/path runtime.
- Real backend tests must be env-gated and skipped by default when Ruff/pylsp are unavailable.

---

### Task 1: Package Metadata, Models, Errors, And Limits

**Status:** Complete. Local verification on 2026-06-21: `test_lsp_models.py` passed with 40 tests, Bandit reported zero findings for `mcp_unified/lsp`, and `git diff --check` was clean.

**Files:**
- Create: `mcp_unified/lsp/__init__.py`
- Create: `mcp_unified/lsp/errors.py`
- Create: `mcp_unified/lsp/models.py`
- Create: `mcp_unified/lsp/config.py`
- Modify: `mcp_unified/pyproject.toml`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_lsp_models.py`

- [x] **Step 1: Write failing tests for the public model contract**

Add tests covering:

```python
def test_lsp_position_is_zero_based_utf16_contract():
    position = LspPosition(line=0, character=4)
    assert position.line == 0
    assert position.character == 4

def test_lsp_position_rejects_negative_offsets():
    with pytest.raises(ValueError, match="line"):
        LspPosition(line=-1, character=0)

def test_lsp_error_payload_redacts_absolute_paths(tmp_path):
    error = LspToolError("backend_unhealthy", detail=f"failed in {tmp_path}")
    payload = error.to_payload(workspace_root=tmp_path)
    assert str(tmp_path) not in str(payload)
    assert payload["reason_code"] == "backend_unhealthy"
```

- [x] **Step 2: Run the failing model tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_lsp_models.py -q
```

Expected: FAIL because `mcp_unified.lsp` does not exist yet.

- [x] **Step 3: Implement `errors.py`**

Define:

```python
LSP_REASON_CODES = frozenset({
    "tool_not_granted",
    "path_denied",
    "invalid_path",
    "invalid_position",
    "backend_missing",
    "backend_unhealthy",
    "backend_timeout",
    "capability_unavailable",
    "response_truncated",
    "preview_too_large",
    "unsupported_action_shape",
    "unsupported_language",
    "workspace_not_supported",
    "config_error",
})

class LspToolError(RuntimeError):
    def __init__(self, reason_code: str, message: str | None = None, *, detail: str | None = None):
        super().__init__(message or reason_code)
        self.reason_code = reason_code
        self.detail = detail

    def to_payload(self, *, workspace_root: Path | None = None) -> dict[str, object]:
        safe_detail = redact_lsp_detail(self.detail, workspace_root=workspace_root)
        return {"reason_code": self.reason_code, "message": str(self), "detail": safe_detail}
```

Redaction rule: replace the workspace root string with `<workspace>` and truncate detail to the configured safe detail length.

- [x] **Step 4: Implement `models.py`**

Include frozen/slotted dataclasses or Pydantic models:

```python
@dataclass(frozen=True, slots=True)
class LspPosition:
    line: int
    character: int

@dataclass(frozen=True, slots=True)
class LspRange:
    start: LspPosition
    end: LspPosition

@dataclass(frozen=True, slots=True)
class LspLocation:
    path: str
    range: LspRange
```

Also add diagnostics, symbols, hover, signature, status, preview, and code-action result shapes. Keep `to_dict()` methods deterministic and JSON-serializable.

- [x] **Step 5: Implement `config.py`**

Define `LspRuntimeConfig` with conservative defaults:

```python
request_timeout_seconds = 5.0
startup_timeout_seconds = 10.0
idle_ttl_seconds = 300
max_diagnostics = 500
max_symbols = 500
max_references = 500
max_hover_bytes = 16_000
max_preview_bytes = 200_000
max_stderr_bytes = 8_000
```

Include `from_mapping(settings: Mapping[str, object])`.

- [x] **Step 6: Update package metadata**

Modify `mcp_unified/pyproject.toml`:

```toml
[project.optional-dependencies]
lsp = [
  "ruff>=0.13",
  "python-lsp-server>=1.14",
]

[tool.setuptools]
packages = [
  # Keep every existing package entry and add this one:
  "mcp_unified.lsp",
]

[tool.setuptools.package-dir]
# Keep every existing package-dir entry and add this one:
"mcp_unified.lsp" = "lsp"
```

Do not remove existing package entries such as `mcp_unified.gateway`,
`mcp_unified.profiles`, `mcp_unified.smoke`, or storage/reporting packages.

- [x] **Step 7: Run targeted tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_lsp_models.py -q
```

Expected: PASS.

- [x] **Step 8: Commit Task 1**

```bash
git add mcp_unified/lsp mcp_unified/pyproject.toml tldw_Server_API/app/core/MCP_unified/tests/test_lsp_models.py
git commit -m "feat(mcp): add LSP model contracts"
```

---

### Task 2: Fake Backends, Capability Router, And Service Contract

**Status:** Complete. Local verification on 2026-06-21: `test_lsp_models.py`, `test_lsp_router.py`, and `test_lsp_backends_fake.py` passed with 65 tests; Ruff passed on the touched LSP Python files/tests; Bandit reported zero findings for `mcp_unified/lsp`; `git diff --check` was clean.

**Files:**
- Create: `mcp_unified/lsp/backends.py`
- Create: `mcp_unified/lsp/router.py`
- Create: `mcp_unified/lsp/service.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_lsp_router.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_lsp_backends_fake.py`

- [x] **Step 1: Write failing router tests**

Cover:

```python
async def test_router_routes_diagnostics_to_ruff_backend():
    router = LspCapabilityRouter(ruff=FakeLspBackend("ruff"), pylsp=FakeLspBackend("pylsp"))
    result = await router.diagnostics(file_path="pkg/app.py")
    assert result["backend"] == "ruff"

async def test_router_routes_definition_to_pylsp_backend():
    result = await router.definition(file_path="pkg/app.py", position=LspPosition(0, 1))
    assert result["backend"] == "pylsp"

async def test_missing_backend_returns_capability_unavailable():
    router = LspCapabilityRouter(ruff=None, pylsp=None)
    with pytest.raises(LspToolError) as exc:
        await router.definition(file_path="pkg/app.py", position=LspPosition(0, 1))
    assert exc.value.reason_code == "backend_missing"
```

Also add table-driven tests for every tool in the first-slice surface:

```python
@pytest.mark.parametrize(
    ("tool_name", "expected_backend"),
    [
        ("lsp.diagnostics", "ruff"),
        ("lsp.format_preview", "ruff"),
        ("lsp.code_actions", "ruff"),
        ("lsp.document_symbols", "pylsp"),
        ("lsp.workspace_symbols", "pylsp"),
        ("lsp.definition", "pylsp"),
        ("lsp.references", "pylsp"),
        ("lsp.hover", "pylsp"),
        ("lsp.signature_help", "pylsp"),
    ],
)
async def test_router_covers_every_lsp_tool(tool_name, expected_backend):
    result = await call_router_tool(tool_name, router_with_fakes())
    assert result["backend"] == expected_backend
```

- [x] **Step 2: Run the failing router tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_router.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_backends_fake.py -q
```

Expected: FAIL because router/backend modules do not exist.

- [x] **Step 3: Implement backend protocol and fake backend**

`LspBackend` protocol must expose:

```python
name: str
capabilities: frozenset[str]
async def status(self) -> LspBackendStatus:
    raise NotImplementedError
async def diagnostics(self, request: DiagnosticsRequest) -> DiagnosticsResult:
    raise NotImplementedError
async def document_symbols(self, request: DocumentSymbolsRequest) -> SymbolsResult:
    raise NotImplementedError
async def workspace_symbols(self, request: WorkspaceSymbolsRequest) -> SymbolsResult:
    raise NotImplementedError
async def definition(self, request: PositionRequest) -> LocationsResult:
    raise NotImplementedError
async def references(self, request: ReferencesRequest) -> LocationsResult:
    raise NotImplementedError
async def hover(self, request: PositionRequest) -> HoverResult:
    raise NotImplementedError
async def signature_help(self, request: PositionRequest) -> SignatureHelpResult:
    raise NotImplementedError
async def format_preview(self, request: FormatPreviewRequest) -> PreviewResult:
    raise NotImplementedError
async def code_actions(self, request: CodeActionsRequest) -> CodeActionsResult:
    raise NotImplementedError
```

Use fake backends for deterministic tests; do not start subprocesses here.
The fake backend suite must cover:

- diagnostics;
- document symbols;
- workspace symbols;
- definitions;
- references;
- hover;
- signature help;
- format preview with and without `include_text_edits`;
- code actions with explicit text edits;
- code actions that raise `unsupported_action_shape`;
- limit/truncation metadata;
- backend unhealthy and backend crash/degraded status.

- [x] **Step 4: Implement capability router**

Capability mapping:

```python
RUFF_TOOLS = {"lsp.diagnostics", "lsp.format_preview", "lsp.code_actions"}
PYLSP_TOOLS = {
    "lsp.document_symbols",
    "lsp.workspace_symbols",
    "lsp.definition",
    "lsp.references",
    "lsp.hover",
    "lsp.signature_help",
}
```

Return structured `backend_missing`, `backend_unhealthy`, or `capability_unavailable` errors.

- [x] **Step 5: Implement service facade with backend injection**

`LspCodeIntelligenceService` should accept a router/session manager and expose methods that match the ten tool names. Keep profile/path authorization out of this class; the tldw module handles it.

Add `status()` tests that assert backend provenance, version/config metadata when supplied by fakes, install hints for missing backends, degraded capability status, and redaction of absolute executable paths.

- [x] **Step 6: Run targeted tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_router.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_backends_fake.py -q
```

Expected: PASS.

- [x] **Step 7: Commit Task 2**

```bash
git add mcp_unified/lsp tldw_Server_API/app/core/MCP_unified/tests/test_lsp_router.py tldw_Server_API/app/core/MCP_unified/tests/test_lsp_backends_fake.py
git commit -m "feat(mcp): add LSP capability router"
```

---

### Task 3: Async LSP JSON-RPC Client And Session Manager

**Status:** Complete. Local verification on 2026-06-21: `test_lsp_models.py`, `test_lsp_router.py`, `test_lsp_backends_fake.py`, `test_lsp_jsonrpc.py`, and `test_lsp_sessions.py` passed with 73 tests; Ruff passed on touched LSP Python files/tests; Bandit reported zero findings for `mcp_unified/lsp`; `git diff --check` was clean.

**Files:**
- Create: `mcp_unified/lsp/jsonrpc.py`
- Create: `mcp_unified/lsp/sessions.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/fixtures/fake_lsp_stdio_server.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_lsp_jsonrpc.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_lsp_sessions.py`

- [x] **Step 1: Write failing JSON-RPC framing tests**

Use the fake stdio server fixture to verify:

- `initialize` sends LSP headers with `Content-Length`.
- responses are correlated by id.
- stderr is bounded/redacted.
- timeout raises `backend_timeout`.
- `shutdown` and `exit` are sent during close.

- [x] **Step 2: Run the failing JSON-RPC tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_jsonrpc.py -q
```

Expected: FAIL because the client/fixture do not exist.

- [x] **Step 3: Implement fake LSP stdio server fixture**

The fixture script should:

- read LSP `Content-Length` framed messages from stdin;
- respond to `initialize`, `textDocument/definition`, `textDocument/documentSymbol`, `textDocument/hover`, `textDocument/references`, `textDocument/signatureHelp`, and `shutdown`;
- optionally sleep or crash based on a request parameter for tests;
- write diagnostics only to stderr.

- [x] **Step 4: Implement `LspJsonRpcClient`**

Use `asyncio.create_subprocess_exec(*argv, cwd=workspace_root, stdin=PIPE, stdout=PIPE, stderr=PIPE)`.

Requirements:

- direct argv only;
- no shell;
- startup timeout;
- request timeout;
- Content-Length framing;
- notification support;
- `close()` is exception-safe and terminates the process if graceful shutdown fails;
- stderr capture is bounded by config.

- [x] **Step 5: Write failing session manager tests**

Cover:

```python
async def test_session_manager_reuses_workspace_backend_session(tmp_path):
    manager = LspSessionManager(config=LspRuntimeConfig(idle_ttl_seconds=300), backend_factory=factory)
    first = await manager.get_session("ruff", workspace_root=tmp_path)
    second = await manager.get_session("ruff", workspace_root=tmp_path)
    assert first is second

async def test_session_manager_stop_all_is_exception_safe(tmp_path):
    await manager.stop_all()
    assert manager.active_session_count == 0

async def test_session_manager_evicts_idle_sessions(tmp_path):
    manager = LspSessionManager(config=LspRuntimeConfig(idle_ttl_seconds=1), backend_factory=factory)
    session = await manager.get_session("ruff", workspace_root=tmp_path)
    await manager.evict_idle_sessions(now=session.last_used_monotonic + 2)
    assert manager.active_session_count == 0
```

- [x] **Step 6: Implement `LspSessionManager`**

Cache key:

- canonical workspace root;
- backend id;
- backend executable identity/version where available;
- config fingerprint where available.

Do not include profile id in the process cache key. Authorization is request-scoped later.

- [x] **Step 7: Run targeted tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_jsonrpc.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_sessions.py -q
```

Expected: PASS.

- [x] **Step 8: Commit Task 3**

```bash
git add mcp_unified/lsp tldw_Server_API/app/core/MCP_unified/tests/test_lsp_jsonrpc.py tldw_Server_API/app/core/MCP_unified/tests/test_lsp_sessions.py tldw_Server_API/app/core/MCP_unified/tests/fixtures/fake_lsp_stdio_server.py
git commit -m "feat(mcp): add LSP process sessions"
```

---

### Task 4: Executable Resolver And Real Ruff/pylsp Backends

**Status:** Complete. Local verification on 2026-06-21: the focused LSP suite passed with 84 tests and 5 env-gated skips; the explicit real-backend env run passed with 11 tests and 5 skips because `ruff`/`pylsp` are not installed on PATH; Ruff passed on touched LSP files/tests; Bandit reported zero findings for `mcp_unified/lsp`; `git diff --check` was clean. Added a workspace-boundary guard so backend file reads and URIs reject path traversal or absolute-path escapes before invoking LSP.

**Files:**
- Create: `mcp_unified/lsp/executables.py`
- Create: `mcp_unified/lsp/ruff.py`
- Create: `mcp_unified/lsp/pylsp.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_lsp_real_backends.py`

- [x] **Step 1: Write failing executable resolver tests**

Cover explicit config, virtualenv discovery, PATH discovery, and missing executable status. Use temporary executable stubs; do not require real Ruff/pylsp for resolver unit cases.

- [x] **Step 2: Implement `LspExecutableResolver`**

Resolution order:

1. explicit `ruff_command` / `pylsp_command` config as argv list;
2. active project virtualenv bin directory;
3. PATH through `shutil.which`.

Reject:

- shell strings containing spaces/control operators;
- wrapper commands such as `npx`, `docker`, `devbox`, unless later explicitly designed;
- non-executable paths.

- [x] **Step 3: Write env-gated real backend tests**

Use `pytest.mark.skipif` unless:

```python
os.getenv("TLDW_MCP_LSP_REAL_BACKENDS") == "1"
```

and the required executable is present.

Real tests:

- Ruff diagnostics detects an unused import or syntax/lint issue in an isolated `.py` file.
- Ruff `format_preview` returns a `unified_diff` for unformatted code and no `text_edits` unless `include_text_edits=True`.
- pylsp document symbols returns a function/class from a small module.
- pylsp definition resolves a local function call.
- missing Ruff or missing pylsp degrades status without failing the whole service.

- [x] **Step 4: Implement Ruff backend**

Use `ruff server` through `LspJsonRpcClient`. First-slice requirements:

- initialize workspace root;
- open/read file text when needed for LSP document operations;
- file-level diagnostics only;
- format preview returns canonical unified diff;
- code actions return explicit text-edit previews only;
- opaque command-only actions raise `unsupported_action_shape`.

- [x] **Step 5: Implement pylsp backend**

Use `pylsp` through `LspJsonRpcClient`. Implement:

- document symbols;
- workspace symbols;
- definition;
- references;
- hover;
- signature help.

Normalize all returned URIs to workspace-relative paths and preserve UTF-16 LSP positions.

- [x] **Step 6: Run fake and resolver tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_router.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_jsonrpc.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_real_backends.py -q
```

Expected: PASS, with real backend tests skipped unless env-gated.

- [x] **Step 7: Run real backend tests when tools are available**

Run:

```bash
source .venv/bin/activate && TLDW_MCP_LSP_REAL_BACKENDS=1 python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_real_backends.py -q
```

Expected: PASS or documented SKIP for missing `ruff`/`pylsp`.

- [x] **Step 8: Commit Task 4**

```bash
git add mcp_unified/lsp tldw_Server_API/app/core/MCP_unified/tests/test_lsp_real_backends.py
git commit -m "feat(mcp): add Ruff and pylsp backends"
```

---

### Task 5: Result Filtering Contract

**Status:** Complete. Local verification on 2026-06-21: Task 5 filtering tests passed with 7 tests; the focused LSP suite passed with 91 tests and 5 env-gated skips; Ruff passed on touched LSP files/tests; Bandit reported zero findings for `mcp_unified/lsp`; `git diff --check` was clean.

**Files:**
- Create: `mcp_unified/lsp/filtering.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_lsp_filtering.py`

- [x] **Step 1: Write failing filtering tests**

Create fake result payloads for all path-bearing result families:

- diagnostics;
- document symbols;
- workspace symbols;
- definition locations;
- reference locations;
- format preview affected paths;
- code-action affected paths and workspace edits.

Example tests:

```python
def test_filter_locations_removes_denied_paths():
    result = LocationsResult(locations=[
        LspLocation(path="src/allowed.py", range=sample_range()),
        LspLocation(path="private/secret.py", range=sample_range()),
    ])
    filtered = filter_lsp_result_paths(result, is_path_allowed=lambda path: path.startswith("src/"))
    assert [location.path for location in filtered.locations] == ["src/allowed.py"]
    assert filtered.filtered_count == 1

def test_filter_preview_rejects_when_any_affected_path_is_denied():
    result = PreviewResult(
        affected_paths=["src/allowed.py", "private/secret.py"],
        unified_diff="--- a/src/allowed.py\n+++ b/src/allowed.py\n",
    )
    with pytest.raises(LspToolError) as exc:
        filter_lsp_result_paths(result, is_path_allowed=lambda path: path.startswith("src/"))
    assert exc.value.reason_code == "path_denied"
```

- [x] **Step 2: Run failing filtering tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_filtering.py -q
```

Expected: FAIL because `mcp_unified.lsp.filtering` does not exist.

- [x] **Step 3: Implement `filtering.py`**

Expose:

```python
PathAllowPredicate = Callable[[str], bool]

def filter_lsp_result_paths(result: object, *, is_path_allowed: PathAllowPredicate) -> object:
    return filter_locations_or_fail_closed(result, is_path_allowed=is_path_allowed)
```

Rules:

- location/symbol/diagnostic result lists are filtered and include `filtered_count`;
- preview/code-action results fail closed with `path_denied` if any affected path is denied, because returning a partial patch can be unsafe;
- no raw absolute paths are added to error detail;
- workspace-root/index-style tools can use `require_workspace_root_allowed=True` or a separate helper to fail fast when root read is not granted.

- [x] **Step 4: Run filtering tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_filtering.py -q
```

Expected: PASS.

- [x] **Step 5: Commit Task 5**

```bash
git add mcp_unified/lsp/filtering.py tldw_Server_API/app/core/MCP_unified/tests/test_lsp_filtering.py
git commit -m "feat(mcp): add LSP result filtering"
```

---

### Task 6: tldw-hosted `LSPModule` And Path Governance

**Status:** Complete. Local verification on 2026-06-21: `test_lsp_module.py`, `test_lsp_module_registration.py`, and `test_protocol_scope_enforcement.py` passed with 20 tests; the focused LSP suite passed with 111 tests and 5 env-gated skips; Ruff passed on touched LSP files/tests; Bandit reported zero findings for `mcp_unified/lsp` plus the hosted LSP module; `git diff --check` was clean. Existing protocol scope tests covered the protocol side, so no redundant `test_protocol_scope_enforcement.py` edits were needed.

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/modules/implementations/lsp_module.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/server.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_lsp_module.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_scope_enforcement.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_lsp_module_registration.py`

- [x] **Step 1: Write failing tool definition tests**

Assert `get_tools()` exposes:

- `lsp.status`
- `lsp.diagnostics`
- `lsp.document_symbols`
- `lsp.workspace_symbols`
- `lsp.definition`
- `lsp.references`
- `lsp.hover`
- `lsp.signature_help`
- `lsp.format_preview`
- `lsp.code_actions`

Each definition must include:

- `readOnlyHint: True`;
- `uses_filesystem: True`;
- `path_boundable: True`;
- `path_scope_candidate_source: "module"` for file/path-scoped tools;
- `path_scope_action: "read"` where applicable;
- category `retrieval` for read/navigation tools and `analysis` or `retrieval` for preview tools.

- [x] **Step 2: Write failing path-scope candidate tests**

Cases:

```python
async def test_lsp_definition_extracts_read_candidate_for_file():
    candidates = await module.extract_path_scope_candidates("lsp.definition", {"path": "src/app.py", "position": {"line": 1, "character": 2}})
    assert candidates == [PathScopeCandidate(path="src/app.py", action="read", source="lsp.definition", requires_existing_file=True)]

async def test_lsp_workspace_symbols_requires_workspace_root_read_candidate():
    candidates = await module.extract_path_scope_candidates("lsp.workspace_symbols", {"query": "Widget"})
    assert candidates[0].path == "."
    assert candidates[0].action == "read"
    assert candidates[0].source == "lsp.workspace_symbols"
```

- [x] **Step 3: Run failing module tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_module.py -q
```

Expected: FAIL because `LSPModule` does not exist.

- [x] **Step 4: Write failing tldw-hosted execution tests with fake service**

Add tests using an injected fake service:

```python
async def test_lsp_module_filters_denied_definition_results():
    module = LSPModule(config, service=fake_service_returning_paths(["src/app.py", "private/secret.py"]))
    result = await module.execute_tool("lsp.definition", {"path": "src/app.py", "position": {"line": 0, "character": 1}}, context=allowed_src_context)
    assert [item["path"] for item in result["locations"]] == ["src/app.py"]
    assert result["filtered_count"] == 1

async def test_lsp_module_rejects_preview_with_denied_affected_path():
    module = LSPModule(config, service=fake_service_preview_paths(["src/app.py", "private/secret.py"]))
    with pytest.raises(PermissionError, match="path_denied"):
        await module.execute_tool("lsp.format_preview", {"path": "src/app.py"}, context=allowed_src_context)
```

If existing policy APIs make direct effective path checks hard inside module tests, inject a path allow predicate into the module or service for tests and wire production to protocol-provided context metadata.
Do not skip post-result filtering just because protocol preflight passed: LSP
responses can include additional paths that were not part of the initial request.

- [x] **Step 5: Implement `LSPModule`**

Constructor:

```python
class LSPModule(BaseModule):
    def __init__(self, config: ModuleConfig, service: LspCodeIntelligenceService | None = None, workspace_root_resolver: Any | None = None) -> None:
        super().__init__(config)
        self._service = service
        self._workspace_root_resolver = workspace_root_resolver
```

Use `McpHubWorkspaceRootResolver` by default, matching `FilesystemModule`.

Tool execution flow:

1. resolve active workspace root;
2. normalize arguments into service request models;
3. call `LspCodeIntelligenceService`;
4. filter returned paths through `filter_lsp_result_paths()` using request/profile path policy context;
5. return JSON-serializable dicts;
6. convert `LspToolError` into safe module errors without leaking absolute paths.

Production filtering implementation detail: start with a conservative helper
that allows only the request path for file-scoped calls and requires `"."` for
workspace-wide calls unless the context metadata includes a broader
policy-evaluated allowlist. Later policy APIs can replace this helper, but the
first implementation must never return an LSP path that is outside the
request-scoped allow predicate.

- [x] **Step 6: Implement validation**

`validate_tool_arguments()` should reject:

- negative positions;
- missing file paths for file-scoped tools;
- non-Python file paths in first slice;
- `include_text_edits` non-boolean values;
- invalid limits.

- [x] **Step 7: Add optional server registration**

In `server.py`, add after safe read/code modules:

```python
if self._env_flag_enabled("MCP_ENABLE_LSP_MODULE"):
    modules_to_load.append({
        "id": "lsp",
        "class": "tldw_Server_API.app.core.MCP_unified.modules.implementations.lsp_module:LSPModule",
        "enabled": True,
        "name": "LSP Code Intelligence",
        "version": "1.0.0",
        "department": "code",
        "settings": {
            "request_timeout_seconds": "${MCP_LSP_REQUEST_TIMEOUT_SECONDS:-5}",
            "startup_timeout_seconds": "${MCP_LSP_STARTUP_TIMEOUT_SECONDS:-10}",
            "idle_ttl_seconds": "${MCP_LSP_IDLE_TTL_SECONDS:-300}",
            "ruff_command": "${MCP_LSP_RUFF_COMMAND:-}",
            "pylsp_command": "${MCP_LSP_PYLSP_COMMAND:-}",
        },
    })
```

Keep disabled by default.

- [x] **Step 8: Run module and protocol scope tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_scope_enforcement.py -q
```

Expected: PASS.

- [x] **Step 9: Commit Task 6**

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/lsp_module.py tldw_Server_API/app/core/MCP_unified/server.py tldw_Server_API/app/core/MCP_unified/tests/test_lsp_module.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_scope_enforcement.py
git commit -m "feat(mcp): expose LSP tools in tldw MCP"
```

---

### Task 7: Standalone Gateway Runtime And Smoke/UAT Scenario

**Status:** Complete. Local verification on 2026-06-21: the new Task 7 tests passed with 7 tests; the plan-targeted suite (`test_lsp_gateway_runtime.py`, `test_lsp_smoke_scenario.py`, `test_lsp_module.py`, `test_smoke_client.py`) passed with 115 tests after rerunning with escalated loopback permissions for existing WebSocket bind tests; the focused `test_lsp_*.py` suite passed with 111 tests and 5 expected env-gated skips; Ruff passed on touched files; Bandit reported zero findings for `mcp_unified/lsp`, `mcp_unified/smoke/scenarios.py`, and `mcp_unified/smoke/cli.py`; `git diff --check` was clean.

**Files:**
- Create: `mcp_unified/lsp/gateway_runtime.py`
- Modify: `mcp_unified/smoke/scenarios.py`
- Modify: `mcp_unified/smoke/cli.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_lsp_gateway_runtime.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_lsp_smoke_scenario.py`

- [x] **Step 1: Write failing standalone runtime tests**

Use fake backends:

```python
async def test_lsp_gateway_runtime_lists_tools():
    runtime = LspGatewayRuntime(service=fake_service)
    tools = await runtime.list_tools(GatewayRequestContext(request_id="r1"))
    assert "lsp.status" in {tool["name"] for tool in tools}

async def test_lsp_gateway_runtime_calls_status():
    result = await runtime.call_tool("lsp.status", {}, GatewayRequestContext(request_id="r1"))
    assert result["status"] in {"healthy", "degraded"}
```

- [x] **Step 2: Implement `LspGatewayRuntime`**

This runtime is for standalone smoke/UAT and embedders who want only LSP tools.
It should:

- implement `GatewayRuntime`;
- list the same ten tool definitions as `LSPModule`;
- call `LspCodeIntelligenceService`;
- apply `filter_lsp_result_paths()` with a constructor-injected `path_allow_predicate`;
- default the predicate to allowing all workspace-relative paths under the configured workspace and document that embedders with profile policy should wrap or override it;
- use workspace root from constructor/config and reject missing workspace with `workspace_not_supported`.

- [x] **Step 3: Write failing smoke scenario tests**

Add `run_lsp_scenario()` with fake runtime/transport. It should:

- initialize;
- list tools;
- call `lsp.status`;
- call file-level diagnostics against an isolated Python fixture;
- call document symbols or definition when pylsp capability is available;
- report skips in best-effort mode;
- fail in strict mode when required LSP tools are absent.
- run against a standalone `LspGatewayRuntime` transport in unit tests;
- run against a tldw-hosted MCP test app/runtime with `LSPModule` enabled and fake service injection.

- [x] **Step 4: Implement smoke scenario and CLI flag**

Add CLI scenario choice:

```bash
mcp-unified-smoke --scenario lsp --mode best-effort
mcp-unified-smoke --scenario lsp --mode strict
```

Keep existing baseline and real-world scenarios unchanged.

- [x] **Step 5: Run targeted tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_gateway_runtime.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_smoke_scenario.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py -q
```

Expected: PASS.

- [x] **Step 6: Commit Task 7**

```bash
git add mcp_unified/lsp/gateway_runtime.py mcp_unified/smoke/scenarios.py mcp_unified/smoke/cli.py tldw_Server_API/app/core/MCP_unified/tests/test_lsp_gateway_runtime.py tldw_Server_API/app/core/MCP_unified/tests/test_lsp_smoke_scenario.py
git commit -m "feat(mcp): add LSP smoke scenario"
```

---

### Task 8: Profile Presets, Docs, And Operator Guidance

**Status:** Complete. Local verification on 2026-06-21: profile preset tests passed with 28 tests; the Task 8 targeted pytest group passed with 37 tests; package metadata release-gate tests passed; the standalone artifact gate passed with 4 tests; Ruff passed on touched Python files; Bandit reported zero findings for touched package code; `git diff --check` was clean. Implementation note: `code_intelligence` is intentionally deferred rather than direct because several code-oriented presets are already near the 24-tool direct disclosure cap.

**Files:**
- Modify: `mcp_unified/profiles/presets.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py`
- Modify: `mcp_unified/README.md`
- Modify: `mcp_unified/USER_GUIDE.md`
- Modify: `Docs/MCP/Unified/Smoke_Client.md`
- Modify: `mcp_unified/package_metadata.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`

- [x] **Step 1: Write failing profile preset tests**

Add `_LSP_TOOLS` expected set:

```python
LSP_TOOLS = {
    "lsp.status",
    "lsp.diagnostics",
    "lsp.document_symbols",
    "lsp.workspace_symbols",
    "lsp.definition",
    "lsp.references",
    "lsp.hover",
    "lsp.signature_help",
    "lsp.format_preview",
    "lsp.code_actions",
}
```

Assert code-oriented presets include the tools in `metadata["tooling"]["enabled_tools"]`, not necessarily `policy_document.allowed_tools` if existing profile semantics keep executable policy capability-based.

Likely presets:

- `architect`
- `project-researcher`
- `code-reviewer`
- `devops-engineer`
- `backend-engineer`
- `frontend-engineer`
- `qa-engineer`
- `sdet`
- `merge-conflict-resolver`

- [x] **Step 2: Update preset tooling metadata**

Add:

```python
_LSP_TOOLS = [
    "lsp.status",
    "lsp.diagnostics",
    "lsp.document_symbols",
    "lsp.workspace_symbols",
    "lsp.definition",
    "lsp.references",
    "lsp.hover",
    "lsp.signature_help",
    "lsp.format_preview",
    "lsp.code_actions",
]
```

Add enabled capability:

```python
"code_intelligence.lsp"
```

Add deferred category:

```python
"code_intelligence"
```

Keep process/shell risk classes unchanged. Do not put all ten LSP tools in the
direct category by default; they remain available through progressive
disclosure/tool search.

- [x] **Step 3: Update docs**

`mcp_unified/README.md`:

- add `pip install "mcp-unified[lsp]"`;
- mention `MCP_ENABLE_LSP_MODULE=true` for tldw-hosted module;
- mention Ruff/pylsp discovery and degraded status.

`mcp_unified/USER_GUIDE.md`:

- include first-run checklist;
- explain `lsp.status`;
- explain preview-only edits;
- explain workspace-root grant required for `lsp.workspace_symbols`;
- mention operator-managed pylsp plugins are trusted runtime inputs.

`Docs/MCP/Unified/Smoke_Client.md`:

- document `--scenario lsp`;
- document env-gated real backend tests and strict/best-effort modes.

- [x] **Step 4: Run docs/profile tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py -q
```

Expected: PASS.

- [x] **Step 5: Commit Task 8**

```bash
git add mcp_unified/profiles/presets.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py mcp_unified/README.md mcp_unified/USER_GUIDE.md Docs/MCP/Unified/Smoke_Client.md
git commit -m "docs(mcp): document LSP code intelligence tools"
```

---

### Task 9: Full Verification, Bandit, And Backlog Finalization

**Status:** Complete. Local verification on 2026-06-21: the focused LSP suite passed with 109 tests and 5 expected real-backend skips; the regression-adjacent MCP suite passed with 136 tests after rerunning with escalated loopback permissions for existing WebSocket bind tests; the explicitly enabled real-backend suite passed 11 tests with 5 skips; the LSP smoke CLI best-effort scenario passed against the in-process runtime with backend-unavailable/capability-unavailable best-effort notes; Bandit reported zero findings for the touched LSP/smoke/profile/server scope; `git diff --check` was clean.

**Files:**
- Modify: `Docs/superpowers/plans/2026-06-21-mcp-lsp-code-intelligence-tools-implementation-plan.md`
- Modify: `backlog/tasks/task-2281 - Add-LSP-backed-code-intelligence-MCP-tools.md`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_server_batch_and_formatting.py`

- [x] **Step 1: Run focused LSP test suite**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_models.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_router.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_jsonrpc.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_sessions.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_filtering.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_backends_fake.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_real_backends.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_gateway_runtime.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_smoke_scenario.py -q
```

Expected: PASS, with real backend tests skipped unless explicitly enabled.

- [x] **Step 2: Run regression-adjacent MCP tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_scope_enforcement.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_server_batch_and_formatting.py -q
```

Expected: PASS.

- [x] **Step 3: Run real backend tests if available**

Run:

```bash
source .venv/bin/activate && TLDW_MCP_LSP_REAL_BACKENDS=1 python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_lsp_real_backends.py -q
```

Expected: PASS or documented SKIP for missing `ruff`/`pylsp`.

- [x] **Step 4: Run smoke CLI best-effort scenario if implementation exposes local test runtime**

Run:

```bash
source .venv/bin/activate && python -m mcp_unified.smoke.cli --scenario lsp --mode best-effort inprocess
```

Expected: PASS or SKIP steps for missing optional backends; no raw absolute paths or secrets in output.

- [x] **Step 5: Run Bandit on touched Python scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  mcp_unified/lsp \
  mcp_unified/smoke \
  mcp_unified/profiles \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/lsp_module.py \
  tldw_Server_API/app/core/MCP_unified/server.py \
  -f json -o /tmp/bandit_mcp_lsp_code_intelligence.json
```

Expected: no new high/medium findings in touched code. Fix new findings before proceeding.

- [x] **Step 6: Update Backlog task**

Record:

- touched files;
- focused test results;
- real backend skip/pass status;
- Bandit result path;
- known limitations: Python-only, single-root, preview-only, file-level diagnostics.

- [x] **Step 7: Commit verification metadata**

```bash
git add 'backlog/tasks/task-2281 - Add-LSP-backed-code-intelligence-MCP-tools.md'
git commit -m "chore(mcp): record LSP verification"
```

---

## PR Readiness Checklist

- [ ] Branch rebased on latest `origin/dev`.
- [ ] No unrelated worktree changes.
- [ ] All planned tests run or skips documented.
- [ ] Bandit touched-scope report run and new findings fixed.
- [ ] `lsp.*` tools remain preview-only for edits.
- [ ] `lsp.workspace_symbols` requires workspace-root read grant.
- [ ] Tool outputs and errors redact absolute paths/secrets.
- [ ] Real backend tests are env-gated and deterministic when enabled.
- [ ] User-facing docs explain install/enable/UAT workflow.
- [ ] Backlog task includes final summary and verification results.
