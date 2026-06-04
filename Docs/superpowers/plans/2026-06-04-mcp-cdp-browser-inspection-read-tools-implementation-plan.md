# MCP CDP Browser Inspection Read Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional, read-only CDP-backed browser inspection MCP tools for browser-capable profiles.

**Architecture:** Add a small CDP client seam under `tldw_Server_API/app/core/MCP_unified/browser_cdp/` and a `BrowserCDPModule` under MCP module implementations. Keep CDP endpoint configuration operator-owned, use fake clients for unit tests, register the module only when explicitly enabled/configured, and wire profile discovery by adding read-only browser tool names/capabilities to existing browser-oriented presets.

**Tech Stack:** Python 3.11, MCP Unified module framework, `httpx`, `websockets`, pytest, pytest-asyncio, Bandit.

---

## File Structure

- Create: `tldw_Server_API/app/core/MCP_unified/browser_cdp/__init__.py`
  - Package marker and public client exports.
- Create: `tldw_Server_API/app/core/MCP_unified/browser_cdp/client.py`
  - CDP configuration, endpoint validation, HTTP target/version discovery, WebSocket command dispatch, bounded event observation, and reason-coded errors.
- Create: `tldw_Server_API/app/core/MCP_unified/modules/implementations/browser_cdp_module.py`
  - MCP tool descriptors, argument validation, result shaping, CDP client factory injection, and module health.
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_client.py`
  - Client seam tests with fake HTTP/WebSocket adapters or monkeypatched factory functions.
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_module.py`
  - Module descriptor, validation, fake-client execution, and truncation tests.
- Modify: `tldw_Server_API/app/core/MCP_unified/server.py`
  - Optional default module registration gated by `MCP_ENABLE_BROWSER_CDP_MODULE` or `MCP_BROWSER_CDP_URL`.
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_server_registration.py`
  - Add focused server registration coverage for the new module.
- Modify: `mcp_unified/profiles/presets.py`
  - Add browser read tools/capabilities to Frontend Engineer, QA Engineer, and SDET tooling/policy where appropriate.
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py`
  - Add profile metadata/discovery expectations for browser read tools.
- Modify: `mcp_unified/USER_GUIDE.md`
  - Document CDP configuration, read-only browser inspection tools, and non-goals.
- Modify: `backlog/tasks/task-2247 - Implement-MCP-CDP-browser-inspection-read-tools.md`
  - Track implementation notes, verification, final summary, and Definition of Done.

## Task 0: Planning And Baseline

**Files:**
- Modify: `backlog/tasks/task-2247 - Implement-MCP-CDP-browser-inspection-read-tools.md`

- [ ] **Step 1: Confirm the working tree**

Run:

```bash
git status --short --branch
```

Expected: branch `codex/mcp-cdp-browser-inspection` with only the spec, plan, and task tracking changes before implementation code starts.

- [ ] **Step 2: Record plan references in Backlog**

Update `TASK-2247` with:

- `Docs/superpowers/specs/2026-06-04-mcp-cdp-browser-inspection-read-tools-design.md`
- `Docs/superpowers/plans/2026-06-04-mcp-cdp-browser-inspection-read-tools-implementation-plan.md`

- [ ] **Step 3: Commit planning artifacts**

Run:

```bash
git add Docs/superpowers/specs/2026-06-04-mcp-cdp-browser-inspection-read-tools-design.md Docs/superpowers/plans/2026-06-04-mcp-cdp-browser-inspection-read-tools-implementation-plan.md "backlog/tasks/task-2247 - Implement-MCP-CDP-browser-inspection-read-tools.md"
git commit -m "docs: plan mcp cdp browser inspection tools"
```

## Task 1: CDP Client Seam

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/browser_cdp/__init__.py`
- Create: `tldw_Server_API/app/core/MCP_unified/browser_cdp/client.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_client.py`

- [ ] **Step 1: Write failing endpoint validation tests**

Create tests for:

- no configured debugger URL returns `cdp_not_configured`;
- `http://127.0.0.1:9222` and `http://localhost:9222` are accepted;
- non-loopback hosts are rejected unless `allow_non_loopback=True`;
- only literal loopback hosts are accepted by default: `localhost`, `127.0.0.0/8`, and `::1`;
- endpoint validation does not perform DNS resolution to prove a hostname is loopback;
- tool-call-provided URLs are not part of the client API.

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_client.py -q
```

Expected: FAIL because the package/client does not exist.

- [ ] **Step 2: Implement configuration and endpoint validation**

Add:

```python
@dataclass(frozen=True, slots=True)
class CDPClientConfig:
    debugger_url: str | None
    request_timeout_seconds: float = 3.0
    observation_window_ms: int = 250
    max_events: int = 100
    max_snapshot_nodes: int = 200
    screenshot_max_bytes: int = 2_000_000
    allow_non_loopback: bool = False
```

Add `CDPClientError(reason_code: str, message: str)` and endpoint normalization/validation helpers. Keep accepted schemes to `http` and `https`; convert WebSocket debugger URLs only when returned by CDP target discovery.

- [ ] **Step 3: Write failing target discovery tests**

Test `get_version()` and `list_pages()` using a fake async HTTP getter injected into the client. Verify:

- `/json/version` payload is normalized;
- `/json/list` filters to page targets by default;
- page payloads include `target_id`, `title`, `url`, `type`, and `webSocketDebuggerUrl`;
- HTTP failures raise `cdp_unreachable`.

- [ ] **Step 4: Implement HTTP discovery**

Use `httpx.AsyncClient` behind a small overridable method:

```python
async def _get_json(self, path: str) -> Any:
    ...
```

Use timeouts from config, close clients with `async with`, and never log full target URLs with secrets.

- [ ] **Step 5: Write failing WebSocket command/event tests**

Use monkeypatched WebSocket connection objects to test:

- command IDs increment;
- `send_command(page, "Browser.getVersion")` returns the matching `result`;
- CDP error payload raises `cdp_protocol_error`;
- `observe_events()` stops at `max_events` or window timeout and reports `truncated`.

- [ ] **Step 6: Implement WebSocket helpers**

Use `websockets.connect()` inside a helper that can be monkeypatched. Implement:

```python
async def send_command(self, page: CDPPageTarget, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    ...

async def observe_events(
    self,
    page: CDPPageTarget,
    *,
    enable_methods: list[str],
    event_names: set[str],
    window_ms: int,
    max_events: int,
) -> dict[str, Any]:
    ...
```

- [ ] **Step 7: Run client tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_client.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit Task 1**

```bash
git add tldw_Server_API/app/core/MCP_unified/browser_cdp tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_client.py
git commit -m "feat: add cdp browser client seam"
```

## Task 2: Browser CDP MCP Module Schemas And Validation

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/modules/implementations/browser_cdp_module.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_module.py`

- [ ] **Step 1: Write failing descriptor tests**

Test that `BrowserCDPModule.get_tools()` exposes:

- `browser.status`
- `browser.pages.list`
- `browser.snapshot`
- `browser.page_state`
- `browser.screenshot`
- `browser.console`
- `browser.network`

Each descriptor must have `additionalProperties: false`, metadata `category: "browser"`, `readOnlyHint: true`, and suitable capabilities (`browser.inspect`, `browser.debug`, `screenshots.capture`, or `app_state.read`).

- [ ] **Step 2: Implement minimal module and descriptors**

Create `BrowserCDPModule(BaseModule)`. Use `create_tool_definition()`, then set `additionalProperties` false for every tool. Accept an optional `client_factory` in `__init__` for tests.

- [ ] **Step 3: Write failing validation tests**

Cover:

- unknown tool names;
- unknown argument keys;
- optional `target_id` must be a non-empty string;
- `limit`, `max_events`, and `window_ms` must be positive integers within module caps;
- screenshot format must be `png` or `jpeg`;
- no tool accepts `url`, `script`, `expression`, `selector`, or interaction arguments.

- [ ] **Step 4: Implement validation**

Add explicit allowed-key sets per tool and bounded integer helpers. Do not silently coerce strings into booleans/integers for tool-call arguments.

- [ ] **Step 5: Run schema/validation tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_module.py -q -k "metadata or validates"
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/browser_cdp_module.py tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_module.py
git commit -m "feat: add browser cdp tool schemas"
```

## Task 3: Read-Only Tool Execution

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/browser_cdp_module.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_module.py`

- [ ] **Step 1: Write failing fake-client execution tests**

Use a fake client with deterministic pages, command results, screenshot payloads, and observed events. Test:

- `browser.status` when configured/reachable and when not configured;
- `browser.pages.list` filters page targets;
- `browser.snapshot` returns bounded snapshot data and `truncated`;
- `browser.page_state` returns URL/title/readiness/viewport fields;
- `browser.screenshot` returns MIME/base64/byte estimate and rejects oversized payloads with `payload_too_large`;
- `browser.console` and `browser.network` return observed events, `observed_for_ms`, and truncation metadata.

- [ ] **Step 2: Implement target resolution**

Add a helper:

```python
async def _resolve_page(self, client: CDPBrowserClient, target_id: str | None) -> CDPPageTarget:
    ...
```

If no target is given, use the first page target. If none exists, return/raise `target_not_found`.

- [ ] **Step 3: Implement `browser.status` and `browser.pages.list`**

Return structured payloads with reason codes instead of stack traces for unavailable CDP.

- [ ] **Step 4: Implement snapshot and page-state fixed commands**

Use only fixed module-owned CDP commands/scripts. Recommended first commands:

- `Accessibility.getFullAXTree` or `DOM.getDocument` for snapshot data;
- fixed read-only `Runtime.evaluate` script for page state, with no caller-provided expression.

Bound returned nodes/entries by `max_snapshot_nodes`.

- [ ] **Step 5: Implement screenshot, console, and network**

Use:

- `Page.captureScreenshot` with module-controlled format/quality;
- `Runtime.enable`/`Log.enable` event observation for console/logs;
- `Network.enable` event observation for request/response/failure events.

Keep observation windows bounded and return empty event lists truthfully when no events occur.

- [ ] **Step 6: Run module tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_module.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 3**

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/browser_cdp_module.py tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_module.py
git commit -m "feat: execute read only cdp browser tools"
```

## Task 4: Server Registration And Profile Discovery

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/server.py`
- Create or modify: `tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_server_registration.py`
- Modify: `mcp_unified/profiles/presets.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py`

- [ ] **Step 1: Write failing server registration tests**

Verify:

- default server registration does not include browser CDP when no env/config is set;
- `MCP_ENABLE_BROWSER_CDP_MODULE=true` queues/registers `browser_cdp`;
- `MCP_BROWSER_CDP_URL=http://127.0.0.1:9222` queues/registers `browser_cdp`;
- explicit `MCP_ENABLE_BROWSER_CDP_MODULE=false` disables registration even when `MCP_BROWSER_CDP_URL` is set.

- [ ] **Step 2: Implement optional registration**

In `_register_default_modules()`, append:

```python
{
    "id": "browser_cdp",
    "class": "tldw_Server_API.app.core.MCP_unified.modules.implementations.browser_cdp_module:BrowserCDPModule",
    "enabled": True,
    "name": "Browser CDP",
    "department": "browser",
    "settings": {"debugger_url": "${MCP_BROWSER_CDP_URL}"},
}
```

Only add it when enabled or URL is configured. Avoid default-on registration.
Honor explicit disable before URL-based auto-registration.

- [ ] **Step 3: Write failing profile tests**

Assert Frontend Engineer and QA Engineer include browser read tools in `metadata["tooling"]["enabled_tools"]` and have enough policy capability to discover installed backend descriptors for those tools.

- [ ] **Step 4: Update profile metadata and capabilities**

Add a local tuple:

```python
_BROWSER_READ_TOOLS = [
    "browser.status",
    "browser.pages.list",
    "browser.snapshot",
    "browser.page_state",
    "browser.screenshot",
    "browser.console",
    "browser.network",
]
```

Add these to Frontend Engineer and QA Engineer tooling enabled tools. Add `browser.inspect` to their policy capabilities if needed. Add SDET as deferred/recommended only unless the existing preset policy is explicitly browser-capable.

- [ ] **Step 5: Run profile/discovery tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_server_registration.py \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```bash
git add tldw_Server_API/app/core/MCP_unified/server.py tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_server_registration.py mcp_unified/profiles/presets.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py
git commit -m "feat: wire browser cdp tools into profiles"
```

## Task 5: Documentation And Final Verification

**Files:**
- Modify: `mcp_unified/USER_GUIDE.md`
- Modify: `backlog/tasks/task-2247 - Implement-MCP-CDP-browser-inspection-read-tools.md`

- [ ] **Step 1: Update user guide**

Document:

- enabling with `MCP_BROWSER_CDP_URL=http://127.0.0.1:9222`;
- optional `MCP_ENABLE_BROWSER_CDP_MODULE=true`;
- read-only tool list;
- console/network observation windows are not historical logs;
- browser interaction remains a future approval-gated capability.

- [ ] **Step 2: Run focused tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_client.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_server_registration.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  -q
```

Expected: PASS.

- [ ] **Step 3: Run Bandit on touched Python**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/MCP_unified/browser_cdp \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/browser_cdp_module.py \
  tldw_Server_API/app/core/MCP_unified/server.py \
  mcp_unified/profiles/presets.py \
  -f json \
  -o /tmp/bandit_mcp_cdp_browser_tools.json
```

Expected: exit 0 or only known non-touched baseline findings. Fix new findings before continuing.

- [ ] **Step 4: Run lint and whitespace checks**

Run:

```bash
source .venv/bin/activate
python -m ruff check \
  tldw_Server_API/app/core/MCP_unified/browser_cdp \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/browser_cdp_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_client.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_server_registration.py
git diff --check
```

Expected: PASS and no whitespace diagnostics.

- [ ] **Step 5: Optional live CDP smoke**

If a local Chrome/Chromium CDP endpoint is already running, execute `browser.status` and `browser.pages.list` through a lightweight module call or protocol test. Do not start or install browsers for this slice unless the user explicitly asks.

- [ ] **Step 6: Finalize Backlog task**

Record:

- RED/GREEN notes;
- focused test results;
- Bandit report path;
- live CDP smoke status or skip reason;
- final summary and Definition of Done.

- [ ] **Step 7: Commit final docs/task updates**

```bash
git add mcp_unified/USER_GUIDE.md "backlog/tasks/task-2247 - Implement-MCP-CDP-browser-inspection-read-tools.md"
git commit -m "docs: document mcp cdp browser tools"
```
