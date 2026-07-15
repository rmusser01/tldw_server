# Web Scraping Phase 3 Governed Preflight Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the complete pre-scrape analyzer into a governed `Web_Scraping/preflight` package, migrate both scrape consumers to one typed facade, and preserve all successful analyzer, scrape, and legacy import behavior.

**Architecture:** Keep scrape-level policy in `OutboundPolicyChecker`, add a narrower per-dispatch `ProbeEgressGuard`, and inject governed HTTP, browser, and external-tool adapters through one request-scoped `PreflightExecutionContext`. Canonical analyzers use private async implementations; exact historical public callables remain available through compatibility wrappers and explicit old-path re-exports. `Article_Extractor_Lib` and `EnhancedWebScraper` consume only the facade and convert typed results to legacy dictionaries at their existing public boundaries.

**Tech Stack:** Python 3.11+ asyncio, frozen dataclasses, `typing.Protocol`, existing `Security.egress`, existing `http_client.afetch`, curl-cffi async sessions, Playwright async API >=1.48.0, Loguru, project Metrics, pytest/pytest-asyncio, Hypothesis, AST contract tests, Ruff, Black, and Bandit.

**Backlog:** `TASK-12969`

**Design:** `Docs/superpowers/specs/2026-07-14-web-scraping-phase-3-governed-preflight-package-design.md`

## Global Constraints

- Before editing repository files for an implementation task, create or claim a Backlog.md task for that reviewable unit, link this plan and `TASK-12969`, keep its status and verification current, and commit its task record with the code.
- Rebase on the latest `origin/dev` before implementation and repeat the rebase before PR review.
- Preserve the public signatures and sync/async classification of `gather_analysis`, `run_analysis`, and all nine analyzer entry points.
- Preserve analyzer result keys in this exact order: `robots`, `tls`, `js`, `behavioral`, `captcha`, `fingerprint`, `integrity`, `rate_limit`, `waf`.
- Preserve successful analyzer values, score cards, recommendations, advice rules, config keys, scrape signatures, and optional public payload shape.
- Primary scrape policy denial remains blocking; analyzer and overall preflight failures remain advisory and extraction continues unchanged.
- Caller cancellation always propagates. It must not be converted to an analyzer error, timeout, or scrape result.
- All analyzer HTTP, browser, and external-tool work goes through injected governed adapters. Analyzer modules may not import `http_client`, Playwright, curl-cffi, `subprocess`, or concrete policy implementations.
- HTTP dispatches receive a fresh probe-egress decision and central dispatch-time egress validation. Browser and external-tool checks are URL-level only and must not claim DNS pinning.
- Playwright is `>=1.48.0` in the base, `web_research`, and `scrape-analyzers` dependency groups. Service workers are blocked and HTTP/WebSocket routes are installed before page creation.
- Overall deadlines use one monotonic timestamp. Cleanup has one shared two-second shielded grace period and preserves the original cancellation or timeout outcome.
- Request, browser, and active-probe limits default to `None`; counters remain atomic and testable.
- Missing external-tool config keeps enabled-when-installed behavior for Phase 3. Explicit true/false is authoritative; malformed explicit values disable. The legacy fallback warning and bounded metric fire once per process.
- Required tests use deterministic fakes and never access the public network, launch a real browser, or execute a real external tool. The local-browser smoke test remains optional and separately marked.
- Activate `.venv` before every Python, pytest, formatter, linter, or Bandit command.
- A step named "Write failing" is complete only after running that task's exact focused pytest command against the newly added test module and observing failure for the intended missing contract. Record that red result in the task's Backlog notes before any production edit; import/collection failures count only when the task intentionally introduces a not-yet-created module or symbol. The later focused-test step is the green gate and must exit zero before refactoring or commit.

---

## File Map

### Core and adapters

- Create `tldw_Server_API/app/core/Web_Scraping/preflight/__init__.py`: facade exports only.
- Create `tldw_Server_API/app/core/Web_Scraping/preflight/options.py`: the sole production parser for existing preflight config.
- Create `tldw_Server_API/app/core/Web_Scraping/preflight/target.py`: immutable URL-bound `PreflightTarget`.
- Create `tldw_Server_API/app/core/Web_Scraping/preflight/probes.py`: analyzer-facing probe data contracts, protocols, and safe probe exceptions.
- Create `tldw_Server_API/app/core/Web_Scraping/preflight/context.py`: deadline, atomic budgets, deterministic identity, cleanup stack, and execution context.
- Create `tldw_Server_API/app/core/Web_Scraping/preflight/facade.py`: target evaluation, context factory, typed runner boundary, advice application, and payload eligibility.
- Create `tldw_Server_API/app/core/Web_Scraping/preflight/compatibility.py`: background-loop bridge and legacy-call helpers.
- Create `tldw_Server_API/app/core/Web_Scraping/preflight/adapters/__init__.py`: concrete probe adapter exports.
- Create `tldw_Server_API/app/core/Web_Scraping/preflight/adapters/http.py`: explicit redirects, budgets, timeout caps, credential stripping, and async transports.
- Create `tldw_Server_API/app/core/Web_Scraping/preflight/adapters/browser.py`: guarded async Playwright lifecycle and analyzer-facing page wrapper.
- Create `tldw_Server_API/app/core/Web_Scraping/preflight/adapters/external_tools.py`: injected wafw00f discovery/execution, transition observability, and process cleanup.
- Modify `tldw_Server_API/app/core/Web_Scraping/runtime/policy.py`: add protocol-only probe-egress contracts.
- Modify `tldw_Server_API/app/core/Web_Scraping/runtime/browser.py`: add only the route/page protocol hooks required by the guarded adapter.
- Modify `tldw_Server_API/app/core/Web_Scraping/runtime/__init__.py`: export new runtime contracts.
- Modify `tldw_Server_API/app/core/Web_Scraping/policy/adapters.py`: add centralized probe-egress adapter.
- Modify `tldw_Server_API/app/core/Web_Scraping/policy/__init__.py`: export the concrete adapter.
- Modify `pyproject.toml`: raise all three Playwright floors to `>=1.48.0`.

### Analyzer ownership and compatibility

- Move implementation from `scraper_analyzers/analyzers/` to `preflight/analyzers/`, retaining the nine module names.
- Create `tldw_Server_API/app/core/Web_Scraping/preflight/analyzers/_shared.py`: safe probe-error conversion used by canonical analyzers only.
- Move implementation from `scraper_analyzers/scoring/`, `recommendations/`, and `utils/` to matching `preflight/` subpackages.
- Create `tldw_Server_API/app/core/Web_Scraping/preflight/runner.py`: deterministic internal async runner.
- Replace every old `scraper_analyzers` implementation module with an explicit re-export shim and explicit `__all__`.

### Consumers and documentation

- Modify `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`: replace duplicated preflight orchestration with facade calls.
- Modify `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py`: use the same facade and policy adapter.
- Modify `tldw_Server_API/app/core/Web_Scraping/README.md`: document canonical imports, compatibility shims, and existing config behavior.
- Regenerate `Docs/Design/WebScraping_Refactor_Import_Inventory.md` and `Docs/Design/web_scraping_refactor_import_inventory.json`.

### Tests

- Create `tldw_Server_API/tests/Web_Scraping/preflight_fakes.py`: deterministic policy, HTTP, browser, tool, clock, and cleanup fakes.
- Create `test_phase3_preflight_characterization.py`, `test_phase3_preflight_contracts.py`, `test_phase3_probe_egress.py`, `test_phase3_preflight_http.py`, `test_phase3_preflight_browser.py`, `test_phase3_preflight_external_tools.py`, `test_phase3_preflight_nonbrowser_analyzers.py`, `test_phase3_preflight_browser_analyzers.py`, `test_phase3_preflight_runner_facade.py`, `test_phase3_preflight_compatibility.py`, `test_phase3_article_preflight_facade.py`, `test_phase3_enhanced_preflight_facade.py`, and `test_phase3_preflight_architecture.py` under `tldw_Server_API/tests/Web_Scraping/`.
- Create `tldw_Server_API/tests/WebScraping/integration/test_phase3_preflight_browser_smoke.py`: optional local-only browser smoke test.
- Update existing Phase 1/2 compatibility tests that patch private duplicated consumer orchestration so they patch the shared facade instead.

---

### Task 0: Refresh the Branch and Establish Implementation Tracking

**Files:**
- No production edits.
- Backlog.md task records created through the official MCP workflow.

**Interfaces:**
- Consumes: approved design and this plan.
- Produces: clean latest-dev branch and one implementation parent task with child tasks matching Tasks 1-14.

- [ ] **Step 1: Fetch and rebase before any implementation edit**

```bash
git fetch origin
git rebase origin/dev
git status --short --branch
```

Expected: the branch is based on latest `origin/dev`; the worktree is clean. Resolve conflicts by preserving the approved design and rechecking current production behavior, never by discarding newer `dev` changes.

- [ ] **Step 2: Create the implementation Backlog parent and child records**

Use the Backlog MCP task-creation workflow. The parent title is `Implement Web_Scraping Phase 3 governed preflight package`; it references this plan, the approved design, and `TASK-12969`. Create one child for each independently committed Task 1-14 before that task edits files.

Expected: each child has its exact touched files, acceptance criteria, current status, and later verification/PR notes.

- [ ] **Step 3: Record current public analyzer signatures and dependency floors**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -c "import inspect; from tldw_Server_API.app.core.Web_Scraping.scraper_analyzers import runner; print(inspect.signature(runner.gather_analysis)); print(inspect.signature(runner.run_analysis))"
rg -n "playwright>=" pyproject.toml
rg -n "^(async )?def (check_robots_txt|analyze_tls_fingerprint|analyze_js_rendering|detect_honeypots|detect_captcha|analyze_fingerprinting|analyze_function_integrity|profile_rate_limits|detect_waf)" tldw_Server_API/app/core/Web_Scraping/scraper_analyzers/analyzers
```

Expected: signatures match the approved inventory and all three Playwright floors are still discoverable for Task 5.

---

### Task 1: Pin Current Analyzer, Scoring, Recommendation, and Advice Behavior

**Files:**
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_characterization.py`
- Modify: the Task 1 Backlog child record.

**Interfaces:**
- Consumes: current `scraper_analyzers.runner`, scoring engine, recommender, and consumer advice behavior.
- Produces: fixed expected signatures, analyzer order, successful result shapes, score/recommendation values, and advice semantics used by all later tasks.

- [ ] **Step 1: Add signature and runner-order characterization tests**

Create tests with the exact historical signature strings and deterministic runner replacements:

```python
EXPECTED_SIGNATURES = {
    "check_robots_txt": "(url: 'str') -> 'dict[str, Any]'",
    "analyze_tls_fingerprint": "(url: 'str') -> 'dict[str, Any]'",
    "analyze_js_rendering": "(url: 'str') -> 'dict[str, Any]'",
    "detect_honeypots": "(url: 'str', scan_depth: 'ScanDepth' = 'default') -> 'dict[str, Any]'",
    "detect_captcha": "(url: 'str') -> 'dict[str, Any]'",
    "analyze_fingerprinting": "(url: 'str') -> 'dict[str, Any]'",
    "analyze_function_integrity": "(url: 'str') -> 'dict[str, Any]'",
    "profile_rate_limits": "(url: 'str', crawl_delay: 'float | None', impersonate: 'bool' = False) -> 'dict[str, Any]'",
    "detect_waf": "(url: 'str', find_all: 'bool' = False) -> 'dict[str, Any]'",
}

@pytest.mark.asyncio
async def test_gather_analysis_preserves_order_and_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    def sync_result(name: str, payload: dict[str, Any]):
        def call(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            events.append(name)
            return payload
        return call

    async def async_result(name: str, payload: dict[str, Any]) -> dict[str, Any]:
        events.append(name)
        return payload

    monkeypatch.setattr(runner, "check_robots_txt", sync_result("robots", {"status": "success", "crawl_delay": 0.0}))
    monkeypatch.setattr(runner, "analyze_tls_fingerprint", lambda _url: async_result("tls", {"status": "inactive"}))
    monkeypatch.setattr(runner, "analyze_js_rendering", sync_result("js", {"status": "success", "js_required": False, "is_spa": False, "content_difference_%": 0.0}))
    monkeypatch.setattr(runner, "detect_honeypots", sync_result("behavioral", {"status": "success", "honeypot_detected": False}))
    monkeypatch.setattr(runner, "detect_captcha", sync_result("captcha", {"status": "success", "captcha_detected": False}))
    monkeypatch.setattr(runner, "analyze_fingerprinting", sync_result("fingerprint", {"status": "success", "detected_services": []}))
    monkeypatch.setattr(runner, "analyze_function_integrity", sync_result("integrity", {"status": "success", "modified_functions": {}}))
    monkeypatch.setattr(runner, "profile_rate_limits", lambda *_args, **_kwargs: async_result("rate_limit", {"status": "success", "results": {"requests_sent": 12}}))
    monkeypatch.setattr(runner, "detect_waf", sync_result("waf", {"status": "success", "wafs": []}))

    result = await runner.gather_analysis("https://example.com", find_all=True, impersonate=True, scan_depth="deep")

    assert events == ["robots", "tls", "js", "behavioral", "captcha", "fingerprint", "integrity", "rate_limit", "waf"]
    assert list(result) == ["results", "score", "recommendations"]
    assert list(result["results"]) == events
```

- [ ] **Step 2: Pin successful scoring, recommendations, advice, and payload values**

Add this fixed input covering WAF, TLS, JS, honeypot, CAPTCHA, fingerprint, integrity, and rate limits. Assert the complete score, recommendations, and current article/enhanced final advice dictionaries rather than partial thresholds.

```python
analysis_results = {
    "js": {"status": "success", "js_required": True, "is_spa": True},
    "tls": {"status": "active"},
    "captcha": {"status": "success", "captcha_detected": True, "trigger_condition": "on page load"},
    "behavioral": {"status": "success", "honeypot_detected": True},
    "rate_limit": {"status": "success", "results": {"requests_sent": 3, "blocking_code": 429}},
    "waf": {"status": "success", "wafs": [("DataDome", None)]},
    "fingerprint": {
        "status": "success",
        "detected_services": ["DataDome"],
        "canvas_fingerprinting_signal": True,
        "behavioral_listeners_detected": ["mousemove"],
    },
    "integrity": {
        "status": "success",
        "modified_functions": {
            "HTMLCanvasElement.prototype.toDataURL": "patched",
            "Date.now": "patched",
        },
    },
}
score = calculate_difficulty_score(analysis_results)
recommendations = generate_recommendations(analysis_results)
assert score == {"score": 10, "label": "Very Hard"}
assert recommendations == {
    "tools": [
        "A CAPTCHA solving service (e.g. 2Captcha, Anti-Captcha).",
        "A headless browser such as Playwright or Selenium for JavaScript rendering.",
        "A library with browser impersonation (e.g. curl_cffi) or a full headless browser.",
        "A pool of high-quality rotating proxies (residential or mobile).",
        "An anti-detection browser automation library (e.g. playwright-stealth, undetected-chromedriver).",
    ],
    "strategy": [
        "Add delays between requests (3-5 seconds) and rotate request headers.",
        "Add random delays and jitter between actions to appear more human.",
        "Avoid interacting with invisible elements; drive the page like a human.",
        "Canvas fingerprinting detected. Use automation with built-in evasion (not basic requests).",
        "Integrate the CAPTCHA solver when challenges appear.",
        "Site modifies canvas functions (strong fingerprinting). Avoid basic automation.",
        "Site monitors timing patterns. Vary your request timing to look less robotic.",
        "Site monitors user behavior (mouse, keyboard, scroll). Simulate realistic interaction.",
        "Site uses advanced bot detection (DataDome). Use playwright-stealth or undetected-chromedriver.",
        "Standard Python HTTP clients are blocked; impersonate a real browser.",
        "Use a modern, non-generic User-Agent and align headers with real browsers.",
        "Wait for dynamic content to load before extracting data.",
    ],
}
analysis = {"results": analysis_results, "score": score, "recommendations": recommendations}
expected_payload = {
    "analysis": analysis,
    "advice": {"backend": "curl", "method": "playwright", "notes": ["js_required", "tls_active"]},
}
```

Invoke the existing fake article and enhanced consumer paths with `analysis` and assert each returned `preflight_analysis` equals `expected_payload`.

- [ ] **Step 3: Run the baseline characterization suite**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_characterization.py tldw_Server_API/tests/WebScraping/test_scraper_analyzer_sanitizers.py tldw_Server_API/tests/WebScraping/test_scraping_module.py
```

Expected: PASS against the pre-move implementation. If a proposed literal does not match, inspect current behavior and correct the test before implementation; do not alter production in this task.

- [ ] **Step 4: Commit the characterization gate**

```bash
git add tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_characterization.py backlog/tasks
git commit -m "test: characterize governed preflight behavior"
```

---

### Task 2: Add Options, Probe Contracts, Deadlines, Budgets, and Execution Context

**Files:**
- Create: `tldw_Server_API/app/core/Web_Scraping/preflight/__init__.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/preflight/options.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/preflight/target.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/preflight/probes.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/preflight/context.py`
- Create: `tldw_Server_API/tests/Web_Scraping/preflight_fakes.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_contracts.py`
- Modify: the Task 2 Backlog child record.

**Interfaces:**
- Consumes: `PolicyDecision`, `RuntimeRequestContext`, `PreflightResult`, `PreflightAdvice`.
- Produces: `PreflightOptions.from_mapping`, `PreflightTarget`, `ProbeHttpRequest`, `ProbeHttpResponse`, `BrowserProbeOptions`, `BrowserProbePage`, `BrowserProbe`, `ExternalToolResult`, `ExternalToolProbe`, safe `ProbeError` subclasses, `PreflightLimits`, `PreflightRuntimeControls`, and `PreflightExecutionContext`.

- [ ] **Step 1: Write failing option-normalization and property tests**

Cover every approved config key with absent, valid, and malformed values. Include Hypothesis invariants for arbitrary timeout and boolean-like values:

```python
@pytest.mark.unit
def test_options_preserve_legacy_defaults() -> None:
    options = PreflightOptions.from_mapping({})
    assert options == PreflightOptions(
        enabled=False,
        timeout_s=None,
        scan_depth="default",
        find_all_waf=False,
        impersonate=False,
        include_results=False,
        external_tools_enabled=None,
        playwright_no_sandbox=False,
    )

@pytest.mark.property
@given(st.one_of(st.none(), st.floats(allow_nan=True, allow_infinity=True), st.text(), st.booleans()))
def test_timeout_is_none_or_positive_finite(value: object) -> None:
    options = PreflightOptions.from_mapping({"web_scraper_preflight_timeout_s": value})
    assert options.timeout_s is None or (math.isfinite(options.timeout_s) and options.timeout_s > 0)
```

Assert absent external-tool config remains `None`, explicit booleans normalize, and malformed explicit values become `False` with one sanitized warning that contains neither the supplied value nor URL-like text.

- [ ] **Step 2: Write failing budget, deadline, identity, and cleanup tests**

Use a fake monotonic clock and concurrent reservations:

```python
@pytest.mark.asyncio
async def test_request_budget_reservation_is_atomic() -> None:
    controls = PreflightRuntimeControls(
        request_context=RuntimeRequestContext(source="test", stage="preflight"),
        limits=PreflightLimits(requests=8),
        clock=lambda: 10.0,
    )
    outcomes = await asyncio.gather(*(controls.reserve("request") for _ in range(12)), return_exceptions=True)
    assert sum(result is None for result in outcomes) == 8
    assert sum(isinstance(result, ProbeBudgetExhausted) for result in outcomes) == 4
    assert controls.consumed.requests == 8

@pytest.mark.asyncio
async def test_cleanup_uses_one_shared_grace_and_preserves_cancellation() -> None:
    cleanup = FakeCleanupHandle(block_close=True)
    controls.register_cleanup(cleanup)
    task = asyncio.create_task(controls.close(grace_s=2.0))
    await cleanup.close_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleanup.force_close_calls == 1
```

Also assert local `ProbeTimeout` remains analyzer-scoped, `PreflightDeadlineExceeded` is distinct and non-normalizable, observed caller cancellation wins a deadline race, one identity is cached per context, and browser/active-probe counters obey the same invariant.

- [ ] **Step 3: Implement `PreflightOptions` and immutable target/probe contracts**

Use these exact public fields:

```python
ScanDepth = Literal["default", "thorough", "deep"]

@dataclass(frozen=True, slots=True)
class PreflightOptions:
    enabled: bool = False
    timeout_s: float | None = None
    scan_depth: ScanDepth = "default"
    find_all_waf: bool = False
    impersonate: bool = False
    include_results: bool = False
    external_tools_enabled: bool | None = None
    playwright_no_sandbox: bool = False

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "PreflightOptions":
        values = dict(config or {})
        external = values.get("web_scraper_preflight_enable_external_tools", _ABSENT)
        return cls(
            enabled=_legacy_bool(values.get("web_scraper_preflight_analyzers"), False),
            timeout_s=_positive_timeout(values.get("web_scraper_preflight_timeout_s")),
            scan_depth=_scan_depth(values.get("web_scraper_preflight_scan_depth")),
            find_all_waf=_legacy_bool(values.get("web_scraper_preflight_find_all_waf"), False),
            impersonate=_legacy_bool(values.get("web_scraper_preflight_impersonate"), False),
            include_results=_legacy_bool(values.get("web_scraper_preflight_include_results"), False),
            external_tools_enabled=None if external is _ABSENT else _explicit_external_bool(external),
            playwright_no_sandbox=_legacy_bool(values.get("web_scraper_playwright_no_sandbox"), False),
        )
```

`PreflightTarget` has exactly `url`, `decision`, and `request_context`; it rejects a blank URL. Probe exceptions expose only stable `error_code` and `public_message`. Use codes `policy_denied`, `policy_error`, `budget_exhausted`, `timeout`, `unavailable`, `missing_dependency`, `external_tool_disabled`, `redirect_loop`, `invalid_redirect`, `too_many_redirects`, and `probe_error`. Define internal `PreflightDeadlineExceeded` separately from `ProbeTimeout`; analyzers may normalize only the latter.

Define the cross-task probe interfaces exactly once in `probes.py`:

```python
@dataclass(frozen=True, slots=True)
class ProbeHttpRequest:
    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    cookies: Mapping[str, str] = field(default_factory=dict)
    timeout_s: float | None = None
    impersonate: str | None = None
    proxies: Mapping[str, str] | str | None = None
    allow_redirects: bool = True

@dataclass(frozen=True, slots=True)
class ProbeHttpResponse:
    url: str
    status: int
    headers: Mapping[str, str] = field(default_factory=dict)
    text: str = ""

@dataclass(frozen=True, slots=True)
class BrowserProbeOptions:
    user_agent: str | None = None
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    viewport_width: int = 1280
    viewport_height: int = 720
    block_resource_types: tuple[str, ...] = ()
    init_scripts: tuple[str, ...] = ()
    capture_requests: bool = False

@dataclass(frozen=True, slots=True)
class ExternalToolResult:
    returncode: int
    stdout: str
    stderr: str

class HttpProbe(Protocol):
    async def get(self, request: ProbeHttpRequest) -> ProbeHttpResponse:
        raise NotImplementedError

class BrowserProbePage(Protocol):
    async def goto(self, url: str, *, wait_until: str, timeout_ms: float) -> None:
        raise NotImplementedError
    async def reload(self, *, wait_until: str, timeout_ms: float) -> None:
        raise NotImplementedError
    async def wait_for_load_state(self, state: str, *, timeout_ms: float) -> None:
        raise NotImplementedError
    async def wait_for_timeout(self, timeout_ms: float) -> None:
        raise NotImplementedError
    async def content(self) -> str:
        raise NotImplementedError
    async def evaluate(self, expression: str, argument: Any = None) -> Any:
        raise NotImplementedError
    async def link_count(self) -> int:
        raise NotImplementedError
    async def link_is_visible(self, index: int) -> bool:
        raise NotImplementedError
    def captured_request_urls(self) -> tuple[str, ...]:
        raise NotImplementedError
    def clear_captured_request_urls(self) -> None:
        raise NotImplementedError

class BrowserProbe(Protocol):
    def open_page(self, options: BrowserProbeOptions) -> AsyncContextManager[BrowserProbePage]:
        raise NotImplementedError

class ExternalToolProbe(Protocol):
    async def run_waf(
        self,
        url: str,
        *,
        find_all: bool,
        enabled: bool | None,
    ) -> ExternalToolResult:
        raise NotImplementedError
```

Normalize/freeze all mappings and sequences in `__post_init__`; reject blank URLs, nonpositive explicit timeouts, and invalid viewport dimensions. `ProbeError.__init__(error_code, public_message)` stores only those two safe strings. Specialized subclasses provide fixed defaults but accept no raw exception text.

- [ ] **Step 4: Implement controls and context with explicit dependency injection**

Use these exact boundaries so adapters never depend on the facade or analyzers:

```python
BudgetKind = Literal["request", "browser", "active_probe"]

@dataclass(frozen=True, slots=True)
class PreflightLimits:
    requests: int | None = None
    browsers: int | None = None
    active_probes: int | None = None

@dataclass(frozen=True, slots=True)
class PreflightConsumed:
    requests: int = 0
    browsers: int = 0
    active_probes: int = 0

class PreflightRuntimeControls:
    async def reserve(self, kind: BudgetKind, amount: int = 1) -> None:
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 1:
            raise ValueError("amount must be a positive integer")
        async with self._budget_lock:
            field = {"request": "requests", "browser": "browsers", "active_probe": "active_probes"}[kind]
            current = getattr(self._consumed, field)
            limit = getattr(self.limits, field)
            if limit is not None and current + amount > limit:
                raise ProbeBudgetExhausted()
            self._consumed = replace(self._consumed, **{field: current + amount})

    def remaining_seconds(self) -> float | None:
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - self._clock())

    def cap_timeout(self, requested_s: float | None) -> float | None:
        remaining = self.remaining_seconds()
        values = [value for value in (requested_s, remaining) if value is not None]
        if values and min(values) <= 0:
            raise PreflightDeadlineExceeded()
        return min(values) if values else None

    def deadline_exhausted(self) -> bool:
        remaining = self.remaining_seconds()
        return remaining is not None and remaining <= 0

    async def sleep(self, delay_s: float) -> None:
        effective = self.cap_timeout(delay_s)
        if effective is None:
            effective = delay_s
        await self._sleep(effective)
        if effective < delay_s:
            raise PreflightDeadlineExceeded()
```

`PreflightLimits.__post_init__` rejects booleans, negative values, and nonintegral values; `None` is unbounded and zero permits no reservations. Inject `_sleep` with `asyncio.sleep` by default and a deterministic fake in tests. Adapter timeout handlers check `deadline_exhausted()`: true raises `PreflightDeadlineExceeded`, false raises analyzer-scoped `ProbeTimeout`. `PreflightExecutionContext` contains `request_context`, `policy_checker`, `egress_guard`, `controls`, `http`, `browser`, `external_tools`, and `identity_selector`. Its `browser_identity()` copies and caches one selection. Its `close()` delegates to the controls cleanup stack, which creates one shielded cleanup task, waits at most two seconds total, force-closes remaining handles, and re-raises any pending caller cancellation after cleanup. Cleanup errors are logged through sanitized labels and never replace an existing timeout/cancellation.

- [ ] **Step 5: Run focused contract tests**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_contracts.py tldw_Server_API/tests/Web_Scraping/test_phase1_contracts.py
```

Expected: PASS, including property tests and concurrent reservations.

- [ ] **Step 6: Commit the context boundary**

```bash
git add tldw_Server_API/app/core/Web_Scraping/preflight tldw_Server_API/tests/Web_Scraping/preflight_fakes.py tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_contracts.py backlog/tasks
git commit -m "feat: add preflight execution contracts"
```

---

### Task 3: Separate Scrape Policy from Probe Egress

**Files:**
- Modify: `tldw_Server_API/app/core/Web_Scraping/runtime/policy.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/runtime/__init__.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/policy/adapters.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/policy/__init__.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/preflight/facade.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase3_probe_egress.py`
- Modify: the Task 3 Backlog child record.

**Interfaces:**
- Consumes: central synchronous `Security.egress.evaluate_url_policy`, scrape-level `OutboundPolicyChecker`, and Task 2 target/context contracts.
- Produces: immutable `ProbeEgressDecision`, protocol `ProbeEgressGuard.decide(url, *, context)`, concrete `DefaultProbeEgressGuard`, and async `evaluate_target(...) -> PreflightTarget`.

- [ ] **Step 1: Write failing protocol and adapter tests**

```python
@pytest.mark.asyncio
async def test_probe_guard_delegates_without_robots_and_sanitizes_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        egress,
        "evaluate_url_policy",
        lambda url: calls.append(url) or URLPolicyResult(True, None, ("93.184.216.34",)),
    )
    decision = await DefaultProbeEgressGuard().decide(
        "https://example.com/private?token=secret",
        context=RuntimeRequestContext(source="preflight", stage="preflight_subrequest"),
    )
    assert calls == ["https://example.com/private?token=secret"]
    assert decision == ProbeEgressDecision(allowed=True, reason="allowed", resolved_ips=("93.184.216.34",))

@pytest.mark.asyncio
async def test_evaluate_target_uses_scrape_policy_once() -> None:
    checker = FakePolicyChecker(allowed=True)
    target = await evaluate_target(
        "https://example.com/article",
        respect_robots=True,
        user_agent="UA",
        request_context=RuntimeRequestContext(source="article_extract", stage="pre_fetch"),
        config={"web_scraper": {}},
        policy_checker=checker,
    )
    assert len(checker.calls) == 1
    assert target.decision.allowed is True
```

Add denial, policy-evaluator exception, immutable resolved-IP, no-robots-import, and sanitized-log assertions.

- [ ] **Step 2: Implement the narrow runtime protocol and concrete adapter**

```python
@dataclass(frozen=True, slots=True)
class ProbeEgressDecision:
    allowed: bool
    reason: str
    resolved_ips: tuple[str, ...] = ()

class ProbeEgressGuard(Protocol):
    async def decide(self, url: str, *, context: RuntimeRequestContext) -> ProbeEgressDecision:
        raise NotImplementedError

class DefaultProbeEgressGuard:
    async def decide(self, url: str, *, context: RuntimeRequestContext) -> ProbeEgressDecision:
        try:
            raw = await asyncio.to_thread(egress_policy.evaluate_url_policy, url)
        except asyncio.CancelledError:
            raise
        except Exception:
            return ProbeEgressDecision(allowed=False, reason="policy_error")
        return ProbeEgressDecision(
            allowed=bool(raw.allowed),
            reason=_bounded_reason(raw.reason),
            resolved_ips=tuple(raw.resolved_ips or ()),
        )
```

`_bounded_reason` maps known central reasons to stable labels and all unknown values to `other`; logs contain only source/stage and a sanitized host label. Evaluator failure returns a denied `policy_error` decision. The runtime module remains protocol-only and does not import `Security.egress`, `preflight`, or `policy.adapters`.

- [ ] **Step 3: Implement target evaluation through the scrape checker only**

```python
async def evaluate_target(
    url: str,
    *,
    respect_robots: bool,
    user_agent: str | None,
    request_context: RuntimeRequestContext,
    config: Mapping[str, Any] | None,
    policy_checker: OutboundPolicyChecker,
) -> PreflightTarget:
    decision = await policy_checker.decide(
        url,
        respect_robots=respect_robots,
        user_agent=user_agent,
        context=request_context,
        config=config,
    )
    return PreflightTarget(url=url, decision=decision, request_context=request_context)
```

Do not call the probe guard here. `evaluate_target` owns scrape-level egress plus optional robots; every actual adapter dispatch performs the separate fresh probe check later.

- [ ] **Step 4: Run policy boundary tests**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase3_probe_egress.py tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_adapters.py tldw_Server_API/tests/Web_Scraping/test_outbound_policy.py
```

Expected: PASS with no network because all central evaluation and policy calls are patched.

- [ ] **Step 5: Commit the policy split**

```bash
git add tldw_Server_API/app/core/Web_Scraping/runtime tldw_Server_API/app/core/Web_Scraping/policy tldw_Server_API/app/core/Web_Scraping/preflight/facade.py tldw_Server_API/tests/Web_Scraping/test_phase3_probe_egress.py backlog/tasks
git commit -m "feat: separate preflight probe egress policy"
```

---

### Task 4: Add the Governed Async HTTP Probe Adapter

**Files:**
- Create: `tldw_Server_API/app/core/Web_Scraping/preflight/adapters/__init__.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/preflight/adapters/http.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_http.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/preflight_fakes.py`
- Modify: the Task 4 Backlog child record.

**Interfaces:**
- Consumes: `ProbeHttpRequest`, `ProbeHttpResponse`, `ProbeEgressGuard`, `PreflightRuntimeControls`, central `http_client.DEFAULT_MAX_REDIRECTS`, and central `http_client.afetch`.
- Produces: `GuardedHttpProbe.get(request) -> ProbeHttpResponse`, `HttpxProbeTransport`, and `CurlCffiProbeTransport`.

- [ ] **Step 1: Write failing dispatch, redirect, and cleanup tests**

Use an injected queue transport whose responses expose `status_code`, `headers`, `text`, `url`, `aclose`, and call capture:

```python
@pytest.mark.asyncio
async def test_http_probe_checks_every_dispatch_and_closes_responses() -> None:
    guard = FakeProbeEgressGuard([True, True])
    transport = FakeHttpTransport([
        FakeRawResponse(302, headers={"Location": "/next"}),
        FakeRawResponse(200, text="ok"),
    ])
    probe = GuardedHttpProbe(controls=controls(), egress_guard=guard, transport=transport)

    response = await probe.get(ProbeHttpRequest(url="https://example.com/start", timeout_s=20.0))

    assert response.url == "https://example.com/next"
    assert [call.url for call in transport.calls] == ["https://example.com/start", "https://example.com/next"]
    assert guard.urls == ["https://example.com/start", "https://example.com/next"]
    assert all(raw.closed for raw in transport.responses)
```

Add cases for relative and absolute locations, loops, missing/invalid locations, maximum hops, scheme downgrade, denied/private redirects, `policy_error`, request-budget exhaustion before transport, deadline timeout, caller cancellation, response-close exceptions, and central transport exceptions. Assert no denied URL reaches the transport and central retries are disabled so one budget reservation always corresponds to one transport attempt.

- [ ] **Step 2: Write cross-origin credential and impersonated-transport tests**

```python
@pytest.mark.asyncio
async def test_cross_origin_redirect_strips_all_sensitive_credentials() -> None:
    request = ProbeHttpRequest(
        url="https://a.example/start",
        headers={
            "Authorization": "Bearer secret",
            "Proxy-Authorization": "Basic secret",
            "Cookie": "header-secret",
            "X-API-Key": "secret",
            "API-Key": "secret",
            "X-Auth-Token": "secret",
            "Accept": "text/html",
        },
        cookies={"session": "secret"},
    )
    await probe.get(request)
    second = transport.calls[1]
    assert second.headers == {"Accept": "text/html"}
    assert second.cookies == {}
```

For `impersonate="chrome120"`, assert `CurlCffiProbeTransport` is selected, performs an immediate second egress evaluation before `session.get`, disables redirects, and closes the async session/response. Missing curl-cffi produces `ProbeUnavailable(error_code="missing_dependency")` without transport work.

- [ ] **Step 3: Implement explicit-hop governance**

The main loop follows this exact order for each dispatch:

```python
async def get(self, request: ProbeHttpRequest) -> ProbeHttpResponse:
    original_url = request.url
    current_url = request.url
    current_headers = dict(request.headers)
    current_cookies = dict(request.cookies)
    visited: set[str] = set()

    for hop in range(DEFAULT_MAX_REDIRECTS + 1):
        if current_url in visited:
            raise ProbeError("redirect_loop", "Redirect loop detected.")
        visited.add(current_url)
        await self._controls.reserve("request")
        decision = await self._egress_guard.decide(current_url, context=self._subrequest_context())
        if not decision.allowed:
            code = "policy_error" if decision.reason == "policy_error" else "policy_denied"
            raise ProbeError(code, "Probe destination was denied.")
        timeout_s = self._controls.cap_timeout(request.timeout_s)
        raw = await self._transport_for(request).send(
            replace(
                request,
                url=current_url,
                headers=current_headers,
                cookies=current_cookies,
                timeout_s=timeout_s,
                allow_redirects=False,
            )
        )
        try:
            response = await _snapshot_response(raw, fallback_url=current_url)
        finally:
            await _close_response(raw)
        if not request.allow_redirects:
            return response
        location = _redirect_location(response)
        if location is None:
            return response
        if hop == DEFAULT_MAX_REDIRECTS:
            raise ProbeError("too_many_redirects", "Redirect limit exceeded.")
        next_url = _resolve_redirect(current_url, location)
        current_headers, current_cookies = _credentials_for_hop(
            current_headers,
            current_cookies,
            original_url=original_url,
            target_url=next_url,
        )
        current_url = next_url
    raise ProbeError("too_many_redirects", "Redirect limit exceeded.")
```

`_credentials_for_hop` compares normalized `(scheme, host, port)` tuples and drops the same header set as `http_client.SENSITIVE_REDIRECT_HEADERS` plus all explicit cookies on cross-origin hops. It never logs header values.

- [ ] **Step 4: Implement native async transports**

`HttpxProbeTransport.send` delegates to `http_client.afetch(method="GET", allow_redirects=False, ...)`; that boundary performs the required second central egress validation immediately before HTTP transport. `CurlCffiProbeTransport.send` uses an injected async-session factory, repeats `ProbeEgressGuard.decide` immediately before `session.get`, passes `allow_redirects=False`, and never uses `asyncio.to_thread`.

```python
raw = await http_client.afetch(
    method="GET",
    url=request.url,
    headers=dict(request.headers),
    cookies=dict(request.cookies),
    timeout=request.timeout_s,
    allow_redirects=False,
    proxies=request.proxies,
    retry=http_client.RetryPolicy(attempts=1),
)
```

Propagate `CancelledError`. On transport timeout, raise `PreflightDeadlineExceeded` when `controls.deadline_exhausted()` is true and analyzer-scoped `ProbeTimeout` otherwise. Map all other transport failures to safe `ProbeError("probe_error", "HTTP probe failed.")` without raw exception text. The probe adapter owns retries by making none in Phase 3; analyzers already define their intended request counts explicitly.

- [ ] **Step 5: Run HTTP tests and central-client regressions**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_http.py tldw_Server_API/tests/Web_Scraping/test_http_client_fetch.py
```

Expected: PASS with exact dispatch counts, closure assertions, and no external network.

- [ ] **Step 6: Commit the HTTP adapter**

```bash
git add tldw_Server_API/app/core/Web_Scraping/preflight/adapters tldw_Server_API/tests/Web_Scraping/preflight_fakes.py tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_http.py backlog/tasks
git commit -m "feat: add governed preflight HTTP probes"
```

---

### Task 5: Add the Guarded Async Browser Adapter and Playwright Floor

**Files:**
- Modify: `pyproject.toml`
- Modify: `tldw_Server_API/app/core/Web_Scraping/runtime/browser.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/runtime/__init__.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/preflight/adapters/browser.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_browser.py`
- Create: `tldw_Server_API/tests/WebScraping/integration/test_phase3_preflight_browser_smoke.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/preflight_fakes.py`
- Modify: the Task 5 Backlog child record.

**Interfaces:**
- Consumes: Task 2 `BrowserProbe`/`BrowserProbePage`, Task 3 guard, controls, and async Playwright.
- Produces: `GuardedPlaywrightBrowserProbe.open_page(options)`, guarded page wrapper methods, and runtime route protocols.

- [ ] **Step 1: Write failing capability and ordering tests**

```python
@pytest.mark.asyncio
async def test_browser_routes_before_page_and_blocks_service_workers() -> None:
    launcher = FakePlaywrightLauncher()
    probe = GuardedPlaywrightBrowserProbe(
        controls=controls(),
        egress_guard=FakeProbeEgressGuard([True]),
        launcher=launcher,
        capability_check=lambda: True,
    )
    async with probe.open_page(BrowserProbeOptions(user_agent="UA")):
        pass
    assert launcher.events[:4] == [
        "launch",
        "new_context:service_workers=block",
        "route_http",
        "route_web_socket",
    ]
    assert launcher.events[4] == "new_page"

@pytest.mark.asyncio
async def test_missing_websocket_capability_returns_unavailable_without_launch() -> None:
    launcher = FakePlaywrightLauncher()
    probe = GuardedPlaywrightBrowserProbe(
        controls=controls(),
        egress_guard=FakeProbeEgressGuard([]),
        launcher=launcher,
        capability_check=lambda: False,
    )
    with pytest.raises(ProbeUnavailable) as raised:
        async with probe.open_page(BrowserProbeOptions()):
            pytest.fail("page must not be created")
    assert raised.value.error_code == "unavailable"
    assert launcher.events == []
```

Add HTTP route allow/deny/error, subresource, redirect, blocked resource type, WebSocket allow/deny/error, browser budget, deadline cap, `--no-sandbox`, close ordering, cancellation, two-second force cleanup, and redacted log tests.

- [ ] **Step 2: Raise all dependency floors before implementation imports the API**

Change exactly these three entries:

```toml
"playwright>=1.48.0",
```

Expected locations: base Research/Web dependencies, `web_research`, and `scrape-analyzers`.

- [ ] **Step 3: Extend only the required runtime protocols**

Add route/request/WebSocket protocol surfaces to `runtime/browser.py`; keep concrete Playwright imports out of runtime. The analyzer-facing wrapper exposes:

```python
class BrowserProbePage(Protocol):
    async def goto(self, url: str, *, wait_until: str, timeout_ms: float) -> None:
        raise NotImplementedError
    async def reload(self, *, wait_until: str, timeout_ms: float) -> None:
        raise NotImplementedError
    async def wait_for_load_state(self, state: str, *, timeout_ms: float) -> None:
        raise NotImplementedError
    async def wait_for_timeout(self, timeout_ms: float) -> None:
        raise NotImplementedError
    async def content(self) -> str:
        raise NotImplementedError
    async def evaluate(self, expression: str, argument: Any = None) -> Any:
        raise NotImplementedError
    async def link_count(self) -> int:
        raise NotImplementedError
    async def link_is_visible(self, index: int) -> bool:
        raise NotImplementedError
    def captured_request_urls(self) -> tuple[str, ...]:
        raise NotImplementedError
    def clear_captured_request_urls(self) -> None:
        raise NotImplementedError
```

Use `raise NotImplementedError` in the protocol bodies; concrete wrappers implement every method.

- [ ] **Step 4: Implement guarded context creation and routing**

Before launch, `_playwright_has_required_routing()` imports the async `BrowserContext` type and verifies callable `route` and `route_web_socket`; false raises `ProbeUnavailable` without launching. `GuardedPlaywrightBrowserProbe(..., no_sandbox=False)` adds Chromium `args=["--no-sandbox"]` only when the context factory passed the explicit config value. Context creation always passes `service_workers="block"`. Install both handlers before `new_page()`.

```python
async def _route_http(route: RuntimeBrowserRoute) -> None:
    if route.request.resource_type in options.block_resource_types:
        await route.abort()
        return
    decision = await self._guard.decide(route.request.url, context=self._subrequest_context())
    if decision.allowed:
        await route.continue_()
    else:
        await route.abort()

async def _route_web_socket(socket: RuntimeWebSocketRoute) -> None:
    decision = await self._guard.decide(socket.url, context=self._subrequest_context())
    if not decision.allowed:
        await socket.close(code=1008, reason="Policy denied")
        return
    await _connect_web_socket_to_server(socket)
```

Guard exceptions are treated as denied. Never include full URLs in logs. `_connect_web_socket_to_server` resolves the installed version's supported server-connect callable (`connect_to_server` or its supported alias), invokes it exactly once, and awaits the result only when it is awaitable. The runtime capability check returns `unavailable` if neither callable exists. Unit tests cover both synchronous-return and awaitable-return fakes so the adapter remains valid across the declared Playwright range.

- [ ] **Step 5: Implement wrapper operations and bounded cleanup**

Before page creation, add every `BrowserProbeOptions.init_scripts` entry to the context and install request capture when requested. The wrapper delegates page operations, captures request URLs in memory only for the owning analyzer, and caps every Playwright timeout against `controls.remaining_seconds() * 1000`. A Playwright timeout raises `PreflightDeadlineExceeded` when the shared deadline is exhausted and analyzer-scoped `ProbeTimeout` otherwise. `about:blank` is the sole non-HTTP internal navigation allowed without a guard decision. Register browser/context/page handles with controls; graceful close order is page, context, browser, Playwright. Force cleanup closes or terminates remaining handles within the shared grace without replacing the original outcome.

- [ ] **Step 6: Add the optional local-browser smoke test**

Mark it `integration` and `smoke`. Start an in-process loopback HTTP server, skip when async Playwright or a Chromium executable is absent, set test egress overrides only for loopback, render a known HTML marker, assert nonblank content, and stop the server. It must not access a public URL and is not part of required CI.

- [ ] **Step 7: Run browser contract tests**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_browser.py
```

Expected: PASS using fakes only. Report the smoke separately if run:

```bash
python -m pytest -q --tb=short tldw_Server_API/tests/WebScraping/integration/test_phase3_preflight_browser_smoke.py
```

- [ ] **Step 8: Commit the browser adapter and dependency floor**

```bash
git add pyproject.toml tldw_Server_API/app/core/Web_Scraping/runtime tldw_Server_API/app/core/Web_Scraping/preflight/adapters/browser.py tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_browser.py tldw_Server_API/tests/WebScraping/integration/test_phase3_preflight_browser_smoke.py tldw_Server_API/tests/Web_Scraping/preflight_fakes.py backlog/tasks
git commit -m "feat: govern preflight browser probes"
```

---

### Task 6: Add the Governed External-Tool Adapter and Transition Signal

**Files:**
- Create: `tldw_Server_API/app/core/Web_Scraping/preflight/adapters/external_tools.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/preflight/adapters/__init__.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_external_tools.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/preflight_fakes.py`
- Modify: the Task 6 Backlog child record.

**Interfaces:**
- Consumes: `ExternalToolProbe`, controls, probe guard, `asyncio.create_subprocess_exec`, Loguru, and `Metrics.increment_counter`.
- Produces: `GuardedExternalToolProbe.run_waf(url, *, find_all, enabled) -> ExternalToolResult`.

- [ ] **Step 1: Write failing config-transition and process tests**

```python
@pytest.mark.parametrize(
    ("enabled", "installed", "starts", "warns"),
    [(None, True, True, True), (None, False, False, False), (True, True, True, False), (True, False, False, False), (False, True, False, False)],
)
@pytest.mark.asyncio
async def test_external_tool_enablement_matrix(enabled, installed, starts, warns) -> None:
    result = await probe(installed=installed).run_waf(
        "https://example.com", find_all=True, enabled=enabled
    )
    assert process_factory.called is starts
    assert legacy_warning.calls == (1 if warns else 0)
    assert metric.calls == (1 if warns else 0)
```

Run the absent-installed case concurrently and assert exactly one safe warning plus one metric call with `labels={"tool": "wafw00f"}`. Add explicit false, malformed-false via options, missing dependency, guard denial/error, active budget exhaustion, exact argv, timeout, cancellation, terminate/kill/await, nonzero exit, raw-output redaction, and parser-input tests.

- [ ] **Step 2: Implement a concurrency-safe once signal**

```python
class _LegacyExternalToolDefaultObserver:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._observed = False

    def observe(self) -> None:
        with self._lock:
            if self._observed:
                return
            self._observed = True
        logger.warning("Preflight external tool used because its config key is absent")
        increment_counter(
            "web_scraping_preflight_legacy_external_tool_default_total",
            labels={"tool": "wafw00f"},
        )

_LEGACY_DEFAULT_OBSERVER = _LegacyExternalToolDefaultObserver()
```

The production adapter defaults to the process singleton. Tests inject a fresh observer instance, so no production reset hook is needed. The warning contains no URL, command, or executable path.

- [ ] **Step 3: Implement governed subprocess lifecycle**

Resolve `wafw00f` through an injected `which` callable. Before process creation: evaluate enablement, reserve one `active_probe`, obtain a fresh allowed probe-egress decision, and cap 60 seconds against the overall deadline. Construct only `("wafw00f", url)` plus `"-a"` when requested; use `create_subprocess_exec`, never a shell.

```python
process = await self._process_factory(
    executable,
    url,
    *(["-a"] if find_all else []),
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
)
try:
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
except asyncio.CancelledError:
    await _terminate_process(process)
    raise
except TimeoutError as exc:
    await _terminate_process(process)
    if self._controls.deadline_exhausted():
        raise PreflightDeadlineExceeded() from exc
    raise ProbeTimeout() from exc
```

`_terminate_process` calls `terminate`, waits briefly inside the shared cleanup deadline, then `kill` and awaits. `ExternalToolResult` carries decoded stdout/stderr only to the WAF parser; no logs, public exceptions, or typed failures contain those values.

- [ ] **Step 4: Run external-tool tests**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_external_tools.py
```

Expected: PASS without consulting or executing any locally installed `wafw00f` outside injected fakes.

- [ ] **Step 5: Commit the tool adapter**

```bash
git add tldw_Server_API/app/core/Web_Scraping/preflight/adapters tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_external_tools.py tldw_Server_API/tests/Web_Scraping/preflight_fakes.py backlog/tasks
git commit -m "feat: govern preflight external tool probes"
```

---

### Task 7: Add the Compatibility-Only Background Event Loop Bridge

**Files:**
- Create: `tldw_Server_API/app/core/Web_Scraping/preflight/compatibility.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/preflight/facade.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_compatibility.py`
- Modify: the Task 7 Backlog child record.

**Interfaces:**
- Consumes: target evaluation, Tasks 3-6 adapters, and an injected internal analyzer coroutine.
- Produces: `build_execution_context(...)`, `_BackgroundLoopBridge.submit(coroutine, timeout_s)`, `_run_sync_compat(...)`, and `run_legacy_analyzer(...)` without importing analyzer modules.

- [ ] **Step 1: Write failing bridge lifecycle tests**

Assert lazy startup, same-process reuse, process-ID change recreation, return/exception propagation, timeout cancellation, cleanup completion, active caller event-loop support for synchronous wrappers, and process-exit shutdown.

```python
@pytest.mark.asyncio
async def test_sync_bridge_can_be_called_from_active_event_loop() -> None:
    def call_sync() -> str:
        return _run_sync_compat(asyncio.sleep(0, result="ok"), timeout_s=1.0)
    assert call_sync() == "ok"

def test_bridge_timeout_cancels_submission_and_waits_for_cleanup() -> None:
    cleaned = threading.Event()
    async def work() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()
    with pytest.raises(TimeoutError):
        _run_sync_compat(work(), timeout_s=0.01)
    assert cleaned.wait(2.0)
```

- [ ] **Step 2: Implement the isolated process-scoped bridge**

```python
class _BackgroundLoopBridge:
    def submit(self, coroutine: Coroutine[Any, Any, T], *, timeout_s: float | None = None) -> T:
        self._ensure_started_for_pid(os.getpid())
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            try:
                future.result(timeout=2.0)
            except (concurrent.futures.CancelledError, concurrent.futures.TimeoutError):
                pass
            raise TimeoutError("Legacy analyzer timed out.") from exc
```

Protect startup/shutdown with a `threading.Lock`. The daemon thread runs `loop.run_forever()`. At process exit, cancel pending tasks, gather them with `return_exceptions=True`, stop and close the loop, and join the thread for at most two seconds. If PID changes after a fork, discard inherited state and create a new thread lazily.

- [ ] **Step 3: Implement default execution-context construction**

Add this exact facade signature:

```python
@dataclass(frozen=True, slots=True)
class PreflightAdapterOverrides:
    http: HttpProbe | None = None
    browser: BrowserProbe | None = None
    external_tools: ExternalToolProbe | None = None
    egress_guard: ProbeEgressGuard | None = None
    clock: Callable[[], float] | None = None
    sleep: Callable[[float], Awaitable[None]] | None = None

def build_execution_context(
    target: PreflightTarget,
    options: PreflightOptions,
    *,
    policy_checker: OutboundPolicyChecker | None = None,
    limits: PreflightLimits | None = None,
    identity_selector: Callable[[], Mapping[str, str]] | None = None,
    injected_adapters: PreflightAdapterOverrides | None = None,
) -> PreflightExecutionContext:
```

Build controls with one monotonic deadline only when `options.timeout_s` is positive, default unbounded limits, `DefaultProbeEgressGuard`, governed HTTP/browser/tool adapters, and a copied random browser identity selected lazily once. Pass `options.playwright_no_sandbox` to the browser adapter at construction; analyzers cannot override launch security. `PreflightAdapterOverrides` has optional `http`, `browser`, `external_tools`, `egress_guard`, `clock`, and `sleep` fields; it is the only production test seam and avoids monkeypatching adapter internals.

- [ ] **Step 4: Add the generic legacy analyzer helper**

`run_legacy_analyzer` accepts `url`, an internal async callable, positional/keyword analyzer arguments, and optional injected policy/context factories. It evaluates the direct legacy target with `respect_robots=False`, returns a safe analyzer error without probes for `policy_denied`/`policy_error`, otherwise builds one default governed context, invokes the coroutine, and closes context in `finally`. It propagates cancellation inside async use; `_run_sync_compat` is the only thread bridge.

- [ ] **Step 5: Run compatibility bridge tests**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_compatibility.py
```

Expected: PASS with no leaked background tasks or threads after explicit test shutdown.

- [ ] **Step 6: Commit the compatibility bridge**

```bash
git add tldw_Server_API/app/core/Web_Scraping/preflight/compatibility.py tldw_Server_API/app/core/Web_Scraping/preflight/facade.py tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_compatibility.py backlog/tasks
git commit -m "feat: add legacy analyzer async bridge"
```

---

### Task 8: Move Pure Utilities, Scoring, and Recommendations to `preflight`

**Files:**
- Create by move: `tldw_Server_API/app/core/Web_Scraping/preflight/utils/__init__.py`
- Create by move: `tldw_Server_API/app/core/Web_Scraping/preflight/utils/browser_identities.py`
- Create by move: `tldw_Server_API/app/core/Web_Scraping/preflight/utils/impersonate_target.py`
- Create by move: `tldw_Server_API/app/core/Web_Scraping/preflight/utils/waf_result_parser.py`
- Create by move: `tldw_Server_API/app/core/Web_Scraping/preflight/scoring/__init__.py`
- Create by move: `tldw_Server_API/app/core/Web_Scraping/preflight/scoring/scoring_engine.py`
- Create by move: `tldw_Server_API/app/core/Web_Scraping/preflight/recommendations/__init__.py`
- Create by move: `tldw_Server_API/app/core/Web_Scraping/preflight/recommendations/recommender.py`
- Replace with explicit shim: `tldw_Server_API/app/core/Web_Scraping/scraper_analyzers/utils/__init__.py`
- Replace with explicit shim: `tldw_Server_API/app/core/Web_Scraping/scraper_analyzers/utils/browser_identities.py`
- Replace with explicit shim: `tldw_Server_API/app/core/Web_Scraping/scraper_analyzers/utils/impersonate_target.py`
- Replace with explicit shim: `tldw_Server_API/app/core/Web_Scraping/scraper_analyzers/utils/waf_result_parser.py`
- Replace with explicit shim: `tldw_Server_API/app/core/Web_Scraping/scraper_analyzers/scoring/__init__.py`
- Replace with explicit shim: `tldw_Server_API/app/core/Web_Scraping/scraper_analyzers/scoring/scoring_engine.py`
- Replace with explicit shim: `tldw_Server_API/app/core/Web_Scraping/scraper_analyzers/recommendations/__init__.py`
- Replace with explicit shim: `tldw_Server_API/app/core/Web_Scraping/scraper_analyzers/recommendations/recommender.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_characterization.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_compatibility.py`
- Modify: the Task 8 Backlog child record.

**Interfaces:**
- Consumes: current pure helpers and Task 1 literal behavior.
- Produces: canonical `preflight` implementations and old-path callable identity for `calculate_difficulty_score`, `generate_recommendations`, `MODERN_BROWSER_IDENTITIES`, `get_impersonate_target`, and `parse_wafw00f_output`.

- [ ] **Step 1: Add failing canonical-vs-legacy identity tests**

```python
@pytest.mark.unit
def test_scoring_and_recommendation_shims_reexport_canonical_callables() -> None:
    from tldw_Server_API.app.core.Web_Scraping.preflight.scoring.scoring_engine import calculate_difficulty_score as canonical_score
    from tldw_Server_API.app.core.Web_Scraping.preflight.recommendations.recommender import generate_recommendations as canonical_recommend
    from tldw_Server_API.app.core.Web_Scraping.scraper_analyzers.scoring.scoring_engine import calculate_difficulty_score as legacy_score
    from tldw_Server_API.app.core.Web_Scraping.scraper_analyzers.recommendations.recommender import generate_recommendations as legacy_recommend
    assert legacy_score is canonical_score
    assert legacy_recommend is canonical_recommend
```

Run this test and expect `ModuleNotFoundError` for canonical modules.

- [ ] **Step 2: Move implementation files without behavior edits**

```bash
git mv tldw_Server_API/app/core/Web_Scraping/scraper_analyzers/utils tldw_Server_API/app/core/Web_Scraping/preflight/utils
git mv tldw_Server_API/app/core/Web_Scraping/scraper_analyzers/scoring tldw_Server_API/app/core/Web_Scraping/preflight/scoring
git mv tldw_Server_API/app/core/Web_Scraping/scraper_analyzers/recommendations tldw_Server_API/app/core/Web_Scraping/preflight/recommendations
```

Update relative imports only. Do not change constants, scoring weights, recommendation strings, parser regexes, return types, or order.

- [ ] **Step 3: Recreate every old module as an explicit shim**

Use this exact pattern for each old module:

```python
"""Deprecated compatibility re-export; implementation lives in Web_Scraping.preflight."""

from __future__ import annotations

from tldw_Server_API.app.core.Web_Scraping.preflight.scoring.scoring_engine import calculate_difficulty_score

__all__ = ["calculate_difficulty_score"]
```

Package `__init__.py` files re-export the same historical public names and define explicit `__all__`. They contain no function/class bodies, wildcard imports, or runtime warnings.

- [ ] **Step 4: Run characterization and old-import tests**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_characterization.py tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_compatibility.py tldw_Server_API/tests/WebScraping/test_scraping_module.py
```

Expected: PASS with literal score/recommendation values unchanged.

- [ ] **Step 5: Commit the pure implementation move**

```bash
git add tldw_Server_API/app/core/Web_Scraping/preflight tldw_Server_API/app/core/Web_Scraping/scraper_analyzers tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_characterization.py tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_compatibility.py backlog/tasks
git commit -m "refactor: move preflight scoring and utilities"
```

---

### Task 9: Move and Govern Robots, TLS, Rate-Limit, and WAF Analyzers

**Files:**
- Create by move: `tldw_Server_API/app/core/Web_Scraping/preflight/analyzers/robots_checker.py`
- Create by move: `tldw_Server_API/app/core/Web_Scraping/preflight/analyzers/tls_analyzer.py`
- Create by move: `tldw_Server_API/app/core/Web_Scraping/preflight/analyzers/rate_limit_profiler.py`
- Create by move: `tldw_Server_API/app/core/Web_Scraping/preflight/analyzers/waf_detector.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/preflight/analyzers/__init__.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/preflight/analyzers/_shared.py`
- Replace with explicit shims: `tldw_Server_API/app/core/Web_Scraping/scraper_analyzers/analyzers/robots_checker.py`
- Replace with explicit shims: `tldw_Server_API/app/core/Web_Scraping/scraper_analyzers/analyzers/tls_analyzer.py`
- Replace with explicit shims: `tldw_Server_API/app/core/Web_Scraping/scraper_analyzers/analyzers/rate_limit_profiler.py`
- Replace with explicit shims: `tldw_Server_API/app/core/Web_Scraping/scraper_analyzers/analyzers/waf_detector.py`
- Modify as explicit shim exports: `tldw_Server_API/app/core/Web_Scraping/scraper_analyzers/analyzers/__init__.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_nonbrowser_analyzers.py`
- Modify: `tldw_Server_API/tests/WebScraping/test_scraper_analyzer_sanitizers.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_compatibility.py`
- Modify: the Task 9 Backlog child record.

**Interfaces:**
- Consumes: `PreflightExecutionContext.http`, `.external_tools`, `.browser_identity()`, `.controls.sleep()`, and Task 7 compatibility helper.
- Produces private async `_check_robots_txt`, `_analyze_tls_fingerprint`, `_profile_rate_limits`, `_detect_waf` plus historical public wrappers.

- [ ] **Step 1: Write failing fake-probe tests for all successful branches**

Replay fixed HTTP/tool responses and assert exact current dictionaries:

```python
@pytest.mark.asyncio
async def test_robots_parser_preserves_success_shape() -> None:
    context = fake_context(http_responses=[ProbeHttpResponse(
        url="https://example.com/robots.txt",
        status=200,
        headers={"Content-Type": "text/plain"},
        text="User-agent: *\nDisallow: /\nCrawl-delay: 2.5\n",
    )])
    assert await _check_robots_txt("https://example.com/path", context) == {
        "status": "success",
        "crawl_delay": 2.5,
        "scraping_disallowed": True,
    }

@pytest.mark.asyncio
async def test_tls_active_shape_uses_standard_and_impersonated_probe() -> None:
    context = fake_context(http_statuses=[403, 200])
    assert await _analyze_tls_fingerprint("https://example.com", context) == {
        "status": "active",
        "details": "Site blocks standard Python clients but allows browser-like clients.",
    }
    assert context.http.requests[1].impersonate == "chrome"
```

Add every existing success/error branch, WAF tuple parsing, 4 gentle plus 8 concurrent rate requests, early block, crawl-delay selection, optional impersonation, and missing dependency. No test may patch `http_client`, Playwright, curl-cffi, or subprocess from an analyzer module because those imports no longer exist there.

- [ ] **Step 2: Move modules and create private async implementations**

Create `preflight/analyzers/`, add an explicit package `__init__.py`, move the four modules with `git mv`, and immediately recreate their old-path shim files so every intermediate commit remains importable.

Use these exact internal signatures:

```python
async def _check_robots_txt(url: str, context: PreflightExecutionContext) -> dict[str, Any]
async def _analyze_tls_fingerprint(url: str, context: PreflightExecutionContext) -> dict[str, Any]
async def _profile_rate_limits(
    url: str,
    context: PreflightExecutionContext,
    crawl_delay: float | None,
    impersonate: bool = False,
) -> dict[str, Any]
async def _detect_waf(
    url: str,
    context: PreflightExecutionContext,
    find_all: bool = False,
    external_tools_enabled: bool | None = None,
) -> dict[str, Any]
```

The robots URL uses `urlsplit`/`urlunsplit` and `context.http.get`. TLS sends one standard and one impersonated request with the same cached identity. Rate-limit profiling preserves `GENTLE_PROBE_COUNT=4`, `BURST_COUNT=8`, `DEFAULT_DELAY=3.0`, and current blocking codes; use `context.controls.sleep(delay)` and `asyncio.gather` only for the burst. WAF uses `context.external_tools.run_waf` and the moved parser.

- [ ] **Step 3: Map probe failures safely without swallowing cancellation**

Define the same helper once in `preflight/analyzers/_shared.py`; it is private and is not re-exported by compatibility shims:

```python
async def _safe_analyzer_call(call: Awaitable[dict[str, Any]]) -> dict[str, Any]:
    try:
        return await call
    except asyncio.CancelledError:
        raise
    except PreflightDeadlineExceeded:
        raise
    except ProbeError as exc:
        return {"status": "error", "message": exc.public_message, "error_code": exc.error_code}
    except Exception:
        return {"status": "error", "message": "Analyzer failed.", "error_code": "analyzer_error"}
```

Use analyzer-specific safe messages where current public messages are already stable. Never expose exception text, URLs, stdout, or stderr.

- [ ] **Step 4: Add historical public wrappers and old-path shims**

TLS and rate-limit remain coroutine functions and call `run_legacy_analyzer` directly. Robots and WAF remain synchronous and call `_run_sync_compat(run_legacy_analyzer(...))`. Define signatures exactly as Task 1 recorded. Recreate old modules with explicit canonical re-exports and identity assertions.

- [ ] **Step 5: Verify behavior, signatures, and coroutine classification**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_nonbrowser_analyzers.py tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_compatibility.py tldw_Server_API/tests/WebScraping/test_scraper_analyzer_sanitizers.py
```

Expected: PASS; `inspect.iscoroutinefunction` is true only for TLS and rate-limit among these four.

- [ ] **Step 6: Commit the nonbrowser analyzers**

```bash
git add tldw_Server_API/app/core/Web_Scraping/preflight/analyzers tldw_Server_API/app/core/Web_Scraping/scraper_analyzers/analyzers tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_nonbrowser_analyzers.py tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_compatibility.py tldw_Server_API/tests/WebScraping/test_scraper_analyzer_sanitizers.py backlog/tasks
git commit -m "refactor: govern nonbrowser preflight analyzers"
```

---

### Task 10: Move and Govern the Five Browser Analyzers

**Files:**
- Create by move: `tldw_Server_API/app/core/Web_Scraping/preflight/analyzers/js_detector.py`
- Create by move: `tldw_Server_API/app/core/Web_Scraping/preflight/analyzers/behavioral_detector.py`
- Create by move: `tldw_Server_API/app/core/Web_Scraping/preflight/analyzers/captcha_detector.py`
- Create by move: `tldw_Server_API/app/core/Web_Scraping/preflight/analyzers/fingerprint_analyzer.py`
- Create by move: `tldw_Server_API/app/core/Web_Scraping/preflight/analyzers/integrity_analyzer.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/preflight/analyzers/__init__.py`
- Replace with explicit shims: `tldw_Server_API/app/core/Web_Scraping/scraper_analyzers/analyzers/js_detector.py`
- Replace with explicit shims: `tldw_Server_API/app/core/Web_Scraping/scraper_analyzers/analyzers/behavioral_detector.py`
- Replace with explicit shims: `tldw_Server_API/app/core/Web_Scraping/scraper_analyzers/analyzers/captcha_detector.py`
- Replace with explicit shims: `tldw_Server_API/app/core/Web_Scraping/scraper_analyzers/analyzers/fingerprint_analyzer.py`
- Replace with explicit shims: `tldw_Server_API/app/core/Web_Scraping/scraper_analyzers/analyzers/integrity_analyzer.py`
- Modify as explicit shim exports: `tldw_Server_API/app/core/Web_Scraping/scraper_analyzers/analyzers/__init__.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_browser_analyzers.py`
- Modify: `tldw_Server_API/tests/WebScraping/test_scraper_analyzer_sanitizers.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_compatibility.py`
- Modify: the Task 10 Backlog child record.

**Interfaces:**
- Consumes: governed browser/page and HTTP probes, deterministic identity, safe analyzer helper, and compatibility bridge.
- Produces private async `_analyze_js_rendering`, `_detect_honeypots`, `_detect_captcha`, `_analyze_fingerprinting`, `_analyze_function_integrity` plus five historical synchronous wrappers.

- [ ] **Step 1: Write deterministic tests for every successful heuristic**

Use `FakeBrowserProbePage` snapshots/evaluations and fake HTTP responses. Assert complete dictionaries for:

- JS: missing dependency/unavailable, zero rendered text, timeout, 0/25/75 boundary percentages, `js_required`, and `is_spa`.
- Honeypot: zero links, default 33% capped at 250, thorough 66%, deep 100%, and threshold `>3`.
- CAPTCHA: on-load detection, post-ten-reload detection, and none.
- Fingerprinting: script URL matches, globals, canvas signal, deduplicated/sorted listeners, and stable success message.
- Integrity: clean-vs-target signature differences and exact suspicion messages.

```python
@pytest.mark.asyncio
async def test_js_thresholds_preserve_existing_semantics() -> None:
    context = fake_context(
        http_text="x" * 75,
        browser_pages=[FakeBrowserProbePage(html="<body>" + ("x" * 100) + "</body>")],
    )
    result = await _analyze_js_rendering("https://example.com", context)
    assert result == {
        "status": "success",
        "js_required": False,
        "is_spa": False,
        "content_difference_%": 25.0,
    }
```

- [ ] **Step 2: Move modules and replace direct browser/network construction**

Use these exact internal signatures:

```python
async def _analyze_js_rendering(url: str, context: PreflightExecutionContext) -> dict[str, Any]
async def _detect_honeypots(url: str, context: PreflightExecutionContext, scan_depth: ScanDepth = "default") -> dict[str, Any]
async def _detect_captcha(url: str, context: PreflightExecutionContext) -> dict[str, Any]
async def _analyze_fingerprinting(url: str, context: PreflightExecutionContext) -> dict[str, Any]
async def _analyze_function_integrity(url: str, context: PreflightExecutionContext) -> dict[str, Any]
```

All five use `async with context.browser.open_page(BrowserProbeOptions(...))`. JS first uses `context.http.get` for no-JS content. Fingerprint options include `JS_PROBE_SCRIPT`, request capture, and image/font/media blocking. Integrity opens a clean `about:blank` page and a separate target page through the same guarded browser adapter. Preserve pure HTML parsing, JS probe strings, constants, threshold math, and successful output fields.

- [ ] **Step 3: Ensure browser probes remain governed across reloads and requests**

CAPTCHA's ten reloads use the wrapper so routing remains installed. Captured request URLs are observations only; they must not become a second dispatch mechanism. Every HTTP/HTTPS request and WebSocket is allowed or blocked by the adapter route handler before Playwright connects. `about:blank` is the only bypass and has no network transport.

- [ ] **Step 4: Replace unsafe exception output**

Cancellation propagates. `ProbeUnavailable` becomes `status=error`, stable capability code, and safe message. Current fingerprint/integrity branches that return `str(exc)` change to sanitized `"Fingerprint analysis failed."` and `"Function integrity analysis failed."` with `error_code="analyzer_error"`; tests prove credentials/query strings from exceptions never appear.

- [ ] **Step 5: Preserve historical sync wrappers and shims**

Each canonical public function keeps its Task 1 signature and calls `_run_sync_compat(run_legacy_analyzer(...))`. Old modules explicitly re-export that exact callable. Assert `inspect.signature`, `inspect.iscoroutinefunction(...) is False`, and callable identity for all five.

- [ ] **Step 6: Run browser-analyzer and sanitizer tests**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_browser_analyzers.py tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_compatibility.py tldw_Server_API/tests/WebScraping/test_scraper_analyzer_sanitizers.py
```

Expected: PASS with fake browser pages only.

- [ ] **Step 7: Commit the browser analyzers**

```bash
git add tldw_Server_API/app/core/Web_Scraping/preflight/analyzers tldw_Server_API/app/core/Web_Scraping/scraper_analyzers/analyzers tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_browser_analyzers.py tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_compatibility.py tldw_Server_API/tests/WebScraping/test_scraper_analyzer_sanitizers.py backlog/tasks
git commit -m "refactor: govern browser preflight analyzers"
```

---

### Task 11: Add the Deterministic Runner, Complete the Facade, and Shim Legacy Runner APIs

**Files:**
- Create: `tldw_Server_API/app/core/Web_Scraping/preflight/runner.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/preflight/facade.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/preflight/__init__.py`
- Replace with shim: `tldw_Server_API/app/core/Web_Scraping/scraper_analyzers/runner.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/scraper_analyzers/__init__.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_runner_facade.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_compatibility.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_characterization.py`
- Modify: the Task 11 Backlog child record.

**Interfaces:**
- Consumes: all private async analyzers, moved scoring/recommendations, typed contracts, adapters, and compatibility bridge.
- Produces: internal `gather_analysis_with_context`, public async `gather_analysis`, sync `run_analysis`, `build_execution_context`, `run_preflight`, `apply_preflight_advice`, and `public_preflight_payload`.

- [ ] **Step 1: Write failing runner isolation and order tests**

Inject one private analyzer that raises unexpectedly, one that raises `CancelledError`, and scoring/recommendation failures. Assert:

```python
assert list(result["results"]) == ANALYZER_KEYS
assert result["results"]["js"] == {
    "status": "error",
    "message": "JavaScript rendering analysis failed.",
    "error_code": "analyzer_error",
}
```

Unexpected analyzer failure affects only its key and remaining analyzers execute. Cancellation and `PreflightDeadlineExceeded` stop immediately and propagate. A local `ProbeTimeout` remains one analyzer error. Scoring/recommendation failure is raised to the facade as an overall failure.

- [ ] **Step 2: Implement the internal sequential runner**

Call private async functions in the approved order, preserve the rate analyzer's internal burst, then score/recommend:

```python
async def gather_analysis_with_context(
    target: PreflightTarget,
    options: PreflightOptions,
    context: PreflightExecutionContext,
) -> AnalysisOutput:
    results: dict[str, Any] = {}
    results["robots"] = await _isolated("robots", lambda: _check_robots_txt(target.url, context))
    crawl_delay = results["robots"].get("crawl_delay")
    results["tls"] = await _isolated("tls", lambda: _analyze_tls_fingerprint(target.url, context))
    results["js"] = await _isolated("js", lambda: _analyze_js_rendering(target.url, context))
    results["behavioral"] = await _isolated(
        "behavioral",
        lambda: _detect_honeypots(target.url, context, options.scan_depth),
    )
    results["captcha"] = await _isolated("captcha", lambda: _detect_captcha(target.url, context))
    results["fingerprint"] = await _isolated(
        "fingerprint",
        lambda: _analyze_fingerprinting(target.url, context),
    )
    results["integrity"] = await _isolated(
        "integrity",
        lambda: _analyze_function_integrity(target.url, context),
    )
    results["rate_limit"] = await _isolated(
        "rate_limit",
        lambda: _profile_rate_limits(target.url, context, crawl_delay, options.impersonate),
    )
    results["waf"] = await _isolated(
        "waf",
        lambda: _detect_waf(
            target.url,
            context,
            options.find_all_waf,
            options.external_tools_enabled,
        ),
    )
    return {
        "results": results,
        "score": calculate_difficulty_score(results),
        "recommendations": generate_recommendations(results),
    }
```

Define `_isolated(name, call: Callable[[], Awaitable[dict[str, Any]]])`. It creates the coroutine only after entering its analyzer isolation boundary, re-raises `CancelledError` and `PreflightDeadlineExceeded`, normalizes `ProbeError` to its stable public payload, and converts only unexpected analyzer exceptions to the analyzer-specific safe error payload. Tests run with warnings treated as errors to prove that setup failures cannot leak an unawaited coroutine.

- [ ] **Step 3: Verify and export default context construction**

Exercise Task 7 `build_execution_context` with defaults and each `PreflightAdapterOverrides` field. Export it from `preflight.__init__`; do not add a second factory in the runner. Assert the context retains the supplied scrape policy checker, target request context, unbounded default limits, one deadline, and one cached browser identity.

- [ ] **Step 4: Implement typed facade timeout/failure/cancellation behavior**

```python
async def run_preflight(
    target: PreflightTarget,
    options: PreflightOptions,
    context: PreflightExecutionContext,
) -> PreflightResult | None:
    if not options.enabled:
        return None
    if not target.decision.allowed:
        raise ValueError("run_preflight requires an allowed target")
    try:
        analysis = await _run_before_deadline(target, options, context)
        return PreflightResult(analysis=analysis, advice=_derive_advice(analysis))
    except asyncio.CancelledError:
        raise
    except PreflightDeadlineExceeded:
        return _failed_preflight(WebScrapingStatus.TIMEOUT, "Preflight analysis timed out.")
    except Exception:
        return _failed_preflight(WebScrapingStatus.ERROR, "Preflight analysis failed.")
    finally:
        await context.close()
```

Use one monotonic deadline, not nested independent timeout budgets. If caller cancellation races deadline expiration, observed cancellation wins. `context.close()` is the single cleanup path; it is idempotent, preserves pending cancellation, and logs cleanup failures without replacing the established result.

- [ ] **Step 5: Implement advice application and payload eligibility**

Use this exact signature:

```python
def apply_preflight_advice(
    result: PreflightResult | None,
    *,
    backend: str,
    method: str,
    backend_setting: str,
) -> tuple[str, str, PreflightResult | None]
```

JS success recommends Playwright only when `method == "auto"`; TLS active recommends curl only when `backend_setting == "auto"`. Return an updated `PreflightResult` whose advice contains final backend, final method, and only applied notes in `js_required`, `tls_active` order. Missing/error signals never route.

```python
def public_preflight_payload(result: PreflightResult | None, include_results: bool) -> dict[str, Any] | None:
    if not include_results or result is None or result.status is not WebScrapingStatus.OK:
        return None
    return preflight_result_to_public_dict(result)
```

- [ ] **Step 6: Implement public legacy aggregate wrappers**

Public `gather_analysis` keeps its exact async signature. It evaluates direct legacy targets with `respect_robots=False`; denial or checker failure returns all nine analyzer keys populated with safe `policy_denied`/`policy_error` errors and no probes, then calculates score/recommendations. Public `run_analysis` keeps its exact synchronous signature, uses `asyncio.run(gather_analysis(...))`, and raises the historical active-loop error instead of using the background bridge.

Replace old runner/package modules with explicit canonical re-exports. Assert callable identity and exact top-level shape.

- [ ] **Step 7: Run facade, runner, characterization, and compatibility tests**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_runner_facade.py tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_characterization.py tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_compatibility.py tldw_Server_API/tests/WebScraping/test_scraping_module.py
```

Expected: PASS, including overall timeout payload omission and cancellation cleanup.

- [ ] **Step 8: Commit the facade and runner cutover**

```bash
git add tldw_Server_API/app/core/Web_Scraping/preflight tldw_Server_API/app/core/Web_Scraping/scraper_analyzers tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_runner_facade.py tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_characterization.py tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_compatibility.py backlog/tasks
git commit -m "feat: centralize governed preflight orchestration"
```

---

### Task 12: Migrate `Article_Extractor_Lib` to the Shared Facade

**Files:**
- Modify: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase3_article_preflight_facade.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/test_phase2_article_runtime_boundary.py`
- Modify: `tldw_Server_API/tests/WebScraping/test_phase1_compatibility_contracts.py`
- Modify: `tldw_Server_API/tests/WebScraping/test_scraping_module.py`
- Modify: the Task 12 Backlog child record.

**Interfaces:**
- Consumes: `PreflightOptions`, `evaluate_target`, `build_execution_context`, `run_preflight`, `apply_preflight_advice`, and `public_preflight_payload` through the package-level facade import.
- Produces: unchanged public `scrape_article(...) -> dict[str, Any]` with one policy evaluation and centralized optional preflight behavior.

- [ ] **Step 1: Write failing article-facade tests before changing the consumer**

Patch `Article_Extractor_Lib.preflight_facade`, not `scraper_analyzers`. Cover:

- denied primary target performs no context creation, analyzer, or extraction fetch;
- policy checker failure returns the existing generic safe extraction failure;
- disabled options perform no preflight work;
- JS advice chooses Playwright only from successful signals;
- TLS advice chooses curl only for automatic backend;
- overall preflight timeout/error preserves original backend/method and omits payload;
- analyzer-scoped errors retain overall OK and eligible payload;
- payload attaches to lightweight success, Playwright success, extraction failure, and fallback return paths exactly as before;
- cancellation from `evaluate_target` or `run_preflight` propagates.

```python
@pytest.mark.asyncio
async def test_article_preflight_cancellation_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    article = install_article_defaults(monkeypatch, preflight=True)
    monkeypatch.setattr(article.preflight_facade, "evaluate_target", AsyncMock(return_value=allowed_target()))
    monkeypatch.setattr(article.preflight_facade, "build_execution_context", lambda *_args, **_kwargs: fake_context())
    monkeypatch.setattr(article.preflight_facade, "run_preflight", AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await article.scrape_article("https://example.com")
```

- [ ] **Step 2: Replace duplicated config parsing and analysis invocation**

Import the package once:

```python
from tldw_Server_API.app.core.Web_Scraping import preflight as preflight_facade
```

After plan/header/backend resolution, use:

```python
options = preflight_facade.PreflightOptions.from_mapping(ws_cfg)
target = await preflight_facade.evaluate_target(
    url,
    respect_robots=bool(getattr(plan, "respect_robots", True)),
    user_agent=effective_ua,
    request_context=RuntimeRequestContext(source="article_extract", stage="pre_fetch"),
    config={"web_scraper": ws_cfg},
    policy_checker=_ARTICLE_POLICY_CHECKER,
)
if not target.decision.allowed:
    return _attach_preflight(_blocked_article_result(url, target.decision))

preflight_result = None
if options.enabled:
    context = preflight_facade.build_execution_context(
        target,
        options,
        policy_checker=_ARTICLE_POLICY_CHECKER,
    )
    preflight_result = await preflight_facade.run_preflight(target, options, context)
backend_choice, preflight_method, preflight_result = preflight_facade.apply_preflight_advice(
    preflight_result,
    backend=backend_choice,
    method="auto",
    backend_setting=str(getattr(plan, "backend", "auto") or "auto").lower().strip(),
)
preflight_payload = preflight_facade.public_preflight_payload(preflight_result, options.include_results)
```

Remove direct `run_analysis`, `asyncio.to_thread`, duplicated scan-depth/timeout/bool parsing, JS/TLS result inspection, and manual payload construction.

- [ ] **Step 3: Preserve policy and cancellation behavior explicitly**

Keep the current blocked-result adapter. Surround policy evaluation with:

```python
except asyncio.CancelledError:
    raise
except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS:
    return _attach_preflight({
        "url": url,
        "title": "N/A",
        "author": "N/A",
        "date": "N/A",
        "content": "",
        "extraction_successful": False,
        "error": "Outbound policy evaluation failed. Please contact system administrator.",
    })
```

Do not remove `CancelledError` from the module-wide compatibility tuple in this phase; use explicit earlier handlers at touched preflight boundaries to avoid unrelated behavior changes.

- [ ] **Step 4: Update existing tests to patch the facade**

Replace monkeypatches of `scraper_analyzers.run_analysis` with typed `PreflightResult` returns from `preflight_facade.run_preflight`. Keep expected public dictionaries unchanged. Existing policy-before-preflight assertions now assert `evaluate_target`/`run_preflight` calls and no fetch.

- [ ] **Step 5: Run article compatibility suites**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase3_article_preflight_facade.py tldw_Server_API/tests/Web_Scraping/test_phase2_article_runtime_boundary.py tldw_Server_API/tests/WebScraping/test_phase1_compatibility_contracts.py tldw_Server_API/tests/WebScraping/test_scraping_module.py tldw_Server_API/tests/WebScraping/test_curl_backend_pipeline.py
```

Expected: PASS with unchanged public payload assertions and propagated cancellation.

- [ ] **Step 6: Commit the article consumer**

```bash
git add tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py tldw_Server_API/tests/Web_Scraping/test_phase3_article_preflight_facade.py tldw_Server_API/tests/Web_Scraping/test_phase2_article_runtime_boundary.py tldw_Server_API/tests/WebScraping/test_phase1_compatibility_contracts.py tldw_Server_API/tests/WebScraping/test_scraping_module.py backlog/tasks
git commit -m "refactor: route article preflight through facade"
```

---

### Task 13: Migrate `EnhancedWebScraper` to the Same Facade

**Files:**
- Modify: `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase3_enhanced_preflight_facade.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/test_enhanced_web_scraping_guards.py`
- Modify: `tldw_Server_API/tests/WebScraping/test_phase1_compatibility_contracts.py`
- Modify: `tldw_Server_API/tests/WebScraping/test_scraping_module.py`
- Modify: the Task 13 Backlog child record.

**Interfaces:**
- Consumes: `PreflightOptions`, `evaluate_target`, `build_execution_context`, `run_preflight`, `apply_preflight_advice`, `public_preflight_payload`, and `DefaultWebOutboundPolicyChecker`.
- Produces: unchanged `EnhancedWebScraper.scrape_article(...)` public dictionary behavior with no duplicated preflight methods.

- [ ] **Step 1: Write failing enhanced-consumer tests**

Cover the Task 12 matrix for all three enhanced methods (`trafilatura`, `playwright`, `beautifulsoup`) plus unknown-method failure. Assert target evaluation receives source `enhanced_scrape`, stage `pre_fetch`, current user agent, robots option, and `{"web_scraper": self.config}`. Assert cancellation is not caught by the outer noncritical exception tuple.

```python
@pytest.mark.asyncio
async def test_enhanced_outer_handler_does_not_swallow_preflight_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    scraper = install_enhanced_defaults(monkeypatch, preflight=True)
    monkeypatch.setattr(enhanced.preflight_facade, "evaluate_target", AsyncMock(return_value=allowed_target()))
    monkeypatch.setattr(enhanced.preflight_facade, "build_execution_context", lambda *_args, **_kwargs: fake_context())
    monkeypatch.setattr(enhanced.preflight_facade, "run_preflight", AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await scraper.scrape_article("https://example.com")
```

- [ ] **Step 2: Add one scrape-policy adapter dependency**

At module scope:

```python
from tldw_Server_API.app.core.Web_Scraping import preflight as preflight_facade
from tldw_Server_API.app.core.Web_Scraping.policy import DefaultWebOutboundPolicyChecker
from tldw_Server_API.app.core.Web_Scraping.runtime import RuntimeRequestContext

_ENHANCED_POLICY_CHECKER = DefaultWebOutboundPolicyChecker()
```

Use `preflight_facade.evaluate_target(..., policy_checker=_ENHANCED_POLICY_CHECKER)` instead of direct `decide_web_outbound_policy` in the single-article path. Leave crawl-specific policy paths untouched.

- [ ] **Step 3: Replace private duplicated orchestration**

Delete `_run_preflight_analysis` and `_apply_preflight_advice`; use this sequence in `scrape_article` and preserve the caller-supplied method:

```python
options = preflight_facade.PreflightOptions.from_mapping(self.config or {})
target = await preflight_facade.evaluate_target(
    url,
    respect_robots=bool(getattr(plan, "respect_robots", True)),
    user_agent=headers.get("User-Agent", DEFAULT_USER_AGENT),
    request_context=RuntimeRequestContext(source="enhanced_scrape", stage="pre_fetch"),
    config={"web_scraper": self.config or {}},
    policy_checker=_ENHANCED_POLICY_CHECKER,
)
if not target.decision.allowed:
    return _attach_preflight(_blocked_scrape_result(url, target.decision))

preflight_result = None
if options.enabled:
    context = preflight_facade.build_execution_context(
        target,
        options,
        policy_checker=_ENHANCED_POLICY_CHECKER,
    )
    preflight_result = await preflight_facade.run_preflight(target, options, context)
backend_setting = str(getattr(plan, "backend", "auto") or "auto").lower().strip()
backend_choice, method, preflight_result = preflight_facade.apply_preflight_advice(
    preflight_result,
    backend=backend_choice,
    method=method,
    backend_setting=backend_setting,
)
preflight_payload = preflight_facade.public_preflight_payload(preflight_result, options.include_results)
```

Use `preflight_payload` on every existing return path.

- [ ] **Step 4: Make the touched outer exception boundary cancellation-safe**

Immediately before the existing final `except _WEBSCRAPE_NONCRITICAL_EXCEPTIONS`, add:

```python
except asyncio.CancelledError:
    raise
```

Policy-evaluation failure returns the existing structural failure dictionary but uses the safe generic message `"Outbound policy evaluation failed. Please contact system administrator."`; no exception text or URL query is exposed.

- [ ] **Step 5: Update old tests to patch typed facade results**

Replace direct calls/patches of removed private methods with facade tests. Preserve all existing expected final backend, method, note order, and `preflight_analysis` dictionary shapes.

- [ ] **Step 6: Run enhanced compatibility and guard suites**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase3_enhanced_preflight_facade.py tldw_Server_API/tests/Web_Scraping/test_enhanced_web_scraping_guards.py tldw_Server_API/tests/WebScraping/test_phase1_compatibility_contracts.py tldw_Server_API/tests/WebScraping/test_scraping_module.py tldw_Server_API/tests/WebScraping/test_playwright_guard_and_cookies.py
```

Expected: PASS, including policy-before-preflight and cancellation propagation.

- [ ] **Step 7: Commit the enhanced consumer**

```bash
git add tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py tldw_Server_API/tests/Web_Scraping/test_phase3_enhanced_preflight_facade.py tldw_Server_API/tests/Web_Scraping/test_enhanced_web_scraping_guards.py tldw_Server_API/tests/WebScraping/test_phase1_compatibility_contracts.py tldw_Server_API/tests/WebScraping/test_scraping_module.py backlog/tasks
git commit -m "refactor: route enhanced preflight through facade"
```

---

### Task 14: Enforce Architecture, Refresh Documentation, and Run Final Gates

**Files:**
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_architecture.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/README.md`
- Regenerate: `Docs/Design/WebScraping_Refactor_Import_Inventory.md`
- Regenerate: `Docs/Design/web_scraping_refactor_import_inventory.json`
- Modify: final implementation Backlog parent/child records.

**Interfaces:**
- Consumes: the complete Phase 3 package and approved design.
- Produces: structural dependency enforcement, updated import inventory/docs, security report, and final verified branch.

- [ ] **Step 1: Write the AST architecture test before cleanup**

Implement a path-aware import scanner with explicit rules:

```python
FORBIDDEN_ANALYZER_IMPORTS = {
    "asyncio.subprocess",
    "curl_cffi",
    "playwright",
    "subprocess",
    "tldw_Server_API.app.core.http_client",
    "tldw_Server_API.app.core.Security.egress",
}
FORBIDDEN_PREFLIGHT_CONSUMERS = {
    "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib",
    "tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping",
}

def test_preflight_dependency_direction() -> None:
    assert_no_imports(PREFLIGHT_ROOT, FORBIDDEN_PREFLIGHT_CONSUMERS)
    assert_no_imports(PREFLIGHT_ANALYZERS, FORBIDDEN_ANALYZER_IMPORTS)
    assert_no_imports(RUNTIME_ROOT, {"tldw_Server_API.app.core.Web_Scraping.preflight", "tldw_Server_API.app.core.Web_Scraping.policy"})
```

Also assert:

- consumer modules import only package-level `preflight`, not analyzer/scoring/recommendation internals;
- application code outside `scraper_analyzers` contains no new old-package imports;
- every old module from the Phase 0 inventory resolves;
- every shim AST contains only docstring, future import, explicit imports, and `__all__` assignment;
- no wildcard re-exports;
- canonical/legacy public callable identity, signatures, and coroutine classification match;
- all three Playwright dependency floors equal `>=1.48.0`.

- [ ] **Step 2: Run the architecture test and remove violations**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_architecture.py tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py
```

Expected: PASS. Fix only Phase 3 dependency violations; do not broaden into later extraction/crawl/search refactors.

- [ ] **Step 3: Update package documentation**

Document `Web_Scraping.preflight` as canonical; mark `scraper_analyzers` as a temporary Phase 7 compatibility path without runtime warning. List all existing config keys, absent/explicit external-tool behavior and sunset, Playwright 1.48/service-worker requirement, fail-open analyzer semantics, primary policy blocking, optional payload behavior, and browser/external-tool DNS limitations.

- [ ] **Step 4: Regenerate and review the import inventory**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python Helper_Scripts/web_scraping_refactor_inventory.py --root . --json Docs/Design/web_scraping_refactor_import_inventory.json --markdown Docs/Design/WebScraping_Refactor_Import_Inventory.md
git diff -- Docs/Design/WebScraping_Refactor_Import_Inventory.md Docs/Design/web_scraping_refactor_import_inventory.json
```

Expected: both consumers import canonical preflight; old imports remain only compatibility/tests explicitly covered by the architecture allowlist. Review every unexpected new record.

- [ ] **Step 5: Run focused Phase 3 and earlier compatibility suites**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_characterization.py tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_contracts.py tldw_Server_API/tests/Web_Scraping/test_phase3_probe_egress.py tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_http.py tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_browser.py tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_external_tools.py tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_nonbrowser_analyzers.py tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_browser_analyzers.py tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_runner_facade.py tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_compatibility.py tldw_Server_API/tests/Web_Scraping/test_phase3_article_preflight_facade.py tldw_Server_API/tests/Web_Scraping/test_phase3_enhanced_preflight_facade.py tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_architecture.py
python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase1_contracts.py tldw_Server_API/tests/WebScraping/test_phase1_compatibility_contracts.py tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_contracts.py tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_adapters.py tldw_Server_API/tests/Web_Scraping/test_phase2_article_runtime_boundary.py
```

Expected: all required tests pass without public network, real browser, or real executable.

- [ ] **Step 6: Run broad Web_Scraping regressions**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping tldw_Server_API/tests/WebScraping
```

Expected: PASS. Record exact pass/skip counts; optional browser smoke skip is acceptable and reported separately.

- [ ] **Step 7: Run compile, formatting, lint, and Bandit gates**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m compileall -q tldw_Server_API/app/core/Web_Scraping/preflight tldw_Server_API/app/core/Web_Scraping/runtime tldw_Server_API/app/core/Web_Scraping/policy
python -m black --check tldw_Server_API/app/core/Web_Scraping/preflight tldw_Server_API/app/core/Web_Scraping/runtime tldw_Server_API/app/core/Web_Scraping/policy tldw_Server_API/tests/Web_Scraping/test_phase3_*.py
python -m ruff check tldw_Server_API/app/core/Web_Scraping/preflight tldw_Server_API/app/core/Web_Scraping/runtime tldw_Server_API/app/core/Web_Scraping/policy tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py
python -m bandit -r tldw_Server_API/app/core/Web_Scraping/preflight tldw_Server_API/app/core/Web_Scraping/runtime tldw_Server_API/app/core/Web_Scraping/policy tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py -f json -o /tmp/bandit_web_scraping_phase3.json
```

Expected: all commands exit zero. Review the Bandit JSON; fix new findings in touched code and record any unchanged baseline findings explicitly.

- [ ] **Step 8: Rebase latest dev and rerun affected gates**

```bash
git fetch origin
git rebase origin/dev
git status --short --branch
```

After conflict resolution, rerun Tasks 14 Steps 5-7 and the import inventory. Expected: clean verified branch based on latest dev.

- [ ] **Step 9: Complete Backlog records and commit final docs/gates**

Update every child and parent with touched files, verification counts, optional smoke status, Bandit result, known skips, final summary, and eventual PR link. Mark complete only after all required gates pass.

```bash
git add tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_architecture.py tldw_Server_API/app/core/Web_Scraping/README.md Docs/Design/WebScraping_Refactor_Import_Inventory.md Docs/Design/web_scraping_refactor_import_inventory.json backlog/tasks
git commit -m "docs: finalize governed preflight migration"
git status --short --branch
```

Expected: final implementation commit succeeds and worktree is clean.

---

## Completion Checklist

- [ ] `preflight` is the only analyzer implementation owner.
- [ ] Both scrape consumers use only the shared facade.
- [ ] Primary scrape policy and per-dispatch probe egress remain separate and tested.
- [ ] All nine analyzer signatures, coroutine classifications, result keys, successful values, score cards, and recommendations remain compatible.
- [ ] Required probes are governed, deadline-capped, budgeted, redacted, and cancellation-safe.
- [ ] Browser routing is installed before page creation with service workers blocked and runtime capability fallback.
- [ ] External-tool compatibility warning/metric is process-once and explicit config is authoritative.
- [ ] Overall preflight failure omits advice/payload and never fails extraction.
- [ ] Every old import path is an explicit temporary shim with no runtime logic.
- [ ] Required deterministic tests, broad regressions, compile, format, lint, and Bandit pass.
- [ ] Import inventory, README, Backlog records, and human-authored PR change summary are complete.
