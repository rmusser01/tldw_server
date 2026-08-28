# Browser Transport Safety Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver TASK-13139.2 by preventing strict or multi-user retrieval from silently entering Playwright when the browser transport cannot prove that all requests use a governed DNS-pinned, peer-verified path.

**Architecture:** Add one pure browser-transport admission decision in `Web_Scraping` and call it before either existing Playwright adapter reserves capacity or launches a browser. Preserve the existing pre-page HTTP/WebSocket routing, service-worker block, redirect/subresource checks, popup handling, and HTTP-to-browser escalation. Single-user `compat` deployments retain the current URL-guarded browser behavior with explicit `dns_peer_attested=false` metadata. Strict or multi-user deployments fail closed unless both an explicit `attested_proxy` mode and a concrete in-process attestation object are supplied. Wave 0 defines the attestation contract but deliberately wires no production proxy and accepts no self-asserted boolean from configuration.

**Tech Stack:** Python 3.10+ dataclasses, `Literal`, `Protocol`, existing Web_Scraping config/outbound-policy helpers, existing Playwright adapters, pytest, and Bandit.

**Spec:** `Docs/superpowers/specs/2026-08-27-agent-native-web-research-quality-provenance-roadmap.md` sections 2, 8, 9, and TASK-13139.2; `Docs/ADR/026-security-outbound-egress-and-ssrf-policy.md`.

## Global Constraints

- Work under Backlog task `TASK-13139.2`; set it to In Progress before source edits and keep its implementation notes current.
- Existing URL-policy checks remain necessary and unchanged. This task must not claim that they pin browser DNS or verify the connected peer.
- Do not add a second request-routing layer, browser launcher, extraction fallback ladder, proxy client, DNS resolver, cookie store, authenticated browser session, or new dependency.
- `AUTH_MODE=multi_user` and `WEB_OUTBOUND_POLICY_MODE=strict` are the two currently enforceable untrusted boundaries for Wave 0. Either boundary requires an attested transport.
- Configuration alone can select `attested_proxy`, but cannot attest it. Only a concrete `BrowserTransportAttestation` supplied by a future governed transport integration may satisfy admission.
- The default production construction supplies no attestation. Therefore `attested_proxy` configured without an approved integration still fails closed.
- Admission happens before acquisition-pool reservation, browser-budget reservation, Playwright startup, or any network dispatch.
- Credentialless governed HTTP extraction remains available in every mode and is not routed through this browser gate.
- Public metadata is bounded to fixed booleans and enumerated strings. Never expose proxy addresses, headers, cookies, credentials, raw URLs, exceptions, or configuration text.
- Use TDD and preserve all existing redirect, subresource, WebSocket, popup, and service-worker regression tests.

## File Map

- Add: `tldw_Server_API/app/core/Web_Scraping/browser_transport.py`
  - Define modes, attestation evidence, deterministic admission decisions, safe capability metadata, and the default environment/config resolver.
- Modify: `tldw_Server_API/app/core/config.py`
  - Add the fail-closed `web_browser_transport_mode()` resolver and expose its value in the existing `web_scraper` config mapping.
- Modify: `tldw_Server_API/Config_Files/config.txt`
  - Add the documented `web_browser_transport_mode = auto` setting.
- Add: `tldw_Server_API/tests/Web_Scraping/test_browser_transport.py`
  - Cover the complete decision matrix, attestation evidence, safe metadata, malformed inputs, and default config resolution.
- Modify: `tldw_Server_API/tests/Web_Scraping/test_outbound_policy.py`
  - Cover env-over-config precedence and invalid browser-transport config fail-closed behavior.
- Modify: `tldw_Server_API/app/core/Web_Scraping/orchestration/article_browser.py`
  - Enforce admission before pool reservation/launch and expose a read-only capability snapshot.
- Modify: `tldw_Server_API/app/core/Web_Scraping/orchestration/article_models.py`
  - Add the stable `browser_transport_unavailable` public error code and an immutable optional capability snapshot on `ArticleFailure`.
- Modify: `tldw_Server_API/app/core/Web_Scraping/orchestration/article.py`
  - Preserve the safe admission reason as bounded capability metadata in article and raw-browser failures.
- Modify: `tldw_Server_API/tests/Web_Scraping/test_phase4_article_browser.py`
  - Cover denial before pool/launch and preservation of attested redirect/subresource controls.
- Modify: `tldw_Server_API/tests/Web_Scraping/test_phase4_article_models.py`
  - Cover the additive public failure code.
- Modify: `tldw_Server_API/tests/Web_Scraping/test_phase4_article_orchestration.py`
  - Cover structured denial and continued credentialless HTTP success.
- Modify: `tldw_Server_API/app/core/Web_Scraping/preflight/probes.py`
  - Add bounded safe error codes/messages for browser transport denial.
- Modify: `tldw_Server_API/app/core/Web_Scraping/preflight/adapters/browser.py`
  - Enforce the same admission decision before budget reservation/launch.
- Modify: `tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_browser.py`
  - Cover preflight denial before budget/launch and an attested permitted path.
- Modify: `Docs/ADR/026-security-outbound-egress-and-ssrf-policy.md`
  - Record the browser DNS-rebinding limitation and fail-closed gate.
- Modify: `Docs/Published/ADR/026-security-outbound-egress-and-ssrf-policy.md`
  - Keep the published ADR mirror synchronized.
- Modify: `Docs/Design/WebScraping.md`
  - Document configuration, capability metadata, and the lack of authenticated browser support.

## Browser Transport Contract

The new module defines these exact public types and functions:

```python
ConfiguredBrowserTransportMode = Literal[
    "auto",
    "disabled",
    "url_guarded",
    "attested_proxy",
]
EffectiveBrowserTransportMode = Literal[
    "disabled",
    "url_guarded",
    "attested_proxy",
]
BrowserTransportReason = Literal[
    "browser_transport_allowed_legacy",
    "browser_transport_allowed_attested",
    "browser_transport_disabled",
    "browser_transport_unattested",
    "browser_transport_config_invalid",
]


@dataclass(frozen=True, slots=True)
class BrowserTransportAttestation:
    mechanism: Literal["governed_proxy"]
    routes_all_requests: bool
    dns_pinned: bool
    peer_verified: bool


@dataclass(frozen=True, slots=True)
class BrowserTransportDecision:
    allowed: bool
    configured_mode: ConfiguredBrowserTransportMode
    effective_mode: EffectiveBrowserTransportMode
    dns_peer_attested: bool
    reason: BrowserTransportReason

    def to_capability_metadata(self) -> dict[str, str | bool]: ...


def decide_browser_transport(
    *,
    configured_mode: object,
    auth_mode: object,
    outbound_policy_mode: object,
    attestation: BrowserTransportAttestation | None = None,
) -> BrowserTransportDecision: ...


def default_browser_transport_decision(
    *,
    attestation: BrowserTransportAttestation | None = None,
    environ: Mapping[str, str] | None = None,
) -> BrowserTransportDecision: ...
```

`to_capability_metadata()` returns only:

```python
{
    "name": "safe_browser_transport",
    "available": decision.allowed,
    "configured_mode": decision.configured_mode,
    "effective_mode": decision.effective_mode,
    "dns_peer_attested": decision.dns_peer_attested,
    "reason": decision.reason,
}
```

The decision table is normative:

| Configured mode | Auth mode | Outbound mode | Attestation | Result |
| --- | --- | --- | --- | --- |
| `disabled` | any | any | any | deny, `browser_transport_disabled` |
| malformed | any | any | any | deny with sanitized mode `disabled`, `browser_transport_config_invalid` |
| `auto` or `url_guarded` | `single_user` | `compat` | ignored | allow effective `url_guarded`, `dns_peer_attested=false`, `browser_transport_allowed_legacy` |
| `auto` or `url_guarded` | any other combination | any | any | deny, `browser_transport_unattested` |
| `attested_proxy` | any | any | absent or incomplete | deny, `browser_transport_unattested` |
| `attested_proxy` | any | any | mechanism is `governed_proxy` and all three booleans are true | allow effective `attested_proxy`, `dns_peer_attested=true`, `browser_transport_allowed_attested` |

Any auth value other than exact `single_user` and any outbound value other than exact `compat` is treated as requiring attestation. An exception or wrong return type from an injected decision provider is converted to `browser_transport_config_invalid`; it never falls back to legacy admission.

## Public Denial Shape

Direct article retrieval adds `browser_transport_unavailable` to `PUBLIC_FAILURE_CODES`. Only this failure receives capability metadata:

```json
{
  "url": "https://article.example/start",
  "title": "N/A",
  "author": "N/A",
  "date": "N/A",
  "content": "",
  "extraction_successful": false,
  "error": "browser_transport_unavailable",
  "capability": {
    "name": "safe_browser_transport",
    "available": false,
    "configured_mode": "auto",
    "effective_mode": "disabled",
    "dns_peer_attested": false,
    "reason": "browser_transport_unattested"
  }
}
```

All other article failure shapes remain unchanged. The raw-browser helper keeps its compact shape and adds the same `capability` object only for this error.

Preflight raises `ProbeUnavailable` with one of these safe error codes:

- `browser_transport_disabled`
- `browser_transport_unattested`
- `browser_transport_config_invalid`

Each maps to a fixed public message in `_SAFE_ERROR_MESSAGES`; no raw exception or configuration value reaches analyzer output.

## Task 1: Add the Pure Admission Decision and Configuration

**Files:**

- Add: `tldw_Server_API/tests/Web_Scraping/test_browser_transport.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/test_outbound_policy.py`
- Add: `tldw_Server_API/app/core/Web_Scraping/browser_transport.py`
- Modify: `tldw_Server_API/app/core/config.py`
- Modify: `tldw_Server_API/Config_Files/config.txt`

**Success Criteria:** The complete decision matrix is deterministic, malformed inputs fail closed, a valid attestation requires all evidence fields, metadata is bounded, and env-over-config resolution works without initializing AuthNZ.

- [ ] **Step 1: Mark TASK-13139.2 In Progress**

Use the official Backlog.md workflow to set `TASK-13139.2` to In Progress and add this plan path to its documentation.

- [ ] **Step 2: Add failing decision-matrix tests**

Include this table-driven core:

```python
@pytest.mark.parametrize(
    ("configured", "auth_mode", "policy_mode", "allowed", "reason"),
    [
        ("auto", "single_user", "compat", True, "browser_transport_allowed_legacy"),
        ("url_guarded", "single_user", "compat", True, "browser_transport_allowed_legacy"),
        ("auto", "multi_user", "compat", False, "browser_transport_unattested"),
        ("auto", "single_user", "strict", False, "browser_transport_unattested"),
        ("url_guarded", "multi_user", "strict", False, "browser_transport_unattested"),
        ("disabled", "single_user", "compat", False, "browser_transport_disabled"),
        ("bogus", "single_user", "compat", False, "browser_transport_config_invalid"),
    ],
)
def test_browser_transport_decision_matrix(
    configured: str,
    auth_mode: str,
    policy_mode: str,
    allowed: bool,
    reason: str,
) -> None:
    decision = decide_browser_transport(
        configured_mode=configured,
        auth_mode=auth_mode,
        outbound_policy_mode=policy_mode,
    )

    assert decision.allowed is allowed
    assert decision.reason == reason
```

Also test that `attested_proxy` denies for each missing/false evidence field and allows only:

```python
BrowserTransportAttestation(
    mechanism="governed_proxy",
    routes_all_requests=True,
    dns_pinned=True,
    peer_verified=True,
)
```

Assert the exact six-key metadata mapping and that its serialized form contains no environment value, address, URL, header, cookie, or credential.

- [ ] **Step 3: Add failing config-resolution tests**

In `test_outbound_policy.py`, cover:

- `WEB_BROWSER_TRANSPORT_MODE` overrides both supported config section names;
- `[Web-Scraper] web_browser_transport_mode` is used when env is absent;
- legacy `[Web-Scraping]` remains readable;
- missing setting returns `auto`;
- malformed env or config returns `disabled`, not `auto`;
- `default_browser_transport_decision(environ={"AUTH_MODE": ...})` uses the supplied environment mapping while outbound/config helpers are monkeypatched, so tests do not mutate AuthNZ global settings.

- [ ] **Step 4: Run the focused tests and confirm red**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Web_Scraping/test_browser_transport.py tldw_Server_API/tests/Web_Scraping/test_outbound_policy.py -q
```

Expected: FAIL because the module and resolver do not exist.

- [ ] **Step 5: Implement the pure module**

Implementation rules:

- Normalize only exact lowercased enumerated strings; do not accept truthy aliases.
- Sanitize malformed configured modes to `configured_mode="disabled"`.
- A complete attestation is evidence only when configured mode is explicitly `attested_proxy`.
- Catch no exceptions inside the pure decision function; it receives data only.
- `default_browser_transport_decision()` reads `AUTH_MODE` from the injected mapping or `os.environ`, and calls `web_browser_transport_mode()` plus `web_outbound_policy_mode()` for existing env-over-config behavior.
- Do not import `AuthNZ.settings`; this keeps scraping config resolution lightweight and avoids global settings initialization in tests.

- [ ] **Step 6: Add the config resolver and mapping field**

Add:

```python
def web_browser_transport_mode(default: str = "auto") -> str:
    """Resolve auto|disabled|url_guarded|attested_proxy, failing closed when malformed."""
```

Read `WEB_BROWSER_TRANSPORT_MODE`, then `[Web-Scraper]`, then legacy `[Web-Scraping]`. Return `disabled` for any non-empty unsupported value. Add the resolved value as `web_browser_transport_mode` inside the existing `web_scraper` result mapping from `load_and_log_configs()`.

Add to `config.txt` immediately after `web_outbound_policy_mode`:

```ini
# Browser transport admission: auto|disabled|url_guarded|attested_proxy
# auto allows legacy URL-guarded Playwright only in single_user + compat mode.
# attested_proxy still requires a concrete in-process governed transport attestation.
web_browser_transport_mode = auto
```

- [ ] **Step 7: Run focused tests and confirm green**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Web_Scraping/test_browser_transport.py tldw_Server_API/tests/Web_Scraping/test_outbound_policy.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit the contract/config slice**

```bash
git add tldw_Server_API/app/core/Web_Scraping/browser_transport.py tldw_Server_API/app/core/config.py tldw_Server_API/Config_Files/config.txt tldw_Server_API/tests/Web_Scraping/test_browser_transport.py tldw_Server_API/tests/Web_Scraping/test_outbound_policy.py
git commit -m "feat(web): define safe browser transport admission (TASK-13139.2)"
```

## Task 2: Gate Direct Article Browser Acquisition

**Files:**

- Modify: `tldw_Server_API/app/core/Web_Scraping/orchestration/article_browser.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/orchestration/article_models.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/orchestration/article.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/test_phase4_article_browser.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/test_phase4_article_models.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/test_phase4_article_orchestration.py`

**Success Criteria:** Unsafe browser admission fails before pool reservation or launch, the public result is distinguishable and bounded, valid attested admission still uses all existing browser routing controls, and lightweight HTTP success never consults the browser.

- [ ] **Step 1: Make existing browser tests independent of ambient auth config**

Extend the `_adapter()` test helper with an optional `transport_decision` callable. Its default must return the explicit single-user/compat allowed decision from `decide_browser_transport()`. This preserves the intent of existing browser lifecycle/routing tests and prevents unrelated environment state from changing them.

- [ ] **Step 2: Add failing admission tests**

Add tests that:

- inject a strict/multi-user denied decision;
- call `GuardedArticleBrowser.acquire()` with retries greater than zero;
- assert `ArticleFailure.code == "browser_transport_unavailable"` and `stage == "browser_transport_unattested"`;
- assert the fake launcher event list, egress-guard call list, and acquisition-pool active count are all empty/zero;
- inject a malformed provider that raises and a provider returning the wrong type, both producing `browser_transport_config_invalid` without launch;
- inject an allowed `attested_proxy` decision into the existing redirect/subresource/WebSocket dispatch scenario and assert the same fresh egress decisions and route ordering still occur.

- [ ] **Step 3: Add failing public-result tests**

Update the exact `PUBLIC_FAILURE_CODES` test to add `browser_transport_unavailable`. Add orchestration coverage asserting the exact public denial shape from this plan, while an ordinary `browser_error` retains its old shape with no capability object.

Add a credentialless HTTP test whose lightweight response extracts successfully while its browser fake raises if called. Run it with strict policy data in the harness and assert the normal extraction result is returned and the browser was not called.

- [ ] **Step 4: Run the direct-browser tests and confirm red**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Web_Scraping/test_phase4_article_browser.py tldw_Server_API/tests/Web_Scraping/test_phase4_article_models.py tldw_Server_API/tests/Web_Scraping/test_phase4_article_orchestration.py -q
```

Expected: FAIL because admission, the new code, and capability metadata are absent.

- [ ] **Step 5: Add the decision provider to `GuardedArticleBrowser`**

Add the constructor dependency:

```python
transport_decision: Callable[[], BrowserTransportDecision] = default_browser_transport_decision,
```

Add a private resolver that catches provider exceptions/wrong types and returns the fixed config-invalid denial. Add:

```python
def transport_capability(self) -> dict[str, str | bool]:
    return self._resolve_transport_decision().to_capability_metadata()
```

In `acquire()`, keep the existing `retries == 0` early return. Immediately afterward, resolve admission before entering the retry loop or touching `_acquisition_pool`. On denial, preserve the exact decision snapshot:

```python
ArticleFailure(
    "browser_transport_unavailable",
    decision.reason,
    capability=decision.to_capability_metadata(),
)
```

Do not change `_acquire_with_lease()`, `_playwright_has_required_routing()`, or any existing routing/accounting/cleanup logic.

- [ ] **Step 6: Add the bounded public failure metadata**

Add `browser_transport_unavailable` to `PUBLIC_FAILURE_CODES`.

Extend `ArticleFailure.__init__()` with a keyword-only `capability: Mapping[str, object] | None = None` and store a defensive immutable snapshot. Existing two-argument callers remain source compatible. When denying admission, raise:

```python
ArticleFailure(
    "browser_transport_unavailable",
    decision.reason,
    capability=decision.to_capability_metadata(),
)
```

At the public boundary, strictly validate that an attached capability has the exact six safe keys and only enumerated values/booleans; otherwise replace it with the fixed config-invalid denial metadata. Change `_failure_result()` to accept `ArticleFailure | str` and pass the instance through `article_failure_result()` so the captured configured/effective modes are not lost. Update direct browser catch sites to pass `exc`, not only `exc.code`. Make `_raw_failure_result()` accept the same union and include the validated capability object only for this code.

Do not add capability metadata to extraction, ordinary browser, fetch, policy, regex, selector, provider, or size failures.

- [ ] **Step 7: Run the direct-browser tests and confirm green**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Web_Scraping/test_phase4_article_browser.py tldw_Server_API/tests/Web_Scraping/test_phase4_article_models.py tldw_Server_API/tests/Web_Scraping/test_phase4_article_orchestration.py -q
```

Expected: PASS, including all pre-existing browser routing/lifecycle tests.

- [ ] **Step 8: Commit the article-browser gate**

```bash
git add tldw_Server_API/app/core/Web_Scraping/orchestration/article_browser.py tldw_Server_API/app/core/Web_Scraping/orchestration/article_models.py tldw_Server_API/app/core/Web_Scraping/orchestration/article.py tldw_Server_API/tests/Web_Scraping/test_phase4_article_browser.py tldw_Server_API/tests/Web_Scraping/test_phase4_article_models.py tldw_Server_API/tests/Web_Scraping/test_phase4_article_orchestration.py
git commit -m "fix(web): fail closed on unsafe article browser transport (TASK-13139.2)"
```

## Task 3: Gate the Existing Preflight Browser Probe

**Files:**

- Modify: `tldw_Server_API/app/core/Web_Scraping/preflight/probes.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/preflight/adapters/browser.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_browser.py`

**Success Criteria:** Preflight uses the same admission contract, denial happens before browser-budget reservation/launch, safe analyzer error codes survive isolation, and an attested path preserves route-before-page behavior.

- [ ] **Step 1: Make existing preflight browser tests explicit**

Extend the `_probe()` helper with an optional `transport_decision` callable and default it to the explicit single-user/compat allowed decision. Do not make existing route tests depend on ambient environment variables.

- [ ] **Step 2: Add failing preflight denial tests**

For each public denial reason, inject the matching decision and assert:

```python
with pytest.raises(ProbeUnavailable) as raised:
    async with probe.open_page(BrowserProbeOptions()):
        pytest.fail("page must not be created")

assert raised.value.error_code == expected_reason
assert controls.consumed.browsers == 0
assert launcher.events == []
```

Also cover provider exception/wrong type as `browser_transport_config_invalid`.

- [ ] **Step 3: Add an attested route-order regression**

Inject a complete attested-proxy decision into `test_browser_routes_before_page_and_blocks_service_workers` or an equivalent new test. Preserve the exact event order:

```python
[
    "launch",
    "launch_browser",
    "new_context:service_workers=block",
    "route_http",
    "route_web_socket",
    "new_page",
]
```

Retain the existing redirect, subresource, and WebSocket-policy tests unchanged apart from explicit admission injection.

- [ ] **Step 4: Run the preflight tests and confirm red**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_browser.py -q
```

Expected: FAIL because the probe has no transport admission and the safe error vocabulary lacks these codes.

- [ ] **Step 5: Extend the bounded probe error vocabulary**

Add the three denial codes to `_SAFE_ERROR_MESSAGES` with fixed messages and extend the `ProbeUnavailable` literal. Use the same generic safe message, `"Safe browser transport is unavailable."`, for all three codes; the code carries the structured reason.

- [ ] **Step 6: Enforce admission before capability and budget checks**

Add the same constructor dependency to `GuardedPlaywrightBrowserProbe`:

```python
transport_decision: Callable[[], BrowserTransportDecision] = default_browser_transport_decision,
```

At the beginning of `open_page()`, resolve it fail closed. If denied, raise:

```python
ProbeUnavailable(error_code=decision.reason)
```

This must run before `_capability_check()` and `await self._controls.reserve("browser")`. Do not change context creation, `service_workers="block"`, HTTP/WebSocket route installation, `new_page()`, navigation, or cleanup.

Add `transport_capability()` with the same safe metadata shape as the article adapter.

- [ ] **Step 7: Run the preflight tests and confirm green**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_browser.py tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_browser_analyzers.py -q
```

Expected: PASS with admission denial visible through the existing analyzer isolation contract.

- [ ] **Step 8: Commit the preflight gate**

```bash
git add tldw_Server_API/app/core/Web_Scraping/preflight/probes.py tldw_Server_API/app/core/Web_Scraping/preflight/adapters/browser.py tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_browser.py
git commit -m "fix(web): gate preflight browser transport (TASK-13139.2)"
```

## Task 4: Document, Verify, Review, and Finalize

**Files:**

- Modify: `Docs/ADR/026-security-outbound-egress-and-ssrf-policy.md`
- Modify: `Docs/Published/ADR/026-security-outbound-egress-and-ssrf-policy.md`
- Modify: `Docs/Design/WebScraping.md`
- Modify through official Backlog workflow: `TASK-13139.2`

**Success Criteria:** The limitation, decision matrix, configuration, and structured denial are documented accurately; focused and adjacent routing tests pass; Bandit finds no new touched-scope issue; and the Backlog task records exact evidence.

- [ ] **Step 1: Update ADR-026 and its published mirror**

Add a bounded browser-transport section that states:

- URL-policy evaluation checks names and resolved addresses before dispatch but cannot prove which address Chromium later connects to;
- Playwright route interception remains defense in depth for redirects, frames, subresources, HTTP, and WebSockets, but it is not DNS pinning or peer verification;
- single-user `compat` mode preserves the legacy URL-guarded path and reports `dns_peer_attested=false`;
- strict or multi-user mode requires explicit `attested_proxy` configuration plus a concrete all-request/DNS-pin/peer-verification attestation;
- Wave 0 includes no production attestor, proxy, credentialed browser, or cookie capability;
- authenticated browser work remains dependent on TASK-13100.

Keep the source and published ADR content synchronized.

- [ ] **Step 2: Update WebScraping design/configuration documentation**

Document:

- `WEB_BROWSER_TRANSPORT_MODE` and `[Web-Scraper] web_browser_transport_mode` values;
- the exact decision table from this plan;
- the direct article error and capability metadata shape;
- the preflight denial codes;
- credentialless governed HTTP remains available when Playwright is denied;
- setting `attested_proxy` alone does not grant access and there is no production attestor in this task;
- request-scoped cookies do not make an unattested transport safe and authenticated sessions remain unavailable.

- [ ] **Step 3: Run focused and adjacent regression verification**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Web_Scraping/test_browser_transport.py tldw_Server_API/tests/Web_Scraping/test_outbound_policy.py tldw_Server_API/tests/Web_Scraping/test_phase4_article_browser.py tldw_Server_API/tests/Web_Scraping/test_phase4_article_models.py tldw_Server_API/tests/Web_Scraping/test_phase4_article_orchestration.py tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_browser.py tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_browser_analyzers.py tldw_Server_API/tests/Web_Scraping/test_phase3_probe_egress.py -q
python -m bandit -r tldw_Server_API/app/core/Web_Scraping/browser_transport.py tldw_Server_API/app/core/Web_Scraping/orchestration/article_browser.py tldw_Server_API/app/core/Web_Scraping/orchestration/article_models.py tldw_Server_API/app/core/Web_Scraping/orchestration/article.py tldw_Server_API/app/core/Web_Scraping/preflight/probes.py tldw_Server_API/app/core/Web_Scraping/preflight/adapters/browser.py -f json -o /tmp/bandit_task_13139_2.json
git diff --check
```

Expected: pytest passes, including unchanged route/redirect/subresource/WebSocket coverage; Bandit exits zero with no new finding in touched code; `git diff --check` emits nothing.

- [ ] **Step 4: Perform security self-review**

Confirm:

- denial happens before both acquisition and preflight budgets and before Playwright starts;
- `auto` cannot allow strict or multi-user browser access;
- `url_guarded` cannot override strict/multi-user requirements;
- `attested_proxy` without an in-process complete attestation remains denied;
- malformed modes, auth values, policy values, provider exceptions, and wrong provider types fail closed;
- existing HTTP/WebSocket route installation still precedes `new_page()`;
- service workers remain blocked and redirect/subresource/popup checks remain intact;
- public output contains only enumerated strings and booleans;
- no cookie, credential, header, proxy URI, raw URL, DNS address, exception, or config text is exposed;
- no authenticated browser feature or proxy implementation slipped into scope.

- [ ] **Step 5: Request security-focused code review and address findings**

Use `superpowers:requesting-code-review`. Apply `superpowers:receiving-code-review` before changing code in response to findings. Rerun all focused verification after any change.

- [ ] **Step 6: Finalize TASK-13139.2**

Through the official Backlog workflow:

- check all acceptance criteria and Definition of Done items;
- record exact pytest, Bandit, and `git diff --check` results;
- link the ADR/design documentation and all implementation commits;
- document that no production attestor or authenticated browser session was added;
- add the final summary and set the task to Done only after every required check passes.

- [ ] **Step 7: Commit documentation and task finalization**

```bash
git add Docs/ADR/026-security-outbound-egress-and-ssrf-policy.md Docs/Published/ADR/026-security-outbound-egress-and-ssrf-policy.md Docs/Design/WebScraping.md backlog/tasks/task-13139.2\ -\ Gate-browser-fallback-against-unresolved-DNS-rebinding-risk.md
git commit -m "docs(web): explain browser transport safety gate (TASK-13139.2)"
```

## Final Acceptance Checklist

- [ ] Existing Playwright interception and route-order tests remain intact; no duplicate routing layer exists.
- [ ] Capability metadata states whether DNS/peer attestation is present and never equates URL checks with pinning.
- [ ] Multi-user or strict retrieval cannot launch Playwright without explicit complete attestation.
- [ ] Unsafe transport denial is distinct from outbound-policy, extraction, and ordinary browser failures.
- [ ] Complete injected attestation permits the existing guarded browser path; configuration alone never does.
- [ ] Credentialless governed HTTP extraction still succeeds when the browser path is denied.
- [ ] No proxy, resolver, persistent cookie state, authenticated browser session, or new dependency was added.
- [ ] Focused tests, adjacent routing regressions, Bandit, and whitespace verification pass and are recorded in TASK-13139.2.
