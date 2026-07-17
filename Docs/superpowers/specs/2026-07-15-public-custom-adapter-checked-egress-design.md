# Public Custom Adapter Checked Egress Design

**Task:** TASK-12972.1

**Status:** Approved for implementation planning

**Date:** 2026-07-15

## Context

`NovitaAdapter`, `PoeAdapter`, and `TogetherAdapter` inherit from `CustomOpenAIAdapter`. Configured custom OpenAI slots already send non-streaming and streaming requests through the central checked HTTP helpers, but these three public-provider subclasses use a separate module-level `http_client_factory` path. That legacy path constructs the shared HTTPX client without invoking the central outbound policy boundary.

TASK-12972 deliberately preserved this compatibility seam while configured-local endpoint scope was introduced. This follow-up removes the seam without granting public providers the exact-origin exception reserved for trusted configured-local endpoints.

## Goals

- Route Novita, Poe, and Together chat and stream requests through the existing checked `fetch` and `stream_response` helpers.
- Keep public-provider requests on ordinary egress policy with `configured_endpoint=None`.
- Preserve base URL selection, authentication headers, payload shaping, timeout handling, streaming normalization, response cleanup, and error mapping.
- Cover synchronous and asynchronous adapter entry points and reject caller attempts to manufacture transport scope.

## Non-goals

- Do not add a new transport, provider base class, configuration option, endpoint resolver, or dependency.
- Do not grant `ConfiguredEndpointScope` to a public provider or relax global egress policy.
- Do not replace the existing thread-backed `achat` and `astream` wrappers with native async HTTP.
- Do not change provider catalog, setup, WebUI, or browser-extension behavior.
- Do not retain the module-level `http_client_factory` as an alternate outbound path.

## Chosen design

Use one checked transport path in `CustomOpenAIAdapter` for every subclass.

For non-streaming calls, `chat` will always call the adapter's `http_fetcher`. Configured custom slots will continue to pass their trusted scope. Novita, Poe, and Together resolve their existing public base URL and pass `configured_endpoint=None`, so the central helper applies ordinary host, port, private-address, DNS, and TLS checks.

The legacy public client path did not follow redirects. The checked fetch call will therefore set `allow_redirects=False` for public subclasses while leaving configured custom behavior unchanged. Central `RetryPolicy` also preserves the effective public POST behavior because unsafe methods are not retried unless explicitly enabled.

For streaming calls, `stream` will always enter the adapter's `http_streamer` context with the same scoped-or-ordinary distinction. `stream_response` already disables redirects and owns checked connection setup. Existing SSE normalization and final `[DONE]` handling remain unchanged.

`achat` and `astream` continue to delegate to `chat` and `stream`, so they inherit the checked path without another network implementation.

## Compatibility boundary

Provider defaults, supported environment variables, `app_config`, and the existing public `base_url` request override keep the same resolution precedence and URL construction. A selected public URL can now fail ordinary egress policy when it targets a private or otherwise forbidden address, a disallowed host, or a disallowed port. That denial is the intended security change; the adapter must not fall back to the legacy client factory or manufacture scope to preserve reachability.

Public POST redirects remain disabled, matching the legacy HTTPX client behavior. Default central retry policy also retains one effective POST attempt because unsafe methods are not retried unless explicitly enabled. Configured custom-slot redirect behavior remains unchanged.

## Injection and request boundary

The class-level `http_fetcher` and `http_streamer` hooks remain injectable per adapter for deterministic tests and existing internal transport substitution. The module-level `http_client_factory` symbol and its production branches are removed.

Request-owned `http_client_factory`, `http_fetcher`, `http_streamer`, `configured_endpoint`, `configured_endpoint_base_url`, and `configured_endpoint_scope` fields remain reserved and are stripped before validation and payload construction. A caller can select the supported public `base_url` compatibility override, but that URL still receives ordinary checked egress and cannot obtain scope.

## Error and resource behavior

- `EgressPolicyError` remains unmodified so its machine-readable `reason_code` reaches the caller.
- HTTP status failures continue through `CustomOpenAIAdapter.normalize_error` and retain provider-specific authentication, bad-request, rate-limit, and upstream mappings.
- Non-streaming responses close in `finally`, including JSON and error failures.
- Streaming responses remain owned by the checked context manager and close on normal completion, provider failure, policy failure, and consumer cancellation.

## Test strategy

Focused tests will be updated test-first:

1. Parameterize Novita, Poe, and Together non-streaming calls and assert URL suffix, authorization, payload, timeout, `allow_redirects=False`, response closure, checked fetch usage, and `configured_endpoint is None`.
2. Parameterize streaming calls and assert checked streamer usage, no scope, timeout propagation, context cleanup, SSE normalization, and one `[DONE]` event.
3. Exercise `achat` and `astream` for all three providers and prove they reach the same checked hooks.
4. Supply forged reserved transport/scope fields and prove they are neither validated nor serialized and cannot change `configured_endpoint=None`.
5. Raise representative `EgressPolicyError` values through sync and async modes and verify their reason codes survive.
6. Retain configured custom-slot regressions to prove trusted scope and request/BYOK ordinary-egress behavior are unchanged.

The focused baseline is 34 passing tests across `test_custom_openai_native_http.py` and `test_openai_compatible_provider_adapters.py`. Final verification will include those suites, adjacent custom OpenAI adapter tests, a static check that no production client-factory seam remains, scoped Ruff checks, Python compilation, Bandit on the production file, and `git diff --check`.

## Documentation

ADR-030 will be updated only to record that TASK-12972.1 completed the previously deferred public-provider transport migration. No user-facing configuration changes are required.
