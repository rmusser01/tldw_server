"""Certified single-attempt, byte-bounded native OpenAI completion transport."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, NoReturn, TypeVar
from urllib.parse import urlsplit

import httpx

from tldw_Server_API.app.core import http_client as hc
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCallCredentials,
    is_runtime_issued_provider_call_credentials,
)
from tldw_Server_API.app.core.exceptions import EgressPolicyError, raise_detached_error
from tldw_Server_API.app.core.LLM_Calls.provider_config_resolution import TrustedProviderEndpoint
from tldw_Server_API.app.core.MCP_unified.adapters.model_completion import normalization
from tldw_Server_API.app.core.MCP_unified.interfaces.model_completion import (
    ModelCompletionCapabilities,
    ModelCompletionFailure,
    ModelCompletionRequest,
    ModelFailureDomain,
)
from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

_T = TypeVar("_T")
_CAPABILITIES = ModelCompletionCapabilities(
    native_async_cancellation=True,
    response_limit_before_decode=True,
    native_max_output_tokens=True,
    tool_suppression=True,
    automatic_retries_disabled=True,
)
_UNCERTIFIED_CAPABILITIES = replace(_CAPABILITIES, native_async_cancellation=False)
_REQUEST_LIMITS = {
    "max_output_tokens": 8192,
    "max_output_chars": 100000,
    "max_output_bytes": 400000,
    "max_provider_response_bytes": 2000000,
}
_AMBIENT_CREDENTIAL_HEADERS = frozenset(
    {"authorization", "proxy-authorization", "cookie", "x-api-key", "api-key", "openai-organization", "openai-project"}
)


def _fail(code: str, domain: ModelFailureDomain) -> NoReturn:
    """Expose only a stable code, with no retained private exception chain."""
    raise_detached_error(ModelCompletionFailure(code, domain))


def _has_controls(text: str) -> bool:
    """Identify control characters unsuitable for operator identifiers or URLs."""
    return any(ord(char) < 0x20 or 0x7F <= ord(char) <= 0x9F for char in text)


def _canonical_endpoint(endpoint: object) -> TrustedProviderEndpoint:
    """Validate and copy one exact-origin capability with a canonical base slash."""
    if type(endpoint) is not TrustedProviderEndpoint or type(endpoint.scope) is not ConfiguredEndpointScope:
        raise ValueError("Invalid endpoint capability")
    if (
        type(endpoint.scope.scheme) is not str
        or type(endpoint.scope.host) is not str
        or type(endpoint.scope.port) is not int
    ):
        raise ValueError("Invalid endpoint scope types")
    base = endpoint.base_url
    if type(base) is not str or not base or _has_controls(base) or any(char.isspace() for char in base):
        raise ValueError("Invalid endpoint base")
    base.encode("utf-8", errors="strict")
    parsed = urlsplit(base)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or "?" in base
        or "#" in base
        or "\\" in base
    ):
        raise ValueError("Invalid endpoint URL")
    scope = ConfiguredEndpointScope.from_url(base)
    if scope != endpoint.scope:
        raise ValueError("Invalid endpoint scope")
    return TrustedProviderEndpoint(base_url=base.rstrip("/") + "/", scope=scope)


@dataclass(frozen=True, slots=True)
class OpenAITransportPolicy:
    """Operator-selected provider, model, endpoint, timeout and native token field."""

    provider: str
    model: str
    endpoint: TrustedProviderEndpoint
    timeout_seconds: int
    output_token_field: str = "max_completion_tokens"

    def __post_init__(self) -> None:
        if type(self.provider) is not str or self.provider != "openai":
            _fail("model_provider_unsupported", ModelFailureDomain.SHARED_INFRASTRUCTURE)
        invalid = False
        try:
            if (
                type(self.model) is not str
                or not 1 <= len(self.model) <= 256
                or self.model.strip() != self.model
                or _has_controls(self.model)
                or type(self.output_token_field) is not str
                or self.output_token_field not in ("max_tokens", "max_completion_tokens")
                or type(self.timeout_seconds) is not int
                or self.timeout_seconds < 1
                or not math.isfinite(float(self.timeout_seconds))
            ):
                raise ValueError("Invalid transport policy")
            self.model.encode("utf-8", errors="strict")
            object.__setattr__(self, "endpoint", _canonical_endpoint(self.endpoint))
        except Exception:  # noqa: BLE001 - operator state must not escape this boundary
            invalid = True
        if invalid:
            _fail("model_transport_policy_invalid", ModelFailureDomain.SHARED_INFRASTRUCTURE)


def snapshot_model_completion_request(request: object) -> ModelCompletionRequest:
    """Revalidate hard ceilings and snapshot the frozen request before any await."""
    invalid = False
    snapshot = None
    try:
        if type(request) is not ModelCompletionRequest:
            raise ValueError("Invalid request type")
        for name, maximum in _REQUEST_LIMITS.items():
            value = getattr(request, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError("Invalid request limit")
        if type(request.system_prompt) is not str or type(request.user_prompt) is not str:
            raise ValueError("Invalid prompt type")
        if len(request.system_prompt) + len(request.user_prompt) > 100000:
            raise ValueError("Prompt character limit exceeded")
        if (
            len(request.system_prompt.encode("utf-8", errors="strict"))
            + len(request.user_prompt.encode("utf-8", errors="strict"))
            > 400000
        ):
            raise ValueError("Prompt byte limit exceeded")
        snapshot = ModelCompletionRequest(
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            **{name: getattr(request, name) for name in _REQUEST_LIMITS},
        )
    except Exception:  # noqa: BLE001 - untrusted malformed requests fail closed
        invalid = True
    if invalid:
        _fail("model_request_invalid", ModelFailureDomain.REQUEST)
    return snapshot


def _credential_headers(credentials: object, policy: OpenAITransportPolicy) -> dict[str, str]:
    """Require authentic runtime issuance and matching server-owned endpoint state."""
    invalid = False
    headers = {}
    try:
        if not is_runtime_issued_provider_call_credentials(credentials, provider="openai"):
            raise ValueError("Invalid credential issuance")
        if credentials.provider != "openai" or credentials.endpoint_provenance != "server_config":
            raise ValueError("Invalid credential provenance")
        key = credentials.api_key
        if type(key) is not str or not key or any(not 0x21 <= ord(char) <= 0x7E for char in key):
            raise ValueError("Invalid credential header")
        if _canonical_endpoint(credentials.trusted_endpoint) != policy.endpoint:
            raise ValueError("Invalid credential endpoint")
        headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json", "Accept": "application/json"}
    except Exception:  # noqa: BLE001 - credential state is always scope-local and redacted
        invalid = True
    if invalid:
        _fail("model_credentials_invalid", ModelFailureDomain.CREDENTIAL_SCOPE)
    return headers


async def _await_owned_operation(operation: Awaitable[_T], *, cancel_on_cancellation: bool) -> _T:
    """Drain owned native I/O or cleanup even through repeated caller cancellation.

    Only the first caller cancellation reaches provider I/O. Further cancellation
    cannot interrupt that operation's response cleanup or orphan client closure.
    The adapter above this transport owns containment of a non-cooperative child.
    """
    task = asyncio.ensure_future(operation)
    cancellation = None
    cancellation_forwarded = False
    while not task.done():
        try:
            # Unlike shield, wait cannot log a raw late exception after cancellation.
            await asyncio.wait({task})
        except asyncio.CancelledError as error:
            if cancellation is None:
                cancellation = error
                if cancel_on_cancellation and not task.done():
                    cancellation_forwarded = task.cancel(error.args[0] if error.args else None)
    if cancellation is not None:
        if task.cancelled():
            if cancellation_forwarded:
                task.result()
        else:
            task.exception()
        raise cancellation
    return task.result()


class OpenAICompletionTransport:
    """One accounted request using fresh HTTPX state and bounded shared JSON I/O.

    The optional callables are trusted test seams, never request-controlled
    provider factories or HTTP configuration. No provider SDK is involved.
    """

    __slots__ = ("_policy", "_fetch_json", "_client_factory", "_capabilities")

    def __init__(
        self,
        policy: OpenAITransportPolicy,
        *,
        fetch_json: Callable[..., Awaitable[Any]] | None = None,
        client_factory: Callable[..., httpx.AsyncClient] | None = None,
    ) -> None:
        if type(policy) is not OpenAITransportPolicy:
            _fail("model_transport_policy_invalid", ModelFailureDomain.SHARED_INFRASTRUCTURE)
        self._policy = policy
        self._fetch_json = fetch_json if fetch_json is not None else hc.afetch_json
        self._client_factory = client_factory if client_factory is not None else hc.create_async_client
        self._capabilities = _CAPABILITIES
        try:
            self._check_selected_host_pins(hc._parse_pins_from_env())
        except Exception:  # noqa: BLE001 - unknown pin readiness cannot certify cancellation
            self._capabilities = _UNCERTIFIED_CAPABILITIES

    def _check_selected_host_pins(self, pins: Mapping[str, set[str]] | None) -> None:
        """Latch uncertified without disabling operator pins or probing TLS.

        The shared certificate check opens a blocking socket. Until it has an
        async implementation, only this endpoint's configured pins disqualify
        the transport; unrelated hosts do not affect the selected path.
        """
        if pins and pins.get(self._policy.endpoint.scope.host):
            self._capabilities = _UNCERTIFIED_CAPABILITIES

    @property
    def policy(self) -> OpenAITransportPolicy:
        """Return the immutable operator policy without issuing a discovery call."""
        return self._policy

    @property
    def capabilities(self) -> ModelCompletionCapabilities:
        """Return stable frozen flags, replaced only by the uncertified latch."""
        return self._capabilities

    async def complete(
        self,
        request: ModelCompletionRequest,
        credentials: ProviderCallCredentials,
    ) -> normalization.NormalizedModelCompletion:
        """Dispatch once, normalize atomically, close owned state, and detach failures."""
        if not self._capabilities.native_async_cancellation:
            _fail("model_transport_uncertified", ModelFailureDomain.SHARED_INFRASTRUCTURE)
        request = snapshot_model_completion_request(request)
        headers = _credential_headers(credentials, self._policy)
        payload = {
            "model": self._policy.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "stream": False,
            "n": 1,
            "tools": None,
            self._policy.output_token_field: request.max_output_tokens,
        }
        response_received = False
        response_rejected = False

        async def on_response(status: int, _headers: Mapping[str, str]) -> None:
            nonlocal response_received, response_rejected
            response_received = True
            if not 200 <= status < 300:
                response_rejected = True
                _fail("model_response_rejected", ModelFailureDomain.REQUEST)

        client = None
        failure = None
        cancellation = None
        result = None
        phase = "client"
        try:
            client = self._client_factory(trust_env=False, timeout=float(self._policy.timeout_seconds))
            if not isinstance(client, httpx.AsyncClient):
                raise ValueError("Native HTTPX client required")
            phase = "certification"
            self._check_selected_host_pins(hc._get_client_cert_pins(client))
            if not self._capabilities.native_async_cancellation:
                _fail("model_transport_uncertified", ModelFailureDomain.SHARED_INFRASTRUCTURE)
            phase = "headers"
            if (
                client.auth is not None
                or list(client.cookies.jar)
                or _AMBIENT_CREDENTIAL_HEADERS.intersection(client.headers)
            ):
                raise ValueError("Ambient credential state is forbidden")
            phase = "fetch"
            envelope = await _await_owned_operation(
                self._fetch_json(
                    method="POST",
                    url=self._policy.endpoint.base_url + "chat/completions",
                    client=client,
                    headers=headers,
                    json=payload,
                    timeout=float(self._policy.timeout_seconds),
                    max_bytes=request.max_provider_response_bytes,
                    configured_endpoint=credentials.trusted_endpoint.scope,
                    allow_redirects=False,
                    retry=hc.RetryPolicy(attempts=1),
                    sensitive_observability=True,
                    require_json_ct=True,
                    on_response=on_response,
                ),
                cancel_on_cancellation=True,
            )
            phase = "normalize"
            result = normalization.normalize_model_completion_response(envelope, request)
        except asyncio.CancelledError as error:
            cancellation = error
        except EgressPolicyError:
            failure = ModelCompletionFailure("model_egress_denied", ModelFailureDomain.REQUEST)
        except Exception:  # noqa: BLE001 - phase, not exception class, establishes provenance
            if phase == "certification":
                self._capabilities = _UNCERTIFIED_CAPABILITIES
                failure = ModelCompletionFailure(
                    "model_transport_uncertified", ModelFailureDomain.SHARED_INFRASTRUCTURE
                )
            elif phase == "headers":
                failure = ModelCompletionFailure("model_credentials_invalid", ModelFailureDomain.CREDENTIAL_SCOPE)
            elif phase == "normalize":
                failure = ModelCompletionFailure("invalid_model_output", ModelFailureDomain.REQUEST)
            elif response_received:
                failure = ModelCompletionFailure(
                    "model_response_rejected" if response_rejected else "model_response_invalid",
                    ModelFailureDomain.REQUEST,
                )
            else:
                failure = ModelCompletionFailure(
                    "model_transport_unavailable", ModelFailureDomain.SHARED_INFRASTRUCTURE
                )
        finally:
            if client is not None:
                try:
                    await _await_owned_operation(client.aclose(), cancel_on_cancellation=False)
                except asyncio.CancelledError as error:
                    if cancellation is None:
                        cancellation = error
                except Exception:  # noqa: BLE001 - cleanup errors must also be detached
                    if failure is None:
                        failure = ModelCompletionFailure(
                            "model_response_invalid" if response_received else "model_transport_unavailable",
                            (
                                ModelFailureDomain.REQUEST
                                if response_received
                                else ModelFailureDomain.SHARED_INFRASTRUCTURE
                            ),
                        )
        if cancellation is not None:
            raise cancellation
        if failure is not None:
            raise_detached_error(failure)
        return result
