"""Who an LLM call is for, and how it is accounted (LLM usage accounting spec §1)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from fnmatch import fnmatchcase
from typing import Any
from urllib.parse import urlsplit

from tldw_Server_API.app.core.DB_Management.scope_context import get_scope

# The credential source of a call made with the user's own key on a provider that bills that key.
# The operator pays nothing for it, so the LLM quotas neither check nor count it (TASK-13955).
USER_KEY_SOURCE = "user"
# A user key on an endpoint the operator runs (a custom OpenAI-compatible slot, a local server, a
# billed provider pointed at a model server or gateway). It answers whatever key is sent, so the call
# counts like a server-key call.
USER_KEY_UNMETERED_SOURCE = "user_unmetered"
# Each provider that bills per key, and the hosts its vendor documents (fnmatch patterns; ``*`` is one DNS label).
# A user key is the user's own spend only there (TASK-13955).
VENDOR_HOSTS: dict[str, tuple[str, ...]] = {
    "anthropic": ("api.anthropic.com",),
    "bedrock": ("bedrock-runtime.*.amazonaws.com", "bedrock-runtime-fips.*.amazonaws.com"),
    "cohere": ("api.cohere.ai", "api.cohere.com"),
    "deepseek": ("api.deepseek.com",),
    "google": ("generativelanguage.googleapis.com",),
    "groq": ("api.groq.com",),
    "huggingface": ("api-inference.huggingface.co", "router.huggingface.co"),
    "mistral": ("api.mistral.ai",),
    "moonshot": ("api.moonshot.cn", "api.moonshot.ai"),
    "novita": ("api.novita.ai",),
    # and its data-residency domains, <region>.api.openai.com (developers.openai.com/api/docs/guides/your-data)
    "openai": ("api.openai.com", "*.api.openai.com"),
    "openrouter": ("openrouter.ai",),
    "poe": ("api.poe.com",),
    "qwen": ("dashscope-intl.aliyuncs.com", "dashscope-us.aliyuncs.com", "dashscope.aliyuncs.com"),
    "together": ("api.together.xyz",),
    "zai": ("api.z.ai",),
}
# Every provider-section key an adapter takes its base URL from.
_BASE_URL_KEYS = (
    "base_url", "api_base_url", "api_base", "api_url", "api_ip", "endpoint", "runtime_endpoint",
    "router_base_url", "huggingface_router_base_url",
)


@dataclass(frozen=True)
class LLMCallContext:
    """How LLM calls made in the current context are attributed and accounted."""

    user_id: int | None = None
    operation: str = "llm_call"
    endpoint: str = "internal"
    exempt: bool = False
    records_itself: bool = False
    prechecked: bool = False
    credential_source: str | None = None


@dataclass(frozen=True)
class LLMCallState:
    """A snapshot of the call context for a thread or queue hand-off."""

    context: LLMCallContext
    in_accounted_call: bool


_DEFAULT_CONTEXT = LLMCallContext()
_CTX: ContextVar[LLMCallContext] = ContextVar("llm_call_context", default=_DEFAULT_CONTEXT)
_IN_ACCOUNTED_CALL: ContextVar[bool] = ContextVar("llm_in_accounted_call", default=False)


def current_llm_call_context() -> LLMCallContext:
    """The call context in effect."""
    return _CTX.get()


def call_credential_source(request: Any = None) -> str | None:
    """Whose key a call uses: its request's runtime credential handle, else the call context's.

    A request that carries a runtime-issued handle is decided by the handle and the request's own base URL
    (``base_url``, ``api_base_url``).
    Never raises: a request that cannot be read is an unknown source, so its call is checked and counted.
    """
    try:
        from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
            PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
            is_runtime_issued_provider_call_credentials,
        )

        handle = request.get(PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY) if isinstance(request, Mapping) else None
        if is_runtime_issued_provider_call_credentials(handle):
            # A trusted caller's per-request base URL wins over app_config in every adapter, so it is read too.
            urls = (request.get("base_url"), request.get("api_base_url"))
            return billed_credential_source(handle.provider, handle.credential_source, base_urls=urls)
        return _CTX.get().credential_source
    except Exception:  # noqa: BLE001 - accounting must never break an LLM call
        return None


def billed_credential_source(
    provider: str | None, source: str | None, *configs: Any, base_urls: Iterable[Any] = ()
) -> str | None:
    """``source`` as the LLM quotas read it: a user key is the user's own spend only on its vendor's host.

    ``user`` stays ``user`` only for a provider in ``VENDOR_HOSTS`` whose base
    URLs in ``configs`` (a resolved ``app_config``, a server config snapshot) and
    in ``base_urls`` (a trusted caller's per-request override) are all on the
    vendor's host. Otherwise it becomes ``user_unmetered``, which counts: a custom
    OpenAI-compatible slot, a local server, an unknown provider, a billed provider
    with no config section, or a billed provider pointed at a model server or
    gateway. Every other source is returned unchanged, so nothing else ever
    becomes ``user``.
    """
    if source != USER_KEY_SOURCE:
        return source
    from tldw_Server_API.app.core.AuthNZ.byok_config import PROVIDER_APP_CONFIG_KEYS
    from tldw_Server_API.app.core.LLM_Calls.provider_identity import canonical_provider_name

    name = canonical_provider_name(provider) if provider else ""
    hosts = VENDOR_HOSTS.get(name)
    section_key = PROVIDER_APP_CONFIG_KEYS.get(name)
    if hosts is None or section_key is None:
        return USER_KEY_UNMETERED_SOURCE
    urls = list(base_urls)
    for config in configs:
        section = config.get(section_key) if isinstance(config, Mapping) else None
        if isinstance(section, Mapping):
            urls.extend(section.get(key) for key in _BASE_URL_KEYS)
    if any(isinstance(url, str) and url.strip() and not _on_vendor_host(url, hosts) for url in urls):
        return USER_KEY_UNMETERED_SOURCE
    return source


def _on_vendor_host(url: str, hosts: tuple[str, ...]) -> bool:
    """Whether ``url``'s host matches one of the vendor's host patterns, a ``*`` matching one DNS label."""
    try:
        host = urlsplit(url.strip()).hostname or ""
    except ValueError:  # a malformed URL is not the vendor's
        return False
    return any(fnmatchcase(host, pattern) and host.count(".") == pattern.count(".") for pattern in hosts)


def current_llm_user_id() -> int | None:
    """The user the next LLM call is for: the explicit one, else the content scope's, else None."""
    explicit = _CTX.get().user_id
    if explicit is not None:
        return explicit
    scope = get_scope()
    return scope.user_id if scope is not None else None


@contextmanager
def llm_call_context(**overrides: Any) -> Iterator[LLMCallContext]:
    """Overlay fields on the current call context for the duration of the block."""
    token = _CTX.set(replace(_CTX.get(), **overrides))
    try:
        yield _CTX.get()
    finally:
        _CTX.reset(token)


async def with_llm_call_context(awaitable: Awaitable[Any], **overrides: Any) -> Any:
    """Await ``awaitable`` with the overrides applied; for marking a long call without re-indenting it.

    The overrides apply to a coroutine this function awaits itself. A task that was
    started earlier has already copied its context and does not see them.
    """
    with llm_call_context(**overrides):
        return await awaitable


@contextmanager
def accounted_call() -> Iterator[None]:
    """Mark that an accounted provider call is running, so nested provider methods don't account again."""
    token = _IN_ACCOUNTED_CALL.set(True)
    try:
        yield
    finally:
        _IN_ACCOUNTED_CALL.reset(token)


def in_accounted_call() -> bool:
    """True inside an accounted provider call."""
    return _IN_ACCOUNTED_CALL.get()


def capture_llm_call_state() -> LLMCallState:
    """Snapshot the call state, resolving the user so the receiver doesn't need the content scope."""
    return LLMCallState(
        context=replace(_CTX.get(), user_id=current_llm_user_id()),
        in_accounted_call=_IN_ACCOUNTED_CALL.get(),
    )


def run_with_llm_call_state(state: LLMCallState | None, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` with a captured call state applied; with no state, just call it."""
    if state is None:
        return fn(*args, **kwargs)
    ctx_token = _CTX.set(state.context)
    flag_token = _IN_ACCOUNTED_CALL.set(state.in_accounted_call)
    try:
        return fn(*args, **kwargs)
    finally:
        _IN_ACCOUNTED_CALL.reset(flag_token)
        _CTX.reset(ctx_token)


def bind_llm_call_user(user_id: int | None) -> None:
    """Set the user for the rest of the current task; ``None`` binds nothing.

    For a task that serves one user for its whole life, such as a WebSocket
    handler. Everything else uses ``llm_call_context`` or ``llm_call_root``,
    which restore the previous user when they exit.
    """
    if user_id is not None:
        _CTX.set(replace(_CTX.get(), user_id=user_id))


def mark_llm_quota_prechecked() -> None:
    """Mark the rest of the current task as admitted by a boundary quota check (LLM usage accounting spec §4a).

    A plain set, not ``llm_call_context``: a ``with`` block is reset before the
    endpoint runs when the check is a dependency, and before a streaming
    response body runs. Tasks and threads started after this inherit the mark.
    """
    _CTX.set(replace(_CTX.get(), prechecked=True))
