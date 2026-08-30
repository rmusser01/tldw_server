"""Strict endpoint-neutral embedding execution for canonical Notes chunks."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import math
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

from tldw_Server_API.app.core.AuthNZ.byok_config import (
    PROVIDER_APP_CONFIG_KEYS,
    runtime_base_url_override_provenance,
)
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionStatus,
    ResolvedByokCredentials,
    resolve_byok_credentials,
)
from tldw_Server_API.app.core.Embeddings.orchestrator import EmbeddingRequestOrchestrator
from tldw_Server_API.app.core.Embeddings.request_types import EmbeddingRequestContext
from tldw_Server_API.app.core.Embeddings.vector_validation import (
    validated_embedding_vectors,
    validated_indexed_embedding_data,
)
from tldw_Server_API.app.core.LLM_Calls.embeddings_adapter_registry import (
    get_embeddings_registry,
)

from .semantic_content import SemanticChunkInput
from .semantic_settings import DEFAULT_SEMANTIC_INDEX_SETTINGS, SemanticIndexSettings

DIMENSION_PROBE_TEXT = "Public semantic embedding dimension probe."
_REVISION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}\Z")
_ENDPOINT_FIELDS = (
    "base_url",
    "api_base_url",
    "api_base",
    "api_url",
    "api_ip",
    "endpoint",
    "runtime_endpoint",
)

CredentialScope = Literal["user", "server_default"]
CredentialResolver = Callable[..., Awaitable[ResolvedByokCredentials]]
UsageLogger = Callable[..., Awaitable[None]]
DimensionCas = Callable[
    ["PendingSemanticConfig", "ResolvedDimension"],
    bool | Awaitable[bool],
]


class SemanticEmbeddingSystemError(RuntimeError):
    """A stable, content-free systemic semantic embedding failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _origin(value: str | None) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return None
        port = parsed.port
    except ValueError:
        return None
    return f"{parsed.scheme}://{parsed.hostname.lower()}{f':{port}' if port is not None else ''}"


def _safe_revision(value: object) -> str | None:
    if not isinstance(value, str) or _REVISION_PATTERN.fullmatch(value) is None:
        return None
    return value


@dataclass(frozen=True, slots=True)
class PendingSemanticConfig:
    """Consent-bound provider identity before dimensions are necessarily known."""

    provider: str
    model: str
    model_revision: str | None
    endpoint_origin: str | None
    credential_source: CredentialScope
    consented: bool
    dimensions: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider must be non-empty")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be non-empty")
        if self.model_revision is not None and _safe_revision(self.model_revision) is None:
            raise ValueError("model_revision is invalid")
        if self.endpoint_origin is not None and _origin(self.endpoint_origin) != self.endpoint_origin:
            raise ValueError("endpoint_origin must be a canonical HTTP origin")
        if self.credential_source not in {"user", "server_default"}:
            raise ValueError("credential_source must identify one durable scope")
        if type(self.consented) is not bool:
            raise TypeError("consented must be a boolean")
        if self.dimensions is not None and (
            type(self.dimensions) is not int or self.dimensions <= 0
        ):
            raise ValueError("dimensions must be a positive integer or None")


@dataclass(frozen=True, slots=True)
class ResolvedSemanticConfig:
    """Pinned semantic provider identity with an exact vector dimension."""

    provider: str
    model: str
    model_revision: str | None
    endpoint_origin: str | None
    credential_source: CredentialScope
    dimensions: int

    def __post_init__(self) -> None:
        PendingSemanticConfig(
            provider=self.provider,
            model=self.model,
            model_revision=self.model_revision,
            endpoint_origin=self.endpoint_origin,
            credential_source=self.credential_source,
            consented=True,
            dimensions=self.dimensions,
        )


@dataclass(frozen=True, slots=True)
class ResolvedDimension:
    dimensions: int
    provider: str
    model: str
    model_revision: str | None


@dataclass(frozen=True, slots=True)
class SemanticEmbeddingBatch:
    vectors: tuple[tuple[float, ...], ...]
    provider: str
    model: str
    model_revision: str | None
    dimensions: int
    prompt_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class NotesEmbeddingExecutionIdentity:
    model_revision: str | None = None
    endpoint_origin: str | None = None
    endpoint_base_url: str | None = None
    provider_attempt_sequence: int = 0
    provider_input_count: int = 0
    provider_prompt_tokens: int = 0
    provider_request_count: int = 0
    provider_status: str | None = None


@dataclass(frozen=True, slots=True)
class NotesEmbeddingRuntime:
    orchestrator: Any
    execution_identity: Callable[[], object]


class AdapterRegistry(Protocol):
    def get_adapter(self, name: str) -> object | None: ...


class RunMemoryEmbeddingCache:
    """One-run vector cache with no durable or cross-run state."""

    def __init__(self) -> None:
        self._values: dict[str, list[float]] = {}

    async def get(self, key: str) -> list[float] | None:
        value = self._values.get(key)
        return list(value) if value is not None else None

    async def set(self, key: str, value: list[float]) -> None:
        self._values[key] = list(value)


class NotesEmbeddingExecutor:
    """Resolve durable credentials and invoke public embedding provider adapters."""

    def __init__(
        self,
        *,
        config: PendingSemanticConfig | ResolvedSemanticConfig,
        user_id: str,
        credential_resolver: CredentialResolver = resolve_byok_credentials,
        adapter_registry: AdapterRegistry | None = None,
    ) -> None:
        self._config = config
        self._user_id = _owner_user_id(user_id)
        self._credential_resolver = credential_resolver
        self._adapter_registry = adapter_registry or get_embeddings_registry()
        self._identity = NotesEmbeddingExecutionIdentity(
            model_revision=config.model_revision,
            endpoint_origin=config.endpoint_origin,
        )

    def execution_identity(self) -> NotesEmbeddingExecutionIdentity:
        return self._identity

    async def create(
        self,
        texts: list[str],
        *,
        provider: str,
        model: str,
        dimensions: int | None,
    ) -> list[list[float]]:
        if provider != self._config.provider or model != self._config.model:
            raise SemanticEmbeddingSystemError("provider_model_drift")
        if dimensions != self._config.dimensions:
            raise SemanticEmbeddingSystemError("dimension_mismatch")
        adapter = self._adapter_registry.get_adapter(provider)
        if adapter is None or not callable(getattr(adapter, "embed", None)):
            raise SemanticEmbeddingSystemError("provider_unavailable")

        try:
            credentials = await self._credential_resolver(
                provider,
                user_id=self._user_id,
                request=None,
                required_source=self._config.credential_source,
            )
        except Exception:  # noqa: BLE001 - credential details must not cross this boundary
            raise SemanticEmbeddingSystemError("durable_credentials_unavailable") from None
        if (
            credentials.source != self._config.credential_source
            or not credentials.allowlisted
            or credentials.status != ByokResolutionStatus.RESOLVED
        ):
            raise SemanticEmbeddingSystemError("durable_credentials_unavailable")
        resolved_base_url = _credential_base_url(credentials, provider)
        endpoint_origin = _origin(resolved_base_url)
        if endpoint_origin != self._config.endpoint_origin:
            raise SemanticEmbeddingSystemError("endpoint_origin_mismatch")
        if resolved_base_url is None:
            raise SemanticEmbeddingSystemError("endpoint_origin_mismatch")
        previous_identity = self._identity
        if (
            previous_identity.endpoint_base_url is not None
            and resolved_base_url != previous_identity.endpoint_base_url
        ):
            raise SemanticEmbeddingSystemError("endpoint_identity_mismatch")

        app_config = copy.deepcopy(credentials.app_config or {})
        request: dict[str, object] = {
            "input": list(texts),
            "model": model,
            "api_key": credentials.api_key,
            "app_config": app_config,
            "credentials_resolved": True,
            "base_url": resolved_base_url,
            "_runtime_base_url_override": runtime_base_url_override_provenance(),
        }
        if dimensions is not None:
            request["dimensions"] = dimensions

        attempt_sequence = previous_identity.provider_attempt_sequence + 1
        provider_prompt_tokens = sum(_count_tokens(text, model) for text in texts)
        self._identity = NotesEmbeddingExecutionIdentity(
            model_revision=previous_identity.model_revision,
            endpoint_origin=endpoint_origin,
            endpoint_base_url=resolved_base_url,
            provider_attempt_sequence=attempt_sequence,
            provider_input_count=len(texts),
            provider_prompt_tokens=provider_prompt_tokens,
            provider_request_count=1,
            provider_status="started",
        )
        try:
            response = await asyncio.to_thread(adapter.embed, request)
            if inspect.isawaitable(response):
                response = await response
            if not isinstance(response, dict):
                raise SemanticEmbeddingSystemError("invalid_vectors")
            response_model = response.get("model")
            if response_model not in {None, "", model}:
                raise SemanticEmbeddingSystemError("provider_model_drift")
            vectors = validated_indexed_embedding_data(
                response.get("data"),
                expected=len(texts),
            )
            if vectors is None:
                raise SemanticEmbeddingSystemError("invalid_vectors")
            actual_revision = _safe_revision(
                response.get("model_revision") or response.get("model_digest")
            )
            if actual_revision is None:
                capabilities = (
                    adapter.capabilities()
                    if callable(getattr(adapter, "capabilities", None))
                    else {}
                )
                if isinstance(capabilities, dict):
                    actual_revision = _safe_revision(
                        capabilities.get("model_revision")
                        or capabilities.get("model_digest")
                    )
            if (
                self._config.model_revision is not None
                and actual_revision is not None
                and actual_revision != self._config.model_revision
            ):
                raise SemanticEmbeddingSystemError("model_revision_drift")
            if (
                self._config.model_revision is None
                and previous_identity.model_revision is not None
                and actual_revision != previous_identity.model_revision
            ):
                raise SemanticEmbeddingSystemError("model_revision_drift")
            await credentials.touch_last_used()
        except asyncio.CancelledError:
            self._identity = NotesEmbeddingExecutionIdentity(
                model_revision=previous_identity.model_revision,
                endpoint_origin=endpoint_origin,
                endpoint_base_url=resolved_base_url,
                provider_attempt_sequence=attempt_sequence,
                provider_input_count=len(texts),
                provider_prompt_tokens=provider_prompt_tokens,
                provider_request_count=1,
                provider_status="failed",
            )
            raise
        except SemanticEmbeddingSystemError:
            self._identity = NotesEmbeddingExecutionIdentity(
                model_revision=previous_identity.model_revision,
                endpoint_origin=endpoint_origin,
                endpoint_base_url=resolved_base_url,
                provider_attempt_sequence=attempt_sequence,
                provider_input_count=len(texts),
                provider_prompt_tokens=provider_prompt_tokens,
                provider_request_count=1,
                provider_status="failed",
            )
            raise
        except Exception:  # noqa: BLE001 - provider details must not cross this boundary
            self._identity = NotesEmbeddingExecutionIdentity(
                model_revision=previous_identity.model_revision,
                endpoint_origin=endpoint_origin,
                endpoint_base_url=resolved_base_url,
                provider_attempt_sequence=attempt_sequence,
                provider_input_count=len(texts),
                provider_prompt_tokens=provider_prompt_tokens,
                provider_request_count=1,
                provider_status="failed",
            )
            raise SemanticEmbeddingSystemError("provider_execution_failed") from None
        self._identity = NotesEmbeddingExecutionIdentity(
            model_revision=actual_revision or self._config.model_revision,
            endpoint_origin=endpoint_origin,
            endpoint_base_url=resolved_base_url,
            provider_attempt_sequence=attempt_sequence,
            provider_input_count=len(texts),
            provider_prompt_tokens=provider_prompt_tokens,
            provider_request_count=1,
            provider_status="success",
        )
        return vectors


def _owner_user_id(user_id: str) -> int:
    try:
        value = int(user_id)
    except (TypeError, ValueError) as exc:
        raise SemanticEmbeddingSystemError("invalid_owner") from exc
    if value <= 0 or str(value) != str(user_id):
        raise SemanticEmbeddingSystemError("invalid_owner")
    return value


def _credential_base_url(
    credentials: ResolvedByokCredentials,
    provider: str,
) -> str | None:
    direct = credentials.credential_fields.get("base_url")
    normalized = _normalize_base_url(direct)
    if normalized is not None:
        return normalized
    app_config = credentials.app_config or {}
    section_name = PROVIDER_APP_CONFIG_KEYS.get(provider)
    section = app_config.get(section_name) if section_name else None
    if isinstance(section, dict):
        for field_name in _ENDPOINT_FIELDS:
            candidate = section.get(field_name)
            normalized = _normalize_base_url(candidate)
            if normalized is not None:
                return normalized
    return None


def _normalize_base_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().rstrip("/")
    return normalized if _origin(normalized) is not None else None


def _cache_key(
    text: str,
    provider: str,
    model: str,
    dimensions: int | None,
    backend_identity: str | None,
) -> str:
    material = "\x1f".join(
        (provider, model, str(dimensions or ""), backend_identity or "", text)
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _count_tokens(text: str, model: str) -> int:
    del model
    return (len(text.encode("utf-8")) + 3) // 4


def _reject_token_input(tokens_input: object, model: str) -> object:
    del tokens_input, model
    raise SemanticEmbeddingSystemError("token_input_unsupported")


def build_notes_semantic_orchestrator(
    config: PendingSemanticConfig | ResolvedSemanticConfig,
    *,
    user_id: str,
    settings: SemanticIndexSettings = DEFAULT_SEMANTIC_INDEX_SETTINGS,
    executor: object | None = None,
    credential_resolver: CredentialResolver = resolve_byok_credentials,
    adapter_registry: AdapterRegistry | None = None,
) -> NotesEmbeddingRuntime:
    """Build one strict run-scoped orchestrator with no durable cache or fallback."""

    resolved_executor = executor or NotesEmbeddingExecutor(
        config=config,
        user_id=user_id,
        credential_resolver=credential_resolver,
        adapter_registry=adapter_registry,
    )
    cache = RunMemoryEmbeddingCache()
    backend_identity = "|".join(
        (
            config.provider,
            config.model,
            config.model_revision or "",
            config.endpoint_origin or "",
            str(config.dimensions or "pending"),
        )
    )
    orchestrator = EmbeddingRequestOrchestrator(
        count_tokens=_count_tokens,
        tokens_to_texts=_reject_token_input,
        cache_key_fn=_cache_key,
        cache=cache,
        executor=resolved_executor,  # type: ignore[arg-type]
        settings_config={},
        max_tokens=max(1, (settings.max_provider_input_bytes + 3) // 4),
        implemented_providers={config.provider},
        allowed_providers={config.provider},
        allowed_models={config.model},
        enforce_policy=True,
        allow_fallback_with_header=False,
        settings_fallback_chain={},
        settings_fallback_model_map={},
        dimension_policy="ignore",
        backend_identity_resolver=lambda provider, model: backend_identity,
        cache_namespace=backend_identity,
        batch_size=settings.max_provider_batch_inputs,
    )
    identity_reader = getattr(resolved_executor, "execution_identity", None)
    return NotesEmbeddingRuntime(
        orchestrator=orchestrator,
        execution_identity=(
            identity_reader
            if callable(identity_reader)
            else lambda: NotesEmbeddingExecutionIdentity(model_revision=config.model_revision)
        ),
    )


async def _default_usage_logger(**kwargs: object) -> None:
    from tldw_Server_API.app.core.Usage.usage_tracker import log_llm_usage

    await log_llm_usage(**kwargs)  # type: ignore[arg-type]


class NotesSemanticEmbedder:
    """Apply strict semantic policy around the shared embedding orchestrator."""

    def __init__(
        self,
        *,
        orchestrator_factory: Callable[
            [PendingSemanticConfig | ResolvedSemanticConfig, str], NotesEmbeddingRuntime
        ]
        | None = None,
        dimension_cas: DimensionCas | None = None,
        usage_logger: UsageLogger = _default_usage_logger,
        settings: SemanticIndexSettings = DEFAULT_SEMANTIC_INDEX_SETTINGS,
    ) -> None:
        self._settings = settings
        self._orchestrator_factory = orchestrator_factory or (
            lambda config, user_id: build_notes_semantic_orchestrator(
                config,
                user_id=user_id,
                settings=settings,
            )
        )
        self._dimension_cas = dimension_cas
        self._usage_logger = usage_logger

    async def resolve_dimensions(
        self,
        config: PendingSemanticConfig,
        *,
        user_id: str,
    ) -> ResolvedDimension:
        """Resolve an unknown dimension with one non-user probe and fenced CAS."""

        if config.dimensions is not None:
            return ResolvedDimension(
                dimensions=config.dimensions,
                provider=config.provider,
                model=config.model,
                model_revision=config.model_revision,
            )
        if not config.consented:
            raise SemanticEmbeddingSystemError("consent_required")
        runtime = self._orchestrator_factory(config, user_id)
        before_sequence = _provider_attempt_sequence(runtime)
        result = None
        try:
            result = await self._execute(
                runtime,
                [DIMENSION_PROBE_TEXT],
                config,
                user_id=user_id,
            )
            _validate_execution_result(result, config)
            vectors = _strict_vectors(result.vectors, expected=1, dimensions=None)
            dimensions = len(vectors[0])
            identity = runtime.execution_identity()
            model_revision = (
                _safe_revision(getattr(identity, "model_revision", None))
                or config.model_revision
            )
        except asyncio.CancelledError:
            await self._record_cancelled_provider_usage(
                runtime=runtime,
                result=result,
                before_sequence=before_sequence,
                config=config,
                user_id=user_id,
                operation="notes_semantic_dimension_probe",
            )
            raise
        except Exception:
            await self._record_provider_usage(
                runtime=runtime,
                result=result,
                before_sequence=before_sequence,
                config=config,
                user_id=user_id,
                operation="notes_semantic_dimension_probe",
                succeeded=False,
            )
            raise
        await self._record_provider_usage(
            runtime=runtime,
            result=result,
            before_sequence=before_sequence,
            config=config,
            user_id=user_id,
            operation="notes_semantic_dimension_probe",
            succeeded=True,
        )
        if self._dimension_cas is None:
            raise SemanticEmbeddingSystemError("dimension_cas_unavailable")
        resolved_dimension = ResolvedDimension(
            dimensions=dimensions,
            provider=config.provider,
            model=config.model,
            model_revision=model_revision,
        )
        published = self._dimension_cas(config, resolved_dimension)
        if inspect.isawaitable(published):
            published = await published
        if published is not True:
            raise SemanticEmbeddingSystemError("dimension_cas_lost")
        return resolved_dimension

    async def embed_chunks(
        self,
        chunks: Sequence[SemanticChunkInput],
        config: ResolvedSemanticConfig,
        *,
        user_id: str,
    ) -> SemanticEmbeddingBatch:
        """Embed a fully admitted bounded run without fallback or durable cache."""

        batches = self._plan_batches(chunks)
        runtime = self._orchestrator_factory(config, user_id)
        vectors: list[list[float]] = []
        prompt_tokens = 0
        total_tokens = 0
        model_revision = config.model_revision
        discovered_revision = config.model_revision
        for batch in batches:
            texts = [chunk.provider_input.text for chunk in batch]
            before_sequence = _provider_attempt_sequence(runtime)
            result = None
            try:
                result = await self._execute(runtime, texts, config, user_id=user_id)
                _validate_execution_result(result, config)
                batch_vectors = _strict_vectors(
                    result.vectors,
                    expected=len(batch),
                    dimensions=config.dimensions,
                )
                identity = runtime.execution_identity()
                actual_revision = _safe_revision(
                    getattr(identity, "model_revision", None)
                )
                if config.model_revision is None:
                    if discovered_revision is None:
                        discovered_revision = actual_revision
                    elif actual_revision != discovered_revision:
                        raise SemanticEmbeddingSystemError("model_revision_drift")
                elif (
                    actual_revision is not None
                    and actual_revision != config.model_revision
                ):
                    raise SemanticEmbeddingSystemError("model_revision_drift")
                model_revision = actual_revision or model_revision
            except asyncio.CancelledError:
                await self._record_cancelled_provider_usage(
                    runtime=runtime,
                    result=result,
                    before_sequence=before_sequence,
                    config=config,
                    user_id=user_id,
                    operation="notes_semantic_embeddings",
                )
                raise
            except Exception:
                await self._record_provider_usage(
                    runtime=runtime,
                    result=result,
                    before_sequence=before_sequence,
                    config=config,
                    user_id=user_id,
                    operation="notes_semantic_embeddings",
                    succeeded=False,
                )
                raise
            provider_prompt_tokens, provider_total_tokens = await self._record_provider_usage(
                runtime=runtime,
                result=result,
                before_sequence=before_sequence,
                config=config,
                user_id=user_id,
                operation="notes_semantic_embeddings",
                succeeded=True,
            )
            vectors.extend(batch_vectors)
            prompt_tokens += provider_prompt_tokens
            total_tokens += provider_total_tokens
        return SemanticEmbeddingBatch(
            vectors=tuple(tuple(vector) for vector in vectors),
            provider=config.provider,
            model=config.model,
            model_revision=model_revision,
            dimensions=config.dimensions,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
        )

    async def _record_cancelled_provider_usage(
        self,
        *,
        runtime: NotesEmbeddingRuntime,
        result: Any,
        before_sequence: int,
        config: PendingSemanticConfig | ResolvedSemanticConfig,
        user_id: str,
        operation: str,
    ) -> None:
        try:
            await asyncio.shield(
                self._record_provider_usage(
                    runtime=runtime,
                    result=result,
                    before_sequence=before_sequence,
                    config=config,
                    user_id=user_id,
                    operation=operation,
                    succeeded=False,
                )
            )
        except asyncio.CancelledError:
            return
        except Exception:  # noqa: BLE001 - cancellation must remain the visible outcome
            return

    async def _record_provider_usage(
        self,
        *,
        runtime: NotesEmbeddingRuntime,
        result: Any,
        before_sequence: int,
        config: PendingSemanticConfig | ResolvedSemanticConfig,
        user_id: str,
        operation: str,
        succeeded: bool,
    ) -> tuple[int, int]:
        identity = runtime.execution_identity()
        if result is not None:
            cache_hits = int(result.cache_hits)
            cache_misses = int(result.cache_misses)
            if cache_misses == 0:
                return 0, 0
        else:
            sequence = getattr(identity, "provider_attempt_sequence", 0)
            if type(sequence) is not int or sequence <= before_sequence:
                return 0, 0
            cache_hits = 0
            cache_misses = int(getattr(identity, "provider_input_count", 0) or 0)
            if cache_misses <= 0:
                return 0, 0

        provider_request_count = getattr(identity, "provider_request_count", 1)
        if type(provider_request_count) is not int or provider_request_count != 1:
            raise SemanticEmbeddingSystemError("provider_usage_unavailable")

        identity_count = getattr(identity, "provider_input_count", 0)
        identity_tokens = getattr(identity, "provider_prompt_tokens", 0)
        if identity_count == cache_misses and type(identity_tokens) is int:
            provider_prompt_tokens = max(0, identity_tokens)
        elif cache_hits == 0:
            provider_prompt_tokens = max(0, int(result.prompt_tokens))
        else:
            raise SemanticEmbeddingSystemError("provider_usage_unavailable")
        await self._usage_logger(
            user_id=_owner_user_id(user_id),
            key_id=None,
            endpoint="/internal/notes/semantic-embeddings",
            operation=operation,
            provider=config.provider,
            model=config.model,
            status=200 if succeeded else 502,
            latency_ms=0,
            prompt_tokens=provider_prompt_tokens,
            completion_tokens=0,
            total_tokens=provider_prompt_tokens,
            estimated=True,
            request=None,
            usage_metadata={
                "attempt_status": "success" if succeeded else "failed",
                "cache_hit_count": cache_hits,
                "cache_miss_count": cache_misses,
                "provider_input_count": cache_misses,
                "provider_request_count": provider_request_count,
            },
        )
        return provider_prompt_tokens, provider_prompt_tokens

    def _plan_batches(
        self,
        chunks: Sequence[SemanticChunkInput],
    ) -> tuple[tuple[SemanticChunkInput, ...], ...]:
        if len(chunks) > self._settings.max_chunks_per_run:
            raise SemanticEmbeddingSystemError("run_chunk_cap_exceeded")
        sizes = [len(chunk.provider_input.text.encode("utf-8")) for chunk in chunks]
        if any(size > self._settings.max_provider_input_bytes for size in sizes):
            raise SemanticEmbeddingSystemError("provider_input_bytes_exceeded")
        if sum(sizes) > self._settings.max_provider_bytes_per_run:
            raise SemanticEmbeddingSystemError("run_byte_cap_exceeded")

        batches: list[tuple[SemanticChunkInput, ...]] = []
        current: list[SemanticChunkInput] = []
        current_bytes = 0
        for chunk, size in zip(chunks, sizes):
            if size > self._settings.max_provider_batch_bytes:
                raise SemanticEmbeddingSystemError("provider_batch_bytes_exceeded")
            if current and (
                len(current) == self._settings.max_provider_batch_inputs
                or current_bytes + size > self._settings.max_provider_batch_bytes
            ):
                batches.append(tuple(current))
                current = []
                current_bytes = 0
            current.append(chunk)
            current_bytes += size
        if current:
            batches.append(tuple(current))
        if len(batches) > self._settings.max_provider_requests_per_run:
            raise SemanticEmbeddingSystemError("provider_request_cap_exceeded")
        return tuple(batches)

    async def _execute(
        self,
        runtime: NotesEmbeddingRuntime,
        texts: list[str],
        config: PendingSemanticConfig | ResolvedSemanticConfig,
        *,
        user_id: str,
    ) -> Any:
        context = EmbeddingRequestContext(
            user_id=user_id,
            model_field=config.model,
            provider_header=config.provider,
            dimensions=config.dimensions,
            encoding_format="float",
            endpoint_path="/internal/notes/semantic-embeddings",
        )
        prepared = runtime.orchestrator.prepare(texts, context)
        plan = prepared.execution_plan
        if (
            prepared.policy_decision.fallback_allowed
            or plan.fallback_chain != [config.provider]
            or plan.provider != config.provider
            or plan.model != config.model
            or plan.dimensions != config.dimensions
        ):
            raise SemanticEmbeddingSystemError("embedding_policy_drift")
        return await runtime.orchestrator.execute(prepared)


def _provider_attempt_sequence(runtime: NotesEmbeddingRuntime) -> int:
    value = getattr(runtime.execution_identity(), "provider_attempt_sequence", 0)
    return value if type(value) is int and value >= 0 else 0


def _validate_execution_result(
    result: Any,
    config: PendingSemanticConfig | ResolvedSemanticConfig,
) -> None:
    if (
        result.provider != config.provider
        or result.model != config.model
        or result.fallback_from is not None
    ):
        raise SemanticEmbeddingSystemError("provider_model_drift")


def _strict_vectors(
    vectors: object,
    *,
    expected: int,
    dimensions: int | None,
) -> list[list[float]]:
    validated = validated_embedding_vectors(vectors, expected=expected)
    if validated is None:
        raise SemanticEmbeddingSystemError("invalid_vectors")
    if dimensions is not None and any(len(vector) != dimensions for vector in validated):
        raise SemanticEmbeddingSystemError("dimension_mismatch")
    if any(math.sqrt(sum(value * value for value in vector)) == 0.0 for vector in validated):
        raise SemanticEmbeddingSystemError("zero_norm_vector")
    return validated


__all__ = [
    "DIMENSION_PROBE_TEXT",
    "NotesEmbeddingExecutor",
    "NotesEmbeddingExecutionIdentity",
    "NotesEmbeddingRuntime",
    "NotesSemanticEmbedder",
    "PendingSemanticConfig",
    "ResolvedDimension",
    "ResolvedSemanticConfig",
    "RunMemoryEmbeddingCache",
    "SemanticEmbeddingBatch",
    "SemanticEmbeddingSystemError",
    "build_notes_semantic_orchestrator",
]
