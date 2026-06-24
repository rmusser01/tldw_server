from __future__ import annotations

"""
MLX-backed local LLM provider (Apple Silicon first).

This adapter mirrors other Chat/Embeddings providers while delegating lifecycle
to an in-process session registry. It keeps a single active model by default,
enforces a small concurrency cap, and provides optional compile/warmup on load
to avoid first-token stalls.
"""

import contextlib
import importlib
import os
import threading
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatBadRequestError,
    ChatProviderError,
    ChatRateLimitError,
)
from tldw_Server_API.app.core.LLM_Calls.capability_registry import validate_payload
from tldw_Server_API.app.core.LLM_Calls.sse import (
    finalize_stream,
    openai_delta_chunk,
)
from tldw_Server_API.app.core.Metrics.metrics_manager import (
    get_metrics_registry,
    increment_counter,
    observe_histogram,
    set_gauge,
)
from tldw_Server_API.app.core.Utils.common import parse_boolean

from .base import ChatProvider, EmbeddingsProvider

_MLX_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _coerce_int(val: str | None, default: int | None = None) -> int | None:
    try:
        if val is None:
            return default
        return int(val)
    except (TypeError, ValueError):
        return default


def _extract_unexpected_kwarg(err: TypeError) -> str | None:
    message = str(err)
    marker = "unexpected keyword argument"
    if marker not in message:
        return None
    try:
        return message.split(marker, 1)[1].strip().strip("'\"")
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def _retry_load_without_unknown_kwargs(load_fn, model_path: str, load_kwargs: dict[str, Any], err: TypeError):
    remaining = dict(load_kwargs)
    current_err: TypeError = err
    while remaining:
        bad_key = _extract_unexpected_kwarg(current_err)
        if not bad_key or bad_key not in remaining:
            break
        remaining.pop(bad_key, None)
        try:
            return load_fn(model_path, **remaining)
        except TypeError as exc:
            current_err = exc
            continue
    raise current_err


def _default_settings() -> dict[str, Any]:
    """Load MLX defaults from env/config shape (env-first)."""
    return {
        "model_path": os.getenv("MLX_MODEL_PATH"),
        "max_seq_len": _coerce_int(os.getenv("MLX_MAX_SEQ_LEN")),
        "max_batch_size": _coerce_int(os.getenv("MLX_MAX_BATCH_SIZE")),
        "device": os.getenv("MLX_DEVICE", "auto"),
        "dtype": os.getenv("MLX_DTYPE"),
        "quantization": os.getenv("MLX_QUANTIZATION"),
        "compile": parse_boolean(os.getenv("MLX_COMPILE"), default=True),
        "prompt_template": os.getenv("MLX_PROMPT_TEMPLATE"),
        "revision": os.getenv("MLX_REVISION"),
        "trust_remote_code": parse_boolean(os.getenv("MLX_TRUST_REMOTE_CODE"), default=False),
        "tokenizer": os.getenv("MLX_TOKENIZER"),
        "adapter": os.getenv("MLX_ADAPTER"),
        "adapter_weights": os.getenv("MLX_ADAPTER_WEIGHTS"),
        "max_kv_cache_size": _coerce_int(os.getenv("MLX_MAX_KV_CACHE_SIZE")),
        "max_concurrent": max(1, _coerce_int(os.getenv("MLX_MAX_CONCURRENT"), default=1) or 1),
        "warmup": parse_boolean(os.getenv("MLX_WARMUP"), default=True),
    }


@dataclass
class MLXSession:
    model_id: str
    model: Any
    tokenizer: Any
    generate_fn: Any
    generate_stream_fn: Any
    embed_fn: Any
    supports_embeddings: bool
    config: dict[str, Any]
    loaded_at: float = field(default_factory=time.time)
    warmup_completed: bool = False


class MLXSessionRegistry:
    """Singleton-ish registry holding the active MLX model/session."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sema = threading.BoundedSemaphore(1)
        self._max_concurrent: int = 1
        self._session: MLXSession | None = None
        self._inflight: int = 0
        self._metrics_registered = False
        self._load_generation: int = 0

    def _set_concurrency(self, max_concurrent: int) -> None:
        max_concurrent = max(1, int(max_concurrent))
        self._max_concurrent = max_concurrent
        self._sema = threading.BoundedSemaphore(max_concurrent)
        set_gauge("mlx_max_concurrent", float(max_concurrent))

    def _ensure_metrics(self) -> None:
        if self._metrics_registered:
            return
        try:
            reg = get_metrics_registry()
            # Histograms and counters for load/chat/embed and gauges for active sessions.
            for metric_def in (
                ("mlx_load_duration_seconds", "histogram", "MLX model load duration", ["model"]),
                ("mlx_load_total", "counter", "Total MLX load attempts", ["model", "status"]),
                ("mlx_chat_latency_seconds", "histogram", "MLX chat latency seconds", ["model", "streaming"]),
                ("mlx_tokens_generated_total", "counter", "Total tokens generated by MLX", ["model", "streaming"]),
                ("mlx_embeddings_requests_total", "counter", "Total MLX embedding requests", ["model"]),
                ("mlx_embeddings_latency_seconds", "histogram", "MLX embedding latency seconds", ["model"]),
            ):
                name, kind, desc, labels = metric_def
                if name in reg.metrics:
                    continue
                from tldw_Server_API.app.core.Metrics.metrics_manager import MetricDefinition, MetricType

                reg.register_metric(
                    MetricDefinition(
                        name=name,
                        type=MetricType.COUNTER if kind == "counter" else MetricType.HISTOGRAM,
                        description=desc,
                        labels=labels,
                    )
                )
            # Gauges
            for name, desc in (
                ("mlx_active_sessions", "Number of active MLX sessions"),
                ("mlx_requests_inflight", "Number of in-flight MLX chat/embedding requests"),
                ("mlx_queue_depth", "MLX request queue depth (always 0 - overflow rejects)"),
                ("mlx_max_concurrent", "Configured MLX max concurrency"),
            ):
                if name not in reg.metrics:
                    from tldw_Server_API.app.core.Metrics.metrics_manager import MetricDefinition, MetricType

                    reg.register_metric(
                        MetricDefinition(
                            name=name,
                            type=MetricType.GAUGE,
                            description=desc,
                            labels=[],
                        )
                    )
            self._metrics_registered = True
        except _MLX_NONCRITICAL_EXCEPTIONS as exc:  # pragma: no cover - metrics must not break flow
            with contextlib.suppress(_MLX_NONCRITICAL_EXCEPTIONS):
                logger.debug(f"MLX metrics registration failed: {exc}")

    def _import_mlx(self):
        try:
            return importlib.import_module("mlx_lm")
        except ImportError as exc:  # pragma: no cover - env/optional
            raise ChatProviderError(provider="mlx", message="mlx-lm is not installed") from exc

    def load(self, *, model_path: str | None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """Load or swap the active model. Keeps previous model on failure."""
        if isinstance(model_path, str):
            model_path = model_path.strip() or None
        if not model_path:
            raise ChatBadRequestError(provider="mlx", message="model_path is required")
        self._ensure_metrics()
        start = time.time()
        settings = _default_settings()
        if overrides:
            for k, v in overrides.items():
                if v is not None:
                    settings[k] = v

        prev_session: MLXSession | None = None
        load_generation: int
        with self._lock:
            prev_session = self._session
            self._load_generation += 1
            load_generation = self._load_generation

        mlx_mod = self._import_mlx()
        load_fn = getattr(mlx_mod, "load", None)
        if not callable(load_fn):
            raise ChatProviderError(provider="mlx", message="mlx_lm.load not available")
        generate_fn = getattr(mlx_mod, "generate", None)
        generate_stream_fn = getattr(mlx_mod, "generate_stream", None)
        embed_fn = getattr(mlx_mod, "embed", None)

        try:
            # Pass only supported kwargs conservatively
            load_kwargs = {}
            for key in ("max_seq_len", "max_batch_size", "device", "dtype", "revision", "trust_remote_code"):
                if settings.get(key) is not None:
                    load_kwargs[key] = settings[key]
            if settings.get("tokenizer"):
                load_kwargs["tokenizer"] = settings["tokenizer"]
            if settings.get("adapter"):
                load_kwargs["adapter"] = settings["adapter"]
            if settings.get("adapter_weights"):
                load_kwargs["adapter_weights"] = settings["adapter_weights"]
            try:
                model, tokenizer = load_fn(model_path, **load_kwargs)
            except TypeError as exc:
                model, tokenizer = _retry_load_without_unknown_kwargs(load_fn, model_path, load_kwargs, exc)
            session = MLXSession(
                model_id=model_path,
                model=model,
                tokenizer=tokenizer,
                generate_fn=generate_fn,
                generate_stream_fn=generate_stream_fn,
                embed_fn=embed_fn,
                supports_embeddings=callable(embed_fn),
                config=settings,
            )

            if settings.get("compile", True) or settings.get("warmup", True):
                try:
                    self._warmup(session)
                    session.warmup_completed = True
                except Exception as warm_err:
                    logger.warning(f"MLX warmup failed: {warm_err}")
                    # On warmup failure, keep previous model and surface error
                    raise

            applied = False
            with self._lock:
                if load_generation == self._load_generation:
                    self._session = session
                    self._set_concurrency(settings.get("max_concurrent", 1))
                    applied = True
            try:
                duration = time.time() - start
                observe_histogram(
                    "mlx_load_duration_seconds",
                    duration,
                    labels={"model": model_path},
                )
                status = "success" if applied else "superseded"
                increment_counter("mlx_load_total", labels={"model": model_path, "status": status})
                if applied:
                    set_gauge("mlx_active_sessions", 1.0)
                    set_gauge("mlx_requests_inflight", float(self._inflight))
                    set_gauge("mlx_queue_depth", 0.0)
            except _MLX_NONCRITICAL_EXCEPTIONS:
                pass
            return self.status()
        except Exception as exc:
            # Restore prior session on failure
            with self._lock:
                if load_generation == self._load_generation:
                    self._session = prev_session
            try:
                duration = time.time() - start
                observe_histogram(
                    "mlx_load_duration_seconds",
                    duration,
                    labels={"model": model_path},
                )
                increment_counter("mlx_load_total", labels={"model": model_path, "status": "failure"})
            except _MLX_NONCRITICAL_EXCEPTIONS:
                pass
            if isinstance(exc, ChatProviderError):
                raise
            raise ChatProviderError(provider="mlx", message=str(exc)) from exc

    def _warmup(self, session: MLXSession) -> None:
        """Best-effort warmup/compile to avoid first-token stalls."""
        if not callable(session.generate_fn):
            return
        prompt = "Hello"
        try:
            session.generate_fn(session.model, session.tokenizer, prompt, max_tokens=1, temp=0.1, verbose=False)
        except TypeError:
            # Fallback without kwargs if the signature differs
            session.generate_fn(session.model, session.tokenizer, prompt)

    def unload(self) -> dict[str, Any]:
        with self._lock:
            self._session = None
        try:
            set_gauge("mlx_active_sessions", 0.0)
            set_gauge("mlx_requests_inflight", 0.0)
            set_gauge("mlx_queue_depth", 0.0)
        except _MLX_NONCRITICAL_EXCEPTIONS:
            pass
        return {"status": "unloaded"}

    @contextlib.contextmanager
    def session_scope(self) -> Iterable[MLXSession]:
        self._ensure_metrics()
        # Snapshot the current semaphore so that concurrency updates
        # (which swap out self._sema) do not affect in-flight contexts.
        sema = self._sema
        acquired = sema.acquire(blocking=False)
        if not acquired:
            raise ChatRateLimitError(provider="mlx", message="MLX busy (max concurrency reached)")
        counted = False
        try:
            with self._lock:
                if not self._session:
                    raise ChatBadRequestError(provider="mlx", message="No active MLX model; load one first")
                session = self._session
                self._inflight += 1
                counted = True
                try:
                    set_gauge("mlx_requests_inflight", float(self._inflight))
                    set_gauge("mlx_queue_depth", 0.0)
                except _MLX_NONCRITICAL_EXCEPTIONS:
                    pass
            yield session
        finally:
            if acquired:
                with contextlib.suppress(ValueError):
                    sema.release()
            if counted:
                with self._lock:
                    self._inflight = max(0, self._inflight - 1)
                    with contextlib.suppress(_MLX_NONCRITICAL_EXCEPTIONS):
                        set_gauge("mlx_requests_inflight", float(self._inflight))

    def status(self) -> dict[str, Any]:
        with self._lock:
            if not self._session:
                return {
                    "active": False,
                    "model": None,
                    "loaded_at": None,
                    "supports_embeddings": False,
                    "warmup_completed": False,
                    "max_concurrent": self._max_concurrent,
                }
            s = self._session
            unapplied_overrides: dict[str, Any] = {}
            for key in ("quantization", "max_kv_cache_size"):
                value = s.config.get(key)
                if value is not None:
                    unapplied_overrides[key] = value
            config = {
                "device": s.config.get("device"),
                "dtype": s.config.get("dtype"),
                "compile": bool(s.config.get("compile", True)),
                "warmup": bool(s.config.get("warmup", True)),
                "max_seq_len": s.config.get("max_seq_len"),
                "max_batch_size": s.config.get("max_batch_size"),
            }
            if unapplied_overrides:
                config["unapplied_runtime_overrides"] = unapplied_overrides
            return {
                "active": True,
                "model": s.model_id,
                "loaded_at": s.loaded_at,
                "supports_embeddings": s.supports_embeddings,
                "warmup_completed": s.warmup_completed,
                "max_concurrent": self._max_concurrent,
                "config": config,
            }

    def get_active_tokenizer(self, model_id: str | None = None) -> tuple[Any, str] | None:
        with self._lock:
            if not self._session:
                return None
            if model_id and str(model_id).strip() and self._session.model_id != str(model_id).strip():
                return None
            return self._session.tokenizer, self._session.model_id


_registry: MLXSessionRegistry | None = None
_registry_lock = threading.Lock()


def get_mlx_registry() -> MLXSessionRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = MLXSessionRegistry()
    return _registry


def get_active_mlx_tokenizer(model_id: str | None = None) -> tuple[Any, str] | None:
    """Return active tokenizer/model from MLX registry when loaded."""
    return get_mlx_registry().get_active_tokenizer(model_id=model_id)


def _messages_to_prompt(messages: Any, tokenizer: Any, system_message: str | None, template_override: str | None) -> str:
    """Convert OpenAI-style messages to a prompt string using tokenizer chat template when available."""
    msgs = messages or []
    if system_message:
        msgs = [{"role": "system", "content": system_message}] + list(msgs)
    try:
        if hasattr(tokenizer, "apply_chat_template"):
            if template_override and hasattr(tokenizer, "chat_template"):
                original_template = getattr(tokenizer, "chat_template", None)
                try:
                    tokenizer.chat_template = template_override
                    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                finally:
                    with contextlib.suppress(_MLX_NONCRITICAL_EXCEPTIONS):
                        tokenizer.chat_template = original_template
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except _MLX_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(f"MLX chat template application failed; falling back: {exc}")
    # Fallback: naive concatenation
    parts = []
    for m in msgs:
        role = m.get("role") or "user"
        content = m.get("content") or ""
        parts.append(f"[{role}] {content}")
    return "\n".join(parts)


class MLXChatAdapter(ChatProvider):
    name = "mlx"

    def __init__(self) -> None:
        self.registry = None

    def capabilities(self) -> dict[str, Any]:
        status = get_mlx_registry().status()
        return {
            "supports_streaming": True,
            "supports_tools": False,
            "default_timeout_seconds": 120,
            "max_output_tokens_default": status.get("config", {}).get("max_seq_len"),
            "supports_embeddings": status.get("supports_embeddings", False),
        }

    def _generate_kwargs(self, request: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k in ("max_tokens",):
            if request.get(k) is not None:
                out[k] = request[k]
        temp = request.get("temperature")
        if temp is None:
            temp = request.get("temp")
        if temp is not None:
            out["temp"] = temp
        top_p = request.get("top_p")
        if top_p is None:
            top_p = request.get("topp")
        if top_p is not None:
            out["top_p"] = top_p
        top_k = request.get("top_k")
        if top_k is None:
            top_k = request.get("topk")
        if top_k is not None:
            out["top_k"] = top_k
        return out

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        request = validate_payload(self.name, request or {})
        with get_mlx_registry().session_scope() as session:
            prompt = _messages_to_prompt(
                request.get("messages"),
                session.tokenizer,
                request.get("system_message"),
                request.get("prompt_template") or session.config.get("prompt_template"),
            )
            generate_kwargs = self._generate_kwargs(request)
            start_time = time.time()
            try:
                output = session.generate_fn(
                    session.model,
                    session.tokenizer,
                    prompt,
                    stream=False,
                    verbose=False,
                    **generate_kwargs,
                )
            except TypeError:
                output = session.generate_fn(session.model, session.tokenizer, prompt)
            content = output if isinstance(output, str) else str(output)
            created = int(time.time())
            tokens = len(str(content).split())
            try:
                observe_histogram(
                    "mlx_chat_latency_seconds",
                    float(time.time() - start_time),
                    labels={"model": session.model_id, "streaming": "false"},
                )
                increment_counter(
                    "mlx_tokens_generated_total",
                    value=float(tokens),
                    labels={"model": session.model_id, "streaming": "false"},
                )
            except _MLX_NONCRITICAL_EXCEPTIONS:
                pass
            return {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": created,
                "model": session.model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            }

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        request = validate_payload(self.name, request or {})
        with get_mlx_registry().session_scope() as session:
            prompt = _messages_to_prompt(
                request.get("messages"),
                session.tokenizer,
                request.get("system_message"),
                request.get("prompt_template") or session.config.get("prompt_template"),
            )
            generate_kwargs = self._generate_kwargs(request)
            start_time = time.time()
            if callable(session.generate_stream_fn):
                try:
                    stream = session.generate_stream_fn(
                        session.model,
                        session.tokenizer,
                        prompt,
                        verbose=False,
                        **generate_kwargs,
                    )
                except TypeError:
                    stream = session.generate_stream_fn(session.model, session.tokenizer, prompt)
                try:
                    total_tokens = 0
                    for chunk in stream:
                        if chunk:
                            total_tokens += len(str(chunk).split())
                            yield openai_delta_chunk(str(chunk))
                    try:
                        observe_histogram(
                            "mlx_chat_latency_seconds",
                            float(time.time() - start_time),
                            labels={"model": session.model_id, "streaming": "true"},
                        )
                        if total_tokens:
                            increment_counter(
                                "mlx_tokens_generated_total",
                                value=float(total_tokens),
                                labels={"model": session.model_id, "streaming": "true"},
                            )
                    except _MLX_NONCRITICAL_EXCEPTIONS:
                        pass
                    yield from finalize_stream(None)
                    return
                except _MLX_NONCRITICAL_EXCEPTIONS as exc:
                    logger.error(f"MLX streaming failed, falling back to non-stream: {exc}")
            # Fallback to single-shot if streaming not available
            try:
                output = session.generate_fn(
                    session.model,
                    session.tokenizer,
                    prompt,
                    stream=False,
                    verbose=False,
                    **generate_kwargs,
                )
            except TypeError:
                output = session.generate_fn(session.model, session.tokenizer, prompt)
            content = output if isinstance(output, str) else str(output)
            tokens = len(str(content).split())
            try:
                observe_histogram(
                    "mlx_chat_latency_seconds",
                    float(time.time() - start_time),
                    labels={"model": session.model_id, "streaming": "true"},
                )
                if tokens:
                    increment_counter(
                        "mlx_tokens_generated_total",
                        value=float(tokens),
                        labels={"model": session.model_id, "streaming": "true"},
                    )
            except _MLX_NONCRITICAL_EXCEPTIONS:
                pass
            yield openai_delta_chunk(content)
            yield from finalize_stream(None)


class MLXEmbeddingsAdapter(EmbeddingsProvider):
    name = "mlx-embeddings"

    def __init__(self) -> None:
        self.registry = get_mlx_registry()

    def capabilities(self) -> dict[str, Any]:
        status = self.registry.status()
        return {
            "dimensions_default": None,
            "max_batch_size": status.get("config", {}).get("max_batch_size"),
            "default_timeout_seconds": 60,
            "supports_embeddings": status.get("supports_embeddings", False),
        }

    def embed(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        inputs = request.get("input")
        if inputs is None:
            raise ChatBadRequestError(provider="mlx", message="'input' is required for embeddings")
        with self.registry.session_scope() as session:
            start_time = time.time()
            if not session.supports_embeddings or not callable(session.embed_fn):
                raise ChatBadRequestError(
                    provider="mlx",
                    message="Active MLX model does not support embeddings",
                )
            try:
                if isinstance(inputs, list):
                    vectors = [session.embed_fn(session.model, session.tokenizer, text) for text in inputs]
                else:
                    vectors = session.embed_fn(session.model, session.tokenizer, inputs)
            except Exception as exc:
                raise ChatProviderError(provider="mlx", message=str(exc)) from exc
            try:
                observe_histogram(
                    "mlx_embeddings_latency_seconds",
                    float(time.time() - start_time),
                    labels={"model": session.model_id},
                )
                increment_counter(
                    "mlx_embeddings_requests_total",
                    labels={"model": session.model_id},
                )
            except _MLX_NONCRITICAL_EXCEPTIONS:
                pass

        # Normalize to OpenAI-like response shape
        if isinstance(inputs, list):
            data = [{"index": i, "embedding": vec} for i, vec in enumerate(vectors)]  # type: ignore[arg-type]
        else:
            data = [{"index": 0, "embedding": vectors}]  # type: ignore[list-item]
        return {"data": data, "object": "list", "model": session.model_id}


__all__ = [
    "MLXChatAdapter",
    "MLXEmbeddingsAdapter",
    "MLXSessionRegistry",
    "get_mlx_registry",
    "get_active_mlx_tokenizer",
]
