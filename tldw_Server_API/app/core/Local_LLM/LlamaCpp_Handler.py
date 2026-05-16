# /tldw_Server_API/app/core/Local_LLM/LlamaCpp_Handler.py
# Description: Handler for Llama.cpp models, managing server processes and inference.
#
import asyncio
import contextlib
import os
import platform
import signal
# Used without shell for managed process flags and bounded process cleanup.
import subprocess  # nosec B404
import time
from pathlib import Path
from typing import Any, Optional

from tldw_Server_API.app.core.exceptions import NetworkError, RetryExhaustedError
from tldw_Server_API.app.core.LLM_Calls.sse import (
    ensure_sse_line,
    openai_delta_chunk,
    sse_data,
    sse_done,
)
from tldw_Server_API.app.core.Local_LLM import handler_utils, http_utils

# Local imports
from .LLM_Base_Handler import BaseLLMHandler
from .LLM_Inference_Exceptions import InferenceError, ModelNotFoundError, ServerError
from .LLM_Inference_Schemas import LlamaCppConfig

try:
    import httpx as _httpx
except ImportError:  # pragma: no cover - optional dependency in some test setups
    _httpx = None  # type: ignore[assignment]

_LLAMACPP_HTTP_EXCEPTIONS: tuple[type[BaseException], ...] = ()
if _httpx is not None:
    _LLAMACPP_HTTP_EXCEPTIONS = (
        _httpx.HTTPError,
        _httpx.TimeoutException,
    )

_LLAMACPP_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    NetworkError,
    OSError,
    PermissionError,
    RetryExhaustedError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
    subprocess.SubprocessError,
) + _LLAMACPP_HTTP_EXCEPTIONS


def create_async_client(*args, **kwargs):
    """Proxy create_async_client so tests can monkeypatch either module."""
    return http_utils.create_async_client(*args, **kwargs)


async def request_json(*args, **kwargs):
    """Proxy request_json for test monkeypatching."""
    return await http_utils.request_json(*args, **kwargs)


async def wait_for_http_ready(*args, **kwargs):
    """Proxy wait_for_http_ready preserving signature expectations."""
    return await http_utils.wait_for_http_ready(*args, **kwargs)
#########################################################################################################################
#
# Functions:

class LlamaCppHandler(BaseLLMHandler):
    def __init__(self, config: LlamaCppConfig, global_app_config: dict[str, Any]):
        super().__init__(config, global_app_config)
        self.config: LlamaCppConfig  # For type hinting
        # self.logger = logger # Or use self.logger from BaseLLMHandler if already set

        self.models_dir = Path(self.config.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # For Llama.cpp, we usually manage one server instance that can have its model swapped.
        # If you need multiple concurrent Llama.cpp servers, this would be a Dict like in LlamafileHandler
        self._active_server_process: Optional[asyncio.subprocess.Process] = None
        self._active_server_model: Optional[str] = None
        self._active_server_port: Optional[int] = None
        self._active_server_host: Optional[str] = None
        self._active_server_log_handle = None
        self._stream_tasks: list[asyncio.Task] = []

        # Apply environment overrides for handler config
        handler_utils.apply_env_overrides(self.config)

        self._setup_signal_handlers()  # For cleaning up on exit
        self._cleanup_done = False  # Guard against duplicate cleanup
        # Lightweight metrics
        self.metrics = {
            "starts": 0,
            "stops": 0,
            "start_errors": 0,
            "readiness_time_sum": 0.0,
            "readiness_count": 0,
            "inference_count": 0,
            "inference_error_count": 0,
            "inference_time_sum": 0.0,
        }

    def get_metrics(self) -> dict[str, Any]:
        return dict(self.metrics)

    # --- Utilities (delegating to shared handler_utils) ---
    def _is_port_free(self, host: str, port: int) -> bool:
        return handler_utils.is_port_free(host, port)

    def _pick_port(self, host: str, start_port: int) -> int:
        """Find an available port. Uses self._is_port_free for testability."""
        if not getattr(self.config, "port_autoselect", True):
            return start_port
        max_probe = int(getattr(self.config, "port_probe_max", 10) or 0)
        for i in range(max_probe + 1):
            candidate = start_port + i
            if self._is_port_free(host, candidate):
                return candidate
        return start_port  # Fallback

    def _denylist_check(self, args: dict[str, Any]):
        try:
            handler_utils.check_denylist(
                args,
                allow_secrets=getattr(self.config, "allow_cli_secrets", False),
            )
        except ValueError as e:
            raise ServerError(str(e)) from e

    def _is_path_allowed(self, p: Path) -> bool:
        """Check if path is under allowed directories."""
        base_dirs = handler_utils.build_allowed_paths(
            self.models_dir,
            getattr(self.config, "allowed_paths", None),
        )
        return handler_utils.is_path_allowed(p, base_dirs)

    def _safe_log(self, level: str, msg: str, *args):
        """Log defensively to avoid errors when sinks are closed during atexit."""
        handler_utils.safe_log(self.logger, level, msg, *args)

    async def _terminate_process(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        is_windows = platform.system().lower().startswith("win")
        if is_windows:
            if hasattr(process, "terminate"):
                process.terminate()
        else:
            try:
                pgid = await asyncio.to_thread(os.getpgid, process.pid)
                await asyncio.to_thread(os.killpg, pgid, signal.SIGTERM)
            except ProcessLookupError:
                if hasattr(process, "terminate"):
                    process.terminate()
            except _LLAMACPP_NONCRITICAL_EXCEPTIONS:
                if hasattr(process, "terminate"):
                    process.terminate()
        try:
            if hasattr(process, "wait"):
                await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            if is_windows:
                if hasattr(process, "kill"):
                    process.kill()
            else:
                try:
                    pgid = await asyncio.to_thread(os.getpgid, process.pid)
                    await asyncio.to_thread(os.killpg, pgid, signal.SIGKILL)
                except _LLAMACPP_NONCRITICAL_EXCEPTIONS:
                    if hasattr(process, "kill"):
                        process.kill()
            try:
                if hasattr(process, "wait"):
                    await process.wait()
            except _LLAMACPP_NONCRITICAL_EXCEPTIONS:
                pass

    async def _drain_stream(self, stream, label: str) -> None:
        if stream is None:
            return
        try:
            while True:
                chunk = await stream.read(1024)
                if not chunk:
                    break
        except asyncio.CancelledError:
            return
        except _LLAMACPP_NONCRITICAL_EXCEPTIONS:
            # Best-effort drain; ignore errors
            return

    def _start_stream_drainers(self, process: asyncio.subprocess.Process) -> None:
        tasks: list[asyncio.Task] = []
        if getattr(process, "stdout", None) is not None:
            tasks.append(asyncio.create_task(self._drain_stream(process.stdout, "stdout")))
        if getattr(process, "stderr", None) is not None:
            tasks.append(asyncio.create_task(self._drain_stream(process.stderr, "stderr")))
        if tasks:
            self._stream_tasks = tasks

    def _stop_stream_drainers(self) -> None:
        for task in self._stream_tasks:
            if not task.done():
                task.cancel()
        self._stream_tasks = []

    async def _stop_unmanaged_pid(self, pid: int) -> str:
        try:
            if platform.system() == "Windows":
                await asyncio.to_thread(
                    subprocess.run,
                    ["taskkill", "/F", "/PID", str(pid)],
                    check=True,
                    capture_output=True,
                )
            else:
                await asyncio.to_thread(os.kill, pid, signal.SIGTERM)
            return f"Attempted to send SIGTERM to unmanaged llama.cpp server with PID {pid}."
        except ProcessLookupError:
            return f"No process found with PID {pid}."
        except subprocess.CalledProcessError as e_taskkill:
            self.logger.exception(f"taskkill failed for PID {pid}: {e_taskkill.stderr.decode()}")
            return f"Failed to stop unmanaged PID {pid} with taskkill."
        except Exception as e:
            self.logger.error(f"Error stopping unmanaged PID {pid}: {e}", exc_info=True)
            raise ServerError(f"Error stopping unmanaged PID {pid}: {e}") from e

    def _is_chat_endpoint(self, api_endpoint: str) -> bool:
        endpoint = f"/{api_endpoint.lstrip('/')}"
        return endpoint.lower().endswith("/chat/completions")

    def _messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        parts: list[str] = []
        for msg in messages:
            if isinstance(msg, dict):
                role = str(msg.get("role", "user"))
                content = msg.get("content", "")
            else:
                role = str(getattr(msg, "role", "user"))
                content = getattr(msg, "content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    async def list_models(self) -> list[str]:
        """Lists locally available GGUF models."""
        if not self.models_dir.exists():
            return []

        def _scan_dir():
            return [f.name for f in self.models_dir.glob("*.gguf")]

        return await asyncio.to_thread(_scan_dir)

    async def is_model_available(self, model_filename: str) -> bool:
        """Checks if a GGUF model file is available locally."""
        return (self.models_dir / model_filename).is_file()

    # --- Server Management (Core of the swapping logic) ---
    async def start_server(self, model_filename: str, server_args: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """
        Starts the Llama.cpp server with the specified model.
        If a server is already running managed by this handler, it will be stopped first (model swap).
        """
        requested = Path(model_filename)
        if requested.is_absolute() or ".." in requested.parts:
            raise ServerError("Model filename must be a relative filename under the configured models directory.")
        return await self.start_server_by_path(
            self.models_dir / model_filename,
            model_label=model_filename,
            server_args=server_args,
        )

    async def start_server_by_path(
        self,
        model_path: Path,
        *,
        model_label: str | None = None,
        server_args: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Starts the managed Llama.cpp server using a validated local GGUF path.
        """
        try:
            model_path = model_path.expanduser().resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            raise ServerError("Model path could not be resolved.") from exc
        if not self._is_path_allowed(model_path):
            raise ServerError("Model path must be under allowed directories.")
        if model_path.suffix.lower() != ".gguf":
            raise ServerError("Model path must reference a GGUF file.")
        if not model_path.is_file():
            raise ModelNotFoundError(f"Model file {model_path.name} was not found.")
        return await self._start_server_for_model_path(
            model_path,
            model_label=model_label or model_path.name,
            server_args=server_args,
        )

    async def _start_server_for_model_path(
        self,
        model_path: Path,
        *,
        model_label: str,
        server_args: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if not Path(self.config.executable_path).is_file():
            raise ServerError(f"Llama.cpp server executable not found at {self.config.executable_path}")

        # --- Model Swapping Logic with Rollback ---
        prev_process = None
        prev_model = None
        prev_port = None
        prev_host = None
        prev_log_handle = None

        if self._active_server_process and self._active_server_process.returncode is None:
            # Save state for potential rollback
            prev_process = self._active_server_process
            prev_model = self._active_server_model
            prev_port = self._active_server_port
            prev_host = self._active_server_host
            prev_log_handle = self._active_server_log_handle

            self.logger.info(
                f"Stopping existing Llama.cpp server (PID: {self._active_server_process.pid}) to swap model.")
            try:
                await self.stop_server()
            except ServerError as e:
                # Restore previous state on stop failure
                self.logger.warning(f"Failed to stop server for model swap: {e}. Restoring previous state.")
                self._active_server_process = prev_process
                self._active_server_model = prev_model
                self._active_server_port = prev_port
                self._active_server_host = prev_host
                self._active_server_log_handle = prev_log_handle
                raise ServerError(f"Model swap failed: could not stop existing server: {e}") from e

        args = {k: v for k, v in (server_args or {}).items() if v is not None and v != ""}
        self._denylist_check(args)

        # Allowlist of supported args (internal key -> formatter)
        # Each formatter extends the command list with the mapped CLI args.
        allowed_formatters: dict[str, Any] = {
            "port": lambda v: ["--port", str(int(v))],
            "host": lambda v: ["--host", str(v)],
            "threads": lambda v: ["-t", str(int(v))],
            "t": lambda v: ["-t", str(int(v))],
            "threads_batch": lambda v: ["--threads-batch", str(int(v))],
            "tb": lambda v: ["--threads-batch", str(int(v))],
            "ctx_size": lambda v: ["-c", str(int(v))],
            "c": lambda v: ["-c", str(int(v))],
            "n_ctx": lambda v: ["-c", str(int(v))],
            "n_gpu_layers": lambda v: ["-ngl", str(int(v))],
            "ngl": lambda v: ["-ngl", str(int(v))],
            "gpu_layers": lambda v: ["-ngl", str(int(v))],
            "batch_size": lambda v: ["-b", str(int(v))],
            "b": lambda v: ["-b", str(int(v))],
            "n_batch": lambda v: ["-b", str(int(v))],
            "ubatch_size": lambda v: ["--ubatch-size", str(int(v))],
            "ub": lambda v: ["--ubatch-size", str(int(v))],
            "n_ubatch": lambda v: ["--ubatch-size", str(int(v))],
            "verbose": lambda v: (["--verbose"] if v else []),
            "log_disable": lambda v: (["--log-disable"] if v else []),
            # Extended safe flags
            "no_mmap": lambda v: (["--no-mmap"] if v else []),
            "mlock": lambda v: (["--mlock"] if v else []),
            "main_gpu": lambda v: ["--main-gpu", str(int(v))],
            "mg": lambda v: ["--main-gpu", str(int(v))],
            "split_mode": lambda v: ["--split-mode", str(v)],
            "sm": lambda v: ["--split-mode", str(v)],
            "row_split": lambda v: (["--split-mode", "row"] if v else []),
            # Additional extended flags
            # Note: Some of these may be build-dependent in llama.cpp;
            # keeping them in allowlist enables safe, explicit usage when supported.
            "main_kv": lambda v: ["--main-kv", str(int(v))],
            "no_kv_offload": lambda v: (["--no-kv-offload"] if v else []),
            "cpu_moe": lambda v: (["--cpu-moe"] if v else []),
            "n_cpu_moe": lambda v: ["--n-cpu-moe", str(int(v))],
            # Rope scaling type (e.g., "linear", "yarn", etc.)
            "rope_scaling_type": lambda v: ["--rope-scaling", str(v)],
            "rope_scaling": lambda v: ["--rope-scaling", str(v)],
            "tensor_split": lambda v: ["--tensor-split", ",".join(map(str, v))] if isinstance(v, (list, tuple)) else ["--tensor-split", str(v)],
            "rope_freq_base": lambda v: ["--rope-freq-base", str(float(v))],
            "rope_freq_scale": lambda v: ["--rope-freq-scale", str(float(v))],
            # Aliases and additional safe toggles
            "rope_scale": lambda v: ["--rope-freq-scale", str(float(v))],
            "flash_attn": lambda v: (["--flash-attn", str(v)] if isinstance(v, str) else (["--flash-attn"] if v else [])),
            "cont_batching": lambda v: (["--cont-batching"] if v else ["--no-cont-batching"]),
            "no_cont_batching": lambda v: (["--no-cont-batching"] if v else []),
            "context_shift": lambda v: (["--context-shift"] if v else ["--no-context-shift"]),
            "streaming_llm": lambda v: (["--context-shift"] if v else []),
            # LoRA support (repeatable)
            "lora": lambda v: sum((["--lora", str(x)] for x in (v if isinstance(v, (list, tuple)) else [v])), []),
            "lora_scaled": lambda v: (["--lora-scaled", str(v[0]), str(v[1])]
                                      if isinstance(v, (list, tuple)) and len(v) == 2
                                      else (["--lora-scaled", str(v)] if v is not None else [])),
            "lora_base": lambda v: ["--lora-base", str(v)],
            "control_vector": lambda v: ["--control-vector", str(v)],
            # KV cache type hints
            "cache_type_k": lambda v: ["--cache-type-k", str(v)],
            "cache_type_v": lambda v: ["--cache-type-v", str(v)],
            "cache_type": lambda v: ["--cache-type-k", str(v), "--cache-type-v", str(v)],
            # Model download / HF
            # Note: hf_token is intentionally NOT in allowlist (use HF_TOKEN env var instead)
            "hf_repo": lambda v: ["--hf-repo", str(v)],
            "hf_file": lambda v: ["--hf-file", str(v)],
            "offline": lambda v: (["--offline"] if v else []),
            # Chat / conversation toggles
            "conversation": lambda v: (["--conversation"] if v else []),
            "cnv": lambda v: (["--conversation"] if v else []),
            "no_conversation": lambda v: (["--no-conversation"] if v else []),
            "no_cnv": lambda v: (["--no-conversation"] if v else []),
            "interactive": lambda v: (["--interactive"] if v else []),
            "i": lambda v: (["--interactive"] if v else []),
            "interactive_first": lambda v: (["--interactive-first"] if v else []),
            "if": lambda v: (["--interactive-first"] if v else []),
            "single_turn": lambda v: (["--single-turn"] if v else []),
            "st": lambda v: (["--single-turn"] if v else []),
            "jinja": lambda v: (["--jinja"] if v else []),
            "chat_template": lambda v: ["--chat-template", str(v)],
            "chat_template_file": lambda v: ["--chat-template-file", str(v)],
            # Input/output control
            "in_prefix": lambda v: ["--in-prefix", str(v)],
            "in_suffix": lambda v: ["--in-suffix", str(v)],
            "in_prefix_bos": lambda v: (["--in-prefix-bos"] if v else []),
            "reverse_prompt": lambda v: ["--reverse-prompt", str(v)],
            "r": lambda v: ["--reverse-prompt", str(v)],
            # Text generation controls
            "predict": lambda v: ["-n", str(int(v))],
            "n": lambda v: ["-n", str(int(v))],
            "keep": lambda v: ["--keep", str(int(v))],
            "ignore_eos": lambda v: (["--ignore-eos"] if v else []),
            "no_context_shift": lambda v: (["--no-context-shift"] if v else []),
            # Sampling controls
            "temp": lambda v: ["--temp", str(float(v))],
            "seed": lambda v: ["-s", str(int(v))],
            "dynatemp_range": lambda v: ["--dynatemp-range", str(float(v))],
            "top_k": lambda v: ["--top-k", str(int(v))],
            "top_p": lambda v: ["--top-p", str(float(v))],
            "min_p": lambda v: ["--min-p", str(float(v))],
            "typical": lambda v: ["--typical", str(float(v))],
            # Repetition controls
            "repeat_penalty": lambda v: ["--repeat-penalty", str(float(v))],
            "repeat_last_n": lambda v: ["--repeat-last-n", str(int(v))],
            "presence_penalty": lambda v: ["--presence-penalty", str(float(v))],
            "frequency_penalty": lambda v: ["--frequency-penalty", str(float(v))],
            # DRY sampling
            "dry_multiplier": lambda v: ["--dry-multiplier", str(float(v))],
            "dry_base": lambda v: ["--dry-base", str(float(v))],
            "dry_allowed_length": lambda v: ["--dry-allowed-length", str(int(v))],
            # Mirostat
            "mirostat": lambda v: ["--mirostat", str(int(v))],
            "mirostat_lr": lambda v: ["--mirostat-lr", str(float(v))],
            "mirostat_ent": lambda v: ["--mirostat-ent", str(float(v))],
            # CPU/GPU/NUMA
            "cpu_mask": lambda v: ["--cpu-mask", str(v)],
            "cpu_range": lambda v: ["--cpu-range", str(v)],
            "numa": lambda v: (["--numa", str(v)] if isinstance(v, str) else (["--numa"] if v else [])),
            # Structured generation
            "grammar": lambda v: ["--grammar", str(v)],
            "grammar_file": lambda v: ["--grammar-file", str(v)],
            "json_schema": lambda v: ["--json-schema", str(v)],
            "json_schema_file": lambda v: ["--json-schema-file", str(v)],
            "j": lambda v: ["-j", str(v)],
            # Reasoning
            "reasoning_format": lambda v: ["--reasoning-format", str(v)],
            "reasoning_budget": lambda v: ["--reasoning-budget", str(int(v))],
            # Caching
            "prompt_cache": lambda v: ["--prompt-cache", str(v)],
            "prompt_cache_all": lambda v: (["--prompt-cache-all"] if v else []),
            "prompt_cache_ro": lambda v: (["--prompt-cache-ro"] if v else []),
            "cache_prompt": lambda v: (["--cache-prompt"] if v else ["--no-cache-prompt"]),
            "cache_reuse": lambda v: ["--cache-reuse", str(int(v))],
            # Server runtime
            "parallel": lambda v: ["--parallel", str(int(v))],
            "threads_http": lambda v: ["--threads-http", str(int(v))],
            "timeout": lambda v: ["--timeout", str(int(v))],
            # Multimodal
            "mmproj": lambda v: ["--mmproj", str(v)],
            "mmproj_url": lambda v: ["--mmproj-url", str(v)],
            "mmproj_auto": lambda v: (["--mmproj-auto"] if v else ["--no-mmproj"]),
            "no_mmproj": lambda v: (["--no-mmproj"] if v else []),
            "mmproj_offload": lambda v: (["--mmproj-offload"] if v else ["--no-mmproj-offload"]),
            "no_mmproj_offload": lambda v: (["--no-mmproj-offload"] if v else []),
            "image_min_tokens": lambda v: ["--image-min-tokens", str(int(v))],
            "image_max_tokens": lambda v: ["--image-max-tokens", str(int(v))],
            # Speculative decoding
            "model_draft": lambda v: ["--model-draft", str(v)],
            "draft_max": lambda v: ["--draft-max", str(int(v))],
            "draft_min": lambda v: ["--draft-min", str(int(v))],
            "draft_p_min": lambda v: ["--draft-p-min", str(float(v))],
            "ctx_size_draft": lambda v: ["--ctx-size-draft", str(int(v))],
            "gpu_layers_draft": lambda v: ["--gpu-layers-draft", str(int(v))],
            "cpu_moe_draft": lambda v: (["--cpu-moe-draft"] if v else []),
            "n_cpu_moe_draft": lambda v: ["--n-cpu-moe-draft", str(int(v))],
            # Logging
            "log_file": lambda v: ["--log-file", str(v)],
            "log_colors": lambda v: (["--log-colors"] if v else []),
            "log_timestamps": lambda v: (["--log-timestamps"] if v else []),
            "log_verbosity": lambda v: ["--log-verbosity", str(v)],
            "no_perf": lambda v: (["--no-perf"] if v else []),
            # System prompt/chat
            "system_prompt": lambda v: ["--system-prompt", str(v)],
            "sys": lambda v: ["--system-prompt", str(v)],
            # Prompt convenience
            "prompt": lambda v: ["-p", str(v)],
        }

        def _coerce_port(value: Any, fallback: Any) -> int:
            if value is None or value == "":
                return int(fallback)
            try:
                return int(value)
            except (TypeError, ValueError) as exc:
                raise ServerError(f"Invalid port value: {value!r}") from exc

        host_value = args.get("host", self.config.default_host)
        if not host_value:
            host_value = self.config.default_host or "127.0.0.1"
        host = handler_utils.strip_host_brackets(str(host_value))
        client_host = handler_utils.resolve_client_host(host)
        # Required defaults
        raw_port = _coerce_port(args.get("port"), self.config.default_port)
        port = self._pick_port(host, raw_port)
        n_gpu_layers = int(
            args.get("n_gpu_layers", args.get("ngl", args.get("gpu_layers", self.config.default_n_gpu_layers)))
        )
        ctx_size = int(args.get("ctx_size", args.get("c", args.get("n_ctx", self.config.default_ctx_size))))
        threads = args.get("threads", args.get("t", self.config.default_threads))

        command = [str(self.config.executable_path), "-m", str(model_path)]
        # Always include core args
        command += ["--host", host, "--port", str(port), "-c", str(ctx_size), "-ngl", str(n_gpu_layers)]
        if threads is not None:
            command += ["-t", str(int(threads))]

        core_keys = {
            "port",
            "host",
            "threads",
            "t",
            "ctx_size",
            "c",
            "n_ctx",
            "n_gpu_layers",
            "ngl",
            "gpu_layers",
        }
        # Validate all provided keys are allowed
        invalid = [k for k in args if k not in allowed_formatters and k not in core_keys]
        if invalid and not getattr(self.config, "allow_unvalidated_args", False):
            raise ServerError(f"Unsupported llama.cpp server args: {sorted(invalid)}")

        # Apply boolean/kv flags from allowlist (exclude ones already encoded above)
        for k, v in args.items():
            if k in core_keys:
                continue
            fmt = allowed_formatters.get(k)
            if fmt:
                # Path safety for file arguments
                if k in {"grammar_file", "json_schema_file", "chat_template_file", "prompt_cache", "log_file", "lora_base", "control_vector", "mmproj"}:
                    p = Path(v)
                    if not self._is_path_allowed(p):
                        raise ServerError(f"File path for '{k}' must be under allowed directories.")
                if k == "lora":
                    vals = v if isinstance(v, (list, tuple)) else [v]
                    for item in vals:
                        if not self._is_path_allowed(Path(item)):
                            raise ServerError("LoRA path must be under allowed directories.")
                command += fmt(v)
            elif getattr(self.config, "allow_unvalidated_args", False):
                # Pass-through unknown args as --key value (bool True -> flag only)
                flag = f"--{k.replace('_', '-')}"
                if v is True:
                    command.append(flag)
                elif v is False or v is None:
                    pass
                else:
                    command += [flag, str(v)]

        from .http_utils import redact_cmd_args
        redacted_cmd = redact_cmd_args(command)
        self.logger.info(
            f"Starting Llama.cpp server for {model_label} on {host}:{port} with command: {' '.join(redacted_cmd)}")

        stdout_redir = asyncio.subprocess.PIPE
        stderr_redir = asyncio.subprocess.PIPE
        log_file_handle = None

        if self.config.log_output_file:
            try:
                log_file_handle = open(self.config.log_output_file, "ab")  # Append binary
                stdout_redir = log_file_handle
                stderr_redir = log_file_handle
                self.logger.info(f"Llama.cpp server logs will be written to: {self.config.log_output_file}")
            except _LLAMACPP_NONCRITICAL_EXCEPTIONS as e:
                self.logger.exception(f"Could not open log file {self.config.log_output_file}: {e}. Logging to PIPE.")

        try:
            cpe_kwargs = {
                "stdout": stdout_redir,
                "stderr": stderr_redir,
            }
            if platform.system() != "Windows":
                cpe_kwargs["preexec_fn"] = os.setsid
            else:
                # Use a fresh process group allowing CTRL_BREAK_EVENT (Windows-only)
                cpe_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            process = await asyncio.create_subprocess_exec(
                *command,
                **cpe_kwargs
            )
            # Poll HTTP health instead of fixed sleep
            base_url = handler_utils.build_base_url(client_host, port)
            t0 = time.perf_counter()
            readiness_timeout = getattr(self.config, "readiness_timeout", 30.0) or 30.0
            is_ready = await wait_for_http_ready(base_url, timeout_total=readiness_timeout, interval=0.5)

            if process.returncode is not None or not is_ready:
                # If logging to file, error might not be in stderr pipe here.
                # Consider reading last few lines of log_output_file if it exists.
                stderr_output = ""
                if stderr_redir == asyncio.subprocess.PIPE and process.stderr:  # Check if stderr was piped
                    try:
                        # Use timeout to prevent blocking indefinitely if server is still writing
                        stderr_timeout = getattr(self.config, "stderr_read_timeout", 5.0) or 5.0
                        err_bytes = await asyncio.wait_for(process.stderr.read(), timeout=stderr_timeout)
                        stderr_output = err_bytes.decode(errors='ignore').strip()
                    except asyncio.TimeoutError:
                        stderr_output = "(stderr read timed out after 5s)"
                        self.logger.warning("stderr read timed out during startup failure diagnosis")
                self.logger.error(
                    f"Llama.cpp server failed to start or become ready for {model_label}. Exit code: {process.returncode}. Stderr: {stderr_output}"
                )
                if log_file_handle:
                    log_file_handle.close()
                try:
                    if process.returncode is None:
                        # Stop server if it started but not ready
                        await self._terminate_process(process)
                except _LLAMACPP_NONCRITICAL_EXCEPTIONS:
                    pass
                self.metrics["start_errors"] += 1
                raise ServerError(f"Llama.cpp server failed to start. Stderr: {stderr_output}")

            self._active_server_process = process
            self._active_server_model = model_label
            self._active_server_port = port
            self._active_server_host = host
            t1 = time.perf_counter()
            self.metrics["starts"] += 1
            self.metrics["readiness_time_sum"] += max(0.0, t1 - t0)
            self.metrics["readiness_count"] += 1
            if log_file_handle:  # Store handle to close it later
                self._active_server_log_handle = log_file_handle
            else:
                self._active_server_log_handle = None
            self._start_stream_drainers(process)

            self.logger.info(f"Llama.cpp server started for {model_label} on {host}:{port} with PID {process.pid}.")
            return {"status": "started", "pid": process.pid, "model": model_label, "port": port, "host": host,
                    "command": ' '.join(redacted_cmd)}
        except Exception as e:
            if log_file_handle:
                log_file_handle.close()
            self.logger.error(f"Exception starting Llama.cpp server for {model_label}: {e}", exc_info=True)
            raise ServerError(f"Exception starting Llama.cpp server: {e}") from e

    async def stop_server(self, pid: Optional[int] = None, port: Optional[int] = None) -> str:
        if pid is not None:
            if self._active_server_process and pid == self._active_server_process.pid:
                port = None
            else:
                return await self._stop_unmanaged_pid(pid)
        if port is not None:
            if not self._active_server_process or self._active_server_port is None or int(port) != int(self._active_server_port):
                return f"No managed Llama.cpp server found on port {port}."

        if not self._active_server_process:
            return "No Llama.cpp server managed by this handler is currently running."

        process_to_stop = self._active_server_process
        pid = process_to_stop.pid
        model_name = self._active_server_model
        self.logger.info(f"Stopping Llama.cpp server (PID: {pid}, Model: {model_name}).")

        try:
            if process_to_stop.returncode is None:  # Still running
                if platform.system() == "Windows":
                    process_to_stop.terminate()
                else:
                    try:
                        pgid = await asyncio.to_thread(os.getpgid, pid)
                        await asyncio.to_thread(os.killpg, pgid, signal.SIGTERM)
                        self.logger.info(f"Sent SIGTERM to process group {pgid} (leader PID: {pid}).")
                    except ProcessLookupError:
                        self.logger.warning(f"Process {pid} not found for SIGTERM, likely already terminated.")
                        process_to_stop.terminate()  # Fallback
                    except _LLAMACPP_NONCRITICAL_EXCEPTIONS as e_pg:
                        self.logger.warning(
                            f"Failed to send SIGTERM to process group {pid}: {e_pg}. Falling back to PID.")
                        process_to_stop.terminate()

                try:
                    await asyncio.wait_for(process_to_stop.wait(), timeout=10)
                    self.logger.info(f"Llama.cpp server PID {pid} (Model: {model_name}) terminated gracefully.")
                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"Llama.cpp server PID {pid} (Model: {model_name}) did not terminate gracefully. Killing.")
                    if platform.system() == "Windows":
                        if hasattr(process_to_stop, "send_signal"):
                            with contextlib.suppress(_LLAMACPP_NONCRITICAL_EXCEPTIONS):
                                process_to_stop.send_signal(signal.CTRL_BREAK_EVENT)
                        if hasattr(process_to_stop, "terminate"):
                            process_to_stop.terminate()
                        if hasattr(process_to_stop, "kill"):
                            process_to_stop.kill()
                    else:
                        try:
                            pgid = await asyncio.to_thread(os.getpgid, pid)  # Re-fetch pgid in case
                            await asyncio.to_thread(os.killpg, pgid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError, OSError) as e:
                            # pgid may not be available if getpgid failed; log using pid
                            self.logger.warning(
                                f"Failed to kill process group for PID {pid}: {e}. Attempting PID SIGKILL fallback.")
                            # Try direct SIGKILL to PID; if that fails, try process.kill() if available
                            try:
                                await asyncio.to_thread(os.kill, pid, signal.SIGKILL)
                                self.logger.info(f"Sent SIGKILL to PID {pid} (fallback).")
                            except ProcessLookupError:
                                self.logger.warning(
                                    f"PID {pid} already exited when attempting SIGKILL fallback.")
                            except _LLAMACPP_NONCRITICAL_EXCEPTIONS as e_killpid:
                                self.logger.debug(
                                    f"os.kill fallback failed for PID {pid}: {e_killpid}; checking for process.kill()"
                                )
                                if hasattr(process_to_stop, "kill"):
                                    try:
                                        process_to_stop.kill()
                                        self.logger.info(
                                            f"Invoked process.kill() for PID {pid} (final fallback)."
                                        )
                                    except _LLAMACPP_NONCRITICAL_EXCEPTIONS as e_pkill:
                                        self.logger.warning(
                                            f"process.kill() failed for PID {pid}: {e_pkill}")
                    await process_to_stop.wait()
            else:
                self.logger.info(
                    f"Llama.cpp server PID {pid} (Model: {model_name}) was already stopped (return code: {process_to_stop.returncode}).")

            if self._active_server_log_handle:
                self._active_server_log_handle.close()
                self._active_server_log_handle = None
            self._stop_stream_drainers()

            self._active_server_process = None
            self._active_server_model = None
            self._active_server_port = None
            self._active_server_host = None
            self.metrics["stops"] += 1
            return f"Llama.cpp server PID {pid} (Model: {model_name}) stopped."

        except Exception as e:
            self.logger.error(f"Error stopping Llama.cpp server PID {pid}: {e}", exc_info=True)
            # Clear state even on error to allow trying to start a new one
            if self._active_server_log_handle:
                self._active_server_log_handle.close()
                self._active_server_log_handle = None
            self._stop_stream_drainers()
            self._active_server_process = None
            self._active_server_model = None
            self._active_server_port = None
            self._active_server_host = None
            raise ServerError(f"Error stopping Llama.cpp server: {e}") from e

    async def get_server_status(self) -> dict[str, Any]:
        if self._active_server_process and self._active_server_process.returncode is None:
            return {
                "status": "running",
                "pid": self._active_server_process.pid,
                "model": self._active_server_model,
                "port": self._active_server_port,
                "host": self._active_server_host,
                "log_file": str(
                    self.config.log_output_file) if self.config.log_output_file and self._active_server_log_handle else None
            }
        return {"status": "stopped", "model": None, "pid": None, "port": None, "host": None}

    async def inference(self, prompt: Optional[str] = None, messages: Optional[list[dict[str, str]]] = None,
                        api_endpoint: str = "/v1/chat/completions",  # or /completion
                        **kwargs) -> dict[str, Any]:
        if not self._active_server_process or self._active_server_process.returncode is not None:
            raise ServerError("Llama.cpp server is not running or not managed by this handler.")

        active_host = self._active_server_host or self.config.default_host or "127.0.0.1"
        active_port = self._active_server_port or self.config.default_port
        client_host = handler_utils.resolve_client_host(active_host)
        base_url = handler_utils.build_base_url(client_host, active_port)
        # Ensure exactly one slash between base and path
        target_url = f"{base_url}/{api_endpoint.lstrip('/')}"

        payload = kwargs.copy()
        timeout = payload.pop("timeout", None)
        payload.pop("stream", None)
        payload.pop("api_endpoint", None)
        prompt_value = prompt
        if prompt_value is None and "prompt" in payload:
            prompt_value = payload.pop("prompt")
        messages_value = messages
        if messages_value is None and "messages" in payload:
            messages_value = payload.pop("messages")

        if self._is_chat_endpoint(api_endpoint):
            if messages_value:
                payload["messages"] = messages_value
            elif prompt_value is not None:
                payload["messages"] = [{"role": "user", "content": prompt_value}]
            else:
                raise InferenceError("Either 'prompt' or 'messages' must be provided for inference.")
        else:
            if prompt_value is None and messages_value:
                prompt_value = self._messages_to_prompt(messages_value)
            if prompt_value is None:
                raise InferenceError("Prompt is required for completion endpoint inference.")
            payload["prompt"] = prompt_value

        payload["stream"] = False

        if self._is_chat_endpoint(api_endpoint):
            msg_count = len(payload.get("messages", [])) if isinstance(payload.get("messages"), list) else 0
            self.logger.debug(
                f"Sending Llama.cpp chat inference request to {target_url} (messages={msg_count})"
            )
        else:
            prompt_len = len(str(payload.get("prompt", "")))
            self.logger.debug(
                f"Sending Llama.cpp completion inference request to {target_url} (prompt_len={prompt_len})"
            )

        t0 = time.perf_counter()
        http_timeout = timeout if timeout is not None else getattr(self.config, "http_timeout", 120.0)
        async with create_async_client(timeout=http_timeout) as client:
            try:
                result = await request_json(client, "POST", target_url, json=payload, headers={"Content-Type": "application/json"})
                t1 = time.perf_counter()
                self.metrics["inference_count"] += 1
                self.metrics["inference_time_sum"] += max(0.0, t1 - t0)
                self.logger.debug("Llama.cpp inference successful.")
                return result
            except Exception as e:
                status = http_utils.get_http_status_from_exception(e)
                if status is not None:
                    error_text = http_utils.get_http_error_text(e)
                    self.logger.error(
                        f"Llama.cpp API error ({status}) from {target_url}: {error_text}",
                        exc_info=True
                    )
                    self.metrics["inference_error_count"] += 1
                    raise InferenceError(f"Llama.cpp API error ({status}): {error_text}") from e
                if http_utils.is_network_error(e):
                    self.logger.error(
                        f"Could not connect or communicate with Llama.cpp server at {target_url}: {e}",
                        exc_info=True
                    )
                    self.metrics["inference_error_count"] += 1
                    raise ServerError(f"Could not connect/communicate with Llama.cpp server at {target_url}: {e}") from e
                error_text = http_utils.get_http_error_text(e)
                self.logger.error(
                    f"Unexpected error during Llama.cpp inference to {target_url}: {error_text}",
                    exc_info=True,
                )
                self.metrics["inference_error_count"] += 1
                raise InferenceError(f"Unexpected error during Llama.cpp inference: {error_text}") from e

    async def stream_inference(self, prompt: Optional[str] = None, messages: Optional[list[dict[str, str]]] = None,
                               api_endpoint: str = "/v1/chat/completions", **kwargs):
        if not self._active_server_process or self._active_server_process.returncode is not None:
            raise ServerError("Llama.cpp server is not running or not managed by this handler.")
        active_host = self._active_server_host or self.config.default_host or "127.0.0.1"
        active_port = self._active_server_port or self.config.default_port
        client_host = handler_utils.resolve_client_host(active_host)
        base_url = handler_utils.build_base_url(client_host, active_port)
        target_url = f"{base_url}/{api_endpoint.lstrip('/')}"
        payload = kwargs.copy()
        timeout = payload.pop("timeout", None)
        payload.pop("stream", None)
        payload.pop("api_endpoint", None)
        prompt_value = prompt
        if prompt_value is None and "prompt" in payload:
            prompt_value = payload.pop("prompt")
        messages_value = messages
        if messages_value is None and "messages" in payload:
            messages_value = payload.pop("messages")

        if self._is_chat_endpoint(api_endpoint):
            if messages_value:
                payload["messages"] = messages_value
            elif prompt_value is not None:
                payload["messages"] = [{"role": "user", "content": prompt_value}]
            else:
                raise InferenceError("Either 'prompt' or 'messages' must be provided for stream_inference.")
        else:
            if prompt_value is None and messages_value:
                prompt_value = self._messages_to_prompt(messages_value)
            if prompt_value is None:
                raise InferenceError("Prompt is required for completion endpoint streaming.")
            payload["prompt"] = prompt_value
        payload["stream"] = True
        headers = {"Content-Type": "application/json"}

        http_timeout = timeout if timeout is not None else getattr(self.config, "http_timeout", 120.0)
        async with create_async_client(timeout=http_timeout) as client:
            try:
                async with client.stream("POST", target_url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    done_sent = False
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        l = line.strip()
                        if not l:
                            continue
                        if l.startswith("data:"):
                            if l.strip().lower() == "data: [done]".lower():
                                done_sent = True
                            yield ensure_sse_line(l)
                        else:
                            yield openai_delta_chunk(l)
                    if not done_sent:
                        yield sse_done()
            except _LLAMACPP_NONCRITICAL_EXCEPTIONS as e:
                status = http_utils.get_http_status_from_exception(e)
                if status is not None:
                    msg = http_utils.get_http_error_text(e)
                    yield sse_data({"error": {"message": f"HTTP error ({status}): {msg}", "type": "llamacpp_stream_error"}})
                elif http_utils.is_network_error(e):
                    yield sse_data({"error": {"message": f"Network error: {e}", "type": "llamacpp_stream_error"}})
                else:
                    yield sse_data({"error": {"message": f"Stream error: {str(e)}", "type": "llamacpp_stream_error"}})

    # --- Cleanup ---
    def _cleanup_managed_server_sync(self):
        """Synchronous cleanup for atexit/signal handlers.

        Uses _safe_log to avoid errors when logging sinks are closed during shutdown.
        """
        # Guard against duplicate cleanup
        if self._cleanup_done:
            return
        self._cleanup_done = True

        if self._active_server_process and self._active_server_process.returncode is None:
            proc = self._active_server_process
            pid = proc.pid
            self._safe_log("info", f"Cleanup: stopping Llama.cpp server PID {pid}")
            try:
                if platform.system() == "Windows":
                    if hasattr(proc, "terminate"):
                        proc.terminate()
                else:
                    try:
                        pgid = os.getpgid(pid)
                        os.killpg(pgid, signal.SIGTERM)
                        self._safe_log("debug", f"Sent SIGTERM to process group {pgid}")
                    except ProcessLookupError:
                        if hasattr(proc, "terminate"):
                            proc.terminate()
                    except _LLAMACPP_NONCRITICAL_EXCEPTIONS:
                        if hasattr(proc, "terminate"):
                            proc.terminate()

                # Best-effort wait: try to reap using running event loop if available
                try:
                    loop = asyncio.get_running_loop()
                    # Schedule async wait in the event loop
                    asyncio.run_coroutine_threadsafe(proc.wait(), loop)
                except RuntimeError:
                    # No running event loop - best effort, OS will reap eventually
                    pass

            except _LLAMACPP_NONCRITICAL_EXCEPTIONS:
                # Best-effort kill if needed
                if proc.returncode is None:
                    if platform.system() == "Windows":
                        if hasattr(proc, "kill"):
                            with contextlib.suppress(_LLAMACPP_NONCRITICAL_EXCEPTIONS):
                                proc.kill()
                    else:
                        try:
                            os.killpg(os.getpgid(pid), signal.SIGKILL)
                        except (ProcessLookupError, PermissionError, OSError):
                            if hasattr(proc, "kill"):
                                with contextlib.suppress(_LLAMACPP_NONCRITICAL_EXCEPTIONS):
                                    proc.kill()
            if self._active_server_log_handle:
                with contextlib.suppress(_LLAMACPP_NONCRITICAL_EXCEPTIONS):
                    self._active_server_log_handle.close()
            self._stop_stream_drainers()

        self._active_server_process = None
        self._active_server_model = None
        self._active_server_port = None
        self._active_server_host = None
        self._active_server_log_handle = None
        self._safe_log("debug", "Cleanup complete")

    def _signal_handler(self, sig, frame):
        self._cleanup_managed_server_sync()
        # Let other handlers run, don't sys.exit here unless it's the main app's job
        # sys.exit(0)

    def _setup_signal_handlers(self):
        import atexit
        # Register for process exit
        atexit.register(self._cleanup_managed_server_sync)
        self.logger.info("Registered atexit synchronous cleanup for LlamaCppHandler.")
        # Signal handling for Ctrl+C etc.
        # Note: Multiple signal handlers can be tricky. Ensure this doesn't conflict.
        # signal.signal(signal.SIGINT, self._signal_handler)
        # signal.signal(signal.SIGTERM, self._signal_handler)

#
# End of LlamaCpp_Handler.py
########################################################################################################################
