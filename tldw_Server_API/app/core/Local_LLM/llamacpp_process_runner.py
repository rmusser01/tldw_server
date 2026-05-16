"""One-process runner for managed llama.cpp runtime profiles."""

from __future__ import annotations

import asyncio
import contextlib
import os
import platform
import signal
import subprocess  # nosec B404 - used without shell for llama-server process control
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from tldw_Server_API.app.core.Local_LLM import handler_utils, http_utils
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ModelNotFoundError, ServerError
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Schemas import LlamaCppConfig
from tldw_Server_API.app.core.Local_LLM.llamacpp_provider_service import _redact_log_line
from tldw_Server_API.app.core.Local_LLM.llamacpp_runtime_models import (
    LlamaCppPortPolicy,
    LlamaCppProfile,
    LlamaCppRuntime,
    LlamaCppRuntimeState,
)

_MAX_LOG_LINES = 1000
_MAX_LOG_BYTES = 256 * 1024
_PATH_ARG_KEYS = {
    "grammar_file",
    "json_schema_file",
    "chat_template_file",
    "prompt_cache",
    "log_file",
    "lora_base",
    "control_vector",
    "mmproj",
    "model_draft",
}


async def wait_for_http_ready(*args: Any, **kwargs: Any) -> bool:
    """Proxy readiness helper so tests can patch this module only."""
    return await http_utils.wait_for_http_ready(*args, **kwargs)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _flatten_repeatable(flag: str, value: Any) -> list[str]:
    values = value if isinstance(value, (list, tuple)) else [value]
    command: list[str] = []
    for item in values:
        command.extend([flag, str(item)])
    return command


def _server_arg_formatters() -> dict[str, Callable[[Any], list[str]]]:
    """Return the supported llama-server args used by the Admin UI and V1 APIs.

    TODO(TASK-397): consolidate this map with LlamaCppHandler once the new
    supervisor owns the default-profile compatibility path.
    """
    return {
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
        "verbose": lambda v: ["--verbose"] if v else [],
        "log_disable": lambda v: ["--log-disable"] if v else [],
        "no_mmap": lambda v: ["--no-mmap"] if v else [],
        "mlock": lambda v: ["--mlock"] if v else [],
        "main_gpu": lambda v: ["--main-gpu", str(int(v))],
        "mg": lambda v: ["--main-gpu", str(int(v))],
        "split_mode": lambda v: ["--split-mode", str(v)],
        "sm": lambda v: ["--split-mode", str(v)],
        "row_split": lambda v: ["--split-mode", "row"] if v else [],
        "main_kv": lambda v: ["--main-kv", str(int(v))],
        "no_kv_offload": lambda v: ["--no-kv-offload"] if v else [],
        "cpu_moe": lambda v: ["--cpu-moe"] if v else [],
        "n_cpu_moe": lambda v: ["--n-cpu-moe", str(int(v))],
        "rope_scaling_type": lambda v: ["--rope-scaling", str(v)],
        "rope_scaling": lambda v: ["--rope-scaling", str(v)],
        "tensor_split": lambda v: ["--tensor-split", ",".join(map(str, v))]
        if isinstance(v, (list, tuple))
        else ["--tensor-split", str(v)],
        "rope_freq_base": lambda v: ["--rope-freq-base", str(float(v))],
        "rope_freq_scale": lambda v: ["--rope-freq-scale", str(float(v))],
        "rope_scale": lambda v: ["--rope-freq-scale", str(float(v))],
        "flash_attn": lambda v: ["--flash-attn", str(v)] if isinstance(v, str) else (["--flash-attn"] if v else []),
        "cont_batching": lambda v: ["--cont-batching"] if v else ["--no-cont-batching"],
        "no_cont_batching": lambda v: ["--no-cont-batching"] if v else [],
        "context_shift": lambda v: ["--context-shift"] if v else ["--no-context-shift"],
        "streaming_llm": lambda v: ["--context-shift"] if v else [],
        "lora": lambda v: _flatten_repeatable("--lora", v),
        "lora_scaled": lambda v: ["--lora-scaled", str(v[0]), str(v[1])]
        if isinstance(v, (list, tuple)) and len(v) == 2
        else (["--lora-scaled", str(v)] if v is not None else []),
        "lora_base": lambda v: ["--lora-base", str(v)],
        "control_vector": lambda v: ["--control-vector", str(v)],
        "cache_type_k": lambda v: ["--cache-type-k", str(v)],
        "cache_type_v": lambda v: ["--cache-type-v", str(v)],
        "cache_type": lambda v: ["--cache-type-k", str(v), "--cache-type-v", str(v)],
        "hf_repo": lambda v: ["--hf-repo", str(v)],
        "hf_file": lambda v: ["--hf-file", str(v)],
        "offline": lambda v: ["--offline"] if v else [],
        "conversation": lambda v: ["--conversation"] if v else [],
        "cnv": lambda v: ["--conversation"] if v else [],
        "no_conversation": lambda v: ["--no-conversation"] if v else [],
        "no_cnv": lambda v: ["--no-conversation"] if v else [],
        "interactive": lambda v: ["--interactive"] if v else [],
        "i": lambda v: ["--interactive"] if v else [],
        "interactive_first": lambda v: ["--interactive-first"] if v else [],
        "if": lambda v: ["--interactive-first"] if v else [],
        "single_turn": lambda v: ["--single-turn"] if v else [],
        "st": lambda v: ["--single-turn"] if v else [],
        "jinja": lambda v: ["--jinja"] if v else [],
        "chat_template": lambda v: ["--chat-template", str(v)],
        "chat_template_file": lambda v: ["--chat-template-file", str(v)],
        "in_prefix": lambda v: ["--in-prefix", str(v)],
        "in_suffix": lambda v: ["--in-suffix", str(v)],
        "in_prefix_bos": lambda v: ["--in-prefix-bos"] if v else [],
        "reverse_prompt": lambda v: ["--reverse-prompt", str(v)],
        "r": lambda v: ["--reverse-prompt", str(v)],
        "predict": lambda v: ["-n", str(int(v))],
        "n": lambda v: ["-n", str(int(v))],
        "keep": lambda v: ["--keep", str(int(v))],
        "ignore_eos": lambda v: ["--ignore-eos"] if v else [],
        "no_context_shift": lambda v: ["--no-context-shift"] if v else [],
        "temp": lambda v: ["--temp", str(float(v))],
        "seed": lambda v: ["-s", str(int(v))],
        "dynatemp_range": lambda v: ["--dynatemp-range", str(float(v))],
        "top_k": lambda v: ["--top-k", str(int(v))],
        "top_p": lambda v: ["--top-p", str(float(v))],
        "min_p": lambda v: ["--min-p", str(float(v))],
        "typical": lambda v: ["--typical", str(float(v))],
        "repeat_penalty": lambda v: ["--repeat-penalty", str(float(v))],
        "repeat_last_n": lambda v: ["--repeat-last-n", str(int(v))],
        "presence_penalty": lambda v: ["--presence-penalty", str(float(v))],
        "frequency_penalty": lambda v: ["--frequency-penalty", str(float(v))],
        "dry_multiplier": lambda v: ["--dry-multiplier", str(float(v))],
        "dry_base": lambda v: ["--dry-base", str(float(v))],
        "dry_allowed_length": lambda v: ["--dry-allowed-length", str(int(v))],
        "mirostat": lambda v: ["--mirostat", str(int(v))],
        "mirostat_lr": lambda v: ["--mirostat-lr", str(float(v))],
        "mirostat_ent": lambda v: ["--mirostat-ent", str(float(v))],
        "cpu_mask": lambda v: ["--cpu-mask", str(v)],
        "cpu_range": lambda v: ["--cpu-range", str(v)],
        "numa": lambda v: ["--numa", str(v)] if isinstance(v, str) else (["--numa"] if v else []),
        "grammar": lambda v: ["--grammar", str(v)],
        "grammar_file": lambda v: ["--grammar-file", str(v)],
        "json_schema": lambda v: ["--json-schema", str(v)],
        "json_schema_file": lambda v: ["--json-schema-file", str(v)],
        "j": lambda v: ["-j", str(v)],
        "reasoning_format": lambda v: ["--reasoning-format", str(v)],
        "reasoning_budget": lambda v: ["--reasoning-budget", str(int(v))],
        "prompt_cache": lambda v: ["--prompt-cache", str(v)],
        "prompt_cache_all": lambda v: ["--prompt-cache-all"] if v else [],
        "prompt_cache_ro": lambda v: ["--prompt-cache-ro"] if v else [],
        "cache_prompt": lambda v: ["--cache-prompt"] if v else ["--no-cache-prompt"],
        "cache_reuse": lambda v: ["--cache-reuse", str(int(v))],
        "parallel": lambda v: ["--parallel", str(int(v))],
        "threads_http": lambda v: ["--threads-http", str(int(v))],
        "timeout": lambda v: ["--timeout", str(int(v))],
        "mmproj": lambda v: ["--mmproj", str(v)],
        "mmproj_url": lambda v: ["--mmproj-url", str(v)],
        "mmproj_auto": lambda v: ["--mmproj-auto"] if v else ["--no-mmproj"],
        "no_mmproj": lambda v: ["--no-mmproj"] if v else [],
        "mmproj_offload": lambda v: ["--mmproj-offload"] if v else ["--no-mmproj-offload"],
        "no_mmproj_offload": lambda v: ["--no-mmproj-offload"] if v else [],
        "image_min_tokens": lambda v: ["--image-min-tokens", str(int(v))],
        "image_max_tokens": lambda v: ["--image-max-tokens", str(int(v))],
        "model_draft": lambda v: ["--model-draft", str(v)],
        "draft_max": lambda v: ["--draft-max", str(int(v))],
        "draft_min": lambda v: ["--draft-min", str(int(v))],
        "draft_p_min": lambda v: ["--draft-p-min", str(float(v))],
        "ctx_size_draft": lambda v: ["--ctx-size-draft", str(int(v))],
        "gpu_layers_draft": lambda v: ["--gpu-layers-draft", str(int(v))],
        "cpu_moe_draft": lambda v: ["--cpu-moe-draft"] if v else [],
        "n_cpu_moe_draft": lambda v: ["--n-cpu-moe-draft", str(int(v))],
        "log_file": lambda v: ["--log-file", str(v)],
        "log_colors": lambda v: ["--log-colors"] if v else [],
        "log_timestamps": lambda v: ["--log-timestamps"] if v else [],
        "log_verbosity": lambda v: ["--log-verbosity", str(v)],
        "no_perf": lambda v: ["--no-perf"] if v else [],
        "system_prompt": lambda v: ["--system-prompt", str(v)],
        "sys": lambda v: ["--system-prompt", str(v)],
        "prompt": lambda v: ["-p", str(v)],
    }


class LlamaCppProcessRunner:
    """Owns one llama.cpp server process for one managed profile."""

    def __init__(self, config: LlamaCppConfig, profile_id: str):
        self.config = config
        self.profile_id = profile_id
        self.models_dir = Path(config.models_dir)
        self._process: asyncio.subprocess.Process | None = None
        self._profile: LlamaCppProfile | None = None
        self._model_path: Path | None = None
        self._host: str | None = None
        self._port: int | None = None
        self._endpoint: str | None = None
        self._started_at: str | None = None
        self._stopped_at: str | None = None
        self._last_health_at: str | None = None
        self._exit_code: int | None = None
        self._log_handle: Any | None = None
        self._log_file_path: Path | None = None
        self._redacted_command: list[str] = []
        self._last_error: str | None = None
        self._warnings: list[str] = []
        self._health: dict[str, object] = {}
        self._failed = False
        self._message: str | None = None
        self._stream_drain_tasks: list[asyncio.Task[None]] = []

    def _is_port_free(self, host: str, port: int) -> bool:
        return handler_utils.is_port_free(host, port)

    def _is_path_allowed(self, path: Path) -> bool:
        allowed_paths = handler_utils.build_allowed_paths(
            self.models_dir,
            getattr(self.config, "allowed_paths", None),
        )
        return handler_utils.is_path_allowed(path, allowed_paths)

    async def start(self, model_path: Path, profile: LlamaCppProfile) -> LlamaCppRuntime:
        """Start this runner's process for the supplied profile and model."""
        if self._process is not None and self._process.returncode is None:
            await self.stop()

        resolved_model_path = self._validate_model_path(model_path)
        executable_path = Path(self.config.executable_path).expanduser()
        if not executable_path.is_file():
            raise ServerError(f"Llama.cpp server executable not found at {self.config.executable_path}")

        args = dict(profile.server_args)
        self._check_denylist(args)
        host = handler_utils.strip_host_brackets(profile.host or self.config.default_host or "127.0.0.1")
        port = self._resolve_port(host, profile)
        command = self._build_command(executable_path, resolved_model_path, host, port, args)
        redacted_command = http_utils.redact_cmd_args(command)
        stdout_target, stderr_target, log_handle, log_file_path = self._open_log_targets()
        client_host = handler_utils.resolve_client_host(host)
        base_url = handler_utils.build_base_url(client_host, port)

        self._record_start_attempt(
            profile=profile,
            model_path=resolved_model_path,
            host=host,
            port=port,
            endpoint=base_url,
            log_handle=log_handle,
            log_file_path=log_file_path,
            redacted_command=redacted_command,
        )

        try:
            process = await self._spawn(command, stdout_target=stdout_target, stderr_target=stderr_target)
            self._process = process
            self._start_stream_drainers(process)
            readiness_timeout = getattr(self.config, "readiness_timeout", 30.0) or 30.0
            is_ready = await wait_for_http_ready(base_url, timeout_total=readiness_timeout, interval=0.5)
            if process.returncode is not None or not is_ready:
                await self._terminate_process(process)
                message = "Llama.cpp server failed to start or become ready."
                self._record_failure(message, process)
                raise ServerError(message)
        except Exception as exc:
            if log_handle is not None and not log_handle.closed:
                self._close_log_handle()
            if isinstance(exc, ServerError):
                raise
            message = f"Exception starting Llama.cpp server: {exc}"
            self._record_failure(message)
            raise ServerError(message) from exc

        self._started_at = _utc_now()
        self._stopped_at = None
        self._exit_code = None
        self._failed = False
        self._last_error = None
        self._last_health_at = _utc_now()
        self._health = {"ready": True}
        self._message = None
        logger.info("Started llama.cpp profile {} on {}:{} with PID {}", profile.profile_id, host, port, process.pid)
        return self.status()

    async def stop(self) -> LlamaCppRuntime:
        """Stop this runner's process if it is currently running."""
        process = self._process
        if process is None:
            self._stopped_at = self._stopped_at or _utc_now()
            return self.status()

        if process.returncode is None:
            await self._terminate_process(process)
        else:
            self._exit_code = process.returncode
        self._stop_stream_drainers()
        self._close_log_handle()
        self._process = None
        self._failed = False
        self._stopped_at = _utc_now()
        self._message = "Stopped"
        return self.status()

    def status(self) -> LlamaCppRuntime:
        """Return the current observed runtime state."""
        process = self._process
        if process is not None and process.returncode is None:
            state = LlamaCppRuntimeState.RUNNING
            pid = process.pid
            exit_code = None
        elif self._failed:
            state = LlamaCppRuntimeState.FAILED
            pid = None
            exit_code = self._exit_code if self._exit_code is not None else getattr(process, "returncode", None)
        elif process is not None and process.returncode is not None:
            state = LlamaCppRuntimeState.FAILED
            pid = None
            exit_code = process.returncode
            self._exit_code = process.returncode
            self._last_error = self._last_error or "Llama.cpp server process exited unexpectedly."
            self._message = self._message or self._last_error
            self._health = {"ready": False}
        elif self._started_at is None and self._stopped_at is None:
            state = LlamaCppRuntimeState.DEFINED
            pid = None
            exit_code = None
        else:
            state = LlamaCppRuntimeState.STOPPED
            pid = None
            exit_code = self._exit_code if self._exit_code is not None else getattr(process, "returncode", None)
        return self._runtime(state=state, pid=pid, exit_code=exit_code)

    def tail_logs(self, lines: int) -> dict[str, object]:
        """Return a bounded redacted tail from this runner's owned log file."""
        line_count = max(1, min(int(lines), _MAX_LOG_LINES))
        log_path = self._log_file_path
        if log_path is None:
            return {"lines": [], "truncated": False, "warnings": ["No managed llama.cpp log file is configured."]}
        try:
            resolved_log_path = log_path.expanduser().resolve(strict=False)
        except OSError:
            return {"lines": [], "truncated": False, "warnings": ["Managed llama.cpp log file is unavailable."]}
        if not resolved_log_path.is_file():
            return {"lines": [], "truncated": False, "warnings": ["Managed llama.cpp log file is unavailable."]}

        try:
            file_size = resolved_log_path.stat().st_size
            read_size = min(file_size, _MAX_LOG_BYTES)
            with resolved_log_path.open("rb") as handle:
                handle.seek(max(0, file_size - read_size))
                raw = handle.read(read_size)
        except OSError:
            return {"lines": [], "truncated": False, "warnings": ["Managed llama.cpp log file is unavailable."]}

        decoded = raw.decode("utf-8", errors="replace")
        all_lines = decoded.splitlines()
        selected = all_lines[-line_count:]
        return {
            "lines": [_redact_log_line(line) for line in selected],
            "truncated": file_size > read_size or len(all_lines) > len(selected),
            "warnings": [],
        }

    def cleanup_sync(self) -> None:
        """Best-effort synchronous cleanup for app shutdown paths."""
        process = self._process
        if process is not None and process.returncode is None:
            with contextlib.suppress(Exception):
                if platform.system() == "Windows":
                    process.terminate()
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        self._stop_stream_drainers()
        self._close_log_handle()
        self._process = None
        self._stopped_at = self._stopped_at or _utc_now()

    def _runtime(
        self,
        *,
        state: LlamaCppRuntimeState,
        pid: int | None,
        exit_code: int | None,
    ) -> LlamaCppRuntime:
        return LlamaCppRuntime(
            profile_id=self.profile_id,
            state=state,
            pid=pid,
            host=self._host,
            port=self._port,
            endpoint=self._endpoint,
            model_id=self._profile.model_id if self._profile else None,
            model_path=str(self._model_path) if self._model_path else None,
            resolved_args=list(self._redacted_command),
            started_at=self._started_at,
            stopped_at=self._stopped_at,
            last_health_at=self._last_health_at,
            exit_code=exit_code,
            last_error=self._last_error,
            log_tail_available=self._log_file_path is not None,
            log_file=str(self._log_file_path) if self._log_file_path else None,
            command=list(self._redacted_command),
            warnings=list(self._warnings),
            health=dict(self._health),
            message=self._message,
        )

    def _record_start_attempt(
        self,
        *,
        profile: LlamaCppProfile,
        model_path: Path,
        host: str,
        port: int,
        endpoint: str,
        log_handle: Any | None,
        log_file_path: Path | None,
        redacted_command: list[str],
    ) -> None:
        self._profile = profile
        self._model_path = model_path
        self._host = host
        self._port = port
        self._endpoint = endpoint
        self._started_at = _utc_now()
        self._stopped_at = None
        self._last_health_at = None
        self._exit_code = None
        self._log_handle = log_handle
        self._log_file_path = log_file_path
        self._redacted_command = redacted_command
        self._last_error = None
        self._warnings = []
        self._health = {"ready": False}
        self._failed = False
        self._message = None

    def _record_failure(self, message: str, process: asyncio.subprocess.Process | None = None) -> None:
        self._failed = True
        self._message = message
        self._last_error = message
        self._health = {"ready": False}
        self._stopped_at = _utc_now()
        if process is not None:
            self._exit_code = process.returncode
        self._stop_stream_drainers()
        self._close_log_handle()
        self._process = None

    def _validate_model_path(self, model_path: Path) -> Path:
        try:
            resolved_model_path = Path(model_path).expanduser().resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            raise ServerError("Model path could not be resolved.") from exc
        if not self._is_path_allowed(resolved_model_path):
            raise ServerError("Model path must be under allowed directories.")
        if resolved_model_path.suffix.lower() != ".gguf":
            raise ServerError("Model path must reference a GGUF file.")
        if not resolved_model_path.is_file():
            raise ModelNotFoundError(f"Model file {resolved_model_path.name} was not found.")
        return resolved_model_path

    def _check_denylist(self, args: dict[str, Any]) -> None:
        try:
            handler_utils.check_denylist(
                args,
                allow_secrets=getattr(self.config, "allow_cli_secrets", False),
            )
        except ValueError as exc:
            raise ServerError(str(exc)) from exc

    def _resolve_port(self, host: str, profile: LlamaCppProfile) -> int:
        if profile.port_policy == LlamaCppPortPolicy.EXPLICIT:
            port = int(profile.port)
            if not self._is_port_free(host, port):
                raise ServerError(f"Explicit llama.cpp port {port} is not available on {host}.")
            return port
        max_probe = int(getattr(self.config, "port_probe_max", 10) or 0)
        for offset in range(max_probe + 1):
            candidate = int(profile.port) + offset
            if self._is_port_free(host, candidate):
                return candidate
        return int(profile.port)

    def _build_command(
        self,
        executable_path: Path,
        model_path: Path,
        host: str,
        port: int,
        args: dict[str, Any],
    ) -> list[str]:
        ctx_size = int(args.get("ctx_size", args.get("c", args.get("n_ctx", self.config.default_ctx_size))))
        n_gpu_layers = int(
            args.get("n_gpu_layers", args.get("ngl", args.get("gpu_layers", self.config.default_n_gpu_layers)))
        )
        command = [
            str(executable_path),
            "-m",
            str(model_path),
            "--host",
            host,
            "--port",
            str(port),
            "-c",
            str(ctx_size),
            "-ngl",
            str(n_gpu_layers),
        ]
        threads = args.get("threads", args.get("t", self.config.default_threads))
        if threads is not None:
            command.extend(["-t", str(int(threads))])

        core_keys = {"port", "host", "threads", "t", "ctx_size", "c", "n_ctx", "n_gpu_layers", "ngl", "gpu_layers"}
        formatters = _server_arg_formatters()
        invalid = [key for key in args if key not in formatters and key not in core_keys]
        if invalid and not getattr(self.config, "allow_unvalidated_args", False):
            raise ServerError(f"Unsupported llama.cpp server args: {sorted(invalid)}")

        for key, value in args.items():
            if key in core_keys:
                continue
            formatter = formatters.get(key)
            if formatter is not None:
                self._validate_arg_path(key, value)
                command.extend(formatter(value))
                continue
            if getattr(self.config, "allow_unvalidated_args", False):
                flag = f"--{key.replace('_', '-')}"
                if value is True:
                    command.append(flag)
                elif value is False or value is None:
                    continue
                else:
                    command.extend([flag, str(value)])
        return command

    def _validate_arg_path(self, key: str, value: Any) -> None:
        if key in _PATH_ARG_KEYS and not self._is_path_allowed(Path(value)):
            raise ServerError(f"File path for '{key}' must be under allowed directories.")
        if key == "lora":
            values = value if isinstance(value, (list, tuple)) else [value]
            for item in values:
                if not self._is_path_allowed(Path(item)):
                    raise ServerError("LoRA path must be under allowed directories.")
        if key == "lora_scaled":
            paths = [value[0]] if isinstance(value, (list, tuple)) and value else [value]
            for item in paths:
                if item is not None and not self._is_path_allowed(Path(item)):
                    raise ServerError("LoRA path must be under allowed directories.")

    def _open_log_targets(self) -> tuple[Any, Any, Any | None, Path | None]:
        log_file = getattr(self.config, "log_output_file", None)
        if not log_file:
            return asyncio.subprocess.PIPE, asyncio.subprocess.PIPE, None, None
        log_path = Path(log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("ab")
        return log_handle, log_handle, log_handle, log_path

    async def _spawn(self, command: list[str], *, stdout_target: Any, stderr_target: Any) -> asyncio.subprocess.Process:
        create_kwargs: dict[str, Any] = {"stdout": stdout_target, "stderr": stderr_target}
        if platform.system() == "Windows":
            create_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            create_kwargs["preexec_fn"] = os.setsid
        return await asyncio.create_subprocess_exec(*command, **create_kwargs)

    async def _drain_stream(self, stream: Any, label: str) -> None:
        try:
            while True:
                chunk = await stream.readline()
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace").rstrip()
                if text:
                    logger.debug("llama.cpp {}: {}", label, _redact_log_line(text))
        except Exception:
            logger.debug("llama.cpp {} stream drain stopped", label, exc_info=True)

    def _start_stream_drainers(self, process: asyncio.subprocess.Process) -> None:
        tasks: list[asyncio.Task[None]] = []
        if getattr(process, "stdout", None) is not None:
            tasks.append(asyncio.create_task(self._drain_stream(process.stdout, "stdout")))
        if getattr(process, "stderr", None) is not None:
            tasks.append(asyncio.create_task(self._drain_stream(process.stderr, "stderr")))
        for task in tasks:
            task.add_done_callback(self._discard_stream_drain_result)
        self._stream_drain_tasks = tasks

    @staticmethod
    def _discard_stream_drain_result(task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug("llama.cpp stream drain task failed", exc_info=True)

    def _stop_stream_drainers(self) -> None:
        for task in self._stream_drain_tasks:
            if not task.done():
                task.cancel()
        self._stream_drain_tasks = []

    async def _terminate_process(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        if platform.system() == "Windows":
            process.terminate()
        else:
            try:
                pgid = await asyncio.to_thread(os.getpgid, process.pid)
                await asyncio.to_thread(os.killpg, pgid, signal.SIGTERM)
            except ProcessLookupError:
                process.terminate()
            except Exception:
                process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except asyncio.TimeoutError:
            if platform.system() == "Windows":
                process.kill()
            else:
                try:
                    pgid = await asyncio.to_thread(os.getpgid, process.pid)
                    await asyncio.to_thread(os.killpg, pgid, signal.SIGKILL)
                except Exception:
                    process.kill()
            await process.wait()
        self._exit_code = process.returncode

    def _close_log_handle(self) -> None:
        if self._log_handle is not None:
            with contextlib.suppress(Exception):
                self._log_handle.close()
        self._log_handle = None
