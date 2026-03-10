"""Structured command construction for managed vLLM instances."""

from __future__ import annotations

from typing import Any

_STRUCTURED_FLAG_MAP: dict[str, str] = {
    "host": "--host",
    "port": "--port",
    "served_model_name": "--served-model-name",
    "tensor_parallel_size": "--tensor-parallel-size",
    "gpu_memory_utilization": "--gpu-memory-utilization",
    "max_model_len": "--max-model-len",
    "dtype": "--dtype",
    "quantization": "--quantization",
    "api_key": "--api-key",
    "chat_template": "--chat-template",
    "limit_mm_per_prompt": "--limit-mm-per-prompt",
}
_BOOLEAN_FLAG_MAP: dict[str, str] = {
    "trust_remote_code": "--trust-remote-code",
}
_SHELL_METACHARACTERS = (";", "&", "|", "`", "$", ">", "<", "\n", "\r")


def _ensure_safe_token(token: Any) -> str:
    value = str(token).strip()
    if not value:
        raise ValueError("vLLM extra_args entries cannot be empty")
    if any(marker in value for marker in _SHELL_METACHARACTERS):
        raise ValueError(f"vLLM extra_args token '{value}' is not allowed")
    return value


def _append_structured_flag(argv: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    argv.extend([flag, str(value)])


def _append_boolean_flag(argv: list[str], flag: str, value: Any) -> None:
    if bool(value):
        argv.append(flag)


def _skip_flags_for_launch_spec(launch_spec: dict[str, Any]) -> set[str]:
    flags = {
        flag
        for key, flag in _STRUCTURED_FLAG_MAP.items()
        if launch_spec.get(key) is not None
    }
    flags.update(
        flag
        for key, flag in _BOOLEAN_FLAG_MAP.items()
        if bool(launch_spec.get(key))
    )
    return flags


def _append_extra_args(argv: list[str], extra_args: Any, *, skip_flags: set[str]) -> None:
    if not extra_args:
        return
    if not isinstance(extra_args, list):
        raise ValueError("vLLM extra_args must be a list of argv tokens")

    index = 0
    while index < len(extra_args):
        token = _ensure_safe_token(extra_args[index])
        if not token.startswith("--"):
            raise ValueError("vLLM extra_args values must follow a flag")
        if token in skip_flags:
            index += 1
            if index < len(extra_args) and not str(extra_args[index]).strip().startswith("--"):
                index += 1
            continue
        argv.append(token)
        index += 1
        if index < len(extra_args):
            next_token = _ensure_safe_token(extra_args[index])
            if not next_token.startswith("--"):
                argv.append(next_token)
                index += 1


def build_vllm_serve_argv(launch_spec: dict[str, Any]) -> list[str]:
    """Build a safe ``vllm serve`` argv from structured instance settings."""

    if not isinstance(launch_spec, dict):
        raise ValueError("vLLM launch_spec must be a dictionary")
    model = str(launch_spec.get("model") or "").strip()
    if not model:
        raise ValueError("vLLM launch_spec.model is required")

    argv = ["vllm", "serve", model]
    for key, flag in _STRUCTURED_FLAG_MAP.items():
        _append_structured_flag(argv, flag, launch_spec.get(key))
    for key, flag in _BOOLEAN_FLAG_MAP.items():
        _append_boolean_flag(argv, flag, launch_spec.get(key))
    _append_extra_args(argv, launch_spec.get("extra_args"), skip_flags=_skip_flags_for_launch_spec(launch_spec))
    return argv
