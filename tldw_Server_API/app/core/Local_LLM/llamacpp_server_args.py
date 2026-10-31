"""Shared llama.cpp server argument formatting helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any

from pydantic import AfterValidator

CORE_SERVER_ARG_KEYS: set[str] = {
    "threads",
    "t",
    "ctx_size",
    "c",
    "n_ctx",
    "n_gpu_layers",
    "ngl",
    "gpu_layers",
}

RESERVED_STRUCTURED_ARG_KEYS: set[str] = {"model", "model_path", "m", "host", "port"}
SNAPSHOT_OWNED_ARG_KEYS: set[str] = {"slot_save_path", "slots", "no_slots"}

PATH_ARG_KEYS: set[str] = {
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


def flatten_repeatable(flag: str, value: Any) -> list[str]:
    values = value if isinstance(value, (list, tuple)) else [value]
    command: list[str] = []
    for item in values:
        command.extend([flag, str(item)])
    return command


def validate_cache_options(server_args: dict[str, Any]) -> dict[str, Any]:
    """Reject ambiguous full-cache settings before cleaning or launching."""
    if "swa_full" in server_args and type(server_args["swa_full"]) is not bool:
        raise ValueError("swa_full must be a boolean")
    return server_args


ServerArgs = Annotated[dict[str, object], AfterValidator(validate_cache_options)]


def clean_server_args(server_args: dict[str, Any] | None) -> dict[str, Any]:
    validate_cache_options(server_args or {})
    return {key: value for key, value in (server_args or {}).items() if value is not None and value != ""}


def server_arg_formatters() -> dict[str, Callable[[Any], list[str]]]:
    """Return supported llama-server arg formatters shared by legacy and supervisor paths."""

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
        "tensor_split": lambda v: (
            ["--tensor-split", ",".join(map(str, v))] if isinstance(v, (list, tuple)) else ["--tensor-split", str(v)]
        ),
        "rope_freq_base": lambda v: ["--rope-freq-base", str(float(v))],
        "rope_freq_scale": lambda v: ["--rope-freq-scale", str(float(v))],
        "rope_scale": lambda v: ["--rope-freq-scale", str(float(v))],
        "flash_attn": lambda v: ["--flash-attn", str(v)] if isinstance(v, str) else (["--flash-attn"] if v else []),
        "cont_batching": lambda v: ["--cont-batching"] if v else ["--no-cont-batching"],
        "no_cont_batching": lambda v: ["--no-cont-batching"] if v else [],
        "context_shift": lambda v: ["--context-shift"] if v else ["--no-context-shift"],
        "streaming_llm": lambda v: ["--context-shift"] if v else [],
        "lora": lambda v: flatten_repeatable("--lora", v),
        "lora_scaled": lambda v: (
            ["--lora-scaled", str(v[0]), str(v[1])]
            if isinstance(v, (list, tuple)) and len(v) == 2
            else (["--lora-scaled", str(v)] if v is not None else [])
        ),
        "lora_base": lambda v: ["--lora-base", str(v)],
        "control_vector": lambda v: ["--control-vector", str(v)],
        "cache_type_k": lambda v: ["--cache-type-k", str(v)],
        "swa_full": lambda v: ["--swa-full"] if validate_cache_options({"swa_full": v})["swa_full"] else [],
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
