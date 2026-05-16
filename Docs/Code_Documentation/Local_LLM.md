Local LLM Module

Overview
- Manages local inference backends: llama.cpp and llamafile.
- Standardizes HTTP via httpx, with retry/backoff helpers and readiness polling.
- Adds safety: strict allowlists, optional passthrough, denylist for secrets, and path restrictions.

Key Paths
- Handlers: tldw_Server_API/app/core/Local_LLM/{LlamaCpp_Handler.py,Llamafile_Handler.py}
- Schemas: tldw_Server_API/app/core/Local_LLM/LLM_Inference_Schemas.py
- HTTP utils: tldw_Server_API/app/core/Local_LLM/http_utils.py
- SSE helpers: tldw_Server_API/app/core/LLM_Calls/sse.py

Config (LlamaCppConfig/LlamafileConfig)
- allow_unvalidated_args: bool (default False)
  - When True, unknown server_args keys pass through as --key value.
  - Still subject to denylist and path safety.
- allow_cli_secrets: bool (default False)
  - When False, secret-like args (e.g., hf_token) are rejected. Use env vars.
  - Hugging Face token: set HF_TOKEN in environment.
- port_autoselect: bool (default True)
  - If the requested port is busy, probe a small range for a free port.
- port_probe_max: int (default 10)
  - Number of ports to probe beyond the starting port.
- allowed_paths: list[Path] (optional)
  - Additional base directories permitted for file flags (grammar_file, lora, caches, logs, etc.).

Flag Mapping (llama.cpp)
- Core: port, host, threads/t (-t), threads_batch/tb (--threads-batch), ctx_size/c (-c), n_gpu_layers/ngl/gpu_layers (-ngl)
- Batching: batch_size/b (-b), ubatch_size/ub (--ubatch-size)
- GPU & memory: main_gpu/mg (--main-gpu), split_mode/sm (--split-mode), tensor_split (--tensor-split)
- Rope & scaling: rope_freq_base, rope_freq_scale/rope_scale (--rope-freq-*)
- Rope scaling type: rope_scaling/rope_scaling_type (--rope-scaling)
- KV & offload: main_kv (--main-kv), no_kv_offload (--no-kv-offload), cache_type_k/cache_type_v
- Features: flash_attn (--flash-attn), cont_batching (--cont-batching)
- Adapters: lora (repeatable), lora_scaled, lora_base, control_vector
- HF model download: hf_repo, hf_file, hf_token (DENY by default; prefer HF_TOKEN env), offline
- Chat: conversation/cnv, no_conversation/no_cnv, interactive/i, interactive_first/if, single_turn/st, jinja,
        chat_template, chat_template_file
- I/O & prompts: in_prefix, in_suffix, in_prefix_bos, reverse_prompt/r, system_prompt/sys, prompt/-p
- Generation: predict/n (-n), keep, ignore_eos, no_context_shift
- Sampling: temp, seed/-s, dynatemp_range, top_k, top_p, min_p, typical
- Penalties: repeat_penalty, repeat_last_n, presence_penalty, frequency_penalty
- DRY/Mirostat: dry_multiplier, dry_base, dry_allowed_length, mirostat, mirostat_lr, mirostat_ent
- CPU/NUMA: cpu_mask, cpu_range, numa
- Structured: grammar, grammar_file, json_schema, json_schema_file, j
- Caching/logging: prompt_cache, prompt_cache_all, prompt_cache_ro, log_file, log_colors, log_timestamps, log_verbosity, no_perf

Security Guidance
- Use env vars for secrets. Example: export HF_TOKEN=...
- Denylist blocks secret flags unless allow_cli_secrets=True. Prefer env over CLI to avoid leaking via ps/args.
- Path safety: file flags must resolve under models_dir or allowed_paths. Absolute/traversal paths are rejected by default.

Windows Notes
- Handlers create processes with CREATE_NEW_PROCESS_GROUP.
- stop_server prefers CTRL_BREAK_EVENT plus terminate/kill fallback.
- On POSIX, processes start in new session via setsid and are terminated via process group signals (SIGTERM/SIGKILL).

Readiness & Observability
- Handlers poll base_url for readiness (configurable window) before returning.
- Metrics (no-op friendly): starts, stops, start_errors, readiness_time_sum/readiness_count, inference_count, inference_error_count, inference_time_sum (llama.cpp).
- Query via handler.get_metrics() or observe logs.
 - HTTP endpoints:
   - GET /api/v1/llamacpp/status
   - GET /api/v1/llamacpp/metrics
   - GET /api/v1/llamafile/metrics

Streaming Parity (llama.cpp)
- stream_inference(prompt|messages, stream=True) returns SSE lines (data: ...), normalizing to OpenAI chunk format and [DONE].
- Uses httpx AsyncClient with proper backpressure via aiter_lines().

Examples
- Start llama.cpp server with safe flags:
  handler.start_server("q4.gguf", server_args={"port": 8081, "ngl": 30, "ctx_size": 4096, "threads": 8, "flash_attn": True})

- Allow passthrough in trusted environments (not recommended):
  cfg = LlamaCppConfig(..., allow_unvalidated_args=True, allow_cli_secrets=False)

- Constrain file flags to models dir:
  cfg = LlamaCppConfig(..., allowed_paths=[Path("/opt/local_llm/templates")])

- Streaming inference:
  async for line in handler.stream_inference(prompt="Hello"):
      print(line.strip())

Managed vLLM Instances
- `Local_LLM` still covers the in-process/self-hosted `llama.cpp` and `llamafile` handlers.
- Managed `vLLM` now uses a separate control plane under `tldw_Server_API/app/core/VLLM_Management/`.
- Admin APIs live under `/api/v1/llm/providers/vllm/instances` with Jobs-backed `start`, `stop`, `restart`, and `probe` actions.
- Request-scoped routing uses `provider_instance_id` instead of mutating global `vllm_api` config. Chat requests target `api_provider="vllm"`; embeddings requests may set `provider="vllm"` plus `provider_instance_id`.
- The managed `vLLM` path supports multiple local or SSH-backed instances, but `tldw_server` must be able to reach each resolved `base_url` directly.
- See `Docs/User_Guides/Integrations_Experiments/Managed_vLLM.md` for the lifecycle, SSH launcher contract, and routing examples.
