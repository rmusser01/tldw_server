# Local_LLM

Local_LLM manages local inference backends such as llama.cpp, llamafile, Ollama,
and Hugging Face runtimes. It wraps process lifecycle, model inventory,
configuration, profiles, acquisition jobs, and OpenAI-like inference routing for
offline or self-hosted model deployments.

## Start Here

- Unified manager and handlers: `LLM_Inference_Manager.py`,
  `LLM_Base_Handler.py`, `LlamaCpp_Handler.py`, `Llamafile_Handler.py`,
  `Ollama_Handler.py`, and `Huggingface_Handler.py`.
- llama.cpp control plane: `llamacpp_*` modules.
- API endpoint and schemas: `app/api/v1/endpoints/llamacpp.py`,
  `app/api/v1/schemas/llamacpp_schemas.py`, and
  `app/api/v1/schemas/llamacpp_admin_schemas.py`.
- Tests: `tests/LLM_Local/`.

## Responsibilities

- Route local inference requests to the selected backend handler.
- Start, stop, and inspect managed llama.cpp/llamafile processes.
- Track llama.cpp profiles, hardware capabilities, inventory, runtime state, and
  configuration locks.
- Acquire local model assets through Jobs-backed acquisition services.
- Normalize local backend failures into typed inference exceptions.

## Module Map

- `LLM_Inference_Manager.py` owns handler registration and request delegation.
- `LLM_Inference_Schemas.py` and `LLM_Inference_Exceptions.py` define shared
  request/result/error shapes.
- `handler_utils.py` and `http_utils.py` hold backend utility code.
- `llamacpp_config_service.py`, `llamacpp_profile_store.py`,
  `llamacpp_inventory_service.py`, `llamacpp_process_runner.py`, and
  `llamacpp_supervisor_service.py` implement the llama.cpp control plane.
- `llamacpp_acquisition_service.py` and `llamacpp_acquisition_jobs.py` enqueue
  and process model acquisition work.

## How It Connects

- The `/llamacpp` endpoint exposes management, profile, inventory, acquisition,
  and inference routes.
- Jobs tracks acquisition and long-running local model operations.
- LLM provider registries can route local model calls through configured local
  OpenAI-compatible endpoints.

## Extension Points

- Add a new backend by implementing a handler with the base-handler contract and
  registering it in the inference manager.
- Add llama.cpp admin settings through runtime models, config service validation,
  endpoint schemas, and tests together.
- Keep process execution in `llamacpp_process_runner.py` or the supervisor so
  lifecycle cleanup remains centralized.

## Testing

- Management and lifecycle APIs: `tests/LLM_Local/test_llamacpp_management_api.py`,
  `tests/LLM_Local/test_llamacpp_lifecycle_api_contract.py`, and
  `tests/LLM_Local/test_llamacpp_runtime_api.py`.
- Config/profile/inventory: `tests/LLM_Local/test_llamacpp_admin_config_api.py`,
  `tests/LLM_Local/test_llamacpp_profile_store.py`, and
  `tests/LLM_Local/test_llamacpp_inventory_api.py`.
- Acquisition: `tests/LLM_Local/test_llamacpp_acquisition_service.py`,
  `tests/LLM_Local/test_llamacpp_acquisition_api.py`, and
  `tests/LLM_Local/test_llamacpp_acquisition_jobs_worker.py`.

## Gotchas

- Local process management is OS- and port-sensitive. Tests should use fakes or
  temporary profiles unless they explicitly exercise real backends.
- Never trust model paths from requests without using the config/profile
  validation helpers.
