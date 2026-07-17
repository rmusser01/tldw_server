# Providers API Documentation

## Overview
- Purpose: Discover configured LLM providers, their models and capabilities, and the health of the LLM inference subsystem. These endpoints are useful for client UIs and for operational monitoring.
- OpenAPI tag: `llm`
 - Base prefix: `/api/v1`

## Auth + Rate Limits
- These endpoints follow the server’s standard AuthNZ settings and typically require authentication.
- Headers:
  - Single-user: `X-API-KEY: <key>`
  - Multi-user: `Authorization: Bearer <JWT>`
- Standard limits apply; these endpoints are lightweight and primarily used by UIs/ops. Deployments can choose to relax access via gateway configuration, but the default app configuration applies global security to API routes.

## Endpoints

### 1) GET /api/v1/llm/health
- Summary: Health status for the LLM inference subsystem (provider manager, request queue, rate limiter).
- Response (example):
```json
{
  "service": "llm_inference",
  "status": "healthy",
  "timestamp": "2025-01-01T00:00:00.000000",
  "components": {
    "providers": {
      "initialized": true,
      "count": 3,
      "report": {
        "openai": {
          "status": "healthy",
          "success_count": 42,
          "failure_count": 1,
          "consecutive_failures": 0,
          "average_response_time": 0.35,
          "circuit_breaker_state": "CLOSED",
          "last_success": 1735689600.0,
          "last_failure": null
        }
      }
    },
    "queue": {
      "initialized": true,
      "queue_size": 0,
      "processing_count": 0,
      "max_queue_size": 100,
      "max_concurrent": 10,
      "total_processed": 0,
      "total_rejected": 0,
      "is_running": true
    },
    "rate_limiter": {
      "initialized": true,
      "limits": {
        "global_rpm": 60,
        "per_user_rpm": 20,
        "per_conversation_rpm": 10,
        "per_user_tokens_per_minute": 10000
      }
    }
  }
}
```

### 2) GET /api/v1/llm/providers
- Summary: List configured LLM providers and their models.
- Query params: `include_deprecated` (bool, default false)
- llama.cpp note:
  - The `Llama.cpp` provider entry may include a first-class `llama_cpp_controls` block for UI-facing capability discovery.
  - This block is separate from `extra_body_compat`. `extra_body_compat` documents raw passthrough keys; `llama_cpp_controls` documents the stable app-level contract for llama.cpp grammar/thinking controls.
- Response (example):
```json
{
  "providers": [
    {
      "name": "openai",
      "display_name": "OpenAI",
      "type": "commercial",
      "is_configured": true,
      "models": ["gpt-4o"],
      "models_info": [
        {
          "name": "gpt-4o",
          "context_window": 128000,
          "max_output_tokens": 4096,
          "capabilities": {"vision": true, "tool_use": true, "streaming": true},
          "modalities": {"input": ["text", "image"], "output": ["text"]}
        }
      ],
      "default_model": "gpt-4o",
      "supports_streaming": true,
      "requires_api_key": true,
      "capabilities": {"supports_streaming": true, "supports_tools": true, "default_timeout_seconds": 60},
      "health": {
        "status": "healthy",
        "success_count": 42,
        "failure_count": 1,
        "consecutive_failures": 0,
        "average_response_time": 0.35,
        "circuit_breaker_state": "CLOSED",
        "last_success": 1735689600.0,
        "last_failure": null
      }
    }
  ],
  "diagnostics_ui": {
    "queue_status_auto": {"min": 3, "max": 30},
    "queue_activity_auto": {"min": 3, "max": 30}
  },
  "default_provider": "openai",
  "total_configured": 1
}
```

llama.cpp capability example:
```json
{
  "name": "llama",
  "display_name": "Llama.cpp",
  "llama_cpp_controls": {
    "grammar": {
      "supported": true,
      "effective_reason": "supported in current deployment",
      "source": "first_class+extra_body"
    },
    "thinking_budget": {
      "supported": false,
      "request_key": null,
      "effective_reason": "no configured thinking-budget mapping for this deployment"
    },
    "reserved_extra_body_keys": ["grammar"]
  }
}
```

`llama_cpp_controls` semantics:
- `grammar.supported=false` means llama.cpp advanced controls are disabled for the current runtime, typically because `strict_openai_compat` is effective.
- `thinking_budget.supported=true` only when the deployment explicitly maps the app-level field to an upstream llama.cpp request key.
- `thinking_budget.request_key` is populated from:
  - environment: `LLAMA_CPP_THINKING_BUDGET_PARAM`
  - config: `Local-API.llama_cpp_thinking_budget_param`
- `reserved_extra_body_keys` is the list UIs should treat as first-class/reserved when editing raw `extra_body`.

### 3) GET /api/v1/llm/providers/{provider_name}
- Summary: Details for a single provider.
- Path params: `provider_name` (e.g., `openai`, `anthropic`)
- Query params: `include_deprecated` (bool, default false)
- Response: Same shape as a single item in `providers` above. `404` if provider not configured.

### 4) GET /api/v1/llm/models
- Summary: Flat list of available models across all providers (chat, embeddings, image).
- Query params:
  - `include_deprecated` (bool, default false)
  - `type` (repeatable; `chat`, `embedding`, `image`)
  - `input_modality` (repeatable; e.g., `text`, `image`)
  - `output_modality` (repeatable; e.g., `text`, `image`, `embedding`)
- Response (example):
```json
[
  "openai/gpt-4o",
  "anthropic/claude-opus-4.1",
  "image/stable_diffusion_cpp"
]
```
Tip: For chat-only models, use `/api/v1/llm/models?type=chat`.

### 5) GET /api/v1/llm/models/metadata
- Summary: Flattened model capability metadata across providers.
- Query params:
  - `include_deprecated` (bool, default false)
  - `type` (repeatable; `chat`, `embedding`, `image`)
  - `input_modality` (repeatable)
  - `output_modality` (repeatable)
- Response (example):
```json
{
  "models": [
    {
      "provider": "openai",
      "name": "gpt-4o",
      "context_window": 128000,
      "max_output_tokens": 4096,
      "capabilities": {
        "vision": true,
        "audio_input": false,
        "audio_output": false,
        "tool_use": true,
        "json_mode": true,
        "function_calling": true,
        "streaming": true,
        "thinking": false
      },
      "modalities": {"input": ["text", "image"], "output": ["text"]},
      "notes": "Vision multimodal; tool use supported."
    },
    {
      "provider": "image",
      "id": "image/stable_diffusion_cpp",
      "name": "stable_diffusion_cpp",
      "type": "image",
      "capabilities": {
        "image_generation": true,
        "image_reference_input": false
      },
      "modalities": {"input": ["text"], "output": ["image"]},
      "supported_formats": ["png", "jpg", "webp"],
      "is_configured": true
    },
    {
      "provider": "image",
      "id": "image/modelstudio",
      "name": "modelstudio",
      "type": "image",
      "capabilities": {
        "image_generation": true,
        "image_reference_input": true
      },
      "modalities": {"input": ["text", "image"], "output": ["image"]},
      "supported_formats": ["png", "jpg", "webp"],
      "is_configured": true
    }
  ],
  "total": 2
}
```

Image capability notes:
- `capabilities.image_reference_input` indicates whether the backend or resolved backend/model combination accepts a managed image reference input for `POST /api/v1/files/create`.
- Clients should treat missing or false `image_reference_input` as unsupported and avoid assuming prompt-only fallback.
- Reference-guided requests use `payload.reference_file_id` and are expected to source eligible files from `GET /api/v1/files/reference-images`.

## Notes
- The presence and values for models depend on your `tldw_Server_API/Config_Files/config.txt` and environment variables. Some providers may be listed as not configured if API keys or endpoints are missing.
- By default, deprecated models are filtered from responses; set `include_deprecated=true` to include them.
- Health information reflects in-process runtime state, including circuit-breaker status and queue statistics.
- Provider objects may include additional fields when present in config, such as `endpoint` (for local providers), `default_temperature`, `max_tokens`, `supports_streaming`, `requires_api_key`, `capabilities`, and a `health` sub-object when the provider manager is initialized.
- `models_info[]` entries may also repeat provider-level llama.cpp control metadata so clients that work from model selections can gate controls without hard-coding provider-specific rules.
- Note: `total_configured` reflects the number of providers returned in the response; individual providers indicate configuration with the `is_configured` flag.
- The response from `/llm/providers` may also include `diagnostics_ui` with UI auto-refresh intervals for queue status/activity, derived from the server configuration.
- Recognized providers include (commercial): OpenAI, AWS Bedrock, Anthropic, Google, Mistral, Cohere, Groq, HuggingFace, OpenRouter, DeepSeek, Qwen, Moonshot, Z.AI; and (local/OpenAI-compatible): Llama.cpp, Kobold.cpp, Oobabooga, TabbyAPI, vLLM, Local LLM, Ollama, Aphrodite, Custom OpenAI API (1/2).
- When no models are configured for Anthropic, safe defaults may be injected: `claude-opus-4.1`, `claude-sonnet-4`.
