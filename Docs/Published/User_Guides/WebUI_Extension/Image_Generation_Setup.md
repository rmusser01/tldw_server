# Image Generation Setup (stable-diffusion.cpp)

This guide explains how to enable image generation in tldw_server using the stable-diffusion.cpp adapter, plus supported remote backends (SwarmUI, OpenRouter, Novita, Together, Alibaba Model Studio). Outputs are **inline-only** (base64 in the response) and are **not stored** on the server.

## What You Get
- `POST /api/v1/files/create` with `file_type: "image"`.
- Inline PNG/JPG/WebP output in the response (no server-side files).
- Backend selection via config + `payload.backend`.
- Remote image backends with API-key auth and backend-specific timeouts.

## Prerequisites
- A working `sd-cli` binary from stable-diffusion.cpp.
- Model weights for the diffusion model, VAE, and (for Z-Image/FLUX.2) an LLM text encoder.
- Sufficient RAM/VRAM for your chosen model and resolution.

### Get the stable-diffusion.cpp CLI
Pick **one**:
- Download a prebuilt binary: https://github.com/leejet/stable-diffusion.cpp/releases
- Build from source: https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/build.md

Make sure you know the full path to `sd-cli` (or `sd-cli.exe` on Windows).

### Platform Notes (macOS / Windows / Linux)
- The adapter is OS-agnostic: it executes the configured binary for the host OS.
- Use an absolute path for `sd_cpp_binary_path` (Windows: `sd-cli.exe`, macOS/Linux: `sd-cli`).
- Ensure execute permissions on Unix (`chmod +x /path/to/sd-cli`).
- Any platform-specific GPU/runtime dependencies required by stable-diffusion.cpp must be installed on the host (CUDA/Vulkan/Metal, etc.).

## Download Model Weights

### Z-Image (Turbo)
From `docs/z_image.md`:
- Diffusion model:
  - GGUF: https://huggingface.co/leejet/Z-Image-Turbo-GGUF/tree/main
  - Safetensors: https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files/diffusion_models
- VAE:
  - https://huggingface.co/black-forest-labs/FLUX.1-schnell/tree/main
- LLM text encoder (Qwen3 4B):
  - GGUF: https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/tree/main
  - Safetensors: https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files/text_encoders

### FLUX.2
From `docs/flux2.md`:

#### FLUX.2-dev
- Diffusion model (GGUF): https://huggingface.co/city96/FLUX.2-dev-gguf/tree/main
- VAE (safetensors): https://huggingface.co/black-forest-labs/FLUX.2-dev/tree/main
- LLM text encoder (Mistral-Small 3.2 24B GGUF): https://huggingface.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF/tree/main

#### FLUX.2 klein 4B / 9B
- Diffusion model:
  - 4B GGUF: https://huggingface.co/leejet/FLUX.2-klein-4B-GGUF/tree/main
  - 9B GGUF: https://huggingface.co/leejet/FLUX.2-klein-9B-GGUF/tree/main
- VAE (safetensors): https://huggingface.co/black-forest-labs/FLUX.2-dev/tree/main
- LLM text encoder:
  - 4B: https://huggingface.co/unsloth/Qwen3-4B-GGUF/tree/main
  - 9B: https://huggingface.co/unsloth/Qwen3-8B-GGUF/tree/main

## Configure tldw_server
Edit `tldw_Server_API/Config_Files/config.txt` and add/update:

```ini
[Image-Generation]
# Required
enabled_backends = ["stable_diffusion_cpp"]
# Optional (used when payload.backend is omitted)
default_backend = stable_diffusion_cpp

# Limits (tune to your hardware)
max_width = 1024
max_height = 1024
max_pixels = 1048576
max_steps = 50
max_prompt_length = 1000
inline_max_bytes = 4000000

# stable-diffusion.cpp paths
sd_cpp_binary_path = /abs/path/to/sd-cli
sd_cpp_diffusion_model_path = /abs/path/to/your_diffusion_model.gguf
sd_cpp_llm_path = /abs/path/to/your_llm_text_encoder.gguf
sd_cpp_vae_path = /abs/path/to/your_vae.safetensors

# Optional
sd_cpp_model_path =
sd_cpp_lora_paths = []
sd_cpp_allowed_extra_params = []
sd_cpp_default_steps = 25
sd_cpp_default_cfg_scale = 7.5
sd_cpp_default_sampler = euler_a
sd_cpp_device = auto
sd_cpp_timeout_seconds = 120

# SwarmUI defaults
swarmui_base_url = http://127.0.0.1:7801
swarmui_default_model =
swarmui_swarm_token =
swarmui_allowed_extra_params = []
swarmui_timeout_seconds = 120

# OpenRouter image defaults
openrouter_image_base_url = https://openrouter.ai/api/v1
openrouter_image_api_key =
openrouter_image_default_model = openai/gpt-image-1
openrouter_image_allowed_extra_params = []
openrouter_image_timeout_seconds = 120

# Novita image defaults
novita_image_base_url = https://api.novita.ai
novita_image_api_key =
novita_image_default_model = sd_xl_base_1.0.safetensors
novita_image_allowed_extra_params = []
novita_image_timeout_seconds = 180
novita_image_poll_interval_seconds = 2

# Together image defaults
together_image_base_url = https://api.together.xyz/v1
together_image_api_key =
together_image_default_model = black-forest-labs/FLUX.1-schnell-Free
together_image_allowed_extra_params = []
together_image_timeout_seconds = 120

# Alibaba Model Studio image defaults
# Optional explicit endpoint override (leave blank to use region preset)
modelstudio_image_base_url =
modelstudio_image_api_key =
modelstudio_image_default_model = qwen-image
# Region presets for base URL routing: sg | cn | us
modelstudio_image_region = sg
# sync | async | auto (auto prefers sync endpoint, then falls back)
modelstudio_image_mode = auto
modelstudio_image_poll_interval_seconds = 2
modelstudio_image_timeout_seconds = 180
modelstudio_image_allowed_extra_params = []
```

Notes:
- **For Z-Image and FLUX.2**, set `sd_cpp_diffusion_model_path` + `sd_cpp_llm_path` + `sd_cpp_vae_path`.
- **For full SD models** (single combined model), set `sd_cpp_model_path` instead of `sd_cpp_diffusion_model_path`.
- `inline_max_bytes` controls the max inline payload size; larger images will fail with `export_size_exceeded`.
- `sd_cpp_allowed_extra_params` controls which `extra_params` keys are accepted (default: deny all).
- For Model Studio, set `modelstudio_image_region` to `sg`, `cn`, or `us` and provide `modelstudio_image_api_key` (or env `DASHSCOPE_API_KEY` / `QWEN_API_KEY`). Leave `modelstudio_image_base_url` blank unless you need a custom endpoint override.
- For Model Studio mode control, set `modelstudio_image_mode` globally or pass `payload.extra_params.mode` (`sync` or `async`) per request.

## Security Considerations (Binary Execution)
- **Treat the CLI as privileged**: `sd_cpp_binary_path` is executed directly. Only use trusted binaries from known sources.
- **Lock down config**: `tldw_Server_API/Config_Files/config.txt` should be writable only by administrators.
- **Least privilege**: run the server under a dedicated, unprivileged user; keep model/binary directories non-writable by untrusted users.
- **Restrict extra flags**: `payload.extra_params` is passed through as CLI flags, but only keys in `sd_cpp_allowed_extra_params` are accepted. Keep the allowlist minimal, and avoid exposing this to untrusted clients.
- **Containerize if needed**: for stronger isolation, run tldw_server inside a container or VM with minimal filesystem access.

## Test the API

```bash
curl -sS http://127.0.0.1:8000/api/v1/files/create \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: <your-key>" \
  -d '{
    "file_type": "image",
    "payload": {
      "backend": "stable_diffusion_cpp",
      "prompt": "A neon-lit city street in the rain",
      "prompt_refinement": "auto",
      "negative_prompt": "blurry, low quality",
      "width": 768,
      "height": 512,
      "steps": 25,
      "cfg_scale": 7.5,
      "sampler": "euler_a"
    },
    "export": {"format": "webp", "mode": "inline", "async_mode": "sync"},
    "options": {"persist": true}
  }'
```

The response includes:
- `artifact.export.content_b64` (base64 bytes)
- `artifact.export.content_type` (image MIME type)

Prompt refinement:
- `payload.prompt_refinement` accepts `true|false` or `basic|auto|off`.
- `auto` (default) keeps detailed prompts unchanged and enriches sparse prompts with a quality suffix.
- `off` keeps the prompt unchanged (after whitespace normalization).

### Model Studio example (`backend=modelstudio`)

```bash
curl -sS http://127.0.0.1:8000/api/v1/files/create \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: <your-key>" \
  -d '{
    "file_type": "image",
    "payload": {
      "backend": "modelstudio",
      "prompt": "A watercolor skyline at sunrise",
      "width": 1024,
      "height": 1024,
      "extra_params": {"mode": "async"}
    },
    "export": {"format": "png", "mode": "inline", "async_mode": "sync"}
  }'
```

## Parameter Mapping
- `sd_cpp_binary_path` -> `sd-cli` binary path.
- `sd_cpp_diffusion_model_path` -> `--diffusion-model`.
- `sd_cpp_model_path` -> `--model` (for full models).
- `sd_cpp_llm_path` -> `--llm`.
- `sd_cpp_vae_path` -> `--vae`.
- `sd_cpp_lora_paths` -> repeated `--lora-model-dir`.
- `sampler` -> `--sampling-method`.

Extra flags:
- `payload.extra_params` accepts backend flags as key/value pairs and maps them to `--<flag>`.
- Only keys in `sd_cpp_allowed_extra_params` are accepted (default: deny all).
- `cli_args` (list) passes raw CLI tokens; only use if you explicitly allowlist `cli_args`.
- Example: `"extra_params": {"offload-to-cpu": true, "diffusion-fa": true}`
- Example allowlist: `sd_cpp_allowed_extra_params = ["offload-to-cpu", "diffusion-fa"]`

### Allowlist Presets (Examples)
Always confirm flags against your `sd-cli --help` output for the version you run. Start with an empty allowlist and expand only as needed.

Minimal (recommended default):
```ini
sd_cpp_allowed_extra_params = []
```

Performance/memory toggles (example, verify for your build):
```ini
sd_cpp_allowed_extra_params = ["offload-to-cpu", "diffusion-fa"]
```

Trusted internal use only (still avoid path/output flags):
```ini
sd_cpp_allowed_extra_params = ["offload-to-cpu", "diffusion-fa", "clip-on-cpu", "vae-on-cpu"]
```

Avoid allowlisting flags that accept paths or control outputs (let the server own these), for example:
`diffusion-model`, `model`, `llm`, `vae`, `lora-model-dir`, `output`, `out`.

## Troubleshooting
- `image_backend_unavailable`: ensure `enabled_backends` includes `stable_diffusion_cpp` and `sd_cpp_binary_path` exists.
- `image_backend_unavailable` (Model Studio): ensure `enabled_backends` includes `modelstudio` and a key is configured (`modelstudio_image_api_key`, `DASHSCOPE_API_KEY`, or `QWEN_API_KEY`).
- `image_generation_failed`: check model/vae/llm paths and CLI compatibility.
- `image_generation_failed` (Model Studio): verify `modelstudio_image_region`, endpoint availability, and whether the selected model supports sync vs async mode.
- `export_size_exceeded`: reduce resolution or raise `inline_max_bytes`.
- CLI flag mismatch: stable-diffusion.cpp changes frequently; verify the CLI doc and adjust `extra_params` as needed.

## References
- CLI docs: https://github.com/leejet/stable-diffusion.cpp/blob/master/examples/cli/README.md
- Z-Image: https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/z_image.md
- FLUX.2: https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/flux2.md
