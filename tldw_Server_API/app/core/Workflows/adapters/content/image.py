"""Image generation and description adapters.

This module includes adapters for image operations:
- image_gen: Generate images
- image_describe: Describe images
"""

from __future__ import annotations

import base64
import time
import uuid
from typing import Any

from loguru import logger

try:
    from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async
except ImportError:
    async def perform_chat_api_call_async(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise ImportError("chat_service_unavailable")

from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.core.Workflows.adapters._common import (
    extract_openai_content,
    resolve_artifacts_dir,
    resolve_workflow_file_path,
)
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.content._config import (
    ImageDescribeConfig,
    ImageGenConfig,
)

# Attempt to import DatabasePaths; if unavailable, fall back to a sensible default
try:
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
except ImportError:
    DatabasePaths = None

# Attempt to import resolve_context_user_id for user ID resolution
try:
    from tldw_Server_API.app.core.Workflows.adapters._common import resolve_context_user_id
except ImportError:
    def resolve_context_user_id(context: dict[str, Any]) -> str | None:
        return context.get("user_id")


@registry.register(
    "image_gen",
    category="content",
    description="Generate images",
    parallelizable=True,
    tags=["content", "image"],
    config_model=ImageGenConfig,
)
async def run_image_gen_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Generate images from text prompts using configured image generation backends.

    Config:
      - prompt: str (templated) - text prompt for image generation
      - negative_prompt: Optional[str] (templated) - negative prompt
      - backend: Literal["stable_diffusion_cpp", "swarmui"] = "stable_diffusion_cpp"
      - width: int = 512
      - height: int = 512
      - steps: int = 20
      - cfg_scale: float = 7.0
      - seed: Optional[int] = None
      - sampler: Optional[str] = None
      - model: Optional[str] = None
      - format: str = "png"
      - save_artifact: bool = True
    Output:
      - {"images": [{"uri": str, "width": int, "height": int, "format": str}], "count": int, "timings": {...}}
    """
    # Cancellation check
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    user_id = resolve_context_user_id(context)
    if not user_id:
        try:
            if DatabasePaths is not None:
                user_id = str(DatabasePaths.get_single_user_id())
        except Exception:
            return {"error": "missing_user_id"}
    user_id = str(user_id)

    # Template rendering for prompt
    prompt_t = str(config.get("prompt") or "").strip()
    if prompt_t:
        prompt = apply_template_to_string(prompt_t, context) or prompt_t
    else:
        # Try to get from last.text
        prompt = None
        try:
            last = context.get("prev") or context.get("last") or {}
            if isinstance(last, dict):
                prompt = str(last.get("text") or last.get("prompt") or "")
        except Exception as prompt_context_error:
            logger.debug("Image adapter failed to read prompt from context fallback", exc_info=prompt_context_error)
    prompt = prompt or ""

    if not prompt.strip():
        return {"error": "missing_prompt"}

    from tldw_Server_API.app.core.Image_Generation.config import get_image_generation_config
    from tldw_Server_API.app.core.Image_Generation.prompt_refinement import (
        normalize_prompt_refinement_mode,
        refine_image_prompt,
    )

    image_generation_cfg = get_image_generation_config()
    prompt_refinement_mode = normalize_prompt_refinement_mode(config.get("prompt_refinement"))
    prompt = refine_image_prompt(
        prompt,
        mode=prompt_refinement_mode,
        max_length=image_generation_cfg.max_prompt_length,
    )

    # Negative prompt
    neg_prompt_t = config.get("negative_prompt")
    negative_prompt = None
    if neg_prompt_t:
        negative_prompt = apply_template_to_string(str(neg_prompt_t), context) or str(neg_prompt_t)

    # Parameters
    backend = str(config.get("backend") or "stable_diffusion_cpp").strip().lower()
    width = int(config.get("width") or 512)
    height = int(config.get("height") or 512)
    steps = int(config.get("steps") or 20)
    cfg_scale = float(config.get("cfg_scale") or 7.0)
    seed = config.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except Exception:
            seed = None
    sampler = config.get("sampler")
    model = config.get("model")
    img_format = str(config.get("format") or "png").strip().lower()
    if img_format not in ("png", "jpg", "jpeg", "webp"):
        img_format = "png"
    save_artifact = config.get("save_artifact")
    save_artifact = True if save_artifact is None else bool(save_artifact)

    # Test mode simulation
    if is_test_mode():
        fake_id = str(uuid.uuid4())[:8]
        step_run_id = str(context.get("step_run_id") or f"test_image_gen_{int(time.time()*1000)}")
        out_dir = resolve_artifacts_dir(step_run_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        img_path = out_dir / f"test_image_{fake_id}.{img_format}"
        return {
            "images": [
                {
                    "uri": f"file://{img_path}",
                    "width": width,
                    "height": height,
                    "format": img_format,
                }
            ],
            "count": 1,
            "timings": {"total_ms": 100.0},
            "prompt": prompt,
            "prompt_refinement_mode": prompt_refinement_mode,
            "backend": backend,
            "simulated": True,
        }

    try:
        from tldw_Server_API.app.core.Image_Generation.adapter_registry import get_registry
        from tldw_Server_API.app.core.Image_Generation.adapters.base import ImageGenRequest

        registry_instance = get_registry()

        # Resolve backend name
        resolved_backend = registry_instance.resolve_backend(backend)
        if not resolved_backend:
            return {"error": f"backend_unavailable:{backend}"}

        adapter = registry_instance.get_adapter(resolved_backend)
        if not adapter:
            return {"error": f"adapter_init_failed:{resolved_backend}"}

        # Build request
        request = ImageGenRequest(
            backend=resolved_backend,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            sampler=sampler,
            model=model,
            format=img_format,
            extra_params=dict(config.get("extra_params") or {}),
        )

        # Generate
        start_ts = time.time()
        result = adapter.generate(request)
        duration_ms = (time.time() - start_ts) * 1000

        # Save image artifact
        images_output = []
        step_run_id = str(context.get("step_run_id") or f"image_gen_{int(time.time()*1000)}")
        out_dir = resolve_artifacts_dir(step_run_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Determine content type
        content_type_map = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
        }
        content_type = content_type_map.get(img_format, "image/png")

        # Save image to file
        img_filename = f"generated_{uuid.uuid4()}.{img_format}"
        img_path = out_dir / img_filename
        img_path.write_bytes(result.content)

        image_info = {
            "uri": f"file://{img_path}",
            "width": width,
            "height": height,
            "format": img_format,
            "size_bytes": result.bytes_len,
        }
        images_output.append(image_info)

        # Register artifact if requested
        artifact_registered = False
        if save_artifact and callable(context.get("add_artifact")):
            try:
                context["add_artifact"](
                    type="generated_image",
                    uri=f"file://{img_path}",
                    size_bytes=result.bytes_len,
                    mime_type=content_type,
                    metadata={
                        "prompt": prompt[:200],
                        "backend": resolved_backend,
                        "width": width,
                        "height": height,
                        "steps": steps,
                        "cfg_scale": cfg_scale,
                        "seed": seed,
                    },
                )
                artifact_registered = True
            except Exception:
                logger.warning("Image gen: failed to register artifact", exc_info=True)

        return {
            "images": images_output,
            "count": len(images_output),
            "timings": {"total_ms": duration_ms},
            "prompt": prompt,
            "prompt_refinement_mode": prompt_refinement_mode,
            "backend": resolved_backend,
            "artifact_registered": artifact_registered,
        }

    except Exception:
        logger.exception("Image gen adapter error")
        return {"error": "image_gen_error"}


@registry.register(
    "image_describe",
    category="content",
    description="Describe images",
    parallelizable=True,
    tags=["content", "image"],
    config_model=ImageDescribeConfig,
)
async def run_image_describe_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Describe an image using VLM/multimodal LLM.

    Config:
      - image_path: str - Path to image file
      - image_url: str - URL of image
      - image_base64: str - Base64 encoded image
      - prompt: str - Description prompt (default: "Describe this image in detail.")
      - provider: str - LLM provider with vision support
      - model: str - Model to use
    Output:
      - description: str
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    prompt = config.get("prompt", "Describe this image in detail.")
    if isinstance(prompt, str):
        prompt = apply_template_to_string(prompt, context) or prompt

    # Get image data
    image_data = None
    image_url = config.get("image_url")

    if config.get("image_base64"):
        image_data = config.get("image_base64")
    elif config.get("image_path"):
        try:
            path = resolve_workflow_file_path(config.get("image_path"), context, config)
            with open(path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return {"description": "", "error": "image_read_error"}

    if not image_data and not image_url:
        return {"description": "", "error": "missing_image"}

    try:
        # Build message with image
        content = [{"type": "text", "text": prompt}]
        if image_url:
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        elif image_data:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}})

        messages = [{"role": "user", "content": content}]
        response = await perform_chat_api_call_async(
            messages=messages,
            api_provider=config.get("provider"),
            model=config.get("model"),
            max_tokens=1000,
        )

        description = extract_openai_content(response) or ""
        return {"description": description, "text": description}

    except Exception:
        logger.exception("Image describe error")
        return {"description": "", "error": "image_describe_error"}
