"""stable-diffusion.cpp backend adapter."""

from __future__ import annotations

import subprocess  # nosec B404 # Required for configured local stable-diffusion.cpp binary invocation.
import tempfile
from pathlib import Path

from loguru import logger

from tldw_Server_API.app.core.Image_Generation.adapters.base import ImageGenRequest, ImageGenResult
from tldw_Server_API.app.core.Image_Generation.adapters.image_format_utils import validate_and_convert_image_output
from tldw_Server_API.app.core.Image_Generation.config import (
    DEFAULT_SD_CPP_CFG_SCALE,
    DEFAULT_SD_CPP_SAMPLER,
    DEFAULT_SD_CPP_STEPS,
    get_image_generation_config,
)
from tldw_Server_API.app.core.Image_Generation.exceptions import ImageBackendUnavailableError, ImageGenerationError
from tldw_Server_API.app.core.Image_Generation.request_validation import effective_inline_max_bytes


class StableDiffusionCppAdapter:
    name = "stable_diffusion_cpp"
    supported_formats = {"png", "jpg", "webp"}

    def __init__(self) -> None:
        self._config = get_image_generation_config()

    def generate(self, request: ImageGenRequest) -> ImageGenResult:
        binary_path = self._resolve_path(self._config.sd_cpp_binary_path, "sd_cpp_binary_path")
        if self._config.sd_cpp_diffusion_model_path:
            diffusion_model_path = request.model or self._config.sd_cpp_diffusion_model_path
            resolved_model_path = self._resolve_path(diffusion_model_path, "sd_cpp_diffusion_model_path")
            model_flag = "--diffusion-model"
        else:
            model_path = request.model or self._config.sd_cpp_model_path
            resolved_model_path = self._resolve_path(model_path, "sd_cpp_model_path")
            model_flag = "--model"
        vae_path = self._resolve_optional_path(self._config.sd_cpp_vae_path)
        lora_paths = [
            Path(path).expanduser()
            for path in self._config.sd_cpp_lora_paths
            if str(path).strip()
        ]
        output_format = request.format.lower()
        if output_format not in self.supported_formats:
            raise ImageGenerationError(f"unsupported output format: {output_format}")

        width = request.width if request.width is not None else 512
        height = request.height if request.height is not None else 512
        steps = request.steps if request.steps is not None else (self._config.sd_cpp_default_steps or DEFAULT_SD_CPP_STEPS)
        cfg_scale = request.cfg_scale if request.cfg_scale is not None else (
            self._config.sd_cpp_default_cfg_scale or DEFAULT_SD_CPP_CFG_SCALE
        )
        sampler = request.sampler or self._config.sd_cpp_default_sampler or DEFAULT_SD_CPP_SAMPLER
        extra_params = dict(request.extra_params or {})
        if "llm" not in extra_params and self._config.sd_cpp_llm_path:
            llm_path = self._resolve_path(self._config.sd_cpp_llm_path, "sd_cpp_llm_path")
            extra_params["llm"] = str(llm_path)

        with tempfile.TemporaryDirectory(prefix="sd_cpp_") as tmp_dir:
            output_path = Path(tmp_dir) / f"image.{output_format}"
            cmd = self._build_command(
                binary_path=binary_path,
                model_flag=model_flag,
                model_path=resolved_model_path,
                output_path=output_path,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=request.seed,
                sampler=sampler,
                vae_path=vae_path,
                lora_paths=lora_paths,
                extra_params=extra_params,
                device=self._config.sd_cpp_device,
            )
            logger.info(
                "stable-diffusion.cpp: running backend binary={} model_flag={} width={} height={} steps={} format={} extra_keys={}",
                binary_path.name,
                model_flag,
                width,
                height,
                steps,
                output_format,
                sorted(str(key) for key in extra_params),
            )
            try:
                result = subprocess.run(  # nosec B603 # cmd is an argv list for a configured local binary; shell=False.
                    cmd,
                    cwd=str(binary_path.parent),
                    capture_output=True,
                    text=True,
                    timeout=self._config.sd_cpp_timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise ImageGenerationError("stable-diffusion.cpp timed out") from exc

            if result.returncode != 0:
                logger.warning(
                    "stable-diffusion.cpp failed exit_code={} stderr_lines={}",
                    result.returncode,
                    len((result.stderr or "").splitlines()),
                )
                raise ImageGenerationError("stable-diffusion.cpp failed")

            if not output_path.exists():
                raise ImageGenerationError("stable-diffusion.cpp did not produce output")

            content = output_path.read_bytes()

        content_type = _content_type_for_format(output_format)
        content, content_type = validate_and_convert_image_output(
            content,
            content_type,
            output_format,
            max_bytes=effective_inline_max_bytes(self._config),
        )
        return ImageGenResult(content=content, content_type=content_type, bytes_len=len(content))

    @staticmethod
    def _resolve_path(raw_path: str | None, label: str) -> Path:
        if not raw_path:
            raise ImageBackendUnavailableError(f"{label} is not configured")
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise ImageBackendUnavailableError(f"{label} does not exist: {path}")
        return path

    @staticmethod
    def _resolve_optional_path(raw_path: str | None) -> Path | None:
        if not raw_path:
            return None
        path = Path(raw_path).expanduser()
        return path if path.exists() else None

    @staticmethod
    def _build_command(
        *,
        binary_path: Path,
        model_flag: str,
        model_path: Path,
        output_path: Path,
        prompt: str,
        negative_prompt: str | None,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        seed: int | None,
        sampler: str,
        vae_path: Path | None,
        lora_paths: list[Path],
        extra_params: dict,
        device: str | None,
    ) -> list[str]:
        cmd = [str(binary_path), model_flag, str(model_path), "-o", str(output_path), "-p", prompt]
        if negative_prompt:
            cmd += ["-n", negative_prompt]
        cmd += ["-W", str(width), "-H", str(height)]
        cmd += ["--steps", str(steps)]
        cmd += ["--cfg-scale", str(cfg_scale)]
        if seed is not None:
            cmd += ["--seed", str(seed)]
        if sampler:
            cmd += ["--sampling-method", sampler]
        if vae_path:
            cmd += ["--vae", str(vae_path)]
        for lora in lora_paths:
            cmd += ["--lora-model-dir", str(lora)]
        if device:
            cmd += ["--device", device]

        if isinstance(extra_params, dict):
            reserved = {
                "prompt",
                "negative_prompt",
                "width",
                "height",
                "steps",
                "cfg_scale",
                "seed",
                "sampler",
                "model",
                "format",
            }
            for key, value in extra_params.items():
                if key in reserved:
                    continue
                if key == "cli_args" and isinstance(value, list):
                    cmd.extend([str(v) for v in value])
                    continue
                flag = f"--{str(key).replace('_', '-')}"
                if isinstance(value, bool):
                    if value:
                        cmd.append(flag)
                    continue
                if isinstance(value, (list, tuple)):
                    for item in value:
                        cmd.extend([flag, str(item)])
                    continue
                cmd.extend([flag, str(value)])
        return cmd


def _content_type_for_format(fmt: str) -> str:
    if fmt == "png":
        return "image/png"
    if fmt == "jpg":
        return "image/jpeg"
    if fmt == "webp":
        return "image/webp"
    return "application/octet-stream"
