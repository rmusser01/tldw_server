"""File artifact adapter for image generation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from contextvars import ContextVar, Token
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.exceptions import FileArtifactsError, FileArtifactsValidationError
from tldw_Server_API.app.core.File_Artifacts.adapters.base import ExportResult, ValidationIssue
from tldw_Server_API.app.core.Image_Generation.capabilities import (
    ResolvedReferenceImage,
    resolve_backend_reference_image_capability,
    resolve_reference_image_capability,
)
from tldw_Server_API.app.core.Image_Generation.adapter_registry import get_registry
from tldw_Server_API.app.core.Image_Generation.adapters.base import ImageGenRequest
from tldw_Server_API.app.core.Image_Generation.config import get_image_generation_config
from tldw_Server_API.app.core.Image_Generation.exceptions import ImageBackendUnavailableError, ImageGenerationError
from tldw_Server_API.app.core.Image_Generation.reference_images import (
    ReferenceImageOperationalError,
    resolve_reference_image,
)
from tldw_Server_API.app.core.Image_Generation.prompt_refinement import (
    normalize_prompt_refinement_mode,
    refine_image_prompt,
)


@dataclass(frozen=True)
class _ImageAdapterRequestContext:
    collections_db: Any
    user_id: str


_REQUEST_CONTEXT: ContextVar[_ImageAdapterRequestContext | None] = ContextVar(
    "file_artifacts_image_adapter_request_context",
    default=None,
)


class _MediaDbCursorProxy:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[dict[str, Any]]:
        return list(self._rows)


class _MediaDbProxy:
    def __init__(self, collections_db: Any) -> None:
        self._collections_db = collections_db
        self.client_id = str(getattr(collections_db, "user_id", "") or "")
        backend = getattr(collections_db, "backend", None)
        backend_config = getattr(backend, "config", None)
        self.db_path_str = str(getattr(backend_config, "sqlite_path", "") or "")
        self.backend_type = getattr(backend, "backend_type", None)
        self._backend = backend

    def execute_query(self, query: str, params: Any = None) -> _MediaDbCursorProxy:
        if self._backend is None:
            raise FileArtifactsValidationError("reference_image_invalid")
        result = self._backend.execute(query, params)
        rows = getattr(result, "rows", None)
        if rows is None and hasattr(result, "fetchall"):
            rows = result.fetchall()
        normalized_rows: list[dict[str, Any]] = []
        for row in rows or []:
            if isinstance(row, dict):
                normalized_rows.append(row)
            else:
                normalized_rows.append(dict(row))
        return _MediaDbCursorProxy(normalized_rows)


def set_image_adapter_request_context(
    *,
    collections_db: Any,
    user_id: int | str,
) -> Token[_ImageAdapterRequestContext | None]:
    """Attach request-scoped collections DB context for image artifact exports."""

    return _REQUEST_CONTEXT.set(
        _ImageAdapterRequestContext(
            collections_db=collections_db,
            user_id=str(user_id),
        )
    )


def reset_image_adapter_request_context(token: Token[_ImageAdapterRequestContext | None]) -> None:
    """Restore the previous image artifact request context."""

    _REQUEST_CONTEXT.reset(token)


def _get_image_adapter_request_context() -> _ImageAdapterRequestContext | None:
    return _REQUEST_CONTEXT.get()


class ImageAdapter:
    file_type = "image"
    export_formats = {"png", "jpg", "webp"}

    def normalize(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise FileArtifactsValidationError("image_params_invalid")

        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise FileArtifactsValidationError("image_params_invalid")

        backend = payload.get("backend")
        backend_name = str(backend).strip() if backend is not None else None
        registry = get_registry()
        resolved_backend = registry.resolve_backend(backend_name)
        if not resolved_backend:
            raise FileArtifactsError("image_backend_unavailable")
        image_config = get_image_generation_config()
        prompt_mode = normalize_prompt_refinement_mode(payload.get("prompt_refinement"))
        normalized_prompt = refine_image_prompt(
            prompt.strip(),
            mode=prompt_mode,
            max_length=image_config.max_prompt_length,
        )
        reference_file_id = self._positive_int_or_none(payload.get("reference_file_id"))

        structured = {
            "backend": resolved_backend,
            "prompt": normalized_prompt,
            "negative_prompt": self._string_or_none(payload.get("negative_prompt")),
            "width": self._int_or_none(payload.get("width")),
            "height": self._int_or_none(payload.get("height")),
            "steps": self._int_or_none(payload.get("steps")),
            "cfg_scale": self._float_or_none(payload.get("cfg_scale")),
            "seed": self._int_or_none(payload.get("seed")),
            "sampler": self._string_or_none(payload.get("sampler")),
            "model": self._string_or_none(payload.get("model")),
            "prompt_refinement": prompt_mode,
            "extra_params": payload.get("extra_params") or {},
        }

        if not isinstance(structured["extra_params"], dict):
            raise FileArtifactsValidationError("image_params_invalid")
        if reference_file_id is not None:
            structured["reference_file_id"] = reference_file_id
            structured["reference_image_provenance"] = {
                "source": "managed_reference_image",
                "reference_file_id": reference_file_id,
            }

        return structured

    def validate(self, structured: dict[str, Any]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        config = get_image_generation_config()

        prompt = structured.get("prompt")
        if isinstance(prompt, str) and len(prompt) > config.max_prompt_length:
            issues.append(
                ValidationIssue(
                    code="image_params_invalid",
                    message="prompt exceeds max length",
                    path="prompt",
                )
            )

        width = structured.get("width")
        height = structured.get("height")
        if width is not None and (width <= 0 or width > config.max_width):
            issues.append(
                ValidationIssue(
                    code="image_params_invalid",
                    message="width out of range",
                    path="width",
                )
            )
        if height is not None and (height <= 0 or height > config.max_height):
            issues.append(
                ValidationIssue(
                    code="image_params_invalid",
                    message="height out of range",
                    path="height",
                )
            )
        if isinstance(width, int) and isinstance(height, int) and width * height > config.max_pixels:
            issues.append(
                ValidationIssue(
                    code="image_params_invalid",
                    message="image dimensions exceed max pixels",
                    path="width,height",
                )
            )

        steps = structured.get("steps")
        if steps is not None and (steps <= 0 or steps > config.max_steps):
            issues.append(
                ValidationIssue(
                    code="image_params_invalid",
                    message="steps out of range",
                    path="steps",
                )
            )

        self._validate_extra_params(structured, config, issues)

        return issues

    def export(self, structured: dict[str, Any], *, format: str) -> ExportResult:
        backend = structured.get("backend")
        if not backend:
            raise FileArtifactsError("image_backend_unavailable")

        registry = get_registry()
        adapter = registry.get_adapter(str(backend))
        if adapter is None:
            raise FileArtifactsError("image_backend_unavailable")

        reference_image = None
        if structured.get("reference_file_id") is not None:
            reference_image = self._resolve_reference_image(structured, backend=str(backend))
            self._attach_reference_image_provenance(structured, reference_image)

        request = ImageGenRequest(
            backend=str(backend),
            prompt=str(structured.get("prompt") or ""),
            negative_prompt=self._string_or_none(structured.get("negative_prompt")),
            width=structured.get("width"),
            height=structured.get("height"),
            steps=structured.get("steps"),
            cfg_scale=structured.get("cfg_scale"),
            seed=structured.get("seed"),
            sampler=self._string_or_none(structured.get("sampler")),
            model=self._string_or_none(structured.get("model")),
            format=format,
            extra_params=structured.get("extra_params") or {},
            reference_image=reference_image,
        )

        try:
            result = adapter.generate(request)
        except ImageBackendUnavailableError as exc:
            raise FileArtifactsError("image_backend_unavailable", detail=str(exc)) from exc
        except ImageGenerationError as exc:
            raise FileArtifactsError("image_generation_failed", detail="image_generation_failed") from exc
        except Exception as exc:
            logger.warning("image adapter: backend generate failed")
            raise FileArtifactsError("image_generation_failed", detail="image_generation_failed") from exc

        return ExportResult(
            status="ready",
            content_type=result.content_type,
            bytes_len=result.bytes_len,
            content=result.content,
        )

    def _resolve_reference_image(self, structured: dict[str, Any], *, backend: str) -> ResolvedReferenceImage:
        reference_file_id = self._positive_int_or_none(structured.get("reference_file_id"))
        if reference_file_id is None:
            raise FileArtifactsValidationError("reference_image_invalid")

        model = self._string_or_none(structured.get("model"))
        image_config = get_image_generation_config()
        backend_capability = resolve_backend_reference_image_capability(backend, config=image_config)
        if not backend_capability.supported:
            raise FileArtifactsValidationError("reference_image_unsupported_by_backend")
        capability = resolve_reference_image_capability(backend, model, config=image_config)
        if not capability.supported:
            raise FileArtifactsValidationError("reference_image_unsupported_by_model")

        context = _get_image_adapter_request_context()
        if context is None or not context.collections_db:
            raise FileArtifactsValidationError("reference_image_invalid")

        proxy = _MediaDbProxy(context.collections_db)
        try:
            return asyncio.run(
                resolve_reference_image(
                    proxy,
                    user_id=context.user_id,
                    file_id=reference_file_id,
                )
            )
        except ReferenceImageOperationalError as exc:
            raise FileArtifactsError("reference_image_storage_unavailable", detail=str(exc)) from exc
        except FileArtifactsValidationError:
            raise
        except Exception as exc:
            logger.warning("image adapter: reference image resolution failed")
            raise FileArtifactsValidationError("reference_image_invalid") from exc

    @staticmethod
    def _attach_reference_image_provenance(
        structured: dict[str, Any],
        reference_image: ResolvedReferenceImage,
    ) -> None:
        provenance = dict(structured.get("reference_image_provenance") or {})
        provenance["snapshot"] = ImageAdapter._reference_image_snapshot(reference_image)
        structured["reference_image_provenance"] = provenance

    @staticmethod
    def _reference_image_snapshot(reference_image: ResolvedReferenceImage) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "filename": reference_image.filename,
            "mime_type": reference_image.mime_type,
        }
        if reference_image.width is not None:
            snapshot["width"] = reference_image.width
        if reference_image.height is not None:
            snapshot["height"] = reference_image.height
        return snapshot

    @staticmethod
    def _string_or_none(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _int_or_none(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            raise FileArtifactsValidationError("image_params_invalid") from None

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            raise FileArtifactsValidationError("image_params_invalid") from None

    @staticmethod
    def _positive_int_or_none(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise FileArtifactsValidationError("reference_image_invalid")
        try:
            candidate = int(value)
        except (TypeError, ValueError):
            raise FileArtifactsValidationError("reference_image_invalid") from None
        if candidate <= 0:
            raise FileArtifactsValidationError("reference_image_invalid")
        return candidate

    @staticmethod
    def _allowed_extra_params(backend: str, config) -> set[str]:
        if backend == "stable_diffusion_cpp":
            return set(config.sd_cpp_allowed_extra_params or [])
        if backend == "swarmui":
            return set(config.swarmui_allowed_extra_params or [])
        if backend == "openrouter":
            return set(config.openrouter_image_allowed_extra_params or [])
        if backend == "novita":
            return set(config.novita_image_allowed_extra_params or [])
        if backend == "together":
            return set(config.together_image_allowed_extra_params or [])
        if backend == "modelstudio":
            return set(config.modelstudio_image_allowed_extra_params or [])
        return set()

    def _validate_extra_params(
        self,
        structured: dict[str, Any],
        config,
        issues: list[ValidationIssue],
    ) -> None:
        extra_params = structured.get("extra_params") or {}
        if not extra_params:
            return
        backend = str(structured.get("backend") or "").strip()
        allowlist = self._allowed_extra_params(backend, config)
        exempt_control_keys: set[str] = set()
        if backend == "modelstudio":
            # `mode` is a local backend control knob, not a passthrough upstream parameter.
            exempt_control_keys.add("mode")
        keys_to_validate = [key for key in extra_params if key not in exempt_control_keys]
        if not keys_to_validate:
            return
        if not allowlist:
            for key in keys_to_validate:
                issues.append(
                    ValidationIssue(
                        code="image_params_invalid",
                        message="extra_params key not allowlisted",
                        path=f"extra_params.{key}",
                    )
                )
            return
        for key in keys_to_validate:
            if key not in allowlist:
                issues.append(
                    ValidationIssue(
                        code="image_params_invalid",
                        message="extra_params key not allowlisted",
                        path=f"extra_params.{key}",
                    )
                )
        if "cli_args" in extra_params and "cli_args" in allowlist:
            cli_args = extra_params.get("cli_args")
            if not isinstance(cli_args, (list, tuple)):
                issues.append(
                    ValidationIssue(
                        code="image_params_invalid",
                        message="cli_args must be a list",
                        path="extra_params.cli_args",
                    )
                )
