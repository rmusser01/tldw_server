from __future__ import annotations

from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.deepseek_ocr import (
    DeepSeekOCRBackend,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.chatllm_ocr import (
    ChatLLMOCRBackend,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.dolphin_ocr import (
    DolphinOCRBackend,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.dots_ocr import (
    DotsOCRBackend,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.hunyuan_ocr import (
    HunyuanOCRBackend,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr import (
    LlamaCppOCRBackend,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.nemotron_parse import (
    NemotronParseBackend,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.points_reader import (
    PointsReaderBackend,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.tesseract_cli import (
    TesseractCLIBackend,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.base import OCRBackend
import os

try:
    # Optional config override
    from tldw_Server_API.app.core.config import loaded_config_data as _loaded_cfg
except Exception:
    _loaded_cfg = None

_BACKENDS: dict[str, type[OCRBackend]] = {
    # Auto-detection priority is the dictionary order below
    # Keep Tesseract first for a lightweight default, but users can force 'dots'
    TesseractCLIBackend.name: TesseractCLIBackend,
    NemotronParseBackend.name: NemotronParseBackend,
    PointsReaderBackend.name: PointsReaderBackend,
    DeepSeekOCRBackend.name: DeepSeekOCRBackend,
    HunyuanOCRBackend.name: HunyuanOCRBackend,
    DotsOCRBackend.name: DotsOCRBackend,
    DolphinOCRBackend.name: DolphinOCRBackend,
    LlamaCppOCRBackend.name: LlamaCppOCRBackend,
    ChatLLMOCRBackend.name: ChatLLMOCRBackend,
}

_AUTO_EXCLUDE = {
    # Opt-in only unless explicitly selected or in backend_priority
    "dolphin",
}


def _resolve_priority_from_config() -> list[str] | None:
    try:
        if not _loaded_cfg:
            return None
        ocr_cfg = _loaded_cfg.get("OCR") or {}
        pr = ocr_cfg.get("backend_priority")
        if not pr:
            return None
        if isinstance(pr, str):
            # comma-separated list
            return [s.strip() for s in pr.split(",") if s.strip()]
        if isinstance(pr, list):
            return [str(s).strip() for s in pr if str(s).strip()]
        return None
    except Exception:
        return None


def _env_flag(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _is_auto_eligible(backend_name: str, *, high_quality: bool) -> bool:
    if backend_name == "llamacpp":
        if high_quality:
            return _env_flag("LLAMACPP_OCR_AUTO_HIGH_QUALITY_ELIGIBLE")
        return _env_flag("LLAMACPP_OCR_AUTO_ELIGIBLE")
    if backend_name == "chatllm":
        if high_quality:
            return _env_flag("CHATLLM_OCR_AUTO_HIGH_QUALITY_ELIGIBLE")
        return _env_flag("CHATLLM_OCR_AUTO_ELIGIBLE")
    return True


def get_backend(name: str | None = None) -> OCRBackend | None:
    """
    Resolve an OCR backend instance by name or choose the first available.

    name = None         -> auto-detect first available backend
    name = "auto"       -> same as None
    name = "tesseract"  -> Tesseract CLI backend, if available
    name = "dots"       -> dots.ocr CLI backend, if available
    name = "points"     -> POINTS-Reader backend, if available
    name = "deepseek"   -> DeepSeek OCR backend, if available
    name = "dolphin"    -> Dolphin backend, if available
    """
    candidates = []
    if name and name not in ("auto", "auto_high_quality"):
        cls = _BACKENDS.get(name)
        if cls and cls.available():
            return cls()
        return None

    # Auto resolution
    preferred_order: list[str] | None = None
    if name == "auto_high_quality":
        preferred_order = _resolve_priority_from_config()
        if not preferred_order:
            preferred_order = [
                "llamacpp",
                "chatllm",
                "nemotron_parse",
                "hunyuan",
                "deepseek",
                "points",
                "dots",
                "dolphin",
                "tesseract",
            ]
    else:
        # Try config override for normal auto
        preferred_order = _resolve_priority_from_config()
        if not preferred_order:
            preferred_order = [
                "tesseract",
                "nemotron_parse",
                "points",
                "deepseek",
                "hunyuan",
                "dots",
                "dolphin",
                "llamacpp",
                "chatllm",
            ]

    if preferred_order:
        for key in preferred_order:
            cls = _BACKENDS.get(key)
            if (
                cls
                and _is_auto_eligible(
                    key,
                    high_quality=name == "auto_high_quality",
                )
                and cls.available()
            ):
                return cls()
        # fallback to discovery order below

    # Default: first available in registration order
    for cls in _BACKENDS.values():
        if getattr(cls, "name", None) in _AUTO_EXCLUDE:
            continue
        if not _is_auto_eligible(
            getattr(cls, "name", ""),
            high_quality=name == "auto_high_quality",
        ):
            continue
        if cls.available():
            candidates.append(cls)
    if not candidates:
        return None
    return candidates[0]()


def list_backends() -> dict[str, dict[str, bool]]:
    """Return a simple availability map for known backends."""
    out: dict[str, dict[str, bool]] = {}
    for k, cls in _BACKENDS.items():
        ok = False
        try:
            ok = bool(cls.available())
        except Exception:
            ok = False
        out[k] = {"available": ok}
    return out
