from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from tldw_Server_API.app.api.v1.schemas.ocr_schemas import OCRBackendsResponse
from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.registry import (
    list_backends as _list_backends,
)
from tldw_Server_API.app.core.Utils.Utils import logging

router = APIRouter(prefix="/ocr", tags=["ocr"])

_OCR_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    ImportError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _describe_mineru_backend() -> dict[str, Any]:
    """Describe MinerU without importing the adapter until discovery needs it."""
    from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.mineru_adapter import (
        describe_mineru_backend,
    )

    return describe_mineru_backend()


def _describe_mineru_backend_error(exc: Exception) -> dict[str, Any]:
    """Return a conservative MinerU capability stub when adapter discovery fails."""
    return {
        "available": False,
        "pdf_only": True,
        "document_level": True,
        "opt_in_only": True,
        "supports_per_page_metrics": False,
        "mode": "cli",
        "error": str(exc),
    }


def _record_backend_discovery_error(
    out: dict[str, Any],
    backend_name: str,
    exc: Exception,
) -> None:
    logging.error(f"OCR backend discovery failed for {backend_name}: {exc}", exc_info=True)
    out.setdefault(backend_name, {})
    out[backend_name]["error"] = str(exc)


@router.get("/backends", response_model=OCRBackendsResponse)
def list_ocr_backends() -> dict[str, Any]:
    """List available OCR backends with lightweight health information."""
    out = _list_backends()
    try:
        out["mineru"] = _describe_mineru_backend()
    except _OCR_NONCRITICAL_EXCEPTIONS as exc:
        out["mineru"] = _describe_mineru_backend_error(exc)

    # Enrich with backend-specific details without heavy loading
    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.points_reader import (
            PointsReaderBackend,
        )
        out.setdefault("points", {})
        out["points"]["mode"] = PointsReaderBackend().describe().get("mode")
        # If SGLang configured, attempt a quick availability check
        url = PointsReaderBackend().describe().get("url")
        if url:
            try:
                from tldw_Server_API.app.core.http_client import create_client as _create_client
                with _create_client(timeout=1.5) as _c:
                    r = _c.get(url.rsplit("/v1", 1)[0] + "/v1/models")
                    out["points"]["sglang_reachable"] = r.status_code in (200, 401)
            except _OCR_NONCRITICAL_EXCEPTIONS:
                out["points"]["sglang_reachable"] = False
    except _OCR_NONCRITICAL_EXCEPTIONS:
        pass

    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.dots_ocr import (
            DotsOCRBackend,
        )
        dots_desc = DotsOCRBackend().describe()
        out.setdefault("dots", {}).update({"prompt": dots_desc.get("prompt")})
        vllm = dots_desc.get("vllm_url")
        if vllm:
            try:
                from tldw_Server_API.app.core.http_client import create_client as _create_client
                with _create_client(timeout=1.5) as _c:
                    r = _c.get(vllm.rsplit("/v1", 1)[0] + "/v1/models")
                    out["dots"]["vllm_reachable"] = r.status_code in (200, 401)
            except _OCR_NONCRITICAL_EXCEPTIONS:
                out["dots"]["vllm_reachable"] = False
    except _OCR_NONCRITICAL_EXCEPTIONS:
        pass

    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.hunyuan_ocr import (
            HunyuanOCRBackend,
        )
        hun_desc = HunyuanOCRBackend().describe()
        out.setdefault("hunyuan", {}).update(
            {
                "mode": hun_desc.get("mode"),
                "prompt_preset": hun_desc.get("prompt_preset"),
            }
        )
        vllm = hun_desc.get("url")
        if vllm:
            try:
                from tldw_Server_API.app.core.http_client import create_client as _create_client
                with _create_client(timeout=1.5) as _c:
                    r = _c.get(vllm.rsplit("/v1", 1)[0] + "/v1/models")
                    out["hunyuan"]["vllm_reachable"] = r.status_code in (200, 401)
            except _OCR_NONCRITICAL_EXCEPTIONS:
                out["hunyuan"]["vllm_reachable"] = False
    except _OCR_NONCRITICAL_EXCEPTIONS:
        pass

    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr import (
            LlamaCppOCRBackend,
        )

        llamacpp_desc = LlamaCppOCRBackend().describe()
        out.setdefault("llamacpp", {}).update(
            {
                "mode": llamacpp_desc.get("mode"),
                "configured_mode": llamacpp_desc.get("configured_mode"),
                "model": llamacpp_desc.get("model"),
                "configured": llamacpp_desc.get("configured"),
                "supports_structured_output": llamacpp_desc.get("supports_structured_output"),
                "supports_json": llamacpp_desc.get("supports_json"),
                "configured_flags": llamacpp_desc.get("configured_flags"),
                "auto_eligible": llamacpp_desc.get("auto_eligible"),
                "auto_high_quality_eligible": llamacpp_desc.get("auto_high_quality_eligible"),
                "url_configured": llamacpp_desc.get("url_configured"),
                "managed_configured": llamacpp_desc.get("managed_configured"),
                "managed_running": llamacpp_desc.get("managed_running"),
                "allow_managed_start": llamacpp_desc.get("allow_managed_start"),
                "cli_configured": llamacpp_desc.get("cli_configured"),
                "backend_concurrency_cap": llamacpp_desc.get("backend_concurrency_cap"),
            }
        )
    except _OCR_NONCRITICAL_EXCEPTIONS as exc:
        _record_backend_discovery_error(out, "llamacpp", exc)

    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.chatllm_ocr import (
            ChatLLMOCRBackend,
        )

        chatllm_desc = ChatLLMOCRBackend().describe()
        out.setdefault("chatllm", {}).update(
            {
                "mode": chatllm_desc.get("mode"),
                "configured": chatllm_desc.get("configured"),
                "supports_structured_output": chatllm_desc.get("supports_structured_output"),
                "supports_json": chatllm_desc.get("supports_json"),
                "auto_eligible": chatllm_desc.get("auto_eligible"),
                "auto_high_quality_eligible": chatllm_desc.get("auto_high_quality_eligible"),
                "managed_configured": chatllm_desc.get("managed_configured"),
                "managed_running": chatllm_desc.get("managed_running"),
                "allow_managed_start": chatllm_desc.get("allow_managed_start"),
                "url_configured": chatllm_desc.get("url_configured"),
                "healthcheck_url_configured": chatllm_desc.get("healthcheck_url_configured"),
                "cli_configured": chatllm_desc.get("cli_configured"),
                "backend_concurrency_cap": chatllm_desc.get("backend_concurrency_cap"),
                "model": chatllm_desc.get("model"),
            }
        )
    except _OCR_NONCRITICAL_EXCEPTIONS as exc:
        _record_backend_discovery_error(out, "chatllm", exc)

    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.nemotron_parse import (
            NemotronParseBackend,
        )
        nemo_desc = NemotronParseBackend().describe()
        out.setdefault("nemotron_parse", {}).update(
            {
                "mode": nemo_desc.get("mode"),
                "prompt": nemo_desc.get("prompt"),
                "text_format": nemo_desc.get("text_format"),
                "table_format": nemo_desc.get("table_format"),
            }
        )
        vllm = nemo_desc.get("url")
        if vllm:
            try:
                from tldw_Server_API.app.core.http_client import create_client as _create_client
                with _create_client(timeout=1.5) as _c:
                    r = _c.get(vllm.rsplit("/v1", 1)[0] + "/v1/models")
                    out["nemotron_parse"]["vllm_reachable"] = r.status_code in (200, 401)
            except _OCR_NONCRITICAL_EXCEPTIONS:
                out["nemotron_parse"]["vllm_reachable"] = False
    except _OCR_NONCRITICAL_EXCEPTIONS:
        pass

    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.dolphin_ocr import (
            DolphinOCRBackend,
        )
        dolph_desc = DolphinOCRBackend().describe()
        out.setdefault("dolphin", {}).update(
            {
                "mode": dolph_desc.get("mode"),
                "remote_mode": dolph_desc.get("remote_mode"),
                "prompt_preset": dolph_desc.get("prompt_preset"),
            }
        )
    except _OCR_NONCRITICAL_EXCEPTIONS:
        pass

    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.deepseek_ocr import (
            DeepSeekOCRBackend,
        )

        deepseek_desc = DeepSeekOCRBackend().describe()
        out.setdefault("deepseek", {}).update(
            {
                "model_id": deepseek_desc.get("model_id"),
                "prompt": deepseek_desc.get("prompt"),
                "base_size": deepseek_desc.get("base_size"),
                "image_size": deepseek_desc.get("image_size"),
                "crop_mode": deepseek_desc.get("crop_mode"),
                "device": deepseek_desc.get("device"),
                "dtype": deepseek_desc.get("dtype"),
                "attn_impl": deepseek_desc.get("attn_impl"),
            }
        )
    except _OCR_NONCRITICAL_EXCEPTIONS:
        pass

    return out


@router.post("/points/preload")
def preload_points_transformers() -> dict[str, Any]:
    """Preload POINTS transformers model to surface errors early."""
    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.points_reader import (
            _load_transformers,
        )
        _load_transformers()
        return {"status": "ok", "mode": "transformers"}
    except _OCR_NONCRITICAL_EXCEPTIONS as e:
        return {"status": "error", "error": str(e)}
