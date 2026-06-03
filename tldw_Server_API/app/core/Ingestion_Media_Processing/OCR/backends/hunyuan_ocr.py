from __future__ import annotations

import base64
import contextlib
import importlib.util
import io
import json
import os
import tempfile
import threading
from typing import Any

from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.base import OCRBackend
from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.hunyuan_llamacpp_runtime import (
    HunyuanLlamaCppRuntime,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.types import (
    OCRBlock,
    OCRResult,
    OCRTable,
    normalize_ocr_format,
)
from tldw_Server_API.app.core.Utils.Utils import logging

_TF_MODEL = None
_TF_PROCESSOR = None
_TF_LOCK = threading.Lock()
_HUNYUAN_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)


def _resolve_mode() -> str:
    mode = (os.getenv("HUNYUAN_MODE") or "auto").lower()
    if mode not in ("auto", "vllm", "transformers"):
        mode = "auto"
    return mode


def _resolve_runtime_family() -> str:
    family = (os.getenv("HUNYUAN_RUNTIME_FAMILY") or "auto").strip().lower() or "auto"
    if family in {"auto", "native", "llamacpp"}:
        return family
    return "auto"


def _native_vllm_configured() -> bool:
    return bool((os.getenv("HUNYUAN_VLLM_URL") or "").strip())


def _native_transformers_intended() -> bool:
    mode = _resolve_mode()
    if mode == "transformers":
        return True
    return bool((os.getenv("HUNYUAN_MODEL_PATH") or "").strip())


def _native_transformers_available() -> bool:
    mode = _resolve_mode()
    if mode not in ("auto", "transformers"):
        return False
    if not _native_transformers_intended():
        return False
    try:
        has_tf = importlib.util.find_spec("transformers") is not None
        has_torch = importlib.util.find_spec("torch") is not None
        has_pil = importlib.util.find_spec("PIL") is not None
        return bool(has_tf and has_torch and has_pil)
    except _HUNYUAN_NONCRITICAL_EXCEPTIONS:
        return False


def _native_family_available() -> bool:
    return _native_vllm_configured() or _native_transformers_available()


def _native_effective_mode() -> str:
    mode = _resolve_mode()
    if mode == "vllm":
        return "vllm"
    if mode == "transformers":
        return "transformers"
    if _native_vllm_configured():
        return "vllm"
    if _native_transformers_available():
        return "transformers"
    return "auto"


def _effective_runtime_family() -> str:
    family = _resolve_runtime_family()
    if family == "native":
        return "native"
    if family == "llamacpp":
        return "llamacpp"
    if _native_family_available():
        return "native"
    if HunyuanLlamaCppRuntime.available():
        return "llamacpp"
    return "native"


def _native_describe() -> dict[str, Any]:
    mode = _native_effective_mode()
    model = (
        os.getenv("HUNYUAN_VLLM_MODEL", "HunyuanOCR")
        if mode == "vllm"
        else (os.getenv("HUNYUAN_MODEL_PATH") or "tencent/HunyuanOCR")
    )
    return {
        "mode": mode,
        "configured": _native_vllm_configured() or _native_transformers_intended(),
        "available": _native_family_available(),
        "url": os.getenv("HUNYUAN_VLLM_URL"),
        "model": model,
        "model_path": os.getenv("HUNYUAN_MODEL_PATH") or "tencent/HunyuanOCR",
        "device": os.getenv("HUNYUAN_DEVICE"),
        "vllm_configured": _native_vllm_configured(),
        "transformers_intended": _native_transformers_intended(),
    }


def _ocr_via_native_family(image_bytes: bytes, prompt: str) -> tuple[str, str, str | None]:
    mode = _native_effective_mode()
    if mode == "vllm" and _native_vllm_configured():
        try:
            return (
                _ocr_via_vllm(image_bytes, prompt),
                "vllm",
                os.getenv("HUNYUAN_VLLM_MODEL", "HunyuanOCR"),
            )
        except _HUNYUAN_NONCRITICAL_EXCEPTIONS as exc:
            logging.error(f"Hunyuan vLLM path failed: {exc}", exc_info=True)
    if _native_transformers_available():
        try:
            return (
                _ocr_via_transformers(image_bytes, prompt),
                "transformers",
                os.getenv("HUNYUAN_MODEL_PATH") or "tencent/HunyuanOCR",
            )
        except _HUNYUAN_NONCRITICAL_EXCEPTIONS as exc:
            logging.error(f"Hunyuan transformers path failed: {exc}", exc_info=True)
    return "", mode, None


_PROMPT_PRESETS: dict[str, str] = {
    "general": "Extract all visible text from the image.",
    "doc": "Parse the document and return all text in Markdown. Render tables as HTML.",
    "table": "Extract tables as HTML. Return all other text in Markdown.",
    "spotting": "Extract all text with bounding boxes. Return JSON only with fields: blocks[].text and blocks[].bbox.",
    "json": "Return JSON only with fields: text (string) and blocks (array of {text,bbox}).",
}


class HunyuanOCRBackend(OCRBackend):
    """
    HunyuanOCR backend supporting:
      - vLLM OpenAI-compatible endpoint
      - local Transformers inference

    Config via env:
      HUNYUAN_MODE: 'auto' (default), 'vllm', 'transformers'
      HUNYUAN_PROMPT: override prompt text
      HUNYUAN_PROMPT_PRESET: default preset name if prompt not provided

      vLLM:
        HUNYUAN_VLLM_URL, HUNYUAN_VLLM_MODEL, HUNYUAN_VLLM_TIMEOUT
        HUNYUAN_VLLM_USE_DATA_URL
        HUNYUAN_MAX_NEW_TOKENS, HUNYUAN_TEMPERATURE

      Transformers:
        HUNYUAN_MODEL_PATH, HUNYUAN_DEVICE
    """

    name = "hunyuan"

    @classmethod
    def auto_eligible(cls, high_quality: bool) -> bool:
        family = _resolve_runtime_family()
        if family == "native":
            return _native_family_available()
        if family == "llamacpp":
            return HunyuanLlamaCppRuntime.available() and HunyuanLlamaCppRuntime.auto_eligible(
                high_quality
            )
        if _native_family_available():
            return True
        if HunyuanLlamaCppRuntime.available():
            return HunyuanLlamaCppRuntime.auto_eligible(high_quality)
        return False

    @classmethod
    def available(cls) -> bool:
        family = _resolve_runtime_family()
        if family == "native":
            return _native_family_available()
        if family == "llamacpp":
            return HunyuanLlamaCppRuntime.available()
        return _native_family_available() or HunyuanLlamaCppRuntime.available()

    def describe(self) -> dict:
        configured_family = _resolve_runtime_family()
        effective_family = _effective_runtime_family()
        native_desc = _native_describe()
        llamacpp_desc = HunyuanLlamaCppRuntime.describe()
        active_mode = (
            llamacpp_desc.get("mode")
            if effective_family == "llamacpp"
            else native_desc.get("mode")
        )
        active_model = (
            llamacpp_desc.get("model")
            if effective_family == "llamacpp"
            else native_desc.get("model")
        )
        info: dict[str, Any] = {
            "mode": active_mode,
            "runtime_family": effective_family,
            "configured_family": configured_family,
            "prompt": os.getenv("HUNYUAN_PROMPT"),
            "prompt_preset": os.getenv("HUNYUAN_PROMPT_PRESET"),
            "model": active_model,
            "configured": bool(
                native_desc.get("configured") or llamacpp_desc.get("configured")
            ),
            "supports_structured_output": True,
            "supports_json": True,
            "auto_eligible": type(self).auto_eligible(False),
            "auto_high_quality_eligible": type(self).auto_eligible(True),
            "backend_concurrency_cap": (
                llamacpp_desc.get("backend_concurrency_cap")
                if effective_family == "llamacpp"
                else None
            ),
            "native": native_desc,
            "llamacpp": llamacpp_desc,
        }
        if effective_family == "native" and active_mode == "vllm":
            info.update(
                {
                    "url": native_desc.get("url"),
                    "model": native_desc.get("model"),
                    "timeout": int(os.getenv("HUNYUAN_VLLM_TIMEOUT", "60")),
                    "use_data_url": os.getenv("HUNYUAN_VLLM_USE_DATA_URL", "true"),
                }
            )
        elif effective_family == "native":
            info.update(
                {
                    "model_path": native_desc.get("model_path"),
                    "device": os.getenv("HUNYUAN_DEVICE"),
                }
            )
        return info

    def ocr_image(self, image_bytes: bytes, lang: str | None = None) -> str:
        result = self.ocr_image_structured(image_bytes, lang=lang, output_format="text")
        return result.text or ""

    def ocr_image_structured(
        self,
        image_bytes: bytes,
        lang: str | None = None,
        output_format: str | None = None,
        prompt_preset: str | None = None,
    ) -> OCRResult:
        if not self.available():
            logging.warning(
                "HunyuanOCRBackend not available: configure native Hunyuan or "
                "HUNYUAN_LLAMACPP_* runtime settings."
            )
            return OCRResult(text="", format="text")

        prompt = _resolve_prompt(prompt_preset, output_format)
        configured_family = _resolve_runtime_family()
        runtime_family = _effective_runtime_family()

        raw_text = ""
        mode: str | None = None
        model: str | None = None
        backend_concurrency_cap: int | None = None

        if runtime_family == "native":
            raw_text, mode, model = _ocr_via_native_family(image_bytes, prompt)
            if (
                not raw_text
                and configured_family == "auto"
                and HunyuanLlamaCppRuntime.available()
            ):
                runtime_family = "llamacpp"

        if runtime_family == "llamacpp":
            try:
                raw_text = HunyuanLlamaCppRuntime.ocr_image(image_bytes, prompt)
                llamacpp_desc = HunyuanLlamaCppRuntime.describe()
                mode = str(llamacpp_desc.get("mode") or "")
                model = (
                    None
                    if llamacpp_desc.get("model") is None
                    else str(llamacpp_desc.get("model"))
                )
                raw_cap = llamacpp_desc.get("backend_concurrency_cap")
                backend_concurrency_cap = raw_cap if isinstance(raw_cap, int) else None
            except _HUNYUAN_NONCRITICAL_EXCEPTIONS as exc:
                logging.error(f"Hunyuan llama.cpp path failed: {exc}", exc_info=True)
                raw_text = ""

        if _should_clean_repeats():
            raw_text = _clean_repeated_substrings(raw_text)

        meta = {
            "backend": self.name,
            "mode": mode,
            "runtime_family": runtime_family,
            "prompt_preset": prompt_preset or os.getenv("HUNYUAN_PROMPT_PRESET"),
            "output_format": output_format,
            "model": model,
        }
        if backend_concurrency_cap is not None:
            meta["backend_concurrency_cap"] = backend_concurrency_cap
        return _build_result_from_output(raw_text, output_format, prompt_preset, meta)


def _resolve_prompt(prompt_preset: str | None, output_format: str | None) -> str:
    env_prompt = os.getenv("HUNYUAN_PROMPT")
    if env_prompt:
        return env_prompt

    preset = (prompt_preset or os.getenv("HUNYUAN_PROMPT_PRESET") or "").strip().lower()
    if not preset:
        fmt = normalize_ocr_format(output_format)
        if fmt == "json":
            preset = "json"
        elif fmt == "markdown":
            preset = "doc"
        else:
            preset = "general"
    return _PROMPT_PRESETS.get(preset, _PROMPT_PRESETS["general"])


def _ocr_via_vllm(image_bytes: bytes, prompt: str) -> str:
    url = os.getenv("HUNYUAN_VLLM_URL")
    if not url:
        raise RuntimeError("HUNYUAN_VLLM_URL not set")
    url = url.rstrip("/")

    model = os.getenv("HUNYUAN_VLLM_MODEL", "HunyuanOCR")
    timeout = int(os.getenv("HUNYUAN_VLLM_TIMEOUT", "60"))
    use_data_url = str(os.getenv("HUNYUAN_VLLM_USE_DATA_URL", "true")).lower() in ("1", "true", "yes")

    content_image = None
    if use_data_url:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        content_image = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
    else:
        # Path-based URL is likely only useful for local file access
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as f:
            f.write(image_bytes)
            f.flush()
            content_image = {"type": "image_url", "image_url": {"url": f.name}}

    def _getf(env: str, cast, default):
        try:
            return cast(os.getenv(env, str(default)))
        except (TypeError, ValueError):
            return default

    max_tokens = _getf("HUNYUAN_MAX_NEW_TOKENS", int, 2048)

    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    content_image,
                ],
            }
        ],
        "max_new_tokens": max_tokens,
        "max_tokens": max_tokens,
        "temperature": _getf("HUNYUAN_TEMPERATURE", float, 0.0),
    }

    from tldw_Server_API.app.core.http_client import fetch_json

    j = fetch_json(method="POST", url=url, json=data, timeout=timeout)
    return (
        j.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    ) or ""


def _load_transformers():
    global _TF_MODEL, _TF_PROCESSOR
    if _TF_MODEL is not None and _TF_PROCESSOR is not None:
        return _TF_MODEL, _TF_PROCESSOR

    with _TF_LOCK:
        if _TF_MODEL is not None and _TF_PROCESSOR is not None:
            return _TF_MODEL, _TF_PROCESSOR

        import torch
        from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

        model_path = os.getenv("HUNYUAN_MODEL_PATH", "tencent/HunyuanOCR")
        model_revision = (os.getenv("HUNYUAN_MODEL_REVISION") or "").strip() or None
        device_env = os.getenv("HUNYUAN_DEVICE")
        device_map = device_env or "auto"

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        _TF_MODEL = HunYuanVLForConditionalGeneration.from_pretrained(
            model_path,
            revision=model_revision,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )  # nosec B615
        _TF_PROCESSOR = AutoProcessor.from_pretrained(  # nosec B615
            model_path,
            revision=model_revision,
            trust_remote_code=True,
        )
        return _TF_MODEL, _TF_PROCESSOR


def _ocr_via_transformers(image_bytes: bytes, prompt: str) -> str:
    model, processor = _load_transformers()

    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Apply chat template using a placeholder image token; the real image is passed via `images=`.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "input_image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt")

    with contextlib.suppress(AttributeError, RuntimeError, TypeError, ValueError):
        inputs = inputs.to(model.device)

    max_new_tokens = int(os.getenv("HUNYUAN_MAX_NEW_TOKENS", "2048"))
    do_sample = str(os.getenv("HUNYUAN_DO_SAMPLE", "false")).lower() in ("1", "true", "yes")

    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    decoded = processor.batch_decode(generated, skip_special_tokens=True)
    return decoded[0] if decoded else ""


def _should_clean_repeats() -> bool:
    return str(os.getenv("HUNYUAN_CLEAN_REPEATS", "true")).lower() in ("1", "true", "yes")


def _clean_repeated_substrings(text: str) -> str:
    if not text:
        return text
    # Safer heuristic: collapse consecutive duplicate lines (keeps first 2).
    lines = text.splitlines()
    if not lines:
        return text
    out = []
    prev = None
    repeat = 0
    for line in lines:
        if line == prev:
            repeat += 1
            if repeat <= 2:
                out.append(line)
        else:
            prev = line
            repeat = 1
            out.append(line)
    return "\n".join(out)


def _build_result_from_output(
    raw_output: str,
    output_format: str | None,
    prompt_preset: str | None,
    meta: dict[str, Any],
) -> OCRResult:
    fmt = normalize_ocr_format(output_format)
    if fmt == "unknown":
        preset = (prompt_preset or "").lower()
        fmt = "markdown" if preset in ("doc", "table") else "text"

    parsed = None
    if fmt == "json" or (prompt_preset or "").lower() == "json":
        parsed = _try_parse_json(raw_output)

    if parsed is not None:
        result = OCRResult(text=_extract_text_from_parsed(parsed), format="json", raw=parsed, meta=meta)
        _fill_blocks_tables_from_parsed(result, parsed)
        if not result.text:
            result.text = raw_output or ""
        return result

    return OCRResult(text=raw_output or "", format=fmt, raw=raw_output, meta=meta)


def _try_parse_json(raw_text: str) -> Any | None:
    if not raw_text:
        return None
    txt = raw_text.strip()
    if not txt:
        return None

    # Try full string first
    try:
        return json.loads(txt)
    except (TypeError, ValueError, json.JSONDecodeError):
        pass

    # Try to extract a JSON object or array from the output
    obj_start = txt.find("{")
    obj_end = txt.rfind("}")
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        try:
            return json.loads(txt[obj_start : obj_end + 1])
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    arr_start = txt.find("[")
    arr_end = txt.rfind("]")
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        try:
            return json.loads(txt[arr_start : arr_end + 1])
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    return None


def _extract_text_from_parsed(parsed: Any) -> str:
    if parsed is None:
        return ""
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, dict):
        for key in ("text", "content", "result", "output"):
            val = parsed.get(key)
            if isinstance(val, str) and val.strip():
                return val
        blocks = parsed.get("blocks") or parsed.get("items") or parsed.get("lines")
        if isinstance(blocks, list):
            parts = []
            for b in blocks:
                if isinstance(b, str):
                    parts.append(b)
                elif isinstance(b, dict) and isinstance(b.get("text"), str):
                    parts.append(b.get("text"))
            return "\n".join([p for p in parts if p])
    if isinstance(parsed, list):
        parts = []
        for item in parsed:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item.get("text"))
        return "\n".join([p for p in parts if p])
    return str(parsed)


def _fill_blocks_tables_from_parsed(result: OCRResult, parsed: Any) -> None:
    try:
        if isinstance(parsed, dict):
            blocks = parsed.get("blocks") or parsed.get("items") or parsed.get("lines")
            if isinstance(blocks, list):
                for b in blocks:
                    if isinstance(b, dict) and isinstance(b.get("text"), str):
                        result.blocks.append(
                            OCRBlock(
                                text=b.get("text"),
                                bbox=b.get("bbox"),
                                block_type=b.get("type") or b.get("block_type"),
                            )
                        )
                    elif isinstance(b, str):
                        result.blocks.append(OCRBlock(text=b))
            tables = parsed.get("tables")
            if isinstance(tables, list):
                for t in tables:
                    if isinstance(t, str):
                        result.tables.append(OCRTable(format="html", content=t))
                    elif isinstance(t, dict):
                        fmt = t.get("format") if isinstance(t.get("format"), str) else "html"
                        result.tables.append(OCRTable(format=fmt, content=t.get("content", t)))
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    result.blocks.append(
                        OCRBlock(
                            text=item.get("text"),
                            bbox=item.get("bbox"),
                            block_type=item.get("type") or item.get("block_type"),
                        )
                    )
                elif isinstance(item, str):
                    result.blocks.append(OCRBlock(text=item))
    except _HUNYUAN_NONCRITICAL_EXCEPTIONS:
        return
