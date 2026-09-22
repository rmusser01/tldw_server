from __future__ import annotations

import base64
import importlib.util
import io
import json
import contextlib
import os
import tempfile
import threading
from typing import Any
from urllib.parse import urlparse

from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.base import OCRBackend
from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.types import (
    OCRResult,
    normalize_ocr_format,
)
from tldw_Server_API.app.core.Utils.Utils import logging

_TF_MODEL = None
_TF_PROCESSOR = None
_TF_TOKENIZER = None
_TF_DEVICE = None
_TF_DTYPE = None
_TF_LOCK = threading.Lock()
_DOLPHIN_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)


def _resolve_mode() -> str:
    mode = (os.getenv("DOLPHIN_MODE") or "auto").lower()
    if mode in ("vllm", "trt", "trtllm", "trt-llm", "remote"):
        return "remote"
    if mode not in ("auto", "transformers", "remote"):
        return "auto"
    return mode


def _resolve_remote_mode() -> str:
    mode = (os.getenv("DOLPHIN_REMOTE_MODE") or "dolphin_trt").lower()
    if mode in ("openai", "oai", "openai_compat", "openai-compatible"):
        return "openai"
    if mode in ("dolphin_vllm", "vllm"):
        return "dolphin_vllm"
    if mode in ("dolphin_trt", "dolphin_trtllm", "dolphin_generate", "generate", "trt", "trtllm"):
        return "dolphin_trt"
    return "dolphin_trt"


_PROMPT_PRESETS: dict[str, str] = {
    "general": "Extract all visible text from the image.",
    "doc": "Parse the document and return all text in Markdown. Render tables as Markdown.",
    "table": "Extract tables as Markdown. Return all other text in Markdown.",
    "json": "Return JSON only. Format: an array of objects with fields: label, bbox, reading_order, text.",
}


class DolphinOCRBackend(OCRBackend):
    """
    Dolphin OCR backend supporting:
      - local Transformers inference (ByteDance/Dolphin-v2)
      - remote Dolphin generate API (vLLM or TRT-LLM server)
      - optional OpenAI-compatible endpoint

    Config via env:
      DOLPHIN_MODE: auto|transformers|remote
      DOLPHIN_PROMPT: override main prompt
      DOLPHIN_PROMPT_PRESET: general|doc|table|json
      DOLPHIN_JSON_PROMPT: override JSON prompt (empty disables)
      DOLPHIN_DISABLE_JSON: set to true to skip JSON pass

      Remote:
        DOLPHIN_URL
        DOLPHIN_REMOTE_MODE: dolphin_vllm|dolphin_trt|openai
        DOLPHIN_REMOTE_MODEL
        DOLPHIN_TIMEOUT
        DOLPHIN_USE_DATA_URL

      Transformers:
        DOLPHIN_MODEL_PATH
        DOLPHIN_DEVICE

      Generation:
        DOLPHIN_MAX_NEW_TOKENS
        DOLPHIN_MAX_LENGTH
        DOLPHIN_TEMPERATURE
        DOLPHIN_TOP_P
        DOLPHIN_TOP_K
        DOLPHIN_REPETITION_PENALTY
        DOLPHIN_DO_SAMPLE
        DOLPHIN_NUM_BEAMS
    """

    name = "dolphin"

    @classmethod
    def available(cls) -> bool:
        mode = _resolve_mode()
        if mode in ("auto", "remote") and os.getenv("DOLPHIN_URL"):
            return True
        if mode in ("auto", "transformers"):
            try:
                has_tf = importlib.util.find_spec("transformers") is not None
                has_torch = importlib.util.find_spec("torch") is not None
                has_pil = importlib.util.find_spec("PIL") is not None
                return bool(has_tf and has_torch and has_pil)
            except _DOLPHIN_NONCRITICAL_EXCEPTIONS:
                return False
        return False

    def describe(self) -> dict:
        mode = _resolve_mode()
        info: dict[str, Any] = {
            "mode": mode,
            "remote_mode": _resolve_remote_mode(),
            "prompt": os.getenv("DOLPHIN_PROMPT"),
            "prompt_preset": os.getenv("DOLPHIN_PROMPT_PRESET"),
            "json_prompt": os.getenv("DOLPHIN_JSON_PROMPT"),
        }
        if mode == "remote" or (mode == "auto" and os.getenv("DOLPHIN_URL")):
            info.update(
                {
                    "url": os.getenv("DOLPHIN_URL"),
                    "model": os.getenv("DOLPHIN_REMOTE_MODEL", "Dolphin"),
                    "timeout": int(os.getenv("DOLPHIN_TIMEOUT", "60")),
                    "use_data_url": os.getenv("DOLPHIN_USE_DATA_URL", "true"),
                }
            )
        else:
            info.update(
                {
                    "model_path": os.getenv("DOLPHIN_MODEL_PATH", "ByteDance/Dolphin-v2"),
                    "device": os.getenv("DOLPHIN_DEVICE"),
                }
            )
        return info

    def ocr_image(self, image_bytes: bytes, lang: str | None = None) -> str:
        result = self.ocr_image_structured(image_bytes, lang=lang, output_format="markdown")
        return result.text or ""

    def ocr_image_structured(
        self,
        image_bytes: bytes,
        lang: str | None = None,
        output_format: str | None = None,
        prompt_preset: str | None = None,
    ) -> OCRResult:
        if not self.available():
            logging.warning("DolphinOCRBackend not available: set DOLPHIN_URL or install transformers+torch+Pillow.")
            return OCRResult(text="", format="markdown")

        prompt = _resolve_prompt(prompt_preset, output_format)
        json_prompt = _resolve_json_prompt(prompt_preset, output_format)
        mode = _resolve_mode()
        remote_mode = _resolve_remote_mode()

        raw_markdown = _run_prompt(image_bytes, prompt)

        raw_json_text = ""
        parsed_json = None
        if json_prompt:
            raw_json_text = _run_prompt(image_bytes, json_prompt)
            parsed_json = _try_parse_json(raw_json_text)

        warnings = []
        fmt = normalize_ocr_format(output_format)
        if fmt == "unknown":
            fmt = "markdown"
        if fmt == "json" and parsed_json is None:
            warnings.append("JSON output requested but could not be parsed; returning markdown.")
            fmt = "markdown"

        meta = {
            "backend": self.name,
            "mode": mode,
            "remote_mode": remote_mode,
            "prompt_preset": prompt_preset or os.getenv("DOLPHIN_PROMPT_PRESET"),
            "output_format": output_format,
            "markdown_prompt": prompt,
            "json_prompt": json_prompt,
            "model": os.getenv("DOLPHIN_REMOTE_MODEL") if os.getenv("DOLPHIN_URL") else os.getenv("DOLPHIN_MODEL_PATH"),
        }

        raw_payload: Any = None
        if parsed_json is not None:
            raw_payload = parsed_json
        elif raw_json_text:
            raw_payload = {"raw_json": raw_json_text}

        primary_text = raw_markdown or ""
        if fmt == "json" and parsed_json is not None:
            primary_text = _extract_text_from_parsed(parsed_json) or raw_json_text or primary_text

        result = OCRResult(
            text=primary_text,
            format=fmt,
            raw=raw_payload,
            meta=meta,
            warnings=warnings,
        )
        if not result.text and parsed_json is not None:
            result.text = _extract_text_from_parsed(parsed_json)
        return result


def _resolve_prompt(prompt_preset: str | None, output_format: str | None) -> str:
    env_prompt = os.getenv("DOLPHIN_PROMPT")
    if env_prompt:
        return env_prompt

    preset = (prompt_preset or os.getenv("DOLPHIN_PROMPT_PRESET") or "").strip().lower()
    if not preset:
        fmt = normalize_ocr_format(output_format)
        if fmt == "json":
            preset = "json"
        elif fmt == "markdown":
            preset = "doc"
        else:
            preset = "doc"
    return _PROMPT_PRESETS.get(preset, _PROMPT_PRESETS["doc"])


def _resolve_json_prompt(prompt_preset: str | None, output_format: str | None) -> str:
    if _bool_env("DOLPHIN_DISABLE_JSON", False):
        return ""

    env_prompt = os.getenv("DOLPHIN_JSON_PROMPT")
    if env_prompt is not None:
        return env_prompt.strip()

    preset = (prompt_preset or "").strip().lower()
    if preset == "json":
        return _PROMPT_PRESETS["json"]

    fmt = normalize_ocr_format(output_format)
    if fmt == "json":
        return _PROMPT_PRESETS["json"]

    return _PROMPT_PRESETS["json"]


def _run_prompt(image_bytes: bytes, prompt: str) -> str:
    if not prompt:
        return ""

    mode = _resolve_mode()
    raw_text = ""

    if mode == "remote" or (mode == "auto" and os.getenv("DOLPHIN_URL")):
        try:
            remote_mode = _resolve_remote_mode()
            if remote_mode == "openai":
                raw_text = _ocr_via_openai(image_bytes, prompt)
            else:
                raw_text = _ocr_via_generate(image_bytes, prompt, remote_mode)
        except _DOLPHIN_NONCRITICAL_EXCEPTIONS as exc:
            logging.error(f"Dolphin remote path failed: {exc}", exc_info=True)

    if not raw_text:
        try:
            raw_text = _ocr_via_transformers(image_bytes, prompt)
        except _DOLPHIN_NONCRITICAL_EXCEPTIONS as exc:
            logging.error(f"Dolphin transformers path failed: {exc}", exc_info=True)
            raw_text = ""

    return raw_text or ""


def _ocr_via_generate(image_bytes: bytes, prompt: str, remote_mode: str) -> str:
    url = os.getenv("DOLPHIN_URL")
    if not url:
        raise RuntimeError("DOLPHIN_URL not set")
    url = url.rstrip("/")
    if not url.endswith("/generate"):
        url = f"{url}/generate"

    payload: dict[str, Any] = {
        "image_base64": base64.b64encode(image_bytes).decode("ascii"),
        "stream": False,
    }
    if remote_mode == "dolphin_vllm":
        payload["encoder_prompt"] = os.getenv("DOLPHIN_ENCODER_PROMPT", prompt)
        payload["decoder_prompt"] = os.getenv("DOLPHIN_DECODER_PROMPT", "<Answer/>")
    else:
        payload["prompt"] = os.getenv("DOLPHIN_DECODER_PROMPT", prompt)
    _apply_generation_params(payload)

    from tldw_Server_API.app.core.http_client import fetch_json

    timeout = int(os.getenv("DOLPHIN_TIMEOUT", "60"))
    data = fetch_json(method="POST", url=url, json=payload, timeout=timeout)

    text_out = data.get("text") if isinstance(data, dict) else None
    if isinstance(text_out, list):
        return str(text_out[0] or "")
    if isinstance(text_out, str):
        return text_out
    return ""


def _ocr_via_openai(image_bytes: bytes, prompt: str) -> str:
    url = os.getenv("DOLPHIN_URL")
    if not url:
        raise RuntimeError("DOLPHIN_URL not set")
    url = url.rstrip("/")

    model = os.getenv("DOLPHIN_REMOTE_MODEL", "Dolphin")
    timeout = int(os.getenv("DOLPHIN_TIMEOUT", "60"))
    use_data_url = _bool_env("DOLPHIN_USE_DATA_URL", True)
    if not use_data_url and not _is_local_url(url):
        logging.warning("DOLPHIN_USE_DATA_URL=false with non-local URL; forcing data URL for reliability.")
        use_data_url = True

    tmp_path = None
    if use_data_url:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        content_image = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
    else:
        # Path-based URLs are generally only useful for local file access
        # delete=False: the file must outlive this block, because the request below
        # references it by path. With delete=True it is unlinked at the with-exit and
        # the POST names a path that no longer exists.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(image_bytes)
            f.flush()
            tmp_path = f.name
        content_image = {"type": "image_url", "image_url": {"url": tmp_path}}

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
    }
    _apply_generation_params(data)

    from tldw_Server_API.app.core.http_client import fetch_json

    try:
        j = fetch_json(method="POST", url=url, json=data, timeout=timeout)
    finally:
        if tmp_path:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
    return (
        j.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    ) or ""


def _load_transformers():
    global _TF_MODEL, _TF_PROCESSOR, _TF_TOKENIZER, _TF_DEVICE, _TF_DTYPE
    if _TF_MODEL is not None and _TF_PROCESSOR is not None and _TF_TOKENIZER is not None:
        return _TF_MODEL, _TF_PROCESSOR, _TF_TOKENIZER, _TF_DEVICE, _TF_DTYPE
    with _TF_LOCK:
        if _TF_MODEL is not None and _TF_PROCESSOR is not None and _TF_TOKENIZER is not None:
            return _TF_MODEL, _TF_PROCESSOR, _TF_TOKENIZER, _TF_DEVICE, _TF_DTYPE

        import torch
        from transformers import AutoProcessor, VisionEncoderDecoderModel

        model_path = os.getenv("DOLPHIN_MODEL_PATH", "ByteDance/Dolphin-v2")
        model_revision = (os.getenv("DOLPHIN_MODEL_REVISION") or "").strip() or None
        device_env = os.getenv("DOLPHIN_DEVICE")
        device = device_env or ("cuda" if torch.cuda.is_available() else "cpu")

        dtype = torch.float16 if "cuda" in device else torch.float32

        _TF_MODEL = VisionEncoderDecoderModel.from_pretrained(
            model_path,
            revision=model_revision,
            trust_remote_code=True,
            torch_dtype=dtype,
        )  # nosec B615
        _TF_PROCESSOR = AutoProcessor.from_pretrained(  # nosec B615
            model_path,
            revision=model_revision,
            trust_remote_code=True,
        )
        _TF_TOKENIZER = getattr(_TF_PROCESSOR, "tokenizer", None)
        if _TF_TOKENIZER is None:
            from transformers import AutoTokenizer

            _TF_TOKENIZER = AutoTokenizer.from_pretrained(  # nosec B615
                model_path,
                revision=model_revision,
                trust_remote_code=True,
            )
        _TF_MODEL.to(device)
        _TF_MODEL.eval()

        _TF_DEVICE = device
        _TF_DTYPE = dtype
        return _TF_MODEL, _TF_PROCESSOR, _TF_TOKENIZER, _TF_DEVICE, _TF_DTYPE


def _ocr_via_transformers(image_bytes: bytes, prompt: str) -> str:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("PIL is required for Dolphin transformers mode") from exc

    model, processor, tokenizer, device, dtype = _load_transformers()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    prompt_text = f"<s>{prompt} <Answer/>"

    inputs = processor(images=[image], return_tensors="pt", padding=True)
    pixel_values = inputs.pixel_values.to(device)
    if dtype is not None and str(dtype).endswith("float16"):
        pixel_values = pixel_values.half()

    prompt_inputs = tokenizer([prompt_text], add_special_tokens=False, return_tensors="pt")
    prompt_input_ids = prompt_inputs["input_ids"].to(device)
    prompt_attention_mask = prompt_inputs["attention_mask"].to(device)

    generation_kwargs: dict[str, Any] = {
        "max_length": _getf("DOLPHIN_MAX_LENGTH", int, 4096),
        "max_new_tokens": _getf("DOLPHIN_MAX_NEW_TOKENS", int, 2048),
        "repetition_penalty": _getf("DOLPHIN_REPETITION_PENALTY", float, 1.0),
        "temperature": _getf("DOLPHIN_TEMPERATURE", float, 0.0),
        "top_p": _getf("DOLPHIN_TOP_P", float, 0.9),
        "top_k": _getf("DOLPHIN_TOP_K", int, 50),
        "do_sample": _getf("DOLPHIN_DO_SAMPLE", lambda x: str(x).lower() in ("1", "true", "yes"), False),
        "num_beams": _getf("DOLPHIN_NUM_BEAMS", int, 1),
    }

    outputs = model.generate(
        pixel_values,
        decoder_input_ids=prompt_input_ids,
        decoder_attention_mask=prompt_attention_mask,
        **generation_kwargs,
    )

    sequences = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    if not sequences:
        return ""

    sequence = sequences[0]
    sequence = sequence.replace("<pad>", "").replace("</s>", "").strip()
    if prompt_text in sequence:
        sequence = sequence.replace(prompt_text, "").strip()
    return sequence


def _apply_generation_params(payload: dict[str, Any]) -> None:
    payload["max_new_tokens"] = _getf("DOLPHIN_MAX_NEW_TOKENS", int, 2048)
    if "messages" in payload:
        payload["max_tokens"] = payload["max_new_tokens"]
    payload["temperature"] = _getf("DOLPHIN_TEMPERATURE", float, 0.0)
    payload["top_p"] = _getf("DOLPHIN_TOP_P", float, 0.9)
    top_k = _getf_optional("DOLPHIN_TOP_K", int)
    if top_k is not None:
        payload["top_k"] = top_k


def _getf(env: str, cast, default):
    try:
        return cast(os.getenv(env, str(default)))
    except (TypeError, ValueError):
        return default


def _getf_optional(env: str, cast):
    val = os.getenv(env)
    if val is None or val == "":
        return None
    try:
        return cast(val)
    except (TypeError, ValueError):
        return None


def _bool_env(env: str, default: bool) -> bool:
    val = os.getenv(env)
    if val is None:
        return default
    return str(val).lower() in ("1", "true", "yes")


def _is_local_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        return host in ("localhost", "127.0.0.1", "::1")
    except (AttributeError, TypeError, ValueError):
        return False


def _try_parse_json(raw_text: str) -> Any | None:
    if not raw_text:
        return None
    txt = raw_text.strip()
    if not txt:
        return None

    try:
        return json.loads(txt)
    except (TypeError, ValueError, json.JSONDecodeError):
        pass

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
        items = parsed.get("blocks") or parsed.get("items") or parsed.get("lines") or parsed.get("elements")
        if isinstance(items, list):
            parts = []
            for item in items:
                if isinstance(item, dict):
                    t = item.get("text") or item.get("content")
                    if isinstance(t, str):
                        parts.append(t)
            if parts:
                return "\n".join(parts)
        return ""
    if isinstance(parsed, list):
        parts = []
        for item in parsed:
            if isinstance(item, dict):
                t = item.get("text") or item.get("content")
                if isinstance(t, str):
                    parts.append(t)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""
