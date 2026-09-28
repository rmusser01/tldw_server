from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import json
import os
import re
import threading
from typing import Any

from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.base import OCRBackend
from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.types import (
    OCRResult,
    normalize_ocr_format,
)
from tldw_Server_API.app.core.Utils.Utils import logging
from tldw_Server_API.app.core.Utils.coercion import env_bool
from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.runtime_support import image_payload

_DEFAULT_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown>"
_MIN_W = 1024
_MIN_H = 1280
_MAX_W = 1648
_MAX_H = 2048

_TF_MODEL = None
_TF_PROCESSOR = None
_TF_LOCK = threading.Lock()

_POSTPROCESS_FUNCS: dict[str, Any] | None = None

_NEMOTRON_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
    json.JSONDecodeError,
)


def _resolve_mode() -> str:
    mode = (os.getenv("NEMOTRON_MODE") or "auto").lower()
    if mode not in ("auto", "vllm", "transformers"):
        mode = "auto"
    return mode


def _env_bool(name: str, default: bool) -> bool:
    return env_bool(name, default=default)


def _resolve_skip_special_tokens(
    output_format: str | None,
    prompt_preset: str | None,
) -> bool:
    env_val = os.getenv("NEMOTRON_SKIP_SPECIAL_TOKENS")
    if env_val is not None:
        return _env_bool("NEMOTRON_SKIP_SPECIAL_TOKENS", False)
    # Default to keeping special tokens so structured tags remain available.
    return False


class NemotronParseBackend(OCRBackend):
    """
    NVIDIA Nemotron-Parse OCR backend with dual modes:
      - vLLM (OpenAI-compatible chat endpoint)
      - Transformers (local HF model with trust_remote_code)

    Plain-text output is returned for OCR; structured layout data is
    returned via ocr_image_structured for downstream UI usage.
    """

    name = "nemotron_parse"

    @classmethod
    def available(cls) -> bool:
        mode = _resolve_mode()
        if mode in ("auto", "vllm") and os.getenv("NEMOTRON_VLLM_URL"):
            return True
        if mode in ("auto", "transformers"):
            try:
                has_tf = importlib.util.find_spec("transformers") is not None
                has_torch = importlib.util.find_spec("torch") is not None
                has_pil = importlib.util.find_spec("PIL") is not None
                return bool(has_tf and has_torch and has_pil)
            except _NEMOTRON_NONCRITICAL_EXCEPTIONS:
                return False
        return False

    def describe(self) -> dict:
        mode = _resolve_mode()
        info = {
            "mode": mode,
            "prompt": os.getenv("NEMOTRON_PROMPT", _DEFAULT_PROMPT),
            "text_format": os.getenv("NEMOTRON_TEXT_FORMAT", "plain"),
            "table_format": os.getenv("NEMOTRON_TABLE_FORMAT", "latex"),
        }
        if mode == "vllm" or (mode == "auto" and os.getenv("NEMOTRON_VLLM_URL")):
            info.update({
                "url": os.getenv("NEMOTRON_VLLM_URL"),
                "model": os.getenv("NEMOTRON_VLLM_MODEL", "model"),
                "timeout": int(os.getenv("NEMOTRON_VLLM_TIMEOUT", "60")),
                "use_data_url": os.getenv("NEMOTRON_VLLM_USE_DATA_URL", "true"),
            })
        else:
            info.update({
                "model_path": os.getenv("NEMOTRON_MODEL_PATH", "nvidia/NVIDIA-Nemotron-Parse-v1.1"),
                "device": os.getenv("NEMOTRON_DEVICE"),
                "dtype": os.getenv("NEMOTRON_DTYPE"),
            })
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
            logging.warning("NemotronParseBackend not available: set NEMOTRON_VLLM_URL or install transformers+torch.")
            return OCRResult(text="", format="text")

        prompt = os.getenv("NEMOTRON_PROMPT", _DEFAULT_PROMPT)
        mode = _resolve_mode()

        raw = ""
        skip_special = _resolve_skip_special_tokens(output_format, prompt_preset)
        if mode == "vllm" or (mode == "auto" and os.getenv("NEMOTRON_VLLM_URL")):
            try:
                raw = _ocr_via_vllm(_prepare_image_bytes(image_bytes), prompt, skip_special)
            except _NEMOTRON_NONCRITICAL_EXCEPTIONS as e:
                logging.error(f"Nemotron vLLM path failed: {e}", exc_info=True)
                # fall back to transformers if available

        if not raw:
            try:
                raw = _ocr_via_transformers(_prepare_image_bytes(image_bytes), prompt, skip_special)
            except _NEMOTRON_NONCRITICAL_EXCEPTIONS as e:
                logging.error(f"Nemotron transformers path failed: {e}", exc_info=True)
                return OCRResult(text="", format="text")

        fmt_input = output_format
        if not fmt_input:
            env_fmt = os.getenv("NEMOTRON_TEXT_FORMAT")
            if env_fmt:
                env_fmt = str(env_fmt).strip().lower()
                if env_fmt == "plain":
                    env_fmt = "text"
                fmt_input = env_fmt
        fmt = normalize_ocr_format(fmt_input)
        if fmt == "unknown":
            fmt = "text"
        text_format = "plain"
        if fmt == "markdown":
            text_format = "markdown"
        elif fmt == "html":
            text_format = "html"
        elif fmt == "json":
            text_format = "json"

        structured = _postprocess_output(raw, text_format=text_format)
        text_out = structured.get("text") if isinstance(structured, dict) else None
        if not text_out:
            text_out = _strip_tags(raw)
            if isinstance(structured, dict):
                structured.setdefault("text", text_out)

        meta = {
            "backend": self.name,
            "mode": mode,
            "prompt_preset": prompt_preset,
            "output_format": output_format,
            "model": os.getenv("NEMOTRON_VLLM_MODEL") if os.getenv("NEMOTRON_VLLM_URL") else os.getenv("NEMOTRON_MODEL_PATH"),
        }

        return OCRResult(
            text=text_out or "",
            format=fmt,
            raw=structured,
            meta=meta,
        )


def _prepare_image_bytes(image_bytes: bytes) -> bytes:
    mode = (os.getenv("NEMOTRON_RESIZE_MODE") or "fit_max").lower()
    if mode not in ("fit_max", "fit_range", "none"):
        mode = "fit_max"
    if mode == "none":
        return image_bytes

    try:
        from PIL import Image
    except (ImportError, OSError):
        return image_bytes

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            w, h = img.size
            scale = 1.0
            if mode in ("fit_max", "fit_range"):
                max_scale = min(_MAX_W / max(w, 1), _MAX_H / max(h, 1), 1.0)
                scale = max_scale
                if mode == "fit_range" and (w < _MIN_W or h < _MIN_H):
                    min_scale = max(_MIN_W / max(w, 1), _MIN_H / max(h, 1))
                    scale = max(scale, min_scale)

            if abs(scale - 1.0) < 1e-3:
                return image_bytes

            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            resized = img.convert("RGB").resize((new_w, new_h), Image.LANCZOS)
            out = io.BytesIO()
            resized.save(out, format="PNG")
            return out.getvalue()
    except _NEMOTRON_NONCRITICAL_EXCEPTIONS:
        return image_bytes


def _postprocess_output(raw: str, text_format: str = "plain") -> dict[str, Any]:
    structured: dict[str, Any] = {}
    keep_raw = _env_bool("NEMOTRON_KEEP_RAW_OUTPUT", True)
    if keep_raw:
        structured["raw_output"] = raw

    data = _try_parse_json(raw)
    if data is not None:
        structured["parsed_json"] = data

    funcs = _load_postprocess_funcs()
    if funcs.get("postprocess"):
        table_format = os.getenv("NEMOTRON_TABLE_FORMAT", "latex")
        try:
            structured["text"] = funcs["postprocess"](
                raw, text_format=text_format, table_format=table_format
            )
        except TypeError:
            with contextlib.suppress(_NEMOTRON_NONCRITICAL_EXCEPTIONS):
                structured["text"] = funcs["postprocess"](raw, text_format, table_format)
        except _NEMOTRON_NONCRITICAL_EXCEPTIONS:
            pass

        extractor = funcs.get("extract")
        if callable(extractor):
            try:
                classes, bboxes = extractor(raw)
                structured["classes"] = classes
                structured["bboxes"] = bboxes
            except _NEMOTRON_NONCRITICAL_EXCEPTIONS:
                pass

        if structured.get("text"):
            structured["parser"] = "nemotron_helpers"

    if "parser" not in structured:
        structured["parser"] = "raw"

    return structured


def _load_postprocess_funcs() -> dict[str, Any]:
    global _POSTPROCESS_FUNCS
    if _POSTPROCESS_FUNCS is not None:
        return _POSTPROCESS_FUNCS

    candidates = []
    env_mod = os.getenv("NEMOTRON_POSTPROCESSOR_MODULE")
    if env_mod:
        candidates.append(env_mod)
    candidates.extend([
        "nemotron_parse.postprocess",
        "nemotron_parse",
        "nvidia_nemotron_parse.postprocess",
        "nvidia_nemotron_parse",
    ])

    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        extract = getattr(mod, "extract_classes_bboxes", None)
        postprocess = getattr(mod, "postprocess_text", None)
        if callable(postprocess):
            _POSTPROCESS_FUNCS = {"extract": extract, "postprocess": postprocess}
            return _POSTPROCESS_FUNCS

    _POSTPROCESS_FUNCS = {}
    return _POSTPROCESS_FUNCS


def _try_parse_json(raw: str) -> Any | None:
    text = (raw or "").strip()
    if not text:
        return None
    if not (text.startswith("{") or text.startswith("[")):
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _strip_tags(raw: str) -> str:
    if not raw:
        return ""
    lines = []
    for line in raw.splitlines():
        line = re.sub(r"<[^>]+>", "", line)
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _ocr_via_vllm(image_bytes: bytes, prompt: str, skip_special_tokens: bool) -> str:
    url = os.getenv("NEMOTRON_VLLM_URL", "").rstrip("/")
    model = os.getenv("NEMOTRON_VLLM_MODEL", "model")
    timeout = int(os.getenv("NEMOTRON_VLLM_TIMEOUT", "60"))
    use_data_url = _env_bool("NEMOTRON_VLLM_USE_DATA_URL", True)

    def _getf(env, cast, default):
        try:
            return cast(os.getenv(env, str(default)))
        except (TypeError, ValueError):
            return default

    with image_payload(image_bytes, use_data_url=use_data_url) as content_image:
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
            "max_new_tokens": _getf("NEMOTRON_MAX_NEW_TOKENS", int, 2048),
            "temperature": _getf("NEMOTRON_TEMPERATURE", float, 0.1),
            "repetition_penalty": _getf("NEMOTRON_REPETITION_PENALTY", float, 1.05),
            "top_p": _getf("NEMOTRON_TOP_P", float, 0.8),
            "top_k": _getf("NEMOTRON_TOP_K", int, 20),
            "do_sample": env_bool("NEMOTRON_DO_SAMPLE", default=True),
            "skip_special_tokens": skip_special_tokens,
        }

        from tldw_Server_API.app.core.http_client import fetch_json
        j = fetch_json(method="POST", url=url, json=data, timeout=timeout)
    choice = (j.get("choices") or [{}])[0]
    content = (
        choice.get("message", {}).get("content")
        or choice.get("text")
        or ""
    )
    return content


def _load_transformers():
    global _TF_MODEL, _TF_PROCESSOR
    if _TF_MODEL is not None and _TF_PROCESSOR is not None:
        return _TF_MODEL, _TF_PROCESSOR

    with _TF_LOCK:
        if _TF_MODEL is not None and _TF_PROCESSOR is not None:
            return _TF_MODEL, _TF_PROCESSOR

        import torch
        from transformers import AutoModel, AutoProcessor

        model_path = os.getenv("NEMOTRON_MODEL_PATH", "nvidia/NVIDIA-Nemotron-Parse-v1.1")
        model_revision = (os.getenv("NEMOTRON_MODEL_REVISION") or "").strip() or None
        device_env = os.getenv("NEMOTRON_DEVICE")
        device_map = device_env or "auto"

        dtype_env = (os.getenv("NEMOTRON_DTYPE") or "").lower()
        if dtype_env in ("fp16", "float16"):
            dtype = torch.float16
        elif dtype_env in ("bf16", "bfloat16"):
            dtype = torch.bfloat16
        elif dtype_env in ("fp32", "float32"):
            dtype = torch.float32
        else:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        _TF_MODEL = AutoModel.from_pretrained(
            model_path,
            revision=model_revision,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device_map,
        )  # nosec B615
        _TF_PROCESSOR = AutoProcessor.from_pretrained(  # nosec B615
            model_path,
            revision=model_revision,
            trust_remote_code=True,
        )
        return _TF_MODEL, _TF_PROCESSOR


def _ocr_via_transformers(image_bytes: bytes, prompt: str, skip_special_tokens: bool) -> str:
    model, processor = _load_transformers()
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError("Pillow is required for transformers mode (PIL).") from None

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = _build_transformer_inputs(processor, image, prompt)

    import torch
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(model.device)

    def _getf(env, cast, default):
        try:
            return cast(os.getenv(env, str(default)))
        except (TypeError, ValueError):
            return default

    gen_kwargs = {
        "max_new_tokens": _getf("NEMOTRON_MAX_NEW_TOKENS", int, 2048),
        "temperature": _getf("NEMOTRON_TEMPERATURE", float, 0.1),
        "repetition_penalty": _getf("NEMOTRON_REPETITION_PENALTY", float, 1.05),
        "top_p": _getf("NEMOTRON_TOP_P", float, 0.8),
        "top_k": _getf("NEMOTRON_TOP_K", int, 20),
        "do_sample": env_bool("NEMOTRON_DO_SAMPLE", default=True),
    }

    output = model.generate(**inputs, **gen_kwargs)
    try:
        text = processor.batch_decode(output, skip_special_tokens=skip_special_tokens)[0]
    except _NEMOTRON_NONCRITICAL_EXCEPTIONS:
        text = processor.decode(output[0], skip_special_tokens=skip_special_tokens)
    return text or ""


def _build_transformer_inputs(processor, image, prompt: str):
    if hasattr(processor, "apply_chat_template"):
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image},
                    ],
                }
            ]
            text_input = processor.apply_chat_template(messages, add_generation_prompt=True)
            return processor(images=image, text=text_input, return_tensors="pt")
        except _NEMOTRON_NONCRITICAL_EXCEPTIONS:
            pass
    return processor(images=image, text=prompt, return_tensors="pt")
