from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import threading

from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.base import OCRBackend
from tldw_Server_API.app.core.Utils.Utils import logging

_TF_MODEL = None
_TF_TOKENIZER = None
_TF_IMAGE_PROCESSOR = None
_TF_LOCK = threading.Lock()
_POINTS_RUNTIME_EXCEPTIONS = (
    ConnectionError,
    ImportError,
    ModuleNotFoundError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)

_WEPOINTS_MODULE = "wepoints"


def _resolve_mode() -> str:
    mode = (os.getenv("POINTS_MODE") or "auto").lower()
    if mode not in ("auto", "sglang", "transformers"):
        mode = "auto"
    return mode


class PointsReaderBackend(OCRBackend):
    """
    POINTS-Reader backend with two modes:
      - Transformers (local HF model: tencent/POINTS-Reader)
      - SGLang server (OpenAI-compatible /v1/chat/completions)

    Configure via env:
      - POINTS_MODE: 'auto' (default), 'sglang', or 'transformers'
      - POINTS_PROMPT: override default extraction prompt
      - Transformers mode:
          POINTS_MODEL_PATH: HF model id or local path (default: 'tencent/POINTS-Reader')
      - SGLang mode:
          POINTS_SGLANG_URL: server URL (default: 'http://127.0.0.1:8081/v1/chat/completions')
          POINTS_SGLANG_MODEL: model param for request (default: 'WePoints')
          Optional generation params: POINTS_MAX_NEW_TOKENS, POINTS_TEMPERATURE,
          POINTS_REPETITION_PENALTY, POINTS_TOP_P, POINTS_TOP_K, POINTS_DO_SAMPLE
    """

    name = "points"

    @classmethod
    def available(cls) -> bool:
        mode = _resolve_mode()
        if mode in ("auto", "sglang"):
            # Require a URL to be configured for availability in SGLang path
            url = os.getenv("POINTS_SGLANG_URL")
            if url:
                return True
        if mode in ("auto", "transformers"):
            try:
                has_tf = importlib.util.find_spec("transformers") is not None
                has_torch = importlib.util.find_spec("torch") is not None
                has_wepoints = importlib.util.find_spec(_WEPOINTS_MODULE) is not None
                if has_tf and has_torch and not has_wepoints:
                    logging.warning("PointsReaderBackend transformers mode unavailable: WePOINTS missing")
                return bool(has_tf and has_torch and has_wepoints)
            except (AttributeError, ImportError, ValueError):
                return False
        return False

    def describe(self) -> dict:
        mode = _resolve_mode()
        info = {
            "mode": mode,
            "prompt": os.getenv("POINTS_PROMPT"),
        }
        if mode == "sglang" or (mode == "auto" and os.getenv("POINTS_SGLANG_URL")):
            info.update({
                "url": os.getenv("POINTS_SGLANG_URL"),
                "model": os.getenv("POINTS_SGLANG_MODEL", "WePoints"),
                "timeout": int(os.getenv("POINTS_SGLANG_TIMEOUT", "60")),
                "use_data_url": os.getenv("POINTS_SGLANG_USE_DATA_URL", "false"),
            })
        else:
            info.update({
                "model_path": os.getenv("POINTS_MODEL_PATH", "tencent/POINTS-Reader"),
                "device": os.getenv("POINTS_DEVICE"),
            })
        return info

    def ocr_image(self, image_bytes: bytes, lang: str | None = None) -> str:
        if not self.available():
            logging.warning(
                "PointsReaderBackend not available: install 'transformers'+'torch' plus WePOINTS, "
                "or configure POINTS_SGLANG_URL for SGLang"
            )
            return ""

        prompt = os.getenv("POINTS_PROMPT") or (
            "Please extract all the text from the image with the following requirements:\n"
            "1. Return tables in HTML format.\n"
            "2. Return all other text in Markdown format."
        )

        with tempfile.TemporaryDirectory(prefix="points_reader_") as tmpdir:
            img_path = os.path.join(tmpdir, "page.png")
            try:
                with open(img_path, "wb") as f:
                    f.write(image_bytes)
            except OSError as e:
                logging.error(f"PointsReaderBackend: failed to write temp image: {e}", exc_info=True)
                return ""

            mode = _resolve_mode()
            # Prefer SGLang when explicitly chosen or when auto and URL is set
            if mode == "sglang" or (mode == "auto" and os.getenv("POINTS_SGLANG_URL")):
                try:
                    return _ocr_via_sglang(img_path, prompt)
                except _POINTS_RUNTIME_EXCEPTIONS as e:
                    logging.error(f"POINTS SGLang path failed: {e}", exc_info=True)
                    # fall through to transformers if available

            try:
                return _ocr_via_transformers(img_path, prompt)
            except _POINTS_RUNTIME_EXCEPTIONS as e:
                if isinstance(e, (ImportError, ModuleNotFoundError)) and "wepoints" in str(e).lower():
                    logging.error(f"POINTS Transformers path failed: WePOINTS missing: {e}", exc_info=True)
                else:
                    logging.error(f"POINTS Transformers path failed: {e}", exc_info=True)
                return ""


def _ocr_via_sglang(image_path: str, prompt: str) -> str:
    import base64

    url = os.getenv("POINTS_SGLANG_URL", "http://127.0.0.1:8081/v1/chat/completions")
    model = os.getenv("POINTS_SGLANG_MODEL", "WePoints")
    timeout = int(os.getenv("POINTS_SGLANG_TIMEOUT", "60"))
    use_data_url = str(os.getenv("POINTS_SGLANG_USE_DATA_URL", "false")).lower() in ("1","true","yes")

    def _getf(env, cast, default):
        try:
            return cast(os.getenv(env, str(default)))
        except (TypeError, ValueError):
            return default

    content_image = None
    if use_data_url:
        with open(image_path, "rb") as f:
            b = f.read()
        b64 = base64.b64encode(b).decode("ascii")
        content_image = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
    else:
        content_image = {"type": "image_url", "image_url": {"url": image_path}}

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
        "max_new_tokens": _getf("POINTS_MAX_NEW_TOKENS", int, 2048),
        "temperature": _getf("POINTS_TEMPERATURE", float, 0.7),
        "repetition_penalty": _getf("POINTS_REPETITION_PENALTY", float, 1.05),
        "top_p": _getf("POINTS_TOP_P", float, 0.8),
        "top_k": _getf("POINTS_TOP_K", int, 20),
        "do_sample": _getf("POINTS_DO_SAMPLE", lambda x: str(x).lower() in ("1","true","yes"), True),
    }

    from tldw_Server_API.app.core.http_client import fetch_json
    j = fetch_json(method="POST", url=url, json=data, timeout=timeout)

    # OpenAI-compatible shape
    content = (
        j.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    return content or ""


def _load_transformers():
    global _TF_MODEL, _TF_TOKENIZER, _TF_IMAGE_PROCESSOR
    if _TF_MODEL is not None and _TF_TOKENIZER is not None and _TF_IMAGE_PROCESSOR is not None:
        return _TF_MODEL, _TF_TOKENIZER, _TF_IMAGE_PROCESSOR
    with _TF_LOCK:
        if _TF_MODEL is not None and _TF_TOKENIZER is not None and _TF_IMAGE_PROCESSOR is not None:
            return _TF_MODEL, _TF_TOKENIZER, _TF_IMAGE_PROCESSOR
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2VLImageProcessor

        model_path = os.getenv("POINTS_MODEL_PATH", "tencent/POINTS-Reader")
        model_revision = (os.getenv("POINTS_MODEL_REVISION") or "").strip() or None
        # device and dtype selection
        device_env = os.getenv("POINTS_DEVICE")
        device_map = device_env or "auto"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        _TF_MODEL = AutoModelForCausalLM.from_pretrained(
            model_path,
            revision=model_revision,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device_map,
        )  # nosec B615
        _TF_TOKENIZER = AutoTokenizer.from_pretrained(  # nosec B615
            model_path,
            revision=model_revision,
            trust_remote_code=True,
        )
        _TF_IMAGE_PROCESSOR = Qwen2VLImageProcessor.from_pretrained(  # nosec B615
            model_path,
            revision=model_revision,
        )
        return _TF_MODEL, _TF_TOKENIZER, _TF_IMAGE_PROCESSOR


def _ocr_via_transformers(image_path: str, prompt: str) -> str:
    model, tokenizer, image_processor = _load_transformers()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    def _getf(env, cast, default):
        try:
            return cast(os.getenv(env, str(default)))
        except (TypeError, ValueError):
            return default

    generation_config = {
        "max_new_tokens": _getf("POINTS_MAX_NEW_TOKENS", int, 2048),
        "repetition_penalty": _getf("POINTS_REPETITION_PENALTY", float, 1.05),
        "temperature": _getf("POINTS_TEMPERATURE", float, 0.7),
        "top_p": _getf("POINTS_TOP_P", float, 0.8),
        "top_k": _getf("POINTS_TOP_K", int, 20),
        "do_sample": _getf("POINTS_DO_SAMPLE", lambda x: str(x).lower() in ("1","true","yes"), True),
    }

    try:
        response = model.chat(messages, tokenizer, image_processor, generation_config)
        # response is expected to be a string
        if not isinstance(response, str):
            try:
                return json.dumps(response)
            except (TypeError, ValueError):
                return str(response)
        return response
    except _POINTS_RUNTIME_EXCEPTIONS as e:
        logging.error(f"POINTS local inference failed: {e}", exc_info=True)
        return ""
