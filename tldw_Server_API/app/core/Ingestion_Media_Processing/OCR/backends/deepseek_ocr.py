from __future__ import annotations

import importlib.util
import os
import tempfile
import threading

from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.base import OCRBackend
from tldw_Server_API.app.core.Utils.Utils import logging
from tldw_Server_API.app.core.Utils.coercion import env_bool

_TF_MODEL = None
_TF_TOKENIZER = None
_TF_LOCK = threading.Lock()

_DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."

_DEEPSEEK_COERCE_EXCEPTIONS = (
    TypeError,
    ValueError,
    OverflowError,
)

_DEEPSEEK_IMPORT_EXCEPTIONS = (
    ImportError,
    OSError,
    RuntimeError,
    AttributeError,
)

_DEEPSEEK_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ImportError,
)


def _env_bool(name: str, default: bool) -> bool:
    return env_bool(name, default=default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except _DEEPSEEK_COERCE_EXCEPTIONS:
        return default


def _resolve_device() -> str:
    device = (os.getenv("DEEPSEEK_OCR_DEVICE") or "cuda").strip().lower()
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except _DEEPSEEK_IMPORT_EXCEPTIONS:
            return "cpu"
    return device


def _resolve_dtype():
    dtype_name = (os.getenv("DEEPSEEK_OCR_DTYPE") or "bfloat16").strip().lower()
    try:
        import torch

        mapping = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return mapping.get(dtype_name, torch.bfloat16)
    except _DEEPSEEK_IMPORT_EXCEPTIONS:
        return None


def _resolve_attn_impl(device: str | None = None) -> str:
    env_val = os.getenv("DEEPSEEK_OCR_ATTN_IMPL")
    attn_impl = (env_val or "flash_attention_2").strip()
    if env_val is None and device and not device.startswith("cuda") and attn_impl == "flash_attention_2":
        return "eager"
    return attn_impl


def _resolve_prompt() -> str:
    return os.getenv("DEEPSEEK_OCR_PROMPT") or _DEFAULT_PROMPT


def _resolve_sizes() -> tuple[int, int, bool]:
    base_size = _env_int("DEEPSEEK_OCR_BASE_SIZE", 1024)
    image_size = _env_int("DEEPSEEK_OCR_IMAGE_SIZE", 640)
    crop_mode = _env_bool("DEEPSEEK_OCR_CROP_MODE", True)
    return base_size, image_size, crop_mode


def _resolve_output_dir(save_results: bool, default_dir: str) -> str:
    if not save_results:
        return default_dir

    base_dir = os.getenv("DEEPSEEK_OCR_OUTPUT_DIR")
    if not base_dir:
        logging.warning(
            "DEEPSEEK_OCR_SAVE_RESULTS enabled but DEEPSEEK_OCR_OUTPUT_DIR not set; "
            "results will be stored in a temporary directory."
        )
        return default_dir

    try:
        os.makedirs(base_dir, exist_ok=True)
    except OSError as exc:
        logging.warning(
            f"DeepSeek OCR: failed to create DEEPSEEK_OCR_OUTPUT_DIR '{base_dir}': {exc}. "
            "Falling back to temporary directory."
        )
        return default_dir

    try:
        return tempfile.mkdtemp(prefix="deepseek_ocr_", dir=base_dir)
    except (OSError, ValueError):
        return base_dir


class DeepSeekOCRBackend(OCRBackend):
    """
    DeepSeek OCR backend (Transformers-only).

    Runs local inference using the DeepSeek-OCR `model.infer(...)` API.
    """

    name = "deepseek"

    @classmethod
    def available(cls) -> bool:
        try:
            has_tf = importlib.util.find_spec("transformers") is not None
            has_torch = importlib.util.find_spec("torch") is not None
            if not (has_tf and has_torch):
                return False

            device = _resolve_device()
            if device.startswith("cuda"):
                try:
                    import torch

                    if not torch.cuda.is_available():
                        return False
                except _DEEPSEEK_IMPORT_EXCEPTIONS:
                    return False

            attn_impl = _resolve_attn_impl(device)
            if device.startswith("cuda") and attn_impl == "flash_attention_2":
                if importlib.util.find_spec("flash_attn") is None:
                    return False
            return True
        except _DEEPSEEK_NONCRITICAL_EXCEPTIONS:
            return False

    def describe(self) -> dict:
        base_size, image_size, crop_mode = _resolve_sizes()
        device = _resolve_device()
        attn_impl = _resolve_attn_impl(device)
        return {
            "model_id": os.getenv("DEEPSEEK_OCR_MODEL_ID", "deepseek-ai/DeepSeek-OCR"),
            "prompt": _resolve_prompt(),
            "base_size": base_size,
            "image_size": image_size,
            "crop_mode": crop_mode,
            "save_results": _env_bool("DEEPSEEK_OCR_SAVE_RESULTS", False),
            "test_compress": _env_bool("DEEPSEEK_OCR_TEST_COMPRESS", False),
            "dtype": os.getenv("DEEPSEEK_OCR_DTYPE", "bfloat16"),
            "attn_impl": attn_impl,
            "device": device,
            "output_dir": os.getenv("DEEPSEEK_OCR_OUTPUT_DIR"),
        }

    def ocr_image(self, image_bytes: bytes, lang: str | None = None) -> str:
        if not self.available():
            logging.warning(
                "DeepSeekOCRBackend not available: install transformers+torch and ensure GPU/FlashAttention if required."
            )
            return ""

        try:
            model, tokenizer = _load_transformers()
        except _DEEPSEEK_NONCRITICAL_EXCEPTIONS as exc:
            logging.error(f"DeepSeek OCR model load failed: {exc}", exc_info=True)
            return ""

        prompt = _resolve_prompt()
        base_size, image_size, crop_mode = _resolve_sizes()
        save_results = _env_bool("DEEPSEEK_OCR_SAVE_RESULTS", False)
        test_compress = _env_bool("DEEPSEEK_OCR_TEST_COMPRESS", False)

        with tempfile.TemporaryDirectory(prefix="deepseek_ocr_") as tmpdir:
            img_path = os.path.join(tmpdir, "page.png")
            try:
                with open(img_path, "wb") as f:
                    f.write(image_bytes)
            except (OSError, ValueError, TypeError) as exc:
                logging.error(f"DeepSeek OCR: failed to write temp image: {exc}", exc_info=True)
                return ""

            output_path = _resolve_output_dir(save_results, os.path.join(tmpdir, "output"))
            if output_path.startswith(tmpdir):
                try:
                    os.makedirs(output_path, exist_ok=True)
                except OSError:
                    output_path = tmpdir

            try:
                result = model.infer(
                    tokenizer,
                    prompt=prompt,
                    image_file=img_path,
                    output_path=output_path,
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                    save_results=save_results,
                    test_compress=test_compress,
                )
            except _DEEPSEEK_NONCRITICAL_EXCEPTIONS as exc:
                logging.error(f"DeepSeek OCR inference failed: {exc}", exc_info=True)
                return ""

        return _extract_text_from_any(result)


def _load_transformers():
    global _TF_MODEL, _TF_TOKENIZER
    if _TF_MODEL is not None and _TF_TOKENIZER is not None:
        return _TF_MODEL, _TF_TOKENIZER

    with _TF_LOCK:
        if _TF_MODEL is not None and _TF_TOKENIZER is not None:
            return _TF_MODEL, _TF_TOKENIZER

        from transformers import AutoModel, AutoTokenizer

        model_id = os.getenv("DEEPSEEK_OCR_MODEL_ID", "deepseek-ai/DeepSeek-OCR")
        model_revision = (os.getenv("DEEPSEEK_OCR_MODEL_REVISION") or "").strip() or None
        device = _resolve_device()
        attn_impl = _resolve_attn_impl(device)

        _TF_TOKENIZER = AutoTokenizer.from_pretrained(  # nosec B615
            model_id,
            revision=model_revision,
            trust_remote_code=True,
        )
        dtype = _resolve_dtype()
        model_kwargs = {
            "trust_remote_code": True,
            "_attn_implementation": attn_impl,
        }
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        try:
            _TF_MODEL = AutoModel.from_pretrained(  # nosec B615
                model_id,
                revision=model_revision,
                use_safetensors=True,
                **model_kwargs,
            )
        except _DEEPSEEK_NONCRITICAL_EXCEPTIONS as exc:
            logging.warning(
                f"DeepSeek OCR: safetensors load failed ({exc}); retrying with use_safetensors=False."
            )
            _TF_MODEL = AutoModel.from_pretrained(  # nosec B615
                model_id,
                revision=model_revision,
                use_safetensors=False,
                **model_kwargs,
            )

        try:
            if device.startswith("cuda"):
                _TF_MODEL = _TF_MODEL.cuda()
            elif device:
                _TF_MODEL = _TF_MODEL.to(device)
        except _DEEPSEEK_NONCRITICAL_EXCEPTIONS:
            pass

        _TF_MODEL.eval()
        return _TF_MODEL, _TF_TOKENIZER


def _extract_text_from_any(obj) -> str:
    try:
        if obj is None:
            return ""
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            for key in ("text", "ocr_text", "content", "result", "output"):
                val = obj.get(key)
                if isinstance(val, str) and val.strip():
                    return val
                if isinstance(val, list):
                    parts = []
                    for item in val:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict):
                            txt = item.get("text") or item.get("content")
                            if isinstance(txt, str):
                                parts.append(txt)
                    if parts:
                        return "\n".join(parts)
            return str(obj)
        if isinstance(obj, (list, tuple)):
            parts = []
            for item in obj:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    txt = item.get("text") or item.get("content")
                    if isinstance(txt, str):
                        parts.append(txt)
            return "\n".join(parts)
        return str(obj)
    except _DEEPSEEK_NONCRITICAL_EXCEPTIONS:
        return ""
