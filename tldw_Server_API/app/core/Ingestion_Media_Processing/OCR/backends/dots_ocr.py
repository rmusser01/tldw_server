from __future__ import annotations

import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tempfile

from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.base import OCRBackend
from tldw_Server_API.app.core.Utils.Utils import logging
from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.runtime_support import image_payload


class DotsOCRBackend(OCRBackend):
    """
    dots.ocr backend wrapper using the project-provided parser CLI.

    This backend writes image bytes to a temporary PNG and calls
    `python -m dots_ocr.parser <image> --prompt <prompt>` to retrieve text.

    Assumptions/Requirements:
    - dots.ocr is installed and importable as the `dots_ocr` package.
    - Its parser CLI can run in the current environment and is configured
      to use either vLLM or HuggingFace per the upstream installation docs.

    Configuration:
    - DOTS_OCR_PROMPT: optional prompt to pass to the parser (default: "prompt_ocr").
      Example prompts from the project include: `prompt_ocr`, `prompt_layout_only_en`, etc.
    """

    name = "dots"

    @classmethod
    def available(cls) -> bool:
        # Available if module importable OR a VLLM endpoint is configured
        try:
            if os.getenv("DOTS_VLLM_URL"):
                return True
            return importlib.util.find_spec("dots_ocr") is not None
        except (AttributeError, ImportError, OSError, TypeError, ValueError):
            return False

    def describe(self) -> dict:
        return {
            "prompt": os.getenv("DOTS_OCR_PROMPT", "prompt_ocr"),
            "cmd_override": bool(os.getenv("DOTS_OCR_CMD")),
            "vllm_url": os.getenv("DOTS_VLLM_URL"),
            "vllm_model": os.getenv("DOTS_VLLM_MODEL", "model"),
            "vllm_timeout": int(os.getenv("DOTS_VLLM_TIMEOUT", "60")),
            "vllm_use_data_url": os.getenv("DOTS_VLLM_USE_DATA_URL", "false"),
        }

    def _build_command(self, img_path: str) -> list:
        python_exe = sys.executable or "python3"
        env_cmd = os.getenv("DOTS_OCR_CMD")
        if env_cmd:
            return shlex.split(env_cmd) + [img_path]
        return [python_exe, "-m", "dots_ocr.parser", img_path, "--prompt", os.getenv("DOTS_OCR_PROMPT", "prompt_ocr")]

    def ocr_image(self, image_bytes: bytes, lang: str | None = None) -> str:
        if not self.available():
            logging.warning("DotsOCRBackend not available: set DOTS_VLLM_URL or install dots_ocr module.")
            return ""

        # If DOTS_VLLM_URL configured, use OpenAI-compatible endpoint
        if os.getenv("DOTS_VLLM_URL"):
            try:
                return _ocr_via_vllm(image_bytes, os.getenv("DOTS_OCR_PROMPT", "prompt_ocr"))
            except (ConnectionError, OSError, RuntimeError, TimeoutError, TypeError, ValueError) as e:
                logging.error(f"Dots vLLM path failed, falling back to CLI: {e}", exc_info=True)

        # Write the image to a temporary PNG file and run the parser
        with tempfile.TemporaryDirectory(prefix="dots_ocr_") as tmpdir:
            img_path = os.path.join(tmpdir, "page.png")
            try:
                with open(img_path, "wb") as f:
                    f.write(image_bytes)
            except OSError as e:
                logging.error(f"DotsOCRBackend: failed to write temp image: {e}", exc_info=True)
                return ""

            cmd = self._build_command(img_path)
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except (OSError, ValueError) as e:
                logging.error(f"DotsOCRBackend: parser invocation failed: {e}", exc_info=True)
                return ""

            if proc.returncode != 0:
                # Log stderr for diagnostics; return best-effort output
                if proc.stderr:
                    logging.warning(f"DotsOCRBackend: parser stderr: {proc.stderr.strip()[:500]}")
                # Some tools still emit useful content to stdout even on nonzero exit
            raw_out = (proc.stdout or "").strip()

            # Attempt to parse JSON if emitted; fall back to plain text
            # Heuristic: try last JSON object if multiple lines/logs present
            text_out = ""
            if raw_out:
                try:
                    # Try whole stdout first
                    data = json.loads(raw_out)
                    text_out = _extract_text_from_any(data)
                except (TypeError, ValueError, json.JSONDecodeError):
                    # Try line-by-line for a JSON object
                    for line in reversed(raw_out.splitlines()):
                        line = line.strip()
                        if not (line.startswith("{") and line.endswith("}")):
                            continue
                        try:
                            data = json.loads(line)
                            text_out = _extract_text_from_any(data)
                            if text_out:
                                break
                        except (TypeError, ValueError, json.JSONDecodeError):
                            continue
                    if not text_out:
                        # As a last resort, return raw stdout
                        text_out = raw_out

            return text_out or ""


def _extract_text_from_any(obj) -> str:
    """
    Best-effort extraction of textual content from various possible structures.
    """
    try:
        if obj is None:
            return ""
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            # Common fields that might contain OCR text
            for k in ("text", "ocr_text", "content", "result", "output"):
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    return v
                if isinstance(v, list):
                    # Flatten list of strings/segments
                    items = []
                    for item in v:
                        if isinstance(item, str):
                            items.append(item)
                        elif isinstance(item, dict):
                            t = item.get("text") or item.get("content")
                            if isinstance(t, str):
                                items.append(t)
                    if items:
                        return "\n".join(items)
            # Fallback: stringify
            return "".join(
                s for s in [
                    obj.get("text") if isinstance(obj.get("text"), str) else None,
                    obj.get("content") if isinstance(obj.get("content"), str) else None,
                ]
                if s
            )
        if isinstance(obj, (list, tuple)):
            parts = []
            for el in obj:
                if isinstance(el, str):
                    parts.append(el)
                elif isinstance(el, dict):
                    t = el.get("text") or el.get("content")
                    if isinstance(t, str):
                        parts.append(t)
            return "\n".join(parts)
        # Fallback to string representation
        return str(obj)
    except (AttributeError, TypeError, ValueError):
        return ""


def _ocr_via_vllm(image_bytes: bytes, prompt: str) -> str:

    url = os.getenv("DOTS_VLLM_URL").rstrip("/")
    model = os.getenv("DOTS_VLLM_MODEL", "model")
    timeout = int(os.getenv("DOTS_VLLM_TIMEOUT", "60"))
    use_data_url = str(os.getenv("DOTS_VLLM_USE_DATA_URL", "true")).lower() in ("1", "true", "yes")

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
            "max_new_tokens": _getf("DOTS_VLLM_MAX_NEW_TOKENS", int, 2048),
            "temperature": _getf("DOTS_VLLM_TEMPERATURE", float, 0.7),
            "repetition_penalty": _getf("DOTS_VLLM_REPETITION_PENALTY", float, 1.05),
            "top_p": _getf("DOTS_VLLM_TOP_P", float, 0.8),
            "top_k": _getf("DOTS_VLLM_TOP_K", int, 20),
            "do_sample": _getf("DOTS_VLLM_DO_SAMPLE", lambda x: str(x).lower() in ("1","true","yes"), True),
        }

        from tldw_Server_API.app.core.http_client import fetch_json
        j = fetch_json(method="POST", url=url, json=data, timeout=timeout)
    return (
        j.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    ) or ""
