from __future__ import annotations

from abc import ABC, abstractmethod

from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.types import (
    OCRResult,
    normalize_ocr_format,
)


class OCRBackend(ABC):
    """
    Abstract OCR backend interface.

    Implementations should support OCR on raw image bytes.
    PDF page handling is performed by the ingestion code (which renders pages to images).
    """

    name: str = "abstract"

    @classmethod
    @abstractmethod
    def available(cls) -> bool:
        """Return True if this backend is usable on this system (e.g. binary/library available)."""
        raise NotImplementedError

    @abstractmethod
    def ocr_image(self, image_bytes: bytes, lang: str | None = None) -> str:
        """Run OCR on an image (bytes) and return extracted text (UTF-8)."""
        raise NotImplementedError

    def ocr_image_structured(
        self,
        image_bytes: bytes,
        lang: str | None = None,
        output_format: str | None = None,
        prompt_preset: str | None = None,
    ) -> OCRResult:
        """
        Best-effort structured OCR result. Backends can override for richer output.
        """
        text = self.ocr_image(image_bytes, lang)
        fmt = normalize_ocr_format(output_format)
        if fmt == "unknown":
            fmt = "text"
        return OCRResult(
            text=text or "",
            format=fmt,
            raw=None,
            meta={
                "backend": getattr(self, "name", type(self).__name__),
                "prompt_preset": prompt_preset,
                "output_format": output_format,
            },
        )
