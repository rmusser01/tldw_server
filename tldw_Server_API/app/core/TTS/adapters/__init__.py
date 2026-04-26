# TTS Adapters Package
"""
This package contains adapter implementations for various TTS providers.
Each adapter provides a unified interface for different TTS engines.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "TTSAdapter",
    "TTSCapabilities",
    "TTSRequest",
    "TTSResponse",
    "AudioFormat",
    "VoiceInfo",
    "ProviderStatus",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        submodule_name = f"{__name__}.{name}"
        try:
            module = import_module(submodule_name)
            globals()[name] = module
            return module
        except ModuleNotFoundError as exc:
            if exc.name != submodule_name:
                raise
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    from .base import AudioFormat, ProviderStatus, TTSAdapter, TTSCapabilities, TTSRequest, TTSResponse, VoiceInfo

    exports = {
        "TTSAdapter": TTSAdapter,
        "TTSCapabilities": TTSCapabilities,
        "TTSRequest": TTSRequest,
        "TTSResponse": TTSResponse,
        "AudioFormat": AudioFormat,
        "VoiceInfo": VoiceInfo,
        "ProviderStatus": ProviderStatus,
    }
    export = exports[name]
    globals()[name] = export
    return export


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
