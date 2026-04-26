# TTS Adapters Package
"""
This package contains adapter implementations for various TTS providers.
Each adapter provides a unified interface for different TTS engines.
"""

from .base import AudioFormat, ProviderStatus, TTSAdapter, TTSCapabilities, TTSRequest, TTSResponse, VoiceInfo

__all__ = [
    'TTSAdapter',
    'TTSCapabilities',
    'TTSRequest',
    'TTSResponse',
    'AudioFormat',
    'VoiceInfo',
    'ProviderStatus'
]
