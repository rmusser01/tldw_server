# vibevoice_realtime_adapter.py
# Description: VibeVoice Realtime TTS adapter skeleton (streaming text -> audio)
#
# Imports
import asyncio
import contextlib
import json
import os
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse

#
# Third-party Imports
from loguru import logger

from ...Security.egress import evaluate_url_policy
from ..realtime_session import RealtimeSessionConfig, RealtimeTTSSession
from ..tts_exceptions import (
    TTSProviderInitializationError,
    TTSProviderNotConfiguredError,
    TTSProviderUnavailableError,
)
from ..tts_validation import validate_tts_request
from ..utils import parse_bool

#
# Local Imports
from .base import (
    AudioFormat,
    ProviderStatus,
    TTSAdapter,
    TTSCapabilities,
    TTSRequest,
    TTSResponse,
)

#
#######################################################################################################################
#
# VibeVoice Realtime Adapter (Skeleton)


def _websocket_url_to_policy_url(ws_url: str) -> str:
    try:
        parsed = urlparse(ws_url)
    except (TypeError, ValueError) as exc:
        raise TTSProviderInitializationError(
            "Invalid VibeVoice Realtime websocket URL",
            provider=VibeVoiceRealtimeAdapter.PROVIDER_KEY,
            details={"error": str(exc)},
        ) from exc

    scheme = (parsed.scheme or "").lower()
    if scheme not in {"ws", "wss"}:
        raise TTSProviderInitializationError(
            "VibeVoice Realtime websocket URL must use ws:// or wss://",
            provider=VibeVoiceRealtimeAdapter.PROVIDER_KEY,
            details={"scheme": scheme or None},
        )
    policy_scheme = "https" if scheme == "wss" else "http"
    return urlunparse(parsed._replace(scheme=policy_scheme))


def _validate_websocket_egress(ws_url: str) -> None:
    policy_url = _websocket_url_to_policy_url(ws_url)
    result = evaluate_url_policy(policy_url)
    if not result.allowed:
        raise TTSProviderInitializationError(
            f"VibeVoice Realtime websocket URL denied by egress policy: {result.reason}",
            provider=VibeVoiceRealtimeAdapter.PROVIDER_KEY,
            details={"reason": result.reason, "url": policy_url},
        )


class VibeVoiceRealtimeAdapter(TTSAdapter):
    """
    Adapter skeleton for Microsoft VibeVoice-Realtime (0.5B).

    This adapter is intentionally minimal; it provides configuration wiring and
    capability metadata, while returning "unavailable" until a realtime backend
    is integrated (local or remote).
    """

    PROVIDER_KEY = "vibevoice_realtime"
    SUPPORTED_FORMATS: set[AudioFormat] = {
        AudioFormat.PCM,
        AudioFormat.WAV,
        AudioFormat.MP3,
        AudioFormat.OPUS,
        AudioFormat.FLAC,
    }
    SUPPORTED_LANGUAGES = {"en"}
    MAX_TEXT_LENGTH = 8192
    DEFAULT_SAMPLE_RATE = 24000

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)
        cfg = config or {}
        extra_cfg = cfg.get("extra_params") if isinstance(cfg.get("extra_params"), dict) else {}

        self.model_path = (
            cfg.get("vibevoice_realtime_model_path")
            or cfg.get("model_path")
            or extra_cfg.get("model_path")
            or "microsoft/VibeVoice-Realtime-0.5B"
        )
        self.device = (
            cfg.get("vibevoice_realtime_device")
            or cfg.get("device")
            or extra_cfg.get("device")
            or "cpu"
        )
        self.sample_rate = int(
            cfg.get("vibevoice_realtime_sample_rate")
            or cfg.get("sample_rate")
            or extra_cfg.get("sample_rate")
            or self.DEFAULT_SAMPLE_RATE
        )
        self.ws_url = (
            cfg.get("vibevoice_realtime_ws_url")
            or cfg.get("ws_url")
            or extra_cfg.get("ws_url")
            or os.getenv("VIBEVOICE_REALTIME_WS_URL")
        )
        self.ws_timeout = float(
            cfg.get("vibevoice_realtime_ws_timeout")
            or cfg.get("ws_timeout")
            or extra_cfg.get("ws_timeout")
            or os.getenv("VIBEVOICE_REALTIME_WS_TIMEOUT", "30")
        )
        self.ws_headers = self._coerce_headers(
            cfg.get("vibevoice_realtime_ws_headers")
            or cfg.get("ws_headers")
            or extra_cfg.get("ws_headers")
            or os.getenv("VIBEVOICE_REALTIME_WS_HEADERS")
        )
        self.auto_download = parse_bool(
            cfg.get("vibevoice_realtime_auto_download")
            or cfg.get("auto_download")
            or extra_cfg.get("auto_download"),
            default=False,
        )
        self.stream_chunk_ms = int(
            cfg.get("vibevoice_realtime_stream_chunk_ms")
            or cfg.get("stream_chunk_size_ms")
            or cfg.get("stream_chunk_ms")
            or extra_cfg.get("stream_chunk_size_ms")
            or extra_cfg.get("stream_chunk_ms")
            or 40
        )
    async def initialize(self) -> bool:
        """Initialize VibeVoice Realtime adapter (websocket backend)."""
        if not self.ws_url:
            raise TTSProviderNotConfiguredError(
                "VibeVoice Realtime websocket URL not configured",
                provider=self.PROVIDER_KEY,
            )
        self._status = ProviderStatus.AVAILABLE
        return True

    async def get_capabilities(self) -> TTSCapabilities:
        return TTSCapabilities(
            provider_name="VibeVoice-Realtime",
            supported_languages=self.SUPPORTED_LANGUAGES,
            supported_voices=[],
            supported_formats=self.SUPPORTED_FORMATS,
            max_text_length=self.MAX_TEXT_LENGTH,
            supports_streaming=True,
            supports_voice_cloning=False,
            supports_emotion_control=False,
            supports_speech_rate=True,
            supports_pitch_control=False,
            supports_volume_control=False,
            supports_ssml=False,
            supports_phonemes=False,
            supports_multi_speaker=False,
            supports_background_audio=False,
            latency_ms=200,
            sample_rate=self.sample_rate,
            default_format=AudioFormat.PCM,
        )

    async def generate(self, request: TTSRequest) -> TTSResponse:
        """Realtime adapter does not yet support non-session generation."""
        # Validate input for consistent error handling.
        validate_tts_request(request, provider=self.PROVIDER_KEY)
        raise TTSProviderUnavailableError(
            "VibeVoice Realtime adapter is not yet implemented",
            provider=self.PROVIDER_KEY,
        )

    async def create_realtime_session(self, config: RealtimeSessionConfig) -> RealtimeTTSSession:
        """Create a realtime session backed by a websocket server."""
        if not await self.ensure_initialized():
            raise TTSProviderNotConfiguredError(
                "VibeVoice Realtime adapter not initialized",
                provider=self.PROVIDER_KEY,
            )
        session = _VibeVoiceRealtimeWebSocketSession(
            ws_url=self.ws_url,
            ws_headers=self.ws_headers,
            ws_timeout=self.ws_timeout,
            config=config,
        )
        await session.start()
        return session

    @staticmethod
    def _coerce_headers(raw: Any) -> Optional[dict[str, str]]:
        if raw is None:
            return None
        if isinstance(raw, dict):
            return {str(k): str(v) for k, v in raw.items()}
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items()}
            except Exception:
                return None
        return None


class _VibeVoiceRealtimeWebSocketSession(RealtimeTTSSession):
    def __init__(
        self,
        *,
        ws_url: str,
        ws_headers: Optional[dict[str, str]],
        ws_timeout: float,
        config: RealtimeSessionConfig,
    ) -> None:
        self._ws_url = ws_url
        self._ws_headers = ws_headers
        self._ws_timeout = ws_timeout
        self._config = config
        self._session = None
        self._ws = None
        self._recv_task: Optional[asyncio.Task] = None
        self._queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._closed = False
        self._error: Optional[Exception] = None

    @property
    def error(self) -> Optional[Exception]:
        return self._error

    async def start(self) -> None:
        _validate_websocket_egress(self._ws_url)

        try:
            import aiohttp
        except Exception as exc:
            raise TTSProviderInitializationError(
                "aiohttp is required for VibeVoice Realtime websocket sessions",
                provider="vibevoice_realtime",
                details={"error": str(exc)},
            ) from exc

        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(
            self._ws_url,
            headers=self._ws_headers,
            timeout=self._ws_timeout,
        )

        await self._send_config()
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def push_text(self, delta: str) -> None:
        if not delta or self._closed:
            return
        await self._send_json({"type": "text", "delta": delta})

    async def commit(self) -> None:
        if self._closed:
            return
        await self._send_json({"type": "commit"})

    async def finish(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            await self._send_json({"type": "final"})
        await self._close()

    async def audio_stream(self):
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                break
            if chunk:
                yield chunk

    async def _send_config(self) -> None:
        payload = {
            "type": "config",
            "model": self._config.model,
            "voice": self._config.voice,
            "format": self._config.response_format,
            "speed": self._config.speed,
            "lang": self._config.lang_code,
            "extra_params": self._config.extra_params,
        }
        await self._send_json(payload)

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if not self._ws:
            return
        await self._ws.send_json(payload)

    async def _recv_loop(self) -> None:
        try:
            if not self._ws:
                return
            import aiohttp

            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    await self._queue.put(msg.data)
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        data = None
                    if not isinstance(data, dict):
                        continue
                    msg_type = str(data.get("type") or "").lower()
                    if msg_type == "error":
                        self._error = RuntimeError(data.get("message") or "Realtime backend error")
                        logger.error(f"VibeVoice Realtime backend error: {data}")
                        break
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        except Exception as exc:
            self._error = exc
        finally:
            await self._queue.put(None)
            await self._close()

    async def _close(self) -> None:
        if self._ws:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None
        if self._session:
            with contextlib.suppress(Exception):
                await self._session.close()
            self._session = None
