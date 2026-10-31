"""Provider-free realtime speech pipeline boundary."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from tldw_Server_API.app.core.Audio.Realtime.models import RealtimeSessionConfig


@dataclass(frozen=True)
class RealtimePipelineTextDelta:
    delta: str


@dataclass(frozen=True)
class RealtimePipelineTranscriptDelta:
    delta: str


@dataclass(frozen=True)
class RealtimePipelineAudioDelta:
    audio: bytes


@dataclass(frozen=True)
class RealtimePipelineTextDone:
    pass


@dataclass(frozen=True)
class RealtimePipelineTranscriptDone:
    pass


@dataclass(frozen=True)
class RealtimePipelineAudioDone:
    pass


@dataclass(frozen=True)
class RealtimePipelineTurnDone:
    status: str = "completed"


RealtimePipelineEvent = (
    RealtimePipelineTextDelta
    | RealtimePipelineTranscriptDelta
    | RealtimePipelineAudioDelta
    | RealtimePipelineTextDone
    | RealtimePipelineTranscriptDone
    | RealtimePipelineAudioDone
    | RealtimePipelineTurnDone
)


class RealtimePipeline(Protocol):
    async def transcribe_pcm16(self, audio: bytes, *, sample_rate_hz: int, language: str | None) -> str:
        raise NotImplementedError

    def stream_turn(self, transcript: str, *, config: RealtimeSessionConfig) -> AsyncIterator[RealtimePipelineEvent]:
        raise NotImplementedError
