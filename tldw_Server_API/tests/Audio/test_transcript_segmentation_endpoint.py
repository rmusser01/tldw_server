import pytest
from fastapi import HTTPException
from starlette.requests import Request

import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_transcriptions
from tldw_Server_API.app.api.v1.schemas.audio_schemas import TranscriptSegmentationRequest


pytestmark = pytest.mark.unit


def _req_entries(n_a=5, n_b=5):
    entries = []
    for i in range(n_a):
        entries.append({"composite": f"TOPIC_A {i}", "speaker": "A"})
    for i in range(n_b):
        entries.append({"composite": f"TOPIC_B {i}", "speaker": "B"})
    return entries


def _make_request() -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/audio/segment/transcript",
        "headers": [],
        "query_string": b"",
        "server": ("testserver", 80),
        "client": ("testclient", 12345),
    }

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, _receive)


async def _stub_embedder(chunks):
    def one_hot(line):
        if "TOPIC_A" in line:
            return [1.0, 0.0]
        if "TOPIC_B" in line:
            return [0.0, 1.0]
        return [0.1, 0.1]

    embs = []
    for c in chunks:
        parts = [p for p in c.splitlines() if p.strip()]
        if not parts:
            embs.append([0.0, 0.0])
            continue
        v = [0.0, 0.0]
        for p in parts:
            oh = one_hot(p)
            v[0] += oh[0]
            v[1] += oh[1]
        v[0] /= len(parts)
        v[1] /= len(parts)
        embs.append(v)
    return embs


@pytest.mark.asyncio
async def test_segment_transcript_endpoint(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Transcript_TreeSegmentation import TreeSegmenter

    # Preserve original classmethod to avoid recursion
    orig_create = TreeSegmenter.create_async

    async def _create_async(configs, entries, embedder=None):
        # Force stub embedder by delegating to the original implementation
        return await orig_create(
            configs=configs, entries=entries, embedder=_stub_embedder
        )

    # Monkeypatch the classmethod to ensure no network
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Transcript_TreeSegmentation.TreeSegmenter.create_async",
        _create_async,
        raising=False,
    )

    req = TranscriptSegmentationRequest(
        entries=_req_entries(6, 6),
        K=2,
        min_segment_size=2,
        lambda_balance=0.01,
        utterance_expansion_width=0,
    )

    resp = await audio_transcriptions.segment_transcript(req, _make_request())
    j = resp.model_dump()
    # Should split around half based on topics
    ones = [i for i, v in enumerate(j["transitions"]) if v == 1]
    assert len(ones) == 1 and ones[0] in (6, 7)  # allow slight variation
    assert len(j["segments"]) == 2


@pytest.mark.asyncio
async def test_segment_transcript_failure_log_is_sanitized(monkeypatch):
    class _LoggerStub:
        def __init__(self) -> None:
            self.errors: list[tuple[str, tuple, dict]] = []

        def error(self, message, *args, **kwargs) -> None:
            self.errors.append((message, args, kwargs))

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_transcriptions, "logger", logger_stub)

    async def _raise_segmentation_failure(*_args, **_kwargs):
        raise RuntimeError("tree segmentation leaked /private/segments.json")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Transcript_TreeSegmentation.TreeSegmenter.create_async",
        _raise_segmentation_failure,
        raising=False,
    )

    req = TranscriptSegmentationRequest(
        entries=_req_entries(6, 6),
        K=2,
        min_segment_size=2,
        lambda_balance=0.01,
        utterance_expansion_width=0,
    )

    with pytest.raises(HTTPException) as exc_info:
        await audio_transcriptions.segment_transcript(req, _make_request())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Transcript segmentation failed"
    assert logger_stub.errors == [("Transcript segmentation failed", (), {})]
