from __future__ import annotations

from pydantic import BaseModel

from tldw_Server_API.app.api.v1.schemas.response_envelope import ResponseEnvelope
from tldw_Server_API.app.api.v1.utils.response_envelope import (
    envelope_error,
    envelope_success,
    is_response_envelope,
)


class Payload(BaseModel):
    value: str


def test_success_envelope_serializes_standard_contract() -> None:
    response = envelope_success(Payload(value="ok"), metadata={"request_id": "req-1"})

    assert isinstance(response, ResponseEnvelope)
    assert response.model_dump(mode="json") == {
        "success": True,
        "data": {"value": "ok"},
        "error": None,
        "error_code": None,
        "metadata": {"request_id": "req-1"},
    }


def test_error_envelope_serializes_standard_contract() -> None:
    response = envelope_error(
        "Failed to load resource",
        error_code="RESOURCE_LOAD_FAILED",
        metadata={"request_id": "req-2"},
    )

    assert response.model_dump(mode="json") == {
        "success": False,
        "data": None,
        "error": "Failed to load resource",
        "error_code": "RESOURCE_LOAD_FAILED",
        "metadata": {"request_id": "req-2"},
    }


def test_is_response_envelope_accepts_only_wrapped_payloads() -> None:
    assert is_response_envelope({"success": True, "data": {"id": 1}})
    assert is_response_envelope({"success": False, "error": "Nope"})
    assert is_response_envelope({"success": True, "metadata": {"request_id": "req-3"}})

    assert not is_response_envelope({"success": True, "file_id": "generated-file"})
    assert not is_response_envelope({"data": {"id": 1}})
    assert not is_response_envelope(None)
