import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.vn_play_schemas import (
    VNPlaySessionCreate,
    VNPlayTurnRequest,
)


def test_turn_request_requires_exactly_one_input_field() -> None:
    VNPlayTurnRequest(input_text="hello", client_scene_version=0, idempotency_key="k")

    with pytest.raises(ValidationError):
        VNPlayTurnRequest(
            input_text="hello",
            choice_id="choice-1",
            client_scene_version=0,
            idempotency_key="k",
        )


def test_turn_request_rejects_missing_input_field() -> None:
    with pytest.raises(ValidationError):
        VNPlayTurnRequest(client_scene_version=0, idempotency_key="k")


def test_create_session_defaults_linked_chat_to_read_only() -> None:
    request = VNPlaySessionCreate(
        mode="freeform",
        title="Test",
        primary_character_id=1,
        vn_asset_pack_id=2,
    )

    assert request.linked_chat_mode == "read_only_context"
