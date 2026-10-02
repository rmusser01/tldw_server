"""Durable turns must not admit a second persistence protocol."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("protocol", ["selection", "retry", "regeneration"])
def test_durable_turn_cannot_mix_persistence_admission_protocols(protocol):
    cid = str(uuid4())
    overrides = {
        "selection": {
            "tldw_history_selection_v1": {
                "version": 1,
                "owner_key": "owner",
                "conversation_id": cid,
                "interpretation": {"kind": "parent_graph_v1"},
                "cursor": {"kind": "empty"},
                "selection_revision": 0,
                "purpose": "send",
                "messages": [],
                "fences": {"conversation": "1", "history": "1", "settings": "1"},
                "storage_context_digest": "storage",
                "request_context_digest": "request",
                "selection_digest": "selection",
            }
        },
        "retry": {"metadata": {"tldw_retry_failed_turn": True}},
        "regeneration": {"metadata": {"tldw_regenerate_from_message_id": str(uuid4())}},
    }
    with pytest.raises(ValidationError, match="tldw_turn"):
        ChatCompletionRequest.model_validate(
            {
                "model": "local-model",
                "conversation_id": cid,
                "save_to_db": True,
                "tldw_turn": {"user_message_id": str(uuid4())},
                "messages": [{"role": "user", "content": "Question"}],
                **overrides[protocol],
            }
        )
