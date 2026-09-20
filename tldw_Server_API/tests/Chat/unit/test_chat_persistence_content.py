import pytest

from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionRequestMessageContentPartText,
)


@pytest.mark.asyncio
async def test_process_content_handles_pydantic_parts():
    part = ChatCompletionRequestMessageContentPartText(type="text", text="hello")

    text_parts, images = await chat_endpoint._process_content_for_db_sync([part], "conv-1")

    assert text_parts == ["hello"]
    assert images == []


@pytest.mark.parametrize("extra", [None, {}, {"image_details": ["auto", "auto"]}])
def test_saved_image_details_default_legacy_rows_to_auto(extra):
    from tldw_Server_API.app.core.Chat.chat_service import _saved_image_details

    assert _saved_image_details(extra, 2) == ["auto", "auto"]


@pytest.mark.parametrize("details", [[], ["high"], ["high", "invalid"], "high", [{}, "low"]])
def test_saved_image_details_reject_partial_or_unsupported_options(details):
    from fastapi import HTTPException

    from tldw_Server_API.app.core.Chat.chat_service import _saved_image_details

    with pytest.raises(HTTPException) as error:
        _saved_image_details({"image_details": details}, 2)
    assert error.value.status_code == 409
