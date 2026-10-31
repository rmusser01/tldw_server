"""Selected native history preserves image-only turns and their ordered options."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.Chat.chat_service import build_context_and_messages
from tldw_Server_API.app.core.Chat.persistence_service import prepare_native_history_message


@pytest.fixture
def selected_history():
    db = MagicMock(client_id="alice")
    db.transaction.side_effect = lambda: nullcontext(object())
    db.get_roleplay_resume_state.return_value = {
        "conversation": {"id": "conv", "client_id": "alice"},
        "settings": None, "behavior_snapshot": {"status": "missing"},
    }
    db.append_selected_history_inputs.return_value = {"input_message_id": "new-input"}

    async def build(item, role="user"):
        node = {"id": item["id"], "role": role, "revision": item["revision"]}
        db.validate_history_selection.return_value = (SimpleNamespace(nodes=[node]), [item])
        request = SimpleNamespace(
            metadata=None, character_id=None, messages=[], save_to_db=True,
            tldw_history_selection_v1=SimpleNamespace(model_dump=lambda **_: {}),
        )
        result = await build_context_and_messages(
            chat_db=db, request_data=request, loop=asyncio.get_running_loop(),
            metrics=MagicMock(), default_save_to_db=True, final_conversation_id="conv",
            save_message_fn=AsyncMock(), runtime_state={
                "history_owner_key": "owner", "history_owner_client_id": "alice", "history_inputs": [],
            },
        )
        return result[4][0]

    return build, db


def image_item(**overrides):
    return {
        "id": "saved", "revision": "sha256:bound-content", "_message_version": 1, "message": "<Image attachment x2>",
        "images": ["data:image/png;base64,YQ==", "data:image/png;base64,Yg=="],
        "extra_metadata": {"content_placeholder_reason": "image_attachment", "image_details": ["high", "low"]},
        **overrides,
    }


@pytest.mark.asyncio
async def test_selected_image_only_turn_preserves_ordered_images_and_detail(selected_history):
    build, _ = selected_history
    item = image_item()
    entry = await build(item)
    assert entry["content"] == [
        {"type": "image_url", "image_url": {"url": url, "detail": detail}}
        for url, detail in zip(item["images"], ["high", "low"])
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("overrides", [
    {"message": "User-edited image description", "_message_version": 2},
    {"_message_version": 2},
    {"message": "An explicit caption"},
])
async def test_selected_image_placeholder_metadata_does_not_erase_literal_text(selected_history, overrides):
    build, _ = selected_history
    item = image_item(**overrides)
    entry = await build(item)
    assert entry["content"][0] == {"type": "text", "text": item["message"]}
    assert len(entry["content"]) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["tool_calls", "function_call"])
async def test_selected_tool_placeholders_remain_null_content(selected_history, reason):
    build, _ = selected_history
    metadata = {"content_placeholder_reason": reason}
    item = image_item(message=f"[{reason}]", images=[], extra_metadata=metadata)
    if reason == "tool_calls":
        item[reason] = [{"id": "call", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]
    else:
        metadata[reason] = {"name": "lookup", "arguments": "{}"}
    entry = await build(item, role="assistant")
    assert entry["content"] is None
    assert entry[reason] == (item.get(reason) or metadata[reason])


@pytest.mark.asyncio
async def test_selected_unknown_placeholder_reason_preserves_literal_content(selected_history):
    build, _ = selected_history
    entry = await build(image_item(message="literal", images=[], extra_metadata={"content_placeholder_reason": "unknown"}))
    assert entry["content"] == "literal"


@pytest.mark.asyncio
@pytest.mark.parametrize("details", [["high"], ["high", "invalid"], "high", [{}, "low"]])
async def test_selected_corrupt_image_options_fail_before_admission(selected_history, details):
    build, db = selected_history
    with pytest.raises(HTTPException) as failure:
        await build(image_item(extra_metadata={"image_details": details}))
    assert failure.value.status_code == 409
    db.append_selected_history_inputs.assert_not_called()


@pytest.mark.asyncio
async def test_native_preparation_persists_ordered_image_details():
    async def processor(_content, _conversation_id, *, image_details=None):
        if image_details is not None:
            image_details.extend(["high", "low"])
        return [], [(b"first", "image/png"), (b"second", "image/png")]

    prepared = await prepare_native_history_message({"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url, "detail": detail}}
        for url, detail in zip(image_item()["images"], ["high", "low"])
    ]}, "conv", processor)
    assert prepared["extra_metadata"]["image_details"] == ["high", "low"]
    assert [image["data"] for image in prepared["images"]] == [b"first", b"second"]


@pytest.mark.asyncio
async def test_native_preparation_preserves_details_with_real_content_processor():
    import base64
    import io
    from PIL import Image
    from tldw_Server_API.app.api.v1.endpoints.chat import _process_content_for_db_sync

    output = io.BytesIO()
    Image.new("RGB", (1, 1)).save(output, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")
    prepared = await prepare_native_history_message({"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url, "detail": detail}}
        for detail in ["high", "low"]
    ]}, "conv", _process_content_for_db_sync)
    assert prepared["extra_metadata"]["image_details"] == ["high", "low"]
    assert len(prepared["images"]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [
    {"url": "image", "detail": "invalid"},
    {"url": "image", "detail": {}},
    {"url": "image", "unsupported": True},
])
async def test_native_preparation_rejects_unsupported_options_before_processing(options):
    processor = AsyncMock()
    with pytest.raises(HTTPException) as failure:
        await prepare_native_history_message({"role": "user", "content": [
            {"type": "image_url", "image_url": options},
        ]}, "conv", processor)
    assert failure.value.status_code == 422
    processor.assert_not_called()
