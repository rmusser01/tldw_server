"""Actual Chat context builder; controlled DB boundary and no inference."""
import asyncio
import base64
import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from PIL import Image
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.tests.Chat.unit.test_chat_history_and_streaming import DummyChatDB

@pytest.mark.asyncio
@pytest.mark.parametrize('prior', ['empty', 'different', 'identical'])
async def test_never_dispatched_image_retry_is_a_new_turn(prior):
    buffer = io.BytesIO()
    Image.new('RGB', (2, 2), 'red').save(buffer, format='PNG')
    data = buffer.getvalue()
    content = [{'type':'text','text':'Question'}, {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(data).decode()}}]
    rows = [] if prior == 'empty' else [
        {'id':'old-user','sender':'user','content':'Question' if prior == 'identical' else 'Prior question','timestamp':1,
         'images':[{'image_data':data,'image_mime_type':'image/png'}]},
        {'id':'old-assistant','sender':'assistant','content':'Answered','timestamp':2}
    ]
    class SavedDB(DummyChatDB):
        def get_conversation_by_id(self, _):
            return {'id':'conv','character_id':1,'client_id':'client'}
        def get_message_metadata(self, message_id):
            return {'extra': {'client_message_id':'old-local-user'}} if message_id == 'old-user' else {}
    request = ChatCompletionRequest(model='test',save_to_db=True,conversation_id='conv', messages=[{'role':'user','content':content}], metadata={'tldw_retry_failed_turn':False,'tldw_client_message_id':'new-never-sent-local-user'})
    save = AsyncMock(return_value='new-user')
    state = {}
    await chat_service.build_context_and_messages(chat_db=SavedDB(rows),request_data=request,loop=asyncio.get_running_loop(),metrics=MagicMock(),default_save_to_db=False,final_conversation_id='conv',save_message_fn=save,runtime_state=state)
    assert state['user_message_id'] == 'new-user'
    assert save.await_count == 1
