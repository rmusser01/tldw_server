import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from loguru import logger
logger.remove()
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter import CustomOpenAIAdapter
from tldw_Server_API.tests.Chat.unit.test_chat_history_and_streaming import DummyChatDB
OUT=Path('/private/tmp/source013-diagnosis-20260916')
class DB(DummyChatDB):
 def get_conversation_by_id(self,_id):
  return {'id':'conv','client_id':'client','character_id':None,'assistant_kind':None,'assistant_id':None}
raw='Chat with this media: Rowan.md\nSummarize this source.'
wrapped=f'Context: <doc>Rowan is a fictional observatory. Mara Chen coordinates it. Opens 7 December 2026. Entrance east. Volunteers meet Friday at 19:30.</doc>\nQuestion: {raw}'
canonical=[{'id':'old-user','sender':'user','content':'Who coordinates Cedar?','timestamp':1},{'id':'old-answer','sender':'assistant','content':'Jonah Patel coordinates Cedar.','timestamp':2}]
timeoutpair=[{'role':'user','content':raw},{'role':'assistant','content':'I could not retrieve evidence from the selected sources, so I did not send this as general chat.'}]
async def run(order):
 req=ChatCompletionRequest(model='synthetic',conversation_id='conv',save_to_db=True,history_message_order=order,messages=[{'role':r['sender'],'content':r['content']} for r in canonical]+timeoutpair+[{'role':'user','content':wrapped}],metadata={'tldw_client_message_id':'current-user'})
 appends=[]
 async def save(db,cid,message,**kwargs):
  appends.append(dict(message)); return 'new-'+str(len(appends))
 state={}
 result=await chat_service.build_context_and_messages(chat_db=DB(canonical),request_data=req,loop=asyncio.get_running_loop(),metrics=MagicMock(),default_save_to_db=False,final_conversation_id='conv',save_message_fn=save,runtime_state=state)
 system,messages=chat_service.apply_prompt_templating(req,result[0],result[4])
 args=chat_service.build_call_params_from_request(request_data=req,target_api_provider='custom-openai-api',provider_api_key='synthetic-unused-key',templated_llm_payload=messages,final_system_message=system,app_config={},resolved_model='synthetic')
 _,adapter_req,_=chat_service._build_adapter_request_from_chat_args(args)
 payload=CustomOpenAIAdapter()._build_payload(adapter_req)
 last=payload['messages'][-1]['content']; last=last if isinstance(last,str) else ''.join(p.get('text','') for p in last)
 assert len(appends)==3
 assert [m.get('client_message_id') for m in appends]==[None,None,'current-user']
 assert all(x in last for x in ['Rowan','Mara Chen','7 December 2026','east','Friday at 19:30'])
 return {'order':order,'appends':appends,'ack':state['user_message_id'],'provider_last_user':last,'provider_message_count':len(payload['messages'])}
async def main():
 result=[await run('asc'),await run('desc')]
 (OUT/'backend-results.json').write_text(json.dumps(result,indent=2))
asyncio.run(main())
