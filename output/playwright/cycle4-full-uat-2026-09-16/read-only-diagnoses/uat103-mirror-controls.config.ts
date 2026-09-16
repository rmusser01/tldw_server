import fs from 'node:fs'
import path from 'node:path'
import { defineConfig } from '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/node_modules/.bun/vitest@4.0.18+08ee8852a9d25cb0/node_modules/vitest/dist/config.cjs'
const root='/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui'
const target=path.join(root,'src/hooks/__tests__/useServerChatLoader.mirror.integration.test.tsx')
const fixture=JSON.parse(fs.readFileSync('/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-full-uat-2026-09-16/multi/normal-messages-settled.json','utf8')).body.messages
const addition=`
describe('UAT103 actual loader with ordinary saved-turn mirror shape',()=>{
  it.each([{name:'acknowledged users',includeDraft:false},{name:'genuinely unsynced equal-text draft',includeDraft:true}])('$name',async({includeDraft})=>{
    const serverRows=${JSON.stringify(fixture)};
    state.histories.clear();state.messages.clear();state.controller=new AbortController();
    state.beforeMessagePut.mockReset().mockResolvedValue(undefined);
    vi.spyOn(tldwClient,'initialize').mockResolvedValue(undefined);
    vi.spyOn(tldwClient,'ensureConfigForRequest').mockResolvedValue({serverUrl:'http://chat.test',authMode:'multi-user',accessToken:'synthetic'});
    state.request.mockImplementation(async({path})=>path.includes('/messages')?{messages:serverRows}:{id:'cedar',title:'Standard saved',character_id:null,assistant_kind:null,assistant_id:null,source:'webui-chat',scope_type:'global'});
    const pairs=serverRows.filter(row=>row.sender!=='system').map((remote,index)=>({id:'local-'+index,history_id:'legacy',name:remote.sender==='user'?'You':'Assistant',role:remote.sender,content:remote.content,images:[],createdAt:Date.parse(remote.timestamp),serverMessageId:remote.id,parent_message_id:remote.sender==='assistant'?'local-'+(index-1):null}));
    if(includeDraft)pairs.push({id:'real-unsynced-draft',history_id:'legacy',name:'You',role:'user',content:serverRows[1].content,images:[],createdAt:Date.now(),parent_message_id:null}); pairs.forEach(row=>state.messages.set(row.id,row));
    state.histories.set('legacy',{id:'legacy',title:'Standard saved',server_chat_id:'cedar',is_rag:false,createdAt:1});
    useStoreMessageOption.setState({historyId:'legacy',serverChatId:'cedar',serverChatLoadState:'idle',messages:formatToMessage(pairs),history:formatToChatHistory(pairs),serverChatMetaLoaded:false,serverChatCharacterId:null,serverChatAssistantId:null,serverChatAssistantKind:null,streaming:false,isProcessing:false,temporaryChat:false});
    usePlaygroundSessionStore.getState().clearSession();
    usePlaygroundSessionStore.getState().saveSession({historyId:'legacy',serverChatId:'cedar',scopeKey:'scope-A'});
    const view=mountLoader();
    await waitFor(()=>expect(useStoreMessageOption.getState().serverChatLoadState).toBe('loaded'));
    const rows=useStoreMessageOption.getState().messages;
    console.log('UAT103_RECONCILIATION',JSON.stringify({serverRows:serverRows.length,visibleRows:rows.map(row=>({id:row.id,serverMessageId:row.serverMessageId??null,role:row.role,contentLength:row.message.length})),mirrorRows:mirrorRows().map(row=>({id:row.id,serverMessageId:row.serverMessageId??null,role:row.role,contentLength:row.content.length}))}));
    view.unmount();
    expect(rows).toHaveLength(includeDraft?6:5); if(includeDraft)expect(rows.some(row=>row.id==='real-unsynced-draft')).toBe(true);
  });
});`
export default defineConfig({root,plugins:[{name:'private-uat103-probe',enforce:'pre',load(id){if(id.split('?')[0]===target)return fs.readFileSync(target,'utf8')+addition}}],resolve:{alias:{'@':path.join(root,'src'),'~':path.join(root,'src')}},test:{environment:'jsdom',setupFiles:[path.join(root,'vitest.setup.ts')],include:[target],maxWorkers:1,fileParallelism:false,restoreMocks:true}})
