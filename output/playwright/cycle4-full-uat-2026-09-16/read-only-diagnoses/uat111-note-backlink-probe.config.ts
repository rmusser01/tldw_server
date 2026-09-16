import fs from 'node:fs'
import path from 'node:path'
import { defineConfig } from '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/node_modules/.bun/vitest@4.0.18+08ee8852a9d25cb0/node_modules/vitest/dist/config.cjs'
const root='/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui'
const target=path.join(root,'src/components/Notes/__tests__/NotesManagerPage.stage26.backlink-labels.test.tsx')
const evidence='/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle4-full-uat-2026-09-16/multi'
const bounded=JSON.parse(fs.readFileSync(path.join(evidence,'uat103-read-only-mirror.txt'),'utf8').split('### Result\n')[1].split('\n### Ran')[0])
const remote=JSON.parse(fs.readFileSync(path.join(evidence,'normal-messages-settled.json'),'utf8')).body.messages
const addition=`
  it.each(['captured-saved-mirror','acknowledged-saved','genuine-new-draft'])('UAT111 actual Note menu / %s',async(kind)=>{
    const captured=${JSON.stringify(bounded)};
    const serverRows=${JSON.stringify(remote)};
    configureCommonRequests(captured.chatId);
    mockGetChat.mockResolvedValue({id:captured.chatId,title:'Normal saved chat',character_id:null,assistant_kind:null,assistant_id:null,source:'webui-chat',version:1});
    mockListChatMessages.mockResolvedValue(serverRows);
    chatAuthority.selection=null;
    const acknowledged=serverRows.map(row=>({id:row.id,serverMessageId:row.id,role:row.sender,isBot:row.sender!=='user',name:row.sender==='user'?'You':'Assistant',message:row.content}));
    const capturedRows=captured.visibleRows.map(row=>({...row,isBot:row.role!=='user',name:row.role==='user'?'You':'Assistant',message:'x'.repeat(row.contentLength)}));
    useStoreMessageOption.setState({historyId:captured.activeHistoryId,serverChatId:captured.chatId,serverChatCharacterId:null,serverChatMetaLoaded:true,messages:kind==='captured-saved-mirror'?capturedRows:kind==='genuine-new-draft'?[...acknowledged,{id:'real-unsent',role:'user',isBot:false,name:'You',message:'Keep this genuinely unsent draft'}]:acknowledged,history:[],streaming:false,isProcessing:false});
    renderPage();
    fireEvent.click(await screen.findByTestId('notes-open-button-note-backlink-1'));
    fireEvent.click(await screen.findByTestId('notes-overflow-menu-button'));
    fireEvent.click(await screen.findByText(/open linked conversation/i));
    if(kind==='genuine-new-draft'){
      expect(mockMessageWarning).toHaveBeenCalledWith('Finish or save the current chat before opening the linked conversation.');
      expect(mockNavigate).not.toHaveBeenCalled();
      expect(mockListChatMessages).not.toHaveBeenCalled();
    }else{
      if(kind==='captured-saved-mirror') console.log('UAT111_GUARD',JSON.stringify({warningCalls:mockMessageWarning.mock.calls,chatRowsRequested:mockListChatMessages.mock.calls.length,navigationCalls:mockNavigate.mock.calls,blockedRows:capturedRows.filter(r=>!r.serverMessageId).map(r=>({id:r.id,role:r.role,contentLength:r.message.length}))}));
      await waitFor(()=>expect(mockNavigate).toHaveBeenCalledWith('/chat'));
      expect(useStoreMessageOption.getState().serverChatId).toBe(captured.chatId);
      expect(chatAuthority.selection).toBeNull();
    }
  });
`
export default defineConfig({root,plugins:[{name:'private-uat111-menu-probe',enforce:'pre',load(id){if(id.split('?')[0]===target){const text=fs.readFileSync(target,'utf8');const end=text.lastIndexOf('\n})');if(end<0)throw Error('Missing target suite close');return text.slice(0,end)+addition+text.slice(end)}}}],resolve:{alias:{'@':path.join(root,'src'),'~':path.join(root,'src')}},test:{environment:'jsdom',setupFiles:[path.join(root,'vitest.setup.ts')],include:[target],maxWorkers:1,fileParallelism:false,restoreMocks:true}})
