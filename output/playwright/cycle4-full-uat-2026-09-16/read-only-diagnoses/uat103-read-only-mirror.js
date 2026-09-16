async (page) => page.evaluate(async () => {
  const chatId = 'a9feabc7-38aa-4b06-b08d-0b673b069819';
  const active = window.__tldw_useStoreMessageOption?.getState();
  const select = (row) => ({id:row.id,serverMessageId:row.serverMessageId??null,role:row.role??(row.isBot?'assistant':'user'),contentLength:(row.content??row.message??'').length,createdAt:row.createdAt??null,parentMessageId:row.parent_message_id??row.parentMessageId??null});
  return new Promise((resolve,reject) => {
    const open = indexedDB.open('PageAssistDatabase');
    open.onupgradeneeded = () => {open.transaction.abort();reject(new Error('Expected database missing; creation aborted'));};
    open.onerror = () => reject(open.error);
    open.onsuccess = () => {
      const db=open.result;
      const tx=db.transaction(['chatHistories','messages'],'readonly');
      const histories=tx.objectStore('chatHistories').index('server_chat_id').getAll(chatId);
      const output={checkedAt:new Date().toISOString(),chatId,activeHistoryId:active?.serverChatId===chatId?active.historyId:null,visibleRows:active?.serverChatId===chatId?active.messages.map(select):[],histories:[]};
      histories.onsuccess=()=>{for(const history of histories.result){const request=tx.objectStore('messages').index('history_id').getAll(history.id);request.onsuccess=()=>output.histories.push({id:history.id,serverChatId:history.server_chat_id,rows:request.result.map(select)});}};
      tx.oncomplete=()=>{db.close();resolve(output);};
      tx.onerror=()=>{db.close();reject(tx.error);};
      tx.onabort=()=>{db.close();reject(tx.error??new Error('Read aborted'));};
    };
  });
})
