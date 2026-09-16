import { beforeEach, expect, it, vi } from 'vitest'
const state = vi.hoisted(() => ({ histories: new Map(), messages: new Map() }))
function table(rows) { return {
 get: async id => rows.get(id),
 add: async row => { if (rows.has(row.id)) throw Error('duplicate'); rows.set(row.id, structuredClone(row)); return row.id },
 put: async row => { rows.set(row.id, structuredClone(row)); return row.id },
 update: async (id, changes) => { rows.set(id, {...rows.get(id), ...changes}); return 1 },
 where: field => ({equals: value => ({toArray: async () => [...rows.values()].filter(r => r[field]===value)})}),
} }
vi.mock('@/db/dexie/schema', () => ({db: { chatHistories: table(state.histories), messages: table(state.messages), modelNickname: {}, sessionFiles: {}, transaction: async (_mode, _tables, op) => op({abort: vi.fn()}) }}))
vi.mock('@/db/dexie/chat', () => ({PageAssistDatabase: class { async addMessage(row) { state.messages.set(row.id, structuredClone(row)) } }}))
vi.mock('@/db/index', () => ({}))
vi.mock('@/db/dexie/nickname', () => ({ModelNickname: class {}}))
vi.mock('@/db/dexie/models', () => ({ModelDb: class {}}))
vi.mock('@/services/recipe-persistence-uncertainty', () => ({}))
import { reconcileServerChatMirror, reconcileServerChatMessages } from '@/db/dexie/server-chat-mirror'
import { formatToMessage, saveMessage } from '@/db/dexie/helpers'
const raw = 'Chat with this media: Rowan.md\nSummarize this source.'
const wrapped = `Context: <doc>Rowan is an observatory run by Mara Chen.</doc>\nQuestion: ${raw}`
const remote = (id, role, message, createdAt) => ({ id, serverMessageId: id, serverMessageVersion:1, role, isBot:role==='assistant', name:role, message, images:[], createdAt })
beforeEach(() => { state.histories.clear(); state.messages.clear(); state.histories.set('h',{id:'h',server_chat_id:'c',server_scope_key:'owner'}); vi.useRealTimers() })
it('reproduces acknowledged RAG user moving after assistant on local-mirror reload', async () => {
 vi.spyOn(Date,'now').mockReturnValue(3000)
 await saveMessage({id:'local-u',history_id:'h',role:'user',name:'You',content:raw,images:[''],time:1,serverMessageId:'server-u'})
 await saveMessage({id:'local-a',history_id:'h',role:'assistant',name:'Assistant',content:'Rowan answer',images:[],time:2,serverMessageId:'server-a',parent_message_id:'local-u'})
 const canonical = [remote('server-u','user',wrapped,1000),remote('server-a','assistant','Rowan answer',2000)]
 const inMemory = reconcileServerChatMessages(formatToMessage([...state.messages.values()]),canonical)
 expect(inMemory.map(m => m.role)).toEqual(['user','assistant'])
 const mirror = await reconcileServerChatMirror({historyId:'h',chatId:'c',ownerKey:'owner',messages:canonical,localMessages:inMemory})
 const reloaded = formatToMessage(mirror.rows)
 console.info('ACK_ORDER_PROOF',JSON.stringify(reloaded.map(m=>({role:m.role,id:m.id,serverId:m.serverMessageId,createdAt:m.createdAt,message:m.message}))))
 expect(reloaded.map(m => m.role)).toEqual(['assistant','user'])
 expect(reloaded.map(m=>m.serverMessageId)).toEqual(['server-a','server-u'])
})
it('equal-content ordinary send control keeps user before assistant', async () => {
 for (const row of [{id:'local-u',role:'user',content:raw,serverMessageId:'server-u',createdAt:3001},{id:'local-a',role:'assistant',content:'Rowan answer',serverMessageId:'server-a',createdAt:3002}]) state.messages.set(row.id,{...row,history_id:'h',name:row.role,images:[]})
 const mirror=await reconcileServerChatMirror({historyId:'h',chatId:'c',ownerKey:'owner',messages:[remote('server-u','user',raw,1000),remote('server-a','assistant','Rowan answer',2000)]})
 expect(formatToMessage(mirror.rows).map(m=>m.role)).toEqual(['user','assistant'])
})
it('missing ACK plus transformed user blocks exact client correlation', () => {
 const local=[{...remote('local-u','user',raw,3001),serverMessageId:undefined},{...remote('local-a','assistant','Rowan answer',3002),serverMessageId:'server-a',parentMessageId:'local-u'}]
 const canonical=[{...remote('server-u','user',wrapped,1000),metadataExtra:{client_message_id:'local-u'}},remote('server-a','assistant','Rowan answer',2000)]
 const result=reconcileServerChatMessages(local,canonical)
 expect(result.map(m=>m.role)).toEqual(['user','assistant','user'])
 console.info('MISSING_ACK_PROOF', JSON.stringify(result.map(m=>({id:m.id,serverId:m.serverMessageId,role:m.role}))))
})
it('promoted older local timeout pair without receipts adds two visible duplicates despite current-turn ACKs', async () => {
 const denial='I could not retrieve evidence from the selected sources, so I did not send this as general chat.'
 const localRows=[
  {id:'old-u',role:'user',content:'Cedar question',serverMessageId:'old-u',createdAt:100},
  {id:'old-a',role:'assistant',content:'Cedar answer',serverMessageId:'old-a',createdAt:200},
  {id:'timeout-u',role:'user',content:raw,createdAt:301},
  {id:'timeout-a',role:'assistant',content:denial,createdAt:302,parent_message_id:'timeout-u'},
  {id:'current-u',role:'user',content:raw,serverMessageId:'new-3',createdAt:3001},
  {id:'current-a',role:'assistant',content:'Rowan answer',serverMessageId:'new-4',createdAt:3002,parent_message_id:'current-u'},
 ]
 for(const row of localRows) state.messages.set(row.id,{...row,history_id:'h',name:row.role,images:[]})
 const canonical=[remote('old-u','user','Cedar question',100),remote('old-a','assistant','Cedar answer',200),remote('new-1','user',raw,1000),remote('new-2','assistant',denial,1001),{...remote('new-3','user',wrapped,1002),metadataExtra:{client_message_id:'current-u'}},remote('new-4','assistant','Rowan answer',2000)]
 const first=reconcileServerChatMessages(formatToMessage([...state.messages.values()]),canonical)
 const mirror=await reconcileServerChatMirror({historyId:'h',chatId:'c',ownerKey:'owner',messages:canonical,localMessages:first})
 const result=formatToMessage(mirror.rows)
 expect(result).toHaveLength(8)
 expect(result.filter(m=>m.message===denial)).toHaveLength(2)
 expect(result.filter(m=>m.message===raw)).toHaveLength(3)
 expect(result.slice(-2).map(m=>m.serverMessageId)).toEqual(['new-4','new-3'])
 console.info('PROMOTED_TIMEOUT_PROOF',JSON.stringify(result.map(m=>({id:m.id,serverId:m.serverMessageId,role:m.role,createdAt:m.createdAt}))))
})
