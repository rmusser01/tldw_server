import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/vitest.config"
const root="/Users/macbook-dev/Documents/GitHub/tldw_server2"
const target=root+"/apps/packages/ui/src/hooks/__tests__/usePlaygroundSessionPersistence.test.tsx"
const probe=`
import { Storage } from "@plasmohq/storage"
import { useServerChatLoader } from "@/hooks/chat/useServerChatLoader"
const canonicalConfig = {serverUrl:"http://server.test",authMode:"multi-user", accessToken:"test."+btoa(JSON.stringify({sub:"A"}))+".signature"}
const raceClient=vi.hoisted(()=>({getChat:vi.fn(),listChatMessages:vi.fn(),getCharacter:vi.fn(),ensureConfigForRequest:vi.fn()}))
vi.mock("@/services/service-prompts",()=>({loadServicePromptSnapshot:async(_ids:unknown,{signal}:{signal:AbortSignal})=>({scopeKey:"scope-A",scopeSignal:signal,scopeInvalidatedSignal:signal,requestScope:{config:canonicalConfig,userId:"A"},release:vi.fn()})}))
vi.mock("@/services/chat-settings",()=>({syncChatSettingsForServerChat:async()=>null}))
const ensureRaceHistory=async()=>null
const raceNotification={error:vi.fn()}
const raceTranslation=((key:string)=>key) as never
it("UAT034 late cached restore cannot downgrade an already loaded canonical conversation",async()=>{
 localStorage.clear()
 mocks.getConfig.mockResolvedValue(canonicalConfig)
 raceClient.ensureConfigForRequest.mockResolvedValue(canonicalConfig)
 raceClient.getChat.mockResolvedValue({id:"saved",title:"Canonical title",character_id:6,assistant_kind:"character",assistant_id:"6",source:"webui-character-chat",scope_type:"global",version:7})
 raceClient.listChatMessages.mockResolvedValue([{id:"server-answer",role:"assistant",content:"Canonical answer",version:1}])
 raceClient.getCharacter.mockResolvedValue({id:6,name:"Canonical character"})
 useStoreMessageOption.setState({serverChatId:null,historyId:null,messages:[],history:[],serverChatMetaLoaded:false,serverChatCharacterId:null,serverChatAssistantKind:null,serverChatAssistantId:null,temporaryChat:true,streaming:false,isProcessing:false})
 usePlaygroundSessionStore.getState().clearSession()
 usePlaygroundSessionStore.getState().saveSession({serverChatId:"saved",historyId:null,scopeKey:"global",trackedAssistantSelection:{kind:"character",id:"5",name:"Cached character",metadata:{selectionMode:"tracked"}},trackedAssistantKind:"character",trackedAssistantId:"5",trackedCharacterId:"5",queuedMessages:[]})
 let release!:()=>void
 const held=new Promise<void>(resolve=>{release=resolve})
 const originalSet=Storage.prototype.set
 let started=false
 const spy=vi.spyOn(Storage.prototype,"set").mockImplementation(async function(key:string,value:unknown){
   if(key==="selectedCharacter"&&(value as any)?.id==="5"){started=true;await held}
   return originalSet.call(this,key,value)
 })
 const view=renderHook(()=>{useServerChatLoader({ensureServerChatHistoryId:ensureRaceHistory,notification:raceNotification,t:raceTranslation});return usePlaygroundSessionPersistence()})
 let restoring:Promise<unknown>|undefined
 try{
  await waitFor(()=>expect(view.result.current.sessionScopeReady).toBe(true))
  act(()=>{restoring=view.result.current.restoreSession()})
  await waitFor(()=>expect(started).toBe(true))
  await waitFor(()=>expect(useStoreMessageOption.getState().serverChatLoadState).toBe("loaded"))
  expect(useStoreMessageOption.getState().serverChatMetaLoaded).toBe(true)
  expect(useStoreMessageOption.getState().serverChatTitle).toBe("Canonical title")
  expect(String(useStoreMessageOption.getState().serverChatCharacterId)).toBe("6")
  await act(async()=>{release();await restoring;await new Promise(resolve=>setTimeout(resolve,300))})
  console.log("POST_HELD_RESTORE",{meta:useStoreMessageOption.getState().serverChatMetaLoaded,id:useStoreMessageOption.getState().serverChatCharacterId,title:useStoreMessageOption.getState().serverChatTitle,load:useStoreMessageOption.getState().serverChatLoadState})
  expect(useStoreMessageOption.getState().serverChatMetaLoaded).toBe(true)
  expect(String(useStoreMessageOption.getState().serverChatCharacterId)).toBe("6")
  expect(useStoreMessageOption.getState().messages.map(m=>m.message)).toContain("Canonical answer")
 }finally{release();view.unmount();spy.mockRestore()}
})
`
export default {...base,plugins:[...base.plugins,{
name:"uat034-real-session-loader-ordering",enforce:"pre" as const,transform(code:string,id:string){
 if(id===target){
  code=code.replace('    getConfig: mocks.getConfig','    getConfig: mocks.getConfig, getChat:raceClient.getChat,listChatMessages:raceClient.listChatMessages,getCharacter:raceClient.getCharacter,ensureConfigForRequest:raceClient.ensureConfigForRequest')
  const start=code.indexOf('vi.mock("@/hooks/useSelectedAssistant",')
  const end=code.indexOf('import { usePlaygroundSessionPersistence }',start)
  if(start<0||end<0)throw new Error("Expected selected hook mock")
  return {code:code.slice(0,start)+code.slice(end)+probe,map:null}
 }
 if(id===root+"/apps/packages/ui/src/hooks/chat/useServerChatLoader.ts"){
  const start=code.indexOf('              const canonicalSelection = effectiveAssistantStateToSelection(')
  const end=code.indexOf('              setServerChatTitle(chatTitle || "")',start)
  if(start<0||end<0)throw new Error("Expected current premetadata scaffold; coordinate new source")
  return{code:code.slice(0,start)+code.slice(end),map:null}
 }
}}],test:{...base.test,setupFiles:[root+"/apps/tldw-frontend/vitest.setup.ts"],include:[target]}}
