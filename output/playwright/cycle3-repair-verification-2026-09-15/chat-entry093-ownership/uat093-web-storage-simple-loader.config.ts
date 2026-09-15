import original from "/private/tmp/uat093-web-storage-no-mismatch.config"
const target = "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/hooks/chat/useServerChatLoader.ts"
export default {...original,plugins:[...original.plugins,{
 name:"uat093-remove-mismatch-only-scaffolding",enforce:"pre" as const,transform(code:string,id:string){
  if(id!==target)return
  const start=code.indexOf('              const canonicalSelection = effectiveAssistantStateToSelection(')
  const end=code.indexOf('              setServerChatTitle(chatTitle || "")',start)
  if(start<0||end<0)throw new Error("Expected metadata selection scaffold")
  return {code:code.slice(0,start)+code.slice(end),map:null}
 }
}]}
