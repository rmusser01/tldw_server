import original from "/private/tmp/uat093-web-storage-isolated.config"
const ui = "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui"
export default {...original, plugins:[...original.plugins, {
 name:"uat093-causal-removal-control", enforce:"pre" as const, transform(code:string,id:string) {
  if (id !== ui + "/src/components/Option/Playground/Playground.tsx") return
  const start=code.indexOf('  React.useEffect(() => {\n    if (\n      !serverChatId ||\n      !serverChatMetaLoaded ||')
  const end=code.indexOf('  const setRouteContext = useChatSurfaceCoordinatorStore(',start)
  if(start<0||end<0) throw new Error("Expected mismatch effect")
  return {code:code.slice(0,start)+code.slice(end),map:null}
 }
}]}
