import original from "/private/tmp/uat093-web-storage-cross-tab.config"
const target = "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx"
export default {...original,plugins:[...original.plugins,{
 name:"uat093-vitest-storage-event-adapter",enforce:"pre" as const,transform(code:string,id:string){
  if(id!==target)return
  return {code:code.replace('newValue: next, storageArea: window.localStorage','newValue: next'),map:null}
 }
}]}
