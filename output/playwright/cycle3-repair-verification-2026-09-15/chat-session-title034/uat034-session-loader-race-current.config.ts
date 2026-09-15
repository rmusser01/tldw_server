import original from "/private/tmp/uat034-session-loader-race.config"
const loader="/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/hooks/chat/useServerChatLoader.ts"
export default {...original,plugins:original.plugins.map(plugin=>plugin?.name!=="uat034-real-session-loader-ordering"?plugin:{...plugin,transform(code:string,id:string){if(id===loader)return;return plugin.transform(code,id)}})}
