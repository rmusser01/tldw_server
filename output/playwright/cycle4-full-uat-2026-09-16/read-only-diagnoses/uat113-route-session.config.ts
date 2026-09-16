import fs from 'node:fs'
import base from '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config'
const ui='/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui'
const target=ui+'/src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx'
export default {...base,plugins:[{name:'uat113-real-session-route-boundary',enforce:'pre',transform(code,id){
 if(id!==target)return
 const start=code.indexOf('vi.mock("@/hooks/usePlaygroundSessionPersistence",')
 const end=code.indexOf('vi.mock("@/hooks/playground-session-restore",',start)
 if(start<0||end<0)throw new Error('Expected original session mock boundary')
 code=code.slice(0,start)+code.slice(end)
 code=code.replace('const tldwClientState = vi.hoisted(() => ({','const tldwClientState = vi.hoisted(() => ({\n getConfig: vi.fn(async () => ({serverUrl: "http://chat.test", authMode: "multi-user", accessToken: "test." + btoa(JSON.stringify({sub: "A"})) + ".signature"})),')
 const insert=code.lastIndexOf('\n})')
 if(insert<0)throw new Error('Expected describe end')
 return{code:code.slice(0,insert)+'\n'+fs.readFileSync('/private/tmp/uat113-route-session-cases.txt','utf8')+code.slice(insert),map:null}
}}],test:{...base.test,setupFiles:[ui+'/vitest.setup.ts'],include:[target]}}
