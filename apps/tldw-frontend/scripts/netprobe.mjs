import { chromium } from "@playwright/test"
const WEB="http://localhost:8080", SERVER="http://127.0.0.1:8000", K="THIS-IS-A-SECURE-KEY-123-FAKE-KEY"
const b=await chromium.launch(); const c=await b.newContext()
await c.addInitScript(({s,k})=>{localStorage.setItem("tldwConfig",JSON.stringify({serverUrl:s,authMode:"single-user",apiKey:k,accessToken:""}));for(const[a,v]of Object.entries({isMigrated:"true",serverUrl:s,tldwServerUrl:s,authMode:"single-user",apiKey:k,__tldw_first_run_complete:"true",__tldw_test_bypass:"true",__tldw_allow_offline:"true"}))localStorage.setItem(a,v)},{s:SERVER,k:K})
const p=await c.newPage()
const calls=[]
p.on("response",async r=>{const u=r.url();if(u.includes("/api/")&&(/model|provider|config/i.test(u))){let len="?";try{len=(await r.text()).length}catch{};calls.push(`${r.status()} ${r.request().method()} ${u.split("8000")[1]||u} (${len}b)`)}})
await p.goto(`${WEB}/chat`,{waitUntil:"domcontentloaded"})
await p.getByTestId("chat-input").first().waitFor({state:"visible",timeout:30000})
await p.waitForTimeout(4000)
console.log("=== model/provider/config API responses on /chat load ===")
calls.forEach(c=>console.log("  "+c))
console.log("total:",calls.length)
await b.close()
