import { chromium } from "@playwright/test"
// Override via env; the fake default matches the repo's e2e smoke fixture key.
const WEB=process.env.WEB_URL||"http://localhost:8080", S=process.env.SERVER_URL||"http://127.0.0.1:8000", K=process.env.TLDW_API_KEY||"THIS-IS-A-SECURE-KEY-123-FAKE-KEY"
const b=await chromium.launch(); const c=await b.newContext()
await c.addInitScript(({s,k})=>{localStorage.setItem("tldwConfig",JSON.stringify({serverUrl:s,authMode:"single-user",apiKey:k,accessToken:""}));for(const[a,v]of Object.entries({isMigrated:"true",serverUrl:s,tldwServerUrl:s,authMode:"single-user",apiKey:k,__tldw_first_run_complete:"true",__tldw_test_bypass:"true",__tldw_allow_offline:"true"}))localStorage.setItem(a,v)},{s:S,k:K})
const p=await c.newPage()
const all=[]
const paths=new Set()
p.on("response",async r=>{const u=r.url();if(u.includes("/api/v1/")){const path=u.split("/api/v1")[1].split("?")[0];paths.add(path);let len="?";try{len=(await r.text()).length}catch{};all.push(`${r.status()} ${r.request().method()} ${path} (${len}b)`)}})
const consoleMsgs=[]
p.on("console",m=>{if(/model|provider|config|fetch|server|error/i.test(m.text())) consoleMsgs.push(m.type()+": "+m.text().slice(0,160))})
await p.goto(`${WEB}/chat`,{waitUntil:"domcontentloaded"})
await p.getByTestId("chat-input").first().waitFor({state:"visible",timeout:30000})
await p.waitForTimeout(5000)
// also check what the model selector shows
await p.getByTestId("model-selector").first().click().catch(()=>{}); await p.waitForTimeout(800)
const menuText=await p.evaluate(()=>document.body.innerText.match(/No models available|gpt-4o|Select a model/g))
console.log("=== ALL /api/v1 calls on /chat load (deduped counts) ===")
const counts={}; all.forEach(c=>{const key=c.replace(/\(\d+b\)/,'');counts[key]=(counts[key]||0)+1})
Object.entries(counts).sort().forEach(([k,n])=>console.log(`  ${n}x  ${k}`))
console.log("\nmetadata called:", paths.has("/llm/models/metadata"))
console.log("models called:", paths.has("/llm/models"))
console.log("\n=== relevant console msgs ===")
consoleMsgs.slice(0,15).forEach(m=>console.log("  "+m))
console.log("\nmodel menu tokens:", menuText)
await b.close()
