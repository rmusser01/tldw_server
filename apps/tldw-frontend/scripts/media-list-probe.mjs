import { chromium } from "@playwright/test"
const WEB=process.env.WEB_URL||"http://localhost:8080", S=process.env.SERVER_URL||"http://127.0.0.1:8000", K=process.env.TLDW_API_KEY||"THIS-IS-A-SECURE-KEY-123-FAKE-KEY"
const b=await chromium.launch(); const c=await b.newContext({viewport:{width:1440,height:900}})
await c.addInitScript(({s,k})=>{localStorage.setItem("tldwConfig",JSON.stringify({serverUrl:s,authMode:"single-user",apiKey:k,accessToken:""}));for(const[a,v]of Object.entries({isMigrated:"true",serverUrl:s,tldwServerUrl:s,authMode:"single-user",apiKey:k,__tldw_first_run_complete:"true",__tldw_test_bypass:"true",__tldw_allow_offline:"true"}))localStorage.setItem(a,v)},{s:S,k:K})
const p=await c.newPage()
const t0=Date.now(); const calls=[]
p.on("request",r=>{const u=r.url(); if(/\/api\/v1\/media(\/)?(\?|$)/.test(u)){calls.push({t:Date.now()-t0,m:r.method(),u:u.split("/api/v1")[1]})}})
await p.goto(`${WEB}/media`,{waitUntil:"domcontentloaded"})
await p.getByTestId("media-search-input").first().waitFor({state:"visible",timeout:30000})
await p.waitForTimeout(5000)
console.log("=== /media list requests on PASSIVE load (no interaction) ===")
calls.forEach(c=>console.log(`  ${String(c.t).padStart(5)}ms  ${c.m} ${c.u}`))
console.log("total:",calls.length)
await b.close()
