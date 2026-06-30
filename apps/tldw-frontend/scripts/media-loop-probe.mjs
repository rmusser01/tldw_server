import { chromium } from "@playwright/test"
const WEB=process.env.WEB_URL||"http://localhost:8080", S=process.env.SERVER_URL||"http://127.0.0.1:8000", K=process.env.TLDW_API_KEY||"THIS-IS-A-SECURE-KEY-123-FAKE-KEY"
async function run(label){
  const b=await chromium.launch(); const c=await b.newContext({viewport:{width:1440,height:900}})
  await c.addInitScript(({s,k})=>{localStorage.setItem("tldwConfig",JSON.stringify({serverUrl:s,authMode:"single-user",apiKey:k,accessToken:""}));for(const[a,v]of Object.entries({isMigrated:"true",serverUrl:s,tldwServerUrl:s,authMode:"single-user",apiKey:k,__tldw_first_run_complete:"true",__tldw_test_bypass:"true",__tldw_allow_offline:"true"}))localStorage.setItem(a,v)},{s:S,k:K})
  const p=await c.newPage()
  let loopErrs=0, progress=0
  p.on("console",m=>{if(m.type()==="error"&&/Maximum update depth/i.test(m.text()))loopErrs++})
  p.on("request",r=>{if(/\/api\/v1\/media\/\d+\/progress/.test(r.url()))progress++})
  await p.goto(`${WEB}/media`,{waitUntil:"domcontentloaded"})
  await p.getByTestId("media-search-input").first().waitFor({state:"visible",timeout:30000})
  await p.waitForTimeout(3500)
  // search to change displayResults
  const si=p.getByTestId("media-search-input").first(); await si.fill("the")
  const sub=p.getByTestId("media-search-submit").first(); if(await sub.count())await sub.click(); else await p.keyboard.press("Enter")
  await p.waitForTimeout(4000)
  console.log(`${label}: maxUpdateDepthErrors=${loopErrs} progressRequests=${progress}`)
  await b.close()
}
for(let i=1;i<=3;i++) await run("run"+i)
