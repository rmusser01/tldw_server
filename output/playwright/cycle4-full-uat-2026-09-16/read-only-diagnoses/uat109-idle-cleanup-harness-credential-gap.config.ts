import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config"
const ui = "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui"
const target = ui + "/src/services/__tests__/background-proxy.test.ts"
const probe = `
  it("UAT109 real errored response body has no discarded cleanup rejection", async () => {
    mocks.runtimeId = null
    mocks.storageGet.mockImplementation(async key => key === "tldwConfig" ? {serverUrl:"http://127.0.0.1:19999",authMode:"single-user",apiKey:"synthetic-test-key"} : null)
    const unhandled: unknown[] = []
    const observe = (reason: unknown) => { unhandled.push(reason) }
    process.on("unhandledRejection", observe)
    const fetchSpy = vi.fn(async (_input: unknown, init: RequestInit) => {
      const stream = new ReadableStream({start(controller) {
        init.signal!.addEventListener("abort", () => controller.error(new DOMException("BodyStreamBuffer was aborted", "AbortError")), {once:true})
      }})
      return new Response(stream, {status:200,headers:{"Content-Type":"text/event-stream"}})
    })
    vi.stubGlobal("fetch", fetchSpy)
    try {
      const { bgStream } = await importProxy()
      const consume = async () => { for await (const chunk of bgStream({path:"/api/v1/rag/search/stream",method:"POST",body:{query:"Cedar question"},streamIdleTimeoutMs:10,sanitizeRagProviderStreamError:true})) { void chunk } }
      const handled = await consume().then(() => null, error => error)
      await new Promise(resolve => setTimeout(resolve,30))
      console.log("UAT109_CLEANUP", JSON.stringify({primaryMessage:handled?.message,unhandled:unhandled.map((error:any)=>({name:error?.name,message:error?.message})),fetchCalls:fetchSpy.mock.calls.length}))
      expect(handled?.message).toMatch(/timeout/i)
      expect(unhandled).toEqual([])
    } finally { process.off("unhandledRejection",observe); vi.unstubAllGlobals() }
  })
`
export default {...base,plugins:[{name:"uat109-readonly",enforce:"pre" as const,transform(code:string,id:string){if(id!==target)return;const marker='describe("background proxy fallback safety", () => {';if(!code.includes(marker))throw Error("Missing suite");return{code:code.replace(marker,marker+probe),map:null}}}],test:{...base.test,setupFiles:[ui+"/vitest.setup.ts"],include:[target],testNamePattern:"UAT109 real errored response"}}
