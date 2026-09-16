import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config"
const ui="/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui"
const target=ui+"/src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.streaming.test.tsx"
const probe=`
  it("UAT109 handled timeout does not enter Next console-error overlay channel",async()=>{
    const logger=vi.spyOn(console,"error").mockImplementation(()=>{})
    ragSearchStreamMock.mockImplementation(async function*(){throw new Error("RAG search timed out. Try again.")})
    try {
      render(<KnowledgeQAProvider><ContextProbe/></KnowledgeQAProvider>)
      await waitFor(()=>expect(latestContext).not.toBeNull())
      await act(async()=>{await latestContext!.selectThread("local-uat109-timeout")})
      act(()=>latestContext!.setQuery("Own Cedar question"))
      let fulfilled=false
      await act(async()=>{await latestContext!.search();fulfilled=true})
      console.log("UAT109_QA_HANDLER",JSON.stringify({fulfilled,error:latestContext!.error,query:latestContext!.query,isSearching:latestContext!.isSearching,consoleErrors:logger.mock.calls,fallbackCalls:ragSearchMock.mock.calls.length}))
      expect(fulfilled).toBe(true)
      expect(latestContext!.error).toMatch(/timed out/i)
      expect(latestContext!.isSearching).toBe(false)
      expect(ragSearchMock).not.toHaveBeenCalled()
      expect(logger).not.toHaveBeenCalled()
    } finally {logger.mockRestore()}
  })
`
export default {...base,plugins:[{name:"uat109-handler-readonly",enforce:"pre" as const,transform(code:string,id:string){if(id!==target)return;const marker='describe("KnowledgeQAProvider streaming search", () => {';if(!code.includes(marker))throw Error("Missing suite");return{code:code.replace(marker,marker+probe),map:null}}}],test:{...base.test,setupFiles:[ui+"/vitest.setup.ts"],include:[target],testNamePattern:"UAT109 handled timeout"}}
