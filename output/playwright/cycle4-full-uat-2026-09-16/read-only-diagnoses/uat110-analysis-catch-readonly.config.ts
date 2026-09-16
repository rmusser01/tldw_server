import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config"
const ui = "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui"
const target = ui + "/src/components/Media/__tests__/AnalysisModal.stage3.regression.test.tsx"
const probe = `
  it('UAT110 handled provider rejection does not reach Next console-error overlay channel', async () => {
    const logger = vi.spyOn(console,'error').mockImplementation(()=>{})
    state.selectedModel = 'test-model'
    const failure = new Error('The chat service provider is currently unavailable. (POST /api/v1/chat/completions)')
    mocks.bgStream.mockImplementation(()=> (async function*(){throw failure})())
    mocks.bgRequest.mockRejectedValue(failure)
    const onClose=vi.fn(), onGenerated=vi.fn()
    mocks.handledPromises.length=0
    try {
      render(<AnalysisModal open onClose={onClose} mediaId={42} mediaContent="Original source" onAnalysisGenerated={onGenerated}/>)
      const button=screen.getByRole('button',{name:'Generate Analysis'})
      await waitFor(()=>expect(button).not.toBeDisabled())
      fireEvent.click(button)
      await waitFor(()=>expect(mocks.messageError).toHaveBeenCalledWith('Failed to generate analysis'))
      const settled=await Promise.allSettled(mocks.handledPromises)
      console.log('UAT110_HANDLER',JSON.stringify({settled:settled.map(r=>r.status),localErrors:mocks.messageError.mock.calls.map(c=>c[0]),consoleErrors:logger.mock.calls.map(c=>({label:c[0],error:c[1] instanceof Error?c[1].message:null})),onClose:onClose.mock.calls.length,onGenerated:onGenerated.mock.calls.length,buttonDisabled:(button as HTMLButtonElement).disabled,requestPaths:mocks.bgRequest.mock.calls.map(c=>c[0]?.path)}))
      expect(settled.map(r=>r.status)).toEqual(['fulfilled'])
      expect(onClose).not.toHaveBeenCalled()
      expect(onGenerated).not.toHaveBeenCalled()
      expect(button).not.toBeDisabled()
      expect(logger).not.toHaveBeenCalled()
    } finally {logger.mockRestore()}
  })
`
export default {...base,plugins:[{name:"uat110-readonly",enforce:"pre" as const,transform(code:string,id:string){if(id!==target)return;code=code.replace('bgRequest: vi.fn(),','handledPromises: [] as Promise<unknown>[],\n  bgRequest: vi.fn(),');code=code.replace('onClick={onClick}\n      disabled=', 'onClick={(event:any)=>{const result=onClick?.(event);if(result?.then)mocks.handledPromises.push(result)}}\n      disabled=');const marker="describe('AnalysisModal stage 3 regression coverage', () => {";if(!code.includes(marker))throw Error('Missing suite');return{code:code.replace(marker,marker+probe),map:null}}}],test:{...base.test,setupFiles:[ui+"/vitest.setup.ts"],include:[target],testNamePattern:"UAT110 handled provider"}}
