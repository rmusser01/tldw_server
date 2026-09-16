import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config"
const root = "/Users/macbook-dev/Documents/GitHub/tldw_server2"
const ui = root + "/apps/packages/ui"
const target = ui + "/src/components/Media/__tests__/AnalysisModal.stage3.regression.test.tsx"
const probe = `
  for (const withOwner of [false, true]) {
    it('read-only actual WebUI model selection with store owner=' + withOwner, async () => {
      window.localStorage.clear()
      useStoreMessageOption.setState({selectedModel:'tldw:gemma3:1b'})
      window.localStorage.setItem('selectedModel', JSON.stringify('tldw:gemma3:1b'))
      const chosen='../../../Working/Language_Models/gemma-4-26B-A4B/gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf'
      mocks.getChatModels.mockResolvedValue([{id:'gemma3:1b',name:'gemma3:1b'},{id:chosen,name:chosen}])
      mocks.bgStream.mockImplementation(()=> (async function*(){yield streamChunk('Chosen model answer');yield 'data: [DONE]'})())
      mocks.bgRequest.mockResolvedValue({processing:{analysis:'Chosen model answer'}})
      function ExistingOwner(){ useSelectedModel(); return null }
      const view=render(<>{withOwner && <ExistingOwner/>}<AnalysisModal open onClose={vi.fn()} mediaId={42} mediaContent="Original public source"/></>)
      try {
        const select=screen.getByRole('combobox',{name:'Model'})
        await waitFor(()=>expect(select).toHaveValue('tldw:gemma3:1b'))
        await waitFor(()=>expect(screen.getByRole('option',{name:chosen})).toBeInTheDocument())
        fireEvent.change(select,{target:{value:'tldw:'+chosen}})
        await act(async()=>{await new Promise(resolve=>setTimeout(resolve,20))})
        fireEvent.click(screen.getByRole('button',{name:'Generate Analysis'}))
        await waitFor(()=>expect(mocks.bgStream).toHaveBeenCalled())
        const sent=mocks.bgStream.mock.calls[0][0].body.model
        console.log('MODEL_SELECTION',JSON.stringify({withOwner,selected:(select as HTMLSelectElement).value,persisted:window.localStorage.getItem('selectedModel'),store:useStoreMessageOption.getState().selectedModel,sent}))
        expect(sent).toBe(chosen)
      } finally {view.unmount()}
    })
  }
`
export default {...base,resolve:{alias:{...base.resolve.alias,'@plasmohq/storage/hook':root+'/apps/tldw-frontend/extension/shims/plasmo-storage-hook.tsx','@plasmohq/storage':root+'/apps/tldw-frontend/extension/shims/plasmo-storage.ts'}},plugins:[{name:'readonly-model-selection',enforce:'pre' as const,transform(code:string,id:string){if(id!==target)return;const start=code.indexOf("vi.mock('@plasmohq/storage',");const end=code.indexOf("vi.mock('@/services/background-proxy',",start);if(start<0||end<0)throw Error('Mock boundary missing');code=code.slice(0,start)+code.slice(end);code=code.replace("import React from 'react'","import React from 'react'\nimport { act } from '@testing-library/react'\nimport { useSelectedModel } from '@/hooks/chat/useSelectedModel'\nimport { useStoreMessageOption } from '@/store/option'");const marker="describe('AnalysisModal stage 3 regression coverage', () => {";return{code:code.replace(marker,marker+probe),map:null}}}],test:{...base.test,setupFiles:[ui+'/vitest.setup.ts'],include:[target],testNamePattern:'read-only actual WebUI model selection'}}
