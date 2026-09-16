import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/vitest.config"
const web='/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend'
const target=web+'/__tests__/components/notification-lifecycle-provider.test.tsx'
const probe=`
  it('read-only hidden tab does not reserve a long-lived stream', async()=>{
    vi.spyOn(document,'visibilityState','get').mockReturnValue('hidden')
    const view=renderProvider()
    try {
      await act(async()=>{await new Promise(resolve=>setTimeout(resolve,20))})
      console.log('HIDDEN_STREAM', JSON.stringify({state:document.visibilityState,streamStarts:mocks.subscribeNotificationsStream.mock.calls.length}))
      expect(mocks.subscribeNotificationsStream).not.toHaveBeenCalled()
    } finally {view.unmount();vi.restoreAllMocks()}
  })
  it('read-only visible tab releases stream when hidden', async()=>{
    const visibility=vi.spyOn(document,'visibilityState','get').mockReturnValue('visible')
    const unsubscribe=vi.fn()
    mocks.subscribeNotificationsStream.mockReturnValue(unsubscribe)
    const view=renderProvider()
    try {
      await waitFor(()=>expect(mocks.subscribeNotificationsStream).toHaveBeenCalledTimes(1))
      visibility.mockReturnValue('hidden')
      await act(async()=>{document.dispatchEvent(new Event('visibilitychange'));await Promise.resolve()})
      console.log('HIDDEN_RELEASE',JSON.stringify({state:document.visibilityState,releases:unsubscribe.mock.calls.length}))
      expect(unsubscribe).toHaveBeenCalledTimes(1)
    } finally {view.unmount();vi.restoreAllMocks()}
  })
`
export default {...base,plugins:[...((base.plugins as any[])||[]),{name:'readonly-hidden-stream',enforce:'pre' as const,transform(code:string,id:string){if(id!==target)return;const marker='describe("NotificationLifecycleProvider", () => {';if(!code.includes(marker))throw Error('Suite boundary missing');return{code:code.replace(marker,marker+probe),map:null}}}],test:{...base.test,setupFiles:[web+'/vitest.setup.ts'],include:[target],testNamePattern:'read-only (hidden|visible) tab'}}
