import original from "/private/tmp/uat093-web-storage-simple-loader.config"
const target = "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx"
export default {...original,plugins:[...original.plugins,{
 name:"uat093-actual-cross-tab-storage-control", enforce:"pre" as const, transform(code:string,id:string){
  if(id!==target)return
  code=code.replace('["immediate", "held-profile", "newer-picker"]','["immediate", "held-profile", "newer-picker", "cross-tab"]')
  code=code.replace('  return { ...messageOptionState.value, ...store, selectedAssistant: effectiveAssistantStateToSelection(resolved)', '  ;(globalThis as any).__uat093ResolvedAssistant = resolved\n  return { ...messageOptionState.value, ...store, selectedAssistant: effectiveAssistantStateToSelection(resolved)')
  const marker='      console.log("WEB_STORAGE_TRACE", variant,'
  const i=code.indexOf(marker)
  if(i<0)throw new Error("Expected probe trace")
  code=code.slice(0,i)+`
      if (variant === "cross-tab") {
        const next = JSON.stringify({ kind: "character", id: "7", name: "Other tab choice", metadata: { selectionMode: "tracked" } })
        await act(async () => {
          const previous = window.localStorage.getItem("selectedAssistant")
          window.localStorage.setItem("selectedAssistant", next)
          window.dispatchEvent(new StorageEvent("storage", {key: "selectedAssistant", oldValue: previous, newValue: next, storageArea: window.localStorage}))
        })
        console.log("CROSS_TAB_RENDER", (globalThis as any).__uat093ResolvedAssistant)
        expect(useStoreMessageOption.getState().serverChatId).toBe("robot")
        expect(useStoreMessageOption.getState().messages.map(message => message.message)).toContain("BEEP BOOP")
        expect((globalThis as any).__uat093ResolvedAssistant.id).toBe("5")
        return
      }
  `+code.slice(i)
  return {code,map:null}
 }
}]}
