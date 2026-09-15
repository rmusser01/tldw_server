import original from "/private/tmp/uat093-web-storage.config"
const ui = "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui"
export default { ...original, plugins: [...original.plugins, {
  name: "isolate-web-storage-probe-and-record-mismatch", enforce: "pre" as const,
  transform(code: string, id: string) {
    if (id === ui + "/src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx") {
      const marker = 'describe("UAT093 actual WebUI storage boundary", () => {'
      if (!code.includes(marker)) throw new Error("Expected original probe")
      return {code:code.replace(marker, marker + `
      beforeEach(() => {
        vi.clearAllMocks()
        realLoader.invalidated = new AbortController()
        realLoader.setSelection = null
        realLoader.storageBarrier = null
        usePlaygroundSessionStore.setState({ restoreState: "idle", restoreRevision: 0, didAttemptRestore: false })
      })
      `), map:null}
    }
    if (id === ui + "/src/components/Option/Playground/Playground.tsx") {
      const marker = '    setHistoryId(null, { preserveServerChatId: false });'
      const index = code.indexOf(marker)
      if (index < 0) throw new Error("Expected mismatch reset")
      return {code:code.slice(0,index) + '    console.log("MISMATCH_AT_CLEAR", { serverChatId, serverChatCharacterId, selectedTrackedCharacterId, selectedAssistant, activeCharacterSelection });\n' + code.slice(index), map:null}
    }
  }
}] }
