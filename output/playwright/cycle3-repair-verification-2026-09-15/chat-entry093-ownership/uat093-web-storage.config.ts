import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config"
const ui = "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui"
const web = "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend"
const target = ui + "/src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx"
const probe = `
describe("UAT093 actual WebUI storage boundary", () => {
  it.each(["immediate", "held-profile", "newer-picker"])("preserves saved entry through independent hook state: %s", async variant => {
    realLoader.enabled = true
    realLoader.additionalLoader = true
    useMessageOptionMock.mockImplementation(useRealServerConversation)
    const { selectedAssistantStorage } = await import("@/utils/selected-assistant-storage")
    window.localStorage.clear()
    await selectedAssistantStorage.set("selectedAssistant", { kind: "character", id: "4", name: "Cedar", metadata: { selectionMode: "tracked" } })
    useStoreMessageOption.setState({ serverChatId: null, historyId: null, messages: [], history: [], serverChatMetaLoaded: false, serverChatCharacterId: null, serverChatAssistantKind: null, serverChatAssistantId: null, temporaryChat: true, streaming: false, isProcessing: false })
    let release!: () => void
    const pending = new Promise<void>(resolve => { release = resolve })
    tldwClientState.getChat.mockResolvedValue({ id: "robot", title: "Robot chat", character_id: 5, assistant_kind: "character", assistant_id: "5", source: "webui-character-chat", scope_type: "global" })
    tldwClientState.listChatMessages.mockResolvedValue([{ id: "answer", role: "assistant", content: "BEEP BOOP", version: 1 }])
    tldwClientState.getCharacter.mockImplementation(async () => { if (variant !== "immediate") await pending; return { id: 5, name: "Robot" } })
    window.history.pushState({}, "", "/chat?settingsServerChatId=robot")
    const changes: unknown[] = []
    const stop = useStoreMessageOption.subscribe(state => changes.push({ id: state.serverChatId, meta: state.serverChatMetaLoaded, character: state.serverChatCharacterId, load: state.serverChatLoadState, messages: state.messages.length }))
    const view = render(<Playground />)
    try {
      await waitFor(() => expect(tldwClientState.getCharacter).toHaveBeenCalled())
      if (variant === "newer-picker") await act(async () => { await setTestAssistant({ kind: "character", id: "7", name: "New choice", metadata: { selectionMode: "tracked" } }) })
      await act(async () => { release(); await pending; await new Promise(resolve => setTimeout(resolve, 400)) })
      console.log("WEB_STORAGE_TRACE", variant, JSON.stringify(changes), "stored", JSON.stringify(await selectedAssistantStorage.get("selectedAssistant")))
      if (variant === "newer-picker") {
        expect(await selectedAssistantStorage.get("selectedAssistant")).toMatchObject({ id: "7", name: "New choice" })
        expect(useStoreMessageOption.getState().serverChatId).toBeNull()
      } else {
        expect(useStoreMessageOption.getState().serverChatId).toBe("robot")
        expect(useStoreMessageOption.getState().messages.map(message => message.message)).toContain("BEEP BOOP")
        expect(await selectedAssistantStorage.get("selectedAssistant")).toMatchObject({ id: "5", name: "Robot" })
      }
    } finally { release(); stop(); view.unmount() }
  })
})
`
export default { ...base, resolve: { ...base.resolve, alias: [
  { find: "@plasmohq/storage/hook", replacement: web + "/extension/shims/plasmo-storage-hook.tsx" },
  { find: "@plasmohq/storage", replacement: web + "/extension/shims/plasmo-storage.ts" },
  ...Object.entries(base.resolve.alias).map(([find,replacement]) => ({find,replacement}))
]}, plugins: [{ name: "uat093-real-web-storage", enforce: "pre" as const, transform(code: string, id: string) {
  if (id !== target) return
  const start = code.indexOf('vi.mock("@plasmohq/storage/hook",')
  const end = code.indexOf('vi.mock("@/hooks/useMediaQuery",', start)
  if (start < 0 || end < 0) throw new Error("Expected controlled storage mock")
  code = code.slice(0,start) + code.slice(end)
  code = code.replace('realLoader.additionalLoader ? <AdditionalServerLoader /> : null', 'realLoader.additionalLoader ? <>{Array.from({ length: 5 }, (_, index) => <AdditionalServerLoader key={index} />)}</> : null')
  return {code: code + probe, map:null}
}}], test: { ...base.test, setupFiles: [ui+"/vitest.setup.ts"], include: [target] } }
