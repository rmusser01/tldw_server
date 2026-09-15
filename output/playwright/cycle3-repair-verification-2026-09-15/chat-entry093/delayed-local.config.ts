import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config"
const ui = "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui"
const target = ui + "/src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx"
const probe = `
  it("independent: does not publish a held local settings return after its route unmounts", async () => {
    const { PageAssistDatabase } = await import("@/db/dexie/chat")
    const modelSettings = await import("@/services/model-settings")
    vi.spyOn(modelSettings, "lastUsedChatModelEnabled").mockResolvedValue(false)
    let release!: () => void
    const held = new Promise<void>(resolve => { release = resolve })
    const read = vi.spyOn(PageAssistDatabase.prototype, "getChatHistory").mockImplementation(async () => { await held; return [] })
    vi.spyOn(PageAssistDatabase.prototype, "getHistoryInfo").mockResolvedValue(null)
    window.history.pushState({}, "", "/chat?settingsHistoryId=owned-local")
    const view = render(<Playground />)
    await waitFor(() => expect(read).toHaveBeenCalled())
    view.unmount()
    messageOptionState.value.setMessages.mockClear()
    messageOptionState.value.setHistoryId.mockClear()
    await act(async () => { release(); await held; await Promise.resolve() })
    expect(messageOptionState.value.setMessages).not.toHaveBeenCalled()
    expect(messageOptionState.value.setHistoryId).not.toHaveBeenCalled()
  })
`
export default { ...base, plugins: [{ name: "uat093-delayed-local", enforce: "pre" as const, transform(code: string, id: string) {
  if (id !== target) return
  code=code.replace('getPromptById: vi.fn(async () => null),', 'getPromptById: vi.fn(async () => null), getSessionFiles: vi.fn(async () => []),')
  const end = code.lastIndexOf("})")
  return { code: code.slice(0, end) + probe + code.slice(end), map: null }
} }], test: { ...base.test, setupFiles: [ui + "/vitest.setup.ts"], include: [target] } }
