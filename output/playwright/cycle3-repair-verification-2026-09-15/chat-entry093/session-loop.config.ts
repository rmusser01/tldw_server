import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config"
const ui = "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui"
const target = ui + "/src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx"
const probe = `
  it("independent: settles canonical settings entry with the real session subscription and loader", async () => {
    messageOptionState.value.serverChatId = "settings-chat"
    window.history.pushState({}, "", "/chat?settingsServerChatId=settings-chat")
    const view = render(<Playground />)
    await waitFor(() => expect(screen.getByTestId("playground-chat")).toBeInTheDocument())
    expect(usePlaygroundSessionStore.getState().restoreRevision).toBeLessThan(5)
    view.unmount()
  })
`
export default { ...base, plugins: [{ name: "uat093-real-session-loop", enforce: "pre" as const, transform(code: string, id: string) {
  if (id !== target) return
  code = code.replace('usePlaygroundSessionPersistence: () => sessionPersistenceState.value', 'usePlaygroundSessionPersistence: () => { usePlaygroundSessionStore(); return sessionPersistenceState.value }')
  code = code.replace('vi.mock("@/hooks/useLoadLocalConversation", () => ({\n  useLoadLocalConversation: () => loadLocalConversationState.value\n}))', '')
  const end = code.lastIndexOf("})")
  return { code: code.slice(0, end) + probe + code.slice(end), map: null }
} }], test: { ...base.test, setupFiles: [ui + "/vitest.setup.ts"], include: [target] } }
