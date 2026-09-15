import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config"
const ui = "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui"
const target = ui + "/src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx"
const imports = `
import { useStoreMessageOption } from "@/store/option"
import { useServerChatLoader } from "@/hooks/chat/useServerChatLoader"
import { resolveEffectiveAssistantState, effectiveAssistantStateToSelection } from "@/hooks/chat/effective-assistant-state"
const probeEnsureHistory = async () => null
const probeNotification = { error: vi.fn() }
const probeTranslation = ((key: string) => key) as any
const probeSetAssistant = async (selection: any, options?: { isCurrent?: () => boolean }) => {
  if (options?.isCurrent && !options.isCurrent()) return
  useStoreMessageOption.setState({ _probeAssistant: selection } as any)
}
vi.mock("@/hooks/useSelectedAssistant", () => ({ useSelectedAssistant: () => [useStoreMessageOption((state: any) => state._probeAssistant), probeSetAssistant] }))
vi.mock("@/services/service-prompts", () => ({ loadServicePromptSnapshot: async (_ids: unknown, { signal }: { signal: AbortSignal }) => ({
  scopeKey: "scope-A", scopeSignal: signal, scopeInvalidatedSignal: signal,
  requestScope: { config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "A" }, release: vi.fn()
}) }))
vi.mock("@/services/chat-settings", () => ({ syncChatSettingsForServerChat: async () => null }))
`
const adapter = `useMessageOption: () => {
  const store = useStoreMessageOption()
  useServerChatLoader({ ensureServerChatHistoryId: probeEnsureHistory, notification: probeNotification, t: probeTranslation })
  const draft = (store as any)._probeAssistant
  const resolved = resolveEffectiveAssistantState({ tracked: { assistantKind: store.serverChatAssistantKind, assistantId: store.serverChatAssistantId, characterId: store.serverChatCharacterId }, draftSelection: draft })
  return { ...messageOptionState.value, ...store, selectedAssistant: effectiveAssistantStateToSelection(resolved) ?? draft, setSelectedAssistant: probeSetAssistant }
}`
const probe = `
it("UAT093 keeps the explicit saved target while its profile is loading over a prior character", async () => {
  useStoreMessageOption.setState({ serverChatId: null, historyId: null, messages: [], history: [], serverChatMetaLoaded: false, serverChatCharacterId: null, serverChatAssistantKind: null, serverChatAssistantId: null, temporaryChat: true, streaming: false, isProcessing: false, _probeAssistant: { kind: "character", id: "4", name: "Cedar", metadata: { selectionMode: "tracked" } } } as any)
  let finishProfile!: (value: unknown) => void
  const profile = new Promise(resolve => { finishProfile = resolve })
  let finishMessages!: (value: any) => void
  const messageResponse = new Promise(resolve => { finishMessages = resolve })
  Object.assign(tldwClientState, {
    ensureConfigForRequest: vi.fn(async () => ({ serverUrl: "http://chat.test", authMode: "multi-user", accessToken: 'test.' + btoa(JSON.stringify({ sub: 'A' })) + '.signature' })),
    getChat: vi.fn(async () => ({ id: "robot", title: "Robot chat", character_id: 5, assistant_kind: "character", assistant_id: "5", source: "webui-character-chat", scope_type: "global" })),
    listChatMessages: vi.fn(async () => messageResponse)
  })
  tldwClientState.getCharacter.mockImplementation(() => profile as any)
  window.history.pushState({}, "", "/chat?settingsServerChatId=robot")
  const changes: any[] = []
  const stop = useStoreMessageOption.subscribe(state => changes.push({ id: state.serverChatId, meta: state.serverChatMetaLoaded, character: state.serverChatCharacterId, load: state.serverChatLoadState, messages: state.messages.length }))
  const view = render(<Playground />)
  await waitFor(() => expect(tldwClientState.getCharacter).toHaveBeenCalledWith(5, expect.anything()))
  await act(async () => { await new Promise(resolve => setTimeout(resolve, 30)) })
  await act(async () => { finishProfile({ id: 5, name: "Robot" }); finishMessages([{ id: "answer", role: "assistant", content: "BEEP BOOP", version: 1 }]); await profile; await new Promise(resolve => setTimeout(resolve, 300)) })
  console.log("probe-state", JSON.stringify(changes)); stop(); view.unmount()
  expect(changes.filter(row => row.meta && row.character === 5).length).toBeGreaterThan(0)
  expect(useStoreMessageOption.getState().messages.map(message => message.message)).toContain("BEEP BOOP")
  expect((useStoreMessageOption.getState() as any)._probeAssistant.id).toBe("5")
})
`
export default { ...base, plugins: [{ name: "uat093-metadata-selection", enforce: "pre" as const, transform(code: string, id: string) {
  if (id !== target) return
  code = code.replace(/vi\.mock\("@\/store\/option", \(\) => \(\{[\s\S]*?\}\)\)\n/, '')
  code = code.replace('useMessageOption: () => messageOptionState.value', adapter)
  return { code: imports + code + probe, map: null }
} }], test: { ...base.test, setupFiles: [ui + "/vitest.setup.ts"], include: [target] } }
