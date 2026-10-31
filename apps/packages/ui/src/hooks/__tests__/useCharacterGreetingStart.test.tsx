import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"
import { useSimpleForm } from "../useSimpleForm"
import { useCharacterGreetingStart } from "../useCharacterGreetingStart"
import { useStoreMessageOption } from "@/store/option"
import { useStoreChatModelSettings } from "@/store/model"
import { useMcpToolsStore } from "@/store/mcp-tools"
import { createDefaultActorSettings } from "@/types/actor"
const getActorSceneForChat = vi.hoisted(() => vi.fn<typeof import("@/services/actor-settings").getActorSceneForChat>())
vi.mock("@/services/actor-settings", () => ({ getActorSceneForChat }))
const requestScope = { config: { serverUrl: "https://greeting-command.example", authMode: "single-user" as const }, userId: 1 }
import { buildGreetingOptionsFromEntries, collectGreetingEntries } from "@/utils/character-greetings"

const option = buildGreetingOptionsFromEntries(collectGreetingEntries({ first_message: "  Raw {{user}}  " }))[0]
const renderStart = (retain = vi.fn(async () => true)) => {
  const submit = vi.fn(async (): Promise<import("../chat/chat-action-utils").ChatSubmitResult> => ({ status: "submitted" }))
  const preflight = vi.fn()
  const controller = { getCurrent: () => ({ status: "idle", owner: null, view: null }), fence: () => () => true,
    draftAccount: { requestScope, isCurrent: () => true } } as unknown as import("../chat/useHistorySelection").HistorySelectionController
  const hook = renderHook(({ isConnectionReady }) => {
    const form = useSimpleForm({ initialValues: { message: "UNSENT\nexact  ", image: "" } })
    return { form, ...useCharacterGreetingStart({ form, retainCurrentDraft: retain, submit,
      ready: true, isConnectionReady, busy: false, readContext: () => [isConnectionReady], preflight, controller, focusComposer: vi.fn() }) }
  }, { initialProps: { isConnectionReady: true } })
  return { ...hook, submit, preflight, retain }
}
beforeEach(() => { getActorSceneForChat.mockReset().mockImplementation(async ({ target }) => ({ target: target!, originalRecord: null, settings: createDefaultActorSettings(), origin: "default", profileUnavailable: false })); useStoreMessageOption.getState().startNewChatDraft(); useStoreMessageOption.setState({ streaming: false, isLoading: false, isProcessing: false, history: [], messages: [], historyId: null, serverChatId: null }) })
it("reads the original saved greeting scene before retaining input and forwards its verified account", async () => {
  const original = { ...createDefaultActorSettings(), notes: "Original greeting scene" }
  let stored = original
  getActorSceneForChat.mockImplementation(async ({ target }) => ({ target: target!, originalRecord: { value: structuredClone(stored), updatedAt: 1 },
    settings: structuredClone(stored), origin: "scene", profileUnavailable: false }))
  let readsAtRetention = -1
  const hook = renderStart(vi.fn(async () => { readsAtRetention = getActorSceneForChat.mock.calls.length; stored = { ...original, notes: "New scene after retention" }; return true }))
  await act(async () => { await hook.result.current.start(option, () => true) })
  expect(readsAtRetention).toBe(1)
  expect(hook.submit).toHaveBeenCalledWith(expect.objectContaining({ actorSceneSend: expect.objectContaining({ requestScope,
    scene: expect.objectContaining({ settings: original }) }) }))
})

it("retains input before submitting only the captured raw greeting seed", async () => {
  const hook = renderStart()
  await act(async () => { await hook.result.current.start(option, () => true) })
  expect(hook.retain).toHaveBeenCalledOnce()
  expect(hook.submit).toHaveBeenCalledWith({ message: "", image: "", greetingSeed: option.seed, greetingIsCurrent: expect.any(Function), actorSceneSend: expect.objectContaining({ requestScope }) })
  expect(hook.submit.mock.invocationCallOrder[0]).toBeGreaterThan(hook.retain.mock.invocationCallOrder[0])
  expect(hook.result.current.form.getValues().message).toBe("UNSENT\nexact  ")
})
it("failed draft ACK never creates a chat or clears input", async () => {
  const hook = renderStart(vi.fn(async () => false))
  await act(async () => { await expect(hook.result.current.start(option, () => true)).rejects.toThrow() })
  expect(hook.submit).not.toHaveBeenCalled()
  expect(hook.result.current.form.getValues().message).toBe("UNSENT\nexact  ")
})
it.each(["edit", "new", "choice"])("a held draft ACK rejects %s and keeps current input", async boundary => {
  let release!: (value: boolean) => void, choice = true
  const hook = renderStart(vi.fn(() => new Promise<boolean>(resolve => { release = resolve })))
  let pending!: Promise<void>
  await act(async () => { pending = hook.result.current.start(option, () => choice); await waitFor(() => expect(hook.retain).toHaveBeenCalledOnce()) })
  act(() => {
    if (boundary === "edit") hook.result.current.form.setFieldValue("message", "NEWER TEXT")
    else if (boundary === "new") useStoreMessageOption.getState().startNewChatDraft()
    else choice = false
  })
  await act(async () => { release(true); await expect(pending).rejects.toThrow() })
  expect(hook.submit).not.toHaveBeenCalled()
  expect(hook.result.current.form.getValues().message).toBe(boundary === "edit" ? "NEWER TEXT" : "UNSENT\nexact  ")
})

it("a held ACK never revives after disconnect and reconnect; a new explicit click can start", async () => {
  let release!: (value: boolean) => void
  const retain = vi.fn(() => new Promise<boolean>(resolve => { release = resolve }))
  const hook = renderStart(retain)
  let pending!: Promise<void>
  await act(async () => { pending = hook.result.current.start(option, () => true); await waitFor(() => expect(hook.retain).toHaveBeenCalledOnce()) })
  hook.rerender({ isConnectionReady: false })
  hook.rerender({ isConnectionReady: true })
  await act(async () => { release(true); await expect(pending).rejects.toThrow() })
  expect(hook.submit).not.toHaveBeenCalled()
  expect(hook.result.current.form.getValues().message).toBe("UNSENT\nexact  ")
  retain.mockResolvedValueOnce(true)
  await act(async () => { await hook.result.current.start(option, () => true) })
  expect(hook.submit).toHaveBeenCalledOnce()
})

it("shows scene-specific recovery for a stable failed scene-read outcome and keeps unsent input", async () => {
  const hook = renderStart()
  hook.submit.mockResolvedValueOnce({
    status: "failed",
    errorMessage: "actor_settings_unavailable"
  })
  await act(async () => {
    await expect(hook.result.current.start(option, () => true)).rejects.toThrow(
      "Scene settings could not be loaded"
    )
  })
  expect(hook.result.current.form.getValues().message).toBe("UNSENT\nexact  ")
})


it.each([["scene", "same"], ["retention", "same"], ["scene", "aba"], ["retention", "aba"]] as const)("model intent retires a held %s greeting after %s selection without changing its final model", async (boundary, intent) => {
  const model = "greeting-command-model"
  useStoreMessageOption.getState().setSelectedModel(model, { storageHandled: true })
  let release!: () => void
  const gate = new Promise<void>(resolve => { release = resolve })
  getActorSceneForChat.mockImplementation(async ({ target }) => {
    if (boundary === "scene") await gate
    return { target: target!, originalRecord: null, settings: createDefaultActorSettings(), origin: "default", profileUnavailable: false }
  })
  const hook = renderStart(vi.fn(async () => { if (boundary === "retention") await gate; return true }))
  let pending!: Promise<void>
  await act(async () => {
    pending = hook.result.current.start(option, () => true)
    await waitFor(() => expect(boundary === "scene" ? getActorSceneForChat : hook.retain).toHaveBeenCalledOnce())
  })
  const source = useStoreMessageOption.getState()
  const settings = useStoreChatModelSettings.getState()
  const tools = useMcpToolsStore.getState()
  act(() => {
    if (intent === "aba") useStoreMessageOption.getState().setSelectedModel("another-greeting-model", { storageHandled: true })
    useStoreMessageOption.getState().setSelectedModel(model, { storageHandled: true })
  })
  expect(useStoreMessageOption.getState().selectedModelRevision).toBe(source.selectedModelRevision + (intent === "aba" ? 2 : 1))
  expect(useStoreMessageOption.getState().selectedModel).toBe(source.selectedModel)
  expect(useStoreMessageOption.getState().chatDraftAddress).toBe(source.chatDraftAddress)
  expect(useStoreMessageOption.getState().chatDraftRevision).toBe(source.chatDraftRevision)
  expect(useStoreChatModelSettings.getState()).toBe(settings)
  expect(useMcpToolsStore.getState()).toBe(tools)
  await act(async () => { release(); await expect(pending).rejects.toThrow() })
  expect(hook.submit).not.toHaveBeenCalled()
  expect(hook.result.current.form.getValues().message).toBe("UNSENT\nexact  ")
})

it.each(["scene", "retention"] as const)("unchanged held %s greeting forwards the original scene, account and seed", async boundary => {
  useStoreMessageOption.getState().setSelectedModel("greeting-command-model", { storageHandled: true })
  const original = { ...createDefaultActorSettings(), notes: "Original held greeting scene" }
  let release!: () => void
  const gate = new Promise<void>(resolve => { release = resolve })
  getActorSceneForChat.mockImplementation(async ({ target }) => {
    if (boundary === "scene") await gate
    return { target: target!, originalRecord: { value: structuredClone(original), updatedAt: 1 }, settings: structuredClone(original), origin: "scene", profileUnavailable: false }
  })
  const hook = renderStart(vi.fn(async () => { if (boundary === "retention") await gate; return true }))
  let pending!: Promise<void>
  await act(async () => {
    pending = hook.result.current.start(option, () => true)
    await waitFor(() => expect(boundary === "scene" ? getActorSceneForChat : hook.retain).toHaveBeenCalledOnce())
  })
  await act(async () => { release(); await pending })
  expect(hook.submit).toHaveBeenCalledOnce()
  expect(hook.submit).toHaveBeenCalledWith(expect.objectContaining({ message: "", image: "", greetingSeed: option.seed,
    actorSceneSend: expect.objectContaining({ requestScope, scene: expect.objectContaining({ settings: original }) }) }))
  expect(hook.result.current.form.getValues().message).toBe("UNSENT\nexact  ")
})
