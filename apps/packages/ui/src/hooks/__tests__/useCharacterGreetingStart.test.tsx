import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"
import { useSimpleForm } from "../useSimpleForm"
import { useCharacterGreetingStart } from "../useCharacterGreetingStart"
import { useStoreMessageOption } from "@/store/option"
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
