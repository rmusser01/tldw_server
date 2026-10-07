import { act, renderHook } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"
import { useSimpleForm } from "../useSimpleForm"
import { useCharacterGreetingStart } from "../useCharacterGreetingStart"
import { useStoreMessageOption } from "@/store/option"
import { buildGreetingOptionsFromEntries, collectGreetingEntries } from "@/utils/character-greetings"

const option = buildGreetingOptionsFromEntries(collectGreetingEntries({ first_message: "  Raw {{user}}  " }))[0]
const renderStart = (retain = vi.fn(async () => true)) => {
  const submit = vi.fn(async (): Promise<import("../chat/chat-action-utils").ChatSubmitResult> => ({ status: "submitted" }))
  const preflight = vi.fn()
  const controller = { getCurrent: () => ({ status: "idle", owner: null }), fence: () => () => true } as unknown as import("../chat/useHistorySelection").HistorySelectionController
  const hook = renderHook(({ isConnectionReady }) => {
    const form = useSimpleForm({ initialValues: { message: "UNSENT\nexact  ", image: "" } })
    return { form, ...useCharacterGreetingStart({ form, retainCurrentDraft: retain, submit,
      ready: true, isConnectionReady, busy: false, readContext: () => [isConnectionReady], preflight, controller, focusComposer: vi.fn() }) }
  }, { initialProps: { isConnectionReady: true } })
  return { ...hook, submit, preflight, retain }
}
beforeEach(() => { useStoreMessageOption.getState().startNewChatDraft(); useStoreMessageOption.setState({ streaming: false, isLoading: false, isProcessing: false, history: [], messages: [], historyId: null, serverChatId: null }) })
it("retains input before submitting only the captured raw greeting seed", async () => {
  const hook = renderStart()
  await act(async () => { await hook.result.current.start(option, () => true) })
  expect(hook.retain).toHaveBeenCalledOnce()
  expect(hook.submit).toHaveBeenCalledWith({ message: "", image: "", greetingSeed: option.seed, greetingIsCurrent: expect.any(Function) })
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
  act(() => { pending = hook.result.current.start(option, () => choice) })
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
  act(() => { pending = hook.result.current.start(option, () => true) })
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
