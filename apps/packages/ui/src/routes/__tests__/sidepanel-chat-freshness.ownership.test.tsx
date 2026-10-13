import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, expect, it, vi } from "vitest"
import type { HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import { useStoreMessageOption } from "@/store/option"
import { FOCUS_CHECK_DEBOUNCE_MS, useSidepanelChatFreshness } from "../sidepanel-chat-freshness"

const io = vi.hoisted(() => ({ capture: vi.fn() }))
vi.mock("@/services/chat-history-selection", async original => ({
  ...await original<typeof import("@/services/chat-history-selection")>(),
  captureHistorySnapshot: (...args: unknown[]) => io.capture(...args)
}))

const manifest = { status: "captured", snapshot: { nodes: [{ id: "tip", parent_id: null }] } }
const mount = (nodes: Array<{ id: string; parent_id: string | null }> = manifest.snapshot.nodes) => {
  const capture = { ...manifest, snapshot: { nodes } }
  let tab = "source"
  let epoch = 0
  let owner = { kind: "native", conversation_id: "source-chat", validate_lease: () => true }
  const view = { cursor: { kind: "after_message", message_id: "tip" } }
  const controller = {
    getCurrent: () => ({ status: "ready", owner, view, capture }),
    fence: () => { const source = epoch; return () => epoch === source },
    getSignal: () => new AbortController().signal,
    choose: vi.fn()
  } as unknown as HistorySelectionController
  const hook = renderHook(() => useSidepanelChatFreshness({ historySelection: controller, checkableTabId: () => tab }))
  const depart = () => { tab = "destination"; epoch++; owner = { ...owner, conversation_id: "destination-chat" } }
  return { ...hook, depart, controller, replaceOwner: () => { epoch++; owner = { ...owner, conversation_id: "replacement-chat" } } }
}

beforeEach(() => {
  io.capture.mockReset()
  useStoreMessageOption.getState().startNewChatDraft(false)
})
afterEach(() => vi.useRealTimers())

it("does not broadcast a departed send into the empty replacement composer", async () => {
  let finish!: (value: unknown) => void
  io.capture.mockImplementation(() => new Promise(resolve => { finish = resolve }))
  const hook = mount()
  const restore = vi.fn()
  window.addEventListener("tldw:set-composer-message", restore)
  try {
    let pending!: ReturnType<typeof hook.result.current.beforeSend>
    await act(async () => { pending = hook.result.current.beforeSend({ message: "DEPARTED SOURCE" }); await Promise.resolve() })
    hook.depart()
    let outcome: unknown
    await act(async () => { finish(manifest); outcome = await pending })
    expect(outcome).toEqual({ proceed: false, refreshed: false })
    expect(restore).not.toHaveBeenCalled()
  } finally {
    window.removeEventListener("tldw:set-composer-message", restore)
  }
})

it("keeps a send queued behind a held focus check bound to its original tab", async () => {
  vi.useFakeTimers()
  let finish!: (value: unknown) => void
  io.capture.mockImplementationOnce(() => new Promise(resolve => { finish = resolve })).mockResolvedValue(manifest)
  const hook = mount()
  await act(async () => {
    window.dispatchEvent(new Event("focus"))
    await vi.advanceTimersByTimeAsync(FOCUS_CHECK_DEBOUNCE_MS)
  })
  expect(io.capture).toHaveBeenCalledOnce()
  let pending!: ReturnType<typeof hook.result.current.beforeSend>
  act(() => { pending = hook.result.current.beforeSend({ message: "SOURCE SEND" }) })
  hook.depart()
  let outcome: unknown
  await act(async () => { finish(manifest); outcome = await pending })
  expect(outcome).toEqual({ proceed: false, refreshed: false })
  expect(io.capture).toHaveBeenCalledOnce()
  expect(hook.controller.choose).not.toHaveBeenCalled()
})


it.each(["owner", "draft"])("holds a same-tab send after its %s changes during the manifest read", async (change) => {
  let finish!: (value: unknown) => void
  io.capture.mockImplementationOnce(() => new Promise(resolve => { finish = resolve })).mockResolvedValue(manifest)
  const hook = mount()
  let pending!: ReturnType<typeof hook.result.current.beforeSend>
  await act(async () => { pending = hook.result.current.beforeSend({ message: "SOURCE" }); await Promise.resolve() })
  if (change === "owner") hook.replaceOwner()
  else useStoreMessageOption.getState().startNewChatDraft(false)
  let outcome: unknown
  await act(async () => { finish({ ...manifest, snapshot: { nodes: [{ id: "new-tip", parent_id: "tip" }, { id: "tip", parent_id: null }] } }); outcome = await pending })
  expect(outcome).toEqual({ proceed: false, refreshed: false })
  expect(io.capture).toHaveBeenCalledOnce()
  expect(hook.controller.choose).not.toHaveBeenCalled()
})

it("does not publish a departed focus notice to a replacement owner in the same tab", async () => {
  vi.useFakeTimers()
  let finish!: (value: unknown) => void
  io.capture.mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
  const hook = mount()
  await act(async () => {
    window.dispatchEvent(new Event("focus"))
    await vi.advanceTimersByTimeAsync(FOCUS_CHECK_DEBOUNCE_MS)
  })
  hook.replaceOwner()
  await act(async () => { finish({ ...manifest, snapshot: { nodes: [{ id: "new-tip", parent_id: "tip" }, { id: "tip", parent_id: null }] } }) })
  expect(hook.controller.choose).not.toHaveBeenCalled()
  expect(hook.result.current.notice).toBeNull()
})

it("focus and send checks preserve an armed explicit branch without fetching or reparenting", async () => {
  vi.useFakeTimers()
  const hook = mount()
  Object.assign(hook.controller, { branchPending: true })
  io.capture.mockResolvedValue(manifest)
  await act(async () => {
    window.dispatchEvent(new Event("focus"))
    await vi.advanceTimersByTimeAsync(FOCUS_CHECK_DEBOUNCE_MS)
    expect(
      await hook.result.current.beforeSend({ message: "Branch input" })
    ).toEqual({ proceed: true, refreshed: false })
  })
  expect(io.capture).not.toHaveBeenCalled()
  expect(hook.controller.choose).not.toHaveBeenCalled()
})


it("warns through the original H1 choice when a reopened native bookmark is behind its admitted manifest", async () => {
  const hook = mount([{ id: "tip", parent_id: null }, { id: "new-tip", parent_id: "tip" }])
  const hold = vi.fn((_view: unknown, _current: () => boolean, _reason: string) => true)
  Object.assign(hook.controller, { checkFreshness: vi.fn(), holdStaleSend: hold })
  const originalView = hook.controller.getCurrent().view
  await act(async () => {
    hook.result.current.showTab("tip")
    hook.result.current.checkShownTab()
  })
  expect(hold).toHaveBeenCalledWith(originalView, expect.any(Function), "freshness")
  expect(hook.controller.choose).not.toHaveBeenCalled()
  expect(io.capture).not.toHaveBeenCalled()
  const current = hold.mock.calls[0][1] as () => boolean
  expect(current()).toBe(true)
  hook.depart()
  expect(current()).toBe(false)
})

it.each(["known latest", "explicit branch"])("preserves a native restored view with %s without a new choice", async reason => {
  const hook = mount([{ id: "tip", parent_id: null }, { id: "new-tip", parent_id: "tip" }])
  const hold = vi.fn((_view: unknown, _current: () => boolean, _reason: string) => true)
  Object.assign(hook.controller, {
    checkFreshness: vi.fn(),
    holdStaleSend: hold,
    branchPending: reason === "explicit branch"
  })
  await act(async () => {
    hook.result.current.showTab(reason === "known latest" ? "new-tip" : "tip")
    hook.result.current.checkShownTab()
  })
  expect(hold).not.toHaveBeenCalled()
  expect(hook.controller.choose).not.toHaveBeenCalled()
  expect(io.capture).not.toHaveBeenCalled()
})
