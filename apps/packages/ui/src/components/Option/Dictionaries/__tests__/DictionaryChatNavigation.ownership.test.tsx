// @vitest-environment jsdom
import {
  HistorySelectionContext,
  useHistorySelection
} from "@/hooks/chat/useHistorySelection"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { act, renderHook } from "@testing-library/react"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useDictionaryChatContextNavigation } from "../components/useDictionaryChatContextNavigation"

const mocks = vi.hoisted(() => ({
  select: vi.fn(),
  getChat: vi.fn(),
  initialize: vi.fn(),
  snapshot: vi.fn(),
  sourceCurrent: true,
  release: vi.fn(),
  setter: vi.fn(),
  warning: vi.fn()
}))
vi.mock("@/hooks/chat/useSelectServerChat", () => ({
  useSelectServerChat: () => mocks.select
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: mocks }))
vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: mocks.snapshot
}))

vi.mock("@/services/chat-account-boundary", () => ({
  watchChatAccountChanges: () => () => undefined
}))
const owner = {
  isCurrent: () => true,
  requestScope: {
    config: { serverUrl: "https://one.test", authMode: "multi-user" as const },
    userId: "A"
  }
}
const chat = {
  id: "saved",
  title: "Actual saved title",
  state: "in-progress",
  version: 2,
  scope_type: "global"
}
beforeEach(() => {
  vi.resetAllMocks()
  mocks.sourceCurrent = true
  useStoreMessageOption.getState().startNewChatDraft(false)
  useStoreMessageOption.setState({ historyId: null, serverChatId: null })
  mocks.getChat.mockResolvedValue(chat)
  mocks.select.mockReturnValue(true)
  mocks.snapshot.mockResolvedValue({
    scopeSignal: new AbortController().signal,
    release: mocks.release
  })
})
describe("dictionary linked chat original source and shared admission", () => {
  it("loads actual scoped identity and delegates to Settings/Temporary admission", async () => {
    const hook = renderHook(() =>
      useDictionaryChatContextNavigation(owner, mocks)
    )
    let result
    await act(async () => {
      result = await hook.result.current.openChatContextFromDictionary({
        chat_id: "saved",
        title: "Old hint"
      })
    })
    expect(mocks.getChat).toHaveBeenCalledWith(
      "saved",
      expect.objectContaining({
        requestScope: owner.requestScope,
        scope: { type: "global" },
        requiredMetadata: ["title", "state"]
      })
    )
    expect(mocks.select).toHaveBeenCalledWith(chat, {
      temporaryChat: false,
      isCurrent: expect.any(Function)
    })
    expect(result).toBe(true)
    expect(mocks.setter).not.toHaveBeenCalled()
  })
  it("honors a declined shared admission without directly resetting the current chat", async () => {
    mocks.select.mockReturnValue(false)
    const hook = renderHook(() =>
      useDictionaryChatContextNavigation(owner, mocks)
    )
    await expect(
      hook.result.current.openChatContextFromDictionary({ chat_id: "saved" })
    ).resolves.toBe(false)
    expect(mocks.setter).not.toHaveBeenCalled()
  })
  it("cannot select after the original draft or modal retires during the scoped read", async () => {
    let release!: (value: typeof chat) => void
    mocks.getChat.mockReturnValue(
      new Promise((resolve) => {
        release = resolve
      })
    )
    let modalCurrent = true
    const hook = renderHook(() =>
      useDictionaryChatContextNavigation(owner, mocks)
    )
    const pending = hook.result.current.openChatContextFromDictionary(
      { chat_id: "saved" },
      () => modalCurrent
    )
    await vi.waitFor(() => expect(mocks.getChat).toHaveBeenCalled())
    useStoreMessageOption.getState().startNewChatDraft(false)
    modalCurrent = false
    release(chat)
    await expect(pending).resolves.toBe(false)
    expect(mocks.select).not.toHaveBeenCalled()
  })
  it.each(["../other", "saved?key=value", "saved#fragment", "saved%2Fother"])(
    "refuses an unsafe linked ID before transport: %s",
    async (id) => {
      const hook = renderHook(() =>
        useDictionaryChatContextNavigation(owner, mocks)
      )
      await expect(
        hook.result.current.openChatContextFromDictionary({ chat_id: id })
      ).resolves.toBe(false)
      expect(mocks.getChat).not.toHaveBeenCalled()
      expect(mocks.select).not.toHaveBeenCalled()
    }
  )

  it.each(["New", "restore", "H1 load", "H1 reset"])(
    "refuses an old rendered Open after actual %s before React commits",
    async (change) => {
      let selection!: ReturnType<typeof useHistorySelection>
      const Wrapper = ({ children }: React.PropsWithChildren) => {
        const current = useHistorySelection()
        React.useLayoutEffect(() => {
          selection = current
        }, [current])
        return (
          <HistorySelectionContext.Provider value={current}>
            {children}
          </HistorySelectionContext.Provider>
        )
      }
      const hook = renderHook(
        () => useDictionaryChatContextNavigation(owner, mocks),
        { wrapper: Wrapper }
      )
      const oldOpen = hook.result.current.openChatContextFromDictionary
      await act(async () => {
        if (change === "New")
          useStoreMessageOption.getState().startNewChatDraft(false)
        else if (change === "restore")
          usePlaygroundSessionStore.getState().cancelPendingRestore()
        else if (change === "H1 load") selection.beginLoad()
        else selection.reset()
        expect(await oldOpen({ chat_id: "saved" })).toBe(false)
      })
      expect(mocks.getChat).not.toHaveBeenCalled()
      expect(mocks.select).not.toHaveBeenCalled()
    }
  )

  it("refuses an expired account callback before any lookup", async () => {
    const hook = renderHook(() =>
      useDictionaryChatContextNavigation(
        { ...owner, isCurrent: () => false },
        mocks
      )
    )
    await expect(
      hook.result.current.openChatContextFromDictionary({ chat_id: "saved" })
    ).resolves.toBe(false)
    expect(mocks.getChat).not.toHaveBeenCalled()
    expect(mocks.select).not.toHaveBeenCalled()
  })
})
