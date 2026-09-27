import React from "react"
import { act, cleanup, render, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import Head from "next/head"
import initHeadManager from "next/dist/client/head-manager"
import { HeadManagerContext } from "next/dist/shared/lib/head-manager-context.shared-runtime"

const mocks = vi.hoisted(() => ({
  generateTitle: vi.fn<() => Promise<string>>(),
  saveHistory: vi.fn(async () => ({ id: "saved-chat" })),
  saveMessage: vi.fn(async () => undefined)
}))

vi.mock("next/dynamic", () => ({ default: () => () => <main>Route content</main> }))
vi.mock("@/services/title", () => ({ generateTitle: mocks.generateTitle }))
vi.mock("@/db/dexie/helpers", () => ({
  saveHistory: mocks.saveHistory,
  saveMessage: mocks.saveMessage,
  updateLastUsedModel: vi.fn(),
  updateLastUsedPrompt: vi.fn(),
  updateChatHistoryCreatedAt: vi.fn(),
  addFileToSession: vi.fn(),
  getLastChatHistory: vi.fn(),
  updateMessage: vi.fn()
}))
vi.mock("@/db/dexie/chat-persistence-transaction", () => ({
  runChatPersistenceTransaction: async (_signal: unknown, write: () => Promise<unknown>) => write()
}))

import ServerSettings from "@web/pages/settings/tldw"
import { saveMessageOnSuccess } from "@/hooks/chat-helper"
import { updatePageTitle } from "@/utils/update-page-title"

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.clearAllMocks()
  document.head.innerHTML = ""
  window.history.replaceState({}, "", "/")
})

describe("route title ownership during late Chat persistence", () => {
  it("keeps the real Settings Head title when a pending Chat save finishes after navigation", async () => {
    vi.stubGlobal("__NEXT_DATA__", { page: "/chat" })
    window.history.replaceState({}, "", "/chat")
    const headManager = initHeadManager()
    const { rerender } = render(
      <HeadManagerContext.Provider value={headManager}>
        <Head><title>Chat | tldw</title></Head>
      </HeadManagerContext.Provider>
    )
    await waitFor(() => expect(document.title).toBe("Chat | tldw"))
    let resolveTitle!: (title: string) => void
    mocks.generateTitle.mockReturnValueOnce(new Promise((resolve) => { resolveTitle = resolve }))
    const setHistoryId = vi.fn()
    const pending = saveMessageOnSuccess({
      historyId: null,
      setHistoryId,
      isRegenerate: false,
      selectedModel: "model",
      message: "private question",
      image: "",
      fullText: "private answer",
      source: [],
      userMessageId: "user",
      assistantMessageId: "assistant"
    })
    await waitFor(() => expect(mocks.generateTitle).toHaveBeenCalledOnce())
    window.history.replaceState({}, "", "/settings/tldw")
    rerender(<HeadManagerContext.Provider value={headManager}><ServerSettings /></HeadManagerContext.Provider>)
    await waitFor(() => expect(document.title).toBe("Server Settings | tldw"))

    await act(async () => {
      resolveTitle("Private conversation title")
      await pending
    })

    expect(setHistoryId).toHaveBeenCalledWith("saved-chat")
    expect(mocks.saveMessage).toHaveBeenCalledTimes(2)
    expect(document.title).toBe("Server Settings | tldw")
    // A reconnect can also finish a loader after Settings already owns Head.
    updatePageTitle("Reconnected private conversation")
    expect(document.title).toBe("Server Settings | tldw")
  })

  it("leaves Chat's own Head authoritative when an older Chat callback completes", async () => {
    vi.stubGlobal("__NEXT_DATA__", { page: "/chat" })
    window.history.replaceState({}, "", "/chat?chat_id=current")
    render(
      <HeadManagerContext.Provider value={initHeadManager()}>
        <Head><title>Current conversation | tldw</title></Head>
      </HeadManagerContext.Provider>
    )
    await waitFor(() => expect(document.title).toBe("Current conversation | tldw"))
    updatePageTitle("Old conversation")
    expect(document.title).toBe("Current conversation | tldw")
  })
})
