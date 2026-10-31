import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, expect, it, vi } from "vitest"
import { FolderChatList } from "../FolderChatList"

const state = vi.hoisted(() => ({
  chats: [] as Array<{ id: string; title: string }>,
  select: vi.fn(), initialize: vi.fn(), getChat: vi.fn(), error: vi.fn(),
  refresh: vi.fn(async () => undefined)
}))
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (key: string, options?: { defaultValue?: string }) => options?.defaultValue || key }) }))
vi.mock("antd", () => ({ Empty: () => null, Skeleton: () => null, Modal: () => null, Input: () => null, message: { error: state.error } }))
vi.mock("@/hooks/useConnectionState", () => ({ useConnectionState: () => ({ isConnected: true }) }))
vi.mock("@/store/connection", () => ({ useConnectionStore: (select: (value: unknown) => unknown) => select({ checkOnce: vi.fn() }) }))
vi.mock("@/store/folder", () => ({
  useFolderActions: () => ({ refreshFromServer: state.refresh, createFolder: vi.fn() }),
  useFolderStore: (select: (value: unknown) => unknown) => select({ conversationKeywordLinks: [], folders: [{ id: "folder-1", deleted: false }], isLoading: false })
}))
vi.mock("@/hooks/useServerChatHistory", () => ({ useServerChatHistory: () => ({ data: state.chats, isLoading: false }) }))
vi.mock("@tanstack/react-query", () => ({ useQuery: () => ({ data: [], isLoading: false }) }))
vi.mock("@/hooks/chat/useSelectServerChat", () => ({ useSelectServerChat: () => state.select }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: { initialize: state.initialize, getChat: state.getChat } }))
vi.mock("@/components/Folders", () => ({ FolderTree: ({ onConversationSelect }: { onConversationSelect: (id: string) => void }) => <button onClick={() => onConversationSelect("folder-chat")}>Select folder conversation</button> }))

const chat = { id: "folder-chat", title: "Folder chat" }

beforeEach(() => {
  vi.clearAllMocks()
  state.chats = []
  state.initialize.mockResolvedValue(undefined)
  state.getChat.mockResolvedValue(chat)
  state.select.mockReset()
})
afterEach(() => vi.restoreAllMocks())

it.each(["cached", "fetched"])("reports accepted %s folder selection only after target handoff", async (mode) => {
  if (mode === "cached") state.chats = [chat]
  let resolve!: (value: typeof chat) => void
  if (mode === "fetched") state.getChat.mockReturnValueOnce(new Promise<typeof chat>(done => { resolve = done }))
  const onConversationSelected = vi.fn()
  render(<FolderChatList onConversationSelected={onConversationSelected} />)
  fireEvent.click(screen.getByRole("button", { name: "Select folder conversation" }))
  if (mode === "fetched") {
    await waitFor(() => expect(state.getChat).toHaveBeenCalledWith("folder-chat"))
    expect(onConversationSelected).not.toHaveBeenCalled()
    await act(async () => resolve(chat))
  }
  await waitFor(() => expect(onConversationSelected).toHaveBeenCalledTimes(1))
  expect(state.select).toHaveBeenCalledWith(chat)
  expect(state.select.mock.invocationCallOrder[0]).toBeLessThan(onConversationSelected.mock.invocationCallOrder[0])
})

it.each(["fetch", "handoff"])("does not report a failed folder %s", async (failure) => {
  vi.spyOn(console, "error").mockImplementation(() => undefined)
  if (failure === "fetch") state.getChat.mockRejectedValueOnce(new Error("Synthetic lookup failure"))
  else state.select.mockImplementationOnce(() => { throw new Error("Synthetic handoff failure") })
  const onConversationSelected = vi.fn()
  render(<FolderChatList onConversationSelected={onConversationSelected} />)
  fireEvent.click(screen.getByRole("button", { name: "Select folder conversation" }))
  await waitFor(() => expect(state.error).toHaveBeenCalledTimes(1))
  expect(onConversationSelected).not.toHaveBeenCalled()
})
