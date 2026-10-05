import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, expect, it, vi } from "vitest"

const io = vi.hoisted(() => ({ select: vi.fn(), getChat: vi.fn(), error: vi.fn() }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (key: string, options?: { defaultValue?: string }) => options?.defaultValue || key }) }))
vi.mock("antd", () => ({ Empty: () => null, Skeleton: () => null, Modal: () => null, Input: () => null, Button: () => null, message: { error: io.error } }))
vi.mock("@/hooks/useConnectionState", () => ({ useConnectionState: () => ({ isConnected: true }) }))
vi.mock("@/store/connection", () => ({ useConnectionStore: Object.assign(
  (select: (value: unknown) => unknown) => select({ checkOnce: vi.fn() }),
  { getState: () => ({ state: { isConnected: true } }) }) }))
vi.mock("@/services/chat-account-boundary", () => ({ watchChatAccountChanges: () => () => undefined }))
vi.mock("@tanstack/react-query", () => ({ useQuery: () => ({ data: undefined, isFetching: false, isError: false }) }))
vi.mock("@/hooks/chat/useSelectServerChat", () => ({ useSelectServerChat: () => io.select }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: { getChat: io.getChat } }))
vi.mock("@/services/folder-api", () => ({
  fetchFolders: vi.fn(async () => ({ ok: true, data: [{ id: 1, name: "Folder", deleted: false }] })),
  fetchKeywords: vi.fn(async () => ({ ok: true, data: [{ id: 1, keyword: "tag", deleted: false }] })),
  fetchFolderKeywordLinks: vi.fn(async () => ({ ok: true, data: [{ folder_id: 1, keyword_id: 1 }] })),
  fetchConversationKeywordLinks: vi.fn(async () => ({ ok: true, data: [{ conversation_id: "folder-chat", keyword_id: 1 }] }))
}))
vi.mock("@/db/dexie/schema", () => ({ db: {} }))
vi.mock("@/components/Folders", () => ({ FolderTree: ({ onConversationSelect }: { onConversationSelect: (id: string) => void }) =>
  <button onClick={() => onConversationSelect("folder-chat")}>Select folder conversation</button> }))

import { useFolderStore } from "@/store/folder"
import { useStoreMessageOption } from "@/store/option"
import { FolderChatList } from "../FolderChatList"

const chat = { id: "folder-chat", title: "Folder chat" }
const owner = { key: "verified-folder-owner", requestScope: { config: { serverUrl: "https://server.example", apiKey: "synthetic-key" }, expectedUserId: "7" }, isCurrent: () => true }

beforeEach(() => {
  vi.clearAllMocks()
  const operation = useFolderStore.getState().serverDataScope
  if (operation) useFolderStore.getState().releaseServerData(operation)
  useStoreMessageOption.setState({ temporaryChat: false })
  io.getChat.mockResolvedValue(chat)
  io.select.mockReset()
})
afterEach(() => { cleanup(); vi.restoreAllMocks() })

it.each(["immediate", "deferred"])("reports accepted %s fresh folder selection only after target handoff", async (mode) => {
  let resolve!: (value: typeof chat) => void
  if (mode === "deferred") io.getChat.mockReturnValueOnce(new Promise<typeof chat>(done => { resolve = done }))
  const onConversationSelected = vi.fn()
  render(<FolderChatList owner={owner} verificationState="ready" onConversationSelected={onConversationSelected} />)
  fireEvent.click(await screen.findByRole("button", { name: "Select folder conversation" }))
  await waitFor(() => expect(io.getChat).toHaveBeenCalledWith("folder-chat", expect.objectContaining({ requestScope: owner.requestScope, scope: { type: "global" }, signal: expect.any(AbortSignal) })))
  if (mode === "deferred") {
    expect(onConversationSelected).not.toHaveBeenCalled()
    await act(async () => resolve(chat))
  }
  await waitFor(() => expect(onConversationSelected).toHaveBeenCalledTimes(1))
  expect(io.select).toHaveBeenCalledWith(chat)
  expect(io.select.mock.invocationCallOrder[0]).toBeLessThan(onConversationSelected.mock.invocationCallOrder[0])
})

it.each(["fetch", "handoff"])("does not report a failed folder %s", async (failure) => {
  if (failure === "fetch") io.getChat.mockRejectedValueOnce(new Error("Synthetic lookup failure"))
  else io.select.mockImplementationOnce(() => { throw new Error("Synthetic handoff failure") })
  const onConversationSelected = vi.fn()
  render(<FolderChatList owner={owner} verificationState="ready" onConversationSelected={onConversationSelected} />)
  fireEvent.click(await screen.findByRole("button", { name: "Select folder conversation" }))
  await waitFor(() => expect(io.error).toHaveBeenCalledTimes(1))
  expect(onConversationSelected).not.toHaveBeenCalled()
})
