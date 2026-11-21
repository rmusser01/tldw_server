// @vitest-environment jsdom
/**
 * CS-09 review: one chat's "Add to folder…" shows the folders it is already
 * in, and unticking one removes it, with the captured account's operation.
 */
import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const folderApi = vi.hoisted(() => ({
  refresh: vi.fn(),
  add: vi.fn(async () => true),
  remove: vi.fn(async () => true),
  release: vi.fn()
}))

vi.mock("@/store/folder", async () => {
  const { create } = await import("zustand")
  const folders = [
    { id: 1, name: "Trips", parent_id: null, deleted: false },
    { id: 2, name: "Work", parent_id: null, deleted: false }
  ]
  const useFolderStore = create<Record<string, unknown>>((set) => ({
    folders,
    isLoading: false,
    error: null,
    folderApiAvailable: null,
    serverDataScope: null,
    membership: new Set([1]),
    refreshFromServer: async (operation?: unknown) => {
      folderApi.refresh(operation)
      set({ serverDataScope: operation ?? null, folderApiAvailable: true, isLoading: false })
    },
    getFolderTree: () => folders.map((folder) => ({ key: folder.id, title: folder.name, itemCount: 0, children: [] })),
    getFoldersForConversation: (id: string) => (id === "chat-1" ? [folders[0]] : []),
    addConversationToFolder: folderApi.add,
    removeConversationFromFolder: folderApi.remove,
    createFolder: vi.fn(),
    releaseServerData: folderApi.release
  }))
  return { useFolderStore }
})

import { FolderPickerModal } from "../FolderPickerModal"

const treeNode = (name: string) => screen.getByText(name).closest(".ant-tree-treenode") as HTMLElement
const isChecked = (name: string) =>
  Boolean(treeNode(name).querySelector(".ant-tree-checkbox-checked"))

describe("FolderPickerModal for one chat (CS-09 review)", () => {
  const requestScope = { config: { serverUrl: "https://chat.test", authMode: "multi-user" as const }, userId: 1 }
  let current = true
  const operation = { requestScope, isCurrent: () => current }

  beforeEach(() => {
    vi.clearAllMocks()
    current = true
  })

  it("shows the folders the chat is in, and unticking one removes it with the captured account", async () => {
    const onClose = vi.fn()
    render(<FolderPickerModal open conversationId="chat-1" operation={operation} onClose={onClose} />)
    await waitFor(() => expect(isChecked("Trips")).toBe(true))
    expect(isChecked("Work")).toBe(false)
    expect(folderApi.refresh).toHaveBeenCalledWith(expect.objectContaining({ requestScope }))

    fireEvent.click(treeNode("Trips").querySelector(".ant-tree-checkbox")!)
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() =>
      expect(folderApi.remove).toHaveBeenCalledWith("chat-1", 1, expect.objectContaining({ requestScope }))
    )
    expect(folderApi.add).not.toHaveBeenCalled()
    await waitFor(() => expect(onClose).toHaveBeenCalled())
  })

  it("changes nothing once the account it was opened for is no longer current", async () => {
    render(<FolderPickerModal open conversationId="chat-1" operation={operation} onClose={vi.fn()} />)
    await waitFor(() => expect(isChecked("Trips")).toBe(true))
    fireEvent.click(treeNode("Work").querySelector(".ant-tree-checkbox")!)
    current = false
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(folderApi.add).not.toHaveBeenCalled()
    expect(folderApi.remove).not.toHaveBeenCalled()
  })
})
