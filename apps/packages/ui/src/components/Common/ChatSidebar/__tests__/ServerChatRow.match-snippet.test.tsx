// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { TFunction } from "i18next"

import type { ServerChatHistoryItem } from "@/hooks/useServerChatHistory"

import { ServerChatRow } from "../ServerChatRow"

const t = ((_key: string, defaultValueOrOptions?: string | { defaultValue?: string }) =>
  typeof defaultValueOrOptions === "string"
    ? defaultValueOrOptions
    : defaultValueOrOptions?.defaultValue || _key) as unknown as TFunction

const createChat = (overrides: Partial<ServerChatHistoryItem> = {}): ServerChatHistoryItem => ({
  id: "chat-1",
  title: "Weekly sync",
  created_at: "2026-03-08T00:00:00.000Z",
  updated_at: "2026-03-08T00:01:00.000Z",
  createdAtMs: Date.parse("2026-03-08T00:00:00.000Z"),
  updatedAtMs: Date.parse("2026-03-08T00:01:00.000Z"),
  ...overrides
})

const onTogglePinned = vi.fn()
const onSelectChat = vi.fn()
beforeEach(() => vi.clearAllMocks())

const renderRow = (chat: ServerChatHistoryItem, overrides: { isPinned?: boolean; readOnly?: boolean; isTrashView?: boolean } = {}) => {
  const Row = () => {
    const [openMenuFor, setOpenMenuFor] = React.useState<string | null>(null)
    return <ServerChatRow
      chat={chat}
      selectionMode={false}
      isTrashView={false}
      isPinned={false}
      isActive={false}
      openMenuFor={openMenuFor}
      setOpenMenuFor={setOpenMenuFor}
      onSelectChat={onSelectChat}
      onTogglePinned={onTogglePinned}
      onOpenSettings={vi.fn()}
      onRenameChat={vi.fn()}
      onCreateTable={vi.fn()}
      onEditTopic={vi.fn()}
      onDeleteChat={vi.fn()}
      onRestoreChat={vi.fn()}
      t={t}
      {...overrides}
    />
  }
  return render(<Row />)
}

describe("ServerChatRow search match snippet (CS-02)", () => {
  it("shows what was said under the title when the chat matched by message content", () => {
    renderRow(
      createChat({
        matched_in: ["content"],
        match_snippet: "…the launch codeword is zebrafinch."
      })
    )

    const snippet = screen.getByTestId("server-chat-match-snippet")
    expect(snippet).toHaveTextContent("…the launch codeword is zebrafinch.")
    expect(snippet).toHaveAttribute("title", "…the launch codeword is zebrafinch.")
    // The row stays findable by its title: the snippet follows it inside the same button.
    const rowButton = screen.getByRole("button", { name: /^Weekly sync\b/ })
    expect(rowButton).toContainElement(snippet)
    expect(
      screen.getByText("Weekly sync").compareDocumentPosition(snippet) &
        Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy()
  })

  it("renders no snippet for chats without one", () => {
    renderRow(createChat({ matched_in: ["title"], match_snippet: null }))

    expect(screen.queryByTestId("server-chat-match-snippet")).toBeNull()
    expect(screen.getByRole("button", { name: /^Weekly sync\b/ })).toBeInTheDocument()
  })
})


describe("ServerChatRow conversation actions", () => {
  it.each([false, true])("pins or unpins through the conversation menu (pinned=%s)", async (isPinned) => {
    renderRow(createChat(), { isPinned })
    fireEvent.click(screen.getByRole("button", { name: "More actions: Weekly sync" }))
    fireEvent.click(await screen.findByRole("menuitem", { name: isPinned ? "Unpin" : "Pin", exact: true }))
    expect(onTogglePinned).toHaveBeenCalledExactlyOnceWith("chat-1")
    expect(onSelectChat).not.toHaveBeenCalled()
    expect(screen.queryByRole("button", { name: /^(Unpin|Pin)$/ })).toBeNull()
  })

  it("does not expose pin or management actions for read-only history", () => {
    renderRow(createChat(), { readOnly: true })
    expect(screen.queryByRole("button", { name: "More actions: Weekly sync" })).toBeNull()
    fireEvent.click(screen.getByRole("button", { name: /^Weekly sync/ }))
    expect(onSelectChat).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ id: "chat-1" }))
  })

  it("keeps trashed conversations free of pin actions", async () => {
    renderRow(createChat(), { isTrashView: true })
    fireEvent.click(screen.getByRole("button", { name: "More actions: Weekly sync" }))
    expect(await screen.findByRole("menuitem", { name: "Restore", exact: true })).toBeInTheDocument()
    expect(screen.queryByRole("menuitem", { name: /^(Unpin|Pin)$/ })).toBeNull()
  })
})
