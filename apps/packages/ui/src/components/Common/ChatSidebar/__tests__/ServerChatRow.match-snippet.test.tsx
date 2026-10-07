// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import type { TFunction } from "i18next"

import type { ServerChatHistoryItem } from "@/hooks/useServerChatHistory"

vi.mock("antd", () => ({
  Dropdown: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
  Tooltip: ({ children }: { children?: React.ReactNode }) => <>{children}</>
}))

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

const renderRow = (chat: ServerChatHistoryItem) =>
  render(
    <ServerChatRow
      chat={chat}
      selectionMode={false}
      isTrashView={false}
      isPinned={false}
      isActive={false}
      openMenuFor={null}
      setOpenMenuFor={vi.fn()}
      onSelectChat={vi.fn()}
      onTogglePinned={vi.fn()}
      onOpenSettings={vi.fn()}
      onRenameChat={vi.fn()}
      onCreateTable={vi.fn()}
      onEditTopic={vi.fn()}
      onDeleteChat={vi.fn()}
      onRestoreChat={vi.fn()}
      t={t}
    />
  )

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
