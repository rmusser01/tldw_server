// @vitest-environment jsdom
/**
 * CS-08 (#3101, G07): history rows are easy to scan. The state, last update
 * and topic share one line under the title (the row layout itself is the
 * calm sidebar design, #3316). States read in words with a coloured dot, and
 * a topic that repeats the title is not shown twice.
 */
import React from "react"
import { render, screen, within } from "@testing-library/react"
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
  title: "Meridian: the sensor anomaly",
  created_at: "2026-03-08T00:00:00.000Z",
  updated_at: "2026-03-08T00:01:00.000Z",
  createdAtMs: Date.parse("2026-03-08T00:00:00.000Z"),
  updatedAtMs: Date.parse("2026-03-08T00:01:00.000Z"),
  state: "in-progress",
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

describe("ServerChatRow compact layout (CS-08)", () => {
  it.each([
    ["in-progress", "In progress"],
    ["resolved", "Resolved"],
    ["backlog", "Backlog"],
    ["non-viable", "Not viable"]
  ])("reads the %s state in words", (state, label) => {
    renderRow(createChat({ state }))

    const meta = screen.getByTestId("server-chat-row-meta")
    expect(within(meta).getByText(label)).toBeInTheDocument()
    expect(within(meta).queryByText(state)).toBeNull()
  })

  it("puts the state, last update and topic on one line under the title", () => {
    renderRow(createChat({ topic_label: "Field trials" }))

    const meta = screen.getByTestId("server-chat-row-meta")
    expect(meta).toHaveTextContent("In progress")
    expect(meta).toHaveTextContent("Field trials")
    expect(meta.textContent).toMatch(/ago|just now|\d/)
  })

  it("does not repeat a topic that matches the title", () => {
    renderRow(createChat({ topic_label: " meridian: the sensor anomaly " }))

    expect(screen.getAllByText(/meridian: the sensor anomaly/i)).toHaveLength(1)
  })

  it("still marks a forked chat", () => {
    renderRow(createChat({ parent_conversation_id: "parent-1234567890" }))

    expect(screen.getByText("Forked chat")).toBeInTheDocument()
  })
})
