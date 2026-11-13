// @vitest-environment jsdom
/**
 * CS-09: a saved chat's row menu offers Export (Markdown or JSON) and Move to
 * folder, and calls its settings item "Chat settings".
 */
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import type { TFunction } from "i18next"

import type { ServerChatHistoryItem } from "@/hooks/useServerChatHistory"

type Item = { key: string; label?: React.ReactNode; onClick?: () => void; children?: Item[] }
// Render every menu item, nested ones included, as a plain button.
vi.mock("antd", () => ({
  Dropdown: ({ children, menu }: { children?: React.ReactNode; menu: { items: Item[] } }) => {
    const flatten = (items: Item[]): Item[] =>
      items.flatMap((item) => (item && "key" in item ? [item, ...flatten(item.children ?? [])] : []))
    return (
      <>
        {children}
        {flatten(menu.items).map((item) => (
          <button key={item.key} type="button" data-key={item.key} onClick={item.onClick}>
            {item.label}
          </button>
        ))}
      </>
    )
  },
  Tooltip: ({ children }: { children?: React.ReactNode }) => <>{children}</>
}))

import { ServerChatRow } from "../ServerChatRow"

const t = ((_key: string, defaultValueOrOptions?: string | { defaultValue?: string }) =>
  typeof defaultValueOrOptions === "string"
    ? defaultValueOrOptions
    : defaultValueOrOptions?.defaultValue || _key) as unknown as TFunction

const chat: ServerChatHistoryItem = {
  id: "chat-1",
  title: "Trip plan",
  created_at: "2026-03-08T00:00:00.000Z",
  updated_at: "2026-03-08T00:01:00.000Z",
  createdAtMs: Date.parse("2026-03-08T00:00:00.000Z"),
  updatedAtMs: Date.parse("2026-03-08T00:01:00.000Z"),
  state: "in-progress"
}

const renderRow = (overrides: Record<string, unknown> = {}) => {
  const handlers = {
    onExportChat: vi.fn(),
    onMoveToFolder: vi.fn(),
    onOpenSettings: vi.fn()
  }
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
      onRenameChat={vi.fn()}
      onCreateTable={vi.fn()}
      onEditTopic={vi.fn()}
      onDeleteChat={vi.fn()}
      onRestoreChat={vi.fn()}
      t={t}
      {...handlers}
      {...overrides}
    />
  )
  return handlers
}

describe("ServerChatRow per-chat actions (CS-09)", () => {
  it("exports the chat as Markdown or JSON", () => {
    const { onExportChat } = renderRow()
    expect(screen.getByRole("button", { name: "Export" })).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Markdown" }))
    fireEvent.click(screen.getByRole("button", { name: "JSON" }))
    expect(onExportChat.mock.calls).toEqual([[chat, "markdown"], [chat, "json"]])
  })

  it("moves the chat to a folder", () => {
    const { onMoveToFolder } = renderRow()
    fireEvent.click(screen.getByRole("button", { name: "Move to folder…" }))
    expect(onMoveToFolder).toHaveBeenCalledWith(chat)
  })

  it("calls its settings item Chat settings", () => {
    const { onOpenSettings } = renderRow()
    expect(screen.queryByRole("button", { name: "Open current chat settings" })).toBeNull()
    fireEvent.click(screen.getByRole("button", { name: "Chat settings" }))
    expect(onOpenSettings).toHaveBeenCalledWith(chat)
  })

  it("offers no export or folder move in Trash", () => {
    renderRow({ isTrashView: true })
    expect(screen.queryByRole("button", { name: "Export" })).toBeNull()
    expect(screen.queryByRole("button", { name: "Move to folder…" })).toBeNull()
  })
})
