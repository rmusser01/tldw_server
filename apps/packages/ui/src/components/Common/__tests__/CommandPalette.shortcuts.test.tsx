import React from "react"
import { describe, it, expect, vi } from "vitest"
import { fireEvent, render, screen, within } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { CommandPalette } from "../CommandPalette"
import {
  formatShortcut,
  isMac,
  type ShortcutModifier
} from "@/hooks/useKeyboardShortcuts"
import type { ShortcutConfig } from "@/hooks/keyboard/useShortcutConfig"

const mockShortcutConfig: ShortcutConfig = {
  focusTextarea: { key: "Escape", shiftKey: true },
  newChat: { key: "u", ctrlKey: true, shiftKey: true },
  toggleSidebar: { key: "b", ctrlKey: true },
  toggleChatMode: { key: "e", ctrlKey: true },
  toggleWebSearch: { key: "w", altKey: true },
  toggleQuickChatHelper: { key: "h", ctrlKey: true, shiftKey: true },
  modePlayground: { key: "1", altKey: true },
  modeMedia: { key: "3", altKey: true },
  modeKnowledge: { key: "4", altKey: true },
  modeNotes: { key: "5", altKey: true },
  modePrompts: { key: "6", altKey: true },
  modeFlashcards: { key: "7", altKey: true },
  modeWorldBooks: { key: "8", altKey: true },
  modeDictionaries: { key: "9", altKey: true },
  modeCharacters: { key: "0", altKey: true }
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string) => defaultValue ?? key
  })
}))

vi.mock("@/hooks/keyboard/useShortcutConfig", () => ({
  useShortcutConfig: () => ({
    shortcuts: mockShortcutConfig,
    updateShortcut: vi.fn(),
    resetShortcuts: vi.fn(),
    resetShortcut: vi.fn()
  })
}))

const toCommandModifiers = (shortcut: {
  ctrlKey?: boolean
  altKey?: boolean
  shiftKey?: boolean
  metaKey?: boolean
}): ShortcutModifier[] => {
  const modifiers: ShortcutModifier[] = []
  if (shortcut.metaKey) modifiers.push("meta")
  if (shortcut.ctrlKey) modifiers.push("ctrl")
  if (shortcut.altKey) modifiers.push("alt")
  if (shortcut.shiftKey) modifiers.push("shift")
  return modifiers
}

const expectedShortcutLabel = (shortcut: {
  key: string
  ctrlKey?: boolean
  altKey?: boolean
  shiftKey?: boolean
  metaKey?: boolean
}): string =>
  formatShortcut({
    key: shortcut.key,
    modifiers: toCommandModifiers(shortcut)
  })

describe("CommandPalette shortcut hints", () => {
  it("shows configured shortcut hints only for actions with real keyboard bindings", async () => {
    render(
      <MemoryRouter>
        <CommandPalette
          onNewChat={vi.fn()}
          onToggleRag={vi.fn()}
          onToggleWebSearch={vi.fn()}
          onIngestPage={vi.fn()}
          onSwitchModel={vi.fn()}
          onToggleSidebar={vi.fn()}
        />
      </MemoryRouter>
    )

    window.dispatchEvent(new CustomEvent("tldw:open-command-palette"))

    expect(await screen.findByRole("dialog")).toBeInTheDocument()

    const newChat = screen.getByRole("option", { name: /New Chat/i })
    expect(
      within(newChat).getByText(expectedShortcutLabel(mockShortcutConfig.newChat))
    ).toBeInTheDocument()

    const toggleContext = screen.getByRole("option", {
      name: /Toggle Search & Context/i
    })
    expect(
      within(toggleContext).getByText(
        expectedShortcutLabel(mockShortcutConfig.toggleChatMode)
      )
    ).toBeInTheDocument()

    const toggleWeb = screen.getByRole("option", { name: /Toggle Web Search/i })
    expect(
      within(toggleWeb).getByText(
        expectedShortcutLabel(mockShortcutConfig.toggleWebSearch)
      )
    ).toBeInTheDocument()

    const toggleSidebar = screen.getByRole("option", {
      name: /Toggle Sidebar/i
    })
    expect(
      within(toggleSidebar).getByText(
        expectedShortcutLabel(mockShortcutConfig.toggleSidebar)
      )
    ).toBeInTheDocument()

    const goToMedia = screen.getByRole("option", { name: /Go to Media/i })
    expect(goToMedia.querySelector("kbd")).toBeNull()

    const goToSettings = screen.getByRole("option", { name: /Go to Settings/i })
    expect(goToSettings.querySelector("kbd")).toBeNull()

    const ingestCurrentPage = screen.getByRole("option", {
      name: /Ingest Current Page/i
    })
    expect(ingestCurrentPage.querySelector("kbd")).toBeNull()

    const switchModel = screen.getByRole("option", { name: /Switch Model/i })
    expect(switchModel.querySelector("kbd")).toBeNull()
  })

  it("applies focus-visible ring classes to modal input and command options", async () => {
    render(
      <MemoryRouter>
        <CommandPalette
          onNewChat={vi.fn()}
          onToggleRag={vi.fn()}
          onToggleWebSearch={vi.fn()}
          onIngestPage={vi.fn()}
          onSwitchModel={vi.fn()}
          onToggleSidebar={vi.fn()}
        />
      </MemoryRouter>
    )

    window.dispatchEvent(new CustomEvent("tldw:open-command-palette"))

    expect(await screen.findByRole("dialog")).toBeInTheDocument()

    const searchInput = screen.getByPlaceholderText(/Type a command or search/i)
    expect(searchInput.className).toContain("focus-visible:ring-2")
    expect(searchInput.className).toContain("focus-visible:ring-focus")
    expect(searchInput.className).toContain("focus-visible:ring-offset-2")
    expect(searchInput.className).toContain("focus-visible:ring-offset-bg")

    const newChat = screen.getByRole("option", { name: /New Chat/i })
    expect(newChat.className).toContain("focus-visible:ring-2")
    expect(newChat.className).toContain("focus-visible:ring-focus")
    expect(newChat.className).toContain("focus-visible:ring-offset-2")
    expect(newChat.className).toContain("focus-visible:ring-offset-bg")
  })

  it("disables the global Cmd/Ctrl+K shortcut on the workspace playground route", async () => {
    render(
      <MemoryRouter initialEntries={["/workspace-playground"]}>
        <CommandPalette
          onNewChat={vi.fn()}
          onToggleRag={vi.fn()}
          onToggleWebSearch={vi.fn()}
          onIngestPage={vi.fn()}
          onSwitchModel={vi.fn()}
          onToggleSidebar={vi.fn()}
        />
      </MemoryRouter>
    )

    fireEvent.keyDown(document, isMac
      ? { key: "k", metaKey: true }
      : { key: "k", ctrlKey: true })

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()

    window.dispatchEvent(new CustomEvent("tldw:open-command-palette"))

    expect(await screen.findByRole("dialog")).toBeInTheDocument()
  })

  it("opens from a window-level Ctrl+K event on standard routes", async () => {
    render(
      <MemoryRouter>
        <CommandPalette
          onNewChat={vi.fn()}
          onToggleRag={vi.fn()}
          onToggleWebSearch={vi.fn()}
          onIngestPage={vi.fn()}
          onSwitchModel={vi.fn()}
          onToggleSidebar={vi.fn()}
        />
      </MemoryRouter>
    )

    fireEvent.keyDown(window, { key: "k", ctrlKey: true })

    expect(await screen.findByRole("dialog")).toBeInTheDocument()
  })
})
