import React from "react"
import { describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"

import { PageHelpModal } from "../PageHelpModal"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback ?? key
  })
}))

vi.mock("@/hooks/keyboard/useShortcutConfig", () => ({
  defaultShortcuts: {
    focusTextarea: { key: "Escape", shiftKey: true },
    newChat: { key: "u", ctrlKey: true, shiftKey: true },
    toggleSidebar: { key: "b", ctrlKey: true },
    toggleChatMode: { key: "e", ctrlKey: true },
    toggleWebSearch: { key: "w", altKey: true },
    toggleQuickChatHelper: { key: "h", ctrlKey: true, shiftKey: true },
    modePlayground: { key: "1", altKey: true },
    modeSources: { key: "2", altKey: true },
    modeMedia: { key: "3", altKey: true },
    modeKnowledge: { key: "4", altKey: true },
    modeNotes: { key: "5", altKey: true },
    modePrompts: { key: "6", altKey: true },
    modeFlashcards: { key: "7", altKey: true }
  },
  formatShortcut: ({
    key,
    altKey,
    ctrlKey,
    shiftKey
  }: {
    key: string
    altKey?: boolean
    ctrlKey?: boolean
    shiftKey?: boolean
  }) => [
    ...(ctrlKey ? ["Ctrl"] : []),
    ...(altKey ? ["Alt"] : []),
    ...(shiftKey ? ["Shift"] : []),
    key
  ].join(" + ")
}))

vi.mock("@/hooks/keyboard/useKeyboardShortcuts", () => ({
  isMac: false
}))

vi.mock("@/tutorials", () => ({
  getTutorialsForRoute: () => []
}))

describe("PageHelpModal shortcuts", () => {
  it("lists the Sources navigation shortcut when opened", async () => {
    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <PageHelpModal />
      </MemoryRouter>
    )

    window.dispatchEvent(new CustomEvent("tldw:open-help-modal"))

    expect(await screen.findByText("Go to Sources")).toBeVisible()
    expect(screen.getByText("Alt + 2")).toBeVisible()
  })
})
