import React from "react"
import { beforeAll, describe, expect, it, vi } from "vitest"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { PageHelpModalHost } from "../PageHelpModalHost"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback ?? key
  })
}))

vi.mock("@/hooks/keyboard/useShortcutConfig", () => ({
  useShortcutConfig: () => ({
    shortcuts: {
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
      modeFlashcards: { key: "7", altKey: true },
      modeWorldBooks: { key: "8", altKey: true },
      modeDictionaries: { key: "9", altKey: true },
      modeCharacters: { key: "0", altKey: true }
    }
  }),
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
  }) =>
    [
      ...(ctrlKey ? ["Ctrl"] : []),
      ...(altKey ? ["Alt"] : []),
      ...(shiftKey ? ["Shift"] : []),
      key
    ].join(" + ")
}))

vi.mock("@/hooks/keyboard/useKeyboardShortcuts", () => ({
  isMac: false
}))

describe("PageHelpModal focus styling", () => {
  // The host lazy-loads the modal; warm the chunk so the first test does not
  // race the default findBy timeout on a cold transform.
  beforeAll(async () => {
    await import("../PageHelpModal")
  }, 30_000)

  it("applies focus-visible classes to the close button in modal context", async () => {
    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <PageHelpModalHost />
      </MemoryRouter>
    )
    act(() =>
      window.dispatchEvent(new CustomEvent("tldw:open-shortcuts-modal"))
    )

    const closeButton = await screen.findByRole("button", { name: "Close" })
    expect(closeButton.className).toContain("focus-visible:outline")
    expect(closeButton.className).toContain("focus-visible:outline-2")
    expect(closeButton.className).toContain("focus-visible:outline-focus")
  })

  it("lists the Sources navigation shortcut", async () => {
    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <PageHelpModalHost />
      </MemoryRouter>
    )
    act(() =>
      window.dispatchEvent(new CustomEvent("tldw:open-shortcuts-modal"))
    )

    expect(await screen.findByText("Go to Sources")).toBeVisible()
    expect(screen.getByText("Alt + 2")).toBeVisible()
  })
  // AX-18 (#3122): the shortcut list scrolls inside the dialog, so keyboard
  // users need to be able to focus it to scroll it.
  it("lets keyboard users reach and scroll the shortcut list", async () => {
    render(
      <MemoryRouter initialEntries={["/chat"]}>
        <PageHelpModalHost />
      </MemoryRouter>
    )
    act(() =>
      window.dispatchEvent(new CustomEvent("tldw:open-shortcuts-modal"))
    )

    const list = await screen.findByRole("region", { name: "Shortcut list" })
    expect(list).toHaveAttribute("tabindex", "0")
    expect(list.className).toContain("focus-visible:ring-2")

    const dialog = screen.getByRole("dialog")
    const closeButton = screen.getByRole("button", { name: "Close" })
    closeButton.focus()
    const reached: Element[] = []
    for (let step = 0; step < 6; step += 1) {
      fireEvent.keyDown(dialog, { key: "Tab" })
      reached.push(document.activeElement as Element)
    }
    expect(reached).toContain(list)
  })
})
