import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"

import { CommandPaletteHost } from "../CommandPaletteHost"
import { isMac } from "@/hooks/useKeyboardShortcuts"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string) => defaultValue ?? key
  })
}))

vi.mock("@/hooks/keyboard/useShortcutConfig", () => ({
  useShortcutConfig: () => ({
    shortcuts: {},
    updateShortcut: vi.fn(),
    resetShortcuts: vi.fn(),
    resetShortcut: vi.fn()
  })
}))

describe("CommandPaletteHost", () => {
  beforeEach(() => {
    HTMLElement.prototype.scrollIntoView = vi.fn()
  })

  it("lazy-mounts and opens the palette in response to the global open event", async () => {
    render(
      <MemoryRouter>
        <CommandPaletteHost />
      </MemoryRouter>
    )

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()

    window.dispatchEvent(new CustomEvent("tldw:open-command-palette"))

    expect(await screen.findByRole("dialog")).toBeInTheDocument()
  })

  it("blocks the Cmd/Ctrl+K opener on the workspace playground route while still honoring the event API", async () => {
    render(
      <MemoryRouter initialEntries={["/workspace-playground"]}>
        <CommandPaletteHost />
      </MemoryRouter>
    )

    fireEvent.keyDown(
      document,
      isMac ? { key: "k", metaKey: true } : { key: "k", ctrlKey: true }
    )

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()

    window.dispatchEvent(new CustomEvent("tldw:open-command-palette"))

    expect(await screen.findByRole("dialog")).toBeInTheDocument()
  })

  it("opens the palette with Control+K on standard routes", async () => {
    render(
      <MemoryRouter>
        <CommandPaletteHost />
      </MemoryRouter>
    )

    fireEvent.keyDown(document, { key: "k", ctrlKey: true })

    expect(await screen.findByRole("dialog")).toBeInTheDocument()
  })

  it("opens the palette even when an input is focused", async () => {
    render(
      <MemoryRouter>
        <div>
          <input aria-label="composer" />
          <CommandPaletteHost />
        </div>
      </MemoryRouter>
    )

    const input = screen.getByRole("textbox", { name: "composer" })
    input.focus()
    fireEvent.keyDown(input, { key: "k", ctrlKey: true })

    expect(await screen.findByRole("dialog")).toBeInTheDocument()
  })

  it("opens the palette even when a focused control stops keydown propagation", async () => {
    render(
      <MemoryRouter>
        <div>
          <input
            aria-label="propagation-blocker"
            onKeyDown={(event) => {
              event.stopPropagation()
            }}
          />
          <CommandPaletteHost />
        </div>
      </MemoryRouter>
    )

    const input = screen.getByRole("textbox", { name: "propagation-blocker" })
    input.focus()
    fireEvent.keyDown(input, { key: "k", ctrlKey: true })

    expect(await screen.findByRole("dialog")).toBeInTheDocument()
  })
})
