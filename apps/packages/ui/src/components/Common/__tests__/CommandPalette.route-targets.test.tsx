import React from "react"
import { describe, expect, it, vi } from "vitest"
import { render, screen, within } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"

import { CommandPalette } from "../CommandPalette"
import type { ShortcutConfig } from "@/hooks/keyboard/useShortcutConfig"
import {
  getRouteCommandPaletteLabel,
  getRouteMetadata,
  normalizeRoutePath
} from "@/routes/route-metadata"

const mockShortcutConfig: ShortcutConfig = {
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

const renderOpenGlobalPalette = async (): Promise<HTMLElement[]> => {
  render(
    <MemoryRouter>
      <CommandPalette />
    </MemoryRouter>
  )

  window.dispatchEvent(new CustomEvent("tldw:open-command-palette"))
  await screen.findByRole("dialog")

  return screen
    .getAllByRole("option")
    .filter((option) =>
      option.getAttribute("data-command-id")?.startsWith("nav-")
    )
}

describe("CommandPalette route target governance", () => {
  it("keeps default navigation commands backed by visible route metadata", async () => {
    const navigationOptions = await renderOpenGlobalPalette()
    const invalidTargets = navigationOptions
      .map((option) => ({
        commandId: option.getAttribute("data-command-id") ?? "",
        targetPath: option.getAttribute("data-target-path") ?? ""
      }))
      .filter(({ targetPath }) => !targetPath || !getRouteMetadata(targetPath))

    expect(invalidTargets).toEqual([])

    for (const option of navigationOptions) {
      const commandId = option.getAttribute("data-command-id")
      const targetPath = option.getAttribute("data-target-path")
      const metadata = getRouteMetadata(targetPath ?? "")

      expect(metadata, `${commandId} target ${targetPath} has metadata`).toBeDefined()
      expect(
        metadata?.commandPalette,
        `${commandId} target ${targetPath} must be command-palette visible`
      ).toBe("show")
    }
  })

  it("keeps default navigation command labels aligned with route metadata", async () => {
    const navigationOptions = await renderOpenGlobalPalette()

    for (const option of navigationOptions) {
      const commandId = option.getAttribute("data-command-id")
      const targetPath = option.getAttribute("data-target-path")
      const metadata = getRouteMetadata(targetPath ?? "")

      expect(metadata, `${commandId} target ${targetPath} has metadata`).toBeDefined()
      expect(
        within(option).getByText(getRouteCommandPaletteLabel(metadata!)),
        `${commandId} label should match the command label for ${targetPath}`
      ).toBeInTheDocument()
    }
  })

  it("does not expose duplicate navigation labels for different routes", async () => {
    const navigationOptions = await renderOpenGlobalPalette()
    const targetsByLabel = new Map<string, Set<string>>()

    for (const option of navigationOptions) {
      const label = option.querySelector(".font-medium")?.textContent?.trim() ?? ""
      const targetPath = normalizeRoutePath(
        option.getAttribute("data-target-path") ?? ""
      )

      if (!targetsByLabel.has(label)) {
        targetsByLabel.set(label, new Set())
      }
      targetsByLabel.get(label)?.add(targetPath)
    }

    const duplicateLabels = [...targetsByLabel.entries()]
      .filter(([, targets]) => targets.size > 1)
      .map(([label, targets]) => ({
        label,
        targets: [...targets].sort()
      }))

    expect(duplicateLabels).toEqual([])
  })
})
