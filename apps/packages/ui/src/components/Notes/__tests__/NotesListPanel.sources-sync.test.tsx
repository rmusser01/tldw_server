import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import NotesListPanel from "../NotesListPanel"
import type { ServerCapabilities } from "@/services/tldw/server-capabilities"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
            [key: string]: unknown
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

const baseCapabilities = {
  hasNotes: true,
  hasIngestionSources: true,
  canCreateLocalDirectoryIngestionSource: true
} as ServerCapabilities

const renderPanel = (
  overrides: Partial<React.ComponentProps<typeof NotesListPanel>> = {}
) => {
  const props: React.ComponentProps<typeof NotesListPanel> = {
    listMode: "active",
    isOnline: true,
    isFetching: false,
    demoEnabled: false,
    capsLoading: false,
    capabilities: baseCapabilities,
    notes: [],
    total: 0,
    page: 1,
    pageSize: 20,
    selectedId: null,
    onSelectNote: vi.fn(),
    onChangePage: vi.fn(),
    onResetEditor: vi.fn(),
    onOpenSettings: vi.fn(),
    onOpenHealth: vi.fn(),
    onRestoreNote: vi.fn(),
    onExportAllMd: vi.fn(),
    onExportAllCsv: vi.fn(),
    onExportAllJson: vi.fn(),
    onImportNotes: vi.fn(),
    onSyncFolder: vi.fn(),
    ...overrides
  }

  return {
    ...render(<NotesListPanel {...props} />),
    props
  }
}

const expectTooltipCopy = (button: HTMLElement, expectedText: string) => {
  expect(button).toHaveAttribute("title", expectedText)
}

describe("NotesListPanel Sources folder sync entry", () => {
  it("enables Sync folder in the active online notes view when server capabilities allow it", () => {
    const { props } = renderPanel()

    const button = screen.getByRole("button", { name: "Sync folder" })
    expect(button).toBeEnabled()

    fireEvent.click(button)
    expect(props.onSyncFolder).toHaveBeenCalledTimes(1)
  })

  it("disables Sync folder in trash view with a Notes-view tooltip", async () => {
    renderPanel({ listMode: "trash" })

    const button = screen.getByRole("button", { name: "Sync folder" })
    expect(button).toBeDisabled()
    expectTooltipCopy(button, "Switch to Notes view to sync folders")
  })

  it("disables Sync folder while offline with a connection tooltip", async () => {
    renderPanel({ isOnline: false })

    const button = screen.getByRole("button", { name: "Sync folder" })
    expect(button).toBeDisabled()
    expectTooltipCopy(button, "Connect to sync folders")
  })

  it("disables Sync folder when Sources are unsupported", async () => {
    renderPanel({
      capabilities: {
        ...baseCapabilities,
        hasIngestionSources: false
      }
    })

    const button = screen.getByRole("button", { name: "Sync folder" })
    expect(button).toBeDisabled()
    expectTooltipCopy(button, "Sources are not available on this server")
  })

  it("disables Sync folder when local-directory entitlement is false", async () => {
    renderPanel({
      capabilities: {
        ...baseCapabilities,
        canCreateLocalDirectoryIngestionSource: false
      }
    })

    const button = screen.getByRole("button", { name: "Sync folder" })
    expect(button).toBeDisabled()
    expectTooltipCopy(
      button,
      "Ask an administrator to enable server folder sync for this account"
    )
  })

  it("keeps Sync folder enabled while local-directory entitlement is unknown", async () => {
    const { props } = renderPanel({
      capabilities: {
        ...baseCapabilities,
        canCreateLocalDirectoryIngestionSource: null
      }
    })

    const button = screen.getByRole("button", { name: "Sync folder" })
    expect(button).toBeEnabled()

    fireEvent.click(button)
    expect(props.onSyncFolder).toHaveBeenCalledTimes(1)
  })
})
