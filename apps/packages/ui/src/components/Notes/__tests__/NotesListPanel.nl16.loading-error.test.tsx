import React from "react"
import { render, screen, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import NotesListPanel from "../NotesListPanel"

// NL-16: the first load shows placeholder rows, and a load error keeps the
// raw request path behind a Details disclosure.

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?: string | { defaultValue?: string; [key: string]: unknown }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionActions: () => ({
    checkOnce: vi.fn()
  })
}))

const renderPanel = (overrides: Partial<React.ComponentProps<typeof NotesListPanel>> = {}) => {
  const props: React.ComponentProps<typeof NotesListPanel> = {
    listMode: "active",
    searchQuery: "",
    isOnline: true,
    isFetching: false,
    demoEnabled: false,
    capsLoading: false,
    capabilities: { hasNotes: true } as any,
    notes: [],
    total: 0,
    page: 1,
    pageSize: 20,
    selectedId: null,
    onCreateNote: vi.fn(),
    onSelectNote: vi.fn(),
    onChangePage: vi.fn(),
    onResetEditor: vi.fn(),
    onOpenSettings: vi.fn(),
    onOpenHealth: vi.fn(),
    onRestoreNote: vi.fn(),
    onExportAllMd: vi.fn(),
    onExportAllCsv: vi.fn(),
    onExportAllJson: vi.fn(),
    ...overrides
  }
  render(<NotesListPanel {...props} />)
}

describe("NotesListPanel NL-16 loading and error states", () => {
  it("shows placeholder rows while the first page loads", () => {
    renderPanel({ isFetching: true, notes: undefined })

    const loading = screen.getByTestId("notes-list-loading")
    expect(loading).toHaveAttribute("role", "status")
    expect(within(loading).getAllByTestId("notes-list-skeleton-row").length).toBeGreaterThanOrEqual(3)
  })

  it("explains an unreachable server in plain words and keeps the request path in Details", async () => {
    renderPanel({
      hasError: true,
      errorMessage: "Failed to fetch (GET /api/v1/notes/?limit=20&offset=0&sort_by=last_modified)",
      onRetry: vi.fn()
    })

    const errorState = await screen.findByTestId("notes-list-error-state")
    expect(within(errorState).getByText("Could not reach the notes server.")).toBeInTheDocument()
    const path = within(errorState).getByText(/GET \/api\/v1\/notes\//)
    const details = path.closest("details")
    expect(details).not.toBeNull()
    expect(details).not.toHaveAttribute("open")
  })

  it("keeps a server message that has no request path as it is", async () => {
    renderPanel({ hasError: true, errorMessage: "Server unavailable", onRetry: vi.fn() })

    const errorState = await screen.findByTestId("notes-list-error-state")
    expect(within(errorState).getByText("Server unavailable")).toBeInTheDocument()
    expect(errorState.querySelector("details")).toBeNull()
  })
})
