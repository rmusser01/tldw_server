import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { useCollectionsStore } from "@/store/collections"
import { ReadingItemsList } from "../ReadingItemsList"

const apiMock = vi.hoisted(() => ({
  getReadingList: vi.fn(),
  listReadingSavedSearches: vi.fn(),
  updateReadingItem: vi.fn(),
  deleteReadingItem: vi.fn(),
  bulkUpdateReadingItems: vi.fn(),
  getOutputTemplates: vi.fn(),
  generateOutput: vi.fn(),
  createReadingSavedSearch: vi.fn()
}))

const undoNotificationMock = vi.hoisted(() => ({
  showUndoNotification: vi.fn()
}))

const interpolate = (template: string, values?: Record<string, unknown>) => {
  if (!values) return template
  return template.replace(/\{\{(\w+)\}\}/g, (_match, token) => {
    const value = values[token]
    return value == null ? "" : String(value)
  })
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string } | Record<string, unknown>,
      maybeOptions?: Record<string, unknown>
    ) => {
      if (typeof fallbackOrOptions === "string") {
        return interpolate(fallbackOrOptions, maybeOptions)
      }
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        const maybeDefault = (fallbackOrOptions as { defaultValue?: string }).defaultValue
        if (typeof maybeDefault === "string") {
          return interpolate(maybeDefault, maybeOptions)
        }
      }
      return key
    }
  })
}))

vi.mock("@/hooks/useTldwApiClient", () => ({
  useTldwApiClient: () => apiMock
}))

vi.mock("@/hooks/useUndoNotification", () => ({
  useUndoNotification: () => undoNotificationMock
}))

const isoForLocalDate = (value: string, boundary: "start" | "end") => {
  const [year, month, day] = value.split("-").map(Number)
  const date =
    boundary === "start"
      ? new Date(year, month - 1, day, 0, 0, 0, 0)
      : new Date(year, month - 1, day, 23, 59, 59, 999)
  return date.toISOString()
}

describe("ReadingItemsList date filters", () => {
  beforeEach(() => {
    vi.clearAllMocks()

    if (!window.matchMedia) {
      Object.defineProperty(window, "matchMedia", {
        writable: true,
        value: vi.fn().mockImplementation((query: string) => ({
          matches: false,
          media: query,
          onchange: null,
          addListener: vi.fn(),
          removeListener: vi.fn(),
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          dispatchEvent: vi.fn()
        }))
      })
    }

    useCollectionsStore.getState().resetStore()
    useCollectionsStore.getState().setReadingSavedSearchesEnabled(false)

    apiMock.getReadingList.mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      size: 20
    })
    apiMock.listReadingSavedSearches.mockResolvedValue({ items: [], total: 0 })
  })

  it("renders existing date filters as native date input values", async () => {
    useCollectionsStore.getState().setFilterDateRange(
      isoForLocalDate("2026-03-04", "start"),
      isoForLocalDate("2026-03-14", "end")
    )

    render(<ReadingItemsList />)

    await waitFor(() => {
      expect(apiMock.getReadingList).toHaveBeenCalled()
    })
    expect(screen.getByLabelText("Date from")).toHaveValue("2026-03-04")
    expect(screen.getByLabelText("Date to")).toHaveValue("2026-03-14")
  })

  it("applies native date filters with local start and end day boundaries", async () => {
    render(<ReadingItemsList />)

    await waitFor(() => {
      expect(apiMock.getReadingList).toHaveBeenCalled()
    })

    fireEvent.change(screen.getByLabelText("Date from"), {
      target: { value: "2026-04-01" }
    })

    await waitFor(() => {
      expect(useCollectionsStore.getState().filterDateFrom).toBe(
        isoForLocalDate("2026-04-01", "start")
      )
    })

    fireEvent.change(screen.getByLabelText("Date to"), {
      target: { value: "2026-04-30" }
    })

    await waitFor(() => {
      expect(useCollectionsStore.getState().filterDateTo).toBe(
        isoForLocalDate("2026-04-30", "end")
      )
    })

    fireEvent.change(screen.getByLabelText("Date from"), {
      target: { value: "" }
    })

    await waitFor(() => {
      expect(useCollectionsStore.getState().filterDateFrom).toBeNull()
    })
    expect(useCollectionsStore.getState().filterDateTo).toBe(
      isoForLocalDate("2026-04-30", "end")
    )
  })
})
