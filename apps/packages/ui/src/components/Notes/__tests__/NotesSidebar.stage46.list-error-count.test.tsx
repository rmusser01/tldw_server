import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import NotesSidebar from "../NotesSidebar"
import { buildNotesSidebarProps } from "./notes-sidebar-test-props"

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
      if (defaultValueOrOptions?.defaultValue) {
        return String(defaultValueOrOptions.defaultValue).replace(
          /\{\{(\w+)\}\}/g,
          (_match, token) => String(defaultValueOrOptions[token] ?? "")
        )
      }
      return key
    }
  })
}))

vi.mock("@/components/Notes/NotesListPanel", () => ({
  default: () => <div data-testid="notes-list-panel" />
}))

const baseProps = buildNotesSidebarProps({
  total: 1,
  queryInput: "server failure",
  hasActiveFilters: true,
  activeFilterSummary: {
    countText: "Showing 0 of 1 notes",
    detailsText: 'Query: "server failure"'
  },
  query: "server failure",
  hasListError: true,
  listErrorMessage: "Backend unavailable"
})

describe("NotesSidebar list error count state", () => {
  it("does not present filtered totals as fresh when the list refresh failed", () => {
    render(<NotesSidebar {...baseProps} />)

    expect(screen.getAllByText("Refresh failed").length).toBeGreaterThanOrEqual(1)
    expect(screen.queryByText("0 of 1")).not.toBeInTheDocument()
    expect(screen.queryByText("Showing 0 of 1 notes")).not.toBeInTheDocument()
  })
})
