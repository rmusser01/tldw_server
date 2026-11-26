import React from "react"
import { render, screen, within } from "@testing-library/react"
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

// NL-16: loading shows "Loading…" instead of "0 total"; load errors keep the
// request path behind a Details disclosure.

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

const header = () => screen.getByTestId("notes-header-count")

describe("NotesSidebar NL-16 loading and error states", () => {
  it("says the list is loading instead of showing 0 total before the first rows arrive", () => {
    render(
      <NotesSidebar
        {...baseProps}
        total={0}
        isFetching
        hasListError={false}
        listErrorMessage={null}
        hasActiveFilters={false}
        activeFilterSummary={null}
        queryInput=""
        query=""
      />
    )

    expect(header()).toHaveTextContent("Loading…")
    expect(header()).not.toHaveTextContent("0 total")
  })

  it("shows the count once rows are in", () => {
    render(
      <NotesSidebar
        {...baseProps}
        total={3}
        filteredCount={3}
        isFetching={false}
        hasListError={false}
        listErrorMessage={null}
        hasActiveFilters={false}
        activeFilterSummary={null}
        queryInput=""
        query=""
      />
    )

    expect(header()).toHaveTextContent("3 total")
  })

  it("keeps the request path of a timeline load error behind Details", () => {
    render(
      <NotesSidebar
        {...baseProps}
        listViewMode="timeline"
        listErrorMessage="Failed to fetch (GET /api/v1/notes/?limit=20&offset=0)"
      />
    )

    const errorState = screen.getByTestId("notes-list-error-state")
    expect(within(errorState).getByText("Could not reach the notes server.")).toBeInTheDocument()
    const path = within(errorState).getByText(/GET \/api\/v1\/notes\//)
    const details = path.closest("details")
    expect(details).not.toBeNull()
    expect(details).not.toHaveAttribute("open")
    expect(within(details as HTMLElement).getByText("Details")).toBeInTheDocument()
  })
})
