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

// AX-15 (#3124): the selected list mode and view must be exposed to assistive
// technology, not only shown by button fill colour.
// AX-16 (#3124): the section help icons need at least a 24x24 px target.
// AX-08 (#3124): and a focus ring in the focus colour.

describe("NotesSidebar toggle and help-button accessibility", () => {
  it("exposes the selected list mode with aria-pressed", () => {
    const { rerender } = render(<NotesSidebar {...baseProps} />)

    expect(screen.getByTestId("notes-mode-active")).toHaveAttribute("aria-pressed", "true")
    expect(screen.getByTestId("notes-mode-trash")).toHaveAttribute("aria-pressed", "false")

    rerender(<NotesSidebar {...baseProps} listMode="trash" />)

    expect(screen.getByTestId("notes-mode-active")).toHaveAttribute("aria-pressed", "false")
    expect(screen.getByTestId("notes-mode-trash")).toHaveAttribute("aria-pressed", "true")
  })

  it("exposes the selected view with aria-pressed", () => {
    render(<NotesSidebar {...baseProps} listViewMode="timeline" />)

    const pressed = ["list", "timeline", "inbox", "moodboard", "graph"].map((mode) => [
      mode,
      screen.getByTestId(`notes-view-mode-${mode}`).getAttribute("aria-pressed")
    ])
    expect(pressed).toEqual([
      ["list", "false"],
      ["timeline", "true"],
      ["inbox", "false"],
      ["moodboard", "false"],
      ["graph", "false"]
    ])
  })

  // XP-15 (#3124): the hint rendered at 14px in the body colour.
  it("renders the paging hint as small muted text, not antd Typography", () => {
    render(<NotesSidebar {...baseProps} showLargeListPaginationHint />)

    const hint = screen.getByTestId("notes-large-list-pagination-hint")
    expect(hint).toHaveClass("text-[11px]", "text-text-muted")
    expect(hint.className).not.toMatch(/ant-typography/)
  })

  it("gives section help icons a 24px target and a focus ring", () => {
    render(<NotesSidebar {...baseProps} />)

    const help = screen.getByRole("button", { name: "Organize help" })
    expect(help).toHaveClass("min-h-6", "min-w-6")
    expect(help).toHaveClass("focus-visible:ring-focus")
  })

  // AX-14 (#3123): screen-reader users move through the sidebar by heading.
  it("names its collapsible sections with headings that hold their toggles", () => {
    render(<NotesSidebar {...baseProps} />)

    const views = screen.getByRole("heading", { level: 2, name: "Views" })
    expect(within(views).getByRole("button", { name: "Views" })).toHaveAttribute(
      "aria-expanded",
      "true"
    )
    // Organize sits inside Views.
    expect(screen.getByRole("heading", { level: 3, name: "Organize" })).toBeInTheDocument()
    expect(screen.getByRole("heading", { level: 2, name: "Filters" })).toBeInTheDocument()
  })

  it("puts the timeline's date groups under a Results heading", () => {
    render(
      <NotesSidebar
        {...baseProps}
        hasListError={false}
        listViewMode="timeline"
        timelineSections={[{ key: "today", label: "Today", notes: [] }]}
      />
    )

    const timeline = screen.getByTestId("notes-timeline-view")
    expect(within(timeline).getByRole("heading", { level: 2, name: "Results" })).toBeInTheDocument()
    expect(within(timeline).getByRole("heading", { level: 3, name: "Today" })).toBeInTheDocument()
  })
})
