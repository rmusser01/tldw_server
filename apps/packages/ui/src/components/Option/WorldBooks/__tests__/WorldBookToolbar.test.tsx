import React from "react"
import { describe, expect, it, vi } from "vitest"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter, useLocation } from "react-router-dom"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        return fallbackOrOptions.defaultValue || key
      }
      return key
    }
  })
}))

import { WorldBookToolbar } from "../WorldBookToolbar"

const defaultProps = {
  listSearch: "",
  onSearchChange: vi.fn(),
  enabledFilter: "all" as const,
  onEnabledFilterChange: vi.fn(),
  attachmentFilter: "all" as const,
  onAttachmentFilterChange: vi.fn(),
  onNewWorldBook: vi.fn(),
  onOpenTestMatching: vi.fn(),
  onOpenMatrix: vi.fn(),
  onOpenGlobalStats: vi.fn(),
  onImport: vi.fn(),
  onExportAll: vi.fn(),
  hasWorldBooks: true,
  hasSelection: false,
  globalStatsFetching: false,
  bulkExportAllLoading: false
}

const LocationProbe = () => {
  const location = useLocation()
  return <div data-testid="location-probe">{location.pathname + location.search}</div>
}

describe("WorldBookToolbar", () => {
  it("renders search input and filter dropdowns", () => {
    render(<WorldBookToolbar {...defaultProps} />)

    expect(
      screen.getByRole("textbox", { name: /search world books/i })
    ).toBeInTheDocument()

    expect(
      screen.getByLabelText(/filter by enabled status/i)
    ).toBeInTheDocument()

    expect(
      screen.getByLabelText(/filter by attachment state/i)
    ).toBeInTheDocument()
  })

  it("renders 'New World Book' as a primary button", () => {
    render(<WorldBookToolbar {...defaultProps} />)

    const button = screen.getByTestId("world-books-new-button")
    expect(button).toBeInTheDocument()
    expect(button).toHaveTextContent(/new world book/i)
    expect(button).toHaveClass("ant-btn-primary")
  })

  it("renders a Tools dropdown containing analysis and I/O actions", async () => {
    const user = userEvent.setup()
    render(<WorldBookToolbar {...defaultProps} />)

    const toolsButton = screen.getByRole("button", { name: /tools/i })
    expect(toolsButton).toBeInTheDocument()

    await user.click(toolsButton)

    await waitFor(() => {
      expect(screen.getByText("Test Matching")).toBeInTheDocument()
      expect(screen.getByText("Relationship Matrix")).toBeInTheDocument()
      expect(screen.getByText("Global Statistics")).toBeInTheDocument()
      expect(screen.getByText("Import JSON")).toBeInTheDocument()
      expect(screen.getByText("Export All")).toBeInTheDocument()
    })
  })

  it("disables analysis tools when no world books exist", async () => {
    const user = userEvent.setup()
    render(<WorldBookToolbar {...defaultProps} hasWorldBooks={false} />)

    const toolsButton = screen.getByRole("button", { name: /tools/i })
    await user.click(toolsButton)

    await waitFor(() => {
      const testMatching = screen.getByText("Test Matching").closest("[role='menuitem']")
      const matrix = screen.getByText("Relationship Matrix").closest("[role='menuitem']")
      const stats = screen.getByText("Global Statistics").closest("[role='menuitem']")

      expect(testMatching).toHaveAttribute("aria-disabled", "true")
      expect(matrix).toHaveAttribute("aria-disabled", "true")
      expect(stats).toHaveAttribute("aria-disabled", "true")
    })
  })

  it("calls onNewWorldBook when primary button is clicked", async () => {
    const user = userEvent.setup()
    const onNewWorldBook = vi.fn()

    render(<WorldBookToolbar {...defaultProps} onNewWorldBook={onNewWorldBook} />)

    await user.click(screen.getByTestId("world-books-new-button"))
    expect(onNewWorldBook).toHaveBeenCalledTimes(1)
  })

  it("shows 'Export Selected' in Tools menu when hasSelection is true", async () => {
    const user = userEvent.setup()
    render(<WorldBookToolbar {...defaultProps} hasSelection={true} />)

    const toolsButton = screen.getByRole("button", { name: /tools/i })
    await user.click(toolsButton)

    await waitFor(() => {
      expect(screen.getByText("Export Selected")).toBeInTheDocument()
    })
  })

  it("navigates internally when the chat injection panel tool is selected", async () => {
    const user = userEvent.setup()
    render(
      <MemoryRouter initialEntries={["/world-books"]}>
        <WorldBookToolbar {...defaultProps} />
        <LocationProbe />
      </MemoryRouter>
    )

    await user.click(screen.getByRole("button", { name: /tools/i }))
    await user.click(await screen.findByText("Chat Injection Panel"))

    expect(screen.getByTestId("location-probe")).toHaveTextContent(
      "/chat?from=world-books&focus=lorebook-debug"
    )
  })
})
