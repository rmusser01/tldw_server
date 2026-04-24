import React from "react"
import { describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"

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

import { WorldBookEmptyState } from "../WorldBookEmptyState"

describe("WorldBookEmptyState", () => {
  const defaultProps = {
    onCreateNew: vi.fn(),
    onCreateFromTemplate: vi.fn(),
    onImport: vi.fn()
  }

  it("renders the 3-step visual flow", () => {
    render(<WorldBookEmptyState {...defaultProps} />)

    expect(screen.getByText(/create a world book/i)).toBeInTheDocument()
    expect(screen.getByText(/add entries/i)).toBeInTheDocument()
    expect(screen.getByText(/attach/i)).toBeInTheDocument()
  })

  it("renders the keyword matching example mentioning magic system", () => {
    render(<WorldBookEmptyState {...defaultProps} />)

    expect(screen.getByText(/magic system/i)).toBeInTheDocument()
  })

  it("renders template quick-start buttons for Fantasy, Sci-Fi, and Product", () => {
    render(<WorldBookEmptyState {...defaultProps} />)

    expect(screen.getByRole("button", { name: /fantasy/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /sci-fi/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /product/i })).toBeInTheDocument()
  })

  it("calls onCreateNew when the primary CTA is clicked", async () => {
    const user = userEvent.setup()
    const onCreateNew = vi.fn()

    render(
      <WorldBookEmptyState
        {...defaultProps}
        onCreateNew={onCreateNew}
      />
    )

    await user.click(screen.getByRole("button", { name: /create your first world book/i }))
    expect(onCreateNew).toHaveBeenCalledTimes(1)
  })

  it("calls onCreateFromTemplate with template key when a template button is clicked", async () => {
    const user = userEvent.setup()
    const onCreateFromTemplate = vi.fn()

    render(
      <WorldBookEmptyState
        {...defaultProps}
        onCreateFromTemplate={onCreateFromTemplate}
      />
    )

    await user.click(screen.getByRole("button", { name: /fantasy/i }))
    expect(onCreateFromTemplate).toHaveBeenCalledTimes(1)
    expect(onCreateFromTemplate).toHaveBeenCalledWith("fantasy")
  })

  it("calls onImport when the import button is clicked", async () => {
    const user = userEvent.setup()
    const onImport = vi.fn()

    render(
      <WorldBookEmptyState
        {...defaultProps}
        onImport={onImport}
      />
    )

    await user.click(screen.getByRole("button", { name: /import from json/i }))
    expect(onImport).toHaveBeenCalledTimes(1)
  })
})
