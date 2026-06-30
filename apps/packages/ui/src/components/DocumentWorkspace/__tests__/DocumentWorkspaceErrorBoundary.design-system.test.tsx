import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { DocumentWorkspaceErrorBoundary } from "../DocumentWorkspaceErrorBoundary"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, defaultValue?: string) => defaultValue || _key
  })
}))

const MaybeThrow: React.FC<{ shouldThrow: boolean }> = ({ shouldThrow }) => {
  if (shouldThrow) {
    throw new Error("Document pane crashed")
  }

  return <div>Workspace recovered</div>
}

const suppressExpectedWindowError = (expectedMessage: string): (() => void) => {
  const handler = (event: ErrorEvent) => {
    const message =
      event.error instanceof Error
        ? event.error.message
        : typeof event.message === "string"
          ? event.message
          : ""

    if (message.includes(expectedMessage)) {
      event.preventDefault()
    }
  }

  window.addEventListener("error", handler)
  return () => window.removeEventListener("error", handler)
}

describe("DocumentWorkspaceErrorBoundary design-system fallback", () => {
  let consoleErrorSpy: ReturnType<typeof vi.spyOn>
  let restoreWindowError: () => void

  beforeEach(() => {
    consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {})
    restoreWindowError = suppressExpectedWindowError("Document pane crashed")
  })

  afterEach(() => {
    cleanup()
    restoreWindowError()
    consoleErrorSpy.mockRestore()
  })

  it("renders the default recovery fallback through the design-system EmptyState", () => {
    let shouldThrow = true

    const { container, rerender } = render(
      <DocumentWorkspaceErrorBoundary>
        <MaybeThrow shouldThrow={shouldThrow} />
      </DocumentWorkspaceErrorBoundary>
    )

    const title = screen.getByText("Something went wrong")
    const emptyState = title.closest('[data-ds-component="EmptyState"]')
    expect(emptyState).not.toBeNull()
    expect(emptyState?.querySelector("svg")).toHaveClass("text-warning")
    expect(
      screen.getByText("An error occurred while loading the document workspace.")
    ).toBeInTheDocument()

    shouldThrow = false
    rerender(
      <DocumentWorkspaceErrorBoundary>
        <MaybeThrow shouldThrow={shouldThrow} />
      </DocumentWorkspaceErrorBoundary>
    )
    fireEvent.click(screen.getByRole("button", { name: "Try again" }))

    expect(screen.getByText("Workspace recovered")).toBeInTheDocument()
    expect(container.querySelector('[data-ds-component="EmptyState"]')).toBeNull()
  })
})
