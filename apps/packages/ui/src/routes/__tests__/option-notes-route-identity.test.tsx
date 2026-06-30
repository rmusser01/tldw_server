import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import OptionNotes from "../option-notes"

vi.mock("~/components/Layouts/Layout", () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="option-layout">{children}</div>
  )
}))

vi.mock("@/components/Common/RouteErrorBoundary", () => ({
  RouteErrorBoundary: ({
    children,
    routeId,
    routeLabel
  }: {
    children: React.ReactNode
    routeId: string
    routeLabel: string
  }) => (
    <div data-testid="route-boundary" data-route-id={routeId} data-route-label={routeLabel}>
      {children}
    </div>
  )
}))

vi.mock("@/components/Notes/NotesManagerPage", () => ({
  __esModule: true,
  default: () => <div data-testid="notes-manager-page">Notes manager</div>
}))

describe("notes option route identity", () => {
  it("wraps /notes in the Notes route boundary and page", () => {
    render(<OptionNotes />)

    const boundary = screen.getByTestId("route-boundary")

    expect(screen.getByTestId("option-layout")).toBeVisible()
    expect(boundary).toHaveAttribute("data-route-id", "notes")
    expect(boundary).toHaveAttribute("data-route-label", "Notes")
    expect(screen.getByTestId("notes-manager-page")).toBeVisible()
  })
})
