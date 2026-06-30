import React from "react"
import { describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"

vi.mock("@web/components/layout/WebLayout", () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="extension-option-layout">{children}</div>
  )
}))

vi.mock("@/components/Common/PageShell", () => ({
  PageShell: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="page-shell">{children}</div>
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
    <div
      data-testid={`extension-route-boundary-${routeId}`}
      data-route-id={routeId}
      data-route-label={routeLabel}
    >
      {children}
    </div>
  )
}))

vi.mock("@/components/Option/Evaluations/EvaluationsPlaygroundPage", () => ({
  EvaluationsPlaygroundPage: () => (
    <div data-testid="extension-evaluations-page">Evaluations</div>
  )
}))

vi.mock("@/components/Option/KanbanPlayground", () => ({
  KanbanPlayground: () => (
    <div data-testid="extension-kanban-page">Kanban</div>
  )
}))

import OptionEvaluations from "@web/extension/routes/option-evaluations"
import OptionKanbanPlayground from "@web/extension/routes/option-kanban-playground"

describe("extension study, safety, and specialized route parity", () => {
  it("wraps extension /evaluations with the canonical route boundary", () => {
    render(<OptionEvaluations />)

    const boundary = screen.getByTestId("extension-route-boundary-evaluations")

    expect(boundary).toHaveAttribute("data-route-id", "evaluations")
    expect(boundary).toHaveAttribute("data-route-label", "Evaluations")
    expect(screen.getByTestId("extension-option-layout")).toBeVisible()
    expect(screen.getByTestId("extension-evaluations-page")).toBeVisible()
  })

  it("wraps extension /kanban with the canonical route boundary", () => {
    render(<OptionKanbanPlayground />)

    const boundary = screen.getByTestId("extension-route-boundary-kanban")

    expect(boundary).toHaveAttribute("data-route-id", "kanban")
    expect(boundary).toHaveAttribute("data-route-label", "Kanban")
    expect(screen.getByTestId("extension-option-layout")).toBeVisible()
    expect(screen.getByTestId("extension-kanban-page")).toBeVisible()
  })
})
