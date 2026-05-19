// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

vi.mock("~/components/Layouts/Layout", () => ({
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
    <section
      data-route-id={routeId}
      data-route-label={routeLabel}
      data-testid="route-error-boundary"
    >
      {children}
    </section>
  )
}))

vi.mock("@/components/WorkflowEditor", () => ({
  WorkflowEditor: () => <div data-testid="workflow-editor">Workflow Editor</div>
}))

import OptionWorkflowEditorRoute from "../option-workflow-editor"

describe("option Workflow Editor route shell", () => {
  it("wraps the standalone Workflow Editor route in a route error boundary", () => {
    render(<OptionWorkflowEditorRoute />)

    const boundary = screen.getByTestId("route-error-boundary")
    expect(boundary).toHaveAttribute("data-route-id", "workflow-editor")
    expect(boundary).toHaveAttribute("data-route-label", "Workflow Editor")
    expect(screen.getByTestId("workflow-editor")).toBeVisible()
  })
})
