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

vi.mock("~/components/Option/Skills/SkillsWorkspace", () => ({
  SkillsWorkspace: () => <div data-testid="skills-workspace">Skills workspace</div>
}))

import OptionSkillsRoute from "../option-skills"

describe("option Skills route shell", () => {
  it("wraps the standalone Skills route in a route error boundary", () => {
    render(<OptionSkillsRoute />)

    const boundary = screen.getByTestId("route-error-boundary")
    expect(boundary).toHaveAttribute("data-route-id", "skills")
    expect(boundary).toHaveAttribute("data-route-label", "Skills")
    expect(screen.getByTestId("skills-workspace")).toBeVisible()
  })
})
