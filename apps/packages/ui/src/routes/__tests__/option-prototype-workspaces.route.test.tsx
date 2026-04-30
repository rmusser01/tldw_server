import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import OptionPrototypeWorkspaces from "../option-prototype-workspaces"

vi.mock("~/components/Layouts/Layout", () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="option-layout">{children}</div>
  ),
}))

vi.mock("@/components/Option/PrototypeWorkspace", () => ({
  PrototypeWorkspacePage: () => (
    <div data-testid="prototype-workspace-page">Prototype workspace</div>
  ),
}))

describe("option prototype workspaces route", () => {
  it("uses the standard option shell with flex/min-h-0 layout", () => {
    render(<OptionPrototypeWorkspaces />)

    const shell = screen.getByTestId("prototype-workspace-route-shell")
    expect(shell.className).toContain("flex")
    expect(shell.className).toContain("flex-1")
    expect(shell.className).toContain("min-h-0")
    expect(shell.className).toContain("overflow-hidden")
    expect(screen.getByTestId("option-layout")).toBeInTheDocument()
    expect(screen.getByTestId("prototype-workspace-page")).toBeVisible()
  })
})
