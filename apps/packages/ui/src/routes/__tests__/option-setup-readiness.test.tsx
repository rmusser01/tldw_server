// @vitest-environment jsdom

import React from "react"
import { MemoryRouter } from "react-router-dom"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import OptionSetup from "../option-setup"

const mocks = vi.hoisted(() => ({
  useSetupOnboarding: vi.fn()
}))

vi.mock("~/components/Layouts/Layout", () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="option-layout">{children}</div>
  )
}))

vi.mock("@/components/Common/PageAssistLoader", () => ({
  PageAssistLoader: ({ label }: { label: string }) => (
    <div data-testid="page-assist-loader">{label}</div>
  )
}))

vi.mock("@/components/Option/Onboarding/UnifiedSetupWizard", () => ({
  UnifiedSetupWizard: () => (
    <section data-testid="unified-setup-shell">
      <h1>First-time setup</h1>
    </section>
  )
}))

vi.mock("@/components/ui/state", () => ({
  SetupRequiredPanel: ({
    title,
    message,
    primaryAction
  }: {
    title: string
    message: string
    primaryAction: { label: string; onClick: () => void }
  }) => (
    <section data-testid="setup-required-panel">
      <h2>{title}</h2>
      <p>{message}</p>
      <button type="button" onClick={primaryAction.onClick}>
        {primaryAction.label}
      </button>
    </section>
  )
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, defaultValue?: string) => defaultValue || _key
  })
}))

vi.mock("@/hooks/useSetupOnboarding", () => ({
  useSetupOnboarding: () => mocks.useSetupOnboarding()
}))

const renderRoute = () =>
  render(
    <MemoryRouter>
      <OptionSetup />
    </MemoryRouter>
  )

describe("OptionSetup readiness route", () => {
  beforeEach(() => {
    mocks.useSetupOnboarding.mockReturnValue({
      state: { status: "completed" },
      metadata: null,
      loading: false,
      adoptState: vi.fn()
    })
  })

  it("exposes a route heading when setup does not require the wizard", () => {
    renderRoute()

    const headings = screen.getAllByRole("heading", { level: 1 })
    expect(headings).toHaveLength(1)
    expect(headings[0]).toHaveTextContent("Setup")
    expect(screen.queryByTestId("unified-setup-shell")).not.toBeInTheDocument()
  })

  it("keeps the wizard as the only h1 when setup is required", () => {
    mocks.useSetupOnboarding.mockReturnValue({
      state: { status: "not_started" },
      metadata: null,
      loading: false,
      adoptState: vi.fn()
    })

    renderRoute()

    const headings = screen.getAllByRole("heading", { level: 1 })
    expect(headings).toHaveLength(1)
    expect(headings[0]).toHaveTextContent("First-time setup")
    expect(screen.getByTestId("unified-setup-shell")).toBeInTheDocument()
  })

  it("uses the route heading while initial setup state is loading", () => {
    mocks.useSetupOnboarding.mockReturnValue({
      state: null,
      metadata: null,
      loading: true,
      adoptState: vi.fn()
    })

    renderRoute()

    const headings = screen.getAllByRole("heading", { level: 1 })
    expect(headings).toHaveLength(1)
    expect(headings[0]).toHaveTextContent("Setup")
    expect(screen.getByTestId("page-assist-loader")).toBeInTheDocument()
    expect(screen.queryByTestId("unified-setup-shell")).not.toBeInTheDocument()
  })
})
