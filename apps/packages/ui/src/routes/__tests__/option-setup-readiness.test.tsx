// @vitest-environment jsdom

import React from "react"
import { MemoryRouter } from "react-router-dom"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import OptionSetup from "../option-setup"

const mocks = vi.hoisted(() => ({
  useSetupOnboarding: vi.fn(),
  navigate: vi.fn(),
  setConfigPartial: vi.fn(),
  testConnectionFromOnboarding: vi.fn()
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
    titleHeadingLevel = 2,
    primaryAction
  }: {
    title: string
    message: string
    titleHeadingLevel?: 1 | 2 | 3 | 4 | 5 | 6
    primaryAction: { label: string; onClick: () => void }
  }) => (
    <section data-testid="setup-required-panel">
      {titleHeadingLevel === 1 ? <h1>{title}</h1> : <h2>{title}</h2>}
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

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>(
    "react-router-dom"
  )
  return {
    ...actual,
    useNavigate: () => mocks.navigate
  }
})

vi.mock("@/hooks/useSetupOnboarding", () => ({
  useSetupOnboarding: () => mocks.useSetupOnboarding()
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionActions: () => ({
    setConfigPartial: mocks.setConfigPartial,
    testConnectionFromOnboarding: mocks.testConnectionFromOnboarding
  })
}))

const renderRoute = () =>
  render(
    <MemoryRouter>
      <OptionSetup />
    </MemoryRouter>
  )

describe("OptionSetup readiness route", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
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
    expect(
      screen.getByRole("heading", { level: 2, name: "Setup operator recovery" })
    ).toBeInTheDocument()
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
    expect(
      screen.getByRole("heading", { level: 2, name: "Setup operator recovery" })
    ).toBeInTheDocument()
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
    expect(
      screen.getByRole("heading", { level: 2, name: "Setup operator recovery" })
    ).toBeInTheDocument()
    expect(screen.getByTestId("page-assist-loader")).toBeInTheDocument()
    expect(screen.queryByTestId("unified-setup-shell")).not.toBeInTheDocument()
  })

  it("shows a self-host connection path before operator recovery", () => {
    renderRoute()

    expect(
      screen.getByRole("heading", { level: 2, name: "Connect your tldw server" })
    ).toBeInTheDocument()
    expect(screen.getByLabelText("Server URL")).toBeInTheDocument()
    expect(screen.getByLabelText("API Key")).toHaveAttribute("type", "password")
    expect(
      screen.getByRole("button", { name: "Test connection" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Where do I find my key?" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Skip and explore UI" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("heading", { level: 2, name: "Setup operator recovery" })
    ).toBeInTheDocument()
  })

  it("persists the first-run skip when users choose to explore the UI", () => {
    renderRoute()

    fireEvent.click(screen.getByRole("button", { name: "Skip and explore UI" }))

    expect(localStorage.getItem("assistant_setup_dismissed")).toBe("true")
    expect(mocks.navigate).toHaveBeenCalledWith("/chat")
  })
})
