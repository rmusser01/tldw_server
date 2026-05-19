// @vitest-environment jsdom

import React from "react"
import { MemoryRouter } from "react-router-dom"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import OptionSetup from "../option-setup"

const mocks = vi.hoisted(() => ({
  useConnectionState: vi.fn(),
  useConnectionUxState: vi.fn()
}))

vi.mock("~/components/Layouts/Layout", () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="option-layout">{children}</div>
  )
}))

vi.mock("@/components/Option/Onboarding/OnboardingWizard", () => ({
  OnboardingWizard: () => <div data-testid="onboarding-wizard">Onboarding</div>
}))

vi.mock("@/components/Option/Setup/ReadinessSetupScreen", () => ({
  ReadinessSetupScreen: ({ mode }: { mode?: string }) => (
    <div data-testid="readiness-screen" data-mode={mode}>
      Readiness
    </div>
  )
}))

vi.mock("@/components/ui/state", () => ({
  SetupRequiredPanel: ({ title }: { title: string }) => (
    <div data-testid="setup-required-panel">{title}</div>
  )
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => mocks.useConnectionState(),
  useConnectionUxState: () => mocks.useConnectionUxState()
}))

const renderRoute = () =>
  render(
    <MemoryRouter>
      <OptionSetup />
    </MemoryRouter>
  )

describe("OptionSetup readiness route", () => {
  beforeEach(() => {
    mocks.useConnectionState.mockReturnValue({
      serverUrl: "http://127.0.0.1:8000"
    })
    mocks.useConnectionUxState.mockReturnValue({
      hasCompletedFirstRun: false,
      isConfigOrError: false
    })
  })

  it("renders the native readiness screen when a backend is configured", () => {
    renderRoute()

    expect(screen.getByTestId("readiness-screen")).toHaveAttribute("data-mode", "first-run")
    expect(screen.queryByTestId("onboarding-wizard")).not.toBeInTheDocument()
  })

  it("uses admin readiness mode after first-run setup has completed", () => {
    mocks.useConnectionUxState.mockReturnValue({
      hasCompletedFirstRun: true,
      isConfigOrError: false
    })

    renderRoute()

    expect(screen.getByTestId("readiness-screen")).toHaveAttribute("data-mode", "admin")
  })

  it("keeps the connection onboarding wizard when the server URL is missing", () => {
    mocks.useConnectionState.mockReturnValue({ serverUrl: null })

    renderRoute()

    expect(screen.getByTestId("onboarding-wizard")).toBeInTheDocument()
    expect(screen.queryByTestId("readiness-screen")).not.toBeInTheDocument()
  })

  it("keeps the connection onboarding wizard while connection setup still needs attention", () => {
    mocks.useConnectionUxState.mockReturnValue({ isConfigOrError: true })

    renderRoute()

    expect(screen.getByTestId("onboarding-wizard")).toBeInTheDocument()
    expect(screen.queryByTestId("readiness-screen")).not.toBeInTheDocument()
  })
})
