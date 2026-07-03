// @vitest-environment jsdom

import React from "react"
import { MemoryRouter } from "react-router-dom"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import OptionSetup from "../option-setup"

const mocks = vi.hoisted(() => ({
  useSetupOnboarding: vi.fn(),
  useConnectionState: vi.fn(),
  navigate: vi.fn(),
  setConfigPartial: vi.fn(),
  testConnectionFromOnboarding: vi.fn(),
  refresh: vi.fn(),
  adoptState: vi.fn()
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
  useConnectionState: () => mocks.useConnectionState(),
  useConnectionActions: () => ({
    setConfigPartial: mocks.setConfigPartial,
    testConnectionFromOnboarding: mocks.testConnectionFromOnboarding
  })
}))

const firstRunState = (status: string) => ({ status })

const firstRunMetadata = () => ({
  setup_required: true,
  setup_completed: false,
  remote_setup_enabled: false,
  connection: {
    frontend_origin: null,
    api_origin: "http://127.0.0.1:8000",
    browser_access: "local"
  }
})

const setupReturn = ({
  state = firstRunState("completed"),
  metadata = null,
  loading = false
}: {
  state?: ReturnType<typeof firstRunState> | null
  metadata?: ReturnType<typeof firstRunMetadata> | null
  loading?: boolean
} = {}) => ({
  state,
  metadata,
  loading,
  refresh: mocks.refresh,
  adoptState: mocks.adoptState
})

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
    mocks.useConnectionState.mockReturnValue({
      serverUrl: "http://127.0.0.1:8000"
    })
    mocks.useSetupOnboarding.mockReturnValue(setupReturn())
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

  it("shows the setup entry choice before the wizard when backend setup is incomplete", () => {
    mocks.useSetupOnboarding.mockReturnValue(
      setupReturn({
        state: firstRunState("not_started"),
        metadata: firstRunMetadata()
      })
    )

    renderRoute()

    const headings = screen.getAllByRole("heading", { level: 1 })
    expect(headings).toHaveLength(1)
    expect(headings[0]).toHaveTextContent("Choose where to set up tldw")
    expect(
      screen.queryByRole("heading", { level: 2, name: "Connect your tldw server" })
    ).not.toBeInTheDocument()
    expect(screen.queryByTestId("setup-required-panel")).not.toBeInTheDocument()
    expect(screen.queryByTestId("unified-setup-shell")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Set up in WebUI" }))

    expect(screen.getByTestId("unified-setup-shell")).toBeInTheDocument()
  })

  it.each(["in_progress", "first_chat_complete"])(
    "shows the setup entry choice for %s backend setup",
    (status) => {
      mocks.useSetupOnboarding.mockReturnValue(
        setupReturn({
          state: firstRunState(status),
          metadata: firstRunMetadata()
        })
      )

      renderRoute()

      expect(
        screen.getByRole("heading", {
          level: 1,
          name: "Choose where to set up tldw"
        })
      ).toBeInTheDocument()
      expect(
        screen.queryByRole("heading", {
          level: 2,
          name: "Connect your tldw server"
        })
      ).not.toBeInTheDocument()
      expect(screen.queryByTestId("setup-required-panel")).not.toBeInTheDocument()
    }
  )

  it("uses the configured server URL for the API setup link when metadata has an internal origin", () => {
    mocks.useConnectionState.mockReturnValue({
      serverUrl: "http://api.example.test:9000"
    })
    const metadata = firstRunMetadata()
    mocks.useSetupOnboarding.mockReturnValue(
      setupReturn({
        state: firstRunState("not_started"),
        metadata: {
          ...metadata,
          connection: {
            ...metadata.connection,
            frontend_origin: "http://localhost:8080",
            api_origin: "http://app:8000"
          }
        }
      })
    )

    renderRoute()

    expect(
      screen.getByRole("link", {
        name: /open api server setup.*opens in a new tab/i
      })
    ).toHaveAttribute("href", "http://api.example.test:9000/setup")
  })

  it("enters the existing wizard after choosing WebUI setup and can go back to choices", () => {
    mocks.useSetupOnboarding.mockReturnValue(
      setupReturn({
        state: firstRunState("not_started"),
        metadata: firstRunMetadata()
      })
    )

    renderRoute()

    fireEvent.click(screen.getByRole("button", { name: "Set up in WebUI" }))

    expect(screen.getByTestId("unified-setup-shell")).toBeInTheDocument()
    let headings = screen.getAllByRole("heading", { level: 1 })
    expect(headings).toHaveLength(1)
    expect(headings[0]).toHaveTextContent("First-time setup")

    fireEvent.click(screen.getByRole("button", { name: "Back to setup choices" }))

    headings = screen.getAllByRole("heading", { level: 1 })
    expect(headings).toHaveLength(1)
    expect(headings[0]).toHaveTextContent("Choose where to set up tldw")
    expect(screen.queryByTestId("unified-setup-shell")).not.toBeInTheDocument()
  })

  it("keeps blocked setup on the recovery choice instead of routing into the wizard", () => {
    mocks.useSetupOnboarding.mockReturnValue(
      setupReturn({
        state: firstRunState("blocked"),
        metadata: firstRunMetadata()
      })
    )

    renderRoute()

    expect(screen.getByRole("button", { name: "Set up in WebUI" })).toBeDisabled()
    expect(screen.queryByTestId("unified-setup-shell")).not.toBeInTheDocument()
  })

  it("refreshes first-run state after API setup handoff", () => {
    mocks.useSetupOnboarding.mockReturnValue(
      setupReturn({
        state: firstRunState("not_started"),
        metadata: firstRunMetadata()
      })
    )

    renderRoute()

    fireEvent.click(
      screen.getByRole("link", {
        name: /open api server setup.*opens in a new tab/i
      })
    )
    fireEvent.click(screen.getByRole("button", { name: "I finished API server setup" }))

    expect(mocks.refresh).toHaveBeenCalledTimes(1)
  })

  it("returns to the recovery choice if refreshed state becomes blocked after WebUI mode was selected", () => {
    mocks.useSetupOnboarding.mockReturnValue(
      setupReturn({
        state: firstRunState("not_started"),
        metadata: firstRunMetadata()
      })
    )
    const view = renderRoute()

    fireEvent.click(screen.getByRole("button", { name: "Set up in WebUI" }))
    expect(screen.getByTestId("unified-setup-shell")).toBeInTheDocument()

    mocks.useSetupOnboarding.mockReturnValue(
      setupReturn({
        state: firstRunState("blocked"),
        metadata: firstRunMetadata()
      })
    )
    view.rerender(
      <MemoryRouter>
        <OptionSetup />
      </MemoryRouter>
    )

    expect(
      screen.getByRole("heading", { level: 1, name: "Choose where to set up tldw" })
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Set up in WebUI" })).toBeDisabled()
    expect(screen.queryByTestId("unified-setup-shell")).not.toBeInTheDocument()
  })

  it("keeps the manual connection and recovery UI when setup metadata is missing", () => {
    mocks.useSetupOnboarding.mockReturnValue(
      setupReturn({
        state: firstRunState("not_started"),
        metadata: null,
        loading: false
      })
    )

    renderRoute()

    expect(
      screen.getByRole("heading", { level: 2, name: "Connect your tldw server" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("heading", { level: 2, name: "Setup operator recovery" })
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("heading", { level: 1, name: "Choose where to set up tldw" })
    ).not.toBeInTheDocument()
  })

  it("keeps the manual connection and recovery UI when setup state is missing", () => {
    mocks.useSetupOnboarding.mockReturnValue(
      setupReturn({
        state: null,
        metadata: firstRunMetadata(),
        loading: false
      })
    )

    renderRoute()

    expect(
      screen.getByRole("heading", { level: 2, name: "Connect your tldw server" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("heading", { level: 2, name: "Setup operator recovery" })
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("heading", { level: 1, name: "Choose where to set up tldw" })
    ).not.toBeInTheDocument()
  })

  it("uses the route heading while initial setup state is loading", () => {
    mocks.useSetupOnboarding.mockReturnValue(
      setupReturn({
        state: null,
        metadata: null,
        loading: true
      })
    )

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

  it("associates API key help with the input and exposes toggle state", () => {
    renderRoute()

    const apiKeyInput = screen.getByLabelText("API Key")
    const helpToggle = screen.getByRole("button", {
      name: "Where do I find my key?"
    })

    expect(helpToggle).toHaveAttribute("aria-expanded", "false")
    expect(apiKeyInput).not.toHaveAttribute("aria-describedby")

    fireEvent.click(helpToggle)

    expect(helpToggle).toHaveAttribute("aria-expanded", "true")
    const help = screen.getByText(/SINGLE_USER_API_KEY/)
    expect(help).toHaveAttribute("id", "setup-api-key-help")
    expect(apiKeyInput).toHaveAttribute("aria-describedby", "setup-api-key-help")
  })

  it("saves self-host connection settings and opens health after a successful test", async () => {
    renderRoute()

    fireEvent.change(screen.getByLabelText("Server URL"), {
      target: { value: " http://localhost:9000 " }
    })
    fireEvent.change(screen.getByLabelText("API Key"), {
      target: { value: " secret-key " }
    })
    fireEvent.click(screen.getByRole("button", { name: "Test connection" }))

    await waitFor(() => {
      expect(mocks.setConfigPartial).toHaveBeenCalledWith({
        serverUrl: "http://localhost:9000",
        authMode: "single-user",
        apiKey: "secret-key"
      })
    })
    expect(mocks.testConnectionFromOnboarding).toHaveBeenCalledTimes(1)
    expect(mocks.navigate).toHaveBeenCalledWith("/settings/health")
  })

  it("sanitizes failed self-host connection test details before display", async () => {
    mocks.testConnectionFromOnboarding.mockRejectedValueOnce(
      new Error(
        "POST /api/v1/health?access_token=raw-token failed from /Users/alice/.tldw/config.json with Bearer bearer-token-value"
      )
    )

    renderRoute()

    fireEvent.click(screen.getByRole("button", { name: "Test connection" }))

    const alert = await screen.findByRole("alert")
    expect(alert).toHaveTextContent("POST [server-endpoint] failed")
    expect(alert).not.toHaveTextContent("raw-token")
    expect(alert).not.toHaveTextContent("/Users/alice")
    expect(alert).not.toHaveTextContent("bearer-token-value")
  })
})
