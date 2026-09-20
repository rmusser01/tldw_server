import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"

import { ConnectionPhase } from "@/types/connection"
import OptionIndex from "../option-index"

let hostedMode = false
let hasCompletedFirstRun = false
let phase: ConnectionPhase | null = null
let firstRunState: { status: string } | null = null
let currentLocation = {
  pathname: "/",
  search: "",
  hash: "",
  state: null as unknown
}

const checkOnceMock = vi.fn().mockResolvedValue(undefined)
const beginOnboardingMock = vi.fn().mockResolvedValue(undefined)
const markFirstRunCompleteMock = vi.fn().mockResolvedValue(undefined)
const navigateMock = vi.fn()
const toggleDarkModeMock = vi.fn()

const optionLayoutMock = vi.fn(
  ({
    children
  }: {
    children: React.ReactNode
    hideHeader?: boolean
    hideSidebar?: boolean
  }) => <div data-testid="option-layout">{children}</div>
)

vi.mock("@/services/tldw/deployment-mode", () => ({
  isHostedTldwDeployment: () => hostedMode
}))

vi.mock("~/components/Layouts/Layout", () => ({
  __esModule: true,
  default: (props: {
    children: React.ReactNode
    hideHeader?: boolean
    hideSidebar?: boolean
  }) => optionLayoutMock(props)
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => ({
    phase
  }),
  useConnectionUxState: () => ({
    uxState: null,
    hasCompletedFirstRun
  }),
  useConnectionActions: () => ({
    checkOnce: checkOnceMock,
    beginOnboarding: beginOnboardingMock,
    markFirstRunComplete: markFirstRunCompleteMock
  })
}))

vi.mock("@/hooks/useComposerFocus", () => ({
  useFocusComposerOnConnect: () => undefined
}))

vi.mock("@/hooks/usePostOnboardingMediaReadiness", () => ({
  usePostOnboardingMediaReadiness: () => ({
    status: "idle",
    config: null,
    errorMessage: null,
    recoverWithApiKey: vi.fn(),
    retry: vi.fn()
  })
}))

vi.mock("@/hooks/useDarkmode", () => ({
  useDarkMode: () => ({
    mode: "dark",
    toggleDarkMode: toggleDarkModeMock
  })
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string | { defaultValue?: string }) => {
      if (typeof fallback === "string") {
        return fallback
      }
      return fallback?.defaultValue ?? _key
    }
  })
}))

vi.mock("react-router-dom", () => ({
  useNavigate: () => navigateMock,
  useLocation: () => currentLocation
}))

vi.mock("@/hooks/useSetupOnboarding", () => ({
  useSetupOnboarding: () => ({
    state: firstRunState,
    metadata: null,
    loading: false,
    adoptState: vi.fn()
  })
}))

vi.mock("@/components/Option/Onboarding/UnifiedSetupWizard", () => ({
  UnifiedSetupWizard: () => (
    <div data-testid="unified-setup-wizard">Unified setup wizard</div>
  )
}))

vi.mock("@/components/Option/CompanionHome", () => ({
  CompanionHomeShell: ({ surface }: { surface: "options" | "sidepanel" }) => (
    <div data-testid="companion-home-shell">{surface}</div>
  )
}))

vi.mock("../option-hosted-home", () => ({
  default: () => <div data-testid="hosted-home">Hosted home</div>
}))

describe("OptionIndex setup-flow routing", () => {
  beforeEach(() => {
    hostedMode = false
    hasCompletedFirstRun = false
    phase = null
    firstRunState = null
    window.localStorage.clear()
    currentLocation = {
      pathname: "/",
      search: "",
      hash: "",
      state: null
    }
    optionLayoutMock.mockClear()
    checkOnceMock.mockReset().mockResolvedValue(undefined)
    beginOnboardingMock.mockReset().mockResolvedValue(undefined)
    markFirstRunCompleteMock.mockReset().mockResolvedValue(undefined)
    navigateMock.mockReset()
    toggleDarkModeMock.mockReset()
  })

  it("renders hosted home in the headerless setup shell without self-host checks", async () => {
    hostedMode = true

    render(<OptionIndex />)

    expect(await screen.findByTestId("hosted-home")).toBeInTheDocument()
    expect(optionLayoutMock).toHaveBeenLastCalledWith(
      expect.objectContaining({
        hideHeader: true,
        hideSidebar: true
      })
    )
    expect(checkOnceMock).not.toHaveBeenCalled()
    expect(beginOnboardingMock).not.toHaveBeenCalled()
  })

  it("renders first-run onboarding in a headerless setup shell", async () => {
    phase = ConnectionPhase.UNCONFIGURED
    firstRunState = { status: "not_started" }

    render(<OptionIndex />)

    expect(
      await screen.findByTestId("unified-setup-wizard")
    ).toBeInTheDocument()
    expect(optionLayoutMock).toHaveBeenLastCalledWith(
      expect.objectContaining({
        hideHeader: true,
        hideSidebar: true
      })
    )
  })

  it("shows a resume-setup banner instead of the wizard for connected users", async () => {
    hasCompletedFirstRun = true
    phase = ConnectionPhase.CONNECTED
    firstRunState = { status: "in_progress" }

    render(<OptionIndex />)

    const banner = await screen.findByTestId("resume-setup-banner")
    expect(banner).toBeInTheDocument()
    expect(screen.queryByTestId("unified-setup-wizard")).not.toBeInTheDocument()
    expect(await screen.findByTestId("companion-home-shell")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Resume setup" }))
    expect(navigateMock).toHaveBeenCalledWith("/setup")

    fireEvent.click(screen.getByRole("button", { name: "Dismiss" }))
    await waitFor(() => {
      expect(screen.queryByTestId("resume-setup-banner")).not.toBeInTheDocument()
    })
  })

  it("renders companion home in the normal app shell after first run", async () => {
    hasCompletedFirstRun = true
    phase = ConnectionPhase.CONNECTED
    firstRunState = { status: "completed" }

    render(<OptionIndex />)

    expect(await screen.findByTestId("companion-home-shell")).toBeInTheDocument()
    expect(screen.getByText("options")).toBeInTheDocument()
    const layoutProps = optionLayoutMock.mock.lastCall?.[0]
    expect(layoutProps).not.toHaveProperty("hideHeader")
    expect(layoutProps).not.toHaveProperty("hideSidebar")
  })

  it("walls a true first run (no connection) with the wizard after hydration", async () => {
    phase = ConnectionPhase.UNCONFIGURED
    firstRunState = null

    render(<OptionIndex />)

    expect(
      await screen.findByTestId("unified-setup-wizard")
    ).toBeInTheDocument()
    expect(checkOnceMock).toHaveBeenCalled()
    expect(
      screen.queryByTestId("resume-setup-banner")
    ).not.toBeInTheDocument()
  })

  it("refreshes connection state for completed first-run users", async () => {
    hasCompletedFirstRun = true
    phase = ConnectionPhase.CONNECTED
    firstRunState = { status: "completed" }

    render(<OptionIndex />)

    await waitFor(() => {
      expect(checkOnceMock).toHaveBeenCalled()
    })
    expect(beginOnboardingMock).not.toHaveBeenCalled()
  })
})
