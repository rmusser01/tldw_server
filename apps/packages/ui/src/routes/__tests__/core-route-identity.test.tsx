import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { act, render, screen } from "@testing-library/react"
import OptionIndex from "../option-index"
import OptionSetup from "../option-setup"
import OptionOnboardingTest from "../option-onboarding-test"
import { ConnectionPhase } from "@/types/connection"

const state = {
  hasCompletedFirstRun: false
}
const firstRunState = {
  status: "not_started"
}
let phase: ConnectionPhase | null = null
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

const checkOnceMock = vi.fn().mockResolvedValue(undefined)
const beginOnboardingMock = vi.fn().mockResolvedValue(undefined)
const markFirstRunCompleteMock = vi.fn().mockResolvedValue(undefined)
const navigateMock = vi.fn()
let currentLocation = {
  pathname: "/",
  search: "",
  hash: "",
  state: null as unknown
}

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
    hasCompletedFirstRun: state.hasCompletedFirstRun
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

vi.mock("@/hooks/useDarkmode", () => ({
  useDarkMode: () => ({
    mode: "dark",
    toggleDarkMode: toggleDarkModeMock
  })
}))

vi.mock("react-router-dom", () => ({
  useNavigate: () => navigateMock,
  useLocation: () => currentLocation
}))

vi.mock("@/components/Option/Onboarding/OnboardingWizard", () => ({
  OnboardingWizard: ({
    onFinish
  }: {
    onFinish?: () => void | Promise<void>
  }) => (
    <div>
      <div data-testid="onboarding-wizard">Wizard</div>
      <button
        data-testid="onboarding-finish"
        onClick={() => {
          void onFinish?.()
        }}
      >
        Finish onboarding
      </button>
    </div>
  )
}))

vi.mock("@/components/Option/Onboarding/UnifiedSetupWizard", () => ({
  UnifiedSetupWizard: () => (
    <div data-testid="unified-setup-shell">
      <h1>First-time setup</h1>
      <button type="button">Mock unified setup</button>
    </div>
  )
}))

vi.mock("@/hooks/useSetupOnboarding", () => ({
  useSetupOnboarding: () => ({
    state: {
      status: firstRunState.status,
      completed_steps: [],
      skipped_steps: [],
      step_data: {},
      acknowledged_steps: [],
      first_chat: { completed: firstRunState.status === "completed" }
    },
    metadata: {
      auth_mode: "single_user",
      bundled_single_user_auth_available: true,
      manual_auth_required: false,
      setup_required: firstRunState.status !== "completed",
      setup_completed: firstRunState.status === "completed",
      remote_setup_enabled: false,
      connection: { browser_access: "local" },
      setup_paths: [],
      multi_user_exit: { guide_path: "/docs/multi-user" }
    },
    loading: false,
    error: null
  })
}))

vi.mock("~/components/Option/LandingHub", () => ({
  LandingHub: () => <div data-testid="landing-hub">Hub</div>
}))

vi.mock("@/components/Option/CompanionHome", () => ({
  CompanionHomeShell: ({ surface }: { surface: "options" | "sidepanel" }) => (
    <div data-testid="companion-home-shell">{surface}</div>
  )
}))

describe("core route identity guardrails", () => {
  beforeEach(() => {
    optionLayoutMock.mockClear()
    navigateMock.mockClear()
    checkOnceMock.mockReset().mockResolvedValue(undefined)
    beginOnboardingMock.mockReset().mockResolvedValue(undefined)
    markFirstRunCompleteMock.mockReset().mockResolvedValue(undefined)
    toggleDarkModeMock.mockReset()
    state.hasCompletedFirstRun = false
    firstRunState.status = "not_started"
    phase = null
    currentLocation = {
      pathname: "/",
      search: "",
      hash: "",
      state: null
    }
  })

  it("provides unique route-intent headings for home/setup/onboarding-test", async () => {
    const firstRender = render(<OptionIndex />)
    expect(screen.getByRole("heading", { name: "First-time setup" })).toBeInTheDocument()
    expect(screen.getByTestId("unified-setup-shell")).toBeInTheDocument()
    expect(optionLayoutMock).toHaveBeenLastCalledWith(
      expect.objectContaining({
        hideHeader: true,
        hideSidebar: true
      })
    )
    firstRender.unmount()

    firstRunState.status = "completed"
    const secondRender = render(<OptionSetup />)
    expect(
      screen.getByRole("heading", {
        level: 1,
        name: "Setup operator recovery"
      })
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Return home" })).toBeInTheDocument()
    expect(screen.queryByTestId("unified-setup-shell")).not.toBeInTheDocument()
    expect(optionLayoutMock).toHaveBeenLastCalledWith(
      expect.objectContaining({
        hideHeader: true,
        hideSidebar: true
      })
    )
    secondRender.unmount()

    render(<OptionOnboardingTest />)
    expect(screen.getByText("Onboarding Test Harness")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Go to setup" })).toBeInTheDocument()
    expect(screen.getByTestId("onboarding-wizard")).toBeInTheDocument()
    expect(optionLayoutMock).toHaveBeenLastCalledWith(
      expect.objectContaining({
        hideHeader: true,
        hideSidebar: true
      })
    )
  })

  it("does not use local first-run completion as the home resolver", async () => {
    optionLayoutMock.mockClear()
    state.hasCompletedFirstRun = true
    firstRunState.status = "not_started"

    render(<OptionIndex />)

    expect(screen.getByTestId("unified-setup-shell")).toBeInTheDocument()
    expect(screen.queryByTestId("companion-home-shell")).not.toBeInTheDocument()
    expect(markFirstRunCompleteMock).not.toHaveBeenCalled()
  })

  it("renders Companion Home when backend first-run state is complete", async () => {
    firstRunState.status = "completed"

    render(<OptionIndex />)

    expect(await screen.findByTestId("companion-home-shell")).toBeInTheDocument()
    expect(screen.getByText("options")).toBeInTheDocument()
    expect(screen.queryByTestId("landing-hub")).not.toBeInTheDocument()
  })

  it("does not auto-restart onboarding after hydration when the connection is already resolved", async () => {
    phase = ConnectionPhase.CONNECTED

    let resolveCheck: (() => void) | null = null
    checkOnceMock.mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          resolveCheck = resolve
        })
    )

    render(<OptionIndex />)

    await act(async () => {
      resolveCheck?.()
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(checkOnceMock).toHaveBeenCalledTimes(1)
    expect(beginOnboardingMock).not.toHaveBeenCalled()
  })

  it("renders setup in the focused shell while backend setup is required", () => {
    render(<OptionIndex />)

    expect(screen.getByTestId("unified-setup-shell")).toBeInTheDocument()
    expect(optionLayoutMock).toHaveBeenLastCalledWith(
      expect.objectContaining({
        hideHeader: true,
        hideSidebar: true
      })
    )
  })

  it("uses unified setup for character-chat first-run entry without pre-auth action lane", async () => {
    currentLocation = {
      pathname: "/",
      search:
        "?intent=character-chat&returnTo=%2Fcharacters%3Ffrom%3Dheader-select%26create%3Dtrue",
      hash: "",
      state: null
    }

    render(<OptionIndex />)

    expect(screen.getByRole("heading", { name: "First-time setup" })).toBeInTheDocument()
    expect(
      screen.queryByTestId("character-chat-onboarding-lane")
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Create character" })
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Import character" })
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Choose model" })
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Start character chat" })
    ).not.toBeInTheDocument()
    expect(screen.getByTestId("unified-setup-shell")).toBeInTheDocument()
  })
})
