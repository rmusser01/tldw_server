import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import OptionIndex from "../option-index"
import OptionSetup from "../option-setup"
import OptionOnboardingTest from "../option-onboarding-test"
import { ConnectionPhase } from "@/types/connection"

const state = {
  hasCompletedFirstRun: false
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
    expect(screen.getByText("Home Onboarding")).toBeInTheDocument()
    expect(await screen.findByTestId("onboarding-wizard")).toBeInTheDocument()
    expect(optionLayoutMock).toHaveBeenLastCalledWith(
      expect.objectContaining({
        hideHeader: true,
        hideSidebar: true
      })
    )
    firstRender.unmount()

    const secondRender = render(<OptionSetup />)
    expect(screen.getByText("Setup Wizard")).toBeInTheDocument()
    expect(screen.getByTestId("onboarding-wizard")).toBeInTheDocument()
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

  it("completes onboarding immediately without waiting for connection recheck", async () => {
    optionLayoutMock.mockClear()
    state.hasCompletedFirstRun = false

    let resolveCheck: (() => void) | null = null
    checkOnceMock.mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          resolveCheck = resolve
        })
    )

    render(<OptionIndex />)
    expect(screen.getByTestId("onboarding-wizard")).toBeInTheDocument()

    checkOnceMock.mockClear()
    markFirstRunCompleteMock.mockClear()

    fireEvent.click(screen.getByTestId("onboarding-finish"))

    await waitFor(() => {
      expect(markFirstRunCompleteMock).toHaveBeenCalledTimes(1)
    })
    expect(checkOnceMock).toHaveBeenCalledTimes(1)

    // Prevent unresolved Promise leakage in this test process.
    resolveCheck?.()
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

  it("keeps an explicit theme toggle available on the home onboarding shell", () => {
    render(<OptionIndex />)

    const toggle = screen.getByTestId("chat-header-theme-toggle")
    expect(toggle).toBeInTheDocument()

    fireEvent.click(toggle)

    expect(toggleDarkModeMock).toHaveBeenCalledTimes(1)
  })

  it("surfaces a character-chat first-run lane from preserved route intent", async () => {
    currentLocation = {
      pathname: "/",
      search:
        "?intent=character-chat&returnTo=%2Fcharacters%3Ffrom%3Dheader-select%26create%3Dtrue",
      hash: "",
      state: null
    }

    render(<OptionIndex />)

    expect(screen.getByText("Character Chat Onboarding")).toBeInTheDocument()
    expect(
      screen.getByText("Finish setup, then continue creating and chatting with characters.")
    ).toBeInTheDocument()
    expect(screen.getByTestId("character-chat-onboarding-lane")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Create character" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Import character" })
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Choose model" })).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Start character chat" })
    ).toBeInTheDocument()
    expect(await screen.findByTestId("onboarding-wizard")).toBeInTheDocument()
  })

  it("returns character-chat users to their interrupted route after onboarding finishes", async () => {
    currentLocation = {
      pathname: "/",
      search:
        "?intent=character-chat&returnTo=%2Fcharacters%3Ffrom%3Dheader-select%26create%3Dtrue",
      hash: "",
      state: null
    }

    render(<OptionIndex />)

    fireEvent.click(await screen.findByTestId("onboarding-finish"))

    await waitFor(() => {
      expect(markFirstRunCompleteMock).toHaveBeenCalledTimes(1)
    })
    expect(navigateMock).toHaveBeenCalledWith(
      "/characters?from=header-select&create=true"
    )
  })

  it("renders Companion Home from / after onboarding", async () => {
    state.hasCompletedFirstRun = true

    render(<OptionIndex />)

    expect(await screen.findByTestId("companion-home-shell")).toBeInTheDocument()
    expect(screen.getByText("options")).toBeInTheDocument()
    expect(screen.queryByTestId("landing-hub")).not.toBeInTheDocument()
  })
})
