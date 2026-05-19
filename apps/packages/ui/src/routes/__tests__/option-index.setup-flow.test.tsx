import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"

import { ConnectionPhase } from "@/types/connection"
import OptionIndex from "../option-index"

let hostedMode = false
let hasCompletedFirstRun = false
let phase: ConnectionPhase | null = null
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

vi.mock("@/components/Option/Onboarding/OnboardingWizard", () => ({
  OnboardingWizard: ({
    entryIntent,
    returnTo,
    onFinish
  }: {
    entryIntent?: string
    returnTo?: string | null
    onFinish?: () => void | Promise<void>
  }) => (
    <div>
      <div data-testid="onboarding-wizard">Wizard</div>
      <div data-testid="onboarding-entry-intent">{entryIntent ?? "none"}</div>
      <div data-testid="onboarding-return-to">{returnTo ?? "none"}</div>
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
    render(<OptionIndex />)

    expect(screen.getByRole("heading", { name: "Home Onboarding" })).toBeInTheDocument()
    expect(await screen.findByTestId("onboarding-wizard")).toBeInTheDocument()
    expect(optionLayoutMock).toHaveBeenLastCalledWith(
      expect.objectContaining({
        hideHeader: true,
        hideSidebar: true
      })
    )
  })

  it("preserves character-chat onboarding intent and return target", async () => {
    currentLocation = {
      pathname: "/",
      search:
        "?intent=character-chat&returnTo=%2Fcharacters%3Ffrom%3Dheader-select%26create%3Dtrue",
      hash: "",
      state: null
    }

    render(<OptionIndex />)

    expect(
      screen.getByRole("heading", { name: "Character Chat Onboarding" })
    ).toBeInTheDocument()
    expect(await screen.findByTestId("onboarding-entry-intent")).toHaveTextContent(
      "character-chat"
    )
    expect(screen.getByTestId("onboarding-return-to")).toHaveTextContent(
      "/characters?from=header-select&create=true"
    )

    fireEvent.click(screen.getByTestId("onboarding-finish"))

    await waitFor(() => {
      expect(markFirstRunCompleteMock).toHaveBeenCalledTimes(1)
    })
    expect(navigateMock).toHaveBeenCalledWith(
      "/characters?from=header-select&create=true"
    )
  })

  it("renders companion home in the normal app shell after first run", async () => {
    hasCompletedFirstRun = true
    phase = ConnectionPhase.CONNECTED

    render(<OptionIndex />)

    expect(await screen.findByTestId("companion-home-shell")).toBeInTheDocument()
    expect(screen.getByText("options")).toBeInTheDocument()
    const layoutProps = optionLayoutMock.mock.lastCall?.[0]
    expect(layoutProps).not.toHaveProperty("hideHeader")
    expect(layoutProps).not.toHaveProperty("hideSidebar")
  })

  it("begins onboarding after hydration when first run is unconfigured", async () => {
    phase = ConnectionPhase.UNCONFIGURED

    render(<OptionIndex />)

    await waitFor(() => {
      expect(beginOnboardingMock).toHaveBeenCalledTimes(1)
    })
  })

  it("refreshes connection state for completed first-run users", async () => {
    hasCompletedFirstRun = true
    phase = ConnectionPhase.CONNECTED

    render(<OptionIndex />)

    await waitFor(() => {
      expect(checkOnceMock).toHaveBeenCalled()
    })
    expect(beginOnboardingMock).not.toHaveBeenCalled()
  })
})
