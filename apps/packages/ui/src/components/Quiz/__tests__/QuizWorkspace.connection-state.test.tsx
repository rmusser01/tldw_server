import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { getDesignSystemState } from "@/design-system"
import { QuizWorkspace } from "../QuizWorkspace"

const mocks = vi.hoisted(() => ({
  isOnline: true,
  demoEnabled: false,
  uxState: "connected_ok" as
    | "connected_ok"
    | "testing"
    | "configuring_url"
    | "configuring_auth"
    | "error_auth"
    | "error_unreachable"
    | "unconfigured",
  hasCompletedFirstRun: true,
  capabilities: {
    hasQuizzes: true,
    specVersion: "test-spec"
  } as {
    hasQuizzes: boolean
    specVersion: string | null
  },
  capsLoading: false,
  navigate: vi.fn(),
  scrollToServerCard: vi.fn(),
  checkOnce: vi.fn(),
  setupRequiredLabel: "Registry Setup Required",
  missingDesignSystemStateKeys: new Set<string>()
}))

const interpolate = (template: string, values?: Record<string, unknown>) =>
  template.replace(/\{\{(\w+)\}\}/g, (_, key) => String(values?.[key] ?? ""))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?:
        | string
        | {
            defaultValue?: string
            [k: string]: unknown
          }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      const template = fallbackOrOptions?.defaultValue || key
      return interpolate(template, fallbackOrOptions as Record<string, unknown> | undefined)
    }
  })
}))

vi.mock("react-router-dom", () => ({
  useNavigate: () => mocks.navigate
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => mocks.isOnline
}))

vi.mock("@/context/demo-mode", () => ({
  useDemoMode: () => ({ demoEnabled: mocks.demoEnabled })
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: mocks.capabilities,
    loading: mocks.capsLoading
  })
}))

vi.mock("@/hooks/useScrollToServerCard", () => ({
  useScrollToServerCard: () => mocks.scrollToServerCard
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionUxState: () => ({
    uxState: mocks.uxState,
    hasCompletedFirstRun: mocks.hasCompletedFirstRun
  }),
  useConnectionActions: () => ({
    checkOnce: mocks.checkOnce
  })
}))

vi.mock("@/design-system", async (importActual) => {
  const actual = await importActual<typeof import("@/design-system")>()
  return {
    ...actual,
    getDesignSystemState: vi.fn(
      (key: Parameters<typeof actual.getDesignSystemState>[0]) => {
        if (mocks.missingDesignSystemStateKeys.has(key)) {
          return undefined as unknown as ReturnType<typeof actual.getDesignSystemState>
        }

        const state = actual.getDesignSystemState(key)
        return state
          ? {
              ...state,
              label:
                key === "setup_required" ? mocks.setupRequiredLabel : state.label
            }
          : state
      }
    )
  }
})

vi.mock("../hooks", () => ({
  useQuizzesQuery: () => ({ data: undefined, isLoading: false }),
  useAttemptsQuery: () => ({ data: { count: 0 } })
}))

vi.mock("../QuizPlayground", () => ({
  QuizPlayground: () => <div data-testid="quiz-playground" />
}))

describe("QuizWorkspace connection and availability states", () => {
  beforeEach(() => {
    mocks.isOnline = true
    mocks.demoEnabled = false
    mocks.uxState = "connected_ok"
    mocks.hasCompletedFirstRun = true
    mocks.capabilities = { hasQuizzes: true, specVersion: "test-spec" }
    mocks.capsLoading = false
    mocks.navigate.mockReset()
    mocks.scrollToServerCard.mockReset()
    mocks.checkOnce.mockReset()
    mocks.setupRequiredLabel = "Registry Setup Required"
    mocks.missingDesignSystemStateKeys.clear()
    vi.mocked(getDesignSystemState).mockClear()
  })

  it("renders an interactive local demo quiz flow when offline demo mode is enabled", async () => {
    mocks.isOnline = false
    mocks.demoEnabled = true

    render(<QuizWorkspace />)

    expect(screen.getByText("Explore Quiz Playground in demo mode")).toBeInTheDocument()
    expect(screen.getByTestId("quiz-demo-preview")).toBeInTheDocument()
    expect(
      screen.getAllByText("Demo quiz: Research workflow fundamentals").length
    ).toBeGreaterThan(0)

    fireEvent.click(screen.getByTestId("quiz-demo-start"))
    expect(screen.getByTestId("quiz-demo-taking")).toBeInTheDocument()

    fireEvent.click(screen.getByLabelText("Generate"))
    fireEvent.click(screen.getByRole("button", { name: "Next" }))

    fireEvent.click(screen.getByLabelText("True"))
    fireEvent.click(screen.getByRole("button", { name: "Next" }))

    fireEvent.change(screen.getByPlaceholderText("Enter one word"), {
      target: { value: "trends" }
    })
    fireEvent.click(screen.getByTestId("quiz-demo-submit"))

    await waitFor(() => {
      expect(screen.getByTestId("quiz-demo-results")).toBeInTheDocument()
      expect(screen.getByTestId("quiz-demo-score")).toHaveTextContent("100%")
    })
  })

  it("keeps demo preview visible while surfacing auth guidance", () => {
    mocks.isOnline = false
    mocks.demoEnabled = true
    mocks.uxState = "error_auth"

    render(<QuizWorkspace />)

    expect(screen.getByText("Explore Quiz Playground in demo mode")).toBeInTheDocument()
    expect(screen.getByTestId("quiz-demo-preview")).toBeInTheDocument()
    expect(
      screen.getByText(
        "Demo stays available, but your Quiz Playground credentials need attention."
      )
    ).toBeInTheDocument()

    fireEvent.click(screen.getAllByRole("button", { name: "Go to server card" })[0])
    expect(mocks.scrollToServerCard).toHaveBeenCalled()
  })

  it("shows setup guidance when demo mode is disabled and first-run setup is incomplete", () => {
    mocks.isOnline = false
    mocks.demoEnabled = false
    mocks.uxState = "unconfigured"
    mocks.hasCompletedFirstRun = false

    render(<QuizWorkspace />)

    expect(
      screen.getByText("Finish setup to use Quiz Playground")
    ).toBeInTheDocument()
    expect(screen.getByText(mocks.setupRequiredLabel)).toBeInTheDocument()
    expect(getDesignSystemState).toHaveBeenCalledWith("setup_required")

    fireEvent.click(screen.getByRole("button", { name: "Go to server card" }))
    expect(mocks.scrollToServerCard).toHaveBeenCalled()
  })

  it("keeps setup guidance renderable when the setup state registry entry is unavailable", () => {
    mocks.isOnline = false
    mocks.demoEnabled = false
    mocks.uxState = "unconfigured"
    mocks.hasCompletedFirstRun = false
    mocks.missingDesignSystemStateKeys.add("setup_required")

    expect(() => render(<QuizWorkspace />)).not.toThrow()

    expect(
      screen.getByText("Finish setup to use Quiz Playground")
    ).toBeInTheDocument()
    expect(getDesignSystemState).toHaveBeenCalledWith("setup_required")
  })

  it("shows unreachable guidance without hiding the demo preview", () => {
    mocks.isOnline = false
    mocks.demoEnabled = true
    mocks.uxState = "error_unreachable"

    render(<QuizWorkspace />)

    expect(screen.getByTestId("quiz-demo-preview")).toBeInTheDocument()
    expect(
      screen.getByText("Demo stays available, but your tldw server is unreachable.")
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Retry connection" })).toBeInTheDocument()
  })

  it("shows actionable capability guidance with diagnostics and setup actions", () => {
    mocks.capabilities = { hasQuizzes: false, specVersion: "2026.02" }

    render(<QuizWorkspace />)

    expect(screen.getByText("Quiz API not available on this server")).toBeInTheDocument()
    expect(screen.getByText(/reported: 2026.02/)).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Health & diagnostics" }))
    fireEvent.click(screen.getByRole("button", { name: "Open setup guide" }))

    expect(mocks.navigate).toHaveBeenCalledWith("/settings/health")
    expect(mocks.navigate).toHaveBeenCalledWith("/documentation")
  })

  it("exposes beta semantics via an accessible tooltip badge without blocking main content", async () => {
    render(<QuizWorkspace />)

    expect(screen.getByTestId("quiz-playground")).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("quiz-beta-badge"))

    expect(screen.getByTestId("quiz-beta-tooltip")).toHaveTextContent(
      "Quiz Playground is in beta."
    )

    fireEvent.keyDown(window, { key: "Escape" })

    await waitFor(() => {
      expect(screen.queryByTestId("quiz-beta-tooltip")).not.toBeInTheDocument()
    })
  })
})
