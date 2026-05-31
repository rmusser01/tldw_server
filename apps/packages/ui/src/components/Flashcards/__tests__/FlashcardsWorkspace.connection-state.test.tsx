import React from "react"
import { fireEvent, render, screen, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { getDesignSystemState } from "@/design-system"
import { FlashcardsWorkspace } from "../FlashcardsWorkspace"

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
    hasFlashcards: true
  } as {
    hasFlashcards: boolean
  },
  capsLoading: false,
  navigate: vi.fn(),
  scrollToServerCard: vi.fn(),
  checkOnce: vi.fn(),
  setupRequiredLabel: "Registry Setup Required",
  translationKeys: [] as string[]
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string }
    ) => {
      mocks.translationKeys.push(key)
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      return fallbackOrOptions?.defaultValue ?? key
    }
  })
}))

vi.mock("react-router-dom", () => ({
  useNavigate: () => mocks.navigate
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => mocks.isOnline
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

vi.mock("@/design-system", async (importActual) => {
  const actual = await importActual<typeof import("@/design-system")>()
  return {
    ...actual,
    getDesignSystemState: vi.fn(
      (key: Parameters<typeof actual.getDesignSystemState>[0]) => {
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

describe("FlashcardsWorkspace connection states", () => {
  beforeEach(() => {
    mocks.isOnline = true
    mocks.demoEnabled = false
    mocks.uxState = "connected_ok"
    mocks.hasCompletedFirstRun = true
    mocks.capabilities = { hasFlashcards: true }
    mocks.capsLoading = false
    mocks.navigate.mockReset()
    mocks.scrollToServerCard.mockReset()
    mocks.checkOnce.mockReset()
    mocks.setupRequiredLabel = "Registry Setup Required"
    mocks.translationKeys = []
    vi.mocked(getDesignSystemState).mockClear()
  })

  it("keeps demo preview visible while surfacing auth guidance", () => {
    mocks.isOnline = false
    mocks.demoEnabled = true
    mocks.uxState = "error_auth"

    render(<FlashcardsWorkspace />)

    expect(screen.getByText("Explore Flashcards in demo mode")).toBeInTheDocument()
    expect(screen.getByText("Try sample flashcards")).toBeInTheDocument()
    expect(
      screen.getByText("Demo stays available, but your Flashcards credentials need attention.")
    ).toBeInTheDocument()

    fireEvent.click(screen.getAllByRole("button", { name: "Go to server card" })[0])
    expect(mocks.scrollToServerCard).toHaveBeenCalled()
  })

  it("shows setup guidance when demo mode is disabled", () => {
    mocks.isOnline = false
    mocks.demoEnabled = false
    mocks.uxState = "unconfigured"
    mocks.hasCompletedFirstRun = false

    render(<FlashcardsWorkspace />)

    expect(
      screen.getByText("Finish setup to use Flashcards")
    ).toBeInTheDocument()
    expect(screen.getByText(mocks.setupRequiredLabel)).toBeInTheDocument()
    expect(getDesignSystemState).toHaveBeenCalledWith("setup_required")

    fireEvent.click(screen.getByRole("button", { name: "Go to server card" }))
    expect(mocks.scrollToServerCard).toHaveBeenCalled()
  })

  it("preserves study workspace identity and modes while setup is required", () => {
    mocks.isOnline = false
    mocks.demoEnabled = false
    mocks.uxState = "unconfigured"

    render(<FlashcardsWorkspace />)

    expect(screen.getByRole("heading", { level: 1, name: "Flashcards" })).toBeInTheDocument()
    expect(screen.getByText("Study workspace")).toBeInTheDocument()

    const modes = within(screen.getByRole("navigation", { name: "Flashcards modes" }))
    for (const mode of ["Study", "Manage", "Import / Export", "Templates", "Scheduler"]) {
      expect(modes.getByText(mode)).toBeInTheDocument()
    }
    expect(mocks.translationKeys).toContain("option:flashcards.importExport")
    expect(mocks.translationKeys).not.toContain("option:flashcards.tabImportExport")
  })

  it("keeps demo preview visible while surfacing unreachable guidance", () => {
    mocks.isOnline = false
    mocks.demoEnabled = true
    mocks.uxState = "error_unreachable"

    render(<FlashcardsWorkspace />)

    expect(screen.getByText("Try sample flashcards")).toBeInTheDocument()
    expect(
      screen.getByText("Demo stays available, but your tldw server is unreachable.")
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Retry connection" })).toBeInTheDocument()
  })
})
