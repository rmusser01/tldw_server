import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import type { PlaygroundComposerNoticesProps } from "../PlaygroundComposerNotices"

const navigate = vi.fn()
const useFirstRunCheckMock = vi.fn()

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>(
    "react-router-dom"
  )
  return {
    ...actual,
    useNavigate: () => navigate
  }
})

vi.mock("@/hooks/useFirstRunCheck", () => ({
  useFirstRunCheck: () => useFirstRunCheckMock()
}))

vi.mock("@/components/PersonaGarden/FirstRunBanner", () => ({
  FirstRunBanner: ({
    variant,
    onResume,
    onDismiss
  }: {
    variant: "resume" | "nudge"
    onResume?: () => void
    onDismiss: () => void
  }) => (
    <div data-testid={`first-run-banner-${variant}`}>
      <button type="button" onClick={onResume}>
        Resume setup
      </button>
      <button type="button" onClick={onDismiss}>
        Dismiss banner
      </button>
    </div>
  )
}))

vi.mock("../ModelRecommendationsPanel", () => ({
  ModelRecommendationsPanel: () => null
}))

import { PlaygroundComposerNotices } from "../PlaygroundComposerNotices"

const buildProps = (): PlaygroundComposerNoticesProps => ({
  modeAnnouncement: null,
  characterPendingApply: false,
  selectedCharacterGreeting: null,
  selectedCharacterName: null,
  compareModeActive: false,
  compareSelectedModels: [],
  compareSelectedModelLabels: [],
  compareNeedsMoreModels: false,
  compareSharedContextLabels: [],
  compareInteroperabilityNotices: [],
  noticesExpanded: false,
  setNoticesExpanded: vi.fn(),
  contextDeltaLabels: [],
  contextConflictWarnings: [],
  visibleModelRecommendations: [],
  sessionInsightsTotalTokens: 0,
  jsonMode: false,
  isConnectionReady: true,
  connectionUxState: "connected",
  isProMode: false,
  selectedModel: null,
  systemPrompt: null,
  selectedCharacter: null,
  ragPinnedResultsLength: 0,
  startupTemplateDraftName: "",
  setStartupTemplateDraftName: vi.fn(),
  startupTemplates: [],
  handleSaveStartupTemplate: vi.fn(),
  handleOpenStartupTemplatePreview: vi.fn(),
  setOpenModelSettings: vi.fn(),
  setOpenActorSettings: vi.fn(),
  setMessageValue: vi.fn(),
  textAreaFocus: vi.fn(),
  openModelApiSelector: vi.fn(),
  openSessionInsightsModal: vi.fn(),
  handleModelRecommendationAction: vi.fn(),
  dismissModelRecommendation: vi.fn(),
  getModelRecommendationActionLabel: vi.fn(() => "Do it"),
  wrapComposerProfile: (_id, element) => element,
  t: ((_key: string, fallback?: string | { defaultValue?: string }) => {
    if (typeof fallback === "string") return fallback
    return fallback?.defaultValue ?? _key
  }) as PlaygroundComposerNoticesProps["t"]
})

describe("PlaygroundComposerNotices first-run banner", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
  })

  it("shows the resume banner when setup is in progress even if first-run gating is false", () => {
    useFirstRunCheckMock.mockReturnValue({
      shouldShowSetup: false,
      resumeStep: "commands",
      loading: false
    })

    render(<PlaygroundComposerNotices {...buildProps()} />)

    expect(screen.getByTestId("first-run-banner-resume")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Resume setup" }))

    expect(navigate).toHaveBeenCalledWith("/persona")
  })

  it("warns when banner dismissal cannot be persisted but still hides the banner", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {})
    const setItemSpy = vi
      .spyOn(Storage.prototype, "setItem")
      .mockImplementation(() => {
        throw new Error("storage unavailable")
      })

    useFirstRunCheckMock.mockReturnValue({
      shouldShowSetup: true,
      resumeStep: null,
      loading: false
    })

    render(<PlaygroundComposerNotices {...buildProps()} />)

    fireEvent.click(screen.getByRole("button", { name: "Dismiss banner" }))

    expect(warnSpy).toHaveBeenCalledWith(
      "Failed to persist assistant chat nudge dismissal state",
      expect.any(Error)
    )
    expect(setItemSpy).toHaveBeenCalledWith(
      "assistant_nudge_dismissed_chat",
      "true"
    )
    expect(screen.queryByTestId("first-run-banner-nudge")).not.toBeInTheDocument()
  })
})
