import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"

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

const mockNoFirstRunBanner = () => {
  useFirstRunCheckMock.mockReturnValue({
    shouldShowSetup: false,
    resumeStep: null,
    loading: false
  })
}

const renderNotices = (props: Partial<PlaygroundComposerNoticesProps> = {}) =>
  render(
    <MemoryRouter>
      <PlaygroundComposerNotices {...buildProps()} {...props} />
    </MemoryRouter>
  )

describe("PlaygroundComposerNotices first-run banner", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
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

  it("keeps the inline assistant setup nudge available inside chat", () => {
    useFirstRunCheckMock.mockReturnValue({
      shouldShowSetup: true,
      resumeStep: null,
      loading: false
    })

    render(<PlaygroundComposerNotices {...buildProps()} />)

    expect(screen.getByTestId("first-run-banner-nudge")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Resume setup" }))

    expect(navigate).toHaveBeenCalledWith("/persona")
  })

  it("warns when banner dismissal cannot be persisted but still hides the banner", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {})
    const setItemSpy = vi.fn(() => {
      throw new Error("storage unavailable")
    })
    vi.stubGlobal("localStorage", {
      clear: vi.fn(),
      getItem: vi.fn(() => null),
      setItem: setItemSpy
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

  it("renders disconnected composer recovery through the shared alert primitive", () => {
    useFirstRunCheckMock.mockReturnValue({
      shouldShowSetup: false,
      resumeStep: null,
      loading: false
    })

    render(
      <MemoryRouter>
        <PlaygroundComposerNotices
          {...buildProps()}
          isConnectionReady={false}
        />
      </MemoryRouter>
    )

    const notice = screen.getByTestId("playground-composer-disconnected-notice")

    expect(notice).toHaveAttribute("data-ds-component", "Alert")
    expect(notice).toHaveAttribute("role", "status")
    expect(screen.getByRole("link", { name: "Open settings" })).toHaveAttribute(
      "href",
      "/settings/tldw"
    )
  })

  it("renders degraded composer recovery through the shared alert primitive", () => {
    const openModelApiSelector = vi.fn()

    useFirstRunCheckMock.mockReturnValue({
      shouldShowSetup: false,
      resumeStep: null,
      loading: false
    })

    render(
      <MemoryRouter>
        <PlaygroundComposerNotices
          {...buildProps()}
          isConnectionReady
          connectionUxState="connected_degraded"
          openModelApiSelector={openModelApiSelector}
        />
      </MemoryRouter>
    )

    const notice = screen.getByTestId("playground-composer-degraded-notice")

    expect(notice).toHaveAttribute("data-ds-component", "Alert")
    expect(notice).toHaveAttribute("role", "status")
    fireEvent.click(screen.getByRole("button", { name: "Switch model" }))
    expect(openModelApiSelector).toHaveBeenCalled()
    expect(screen.getByRole("link", { name: "Health & diagnostics" })).toHaveAttribute(
      "href",
      "/settings/health"
    )
  })

  it("shows the expression nudge for a selected character with no expression images", () => {
    mockNoFirstRunBanner()

    renderNotices({
      selectedCharacter: { id: 42, name: "Ada", extensions: {} },
      selectedCharacterName: "Ada"
    })

    expect(screen.getByText(/add expression images/i)).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Edit expressions" })).toHaveAttribute(
      "href",
      "/characters?from=chat-emote-nudge&focus=expressions"
    )
  })

  it("does not show the expression nudge when mood images are configured", () => {
    mockNoFirstRunBanner()

    renderNotices({
      selectedCharacter: {
        id: 42,
        name: "Ada",
        extensions: {
          tldw: {
            mood_images: {
              happy: "https://example.test/happy.png"
            }
          }
        }
      },
      selectedCharacterName: "Ada"
    })

    expect(screen.queryByText(/add expression images/i)).not.toBeInTheDocument()
  })

  it("dismisses the expression nudge with the full scoped storage key", () => {
    mockNoFirstRunBanner()

    renderNotices({
      selectedCharacter: {
        id: 42,
        name: "Ada",
        server_url: "http://localhost:8000",
        user_id: 7,
        extensions: {}
      },
      selectedCharacterName: "Ada"
    })

    fireEvent.click(
      screen.getByRole("button", { name: /dismiss expression image setup/i })
    )

    expect(
      localStorage.getItem(
        "character-expression-nudge:server:http://localhost:8000:user:7:character:42"
      )
    ).toBe("true")
    expect(screen.queryByText(/add expression images/i)).not.toBeInTheDocument()
  })

  it("keeps expression nudge dismissal scoped by server", () => {
    mockNoFirstRunBanner()

    const { unmount } = renderNotices({
      selectedCharacter: {
        id: 42,
        name: "Ada",
        server_url: "http://localhost:8000",
        extensions: {}
      },
      selectedCharacterName: "Ada"
    })

    fireEvent.click(
      screen.getByRole("button", { name: /dismiss expression image setup/i })
    )
    unmount()

    renderNotices({
      selectedCharacter: {
        id: 42,
        name: "Ada",
        server_url: "http://127.0.0.1:8000",
        extensions: {}
      },
      selectedCharacterName: "Ada"
    })

    expect(screen.getByText(/add expression images/i)).toBeInTheDocument()
  })

  it("dismisses the expression nudge without broad storage when no stable character id exists", () => {
    mockNoFirstRunBanner()

    renderNotices({
      selectedCharacter: { name: "Nameless", extensions: {} },
      selectedCharacterName: "Nameless"
    })

    fireEvent.click(
      screen.getByRole("button", { name: /dismiss expression image setup/i })
    )

    expect(screen.queryByText(/add expression images/i)).not.toBeInTheDocument()
    expect(
      Object.keys(localStorage).some((key) =>
        key.startsWith("character-expression-nudge")
      )
    ).toBe(false)
  })
})
