// @vitest-environment jsdom
import React from "react"
import { MemoryRouter } from "react-router-dom"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  createInitialQuickIngestLastRunSummary,
  useQuickIngestStore
} from "@/store/quick-ingest"
import {
  createEmptyQuickIngestSession,
  useQuickIngestSessionStore,
  type QuickIngestSessionRecord
} from "@/store/quick-ingest-session"

const routeMocks = vi.hoisted(() => ({
  firstRunState: {
    current: {
      status: "not_started",
      completed_steps: [],
      skipped_steps: [],
      step_data: {},
      acknowledged_steps: [],
      first_chat: { completed: false }
    }
  },
  requestQuickIngestOpen: vi.fn(),
  tldwConfig: {
    current: {
      serverUrl: "http://localhost:3000",
      authMode: "single-user",
      apiKey: "test-api-key"
    } as Record<string, unknown>
  },
  getConfig: vi.fn(),
  listMedia: vi.fn(),
  updateConfig: vi.fn()
}))

vi.mock("~/components/Layouts/Layout", () => ({
  __esModule: true,
  default: ({ children, hideHeader, hideSidebar }: any) => (
    <main
      data-hide-header={String(Boolean(hideHeader))}
      data-hide-sidebar={String(Boolean(hideSidebar))}
    >
      {children}
    </main>
  )
}))

vi.mock("@/hooks/useDarkmode", () => ({
  useDarkMode: () => ({
    mode: "light",
    toggleDarkMode: vi.fn()
  })
}))

vi.mock("@/hooks/useComposerFocus", () => ({
  useFocusComposerOnConnect: vi.fn()
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionActions: () => ({
    checkOnce: vi.fn().mockResolvedValue(undefined),
    beginOnboarding: vi.fn(),
    markFirstRunComplete: vi.fn()
  }),
  useConnectionState: () => ({
    phase: "connected"
  }),
  useConnectionUxState: () => ({
    hasCompletedFirstRun: true
  })
}))

vi.mock("@/services/tldw/deployment-mode", () => ({
  isHostedTldwDeployment: () => false
}))

vi.mock("@/components/Option/CompanionHome", () => ({
  CompanionHomeShell: () => <section data-testid="companion-home" />
}))

vi.mock("@/utils/quick-ingest-open", () => ({
  isFirstSourceOpenDetail: (detail: any) =>
    Boolean(
      detail &&
        (detail.source === "first_source_milestone" ||
          detail.firstSource === true)
    ),
  isFirstSourceQuickIngestKind: (value: unknown) =>
    value === "web_url" || value === "file_upload" || value === "paste_text",
  requestQuickIngestOpen: routeMocks.requestQuickIngestOpen
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: routeMocks.getConfig,
    listMedia: routeMocks.listMedia,
    updateConfig: routeMocks.updateConfig
  }
}))

vi.mock("@/hooks/useSetupOnboarding", () => ({
  useSetupOnboarding: () => ({
    state: routeMocks.firstRunState.current,
    metadata: {
      auth_mode: "single_user",
      bundled_single_user_auth_available: true,
      manual_auth_required: false,
      setup_required: true,
      setup_completed: false,
      remote_setup_enabled: false,
      connection: { browser_access: "local" },
      setup_paths: [],
      multi_user_exit: { guide_path: "/docs/multi-user" }
    },
    providerCatalog: [],
    audioRecommendations: [],
    loading: false,
    error: null,
    refresh: vi.fn(),
    adoptState: vi.fn(),
    loadProviderCatalog: vi.fn(),
    loadAudioRecommendations: vi.fn(),
    saveStep: vi.fn(),
    skip: vi.fn(),
    saveProvider: vi.fn(),
    saveIngestDefaults: vi.fn(),
    saveAudioDefaults: vi.fn(),
    saveOptionalAdvanced: vi.fn(),
    verifyFirstChat: vi.fn(),
    complete: vi.fn()
  })
}))

const createCompletedFirstRunState = () => ({
  status: "completed",
  completed_steps: ["first_chat"],
  skipped_steps: [],
  step_data: {},
  acknowledged_steps: ["first_chat"],
  first_chat: { completed: true }
})

const seedQuickIngestSession = (
  overrides: Partial<QuickIngestSessionRecord>
) => {
  const base = createEmptyQuickIngestSession()
  useQuickIngestSessionStore.setState((state) => ({
    ...state,
    session: {
      ...base,
      ...overrides,
      resultSummary: {
        ...base.resultSummary,
        ...(overrides.resultSummary || {})
      }
    },
    triggerSummary: { count: 0, label: null, hadFailure: false }
  }))
}

describe("OptionIndex unified setup resolver", () => {
  beforeEach(() => {
    window.localStorage.clear()
    routeMocks.requestQuickIngestOpen.mockReset()
    routeMocks.firstRunState.current = {
      status: "not_started",
      completed_steps: [],
      skipped_steps: [],
      step_data: {},
      acknowledged_steps: [],
      first_chat: { completed: false }
    }
    routeMocks.tldwConfig.current = {
      serverUrl: "http://localhost:3000",
      authMode: "single-user",
      apiKey: "test-api-key"
    }
    routeMocks.getConfig.mockReset()
    routeMocks.getConfig.mockImplementation(
      async () => routeMocks.tldwConfig.current
    )
    routeMocks.listMedia.mockReset()
    routeMocks.listMedia.mockResolvedValue({ items: [] })
    routeMocks.updateConfig.mockReset()
    routeMocks.updateConfig.mockImplementation(async (updates) => {
      routeMocks.tldwConfig.current = {
        ...routeMocks.tldwConfig.current,
        ...(updates as Record<string, unknown>)
      }
    })
    useQuickIngestStore.setState((state) => ({
      ...state,
      queuedCount: 0,
      hadRecentFailure: false,
      lastRunSummary: createInitialQuickIngestLastRunSummary()
    }))
    useQuickIngestSessionStore.setState((state) => ({
      ...state,
      session: null,
      triggerSummary: { count: 0, label: null, hadFailure: false }
    }))
  })

  it("renders setup in focused shell when backend state is not complete", async () => {
    const { default: OptionIndex } = await import("../option-index")

    render(
      <MemoryRouter>
        <OptionIndex />
      </MemoryRouter>
    )

    expect(screen.getByRole("main")).toHaveAttribute("data-hide-header", "true")
    expect(screen.getByRole("main")).toHaveAttribute("data-hide-sidebar", "true")
    expect(
      screen.getByRole("heading", { name: /first-time setup/i })
    ).toBeInTheDocument()
  })

  it("offers the first-source milestone after authenticated media readiness succeeds", async () => {
    routeMocks.firstRunState.current = createCompletedFirstRunState()
    const { default: OptionIndex } = await import("../option-index")

    render(
      <MemoryRouter>
        <OptionIndex />
      </MemoryRouter>
    )

    expect(await screen.findByRole("heading", { name: /add your first source/i })
    ).toBeInTheDocument()
    await waitFor(() => {
      expect(routeMocks.listMedia).toHaveBeenCalledWith({
        results_per_page: 1
      })
    })
    fireEvent.click(screen.getByRole("button", { name: /add source/i }))

    expect(routeMocks.requestQuickIngestOpen).toHaveBeenCalledWith(
      {
        source: "first_source_milestone",
        preferredPreset: "quick",
        firstSource: true,
        firstSourceKind: "web_url"
      },
      { focusTrigger: true }
    )
  })

  it("passes the selected first-source kind into quick ingest", async () => {
    routeMocks.firstRunState.current = createCompletedFirstRunState()
    const { default: OptionIndex } = await import("../option-index")

    render(
      <MemoryRouter>
        <OptionIndex />
      </MemoryRouter>
    )

    expect(
      await screen.findByRole("heading", { name: /add your first source/i })
    ).toBeInTheDocument()
    fireEvent.click(screen.getByRole("radio", { name: /file/i }))
    fireEvent.click(screen.getByRole("button", { name: /add source/i }))
    fireEvent.click(screen.getByRole("radio", { name: /paste/i }))
    fireEvent.click(screen.getByRole("button", { name: /add source/i }))

    expect(routeMocks.requestQuickIngestOpen).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        firstSourceKind: "file_upload"
      }),
      { focusTrigger: true }
    )
    expect(routeMocks.requestQuickIngestOpen).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        firstSourceKind: "paste_text"
      }),
      { focusTrigger: true }
    )
  })

  it("does not offer source chat before quick ingest returns a ready media id", async () => {
    routeMocks.firstRunState.current = createCompletedFirstRunState()
    const { default: OptionIndex } = await import("../option-index")

    render(
      <MemoryRouter>
        <OptionIndex />
      </MemoryRouter>
    )

    expect(
      await screen.findByRole("heading", { name: /add your first source/i })
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: /ask a question about this source/i })
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: /summarize this source/i })
    ).not.toBeInTheDocument()
  })

  it("ignores unrelated quick ingest success when no first-source session owns it", async () => {
    routeMocks.firstRunState.current = createCompletedFirstRunState()
    useQuickIngestStore.setState((state) => ({
      ...state,
      lastRunSummary: {
        ...createInitialQuickIngestLastRunSummary(),
        status: "success",
        attemptedAt: 1,
        completedAt: 2,
        totalCount: 1,
        successCount: 1,
        firstMediaId: "unrelated-42",
        primarySourceLabel: "Unrelated import"
      }
    }))
    const { default: OptionIndex } = await import("../option-index")

    render(
      <MemoryRouter>
        <OptionIndex />
      </MemoryRouter>
    )

    expect(
      await screen.findByRole("heading", { name: /add your first source/i })
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: /ask a question about this source/i })
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: /summarize this source/i })
    ).not.toBeInTheDocument()
  })

  it("uses persisted first-source session result summary after reload", async () => {
    routeMocks.firstRunState.current = createCompletedFirstRunState()
    seedQuickIngestSession({
      lifecycle: "completed",
      openDetail: {
        source: "first_source_milestone",
        preferredPreset: "quick",
        firstSource: true,
        firstSourceKind: "file_upload"
      },
      resultSummary: {
        status: "success",
        attemptedAt: 1,
        completedAt: 2,
        totalCount: 1,
        successCount: 1,
        failedCount: 0,
        cancelledCount: 0,
        firstMediaId: "persisted-42",
        primarySourceLabel: "Saved PDF",
        errorMessage: null
      }
    })
    const discussEvents: Array<CustomEvent> = []
    const listener = (event: Event) => {
      discussEvents.push(event as CustomEvent)
    }
    window.addEventListener("tldw:discuss-media", listener)
    const { default: OptionIndex } = await import("../option-index")

    try {
      render(
        <MemoryRouter>
          <OptionIndex />
        </MemoryRouter>
      )

      expect(
        await screen.findByRole("heading", { name: /add your first source/i })
      ).toBeInTheDocument()

      expect(screen.getByText(/starter questions/i)).toBeInTheDocument()
      expect(
        screen.getByRole("button", { name: /list the key claims/i })
      ).toBeInTheDocument()

      fireEvent.click(
        await screen.findByRole("button", {
          name: /summarize this source/i
        })
      )

      expect(discussEvents).toHaveLength(1)
      expect(discussEvents[0]?.detail).toEqual({
        mediaId: "persisted-42",
        title: "Saved PDF",
        mode: "rag_media",
        content: "Summarize this source."
      })
    } finally {
      window.removeEventListener("tldw:discuss-media", listener)
    }
  })

  it("retries first-source ingest with the persisted source kind after reload", async () => {
    routeMocks.firstRunState.current = createCompletedFirstRunState()
    seedQuickIngestSession({
      lifecycle: "completed",
      openDetail: {
        source: "first_source_milestone",
        preferredPreset: "quick",
        firstSource: true,
        firstSourceKind: "paste_text"
      },
      firstSourceAddMode: "paste_text",
      resultSummary: {
        status: "error",
        attemptedAt: 1,
        completedAt: 2,
        totalCount: 1,
        successCount: 0,
        failedCount: 1,
        cancelledCount: 0,
        firstMediaId: null,
        primarySourceLabel: "Pasted notes",
        errorMessage: "Upload failed"
      }
    })
    const { default: OptionIndex } = await import("../option-index")

    render(
      <MemoryRouter>
        <OptionIndex />
      </MemoryRouter>
    )

    fireEvent.click(await screen.findByRole("button", { name: /retry/i }))

    expect(routeMocks.requestQuickIngestOpen).toHaveBeenCalledWith(
      expect.objectContaining({
        firstSourceKind: "paste_text"
      }),
      { focusTrigger: true }
    )
  })

  it("does not show first-source processing for an unrelated processing session", async () => {
    routeMocks.firstRunState.current = createCompletedFirstRunState()
    seedQuickIngestSession({
      lifecycle: "processing",
      openDetail: { source: "manual" }
    })
    const { default: OptionIndex } = await import("../option-index")

    render(
      <MemoryRouter>
        <OptionIndex />
      </MemoryRouter>
    )

    expect(
      await screen.findByRole("heading", { name: /add your first source/i })
    ).toBeInTheDocument()
    expect(screen.queryByText(/processing your source/i)).not.toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /add source/i })
    ).toBeInTheDocument()
  })

  it("shows inline API key recovery when setup is complete but media auth is missing", async () => {
    routeMocks.firstRunState.current = createCompletedFirstRunState()
    routeMocks.tldwConfig.current = {
      serverUrl: "http://localhost:3000",
      authMode: "single-user"
    }
    const { default: OptionIndex } = await import("../option-index")

    render(
      <MemoryRouter>
        <OptionIndex />
      </MemoryRouter>
    )

    expect(
      await screen.findByRole("heading", { name: /restore media access/i })
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("heading", { name: /add your first source/i })
    ).not.toBeInTheDocument()
  })

  it("saves a recovered API key, rechecks readiness, and then shows the first-source milestone", async () => {
    routeMocks.firstRunState.current = createCompletedFirstRunState()
    routeMocks.tldwConfig.current = {
      serverUrl: "http://localhost:3000",
      authMode: "single-user"
    }
    const { default: OptionIndex } = await import("../option-index")

    render(
      <MemoryRouter>
        <OptionIndex />
      </MemoryRouter>
    )

    const keyInput = await screen.findByLabelText(/single-user API key/i)
    fireEvent.change(keyInput, {
      target: { value: "recovered-api-key" }
    })
    fireEvent.click(screen.getByRole("button", { name: /save API key/i }))

    await waitFor(() => {
      expect(routeMocks.updateConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          serverUrl: "http://localhost:3000",
          authMode: "single-user",
          apiKey: "recovered-api-key"
        })
      )
    })
    expect(
      await screen.findByRole("heading", { name: /add your first source/i })
    ).toBeInTheDocument()
  })

  it("does not render the first-source CTA while media readiness is still checking", async () => {
    routeMocks.firstRunState.current = createCompletedFirstRunState()
    routeMocks.listMedia.mockReturnValue(new Promise(() => undefined))
    const { default: OptionIndex } = await import("../option-index")

    render(
      <MemoryRouter>
        <OptionIndex />
      </MemoryRouter>
    )

    await waitFor(() => {
      expect(routeMocks.listMedia).toHaveBeenCalledWith({
        results_per_page: 1
      })
    })
    expect(
      screen.queryByRole("heading", { name: /add your first source/i })
    ).not.toBeInTheDocument()
  })
})
