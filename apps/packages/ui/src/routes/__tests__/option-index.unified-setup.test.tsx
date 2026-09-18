// @vitest-environment jsdom
import React from "react"
import { MemoryRouter, useLocation } from "react-router-dom"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
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
import { buildChatSurfaceScopeKeyFromConfig } from "@/services/chat-surface-scope"
import { useMilestoneStore } from "@/store/milestones"
import { DISCUSS_MEDIA_PROMPT_SETTING } from "@/services/settings/ui-settings"

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
  runtimeApiKey: {
    current: null as string | null
  },
  getConfig: vi.fn(),
  listMedia: vi.fn(),
  updateConfig: vi.fn(),
  setSetting: vi.fn()
}))

vi.mock("@/services/settings/registry", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/services/settings/registry")>()),
  setSetting: routeMocks.setSetting
}))

function RouteProbe() {
  return <span data-testid="route">{useLocation().pathname}</span>
}

vi.mock("~/components/Layouts/Layout", () => ({
  __esModule: true,
  default: ({
    children,
    hideHeader,
    hideSidebar
  }: {
    children: React.ReactNode
    hideHeader?: boolean
    hideSidebar?: boolean
  }) => (
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
  isFirstSourceOpenDetail: (
    detail?: { source?: string; firstSource?: boolean } | null
  ) =>
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

vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: () => routeMocks.runtimeApiKey.current
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

const ownerScope = buildChatSurfaceScopeKeyFromConfig({
  serverUrl: "http://localhost:3000",
  authMode: "single-user"
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
    useMilestoneStore.getState().resetMilestones()
    routeMocks.setSetting.mockReset().mockResolvedValue(undefined)
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
    routeMocks.runtimeApiKey.current = null
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

  it("keeps connected users in the app with a resume-setup banner when backend state is not complete", async () => {
    // The connection mock reports "connected": incomplete backend setup must
    // no longer wall the home route with the wizard (#2871) - the operator
    // gets the normal shell plus a dismissible resume-setup banner instead.
    const { default: OptionIndex } = await import("../option-index")

    render(
      <MemoryRouter>
        <OptionIndex />
      </MemoryRouter>
    )

    expect(await screen.findByTestId("resume-setup-banner")).toBeInTheDocument()
    expect(screen.getByRole("main")).toHaveAttribute(
      "data-hide-header",
      "false"
    )
    expect(screen.getByRole("main")).toHaveAttribute(
      "data-hide-sidebar",
      "false"
    )
    expect(
      screen.queryByRole("heading", { name: /first-time setup/i })
    ).not.toBeInTheDocument()
  })

  it("offers the first-source milestone after authenticated media readiness succeeds", async () => {
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
    await waitFor(() => {
      expect(routeMocks.listMedia).toHaveBeenCalledWith({
        results_per_page: 1
      })
    })
    fireEvent.click(screen.getByRole("button", { name: /add source/i }))

    expect(routeMocks.requestQuickIngestOpen).toHaveBeenCalledWith(
      {
        source: "first_source_milestone",
        ownerScope,
        preferredPreset: "quick",
        firstSource: true,
        firstSourceKind: "web_url"
      },
      { focusTrigger: true }
    )
  })

  it("accepts runtime single-user auth for post-onboarding media readiness", async () => {
    routeMocks.firstRunState.current = createCompletedFirstRunState()
    routeMocks.tldwConfig.current = {
      serverUrl: "http://localhost:3000",
      authMode: "single-user"
    }
    routeMocks.runtimeApiKey.current = "runtime-api-key"
    const { default: OptionIndex } = await import("../option-index")

    render(
      <MemoryRouter>
        <OptionIndex />
      </MemoryRouter>
    )

    expect(
      await screen.findByRole("heading", { name: /add your first source/i })
    ).toBeInTheDocument()
    await waitFor(() => {
      expect(routeMocks.listMedia).toHaveBeenCalledWith({
        results_per_page: 1
      })
    })
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
      screen.queryByRole("button", {
        name: /ask a question about this source/i
      })
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
      screen.queryByRole("button", {
        name: /ask a question about this source/i
      })
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: /summarize this source/i })
    ).not.toBeInTheDocument()
  })

  it("uses persisted first-source session result summary after reload", async () => {
    routeMocks.firstRunState.current = createCompletedFirstRunState()
    let finishSaving!: () => void
    routeMocks.setSetting.mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          finishSaving = resolve
        })
    )
    seedQuickIngestSession({
      lifecycle: "completed",
      openDetail: {
        source: "first_source_milestone",
        ownerScope,
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
        firstMediaId: "42",
        primarySourceLabel: "Saved PDF",
        errorMessage: null
      }
    })
    const { default: OptionIndex } = await import("../option-index")

    render(
      <MemoryRouter>
        <OptionIndex />
        <RouteProbe />
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

    expect(screen.getByTestId("route")).toHaveTextContent(/^\/$/)
    finishSaving()
    await waitFor(() =>
      expect(screen.getByTestId("route")).toHaveTextContent("/chat")
    )
    expect(routeMocks.setSetting).toHaveBeenCalledWith(
      DISCUSS_MEDIA_PROMPT_SETTING,
      {
        mediaId: "42",
        ownerScope,
        title: "Saved PDF",
        mode: "rag_media",
        content: "Summarize this source."
      }
    )
    expect(
      useMilestoneStore.getState().scopedMilestones[ownerScope]?.first_chat
    ).toBeUndefined()
    expect(
      useMilestoneStore.getState().scopedMilestones[ownerScope]?.first_ingest
    ).toBeTypeOf("number")
  })

  it("does not navigate a saved Home source handoff after the account changes", async () => {
    routeMocks.firstRunState.current = createCompletedFirstRunState()
    let finishSaving!: () => void
    routeMocks.setSetting.mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          finishSaving = resolve
        })
    )
    seedQuickIngestSession({
      lifecycle: "completed",
      openDetail: { source: "first_source_milestone", ownerScope },
      resultSummary: {
        ...createEmptyQuickIngestSession().resultSummary,
        status: "success",
        firstMediaId: "42"
      }
    })
    const { default: OptionIndex } = await import("../option-index")
    render(
      <MemoryRouter>
        <OptionIndex />
        <RouteProbe />
      </MemoryRouter>
    )
    fireEvent.click(
      await screen.findByRole("button", { name: /summarize this source/i })
    )
    await act(async () => {
      routeMocks.tldwConfig.current = {
        ...routeMocks.tldwConfig.current,
        serverUrl: "http://another-server:8000"
      }
      window.dispatchEvent(new Event("tldw:config-updated"))
    })
    await act(async () => {
      finishSaving()
    })
    expect(screen.getByTestId("route")).toHaveTextContent(/^\/$/)
  })

  it("retries first-source ingest with the persisted source kind after reload", async () => {
    routeMocks.firstRunState.current = createCompletedFirstRunState()
    seedQuickIngestSession({
      lifecycle: "completed",
      openDetail: {
        source: "first_source_milestone",
        ownerScope,
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

  it("keeps the source on Home with an actionable error if handoff cannot be saved", async () => {
    routeMocks.firstRunState.current = createCompletedFirstRunState()
    routeMocks.setSetting.mockRejectedValue(new Error("Storage unavailable"))
    seedQuickIngestSession({
      lifecycle: "completed",
      openDetail: {
        source: "first_source_milestone",
        ownerScope,
        firstSource: true
      },
      resultSummary: {
        status: "success",
        firstMediaId: "42",
        primarySourceLabel: "Notes",
        attemptedAt: 1,
        completedAt: 2,
        totalCount: 1,
        successCount: 1,
        failedCount: 0,
        cancelledCount: 0,
        errorMessage: null
      }
    })
    const { default: OptionIndex } = await import("../option-index")
    render(
      <MemoryRouter>
        <OptionIndex />
        <RouteProbe />
      </MemoryRouter>
    )
    fireEvent.click(
      await screen.findByRole("button", { name: /summarize this source/i })
    )
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Please try again"
    )
    expect(screen.getByTestId("route")).toHaveTextContent(/^\/$/)
  })

  it.each([
    {
      lifecycle: "processing",
      status: "success",
      source: "first_source_milestone"
    },
    {
      lifecycle: "completed",
      status: "error",
      source: "first_source_milestone"
    },
    { lifecycle: "completed", status: "success", source: "manual" }
  ] as const)(
    "does not credit ingestion from $source with $lifecycle/$status",
    async ({ lifecycle, status, source }) => {
      routeMocks.firstRunState.current = createCompletedFirstRunState()
      seedQuickIngestSession({
        lifecycle,
        openDetail: { source, ownerScope },
        resultSummary: {
          status,
          firstMediaId: "42",
          primarySourceLabel: "Notes",
          attemptedAt: 1,
          completedAt: 2,
          totalCount: 1,
          successCount: status === "success" ? 1 : 0,
          failedCount: status === "error" ? 1 : 0,
          cancelledCount: 0,
          errorMessage: null
        }
      })
      const { default: OptionIndex } = await import("../option-index")
      render(
        <MemoryRouter>
          <OptionIndex />
        </MemoryRouter>
      )
      await screen.findByRole("heading", { name: /add your first source/i })
      expect(
        useMilestoneStore.getState().scopedMilestones[ownerScope]?.first_ingest
      ).toBeUndefined()
    }
  )

  it.each([undefined, "another-account"])(
    "does not offer or credit a source with owner %s",
    async (sourceOwner) => {
      routeMocks.firstRunState.current = createCompletedFirstRunState()
      seedQuickIngestSession({
        lifecycle: "completed",
        openDetail: {
          source: "first_source_milestone",
          ownerScope: sourceOwner
        },
        resultSummary: {
          ...createEmptyQuickIngestSession().resultSummary,
          status: "success",
          firstMediaId: "42",
          primarySourceLabel: "Private source"
        }
      })
      const { default: OptionIndex } = await import("../option-index")
      render(
        <MemoryRouter>
          <OptionIndex />
        </MemoryRouter>
      )
      await screen.findByRole("heading", { name: /add your first source/i })
      expect(screen.queryByText("Private source")).not.toBeInTheDocument()
      expect(
        screen.queryByRole("button", { name: /summarize this source/i })
      ).not.toBeInTheDocument()
      expect(
        useMilestoneStore.getState().scopedMilestones[ownerScope]?.first_ingest
      ).toBeUndefined()
      expect(
        useMilestoneStore.getState().completedMilestones.first_chat
      ).toBeUndefined()
    }
  )

  it("removes a source offer on identity switch and restores it on same-account reload", async () => {
    routeMocks.firstRunState.current = createCompletedFirstRunState()
    seedQuickIngestSession({
      lifecycle: "completed",
      openDetail: { source: "first_source_milestone", ownerScope },
      resultSummary: {
        ...createEmptyQuickIngestSession().resultSummary,
        status: "success",
        firstMediaId: "42",
        primarySourceLabel: "Private source"
      }
    })
    const { default: OptionIndex } = await import("../option-index")
    const view = render(
      <MemoryRouter>
        <OptionIndex />
      </MemoryRouter>
    )
    await screen.findByRole("button", { name: /summarize this source/i })
    const originalConfig = routeMocks.tldwConfig.current
    routeMocks.tldwConfig.current = {
      ...originalConfig,
      serverUrl: "http://another-server:8000"
    }
    await act(async () => {
      window.dispatchEvent(new Event("tldw:config-updated"))
    })
    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: /summarize this source/i })
      ).not.toBeInTheDocument()
    )
    expect(screen.queryByText("Private source")).not.toBeInTheDocument()
    view.unmount()
    routeMocks.tldwConfig.current = originalConfig
    render(
      <MemoryRouter>
        <OptionIndex />
      </MemoryRouter>
    )
    await screen.findByRole("button", { name: /summarize this source/i })
    expect(
      useMilestoneStore.getState().scopedMilestones[ownerScope]?.first_ingest
    ).toBeTypeOf("number")
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
    expect(
      screen.queryByText(/processing your source/i)
    ).not.toBeInTheDocument()
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
