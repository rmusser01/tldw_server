// @vitest-environment jsdom
import React from "react"
import { MemoryRouter } from "react-router-dom"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

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
    routeMocks.firstRunState.current = {
      status: "completed",
      completed_steps: ["first_chat"],
      skipped_steps: [],
      step_data: {},
      acknowledged_steps: ["first_chat"],
      first_chat: { completed: true }
    }
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
      { source: "first_source_milestone" },
      { focusTrigger: true }
    )
  })

  it("shows inline API key recovery when setup is complete but media auth is missing", async () => {
    routeMocks.firstRunState.current = {
      status: "completed",
      completed_steps: ["first_chat"],
      skipped_steps: [],
      step_data: {},
      acknowledged_steps: ["first_chat"],
      first_chat: { completed: true }
    }
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
    routeMocks.firstRunState.current = {
      status: "completed",
      completed_steps: ["first_chat"],
      skipped_steps: [],
      step_data: {},
      acknowledged_steps: ["first_chat"],
      first_chat: { completed: true }
    }
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
    routeMocks.firstRunState.current = {
      status: "completed",
      completed_steps: ["first_chat"],
      skipped_steps: [],
      step_data: {},
      acknowledged_steps: ["first_chat"],
      first_chat: { completed: true }
    }
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
