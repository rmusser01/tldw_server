// @vitest-environment jsdom

import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"

const mocks = vi.hoisted(() => ({
  fetchCompanionHomeSnapshot: vi.fn(),
  fetchPersonalizationProfile: vi.fn(),
  updatePersonalizationOptIn: vi.fn(),
  loadCompanionHomeLayout: vi.fn(),
  saveCompanionHomeLayout: vi.fn(),
  capabilitiesState: {
    capabilities: { hasPersonalization: true, hasPersona: true },
    loading: false
  } as {
    capabilities: { hasPersonalization: boolean; hasPersona?: boolean } | null
    loading: boolean
  }
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => mocks.capabilitiesState
}))

vi.mock("@/design-system", async () => {
  const actual = await vi.importActual<typeof import("@/design-system")>(
    "@/design-system"
  )

  return {
    ...actual,
    getDesignSystemState: (
      key: Parameters<typeof actual.getDesignSystemState>[0]
    ) => {
      const state = actual.getDesignSystemState(key)
      if (key === "setup_required" && state) {
        return {
          ...state,
          label: "Registry setup required"
        }
      }
      return state
    }
  }
})

vi.mock("@/services/companion-home", () => ({
  fetchCompanionHomeSnapshot: (...args: unknown[]) =>
    mocks.fetchCompanionHomeSnapshot(...args)
}))

vi.mock("@/services/companion", () => ({
  fetchPersonalizationProfile: (...args: unknown[]) =>
    mocks.fetchPersonalizationProfile(...args),
  updatePersonalizationOptIn: (...args: unknown[]) =>
    mocks.updatePersonalizationOptIn(...args)
}))

vi.mock("@/store/companion-home-layout", async () => {
  const actual = await vi.importActual<typeof import("@/store/companion-home-layout")>(
    "@/store/companion-home-layout"
  )

  return {
    ...actual,
    DEFAULT_COMPANION_HOME_LAYOUT: [
      {
        id: "inbox-preview",
        title: "Inbox Preview",
        kind: "system",
        fixed: true,
        visible: true
      },
      {
        id: "needs-attention",
        title: "Needs Attention",
        kind: "system",
        fixed: true,
        visible: true
      },
      {
        id: "resume-work",
        title: "Resume Work",
        kind: "core",
        fixed: false,
        visible: true
      },
      {
        id: "goals-focus",
        title: "Goals / Focus",
        kind: "core",
        fixed: false,
        visible: true
      },
      {
        id: "recent-activity",
        title: "Recent Activity",
        kind: "core",
        fixed: false,
        visible: true
      },
      {
        id: "reading-queue",
        title: "Reading Queue",
        kind: "core",
        fixed: false,
        visible: true
      }
    ],
    loadCompanionHomeLayout: (...args: unknown[]) =>
      mocks.loadCompanionHomeLayout(...args),
    saveCompanionHomeLayout: (...args: unknown[]) =>
      mocks.saveCompanionHomeLayout(...args)
  }
})

import { CompanionHomePage } from "../CompanionHomePage"

const createDeferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return {
    promise,
    resolve,
    reject
  }
}

const buildSnapshot = (overrides: Record<string, unknown> = {}) => ({
  surface: "options",
  inbox: [
    {
      id: "inbox-1",
      entityId: "reflection-1",
      entityType: "notification",
      source: "canonical_inbox",
      title: "Unread reflection",
      summary: "This is still waiting in your inbox.",
      updatedAt: "2026-03-20T10:00:00Z",
      href: "/companion"
    }
  ],
  needsAttention: [
    {
      id: "attention-1",
      entityId: "goal-1",
      entityType: "goal",
      source: "goal",
      title: "Finish queue review",
      summary: "Progress needs an explicit update.",
      updatedAt: "2026-03-18T10:00:00Z",
      href: "/companion"
    }
  ],
  resumeWork: [
    {
      id: "resume-1",
      entityId: "note-1",
      entityType: "note",
      source: "note",
      title: "Draft outline",
      summary: "Turn the queue review into a checklist.",
      updatedAt: "2026-03-19T12:00:00Z",
      href: "/notes"
    }
  ],
  goalsFocus: [
    {
      id: "goal-1",
      entityId: "goal-1",
      entityType: "goal",
      source: "goal",
      title: "Finish queue review",
      summary: "0 / 3 complete",
      updatedAt: "2026-03-19T09:00:00Z",
      href: "/companion"
    }
  ],
  recentActivity: [
    {
      id: "activity-1",
      entityId: "reading-1",
      entityType: "reading_item",
      source: "reading",
      title: "Queue article",
      summary: "Captured while researching.",
      updatedAt: "2026-03-20T08:00:00Z",
      href: "/collections"
    }
  ],
  readingQueue: [
    {
      id: "reading-1",
      entityId: "reading-1",
      entityType: "reading_item",
      source: "reading",
      title: "Queue article",
      summary: "Saved for later reading.",
      updatedAt: "2026-03-20T07:00:00Z",
      href: "/collections"
    }
  ],
  degradedSources: [],
  summary: {
    activityCount: 1,
    inboxCount: 1,
    needsAttentionCount: 1,
    resumeWorkCount: 1
  },
  ...overrides
})

const renderPage = (
  props: Partial<React.ComponentProps<typeof CompanionHomePage>> = {}
) =>
  render(
    <MemoryRouter>
      <CompanionHomePage surface="options" {...props} />
    </MemoryRouter>
  )

describe("CompanionHomePage", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.capabilitiesState.capabilities = { hasPersonalization: true, hasPersona: true }
    mocks.capabilitiesState.loading = false
    mocks.fetchCompanionHomeSnapshot.mockResolvedValue(buildSnapshot())
    mocks.fetchPersonalizationProfile.mockResolvedValue({
      enabled: true,
      updated_at: "2026-03-20T10:00:00Z"
    })
    mocks.updatePersonalizationOptIn.mockResolvedValue({
      enabled: true,
      updated_at: "2026-03-20T10:05:00Z"
    })
    mocks.loadCompanionHomeLayout.mockResolvedValue([
      {
        id: "inbox-preview",
        title: "Inbox Preview",
        kind: "system",
        fixed: true,
        visible: true
      },
      {
        id: "needs-attention",
        title: "Needs Attention",
        kind: "system",
        fixed: true,
        visible: true
      },
      {
        id: "resume-work",
        title: "Resume Work",
        kind: "core",
        fixed: false,
        visible: true
      },
      {
        id: "goals-focus",
        title: "Goals / Focus",
        kind: "core",
        fixed: false,
        visible: true
      },
      {
        id: "recent-activity",
        title: "Recent Activity",
        kind: "core",
        fixed: false,
        visible: true
      },
      {
        id: "reading-queue",
        title: "Reading Queue",
        kind: "core",
        fixed: false,
        visible: true
      }
    ])
    mocks.saveCompanionHomeLayout.mockResolvedValue(undefined)
  })

  it("shows a setup band and inbox card instead of a dead-end when personalization is unavailable", async () => {
    mocks.capabilitiesState.capabilities = { hasPersonalization: false, hasPersona: false }

    renderPage()

    expect(await screen.findByText("Companion setup required")).toBeInTheDocument()
    expect(screen.getByRole("heading", { name: "Inbox Preview" })).toBeInTheDocument()
    expect(screen.getAllByText("Registry setup required").length).toBeGreaterThan(0)
    expect(
      screen.getByText(/Companion inbox items unlock once personalization is available for this workspace\./)
    ).toBeInTheDocument()
    expect(
      screen.queryByText("Companion unavailable")
    ).not.toBeInTheDocument()
    expect(mocks.fetchCompanionHomeSnapshot).not.toHaveBeenCalled()
    expect(mocks.fetchPersonalizationProfile).not.toHaveBeenCalled()
  })

  it("renders the default core dashboard cards", async () => {
    renderPage()

    await waitFor(() => {
      expect(mocks.fetchCompanionHomeSnapshot).toHaveBeenCalledWith("options")
    })

    expect(screen.getByRole("heading", { name: "Inbox Preview" })).toBeInTheDocument()
    expect(screen.getByRole("heading", { name: "Needs Attention" })).toBeInTheDocument()
    expect(screen.getByRole("heading", { name: "Resume Work" })).toBeInTheDocument()
    expect(screen.getByRole("heading", { name: "Goals / Focus" })).toBeInTheDocument()
    expect(screen.getByRole("heading", { name: "Recent Activity" })).toBeInTheDocument()
    expect(screen.getByRole("heading", { name: "Reading Queue" })).toBeInTheDocument()

    const summary = screen.getByTestId("companion-home-summary")
    expect(within(summary).getByText("Inbox")).toBeInTheDocument()
    expect(within(summary).getByText("Goals")).toBeInTheDocument()
    expect(within(summary).getByText("Reading")).toBeInTheDocument()
    expect(within(summary).getByText("Resume")).toBeInTheDocument()
  })

  it("does not flash the default core layout before the persisted layout resolves", async () => {
    const deferred = createDeferred<
      Awaited<ReturnType<typeof mocks.loadCompanionHomeLayout>>
    >()
    mocks.loadCompanionHomeLayout.mockReturnValueOnce(deferred.promise)

    renderPage()

    await screen.findByRole("button", { name: "Customize Home" })
    expect(screen.getByRole("heading", { name: "Inbox Preview" })).toBeInTheDocument()
    expect(screen.getByRole("heading", { name: "Needs Attention" })).toBeInTheDocument()
    expect(screen.queryByRole("heading", { name: "Resume Work" })).not.toBeInTheDocument()
    expect(screen.queryByRole("heading", { name: "Goals / Focus" })).not.toBeInTheDocument()
    expect(screen.queryByRole("heading", { name: "Recent Activity" })).not.toBeInTheDocument()
    expect(screen.queryByRole("heading", { name: "Reading Queue" })).not.toBeInTheDocument()

    deferred.resolve([
      {
        id: "inbox-preview",
        title: "Inbox Preview",
        kind: "system",
        fixed: true,
        visible: true
      },
      {
        id: "needs-attention",
        title: "Needs Attention",
        kind: "system",
        fixed: true,
        visible: true
      },
      {
        id: "recent-activity",
        title: "Recent Activity",
        kind: "core",
        fixed: false,
        visible: true
      },
      {
        id: "resume-work",
        title: "Resume Work",
        kind: "core",
        fixed: false,
        visible: true
      },
      {
        id: "goals-focus",
        title: "Goals / Focus",
        kind: "core",
        fixed: false,
        visible: false
      },
      {
        id: "reading-queue",
        title: "Reading Queue",
        kind: "core",
        fixed: false,
        visible: true
      }
    ])

    const recent = await screen.findByRole("heading", { name: "Recent Activity" })
    const resume = screen.getByRole("heading", { name: "Resume Work" })
    expect(screen.queryByRole("heading", { name: "Goals / Focus" })).not.toBeInTheDocument()
    expect(recent.compareDocumentPosition(resume) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it("renders core cards according to the persisted layout override", async () => {
    mocks.loadCompanionHomeLayout.mockResolvedValueOnce([
      {
        id: "inbox-preview",
        title: "Inbox Preview",
        kind: "system",
        fixed: true,
        visible: true
      },
      {
        id: "needs-attention",
        title: "Needs Attention",
        kind: "system",
        fixed: true,
        visible: true
      },
      {
        id: "recent-activity",
        title: "Recent Activity",
        kind: "core",
        fixed: false,
        visible: true
      },
      {
        id: "resume-work",
        title: "Resume Work",
        kind: "core",
        fixed: false,
        visible: true
      },
      {
        id: "goals-focus",
        title: "Goals / Focus",
        kind: "core",
        fixed: false,
        visible: false
      },
      {
        id: "reading-queue",
        title: "Reading Queue",
        kind: "core",
        fixed: false,
        visible: true
      }
    ])

    renderPage()

    await waitFor(() => {
      expect(mocks.loadCompanionHomeLayout).toHaveBeenCalledWith("options")
    })

    expect(screen.queryByRole("heading", { name: "Goals / Focus" })).not.toBeInTheDocument()

    const recent = screen.getByRole("heading", { name: "Recent Activity" })
    const resume = screen.getByRole("heading", { name: "Resume Work" })
    expect(recent.compareDocumentPosition(resume) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it("opens the customize drawer and persists layout changes", async () => {
    renderPage()

    await screen.findByRole("button", { name: "Customize Home" })
    fireEvent.click(screen.getByRole("button", { name: "Customize Home" }))

    expect(screen.getByRole("dialog", { name: "Customize Home" })).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: /hide goals \/ focus/i }))

    await waitFor(() => {
      expect(mocks.saveCompanionHomeLayout).toHaveBeenCalledWith(
        "options",
        expect.arrayContaining([
          expect.objectContaining({
            id: "goals-focus",
            visible: false
          })
        ])
      )
    })
    expect(screen.queryByRole("heading", { name: "Goals / Focus" })).not.toBeInTheDocument()
  })

  it("does not let a slow persisted load clobber an earlier user layout change", async () => {
    const deferred = createDeferred<
      Awaited<ReturnType<typeof mocks.loadCompanionHomeLayout>>
    >()
    mocks.loadCompanionHomeLayout.mockReturnValueOnce(deferred.promise)

    renderPage()

    await screen.findByRole("button", { name: "Customize Home" })
    fireEvent.click(screen.getByRole("button", { name: "Customize Home" }))
    fireEvent.click(screen.getByRole("button", { name: /hide goals \/ focus/i }))

    await waitFor(() => {
      expect(mocks.saveCompanionHomeLayout).toHaveBeenCalledWith(
        "options",
        expect.arrayContaining([
          expect.objectContaining({
            id: "goals-focus",
            visible: false
          })
        ])
      )
    })
    expect(screen.queryByRole("heading", { name: "Goals / Focus" })).not.toBeInTheDocument()

    deferred.resolve([
      {
        id: "inbox-preview",
        title: "Inbox Preview",
        kind: "system",
        fixed: true,
        visible: true
      },
      {
        id: "needs-attention",
        title: "Needs Attention",
        kind: "system",
        fixed: true,
        visible: true
      },
      {
        id: "resume-work",
        title: "Resume Work",
        kind: "core",
        fixed: false,
        visible: true
      },
      {
        id: "goals-focus",
        title: "Goals / Focus",
        kind: "core",
        fixed: false,
        visible: true
      },
      {
        id: "recent-activity",
        title: "Recent Activity",
        kind: "core",
        fixed: false,
        visible: true
      },
      {
        id: "reading-queue",
        title: "Reading Queue",
        kind: "core",
        fixed: false,
        visible: true
      }
    ])

    await waitFor(() => {
      expect(screen.queryByRole("heading", { name: "Goals / Focus" })).not.toBeInTheDocument()
    })
  })

  it("shows scoped degraded states for mixed cards when upstream sources are unavailable", async () => {
    mocks.fetchCompanionHomeSnapshot.mockResolvedValueOnce(
      buildSnapshot({
        needsAttention: [],
        resumeWork: [],
        degradedSources: ["workspace", "reading"]
      })
    )

    renderPage()

    expect(
      await screen.findByText(
        "Needs-attention signals are limited until companion and reading sources come back."
      )
    ).toBeInTheDocument()
    expect(
      screen.getByText(
        "Resume suggestions are limited until companion, reading, and note sources are available."
      )
    ).toBeInTheDocument()
  })

  it("notifies the shell after enabling personalization successfully", async () => {
    mocks.fetchPersonalizationProfile.mockResolvedValueOnce({
      enabled: false,
      updated_at: "2026-03-20T10:00:00Z"
    })
    const onPersonalizationEnabled = vi.fn()

    renderPage({ onPersonalizationEnabled })

    fireEvent.click(await screen.findByRole("button", { name: "Enable Companion" }))

    await waitFor(() => {
      expect(mocks.updatePersonalizationOptIn).toHaveBeenCalledWith(true)
    })
    await waitFor(() => {
      expect(onPersonalizationEnabled).toHaveBeenCalledTimes(1)
    })
  })
})
