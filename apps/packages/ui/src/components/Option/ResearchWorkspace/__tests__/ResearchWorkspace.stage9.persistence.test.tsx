import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import {
  WORKSPACE_CONFLICT_NOTICE_THROTTLE_MS,
  WORKSPACE_STORAGE_KEY,
  WORKSPACE_STORAGE_QUOTA_EVENT
} from "@/store/workspace-events"
import { ResearchWorkspace } from "../index"

const {
  mockGetMediaDetails,
  useWorkspaceStoreMock,
  workspaceStorage,
  workspaceStorageItems
} = vi.hoisted(() => {
  const storageItems = new Map<string, string>()
  const storage = {
    getItem: vi.fn((key: string) => storageItems.get(key) ?? null),
    setItem: vi.fn(async (key: string, value: string) => {
      storageItems.set(key, value)
    }),
    removeItem: vi.fn(async (key: string) => {
      storageItems.delete(key)
    })
  }

  return {
    mockGetMediaDetails: vi.fn(),
    useWorkspaceStoreMock: vi.fn(),
    workspaceStorage: storage,
    workspaceStorageItems: storageItems
  }
})

const testState = {
  isMobile: false,
  storeHydrated: true,
  leftPaneCollapsed: false,
  rightPaneCollapsed: false,
  workspaceId: "workspace-1",
  workspaceName: "Workspace One",
  initializeWorkspace: vi.fn(),
  duplicateWorkspace: vi.fn(),
  addSources: vi.fn(),
  setSelectedSourceIds: vi.fn(),
  captureToCurrentNote: vi.fn(),
  selectedSourceIds: [] as string[],
  generatedArtifacts: [] as Array<{ id: string }>,
  setLeftPaneCollapsed: vi.fn(),
  setRightPaneCollapsed: vi.fn(),
  focusSourceById: vi.fn(() => true),
  focusChatMessageById: vi.fn(() => true),
  focusWorkspaceNote: vi.fn(),
  setSourceStatusByMediaId: vi.fn(),
  sources: [] as Array<{
    id: string
    mediaId: number
    title: string
    type: "pdf" | "video" | "audio" | "website" | "document" | "text"
    addedAt: Date
    status?: "processing" | "ready" | "error"
    url?: string
  }>,
  workspaceChatSessions: {} as Record<string, { messages: any[] }>,
  currentNote: {
    id: 9,
    title: "",
    content: "",
    keywords: [] as string[],
    isDirty: false
  },
  workspaceBanner: {
    title: "",
    subtitle: "",
    image: null as
      | {
          dataUrl: string
          mimeType: "image/jpeg" | "image/png" | "image/webp"
          width: number
          height: number
          bytes: number
          updatedAt: Date
        }
      | null
  }
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          },
      interpolationOptions?: Record<string, unknown>
    ) => {
      if (typeof defaultValueOrOptions === "string") {
        if (!interpolationOptions) return defaultValueOrOptions
        return defaultValueOrOptions.replace(
          /\{\{(\w+)\}\}/g,
          (_match, token) => String(interpolationOptions[token] ?? "")
        )
      }
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => testState.isMobile
}))

vi.mock("@/store/workspace", () => ({
  useWorkspaceStore: useWorkspaceStoreMock,
  createWorkspaceStorage: () => workspaceStorage
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getMediaDetails: mockGetMediaDetails
  }
}))

vi.mock("@/utils/research-workspace-prefill", () => ({
  consumeResearchWorkspacePrefill: vi.fn().mockResolvedValue(null),
  buildKnowledgeQaSeedNote: vi.fn().mockReturnValue("")
}))

vi.mock("../WorkspaceHeader", () => ({
  WorkspaceHeader: () => <div data-testid="workspace-header" />
}))

vi.mock("../SourcesPane", () => ({
  SourcesPane: () => <div data-testid="workspace-sources-pane">Sources</div>
}))

vi.mock("../ChatPane", () => ({
  ChatPane: () => <div data-testid="workspace-chat-pane">Chat</div>
}))

vi.mock("../StudioPane", () => ({
  StudioPane: () => <div data-testid="workspace-studio-pane">Studio</div>
}))

vi.mock("../WorkspaceStatusBar", () => ({
  WorkspaceStatusBar: () => <div data-testid="workspace-status-bar" />
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

describe("ResearchWorkspace stage 9 persistence resilience", () => {
  const originalMatchMedia = window.matchMedia

  beforeAll(() => {
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: vi.fn().mockImplementation((query: string) => ({
        matches: query.includes("min-width: 1024px"),
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn()
      }))
    })
  })

  afterAll(() => {
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: originalMatchMedia
    })
  })

  beforeEach(() => {
    vi.clearAllMocks()
    workspaceStorageItems.clear()
    useWorkspaceStoreMock.mockImplementation(
      (selector: (state: typeof testState) => unknown) => selector(testState)
    )
    testState.storeHydrated = true
    testState.workspaceId = "workspace-1"
    testState.sources = []
    testState.selectedSourceIds = []
    testState.workspaceChatSessions = {}
    testState.currentNote = {
      id: 9,
      title: "",
      content: "",
      keywords: [],
      isDirty: false
    }
    testState.workspaceBanner = {
      title: "",
      subtitle: "",
      image: null
    }
    mockGetMediaDetails.mockResolvedValue({})
  })

  const dispatchWorkspaceStorageUpdate = (
    oldValue = '{"version":1}',
    newValue = '{"version":2}'
  ) => {
    const event = new Event("storage") as StorageEvent
    Object.defineProperties(event, {
      key: { value: WORKSPACE_STORAGE_KEY },
      oldValue: { value: oldValue },
      newValue: { value: newValue },
      storageArea: { value: window.localStorage }
    })
    window.dispatchEvent(event)
  }

  it("shows a cross-tab sync banner and conflict actions on storage updates", async () => {
    render(<ResearchWorkspace />)

    dispatchWorkspaceStorageUpdate()

    expect(
      await screen.findByTestId("workspace-storage-sync-banner")
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Reload from other tab" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save as new workspace" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Keep this version" })).toBeInTheDocument()
    expect(
      screen.getByText("Recommended: Reload to get the latest version.")
    ).toBeInTheDocument()
    expect(
      screen.getByText(
        "Keep this version ignores the update. Save as new workspace copies your current state."
      )
    ).toBeInTheDocument()
  })

  it("throttles repeated storage conflict prompts", async () => {
    let now = 1708257600000
    const nowSpy = vi.spyOn(Date, "now").mockImplementation(() => now)

    render(<ResearchWorkspace />)
    dispatchWorkspaceStorageUpdate('{"version":0}', '{"version":1}')
    expect(
      await screen.findByTestId("workspace-storage-sync-banner")
    ).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Keep this version" }))
    expect(
      screen.queryByTestId("workspace-storage-sync-banner")
    ).not.toBeInTheDocument()

    dispatchWorkspaceStorageUpdate('{"version":1}', '{"version":2}')
    expect(
      screen.queryByTestId("workspace-storage-sync-banner")
    ).not.toBeInTheDocument()

    now += WORKSPACE_CONFLICT_NOTICE_THROTTLE_MS + 1000
    dispatchWorkspaceStorageUpdate('{"version":2}', '{"version":3}')
    expect(
      await screen.findByTestId("workspace-storage-sync-banner")
    ).toBeInTheDocument()

    nowSpy.mockRestore()
  })

  it("shows changed field hints and can fork workspace from conflict banner", async () => {
    render(<ResearchWorkspace />)

    dispatchWorkspaceStorageUpdate(
      JSON.stringify({
        state: {
          sources: [{ id: "source-1" }],
          workspaceChatSessions: {}
        }
      }),
      JSON.stringify({
        state: {
          sources: [{ id: "source-1" }, { id: "source-2" }],
          workspaceChatSessions: {
            "workspace-1": { messages: [{ id: "message-1" }] }
          }
        }
      })
    )

    expect(
      await screen.findByText("Changed fields: sources, chat history")
    ).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Save as new workspace" }))
    expect(testState.duplicateWorkspace).toHaveBeenCalledWith("workspace-1")
    await waitFor(() => {
      expect(
        screen.queryByTestId("workspace-storage-sync-banner")
      ).not.toBeInTheDocument()
    })
  })

  it("tracks workspaceBanner as cross-tab conflict field", async () => {
    render(<ResearchWorkspace />)

    dispatchWorkspaceStorageUpdate(
      JSON.stringify({
        state: {
          workspaceBanner: {
            title: "Old",
            subtitle: "",
            image: null
          }
        }
      }),
      JSON.stringify({
        state: {
          workspaceBanner: {
            title: "New",
            subtitle: "Updated",
            image: null
          }
        }
      })
    )

    expect(
      await screen.findByText("Changed fields: workspace banner")
    ).toBeInTheDocument()
  })

  it("shows storage quota warning from persistence quota event", async () => {
    render(<ResearchWorkspace />)

    window.dispatchEvent(
      new CustomEvent(WORKSPACE_STORAGE_QUOTA_EVENT, {
        detail: {
          key: WORKSPACE_STORAGE_KEY,
          reason: "Quota exceeded"
        }
      })
    )

    const quotaBanner = await screen.findByTestId("workspace-storage-quota-banner")
    expect(quotaBanner).toBeInTheDocument()

    fireEvent.click(within(quotaBanner).getByRole("button", { name: "Dismiss" }))
    await waitFor(() => {
      expect(
        screen.queryByTestId("workspace-storage-quota-banner")
      ).not.toBeInTheDocument()
    })
  })

  it("renders configured workspace banner title/subtitle", async () => {
    testState.workspaceBanner = {
      title: "Alpha Banner",
      subtitle: "Alpha subtitle",
      image: {
        dataUrl: "data:image/webp;base64,alpha-banner",
        mimeType: "image/webp",
        width: 1400,
        height: 420,
        bytes: 18000,
        updatedAt: new Date("2026-02-25T09:00:00.000Z")
      }
    }

    render(<ResearchWorkspace />)

    expect(await screen.findByTestId("workspace-banner")).toBeInTheDocument()
    expect(screen.getByTestId("workspace-banner-title")).toHaveTextContent(
      "Alpha Banner"
    )
    expect(screen.getByTestId("workspace-banner-subtitle")).toHaveTextContent(
      "Alpha subtitle"
    )
  })

  it("falls back gracefully when banner image is absent", async () => {
    testState.workspaceBanner = {
      title: "Imageless Banner",
      subtitle: "No image present",
      image: null
    }

    render(<ResearchWorkspace />)

    const banner = await screen.findByTestId("workspace-banner")
    expect(banner).toBeInTheDocument()
    // Banner without image defaults to collapsed ribbon — title visible, subtitle hidden
    expect(screen.getByTestId("workspace-banner-title")).toHaveTextContent(
      "Imageless Banner"
    )
    expect(banner.getAttribute("style") || "").not.toContain("url(")

    // Expand by clicking the collapsed ribbon
    fireEvent.click(banner)
    expect(screen.getByTestId("workspace-banner-subtitle")).toHaveTextContent(
      "No image present"
    )
  })
})
