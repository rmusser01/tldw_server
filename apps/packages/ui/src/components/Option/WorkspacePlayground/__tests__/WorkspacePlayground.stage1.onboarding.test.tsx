import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { WorkspacePlayground } from "../index"

const ONBOARDING_KEY = "tldw:workspace-playground:onboarding-dismissed:v1"
const {
  onboardingStorageState,
  mockWorkspaceStorageGetItem,
  mockWorkspaceStorageSetItem,
} = vi.hoisted(() => ({
  onboardingStorageState: {
    value: undefined as string | undefined,
  },
  mockWorkspaceStorageGetItem: vi.fn(async (_key: string) => null as string | null),
  mockWorkspaceStorageSetItem: vi.fn(async (_key: string, _value: string) => undefined),
}))

const mockStartTutorial = vi.fn()

const testState = {
  isMobile: false,
  storeHydrated: true,
  leftPaneCollapsed: false,
  rightPaneCollapsed: false,
  workspaceId: "workspace-1",
  workspaceTag: "",
  initializeWorkspace: vi.fn(),
  createNewWorkspace: vi.fn(),
  addSources: vi.fn(),
  setSelectedSourceIds: vi.fn(),
  captureToCurrentNote: vi.fn(),
  clearCurrentNote: vi.fn(),
  loadNote: vi.fn(),
  selectedSourceIds: [] as string[],
  generatedArtifacts: [] as Array<{ id: string }>,
  setLeftPaneCollapsed: vi.fn(),
  setRightPaneCollapsed: vi.fn(),
  sources: [] as Array<{
    id: string
    mediaId: number
    title: string
    type: "pdf" | "video" | "audio" | "website" | "document" | "text"
    addedAt: Date
    status?: "processing" | "ready" | "error"
  }>,
  currentNote: {
    id: undefined as number | undefined,
    title: "",
    content: "",
    keywords: [] as string[],
    isDirty: false,
  },
  workspaceChatSessions: {} as Record<string, { messages: unknown[] }>,
  focusSourceById: vi.fn(() => true),
  focusChatMessageById: vi.fn(() => true),
  focusWorkspaceNote: vi.fn(),
  setSourceStatusByMediaId: vi.fn(),
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    },
  }),
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => testState.isMobile,
}))

vi.mock("@/store/workspace", () => ({
  useWorkspaceStore: (selector: (state: typeof testState) => unknown) =>
    selector(testState),
  createWorkspaceStorage: () => ({
    getItem: (key: string) => {
      mockWorkspaceStorageGetItem.mockImplementationOnce(async (requestedKey: string) =>
        requestedKey === ONBOARDING_KEY ? onboardingStorageState.value ?? null : null
      )
      return mockWorkspaceStorageGetItem(key)
    },
    setItem: (key: string, value: string) => {
      if (key === ONBOARDING_KEY) {
        onboardingStorageState.value = value
      }
      return mockWorkspaceStorageSetItem(key, value)
    },
    removeItem: vi.fn(),
  }),
}))

vi.mock("@/store/tutorials", () => ({
  useTutorialStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({ startTutorial: mockStartTutorial }),
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getMediaDetails: vi.fn().mockResolvedValue({}),
  },
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: vi.fn().mockResolvedValue([]),
}))

vi.mock("@/utils/workspace-playground-prefill", () => ({
  consumeWorkspacePlaygroundPrefill: vi.fn().mockResolvedValue(null),
  buildKnowledgeQaSeedNote: vi.fn().mockReturnValue(""),
}))

vi.mock("../WorkspaceHeader", () => ({
  WorkspaceHeader: () => <div data-testid="workspace-header" />,
}))

vi.mock("../SourcesPane", () => ({
  SourcesPane: () => <div data-testid="workspace-sources-pane">Sources</div>,
}))

vi.mock("../ChatPane", () => ({
  ChatPane: () => <div data-testid="workspace-chat-pane">Chat</div>,
}))

vi.mock("../StudioPane", () => ({
  StudioPane: () => <div data-testid="workspace-studio-pane">Studio</div>,
}))

vi.mock("../WorkspaceStatusBar", () => ({
  WorkspaceStatusBar: () => <div data-testid="workspace-status-bar" />,
}))

if (!(globalThis as { ResizeObserver?: unknown }).ResizeObserver) {
  ;(globalThis as { ResizeObserver?: unknown }).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

describe("WorkspacePlayground stage 1 onboarding walkthrough", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    onboardingStorageState.value = undefined
    testState.isMobile = false
    testState.storeHydrated = true
    testState.workspaceId = "workspace-1"
    testState.workspaceTag = ""
    testState.leftPaneCollapsed = false
    testState.rightPaneCollapsed = false
    testState.selectedSourceIds = []
    testState.generatedArtifacts = []
    testState.sources = []
    testState.currentNote = {
      id: undefined,
      title: "",
      content: "",
      keywords: [],
      isDirty: false,
    }
    testState.workspaceChatSessions = {}
  })

  it("shows an opt-in tour prompt for first-time users instead of auto-starting", async () => {
    render(<WorkspacePlayground />)

    await waitFor(() => {
      expect(screen.getByText("Start tour")).toBeInTheDocument()
    })
    expect(mockStartTutorial).not.toHaveBeenCalled()
  })

  it("starts the tour and persists dismissal when user clicks Start tour", async () => {
    const user = userEvent.setup()
    render(<WorkspacePlayground />)

    const startButton = await screen.findByText("Start tour")
    await user.click(startButton)

    expect(mockStartTutorial).toHaveBeenCalledWith("workspace-playground-basics")
    await waitFor(() => {
      expect(mockWorkspaceStorageSetItem).toHaveBeenCalledWith(ONBOARDING_KEY, "1")
    })
    expect(onboardingStorageState.value).toBe("1")
    expect(screen.queryByText("Start tour")).not.toBeInTheDocument()
  })

  it("persists dismissal without starting tour when user clicks Dismiss", async () => {
    const user = userEvent.setup()
    render(<WorkspacePlayground />)

    const dismissButton = await screen.findByText("Dismiss")
    await user.click(dismissButton)

    expect(mockStartTutorial).not.toHaveBeenCalled()
    await waitFor(() => {
      expect(mockWorkspaceStorageSetItem).toHaveBeenCalledWith(ONBOARDING_KEY, "1")
    })
    expect(onboardingStorageState.value).toBe("1")
    expect(screen.queryByText("Start tour")).not.toBeInTheDocument()
  })

  it("does not show tour prompt when already dismissed", () => {
    onboardingStorageState.value = "1"

    render(<WorkspacePlayground />)

    expect(screen.queryByText("Start tour")).not.toBeInTheDocument()
    expect(mockStartTutorial).not.toHaveBeenCalled()
  })

  it("does not show a blocking onboarding overlay", () => {
    render(<WorkspacePlayground />)

    expect(
      screen.queryByTestId("workspace-onboarding-overlay")
    ).not.toBeInTheDocument()
  })
})
