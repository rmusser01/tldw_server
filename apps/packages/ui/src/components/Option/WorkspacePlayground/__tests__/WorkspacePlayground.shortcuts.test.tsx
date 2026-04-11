import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { WorkspacePlayground } from "../index"

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
  }>,
  currentNote: {
    id: undefined as number | undefined,
    title: "",
    content: "",
    keywords: [] as string[],
    isDirty: false,
  },
  workspaceChatSessions: {} as Record<string, { messages: unknown[] }>,
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
      if (typeof defaultValueOrOptions === "string")
        return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue)
        return defaultValueOrOptions.defaultValue
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
}))

vi.mock("@/store/tutorials", () => ({
  useTutorialStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({ startTutorial: vi.fn() }),
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

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

const ONBOARDING_KEY = "tldw:workspace-playground:onboarding-dismissed:v1"

describe("WorkspacePlayground keyboard shortcuts modal", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Dismiss onboarding so it doesn't interfere with tests
    window.localStorage.setItem(ONBOARDING_KEY, "1")
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

  it("opens the shortcuts modal when '?' key is pressed", async () => {
    render(<WorkspacePlayground />)

    fireEvent.keyDown(window, { key: "?" })

    await waitFor(() => {
      expect(screen.getByText("Keyboard Shortcuts")).toBeInTheDocument()
    })
  })

  it("does NOT open the modal when '?' is pressed while an input is focused", () => {
    render(
      <>
        <input data-testid="external-input" />
        <WorkspacePlayground />
      </>
    )

    const input = screen.getByTestId("external-input")
    input.focus()
    fireEvent.keyDown(input, { key: "?" })

    expect(screen.queryByText("Keyboard Shortcuts")).not.toBeInTheDocument()
  })

  it("does NOT open the modal when '?' is pressed while a textarea is focused", () => {
    render(
      <>
        <textarea data-testid="external-textarea" />
        <WorkspacePlayground />
      </>
    )

    const textarea = screen.getByTestId("external-textarea")
    textarea.focus()
    fireEvent.keyDown(textarea, { key: "?" })

    expect(screen.queryByText("Keyboard Shortcuts")).not.toBeInTheDocument()
  })

  it("shows all expected keyboard shortcuts in the modal", async () => {
    render(<WorkspacePlayground />)

    fireEvent.keyDown(window, { key: "?" })

    await waitFor(() => {
      expect(screen.getByText("Keyboard Shortcuts")).toBeInTheDocument()
    })

    expect(screen.getByText("Focus sources pane")).toBeInTheDocument()
    expect(screen.getByText("Focus chat pane")).toBeInTheDocument()
    expect(screen.getByText("Focus studio pane")).toBeInTheDocument()
    expect(screen.getByText("Global search")).toBeInTheDocument()
    expect(screen.getByText("New note")).toBeInTheDocument()
    expect(screen.getByText("New workspace")).toBeInTheDocument()
    expect(screen.getByText("Undo")).toBeInTheDocument()
    expect(screen.getByText("Show shortcuts")).toBeInTheDocument()
  })

  it("closes the modal when Escape is pressed", async () => {
    render(<WorkspacePlayground />)

    fireEvent.keyDown(window, { key: "?" })

    await waitFor(() => {
      expect(screen.getByText("Keyboard Shortcuts")).toBeInTheDocument()
    })

    // Antd Modal uses its own close mechanism via onCancel;
    // find and click the close button to trigger it
    const modal = screen.getByRole("dialog")
    const closeButton = modal.querySelector("button.ant-modal-close")
    if (closeButton) {
      fireEvent.click(closeButton)
    } else {
      // Fallback: press Escape on the document, which antd Modal intercepts
      fireEvent.keyDown(document, { key: "Escape" })
    }

    await waitFor(() => {
      // The modal should either be gone or in leave animation
      const dialog = screen.queryByRole("dialog")
      if (dialog) {
        expect(dialog).toHaveClass("ant-zoom-leave")
      } else {
        expect(dialog).toBeNull()
      }
    })
  })
})
