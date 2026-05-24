import { fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { ResearchWorkspace } from "../index"

const testState = {
  isMobile: false,
  storeHydrated: true,
  leftPaneCollapsed: false,
  rightPaneCollapsed: false,
  workspaceId: "workspace-1",
  initializeWorkspace: vi.fn(),
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
  }>,
  workspaceChatSessions: {} as Record<string, { messages: any[] }>,
  currentNote: {
    id: 7,
    title: "",
    content: "",
    keywords: [] as string[],
    isDirty: false
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
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => testState.isMobile
}))

vi.mock("@/store/workspace", () => ({
  useWorkspaceStore: (selector: (state: typeof testState) => unknown) =>
    selector(testState)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getMediaDetails: vi.fn().mockResolvedValue({})
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
  ChatPane: () => {
    throw new Error("chat pane crash")
  }
}))

vi.mock("../StudioPane", () => ({
  StudioPane: () => <div data-testid="workspace-studio-pane">Studio</div>
}))

vi.mock("../WorkspaceStatusBar", () => ({
  WorkspaceStatusBar: () => <div data-testid="workspace-status-bar" />
}))

const suppressExpectedWindowError = (expectedMessage: string): (() => void) => {
  const handler = (event: ErrorEvent) => {
    const message =
      event.error instanceof Error
        ? event.error.message
        : typeof event.message === "string"
          ? event.message
          : ""

    if (message.includes(expectedMessage)) {
      event.preventDefault()
    }
  }

  window.addEventListener("error", handler)
  return () => window.removeEventListener("error", handler)
}

describe("ResearchWorkspace error boundary", () => {
  let consoleErrorSpy: ReturnType<typeof vi.spyOn> | null = null

  beforeEach(() => {
    vi.clearAllMocks()
    consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {})
  })

  afterEach(() => {
    consoleErrorSpy?.mockRestore()
    consoleErrorSpy = null
  })

  it("shows recoverable fallback and reload action on render crash", () => {
    const restoreWindowError = suppressExpectedWindowError("chat pane crash")

    try {
      render(<ResearchWorkspace />)

      expect(screen.getByText("Something went wrong")).toBeInTheDocument()
      expect(screen.getByTestId("workspace-reload-button")).toBeInTheDocument()
      expect(
        screen.getByTestId("workspace-export-recovery-button")
      ).toBeInTheDocument()
      expect(
        screen.getByTestId("workspace-clear-cache-button")
      ).toBeInTheDocument()
      expect(
        screen.getByTestId("workspace-switch-from-error-button")
      ).toBeInTheDocument()
    } finally {
      restoreWindowError()
    }
  })

  it("shows error details when toggled", () => {
    const restoreWindowError = suppressExpectedWindowError("chat pane crash")

    try {
      render(<ResearchWorkspace />)

      expect(
        screen.queryByTestId("workspace-error-details")
      ).not.toBeInTheDocument()

      fireEvent.click(screen.getByTestId("workspace-error-details-toggle"))

      expect(
        screen.getByTestId("workspace-error-details")
      ).toBeInTheDocument()
      expect(screen.getByTestId("workspace-error-details")).toHaveTextContent(
        "chat pane crash"
      )
    } finally {
      restoreWindowError()
    }
  })
})
