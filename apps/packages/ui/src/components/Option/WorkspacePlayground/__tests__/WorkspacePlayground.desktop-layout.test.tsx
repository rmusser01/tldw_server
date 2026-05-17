import { readFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { fireEvent, render, screen } from "@testing-library/react"
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { WorkspacePlayground } from "../index"

const testDirname = dirname(fileURLToPath(import.meta.url))
const chatPanePropsSpy = vi.fn()

const stripTsxComments = (source: string): string =>
  source
    .replace(/\{\/\*[\s\S]*?\*\/\}/g, "")
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/^\s*\/\/.*$/gm, "")

const getWorkspacePlaygroundWrapperClassName = (source: string): string | null => {
  const uncommentedSource = stripTsxComments(source)
  const wrapperMatch = uncommentedSource.match(
    /<([A-Za-z][\w.]*)\b([^>]*)>\s*<WorkspacePlayground\s*\/>\s*<\/\1>/m
  )
  if (!wrapperMatch) return null

  const classNameMatch = wrapperMatch[2].match(/\bclassName=(["'`])([^"'`]+)\1/)
  return classNameMatch?.[2] ?? null
}

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
  setLeftPaneCollapsed: vi.fn(),
  setRightPaneCollapsed: vi.fn(),
  selectedSourceIds: [] as string[],
  generatedArtifacts: [] as Array<{ id: string }>,
  sources: [] as Array<{
    id: string
    mediaId: number
    title: string
    type: "pdf" | "video" | "audio" | "website" | "document" | "text"
    addedAt: Date
  }>,
  currentNote: {
    title: "",
    content: "",
    keywords: [] as string[],
    isDirty: false
  },
  workspaceChatSessions: {} as Record<
    string,
    { messages: Array<{ message: string; sources: unknown[]; isBot: boolean; name: string }> }
  >,
  focusSourceById: vi.fn(),
  focusChatMessageById: vi.fn(),
  focusWorkspaceNote: vi.fn(),
  setSourceStatusByMediaId: vi.fn()
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
  createWorkspaceStorage: () => ({
    getItem: vi.fn().mockResolvedValue("1"),
    setItem: vi.fn().mockResolvedValue(undefined),
    removeItem: vi.fn().mockResolvedValue(undefined)
  }),
  useWorkspaceStore: (
    selector: (state: {
      storeHydrated?: boolean
      workspaceId: string | null
      initializeWorkspace: () => void
      addSources: (
        sources: Array<{ mediaId: number; title: string; type: string }>
      ) => unknown
      setSelectedSourceIds: (ids: string[]) => void
      captureToCurrentNote: (input: {
        title?: string
        content: string
        mode?: "append" | "replace"
      }) => void
      leftPaneCollapsed: boolean
      rightPaneCollapsed: boolean
      setLeftPaneCollapsed: (collapsed: boolean) => void
      setRightPaneCollapsed: (collapsed: boolean) => void
      selectedSourceIds: string[]
      generatedArtifacts: Array<{ id: string }>
      sources: Array<{
        id: string
        mediaId: number
        title: string
        type: "pdf" | "video" | "audio" | "website" | "document" | "text"
        addedAt: Date
      }>
      currentNote: {
        title: string
        content: string
        keywords: string[]
        isDirty: boolean
      }
      workspaceChatSessions: Record<
        string,
        { messages: Array<{ message: string; sources: unknown[]; isBot: boolean; name: string }> }
      >
      focusSourceById: (id: string) => boolean
      focusChatMessageById: (messageId: string) => boolean
      focusWorkspaceNote: (field?: "title" | "content") => void
      setSourceStatusByMediaId: (
        mediaId: number,
        status: "processing" | "ready" | "error",
        statusMessage?: string
      ) => void
    }) => unknown
  ) =>
    selector({
      storeHydrated: testState.storeHydrated,
      workspaceId: testState.workspaceId,
      initializeWorkspace: testState.initializeWorkspace,
      addSources: testState.addSources,
      setSelectedSourceIds: testState.setSelectedSourceIds,
      captureToCurrentNote: testState.captureToCurrentNote,
      leftPaneCollapsed: testState.leftPaneCollapsed,
      rightPaneCollapsed: testState.rightPaneCollapsed,
      setLeftPaneCollapsed: testState.setLeftPaneCollapsed,
      setRightPaneCollapsed: testState.setRightPaneCollapsed,
      selectedSourceIds: testState.selectedSourceIds,
      generatedArtifacts: testState.generatedArtifacts,
      sources: testState.sources,
      currentNote: testState.currentNote,
      workspaceChatSessions: testState.workspaceChatSessions,
      focusSourceById: testState.focusSourceById,
      focusChatMessageById: testState.focusChatMessageById,
      focusWorkspaceNote: testState.focusWorkspaceNote,
      setSourceStatusByMediaId: testState.setSourceStatusByMediaId
    }),
  createWorkspaceStorage: () => ({
    getItem: vi.fn().mockResolvedValue("1"),
    setItem: vi.fn().mockResolvedValue(undefined),
    removeItem: vi.fn().mockResolvedValue(undefined)
  })
}))

vi.mock("@/utils/workspace-playground-prefill", () => ({
  consumeWorkspacePlaygroundPrefill: vi.fn().mockResolvedValue(null),
  buildKnowledgeQaSeedNote: vi.fn().mockReturnValue(""),
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getMediaDetails: vi.fn().mockResolvedValue({})
  }
}))

vi.mock("../WorkspaceHeader", () => ({
  WorkspaceHeader: () => <div data-testid="workspace-header" />
}))

vi.mock("../SourcesPane", () => ({
  SourcesPane: () => <div data-testid="workspace-sources-pane">Sources</div>
}))

vi.mock("../ChatPane", () => ({
  ChatPane: (props: { contentWidthMode?: string }) => {
    chatPanePropsSpy(props)
    return (
      <div
        data-testid="workspace-chat-pane"
        data-content-width-mode={props.contentWidthMode ?? ""}
      >
        Chat
      </div>
    )
  }
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

describe("WorkspacePlayground desktop layout guardrails", () => {
  const originalMatchMedia = window.matchMedia

  beforeAll(() => {
    if (typeof window.matchMedia !== "function") {
      Object.defineProperty(window, "matchMedia", {
        writable: true,
        value: vi.fn().mockImplementation((query: string) => ({
          matches: false,
          media: query,
          onchange: null,
          addListener: vi.fn(),
          removeListener: vi.fn(),
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          dispatchEvent: vi.fn()
        }))
      })
    }
  })

  afterAll(() => {
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: originalMatchMedia
    })
  })

  beforeEach(() => {
    vi.clearAllMocks()
    chatPanePropsSpy.mockClear()
    testState.isMobile = false
    testState.storeHydrated = true
    testState.leftPaneCollapsed = false
    testState.rightPaneCollapsed = false
    testState.workspaceId = "workspace-1"
    testState.selectedSourceIds = []
    testState.generatedArtifacts = []
    testState.sources = []
    testState.currentNote = {
      title: "",
      content: "",
      keywords: [],
      isDirty: false
    }
    testState.workspaceChatSessions = {}
    testState.setSourceStatusByMediaId = vi.fn()
  })

  it("renders the desktop three-panel structure with sources, chat, and studio panes", () => {
    const { container } = render(<WorkspacePlayground />)

    expect(screen.getByTestId("workspace-header")).toBeInTheDocument()
    expect(screen.getByTestId("workspace-sources-pane")).toBeInTheDocument()
    expect(screen.getByTestId("workspace-chat-pane")).toBeInTheDocument()
    expect(screen.getByTestId("workspace-studio-pane")).toBeInTheDocument()

    expect(container.querySelectorAll("aside")).toHaveLength(2)

    const main = container.querySelector("main")
    expect(main).not.toBeNull()
    expect(
      main?.querySelector("[data-testid='workspace-chat-pane']")
    ).not.toBeNull()

    const root = container.firstElementChild as HTMLElement | null
    expect(root).not.toBeNull()
    expect(root?.className).toContain("h-full")
    expect(screen.getByTestId("workspace-chat-pane")).toHaveAttribute(
      "data-content-width-mode",
      "comfortable"
    )
  })

  it("renders hydration skeleton until workspace store hydration completes", () => {
    testState.storeHydrated = false

    render(<WorkspacePlayground />)

    expect(screen.getByTestId("workspace-playground-skeleton")).toBeInTheDocument()
    expect(screen.queryByTestId("workspace-header")).not.toBeInTheDocument()
  })

  it("swaps from hydration skeleton to panes after hydration", () => {
    testState.storeHydrated = false
    const { rerender } = render(<WorkspacePlayground />)

    expect(screen.getByTestId("workspace-playground-skeleton")).toBeInTheDocument()

    testState.storeHydrated = true
    rerender(<WorkspacePlayground />)

    expect(screen.queryByTestId("workspace-playground-skeleton")).not.toBeInTheDocument()
    expect(screen.getByTestId("workspace-header")).toBeInTheDocument()
    expect(screen.getByTestId("workspace-chat-pane")).toBeInTheDocument()
  })

  it("expands chat content width when one sidebar is collapsed", () => {
    testState.rightPaneCollapsed = true

    render(<WorkspacePlayground />)

    expect(screen.getByTestId("workspace-chat-pane")).toHaveAttribute(
      "data-content-width-mode",
      "expanded"
    )
  })

  it("shows a restore rail for the sources pane when the left pane is collapsed", () => {
    testState.leftPaneCollapsed = true

    render(<WorkspacePlayground />)

    expect(screen.queryByTestId("workspace-sources-pane")).not.toBeInTheDocument()

    const restoreSourcesButton = screen.getByRole("button", {
      name: /show sources/i
    })
    expect(restoreSourcesButton).toHaveAttribute(
      "data-testid",
      "workspace-restore-sources"
    )

    fireEvent.click(restoreSourcesButton)

    expect(testState.setLeftPaneCollapsed).toHaveBeenCalledWith(false)
  })

  it("keeps collapsed sidebar restore rails persistent and associated with their panels", () => {
    testState.leftPaneCollapsed = true
    testState.rightPaneCollapsed = true

    render(<WorkspacePlayground />)

    const restoreSourcesButton = screen.getByTestId("workspace-restore-sources")
    const restoreStudioButton = screen.getByTestId("workspace-restore-studio")

    expect(restoreSourcesButton).toHaveAttribute("aria-controls", "workspace-sources-panel")
    expect(restoreSourcesButton).toHaveAttribute("aria-expanded", "false")
    expect(restoreSourcesButton).toHaveClass("sticky")
    expect(restoreSourcesButton).toHaveClass("top-2")
    expect(restoreSourcesButton).toHaveClass("self-stretch")
    expect(restoreSourcesButton).toHaveClass("min-h-[14rem]")
    expect(restoreSourcesButton).toHaveClass("w-11")

    expect(restoreStudioButton).toHaveAttribute("aria-controls", "workspace-studio-panel")
    expect(restoreStudioButton).toHaveAttribute("aria-expanded", "false")
    expect(restoreStudioButton).toHaveClass("sticky")
    expect(restoreStudioButton).toHaveClass("top-2")
    expect(restoreStudioButton).toHaveClass("self-stretch")
    expect(restoreStudioButton).toHaveClass("min-h-[14rem]")
    expect(restoreStudioButton).toHaveClass("w-11")
  })

  it("shows a restore rail for the studio pane when the right pane is collapsed", () => {
    testState.rightPaneCollapsed = true

    render(<WorkspacePlayground />)

    expect(screen.queryByTestId("workspace-studio-pane")).not.toBeInTheDocument()

    const restoreStudioButton = screen.getByRole("button", {
      name: /show studio/i
    })
    expect(restoreStudioButton).toHaveAttribute(
      "data-testid",
      "workspace-restore-studio"
    )

    fireEvent.click(restoreStudioButton)

    expect(testState.setRightPaneCollapsed).toHaveBeenCalledWith(false)
  })

  it("uses full chat content width when both sidebars are collapsed", () => {
    testState.leftPaneCollapsed = true
    testState.rightPaneCollapsed = true

    render(<WorkspacePlayground />)

    expect(screen.getByTestId("workspace-chat-pane")).toHaveAttribute(
      "data-content-width-mode",
      "full"
    )
  })

  it("keeps the shared WebUI and extension route wrappers height-bounded", () => {
    const sharedRoute = readFileSync(
      resolve(testDirname, "../../../../routes/option-workspace-playground.tsx"),
      "utf8"
    )
    const extensionRoute = readFileSync(
      resolve(
        testDirname,
        "../../../../../../../tldw-frontend/extension/routes/option-workspace-playground.tsx"
      ),
      "utf8"
    )
    const sharedWrapperClassName = getWorkspacePlaygroundWrapperClassName(sharedRoute)
    const extensionWrapperClassName =
      getWorkspacePlaygroundWrapperClassName(extensionRoute)

    expect(sharedWrapperClassName?.split(/\s+/)).toEqual(
      expect.arrayContaining(["h-full", "min-h-0", "flex-1", "overflow-hidden"])
    )
    expect(extensionWrapperClassName?.split(/\s+/)).toEqual(
      expect.arrayContaining(["h-full", "min-h-0", "flex-1", "overflow-hidden"])
    )
  })
})
