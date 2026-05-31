import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within
} from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ResearchWorkspace } from "../index"
import { RESEARCH_WORKSPACE_LAST_MOBILE_TAB_STORAGE_KEY } from "../research-workspace-route-state"

const ONBOARDING_KEY = "tldw:research-workspace:onboarding-dismissed:v1"
const {
  onboardingStorageState,
  mockWorkspaceStorageGetItem,
  mockWorkspaceStorageSetItem,
  mockGetResearchWorkspaceCapabilities,
  mockGetResearchBundle,
  mockChatPaneProps,
  mockStudioPaneProps
} = vi.hoisted(() => ({
  onboardingStorageState: {
    value: undefined as string | undefined
  },
  mockWorkspaceStorageGetItem: vi.fn(async (_key: string) => null as string | null),
  mockWorkspaceStorageSetItem: vi.fn(async (_key: string, _value: string) => undefined),
  mockGetResearchWorkspaceCapabilities: vi.fn(async () => ({
    status: "degraded",
    ttl_seconds: 30,
    timestamp: "2026-05-13T00:00:00.000Z",
    capabilities: {
      chat: {
        status: "degraded",
        mode: "warn",
        dependencies: ["llm"],
        reason_code: "llm_health_degraded"
      }
    }
  })),
  mockGetResearchBundle: vi.fn(),
  mockChatPaneProps: [] as any[],
  mockStudioPaneProps: [] as any[]
}))

const mockMessageApi = {
  open: vi.fn(),
  warning: vi.fn(),
  success: vi.fn(),
  destroy: vi.fn()
}

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
  addArtifact: vi.fn(),
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
  }>,
  currentNote: {
    title: "",
    content: "",
    keywords: [] as string[],
    isDirty: false
  },
  workspaceChatSessions: {} as Record<string, { messages: any[] }>,
  focusSourceById: vi.fn(() => true),
  focusChatMessageById: vi.fn(() => true),
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

vi.mock("@/store/workspace", () => {
  const useWorkspaceStore = (selector: (state: typeof testState) => unknown) =>
    selector(testState)
  useWorkspaceStore.getState = () => testState

  return {
    useWorkspaceStore,
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
      removeItem: vi.fn()
    })
  }
})

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getMediaDetails: vi.fn().mockResolvedValue({}),
    getResearchBundle: mockGetResearchBundle,
    getResearchWorkspaceCapabilities: mockGetResearchWorkspaceCapabilities
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
  ChatPane: (props: any) => {
    mockChatPaneProps.push(props)
    return <div data-testid="workspace-chat-pane">Chat</div>
  }
}))

vi.mock("../StudioPane", () => ({
  StudioPane: (props: { onRequestSources?: () => void }) => {
    mockStudioPaneProps.push(props)
    return (
      <div data-testid="workspace-studio-pane">
        Studio
        <button type="button" onClick={props.onRequestSources}>
          Open Sources tab
        </button>
      </div>
    )
  }
}))

vi.mock("../WorkspaceStatusBar", () => ({
  WorkspaceStatusBar: () => <div data-testid="workspace-status-bar" />
}))

vi.mock("antd", () => ({
  Drawer: ({ placement, mask, open, children }: any) => (
    <div
      data-testid={`workspace-drawer-${placement}`}
      data-mask={String(mask)}
      data-open={String(open)}
    >
      {children}
    </div>
  ),
  Tabs: ({ activeKey, items, onChange }: any) => (
    <div data-testid="workspace-mobile-tabs" data-active-key={String(activeKey)}>
      {Array.isArray(items)
        ? items.map((item: any) => (
            <div key={item.key} data-testid={`tab-label-wrapper-${item.key}`}>
              <button
                type="button"
                data-testid={`tab-label-${item.key}`}
                onClick={() => onChange?.(item.key)}
              >
                {item.label}
              </button>
            </div>
          ))
        : null}
      {Array.isArray(items)
        ? items
            .filter((item: any) => item.key === activeKey)
            .map((item: any) => (
              <div key={item.key} data-testid={`tab-${item.key}`}>
                {item.children}
              </div>
            ))
        : null}
    </div>
  ),
  Modal: ({ open, children, title }: any) =>
    open ? <div aria-label={String(title)}>{children}</div> : null,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Input: (props: any) => <input {...props} />,
  Empty: ({ description }: any) => <div>{description}</div>,
  Skeleton: {
    Button: (props: any) => <div {...props} />
  },
  message: {
    useMessage: () => [
      mockMessageApi,
      <div key="message-context" data-testid="workspace-message-context" />
    ]
  }
}))

describe("ResearchWorkspace Stage 2 drawer responsiveness", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    onboardingStorageState.value = "1"
    testState.isMobile = false
    testState.storeHydrated = true
    testState.leftPaneCollapsed = false
    testState.rightPaneCollapsed = false
    testState.workspaceId = "workspace-1"
    testState.workspaceTag = ""
    testState.selectedSourceIds = []
    testState.generatedArtifacts = []
    testState.setSourceStatusByMediaId = vi.fn()
    testState.createNewWorkspace = vi.fn()
    testState.clearCurrentNote = vi.fn()
    testState.loadNote = vi.fn()
    testState.addArtifact = vi.fn()
    mockGetResearchWorkspaceCapabilities.mockClear()
    mockGetResearchBundle.mockReset()
    mockChatPaneProps.length = 0
    mockStudioPaneProps.length = 0
    window.localStorage.removeItem(RESEARCH_WORKSPACE_LAST_MOBILE_TAB_STORAGE_KEY)
    window.history.replaceState(null, "", "/research-workspace")
  })

  const createCompletedDeepResearchBundle = () => ({
    question: "Which intervention gaps remain?",
    report_markdown: "# Deep Report\n\nThe evidence supports follow-up work.",
    claims: [
      {
        text: "Claim one",
        citations: [{ source_id: "src_1", title: "Paper A" }]
      }
    ],
    source_inventory: [{ source_id: "src_1", title: "Paper A" }],
    verification_summary: {
      supported_claim_count: 1,
      unsupported_claim_count: 0
    },
    source_trust: [{ source_id: "src_1", snapshot_policy: "full_artifact" }]
  })

  it("uses non-masked tablet drawers so chat remains visible", () => {
    render(<ResearchWorkspace />)

    expect(screen.getByTestId("workspace-drawer-left")).toHaveAttribute(
      "data-mask",
      "false"
    )
    expect(screen.getByTestId("workspace-drawer-right")).toHaveAttribute(
      "data-mask",
      "false"
    )
  })

  it("loads capability health and passes it to Chat and Studio panes", async () => {
    render(<ResearchWorkspace />)

    await waitFor(() => {
      expect(mockGetResearchWorkspaceCapabilities).toHaveBeenCalledTimes(1)
      expect(
        mockChatPaneProps.at(-1)?.researchWorkspaceCapabilities?.capabilities.chat
      ).toMatchObject({
        status: "degraded",
        mode: "warn",
        reason_code: "llm_health_degraded"
      })
      expect(
        mockStudioPaneProps.at(-1)?.researchWorkspaceCapabilities?.capabilities.chat
      ).toMatchObject({
        status: "degraded",
        mode: "warn",
        reason_code: "llm_health_degraded"
      })
      expect(
        typeof mockStudioPaneProps.at(-1)?.onRefreshResearchWorkspaceCapabilities
      ).toBe("function")
    })
  })

  it("renders mobile tab count badges with AA-safe token classes", () => {
    testState.isMobile = true
    testState.selectedSourceIds = ["source-1", "source-2"]
    testState.generatedArtifacts = [
      { id: "artifact-1" },
      { id: "artifact-2" },
      { id: "artifact-3" }
    ]

    render(<ResearchWorkspace />)

    const sourcesLabel = screen.getByTestId("tab-label-sources")
    const studioLabel = screen.getByTestId("tab-label-studio")
    const sourceCountBadge = within(sourcesLabel).getByText("2", {
      selector: "span"
    })
    const studioCountBadge = within(studioLabel).getByText("3", {
      selector: "span"
    })

    expect(sourcesLabel).toContainElement(sourceCountBadge)
    expect(studioLabel).toContainElement(studioCountBadge)
    expect(sourceCountBadge).toHaveClass("bg-surface2", "text-text")
    expect(studioCountBadge).toHaveClass("bg-surface2", "text-text")
  })

  it("opens Studio from the mobile ?tab=studio route state", () => {
    testState.isMobile = true
    window.history.replaceState(null, "", "/research-workspace?tab=studio")

    render(<ResearchWorkspace />)

    expect(screen.getByTestId("workspace-mobile-tabs")).toHaveAttribute(
      "data-active-key",
      "studio"
    )
    expect(screen.getByTestId("workspace-studio-pane")).toBeInTheDocument()
    expect(screen.queryByTestId("workspace-chat-pane")).not.toBeInTheDocument()
  })

  it("opens Studio from HashRouter mobile ?tab=studio route state", () => {
    testState.isMobile = true
    window.history.replaceState(
      null,
      "",
      "/options.html#/research-workspace?tab=studio"
    )

    render(<ResearchWorkspace />)

    expect(screen.getByTestId("workspace-mobile-tabs")).toHaveAttribute(
      "data-active-key",
      "studio"
    )
    expect(screen.getByTestId("workspace-studio-pane")).toBeInTheDocument()
    expect(screen.queryByTestId("workspace-chat-pane")).not.toBeInTheDocument()
  })

  it("falls back to Chat for invalid mobile tab route state", () => {
    testState.isMobile = true
    window.history.replaceState(null, "", "/research-workspace?tab=banana")

    render(<ResearchWorkspace />)

    expect(screen.getByTestId("workspace-mobile-tabs")).toHaveAttribute(
      "data-active-key",
      "chat"
    )
    expect(screen.getByTestId("workspace-chat-pane")).toBeInTheDocument()
    expect(screen.queryByTestId("workspace-studio-pane")).not.toBeInTheDocument()
  })

  it("opens the persisted mobile tab when no URL tab is present", () => {
    testState.isMobile = true
    window.localStorage.setItem(
      RESEARCH_WORKSPACE_LAST_MOBILE_TAB_STORAGE_KEY,
      "studio"
    )

    render(<ResearchWorkspace />)

    expect(screen.getByTestId("workspace-mobile-tabs")).toHaveAttribute(
      "data-active-key",
      "studio"
    )
    expect(screen.getByTestId("workspace-studio-pane")).toBeInTheDocument()
  })

  it("lets URL tab state override the persisted mobile tab", () => {
    testState.isMobile = true
    window.localStorage.setItem(
      RESEARCH_WORKSPACE_LAST_MOBILE_TAB_STORAGE_KEY,
      "sources"
    )
    window.history.replaceState(null, "", "/research-workspace?tab=studio")

    render(<ResearchWorkspace />)

    expect(screen.getByTestId("workspace-mobile-tabs")).toHaveAttribute(
      "data-active-key",
      "studio"
    )
    expect(screen.getByTestId("workspace-studio-pane")).toBeInTheDocument()
  })

  it("falls back to Chat when persisted mobile tab state is invalid", () => {
    testState.isMobile = true
    window.localStorage.setItem(
      RESEARCH_WORKSPACE_LAST_MOBILE_TAB_STORAGE_KEY,
      "banana"
    )

    render(<ResearchWorkspace />)

    expect(screen.getByTestId("workspace-mobile-tabs")).toHaveAttribute(
      "data-active-key",
      "chat"
    )
    expect(screen.getByTestId("workspace-chat-pane")).toBeInTheDocument()
  })

  it("persists mobile tab changes", () => {
    testState.isMobile = true

    render(<ResearchWorkspace />)

    fireEvent.click(screen.getByTestId("tab-label-studio"))

    expect(screen.getByTestId("workspace-mobile-tabs")).toHaveAttribute(
      "data-active-key",
      "studio"
    )
    expect(window.localStorage.getItem(RESEARCH_WORKSPACE_LAST_MOBILE_TAB_STORAGE_KEY))
      .toBe("studio")
  })

  it("persists later mobile tab changes after initial URL tab state", () => {
    testState.isMobile = true
    window.localStorage.setItem(
      RESEARCH_WORKSPACE_LAST_MOBILE_TAB_STORAGE_KEY,
      "sources"
    )
    window.history.replaceState(null, "", "/research-workspace?tab=chat")

    render(<ResearchWorkspace />)

    fireEvent.click(screen.getByTestId("tab-label-studio"))

    expect(screen.getByTestId("workspace-mobile-tabs")).toHaveAttribute(
      "data-active-key",
      "studio"
    )
    expect(window.localStorage.getItem(RESEARCH_WORKSPACE_LAST_MOBILE_TAB_STORAGE_KEY))
      .toBe("studio")
  })

  it("keeps shared params while opening the requested mobile tab", () => {
    testState.isMobile = true
    window.history.replaceState(
      null,
      "",
      "/research-workspace?shared=abc&tab=studio"
    )

    render(<ResearchWorkspace />)

    expect(window.location.search).toBe("?shared=abc&tab=studio")
    expect(screen.getByTestId("workspace-mobile-tabs")).toHaveAttribute(
      "data-active-key",
      "studio"
    )
  })

  it("surfaces matching Deep Research return context and focuses Studio", async () => {
    window.history.replaceState(
      null,
      "",
      "/research-workspace?source_workspace_id=workspace-1&source_artifact_id=gap-artifact&source_artifact_template=corpus_gap_finder&source_artifact_title=Corpus%20Gap%20Finder&research_run_id=research-run-7"
    )

    render(<ResearchWorkspace />)

    const handoff = screen.getByTestId("workspace-deep-research-return-handoff")
    expect(handoff).toHaveTextContent("Deep Research return ready")
    expect(handoff).toHaveTextContent("Corpus Gap Finder")
    expect(handoff).toHaveTextContent("research-run-7")
    await waitFor(() => {
      expect(testState.setRightPaneCollapsed).toHaveBeenCalledWith(false)
    })
    expect(
      screen.getByRole("complementary", { name: "Studio panel" })
    ).toBeInTheDocument()
  })

  it("imports a completed Deep Research bundle from the return handoff", async () => {
    testState.generatedArtifacts = [
      {
        id: "gap-artifact",
        type: "data_table",
        title: "Corpus Gap Finder",
        status: "completed",
        templateId: "corpus_gap_finder",
        sourceCoverage: {
          selectedSourceIds: ["source-a", "source-b"],
          usableSources: [
            { sourceId: "source-a", mediaId: 101, title: "Paper A" },
            { sourceId: "source-b", mediaId: 102, title: "Paper B" }
          ],
          skippedSources: [],
          truncatedSources: [],
          minimumUsableSourcesMet: true
        },
        sourceLineage: [
          { sourceId: "source-a", mediaId: 101, title: "Paper A" },
          { sourceId: "source-b", mediaId: 102, title: "Paper B" }
        ],
        createdAt: new Date()
      }
    ] as any[]
    mockGetResearchBundle.mockResolvedValueOnce(createCompletedDeepResearchBundle())
    window.history.replaceState(
      null,
      "",
      "/research-workspace?source_workspace_id=workspace-1&source_artifact_id=gap-artifact&source_artifact_template=corpus_gap_finder&source_artifact_title=Corpus%20Gap%20Finder&research_run_id=research-run-7"
    )

    render(<ResearchWorkspace />)

    fireEvent.click(screen.getByRole("button", { name: /import bundle/i }))

    await waitFor(() => {
      expect(mockGetResearchBundle).toHaveBeenCalledWith("research-run-7")
    })
    expect(testState.addArtifact).toHaveBeenCalledTimes(1)
    const artifactPayload = testState.addArtifact.mock.calls[0]?.[0]
    expect(artifactPayload).toMatchObject({
      type: "report",
      status: "completed",
      title: "Deep Research: Corpus Gap Finder",
      producerMetadata: {
        producerType: "deep_research_bundle_import",
        runId: "research-run-7",
        templateId: "corpus_gap_finder"
      }
    })
    expect(artifactPayload.data.deepResearch.runId).toBe("research-run-7")
    expect(artifactPayload.sourceCoverage.selectedSourceIds).toEqual([
      "source-a",
      "source-b"
    ])
    expect(
      screen.getByTestId("workspace-deep-research-return-handoff")
    ).toHaveTextContent("Bundle imported")
  })

  it("cancels a bundle import when the active workspace changes during fetch", async () => {
    let resolveBundle!: (bundle: unknown) => void
    const bundlePromise = new Promise<unknown>((resolve) => {
      resolveBundle = resolve
    })
    mockGetResearchBundle.mockReturnValueOnce(bundlePromise)
    window.history.replaceState(
      null,
      "",
      "/research-workspace?source_workspace_id=workspace-1&source_artifact_id=gap-artifact&source_artifact_template=corpus_gap_finder&source_artifact_title=Corpus%20Gap%20Finder&research_run_id=research-run-7"
    )

    render(<ResearchWorkspace />)

    fireEvent.click(screen.getByRole("button", { name: /import bundle/i }))

    await waitFor(() => {
      expect(mockGetResearchBundle).toHaveBeenCalledWith("research-run-7")
    })
    testState.workspaceId = "workspace-2"

    await act(async () => {
      resolveBundle(createCompletedDeepResearchBundle())
      await bundlePromise
    })

    expect(testState.addArtifact).not.toHaveBeenCalled()
    expect(
      screen.getByTestId("workspace-deep-research-return-handoff")
    ).not.toHaveTextContent("Bundle imported")
  })

  it("cancels a bundle import after the Research Workspace unmounts", async () => {
    let resolveBundle!: (bundle: unknown) => void
    const bundlePromise = new Promise<unknown>((resolve) => {
      resolveBundle = resolve
    })
    mockGetResearchBundle.mockReturnValueOnce(bundlePromise)
    window.history.replaceState(
      null,
      "",
      "/research-workspace?source_workspace_id=workspace-1&source_artifact_id=gap-artifact&source_artifact_template=corpus_gap_finder&source_artifact_title=Corpus%20Gap%20Finder&research_run_id=research-run-7"
    )

    const { unmount } = render(<ResearchWorkspace />)

    fireEvent.click(screen.getByRole("button", { name: /import bundle/i }))

    await waitFor(() => {
      expect(mockGetResearchBundle).toHaveBeenCalledWith("research-run-7")
    })
    unmount()

    await act(async () => {
      resolveBundle(createCompletedDeepResearchBundle())
      await bundlePromise
    })

    expect(testState.addArtifact).not.toHaveBeenCalled()
  })

  it("shows a bundle import error without adding an artifact", async () => {
    mockGetResearchBundle.mockResolvedValueOnce({
      question: "Which intervention gaps remain?"
    })
    window.history.replaceState(
      null,
      "",
      "/research-workspace?source_workspace_id=workspace-1&source_artifact_id=gap-artifact&source_artifact_template=corpus_gap_finder&source_artifact_title=Corpus%20Gap%20Finder&research_run_id=research-run-7"
    )

    render(<ResearchWorkspace />)

    fireEvent.click(screen.getByRole("button", { name: /import bundle/i }))

    await waitFor(() => {
      expect(mockGetResearchBundle).toHaveBeenCalledWith("research-run-7")
    })
    expect(testState.addArtifact).not.toHaveBeenCalled()
    expect(
      screen.getByTestId("workspace-deep-research-return-handoff")
    ).toHaveTextContent("could not be imported")
  })

  it("surfaces matching Deep Research return context from HashRouter search params", async () => {
    window.history.replaceState(
      null,
      "",
      "/options.html#/research-workspace?source_workspace_id=workspace-1&source_artifact_id=gap-artifact&source_artifact_template=corpus_gap_finder&source_artifact_title=Corpus%20Gap%20Finder&research_run_id=research-run-7"
    )

    render(<ResearchWorkspace />)

    const handoff = screen.getByTestId("workspace-deep-research-return-handoff")
    expect(handoff).toHaveTextContent("Deep Research return ready")
    expect(handoff).toHaveTextContent("Corpus Gap Finder")
    expect(handoff).toHaveTextContent("research-run-7")
    await waitFor(() => {
      expect(testState.setRightPaneCollapsed).toHaveBeenCalledWith(false)
    })
  })

  it("ignores Deep Research return context for a different workspace", () => {
    window.history.replaceState(
      null,
      "",
      "/research-workspace?source_workspace_id=workspace-2&source_artifact_id=gap-artifact&source_artifact_template=corpus_gap_finder&source_artifact_title=Corpus%20Gap%20Finder&research_run_id=research-run-7"
    )

    render(<ResearchWorkspace />)

    expect(
      screen.queryByTestId("workspace-deep-research-return-handoff")
    ).not.toBeInTheDocument()
    expect(testState.setRightPaneCollapsed).not.toHaveBeenCalledWith(false)
  })

  it("opens Sources from the mobile Studio source-readiness CTA", async () => {
    testState.isMobile = true
    window.history.replaceState(null, "", "/research-workspace?tab=studio")

    render(<ResearchWorkspace />)

    fireEvent.click(
      await screen.findByRole("button", { name: /open sources tab/i })
    )

    expect(screen.getByTestId("workspace-mobile-tabs")).toHaveAttribute(
      "data-active-key",
      "sources"
    )
    expect(screen.getByTestId("workspace-sources-pane")).toBeInTheDocument()
    expect(screen.queryByTestId("workspace-studio-pane")).not.toBeInTheDocument()
  })

  it("focuses the requested Studio pane on desktop without switching layouts", async () => {
    window.history.replaceState(null, "", "/research-workspace?tab=studio")

    render(<ResearchWorkspace />)

    await waitFor(() => {
      expect(testState.setRightPaneCollapsed).toHaveBeenCalledWith(false)
    })
    expect(screen.getByRole("main")).toHaveAttribute(
      "id",
      "workspace-main-content"
    )
    expect(
      screen.getByRole("complementary", { name: "Studio panel" })
    ).toBeInTheDocument()
  })
})
