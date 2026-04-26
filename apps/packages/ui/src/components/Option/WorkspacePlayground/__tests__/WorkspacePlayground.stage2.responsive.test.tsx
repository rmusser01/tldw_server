import { render, screen, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { WorkspacePlayground } from "../index"

const ONBOARDING_KEY = "tldw:workspace-playground:onboarding-dismissed:v1"
const {
  onboardingStorageState,
  mockWorkspaceStorageGetItem,
  mockWorkspaceStorageSetItem
} = vi.hoisted(() => ({
  onboardingStorageState: {
    value: undefined as string | undefined
  },
  mockWorkspaceStorageGetItem: vi.fn(async (_key: string) => null as string | null),
  mockWorkspaceStorageSetItem: vi.fn(async (_key: string, _value: string) => undefined)
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

vi.mock("@/store/workspace", () => ({
  useWorkspaceStore: (
    selector: (state: typeof testState) => unknown
  ) => selector(testState),
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
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getMediaDetails: vi.fn().mockResolvedValue({})
  }
}))

vi.mock("@/utils/workspace-playground-prefill", () => ({
  consumeWorkspacePlaygroundPrefill: vi.fn().mockResolvedValue(null),
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
  Tabs: ({ items }: any) => (
    <div>
      {Array.isArray(items)
        ? items.map((item: any) => (
            <div key={item.key} data-testid={`tab-${item.key}`}>
              <div data-testid={`tab-label-${item.key}`}>{item.label}</div>
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

describe("WorkspacePlayground Stage 2 drawer responsiveness", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    onboardingStorageState.value = "1"
    testState.isMobile = false
    testState.storeHydrated = true
    testState.leftPaneCollapsed = false
    testState.rightPaneCollapsed = false
    testState.workspaceTag = ""
    testState.setSourceStatusByMediaId = vi.fn()
    testState.createNewWorkspace = vi.fn()
    testState.clearCurrentNote = vi.fn()
    testState.loadNote = vi.fn()
  })

  it("uses non-masked tablet drawers so chat remains visible", () => {
    render(<WorkspacePlayground />)

    expect(screen.getByTestId("workspace-drawer-left")).toHaveAttribute(
      "data-mask",
      "false"
    )
    expect(screen.getByTestId("workspace-drawer-right")).toHaveAttribute(
      "data-mask",
      "false"
    )
  })

  it("renders mobile tab count badges with AA-safe token classes", () => {
    testState.isMobile = true
    testState.selectedSourceIds = ["source-1", "source-2"]
    testState.generatedArtifacts = [
      { id: "artifact-1" },
      { id: "artifact-2" },
      { id: "artifact-3" }
    ]

    render(<WorkspacePlayground />)

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
})
