import React from "react"
import type { QuickIngestOperation } from "@/services/tldw/quick-ingest-authority"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { DocumentWorkspacePage } from "../DocumentWorkspacePage"

const testState = vi.hoisted(() => ({
  openFromPicker: null as null | ((id: number, hint: "pdf", operation: QuickIngestOperation) => Promise<void>),
  workspace: {} as Record<string, unknown>,
  searchParams: new URLSearchParams(),
  getMediaDetails: vi.fn(() => new Promise(() => {})),
  setSearchParams: vi.fn(),
  setStorage: vi.fn(),
  retrySync: vi.fn(),
  forceSync: vi.fn(),
  forceSave: vi.fn(),
  messageError: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

vi.mock("react-router-dom", () => ({
  useSearchParams: () => [testState.searchParams, testState.setSearchParams]
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: () => [false, testState.setStorage]
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => false,
  useTablet: () => false
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    error: testState.messageError
  })
}))

vi.mock("@/store/document-workspace", () => ({
  useDocumentWorkspaceStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector(testState.workspace)
}))

vi.mock("@/hooks/document-workspace", () => ({
  useAnnotations: vi.fn(),
  useAnnotationSync: () => ({
    retrySync: testState.retrySync,
    forceSync: testState.forceSync
  }),
  useAnnotationSyncOnClose: vi.fn(),
  useReadingProgress: vi.fn(),
  useReadingProgressAutoSave: () => ({
    forceSave: testState.forceSave
  }),
  useReadingProgressSaveOnClose: vi.fn(),
  useResizablePanel: () => ({
    width: 320,
    handleMouseDown: vi.fn()
  })
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: vi.fn()
}))

vi.mock("@/services/tldw", () => ({
  tldwClient: {
    getMediaDetails: testState.getMediaDetails
  }
}))

vi.mock("antd", () => ({
  Drawer: ({ open, children }: { open?: boolean; children?: React.ReactNode }) =>
    open ? <div>{children}</div> : null,
  Dropdown: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
  Modal: {
    confirm: vi.fn()
  },
  notification: {
    info: vi.fn(),
    destroy: vi.fn()
  },
  Tabs: ({ items }: { items?: Array<{ key: string; children?: React.ReactNode }> }) => (
    <div>
      {items?.map((item) => (
        <div key={item.key}>{item.children}</div>
      ))}
    </div>
  ),
  Tooltip: ({ children }: { children?: React.ReactNode }) => <>{children}</>
}))

vi.mock("../DocumentWorkspaceErrorBoundary", () => ({
  DocumentWorkspaceErrorBoundary: ({ children }: { children?: React.ReactNode }) => (
    <>{children}</>
  )
}))

vi.mock("../DocumentShortcutsModal", () => ({
  DocumentShortcutsModal: () => null
}))

vi.mock("../DocumentTabBar", () => ({
  DocumentTabBar: () => null
}))

vi.mock("../SyncStatusIndicator", () => ({
  SyncStatusIndicator: () => null
}))

vi.mock("../WorkspaceTips", () => ({
  WorkspaceTour: () => null,
  HighlightTip: () => null,
  MultiDocTip: () => null,
  resetTour: vi.fn(),
  resetAllTips: vi.fn()
}))

vi.mock("../DocumentViewer", () => ({
  DocumentViewer: () => <div data-testid="document-viewer" />
}))

vi.mock("../DocumentPickerModal", () => ({
  default: ({ onOpenDocument }: { onOpenDocument: NonNullable<typeof testState.openFromPicker> }) => { testState.openFromPicker = onOpenDocument; return <div>Picker mounted</div> }
}))

const createWorkspaceState = (
  overrides: Partial<Record<string, unknown>> = {}
): Record<string, unknown> => ({
  activeDocumentId: null,
  openDocuments: [],
  openDocument: vi.fn(),
  annotationsHealth: "ready",
  progressHealth: "ready",
  closeDocument: vi.fn(),
  undoCloseDocument: vi.fn(),
  annotationSyncStatus: "idle",
  recentlyClosed: [],
  ...overrides
})

const findRenderedAlert = async (container: HTMLElement, text: string) =>
  waitFor(() => {
    const alert = Array.from(
      container.querySelectorAll<HTMLElement>('[role="alert"], [role="status"]')
    ).find((node) => node.textContent?.includes(text))

    expect(alert).toBeTruthy()
    return alert as HTMLElement
  })

describe("DocumentWorkspacePage design-system alerts", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    testState.openFromPicker = null
    testState.workspace = createWorkspaceState()
    testState.searchParams = new URLSearchParams()
    testState.getMediaDetails.mockImplementation(() => new Promise(() => {}))
  })

  it.each(["metadata", "file"])("abandons a Picker %s read after the captured owner changes", async (stage) => {
    let release!: (value: unknown) => void
    const delayed = new Promise(resolve => { release = resolve })
    testState.getMediaDetails.mockImplementation(async () => stage === "metadata" ? delayed : ({ type: "pdf", title: "Bob private" }))
    const { bgRequest } = await import("@/services/background-proxy")
    vi.mocked(bgRequest).mockImplementation(async () => delayed)
    render(<DocumentWorkspacePage />)
    fireEvent.click(screen.getByTestId("document-open-picker-button"))
    await screen.findByText("Picker mounted")
    const controller = new AbortController()
    const operation: QuickIngestOperation = { authorityKey: "owner-a", requestScope: { config: { serverUrl: "https://a.test", authMode: "multi-user" }, userId: 1 }, signal: controller.signal, isCurrent: () => !controller.signal.aborted, assertCurrent: () => controller.signal.throwIfAborted() }
    let pending!: Promise<void>
    await act(async () => { pending = testState.openFromPicker!(7, "pdf", operation); await Promise.resolve() })
    expect(testState.getMediaDetails).toHaveBeenCalledWith(7, expect.objectContaining({ requestScope: operation.requestScope, signal: operation.signal }))
    if (stage === "file") expect(bgRequest).toHaveBeenCalledWith(expect.objectContaining({ servicePromptConfig: expect.objectContaining({ serverUrl: "https://a.test", expectedUserId: 1 }), abortSignal: operation.signal }))
    controller.abort()
    await act(async () => { release(stage === "metadata" ? { type: "pdf", title: "Bob private" } : new ArrayBuffer(3)); await pending })
    if (stage === "metadata") expect(bgRequest).not.toHaveBeenCalled()
    expect(testState.workspace.openDocument).not.toHaveBeenCalled()
  })

  it("renders the auto-open loading state through the design-system Alert", async () => {
    testState.searchParams = new URLSearchParams("open=42")

    const { container } = render(<DocumentWorkspacePage />)

    const loadingAlert = await findRenderedAlert(container, "Loading document...")
    expect(loadingAlert).toHaveTextContent("Loading document...")
    expect(loadingAlert).toHaveTextContent(
      "Fetching the document file. This can take a moment for large files."
    )
    expect(loadingAlert.closest('[data-ds-component="Alert"]')).not.toBeNull()
    expect(container.querySelectorAll('[data-ds-component="Alert"]')).toHaveLength(1)
  })

  it("renders workspace storage health issues through the design-system Alert", () => {
    testState.workspace = createWorkspaceState({
      annotationsHealth: "error",
      progressHealth: "error"
    })

    const { container } = render(<DocumentWorkspacePage />)

    const healthMessage = screen.getByText("Document workspace storage unavailable")
    expect(healthMessage.closest('[data-ds-component="Alert"]')).not.toBeNull()
    expect(screen.getByText("Annotations storage is unavailable on the server."))
      .toBeInTheDocument()
    expect(screen.getByText("Reading progress storage is unavailable on the server."))
      .toBeInTheDocument()
    expect(
      screen.getByText(
        "Some workspace features are temporarily unavailable. This usually resolves after restarting the server. If this persists, contact your administrator."
      )
    ).toBeInTheDocument()
    expect(container.querySelectorAll('[data-ds-component="Alert"]')).toHaveLength(1)
  })
})
