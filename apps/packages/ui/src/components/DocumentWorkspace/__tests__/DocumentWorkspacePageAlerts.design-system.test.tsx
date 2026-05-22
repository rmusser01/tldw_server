import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { DocumentWorkspacePage } from "../DocumentWorkspacePage"

const testState = vi.hoisted(() => ({
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
  default: () => null
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
    testState.workspace = createWorkspaceState()
    testState.searchParams = new URLSearchParams()
    testState.getMediaDetails.mockImplementation(() => new Promise(() => {}))
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
