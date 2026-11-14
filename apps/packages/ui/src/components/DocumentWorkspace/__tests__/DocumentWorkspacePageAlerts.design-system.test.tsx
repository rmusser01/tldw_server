import React from "react"
import type { QuickIngestOperation } from "@/services/tldw/quick-ingest-authority"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useDocumentWorkspaceStore as store } from "@/store/document-workspace"
import { DocumentWorkspacePage } from "../DocumentWorkspacePage"

const testState = vi.hoisted(() => ({
  owner: null as QuickIngestOperation | null,
  openFromPicker: null as null | ((id: number, hint: "pdf", operation: QuickIngestOperation) => Promise<void>),
  workspace: {} as Record<string, unknown>,
  searchParams: new URLSearchParams(),
  getMediaDetails: vi.fn(() => new Promise(() => {})),
  setSearchParams: vi.fn(),
  setStorage: vi.fn(),
  retrySync: vi.fn(),
  forceSync: vi.fn(),
  closeFromTabs:null as null|((id:number)=>void),
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

vi.mock("@/hooks/document-workspace/useDocumentOperation", () => ({useDocumentOperation: () => testState.owner}))

vi.mock("@/hooks/document-workspace", () => ({
  useAnnotations: vi.fn(),
  useAnnotationSync: () => ({
    retrySync: testState.retrySync,
    forceSync: testState.forceSync
  }),
  useAnnotationSyncOnClose: vi.fn(),
  useReadingProgress: () => ({isPending:false,isError:false}),
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
  DocumentTabBar: ({onCloseDocument}:{onCloseDocument:(id:number)=>void}) => {testState.closeFromTabs=onCloseDocument;return null}
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
    URL.revokeObjectURL = vi.fn()
    testState.openFromPicker = null
    store.getState().reset()
    const controller = new AbortController()
    testState.owner = {authorityKey:"alice",authorityRevision:1,requestScope:{config:{serverUrl:"https://a.test",authMode:"multi-user"},userId:1},signal:controller.signal,isCurrent:()=>!controller.signal.aborted,assertCurrent:()=>controller.signal.throwIfAborted()}
    testState.workspace = createWorkspaceState()
    store.setState({...testState.workspace,workspaceAuthority:'["alice",1]'} as never)
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
    const operation: QuickIngestOperation = { authorityKey: "owner-a", authorityRevision: 0, requestScope: { config: { serverUrl: "https://a.test", authMode: "multi-user" }, userId: 1 }, signal: controller.signal, isCurrent: () => !controller.signal.aborted, assertCurrent: () => controller.signal.throwIfAborted() }
    let pending!: Promise<void>
    await act(async () => { pending = testState.openFromPicker!(7, "pdf", operation); await Promise.resolve() })
    expect(testState.getMediaDetails).toHaveBeenCalledWith(7, expect.objectContaining({ requestScope: operation.requestScope, signal: expect.any(AbortSignal) }))
    if (stage === "file") expect(bgRequest).toHaveBeenCalledWith(expect.objectContaining({ servicePromptConfig: expect.objectContaining({ serverUrl: "https://a.test", expectedUserId: 1 }), abortSignal: expect.any(AbortSignal) }))
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

    store.setState(testState.workspace)
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
  it("retires another account's documents before displaying the workspace", () => {
    const view=render(<DocumentWorkspacePage />)
    store.setState({openDocuments:[{id:7,title:"Alice private",type:"pdf",url:"blob:alice"}],activeDocumentId:7})
    testState.owner={...testState.owner!,authorityKey:"bob",authorityRevision:2,requestScope:{config:{serverUrl:"https://a.test",authMode:"multi-user"},userId:2}}
    view.rerender(<DocumentWorkspacePage />)
    expect(store.getState().openDocuments).toEqual([])
  })
  it("does not publish a held URL-open after leaving the page", async()=>{
    let release!:(value:unknown)=>void
    testState.searchParams=new URLSearchParams("open=42")
    testState.getMediaDetails.mockReturnValue(new Promise(resolve=>{release=resolve}))
    const {bgRequest}=await import("@/services/background-proxy")
    vi.mocked(bgRequest).mockResolvedValue(new ArrayBuffer(3))
    const view=render(<DocumentWorkspacePage />)
    await waitFor(()=>expect(testState.getMediaDetails).toHaveBeenCalled())
    view.unmount()
    await act(async()=>{release({type:"pdf",title:"Late private"});await Promise.resolve();await Promise.resolve()})
    expect(bgRequest).not.toHaveBeenCalled()
    expect(testState.workspace.openDocument).not.toHaveBeenCalled()
  })

  it("keeps the pending-save confirmation for an errored document",async()=>{
    store.setState({annotationSyncStatus:"error",pendingAnnotations:[{id:"pending",documentId:7,location:1,text:"Pending",color:"yellow",createdAt:new Date(0),updatedAt:new Date(0)}]})
    render(<DocumentWorkspacePage />)
    const {Modal}=await import("antd")
    act(()=>testState.closeFromTabs!(7))
    expect(Modal.confirm).toHaveBeenCalled()
    const dialog=vi.mocked(Modal.confirm).mock.calls[0][0]
    await expect(dialog.onOk!()).rejects.toThrow("Annotations are still saving")
    expect(testState.forceSync).toHaveBeenCalledWith(7)
    expect(testState.workspace.closeDocument).not.toHaveBeenCalled()
  })

  it("does not rewrite a newly requested URL to the previous active document",async()=>{
    testState.searchParams=new URLSearchParams("open=2")
    store.setState({activeDocumentId:1,openDocuments:[{id:1,title:"Existing A",type:"pdf",url:"blob:a"}]})
    render(<DocumentWorkspacePage />)
    await waitFor(()=>expect(testState.getMediaDetails).toHaveBeenCalledWith(2,expect.anything()))
    expect(testState.setSearchParams.mock.calls.some(([params])=>params.get("open")==="1")).toBe(false)
  })

  it("retires owned close notices and fences their old Undo after account replacement",async()=>{
    store.setState({activeDocumentId:7,openDocuments:[{id:7,title:"Private Alice title",type:"pdf",url:"blob:alice"}]})
    const view=render(<DocumentWorkspacePage />)
    const {notification}=await import("antd")
    act(()=>testState.closeFromTabs!(7))
    const notice=vi.mocked(notification.info).mock.calls[0][0]
    testState.owner={...testState.owner!,authorityKey:"bob",authorityRevision:2}
    view.rerender(<DocumentWorkspacePage />)
    const noticeRetired=vi.mocked(notification.destroy).mock.calls.some(([key])=>key===notice.key)
    const undoButton = notice.btn
    if (!React.isValidElement<{onClick:()=>void}>(undoButton)) throw new Error("Expected the owned Undo button")
    act(()=>undoButton.props.onClick())
    expect({undoCalls:vi.mocked(testState.workspace.undoCloseDocument as ()=>void).mock.calls.length,noticeRetired}).toEqual({undoCalls:0,noticeRetired:true})
  })
  it("destroys only its owned pending-close confirmation on account replacement",async()=>{
    const {Modal}=await import("antd")
    const destroy=vi.fn();vi.mocked(Modal.confirm).mockReturnValueOnce({destroy,update:vi.fn()})
    store.setState({pendingAnnotations:[{id:"pending",documentId:7,location:1,text:"Private pending",color:"yellow",createdAt:new Date(0),updatedAt:new Date(0)}]})
    const view=render(<DocumentWorkspacePage />)
    act(()=>testState.closeFromTabs!(7))
    testState.owner={...testState.owner!,authorityKey:"bob",authorityRevision:2}
    view.rerender(<DocumentWorkspacePage />)
    expect(destroy).toHaveBeenCalledTimes(1)
  })

  it("does not rewrite a warm URL target before its tab selection commits",()=>{
    testState.searchParams=new URLSearchParams("open=2")
    store.setState({activeDocumentId:1,openDocuments:[{id:1,title:"A",type:"pdf",url:"blob:a"},{id:2,title:"B",type:"pdf",url:"blob:b"}]})
    render(<DocumentWorkspacePage />)
    expect(store.getState().activeDocumentId).toBe(2)
    expect(testState.setSearchParams.mock.calls.some(([params])=>params.get("open")==="1")).toBe(false)
  })

})
