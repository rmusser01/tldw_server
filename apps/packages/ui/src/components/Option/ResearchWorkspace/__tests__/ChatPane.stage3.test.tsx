import { webcrypto } from "node:crypto"
import { sha256Text } from "@/store/workspace-migration"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import type { ReactNode } from "react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ConnectionPhase } from "@/types/connection"
import { ChatPane } from "../ChatPane"
import type { WebArticleCapturePin, WorkspaceSource } from "@/types/workspace"
import { create, type StoreApi, type UseBoundStore } from "zustand"
import type { WorkspaceState } from "@/store/workspace"
import { createSourcesSlice } from "@/store/workspace-slices/sources-slice"
import { WebCaptureNotCurrentError } from "@/utils/research-web-capture"
import { isWorkspaceSourceSelectable } from "@/store/workspace-source-status"

let realSourcesStore: UseBoundStore<StoreApi<WorkspaceState>> | null = null

const mockCheckConnectionOnce = vi.fn()
const mockSaveWorkspaceChatSession = vi.fn()
const mockGetWorkspaceChatSession = vi.fn()
const mockFocusSourceById = vi.fn()
const mockFocusSourceByMediaId = vi.fn()
const mockSetMessages = vi.fn()
const mockSetHistory = vi.fn()
const mockSetHistoryId = vi.fn()
const mockSetServerChatId = vi.fn()
const mockSetStreaming = vi.fn()
const mockSetIsProcessing = vi.fn()
const mockStopStreamingRequest = vi.fn()
const mockOnSubmit = vi.fn()
const mockRegenerateLastMessage = vi.fn()
const mockDeleteMessage = vi.fn()
const mockEditMessage = vi.fn()
const mockGetMediaDetails = vi.fn()
const mockCaptureHead = vi.fn()
const mockVersion = vi.fn()
const mockWorkspaceSources = vi.fn()
const mockMediaVersions = vi.fn()
const mockScope = vi.fn()
vi.mock("@/utils/research-web-capture", async () => ({
  ...(await vi.importActual<typeof import("@/utils/research-web-capture")>(
    "@/utils/research-web-capture"
  )),
  assertWebCaptureHeadCurrent: (...args: unknown[]) => mockCaptureHead(...args)
}))
vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: (...args: unknown[]) => mockScope(...args)
}))
const mockSetRagMediaIds = vi.fn()
const mockSetChatMode = vi.fn()
const mockSetFileRetrievalEnabled = vi.fn()
const mockSetRagTopK = vi.fn()
const mockSetRagAdvancedOptions = vi.fn()
const mockSetSelectedModel = vi.fn()
const mockCaptureToCurrentNote = vi.fn()
const mockClearChatFocusTarget = vi.fn()

const connectionStoreState = {
  state: {
    phase: ConnectionPhase.CONNECTED,
    isChecking: false,
    lastError: null
  },
  checkOnce: mockCheckConnectionOnce
}

const workspaceStoreState = {
  currentNote: { title: "", content: "", keywords: [], isDirty: true },
  sources: [] as Array<{
    id: string
    mediaId: number
    title: string
    type: "pdf" | "video" | "audio" | "website" | "document" | "text"
    status?: "processing" | "ready" | "error"
    addedAt?: Date
    url?: string
    webCapture?: WebArticleCapturePin
  }>,
  selectedSourceFolderIds: [] as string[],
  sourceFolders: [] as Array<{
    id: string
    name: string
    parentId: string | null
  }>,
  sourceFolderMemberships: [] as Array<{ folderId: string; sourceId: string }>,
  selectedSourceIds: [] as string[],
  getSelectedSources: () =>
    [] as Array<{
      id: string
      mediaId: number
      title: string
      type: "pdf" | "video" | "audio" | "website" | "document" | "text"
      status?: "processing" | "ready" | "error"
    }>,
  getSelectedMediaIds: () => [] as number[],
  setSelectedSourceIds: vi.fn(),
  setSourceStatusById: vi.fn(),
  focusSourceById: mockFocusSourceById,
  focusSourceByMediaId: mockFocusSourceByMediaId,
  chatFocusTarget: null as { messageId: string; token: number } | null,
  clearChatFocusTarget: mockClearChatFocusTarget,
  captureToCurrentNote: mockCaptureToCurrentNote,
  workspaceId: "workspace-a",
  workspaceChatReferenceId: "workspace-a",
  saveWorkspaceChatSession: mockSaveWorkspaceChatSession,
  getWorkspaceChatSession: mockGetWorkspaceChatSession
}

const optionStoreState = {
  selectedModel: "test-model",
  setSelectedModel: mockSetSelectedModel,
  setRagMediaIds: mockSetRagMediaIds,
  setChatMode: mockSetChatMode,
  setFileRetrievalEnabled: mockSetFileRetrievalEnabled,
  ragTopK: 8,
  setRagTopK: mockSetRagTopK,
  ragAdvancedOptions: {
    min_score: 0.2,
    enable_reranking: false
  } as Record<string, unknown>,
  setRagAdvancedOptions: mockSetRagAdvancedOptions
}

const messageOptionState = {
  messages: [] as Array<{
    id: string
    isBot: boolean
    name: string
    message: string
    sources: any[]
  }>,
  setMessages: mockSetMessages,
  history: [] as Array<{
    role: "user" | "assistant" | "system"
    content: string
  }>,
  setHistory: mockSetHistory,
  streaming: false,
  setStreaming: mockSetStreaming,
  isProcessing: false,
  setIsProcessing: mockSetIsProcessing,
  onSubmit: mockOnSubmit,
  stopStreamingRequest: mockStopStreamingRequest,
  regenerateLastMessage: mockRegenerateLastMessage,
  deleteMessage: mockDeleteMessage,
  editMessage: mockEditMessage,
  historyId: null as string | null,
  setHistoryId: mockSetHistoryId,
  serverChatId: null as string | null,
  setServerChatId: mockSetServerChatId
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
    }
  })
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => false
}))

vi.mock("@/hooks/useSmartScroll", () => ({
  useSmartScroll: () => ({
    containerRef: { current: null },
    isAutoScrollToBottom: true,
    autoScrollToBottom: vi.fn()
  })
}))

vi.mock("@/store/connection", () => ({
  useConnectionStore: (
    selector: (state: typeof connectionStoreState) => unknown
  ) => selector(connectionStoreState)
}))

vi.mock("@/store/workspace", async () => ({
  ...(await vi.importActual<typeof import("@/store/workspace")>(
    "@/store/workspace"
  )),
  useWorkspaceStore: Object.assign(
    (selector: (state: typeof workspaceStoreState) => unknown) =>
      realSourcesStore
        ? realSourcesStore(
            selector as unknown as (state: WorkspaceState) => unknown
          )
        : selector(workspaceStoreState),
    {
      getState: () => realSourcesStore?.getState() ?? workspaceStoreState,
      subscribe: (listener: () => void) =>
        realSourcesStore?.subscribe(listener) ?? (() => {})
    }
  )
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (
    selector: (state: typeof optionStoreState) => unknown
  ) => selector(optionStoreState)
}))

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => messageOptionState
}))

vi.mock("@/components/Common/Playground/Message", () => ({
  PlaygroundMessage: ({
    message,
    onSaveToWorkspaceNotes,
    onRegenerate,
    onEditFormSubmit
  }: {
    message: string
    onSaveToWorkspaceNotes?: () => void
    onRegenerate?: () => void
    onEditFormSubmit?: (value: string, isSend: boolean) => void
  }) => (
    <div>
      <div data-testid="playground-message">{message}</div>
      <button onClick={onRegenerate}>Regenerate capture test</button>
      <button onClick={() => onEditFormSubmit?.("edited question", true)}>
        Edit and send capture test
      </button>
      {onSaveToWorkspaceNotes && (
        <button type="button" onClick={onSaveToWorkspaceNotes}>
          Save to Notes
        </button>
      )}
    </div>
  )
}))

vi.mock("@/components/Common/FeatureEmptyState", () => ({
  default: ({
    title,
    examples
  }: {
    title: ReactNode
    examples?: ReactNode[]
  }) => (
    <div>
      <div>{title}</div>
      <ul>
        {(examples ?? []).map((example, index) => (
          <li key={index}>{example}</li>
        ))}
      </ul>
    </div>
  )
}))

vi.mock("../source-location-copy", () => ({
  getWorkspaceChatNoSourcesHint: () =>
    "Select sources from the Sources pane, then ask questions.",
  getWorkspaceChatSourcesExplainer: () =>
    "Sources are documents, PDFs, web pages, or other content you add. Add or select sources from the Sources pane to ask grounded questions about them."
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getMediaDocumentVersion: (...args: unknown[]) => mockVersion(...args),
    getWorkspaceSources: (...args: unknown[]) => mockWorkspaceSources(...args),
    listMediaDocumentVersions: (...args: unknown[]) =>
      mockMediaVersions(...args),
    getMediaDetails: (...args: unknown[]) =>
      (mockGetMediaDetails as (...inner: unknown[]) => unknown)(...args),
    getChatLorebookDiagnostics: vi.fn(async () => ({
      chat_id: "chat",
      total_turns_with_diagnostics: 0,
      turns: [],
      page: 1,
      size: 8
    }))
  }
}))

vi.mock("@/services/tldw-server", () => ({
  fetchChatModels: vi.fn(async () => [
    {
      id: "test-model",
      name: "Test Model",
      provider: "test"
    }
  ])
}))

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  return {
    ...actual,
    Slider: ({
      value,
      min,
      max,
      step,
      onChange
    }: {
      value?: number
      min?: number
      max?: number
      step?: number
      onChange?: (value: number) => void
    }) => (
      <input
        type="range"
        value={typeof value === "number" ? value : 0}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange?.(Number(event.target.value))}
      />
    ),
    Switch: ({
      checked,
      onChange,
      ...rest
    }: {
      checked?: boolean
      onChange?: (checked: boolean) => void
      [key: string]: unknown
    }) => (
      <input
        type="checkbox"
        role="checkbox"
        checked={Boolean(checked)}
        onChange={(event) => onChange?.(event.target.checked)}
        {...(rest as any)}
      />
    )
  }
})

function renderChatPane() {
  return render(
    <MemoryRouter>
      <ChatPane />
    </MemoryRouter>
  )
}

describe("ChatPane Stage 3 adaptive mode controls and settings", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    realSourcesStore = null
    Object.defineProperty(globalThis, "crypto", {
      value: webcrypto,
      configurable: true
    })

    connectionStoreState.state.phase = ConnectionPhase.CONNECTED
    connectionStoreState.state.isChecking = false
    connectionStoreState.state.lastError = null

    workspaceStoreState.workspaceId = "workspace-a"
    workspaceStoreState.workspaceChatReferenceId = "workspace-a"
    workspaceStoreState.sources = []
    workspaceStoreState.selectedSourceFolderIds = []
    workspaceStoreState.sourceFolders = []
    workspaceStoreState.sourceFolderMemberships = []
    workspaceStoreState.selectedSourceIds = []
    workspaceStoreState.getSelectedSources = () => []
    workspaceStoreState.getSelectedMediaIds = () => []
    workspaceStoreState.chatFocusTarget = null
    mockCaptureToCurrentNote.mockReset()
    mockCaptureHead.mockReset()
    mockVersion.mockReset()

    optionStoreState.ragTopK = 8
    optionStoreState.selectedModel = "test-model"
    optionStoreState.ragAdvancedOptions = {
      min_score: 0.2,
      enable_reranking: false
    }

    messageOptionState.messages = []
    messageOptionState.history = []
    messageOptionState.historyId = null
    messageOptionState.serverChatId = null
    messageOptionState.streaming = false
    messageOptionState.isProcessing = false
    mockOnSubmit.mockResolvedValue(undefined)
    mockGetMediaDetails.mockResolvedValue({
      content: {
        text: "Fallback full source text"
      }
    })

    mockGetWorkspaceChatSession.mockReturnValue(null)
  })

  it("blocks captured stale heads at the actual Ask boundary", async () => {
    const source = {
      id: "capture",
      mediaId: 101,
      title: "Article",
      type: "website" as const,
      status: "ready" as const,
      webCapture: {
        clipId: "clip",
        requestedUrl: "https://example.org",
        capturedAt: "2026-10-07T00:00:00Z",
        contentSha256: "hash",
        refreshOf: null,
        mediaId: 101,
        versionNumber: 9,
        versionUuid: "version"
      }
    }
    workspaceStoreState.sources = [source]
    workspaceStoreState.selectedSourceIds = [source.id]
    workspaceStoreState.getSelectedSources = () => [source]
    mockScope.mockResolvedValue({
      requestScope: {},
      scopeSignal: new AbortController().signal,
      scopeInvalidatedSignal: new AbortController().signal,
      release: vi.fn()
    })
    mockCaptureHead.mockRejectedValue(
      new WebCaptureNotCurrentError(
        "Snapshot changed outside refresh",
        "capture"
      )
    )
    const view = renderChatPane()
    fireEvent.change(screen.getByPlaceholderText("Ask about your sources..."), {
      target: { value: "Explain" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send" }))
    await waitFor(() =>
      expect(
        screen.getAllByText(/Snapshot changed outside refresh/).length
      ).toBeGreaterThan(0)
    )
    expect(mockOnSubmit).not.toHaveBeenCalled()
    expect(workspaceStoreState.setSourceStatusById).toHaveBeenCalledWith(
      source.id,
      "error",
      "Snapshot changed outside refresh",
      undefined,
      { statusReason: "capture_head_changed", retryEligible: false }
    )
    // The accepted refresh appends a new pin and preserves selected old evidence.
    const fresh = {
      ...source,
      id: "fresh",
      mediaId: 102,
      webCapture: {
        ...source.webCapture,
        clipId: "fresh-clip",
        mediaId: 102,
        refreshOf: "clip"
      }
    }
    workspaceStoreState.sources = [{ ...source, status: "error" }, fresh]
    workspaceStoreState.selectedSourceIds = [source.id, fresh.id]
    workspaceStoreState.getSelectedSources = () => workspaceStoreState.sources
    mockCaptureHead.mockImplementation(async (item) => {
      if (item.id === source.id)
        throw new WebCaptureNotCurrentError(
          "Snapshot changed outside refresh",
          "capture"
        )
    })
    view.unmount()
    renderChatPane()
    fireEvent.change(screen.getByPlaceholderText("Ask about your sources..."), {
      target: { value: "Ask fresh capture" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send" }))
    await waitFor(() => expect(mockOnSubmit).toHaveBeenCalledOnce())
    expect(mockSetRagMediaIds).toHaveBeenLastCalledWith([102])
    expect(workspaceStoreState.sources[0].webCapture).toEqual(source.webCapture)
  })

  const setupRealCapture = async () => {
    const originalText = "Original accepted snapshot text"
    const pin: WebArticleCapturePin = {
      clipId: "89ed590f-9d1a-4ce8-a738-6870a0df788c",
      requestedUrl: "https://example.org/article",
      capturedAt: "2026-10-07T00:00:00Z",
      contentSha256: await sha256Text(originalText),
      refreshOf: null,
      mediaId: 101,
      versionNumber: 1,
      versionUuid: "9263eb39-0737-4a33-84e7-c766a7b454f8"
    }
    const source: WorkspaceSource = {
      id: `web-clipper:${pin.clipId}`,
      mediaId: 101,
      title: "Captured article",
      type: "website",
      status: "ready",
      url: pin.requestedUrl,
      addedAt: new Date(pin.capturedAt),
      webCapture: pin
    }
    const originalVersion = {
      media_id: 101,
      version_number: 1,
      uuid: pin.versionUuid,
      content: originalText,
      safe_metadata: {
        source: "web_clipper",
        clip_type: "article",
        clip_id: pin.clipId,
        workspace_id: "workspace-a",
        source_url: pin.requestedUrl,
        capture_metadata: {
          web_capture_v1: {
            mode: "server_article",
            requested_url: pin.requestedUrl,
            captured_at: pin.capturedAt,
            content_sha256: pin.contentSha256,
            refresh_of: null
          }
        }
      }
    }
    const original: WorkspaceSource = {
      id: "retrieved-result",
      mediaId: 100,
      title: "Original result",
      type: "website",
      status: "ready",
      addedAt: new Date(pin.capturedAt),
      knowledgeQaEvidence: {
        importId: "import",
        threadId: null,
        snapshot: true,
        sources: [
          {
            originalId: "result",
            mediaId: null,
            title: "Original result",
            type: "website",
            excerpt: "Retrieved excerpt",
            sourceType: "web"
          }
        ]
      }
    }
    realSourcesStore = create<WorkspaceState>(
      (set, get) =>
        ({
          ...workspaceStoreState,
          ...createSourcesSlice(set, get),
          sources: [source, original],
          selectedSourceIds: [source.id]
        }) as unknown as WorkspaceState
    )
    const owner = new AbortController()
    mockScope.mockResolvedValue({
      requestScope: {},
      scopeSignal: new AbortController().signal,
      scopeInvalidatedSignal: owner.signal,
      release: vi.fn()
    })
    mockWorkspaceSources.mockResolvedValue([
      {
        id: source.id,
        workspace_id: "workspace-a",
        media_id: 101,
        url: pin.requestedUrl
      }
    ])
    mockMediaVersions.mockResolvedValue([originalVersion])
    mockGetMediaDetails.mockResolvedValue({
      media_id: 101,
      content: { text: originalText }
    })
    mockVersion.mockImplementation(async (mediaId, version) => {
      if (mediaId !== 101 || version !== 1) throw new Error("Wrong exact pin")
      return originalVersion
    })
    const actual = await vi.importActual<
      typeof import("@/utils/research-web-capture")
    >("@/utils/research-web-capture")
    mockCaptureHead.mockImplementation(actual.assertWebCaptureHeadCurrent)
    return {
      store: realSourcesStore,
      source,
      owner,
      originalText,
      originalVersion,
      original
    }
  }

  const sendQuestion = () => {
    fireEvent.change(screen.getByPlaceholderText("Ask about your sources..."), {
      target: { value: "Explain" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send" }))
  }

  it("refuses Ask with no cached refusal after update then newest-version deletion", async () => {
    const { store, source, originalVersion } = await setupRealCapture()
    // DELETE exposes the original active row again, while current Media/FTS remains changed.
    mockMediaVersions.mockResolvedValue([originalVersion])
    mockGetMediaDetails.mockResolvedValue({
      media_id: 101,
      content: { text: "Later changed current article" }
    })
    expect(store.getState().sources[0].status).not.toBe("error")
    renderChatPane()
    sendQuestion()
    await waitFor(() => expect(store.getState().selectedSourceIds).toEqual([]))
    expect(mockOnSubmit).not.toHaveBeenCalled()
    expect(
      screen.getAllByText(/Snapshot changed outside refresh/).length
    ).toBeGreaterThan(0)
    expect(store.getState().sources[0].statusDetails?.statusReason).toBe(
      "capture_head_changed"
    )
    expect(store.getState().sources[0].webCapture).toBe(source.webCapture)
  })

  it("keeps real-store stale refusal visible after authorized deselection", async () => {
    const { store, source } = await setupRealCapture()
    mockMediaVersions.mockResolvedValue([
      {
        media_id: 101,
        version_number: 2,
        uuid: "a5259cb8-86c6-4d24-b3a5-24ad28d3ee91"
      }
    ])
    renderChatPane()
    sendQuestion()
    await waitFor(() => expect(store.getState().selectedSourceIds).toEqual([]))
    expect(mockOnSubmit).not.toHaveBeenCalled()
    expect(
      screen.getAllByText(/Snapshot changed outside refresh/).length
    ).toBeGreaterThan(0)
    expect(store.getState().sources[0].webCapture).toBe(source.webCapture)
  })

  it("retains the exact old pin and exclusion after queued canonical readiness", async () => {
    const { store, source, originalText, original } = await setupRealCapture()
    mockMediaVersions.mockResolvedValue([
      {
        media_id: 101,
        version_number: 2,
        uuid: "a5259cb8-86c6-4d24-b3a5-24ad28d3ee91"
      }
    ])
    const stop = store.subscribe((state) => {
      if (state.sources[0]?.status === "error") {
        stop()
        queueMicrotask(() =>
          store.getState().setSourceStatusByMediaId(
            101,
            "processing",
            "Text search is available while vector indexing continues.",
            {
              metadata_ready: true,
              text_extracted: true,
              fts_ready: true,
              vector_ready: false,
              citation_ready: false,
              summary_ready: false,
              tool_accessible: true
            },
            {
              lifecycleState: "partially_queryable",
              sourceOfTruth: "workspace-status-projection"
            }
          )
        )
      }
    })
    const view = renderChatPane()
    sendQuestion()
    await waitFor(() => expect(mockMediaVersions).toHaveBeenCalledOnce())
    await act(async () => {})
    const stale = store.getState().sources[0]
    expect(mockOnSubmit).not.toHaveBeenCalled()
    expect(stale.statusMessage).toBe("Snapshot changed outside refresh")
    expect(stale.status).toBe("error")
    expect(store.getState().selectedSourceIds).toEqual([])
    expect(isWorkspaceSourceSelectable(stale)).toBe(false)
    act(() => store.getState().toggleSourceSelection(source.id))
    expect(store.getState().selectedSourceIds).toEqual([])
    expect(stale.webCapture).toBe(source.webCapture)
    expect(store.getState().sources[1]).toBe(original)
    const { tldwClient } = await import("@/services/tldw/TldwApiClient")
    const old = await tldwClient.getMediaDocumentVersion(
      101,
      stale.webCapture!.versionNumber
    )
    expect(old.content).toBe(originalText)
    expect(old.uuid).toBe(source.webCapture!.versionUuid)

    // A fresh recapture has its own source/pin; original snapshot remains evidence.
    const fresh = {
      ...source,
      id: "web-clipper:fresh",
      mediaId: 102,
      webCapture: {
        ...source.webCapture!,
        clipId: "fresh",
        mediaId: 102,
        refreshOf: source.webCapture!.clipId
      }
    }
    act(() => store.setState({ sources: [stale, original, fresh] }))
    act(() => store.getState().setSelectedSourceIds([source.id, fresh.id]))
    expect(store.getState().selectedSourceIds).toEqual([fresh.id])
    mockCaptureHead.mockResolvedValue(undefined)
    view.unmount()
    renderChatPane()
    sendQuestion()
    await waitFor(() => expect(mockOnSubmit).toHaveBeenCalledOnce())
    expect(mockSetRagMediaIds).toHaveBeenLastCalledWith([102])
    expect(store.getState().sources[0].webCapture).toBe(source.webCapture)
    expect(store.getState().sources[1].knowledgeQaEvidence).toBe(
      original.knowledgeQaEvidence
    )
  })

  it.each([
    { fullSource: false, priorChecks: 1 },
    { fullSource: true, priorChecks: 1 },
    { fullSource: true, priorChecks: 2 }
  ])(
    "retains refusal for a later head change (full=$fullSource, prior=$priorChecks)",
    async ({ fullSource, priorChecks }) => {
      const { store, originalVersion } = await setupRealCapture()
      mockMediaVersions.mockResolvedValue([
        {
          ...originalVersion,
          version_number: 2,
          uuid: "a5259cb8-86c6-4d24-b3a5-24ad28d3ee91"
        }
      ])
      for (let check = 0; check < priorChecks; check++) {
        mockMediaVersions.mockResolvedValueOnce([originalVersion])
      }
      renderChatPane()
      if (fullSource)
        fireEvent.click(
          screen.getByRole("checkbox", { name: "Include full source contents" })
        )
      sendQuestion()
      await waitFor(() =>
        expect(mockMediaVersions).toHaveBeenCalledTimes(priorChecks + 1)
      )
      await act(async () => {})
      expect(mockOnSubmit).not.toHaveBeenCalled()
      expect(store.getState().sources[0].statusMessage).toBe(
        "Snapshot changed outside refresh"
      )
      expect(store.getState().selectedSourceIds).toEqual([])
      expect(
        screen.getAllByText(/Snapshot changed outside refresh/).length
      ).toBeGreaterThan(0)
    }
  )

  it.each(["version", "digest", "transport"])(
    "preserves exact-read %s refusal classification",
    async (failure) => {
      const { store, originalVersion } = await setupRealCapture()
      if (failure === "transport")
        mockVersion.mockRejectedValue(
          new Error("Exact read unavailable. Retry.")
        )
      else
        mockVersion.mockResolvedValue({
          ...originalVersion,
          ...(failure === "version"
            ? { version_number: 2 }
            : { content: "Different bytes" })
        })
      renderChatPane()
      fireEvent.click(
        screen.getByRole("checkbox", { name: "Include full source contents" })
      )
      sendQuestion()
      await waitFor(() => expect(mockVersion).toHaveBeenCalledOnce())
      await act(async () => {})
      expect(mockOnSubmit).not.toHaveBeenCalled()
      if (failure === "transport") {
        expect(
          screen.getByText(/Exact read unavailable\. Retry\./)
        ).toBeInTheDocument()
        expect(isWorkspaceSourceSelectable(store.getState().sources[0])).toBe(
          true
        )
      } else {
        expect(store.getState().sources[0].statusMessage).toBe(
          "Snapshot changed outside refresh"
        )
        expect(store.getState().selectedSourceIds).toEqual([])
      }
    }
  )

  it.each(["selection", "workspace", "source", "owner"])(
    "real %s retirement prevents a late head failure from changing feedback",
    async (retirement) => {
      const { store, owner } = await setupRealCapture()
      let reject!: (reason: Error) => void
      mockMediaVersions.mockImplementation(
        () =>
          new Promise((_, fail) => {
            reject = fail
          })
      )
      renderChatPane()
      sendQuestion()
      await waitFor(() => expect(mockMediaVersions).toHaveBeenCalledOnce())
      await act(async () => {
        if (retirement === "selection") store.getState().deselectAllSources()
        if (retirement === "workspace")
          store.setState({ workspaceId: "workspace-b" })
        if (retirement === "source") store.setState({ sources: [] })
        if (retirement === "owner") owner.abort()
        reject(new Error("late head failure"))
      })
      expect(mockOnSubmit).not.toHaveBeenCalled()
      expect(screen.queryByText(/Snapshot changed outside refresh/)).toBeNull()
      expect(
        store.getState().sources.every((source) => source.status === "ready")
      ).toBe(true)
    }
  )

  it("transient head read failure leaves the capture selectable for a legitimate retry", async () => {
    const { store } = await setupRealCapture()
    mockMediaVersions.mockRejectedValueOnce(
      new Error("Service temporarily unavailable. Retry.")
    )
    renderChatPane()
    sendQuestion()
    await waitFor(() => expect(mockMediaVersions).toHaveBeenCalledOnce())
    await act(async () => {})
    expect(mockOnSubmit).not.toHaveBeenCalled()
    expect(store.getState().selectedSourceIds).toEqual([
      store.getState().sources[0].id
    ])
    expect(isWorkspaceSourceSelectable(store.getState().sources[0])).toBe(true)
    expect(
      screen.getByText(/Service temporarily unavailable\. Retry\./)
    ).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    await waitFor(() =>
      expect(
        screen.queryByText(/Service temporarily unavailable\. Retry\./)
      ).toBeNull()
    )
    sendQuestion()
    await waitFor(() => expect(mockOnSubmit).toHaveBeenCalledOnce())
  })

  it("generic readiness still recovers ordinary errors and unaffected captures", async () => {
    const { store, source } = await setupRealCapture()
    const ordinary = {
      ...source,
      id: "ordinary",
      mediaId: 103,
      webCapture: undefined
    }
    act(() => store.setState({ sources: [source, ordinary] }))
    act(() => {
      store
        .getState()
        .setSourceStatusById(
          ordinary.id,
          "error",
          "Snapshot changed outside refresh",
          undefined,
          { statusReason: "capture_head_changed" }
        )
      store.getState().setSourceStatusByMediaId(103, "ready")
      store
        .getState()
        .setSourceStatusById(source.id, "error", "Temporary read failure")
      store.getState().setSourceStatusByMediaId(101, "ready")
      store.getState().setSelectedSourceIds([source.id, ordinary.id])
    })
    expect(store.getState().getSelectedMediaIds()).toEqual([101, 103])
    renderChatPane()
    sendQuestion()
    await waitFor(() => expect(mockOnSubmit).toHaveBeenCalledOnce())
  })

  it("uses exact capture version text and rechecks before dispatch", async () => {
    const source = {
      id: "capture",
      mediaId: 101,
      title: "Article",
      type: "website" as const,
      status: "ready" as const,
      webCapture: {
        clipId: "clip",
        requestedUrl: "https://example.org",
        capturedAt: "2026-10-07T00:00:00Z",
        contentSha256: await sha256Text("exact captured body"),
        refreshOf: null,
        mediaId: 101,
        versionNumber: 9,
        versionUuid: "version"
      }
    }
    workspaceStoreState.sources = [source]
    workspaceStoreState.selectedSourceIds = [source.id]
    workspaceStoreState.getSelectedSources = () => [source]
    mockScope.mockResolvedValue({
      requestScope: {},
      scopeSignal: new AbortController().signal,
      scopeInvalidatedSignal: new AbortController().signal,
      release: vi.fn()
    })
    mockCaptureHead.mockResolvedValue(undefined)
    mockVersion.mockResolvedValue({
      uuid: "version",
      version_number: 9,
      content: "exact captured body"
    })
    renderChatPane()
    fireEvent.click(
      screen.getByRole("checkbox", { name: "Include full source contents" })
    )
    fireEvent.change(screen.getByPlaceholderText("Ask about your sources..."), {
      target: { value: "Explain" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send" }))
    await waitFor(() =>
      expect(mockOnSubmit).toHaveBeenCalledWith(
        expect.objectContaining({
          message: expect.stringContaining("exact captured body"),
          controller: expect.any(AbortController),
          assertCurrent: expect.any(Function)
        })
      )
    )
    expect(mockGetMediaDetails).not.toHaveBeenCalled()
    expect(mockVersion).toHaveBeenCalledWith(
      101,
      9,
      expect.objectContaining({ requestScope: {} })
    )
    expect(mockCaptureHead).toHaveBeenCalledTimes(3)
  })

  it("a source removed during head readback cannot dispatch", async () => {
    const source = {
      id: "capture",
      mediaId: 101,
      title: "Article",
      type: "website" as const,
      status: "ready" as const,
      webCapture: {
        clipId: "clip",
        requestedUrl: "https://example.org",
        capturedAt: "2026-10-07T00:00:00Z",
        contentSha256: "hash",
        refreshOf: null,
        mediaId: 101,
        versionNumber: 9,
        versionUuid: "version"
      }
    }
    workspaceStoreState.sources = [source]
    workspaceStoreState.selectedSourceIds = [source.id]
    workspaceStoreState.getSelectedSources = () => [source]
    mockScope.mockResolvedValue({
      requestScope: {},
      scopeSignal: new AbortController().signal,
      scopeInvalidatedSignal: new AbortController().signal,
      release: vi.fn()
    })
    mockCaptureHead.mockImplementation(async () => {
      workspaceStoreState.sources = []
    })
    renderChatPane()
    fireEvent.change(screen.getByPlaceholderText("Ask about your sources..."), {
      target: { value: "Explain" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send" }))
    await waitFor(() =>
      expect(
        screen.getAllByText(/Research source selection or account changed/)
          .length
      ).toBeGreaterThan(0)
    )
    expect(mockOnSubmit).not.toHaveBeenCalled()
  })

  it("a capture moved out of a selected folder during head readback cannot dispatch", async () => {
    const source = {
      id: "capture",
      mediaId: 101,
      title: "Article",
      type: "website" as const,
      status: "ready" as const,
      webCapture: {
        clipId: "clip",
        requestedUrl: "https://example.org",
        capturedAt: "2026-10-07T00:00:00Z",
        contentSha256: "hash",
        refreshOf: null,
        mediaId: 101,
        versionNumber: 9,
        versionUuid: "version"
      }
    }
    workspaceStoreState.sources = [source]
    workspaceStoreState.selectedSourceIds = []
    workspaceStoreState.selectedSourceFolderIds = ["folder"]
    workspaceStoreState.sourceFolders = [
      { id: "folder", name: "Selected", parentId: null }
    ]
    workspaceStoreState.sourceFolderMemberships = [
      { folderId: "folder", sourceId: source.id }
    ]
    workspaceStoreState.getSelectedSources = () => [source]
    mockScope.mockResolvedValue({
      requestScope: {},
      scopeSignal: new AbortController().signal,
      scopeInvalidatedSignal: new AbortController().signal,
      release: vi.fn()
    })
    mockCaptureHead.mockImplementation(async () => {
      workspaceStoreState.sourceFolderMemberships = []
    })
    renderChatPane()
    fireEvent.change(screen.getByPlaceholderText("Ask about your sources..."), {
      target: { value: "Explain" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send" }))
    await waitFor(() =>
      expect(
        screen.getAllByText(/Research source selection or account changed/)
          .length
      ).toBeGreaterThan(0)
    )
    expect(mockOnSubmit).not.toHaveBeenCalled()
  })

  it.each(["Regenerate capture test", "Edit and send capture test"])(
    "guards actual %s dispatch",
    async (action) => {
      const source = {
        id: "capture",
        mediaId: 101,
        title: "Article",
        type: "website" as const,
        status: "ready" as const,
        webCapture: {
          clipId: "clip",
          requestedUrl: "https://example.org",
          capturedAt: "2026-10-07T00:00:00Z",
          contentSha256: "hash",
          refreshOf: null,
          mediaId: 101,
          versionNumber: 9,
          versionUuid: "version"
        }
      }
      workspaceStoreState.sources = [source]
      workspaceStoreState.selectedSourceIds = [source.id]
      workspaceStoreState.getSelectedSources = () => [source]
      messageOptionState.messages = [
        {
          id: "answer",
          isBot: true,
          name: "Assistant",
          message: "Prior answer",
          sources: []
        }
      ]
      mockScope.mockResolvedValue({
        requestScope: {},
        scopeSignal: new AbortController().signal,
        scopeInvalidatedSignal: new AbortController().signal,
        release: vi.fn()
      })
      mockCaptureHead.mockRejectedValue(
        new WebCaptureNotCurrentError("changed", "capture")
      )
      renderChatPane()
      fireEvent.click(screen.getByRole("button", { name: action }))
      await waitFor(() =>
        expect(
          screen.getAllByText(/Snapshot changed outside refresh/).length
        ).toBeGreaterThan(0)
      )
      expect(mockRegenerateLastMessage).not.toHaveBeenCalled()
      expect(mockEditMessage).not.toHaveBeenCalled()
    }
  )

  it("Save to Notes adds only actually cited capture pins", () => {
    const pin = {
      clipId: "clip",
      requestedUrl: "https://example.org",
      capturedAt: "2026-10-07T00:00:00Z",
      contentSha256: "hash",
      refreshOf: null,
      mediaId: 101,
      versionNumber: 9,
      versionUuid: "version"
    }
    workspaceStoreState.sources = [
      {
        id: "capture",
        mediaId: 101,
        title: "Article",
        type: "website",
        webCapture: pin
      },
      {
        id: "uncited",
        mediaId: 102,
        title: "Uncited article",
        type: "website",
        webCapture: { ...pin, mediaId: 102 }
      }
    ]
    messageOptionState.messages = [
      {
        id: "answer",
        isBot: true,
        name: "Assistant",
        message: "Sourced answer",
        sources: [{ media_id: 101 }]
      }
    ]
    renderChatPane()
    fireEvent.click(screen.getByRole("button", { name: "Save to Notes" }))
    const saved = mockCaptureToCurrentNote.mock.calls[0][0]
    expect(saved.provenance.sources).toEqual([
      expect.objectContaining({
        mediaId: 101,
        originalVersion: 9,
        snapshotMediaId: 101
      })
    ])
  })

  it("refuses inline text whose digest differs from the pin even when version identity matches", async () => {
    const source = {
      id: "capture",
      mediaId: 101,
      title: "Article",
      type: "website" as const,
      status: "ready" as const,
      webCapture: {
        clipId: "clip",
        requestedUrl: "https://example.org",
        capturedAt: "2026-10-07T00:00:00Z",
        contentSha256: await sha256Text("accepted text"),
        refreshOf: null,
        mediaId: 101,
        versionNumber: 9,
        versionUuid: "version"
      }
    }
    workspaceStoreState.sources = [source]
    workspaceStoreState.selectedSourceIds = [source.id]
    workspaceStoreState.getSelectedSources = () => [source]
    mockScope.mockResolvedValue({
      requestScope: {},
      scopeSignal: new AbortController().signal,
      scopeInvalidatedSignal: new AbortController().signal,
      release: vi.fn()
    })
    mockCaptureHead.mockResolvedValue(undefined)
    mockVersion.mockResolvedValue({
      uuid: "version",
      version_number: 9,
      content: "different text"
    })
    renderChatPane()
    fireEvent.click(
      screen.getByRole("checkbox", { name: "Include full source contents" })
    )
    fireEvent.change(screen.getByPlaceholderText("Ask about your sources..."), {
      target: { value: "Explain" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send" }))
    await waitFor(() =>
      expect(
        screen.getAllByText(/Snapshot changed outside refresh/).length
      ).toBeGreaterThan(0)
    )
    expect(mockOnSubmit).not.toHaveBeenCalled()
  })

  it("renders the empty chat shell through the canonical EmptyState primitive", () => {
    const { container } = renderChatPane()

    expect(
      container.querySelector('[data-ds-component="EmptyState"]')
    ).toBeInTheDocument()
    expect(screen.getByText("Start your research")).toBeInTheDocument()
  })

  it("uses contextual source guidance instead of stale left-panel copy", () => {
    renderChatPane()

    const explainer = screen.getByTestId("workspace-chat-sources-explainer")

    expect(explainer).toHaveTextContent("Sources pane")
    expect(explainer.textContent?.toLowerCase()).not.toContain("left panel")
  })

  it("adapts empty-state examples based on selected source types", () => {
    workspaceStoreState.selectedSourceIds = ["source-video-1"]
    workspaceStoreState.getSelectedSources = () => [
      {
        id: "source-video-1",
        mediaId: 10,
        title: "Interview Recording",
        type: "video",
        status: "ready"
      }
    ]
    workspaceStoreState.getSelectedMediaIds = () => [10]

    renderChatPane()

    expect(
      screen.getByText("What was discussed around minute 12?")
    ).toBeInTheDocument()
  })

  it("clicking an empty-state prompt seeds the composer and keeps send enabled", async () => {
    workspaceStoreState.selectedSourceIds = ["source-video-1"]
    workspaceStoreState.getSelectedSources = () => [
      {
        id: "source-video-1",
        mediaId: 10,
        title: "Interview Recording",
        type: "video",
        status: "ready"
      }
    ]
    workspaceStoreState.getSelectedMediaIds = () => [10]

    renderChatPane()

    const prompt = "What was discussed around minute 12?"
    fireEvent.click(screen.getByRole("button", { name: prompt }))

    const textarea = screen.getByPlaceholderText(
      "Ask about your sources..."
    ) as HTMLTextAreaElement
    expect(textarea.value).toBe(prompt)

    const sendButton = screen.getByRole("button", { name: "Send" })
    expect(sendButton).not.toBeDisabled()

    fireEvent.click(sendButton)

    await waitFor(() => {
      expect(mockOnSubmit).toHaveBeenCalledWith({
        message: prompt,
        image: ""
      })
    })
  })

  it("prepends response style and length instructions only when presets are active", async () => {
    renderChatPane()

    fireEvent.change(screen.getByLabelText("Response style"), {
      target: { value: "explain" }
    })
    fireEvent.change(screen.getByLabelText("Answer length"), {
      target: { value: "brief" }
    })
    fireEvent.change(screen.getByLabelText("Chat message"), {
      target: { value: "What should I know?" }
    })
    fireEvent.keyDown(screen.getByLabelText("Chat message"), {
      key: "Enter"
    })

    await waitFor(() => expect(mockOnSubmit).toHaveBeenCalled())
    expect(mockOnSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        message: expect.stringContaining("Response preference:")
      })
    )
    expect(mockOnSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        message: expect.stringContaining("explain the answer")
      })
    )
    expect(mockOnSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        message: expect.stringContaining("Keep the answer brief")
      })
    )
    expect(mockOnSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        message: expect.stringContaining("User question: What should I know?")
      })
    )
  })

  it("groups mode and RAG controls inside one toolbar region", () => {
    workspaceStoreState.selectedSourceIds = ["source-doc-1"]
    workspaceStoreState.getSelectedSources = () => [
      {
        id: "source-doc-1",
        mediaId: 101,
        title: "Policy Document",
        type: "pdf",
        status: "ready"
      }
    ]
    workspaceStoreState.getSelectedMediaIds = () => [101]

    renderChatPane()

    const toolbar = screen.getByTestId("workspace-chat-controls-toolbar")
    expect(toolbar).toBeInTheDocument()
    expect(toolbar).toContainElement(
      screen.getByRole("button", { name: "General chat" })
    )
    expect(toolbar).toContainElement(
      screen.getByRole("button", { name: "Advanced RAG settings" })
    )
  })

  it("allows explicit general mode override even when sources are selected", async () => {
    workspaceStoreState.selectedSourceIds = ["source-doc-1"]
    workspaceStoreState.getSelectedSources = () => [
      {
        id: "source-doc-1",
        mediaId: 101,
        title: "Policy Document",
        type: "pdf",
        status: "ready"
      }
    ]
    workspaceStoreState.getSelectedMediaIds = () => [101]

    renderChatPane()

    await waitFor(() => {
      expect(mockSetRagMediaIds).toHaveBeenCalledWith([101])
      expect(mockSetChatMode).toHaveBeenCalledWith("rag")
      expect(mockSetFileRetrievalEnabled).toHaveBeenCalledWith(true)
    })

    fireEvent.click(screen.getByRole("button", { name: "General chat" }))

    await waitFor(() => {
      expect(mockSetRagMediaIds).toHaveBeenLastCalledWith(null)
      expect(mockSetChatMode).toHaveBeenLastCalledWith("normal")
      expect(mockSetFileRetrievalEnabled).toHaveBeenLastCalledWith(false)
    })

    expect(
      screen.getByText(
        "General chat mode is active. Selected sources will not be used unless RAG mode is enabled."
      )
    ).toBeInTheDocument()
  })

  it("updates advanced RAG settings from UI controls", async () => {
    workspaceStoreState.selectedSourceIds = ["source-doc-1"]
    workspaceStoreState.getSelectedSources = () => [
      {
        id: "source-doc-1",
        mediaId: 101,
        title: "Policy Document",
        type: "pdf",
        status: "ready"
      }
    ]
    workspaceStoreState.getSelectedMediaIds = () => [101]

    renderChatPane()

    fireEvent.click(
      screen.getByRole("button", { name: "Advanced RAG settings" })
    )

    const sliders = screen.getAllByRole("slider")
    fireEvent.change(sliders[0], { target: { value: "70" } })
    fireEvent.change(sliders[1], { target: { value: "0.55" } })

    expect(mockSetRagTopK).toHaveBeenCalledWith(50)
    expect(mockSetRagAdvancedOptions).toHaveBeenCalledWith(
      expect.objectContaining({ top_k: 50 })
    )
    expect(mockSetRagAdvancedOptions).toHaveBeenCalledWith(
      expect.objectContaining({ min_score: 0.55 })
    )

    fireEvent.click(screen.getByRole("checkbox", { name: "Enable reranking" }))
    expect(mockSetRagAdvancedOptions).toHaveBeenCalledWith(
      expect.objectContaining({ enable_reranking: true })
    )
  })

  it("injects complete selected source contents when enabled", async () => {
    workspaceStoreState.selectedSourceIds = ["source-doc-1", "source-doc-2"]
    workspaceStoreState.getSelectedSources = () => [
      {
        id: "source-doc-1",
        mediaId: 101,
        title: "Primary Paper",
        type: "pdf",
        status: "ready"
      },
      {
        id: "source-doc-2",
        mediaId: 102,
        title: "Appendix Notes",
        type: "document",
        status: "ready"
      }
    ]
    workspaceStoreState.getSelectedMediaIds = () => [101, 102]

    mockGetMediaDetails.mockImplementation(async (mediaId: number) => {
      if (mediaId === 101) {
        return {
          content: {
            text: "Primary paper full text block."
          }
        }
      }
      return {
        content: {
          text: "Appendix reference text block."
        }
      }
    })

    renderChatPane()

    fireEvent.click(
      screen.getByRole("checkbox", { name: "Include full source contents" })
    )

    const textarea = screen.getByPlaceholderText("Ask about your sources...")
    fireEvent.change(textarea, { target: { value: "What's the synopsis?" } })
    fireEvent.click(screen.getByRole("button", { name: "Send" }))

    await waitFor(() => {
      expect(mockOnSubmit).toHaveBeenCalledWith(
        expect.objectContaining({
          message: expect.stringContaining("Source 101: Primary Paper")
        })
      )
    })
    expect(mockOnSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        message: expect.stringContaining("Primary paper full text block.")
      })
    )
    expect(mockOnSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        message: expect.stringContaining("User question: What's the synopsis?")
      })
    )
  })

  it("does not duplicate question framing when full source contents and response presets are enabled", async () => {
    workspaceStoreState.selectedSourceIds = ["source-doc-1"]
    workspaceStoreState.getSelectedSources = () => [
      {
        id: "source-doc-1",
        mediaId: 101,
        title: "Primary Paper",
        type: "pdf",
        status: "ready"
      }
    ]
    workspaceStoreState.getSelectedMediaIds = () => [101]
    mockGetMediaDetails.mockResolvedValue({
      content: {
        text: "Primary paper full text block."
      }
    })

    renderChatPane()

    fireEvent.change(screen.getByLabelText("Response style"), {
      target: { value: "source-first" }
    })
    fireEvent.change(screen.getByLabelText("Answer length"), {
      target: { value: "detailed" }
    })
    fireEvent.click(
      screen.getByRole("checkbox", { name: "Include full source contents" })
    )
    fireEvent.change(screen.getByLabelText("Chat message"), {
      target: { value: "How do the findings connect?" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send" }))

    await waitFor(() => expect(mockOnSubmit).toHaveBeenCalled())
    const submittedMessage = mockOnSubmit.mock.calls.at(-1)?.[0].message

    expect(submittedMessage).toContain("Response preference:")
    expect(submittedMessage).toContain("Source 101: Primary Paper")
    expect(submittedMessage).toContain(
      "User question: How do the findings connect?"
    )
    expect(submittedMessage.match(/User question:/g)).toHaveLength(1)
  })

  it("shows keyboard shortcut hint below the composer", () => {
    renderChatPane()

    expect(
      screen.getByText(
        "Enter or Cmd/Ctrl+Enter to send, Shift+Enter for new line"
      )
    ).toBeInTheDocument()
  })

  it("submits discuss-artifact payloads from Studio into chat", async () => {
    renderChatPane()

    window.dispatchEvent(
      new CustomEvent("research-workspace:discuss-artifact", {
        detail: {
          artifactId: "artifact-123",
          artifactType: "summary",
          title: "Research Summary",
          content: "Point 1\\nPoint 2"
        }
      })
    )

    await waitFor(() => {
      expect(mockOnSubmit).toHaveBeenCalledWith(
        expect.objectContaining({
          message: expect.stringContaining("Research Summary")
        })
      )
    })
    expect(mockOnSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        message: expect.stringContaining("Point 1")
      })
    )
  })

  it("exposes save-to-notes action for both user and assistant messages", () => {
    messageOptionState.messages = [
      {
        id: "msg-user",
        isBot: false,
        name: "You",
        message: "User question",
        sources: []
      },
      {
        id: "msg-assistant",
        isBot: true,
        name: "Assistant",
        message: "Assistant answer",
        sources: []
      }
    ]

    renderChatPane()

    expect(
      screen.getAllByRole("button", { name: "Save to Notes" })
    ).toHaveLength(2)
  })

  it("saves chat message content into workspace note draft", () => {
    messageOptionState.messages = [
      {
        id: "msg-assistant",
        isBot: true,
        name: "Assistant",
        message: "Concise summary of findings.",
        sources: []
      }
    ]

    renderChatPane()

    fireEvent.click(screen.getByRole("button", { name: "Save to Notes" }))

    expect(mockCaptureToCurrentNote).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "append",
        content: "Concise summary of findings.",
        title: expect.stringContaining("Assistant:")
      })
    )
  })
})
