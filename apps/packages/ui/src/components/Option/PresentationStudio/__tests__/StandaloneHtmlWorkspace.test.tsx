import React from "react"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  online: true,
  runtimeExtension: false,
  navigate: vi.fn(),
  getConfig: vi.fn(),
  getCurrentUser: vi.fn(),
  getPresentation: vi.fn(),
  getPresentationMetadata: vi.fn(),
  getSlidesCapabilities: vi.fn(),
  saveStandaloneHtmlSource: vi.fn(),
  downloadStandaloneHtmlDraft: vi.fn(),
  listVisualStyles: vi.fn(),
  slidesCapabilities: null as any,
  usePrompt: vi.fn(),
  promptActive: false,
  editorProps: null as Record<string, any> | null,
  monacoEditorDispose: vi.fn(),
  monacoModelDispose: vi.fn(),
  outlineTerminate: vi.fn()
}))

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom")
  const ReactModule = await import("react")
  return {
    ...actual,
    useNavigate: () => mocks.navigate,
    useBlocker: () => ({ state: "unblocked", proceed: undefined, reset: undefined }),
    unstable_usePrompt: (...args: any[]) => {
      mocks.usePrompt(...args)
      ReactModule.useEffect(() => {
        mocks.promptActive = true
        return () => { mocks.promptActive = false }
      }, [])
    }
  }
})

vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: () => mocks.online }))
vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    loading: false,
    capabilities: { hasSlides: true, hasPresentationStudio: true, hasPresentationRender: true }
  })
}))
vi.mock("@/hooks/useSlidesCapabilities", () => ({
  useSlidesCapabilities: () => mocks.slidesCapabilities
}))
vi.mock("@/utils/browser-runtime", () => ({
  isExtensionRuntime: () => mocks.runtimeExtension,
  getBrowserRuntime: () => (mocks.runtimeExtension ? { id: "extension-test" } : null)
}))

vi.mock("@/services/tldw/TldwApiClient", async () => {
  const actual = await vi.importActual<Record<string, any>>("@/services/tldw/TldwApiClient")
  return {
    ...actual,
    buildPresentationVisualStyleSnapshot: (style: any) => ({ ...style }),
    clonePresentationVisualStyleSnapshot: (style: any) => (style ? { ...style } : null),
    tldwClient: {
      ...actual.tldwClient,
      getConfig: (...args: any[]) => mocks.getConfig(...args),
      getPresentation: (...args: any[]) => mocks.getPresentation(...args),
      getPresentationMetadata: (...args: any[]) => mocks.getPresentationMetadata(...args),
      getSlidesCapabilities: (...args: any[]) => mocks.getSlidesCapabilities(...args),
      saveStandaloneHtmlSource: (...args: any[]) => mocks.saveStandaloneHtmlSource(...args),
      downloadStandaloneHtmlDraft: (...args: any[]) => mocks.downloadStandaloneHtmlDraft(...args),
      listVisualStyles: (...args: any[]) => mocks.listVisualStyles(...args)
    }
  }
})

vi.mock("@/services/tldw/TldwAuth", async () => {
  const actual = await vi.importActual<Record<string, any>>("@/services/tldw/TldwAuth")
  return {
    ...actual,
    tldwAuth: { ...actual.tldwAuth, getCurrentUser: (...args: any[]) => mocks.getCurrentUser(...args) }
  }
})

vi.mock("@monaco-editor/react", async () => {
  const ReactModule = await import("react")
  const Monaco = (props: Record<string, any>) => {
    mocks.editorProps = props
    ReactModule.useEffect(() => {
      const model = { dispose: mocks.monacoModelDispose }
      props.onMount?.(
        {
          getModel: () => model,
          getDomNode: () => document.querySelector("[data-workspace-monaco]"),
          getValue: () => props.value,
          setValue: vi.fn(),
          updateOptions: vi.fn(),
          dispose: mocks.monacoEditorDispose
        },
        { languages: {} }
      )
    }, [props])
    return (
      <textarea
        data-workspace-monaco
        id={props.wrapperProps?.id}
        aria-label="HTML source"
        value={props.value}
        onChange={(event) => props.onChange?.(event.target.value)}
      />
    )
  }
  return { default: Monaco }
})

class ImmediateOutlineWorker {
  onmessage: ((event: MessageEvent<any>) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  terminate = mocks.outlineTerminate
  postMessage(request: any) {
    queueMicrotask(() => {
      this.onmessage?.({
        data: {
          type: "result",
          requestId: request.requestId,
          digest: request.digest,
          outline: { digest: request.digest, slides: [], truncated: false }
        }
      } as MessageEvent<any>)
    })
  }
}

const loadWorkspace = () =>
  vi.importActual<Record<string, any>>(["..", "StandaloneHtmlWorkspace"].join("/"))
const loadRecovery = () =>
  vi.importActual<Record<string, any>>(["..", "standalone-html-recovery"].join("/"))

const SOURCE = "<!doctype html><title>Deck</title>"
const SOURCE_DIGEST = "860887583dae29d0a221e3c9315a092fc6b271dd5d11cbe6e89be21a5260223d"
const EDITED = "<!doctype html><title>Edited</title>"
const EDITED_DIGEST = "21346e71978f06e2bdaf4b151a2c272c1b2b639212e11f5d5612c62a115298b8"
const RECOVERED = "<!doctype html><title>Recovered</title>"
const SECOND_EDIT = "<!doctype html><title>Second local edit</title>"
const SECOND_EDIT_DIGEST = "71777a47ce4d4473a7c68a686b27278eddc0c8c4ae21d1bcd82c05495e75d4f5"
const THIRD_EDIT = "<!doctype html><title>Third pending local edit</title>"
const EMPTY_DIGEST = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

const installSourceDigestSentinel = (sentinel: string) => {
  const observed: string[] = []
  const realDigest = crypto.subtle.digest.bind(crypto.subtle)
  vi.spyOn(crypto.subtle, "digest").mockImplementation((algorithm, data) => {
    const source = new TextDecoder().decode(data)
    if (source === sentinel) observed.push(source)
    return realDigest(algorithm, data)
  })
  return observed
}

const installThrowingSessionStorageGetter = () => {
  const descriptor = Object.getOwnPropertyDescriptor(window, "sessionStorage")
  Object.defineProperty(window, "sessionStorage", {
    configurable: true,
    get: () => {
      throw new DOMException("Storage access denied", "SecurityError")
    }
  })
  return () => {
    if (descriptor) Object.defineProperty(window, "sessionStorage", descriptor)
    else Reflect.deleteProperty(window, "sessionStorage")
  }
}

const detail = (overrides: Record<string, unknown> = {}) => ({
  record: {
    id: "html-1",
    content_kind: "standalone_html",
    title: "Deck",
    description: null,
    theme: "black",
    source_type: "prompt",
    source_ref: null,
    source_query: null,
    created_at: "2026-07-15T00:00:00Z",
    last_modified: "2026-07-15T00:00:01Z",
    deleted: false,
    client_id: "42",
    version: 7,
    html_document: SOURCE,
    html_sha256: SOURCE_DIGEST,
    html_bytes: 34,
    html_slide_count: 1,
    generation_provenance: { source_kind: "prompt", provider: "test", model: "model" },
    ...overrides
  },
  etag: '"v7"'
})

const readyCapabilities = {
  capabilities: {
    content_kinds: {
      standalone_html: {
        read: true,
        edit: true,
        export_attachment: true,
        draft_attachment: true,
        reason: null,
        limits: {
          max_document_bytes: 1_048_576,
          max_source_write_bytes: 1_048_576,
          max_draft_attachment_bytes: 1_048_576,
          max_slides: 30,
          max_nesting_depth: 128
        }
      }
    }
  },
  status: "ready",
  reason: null,
  canGenerate: true,
  canReadStandalone: true,
  canDraftStandalone: true,
  canEditStandalone: true,
  retry: vi.fn()
}

describe("StandaloneHtmlWorkspace", () => {
  beforeEach(() => {
    sessionStorage.clear()
    localStorage.clear()
    mocks.online = true
    mocks.runtimeExtension = false
    mocks.navigate.mockReset()
    mocks.getConfig.mockReset().mockResolvedValue({ serverUrl: "https://TLDW.Example/path" })
    mocks.getCurrentUser.mockReset().mockResolvedValue({ id: 42, username: "owner", is_active: true })
    mocks.getPresentation.mockReset().mockResolvedValue(detail())
    mocks.getPresentationMetadata.mockReset().mockResolvedValue({
      record: { id: "html-1", content_kind: "standalone_html" },
      etag: null
    })
    mocks.getSlidesCapabilities.mockReset().mockResolvedValue({})
    mocks.saveStandaloneHtmlSource.mockReset()
    mocks.downloadStandaloneHtmlDraft.mockReset().mockResolvedValue(new TextEncoder().encode(SOURCE))
    mocks.listVisualStyles.mockReset().mockResolvedValue([])
    mocks.slidesCapabilities = readyCapabilities
    mocks.usePrompt.mockReset()
    mocks.promptActive = false
    readyCapabilities.retry.mockReset()
    mocks.editorProps = null
    mocks.monacoEditorDispose.mockReset()
    mocks.monacoModelDispose.mockReset()
    mocks.outlineTerminate.mockReset()
    Object.defineProperty(globalThis, "Worker", {
      configurable: true,
      writable: true,
      value: ImmediateOutlineWorker
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("loads source only after trusted principal scope and keeps the workspace inert and source-local", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    const localWrite = vi.spyOn(Object.getPrototypeOf(sessionStorage), "setItem")
    const pushState = vi.spyOn(history, "pushState")
    const consoleLog = vi.spyOn(console, "log")

    render(<StandaloneHtmlWorkspace presentationId="html-1" />)

    expect(screen.getByText(/Confirming current server and account/i)).toBeVisible()
    const editor = await screen.findByLabelText("HTML source")
    expect(editor).toHaveValue(SOURCE)
    expect(mocks.getConfig).toHaveBeenCalled()
    expect(mocks.getCurrentUser).toHaveBeenCalled()
    expect(mocks.getPresentation).toHaveBeenCalledWith(
      "html-1",
      expect.objectContaining({ abortSignal: expect.any(AbortSignal) })
    )
    expect(screen.getByText("Safe outline: text only; code never runs in Studio")).toBeVisible()
    expect(screen.getByRole("button", { name: "Save" })).toHaveClass("min-h-[44px]")
    expect(screen.getByRole("button", { name: "Download current draft" })).toBeVisible()
    expect(screen.getByRole("button", { name: "Back to presentations" })).toBeVisible()
    const codeTab = screen.getByRole("tab", { name: "Code" })
    const outlineTab = screen.getByRole("tab", { name: "Outline" })
    expect(codeTab).toHaveAttribute("aria-selected", "true")
    expect(outlineTab).toHaveAttribute("aria-selected", "false")
    const codePanel = screen.getByRole("tabpanel", { name: "Code" })
    const outlinePanel = screen.getByRole("tabpanel", { name: "Outline" })
    expect(codeTab).toHaveAttribute("aria-controls", codePanel.id)
    expect(outlineTab).toHaveAttribute("aria-controls", outlinePanel.id)
    expect(screen.queryByTestId("presentation-studio-slide-rail")).not.toBeInTheDocument()
    expect(screen.queryByTestId("presentation-studio-media-rail")).not.toBeInTheDocument()
    expect(localStorage.length).toBe(0)
    expect(pushState).not.toHaveBeenCalled()
    expect(consoleLog).not.toHaveBeenCalledWith(expect.stringContaining(SOURCE))
    expect(localWrite.mock.calls.some(([, value]) => String(value).includes(SOURCE))).toBe(false)
  })

  it("moves and selects the mobile workspace tabs with left and right arrow keys", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)

    await screen.findByLabelText("HTML source")
    const codeTab = screen.getByRole("tab", { name: "Code" })
    const outlineTab = screen.getByRole("tab", { name: "Outline" })
    codeTab.focus()

    expect(codeTab).toHaveFocus()
    expect(codeTab).toHaveAttribute("tabindex", "0")
    expect(outlineTab).toHaveAttribute("tabindex", "-1")

    fireEvent.keyDown(codeTab, { key: "ArrowRight" })

    expect(outlineTab).toHaveFocus()
    expect(outlineTab).toHaveAttribute("aria-selected", "true")
    expect(outlineTab).toHaveAttribute("tabindex", "0")
    expect(codeTab).toHaveAttribute("tabindex", "-1")

    fireEvent.keyDown(outlineTab, { key: "ArrowLeft" })

    expect(codeTab).toHaveFocus()
    expect(codeTab).toHaveAttribute("aria-selected", "true")
  })

  it.each([
    ["a tag list", '"v7", "v8"'],
    ["concatenated weak syntax", '"v7"W/"v8"'],
    ["an internal quote", '"v7"tail"'],
    ["a C0 control", '"v7\u001f"'],
    ["a DEL control", '"v7\u007f"'],
    ["a non-byte code point", '"v7\u0100"'],
    ["the wildcard", "*"],
    ["a weak entity-tag", 'W/"v7"'],
    ["surrounding whitespace", ' "v7"']
  ])("rejects %s as the initial strong ETag authority", async (_case, etag) => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    mocks.getPresentation.mockResolvedValueOnce({ ...detail(), etag })

    render(<StandaloneHtmlWorkspace presentationId="html-1" />)

    expect(
      await screen.findByText("This standalone HTML presentation could not be loaded safely.")
    ).toBeVisible()
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
  })

  it.each(['""', '"v7"', '"!,#~\\"', `"${String.fromCharCode(0x80, 0xff)}"`])(
    "accepts one anchored strong entity-tag with legal etagc bytes: %s",
    async (etag) => {
      const { StandaloneHtmlWorkspace } = await loadWorkspace()
      mocks.getPresentation.mockResolvedValueOnce({ ...detail(), etag })

      render(<StandaloneHtmlWorkspace presentationId="html-1" />)

      expect(await screen.findByLabelText("HTML source")).toHaveValue(SOURCE)
    }
  )

  it("marks the retained outline stale immediately while a preflight-valid candidate digest is pending", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    const editor = await screen.findByLabelText("HTML source")
    const outlineRegion = screen
      .getByRole("heading", { name: /Safe outline: text only/i })
      .closest("section") as HTMLElement
    await waitFor(() => expect(within(outlineRegion).getByRole("status")).toHaveTextContent("Current"))
    const digest = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(EDITED)
    )
    let resolveDigest: ((value: ArrayBuffer) => void) | null = null
    vi.spyOn(crypto.subtle, "digest").mockReturnValueOnce(
      new Promise<ArrayBuffer>((resolve) => { resolveDigest = resolve })
    )

    fireEvent.change(editor, { target: { value: EDITED } })

    await waitFor(() => expect(editor).toHaveValue(EDITED))
    expect(within(outlineRegion).getByRole("status")).toHaveTextContent("Stale")
    resolveDigest?.(digest)
    await waitFor(() => expect(within(outlineRegion).getByRole("status")).toHaveTextContent("Current"))
  })

  it("loads source after React StrictMode replays mount effects", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    let resolveDetail: ((value: any) => void) | null = null
    mocks.getPresentation.mockReturnValue(
      new Promise((resolve) => {
        resolveDetail = resolve
      })
    )

    render(
      <React.StrictMode>
        <StandaloneHtmlWorkspace presentationId="html-1" />
      </React.StrictMode>
    )

    await waitFor(() => expect(mocks.getPresentation).toHaveBeenCalledTimes(1))
    await act(async () => {
      resolveDetail?.(detail())
      await Promise.resolve()
    })
    expect(await screen.findByLabelText("HTML source")).toHaveValue(SOURCE)
    expect(mocks.getPresentation).toHaveBeenCalledWith(
      "html-1",
      expect.objectContaining({ abortSignal: expect.any(AbortSignal) })
    )
  })

  it("does not start a detail request when the parent authority ref advances before passive loading", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    const authorityFence = vi.fn(
      (capturedEpoch: number | null, presentationId: string) =>
        capturedEpoch === 7 && presentationId === "html-1"
    )

    const Parent = () => {
      const authorityEpochRef = React.useRef(7)
      React.useLayoutEffect(() => {
        authorityEpochRef.current = 8
      }, [])
      const isKindAuthorityCurrent = React.useCallback(
        (capturedEpoch: number | null, presentationId: string) => {
          authorityFence(capturedEpoch, presentationId)
          return capturedEpoch === authorityEpochRef.current && presentationId === "html-1"
        },
        []
      )
      return (
        <StandaloneHtmlWorkspace
          presentationId="html-1"
          kindAuthorityEpoch={7}
          isKindAuthorityCurrent={isKindAuthorityCurrent}
        />
      )
    }

    render(<Parent />)

    await waitFor(() => {
      expect(mocks.getConfig).toHaveBeenCalled()
      expect(mocks.getCurrentUser).toHaveBeenCalled()
      expect(authorityFence).toHaveBeenCalledWith(7, "html-1")
    })
    expect(mocks.getPresentation).not.toHaveBeenCalled()
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
  })

  it("preserves verified server source and warns when recovery getItem is unavailable", async () => {
    vi.spyOn(Object.getPrototypeOf(sessionStorage), "getItem").mockImplementation(() => {
      throw new DOMException("Storage access denied", "SecurityError")
    })
    const { StandaloneHtmlWorkspace } = await loadWorkspace()

    render(<StandaloneHtmlWorkspace presentationId="html-1" />)

    expect(await screen.findByLabelText("HTML source")).toHaveValue(SOURCE)
    const warning = screen.getByText(/Recovery unavailable/i)
    expect(warning).toBeVisible()
    expect(warning.textContent?.length).toBeLessThanOrEqual(100)
    expect(screen.queryByText(/could not be loaded safely/i)).not.toBeInTheDocument()
  })

  it("preserves verified server source when the sessionStorage getter throws", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    const restoreStorage = installThrowingSessionStorageGetter()

    try {
      render(<StandaloneHtmlWorkspace presentationId="html-1" />)

      expect(await screen.findByLabelText("HTML source")).toHaveValue(SOURCE)
      const warning = screen.getByText(/Recovery unavailable/i)
      expect(warning).toBeVisible()
      expect(warning.textContent?.length).toBeLessThanOrEqual(100)
      expect(screen.queryByText(/could not be loaded safely/i)).not.toBeInTheDocument()
    } finally {
      restoreStorage()
    }
  })

  it("keeps an accepted in-memory edit and warns when recovery storage becomes inaccessible", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    await screen.findByLabelText("HTML source")
    const restoreStorage = installThrowingSessionStorageGetter()

    try {
      act(() => mocks.editorProps?.onChange?.(EDITED))

      expect(await screen.findByLabelText("HTML source")).toHaveValue(EDITED)
      const warning = await screen.findByText(/Recovery unavailable/i)
      expect(warning).toBeVisible()
      expect(warning.textContent?.length).toBeLessThanOrEqual(100)
      expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Not saved")
    } finally {
      restoreStorage()
    }
  })

  it("resolves a same-scope read warning after a confirmed save clears recovery", async () => {
    const storageRead = vi.spyOn(Object.getPrototypeOf(sessionStorage), "getItem")
      .mockImplementationOnce(() => {
        throw new DOMException("Storage access denied", "SecurityError")
      })
    mocks.saveStandaloneHtmlSource.mockResolvedValueOnce({
      record: detail({
        title: "Edited",
        version: 8,
        html_document: EDITED,
        html_sha256: EDITED_DIGEST,
        html_bytes: 36
      }).record,
      etag: '"v8"'
    })
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)

    await screen.findByLabelText("HTML source")
    expect(screen.getByText(/Recovery unavailable/i)).toBeVisible()
    storageRead.mockRestore()
    fireEvent.change(screen.getByLabelText("HTML source"), { target: { value: EDITED } })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() =>
      expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Saved")
    )
    expect(screen.queryByText(/Recovery unavailable/i)).not.toBeInTheDocument()
  })

  it("clears the exact same-scope read marker after a later successful recovery read", async () => {
    vi.spyOn(Object.getPrototypeOf(sessionStorage), "getItem")
      .mockImplementationOnce(() => {
        throw new DOMException("Storage access denied", "SecurityError")
      })
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    const view = render(<StandaloneHtmlWorkspace presentationId="html-1" />)

    await screen.findByLabelText("HTML source")
    expect(screen.getByText(/Recovery unavailable/i)).toBeVisible()
    mocks.slidesCapabilities = {
      ...readyCapabilities,
      canReadStandalone: false,
      canDraftStandalone: false,
      canEditStandalone: false
    }
    view.rerender(<StandaloneHtmlWorkspace presentationId="html-1" />)
    await waitFor(() => expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument())
    mocks.slidesCapabilities = readyCapabilities
    view.rerender(<StandaloneHtmlWorkspace presentationId="html-1" />)

    expect(await screen.findByLabelText("HTML source")).toHaveValue(SOURCE)
    expect(screen.queryByText(/Recovery unavailable/i)).not.toBeInTheDocument()
  })

  it("resolves a same-scope write warning after a confirmed save clears recovery", async () => {
    mocks.saveStandaloneHtmlSource.mockResolvedValueOnce({
      record: detail({
        title: "Edited",
        version: 8,
        html_document: EDITED,
        html_sha256: EDITED_DIGEST,
        html_bytes: 36
      }).record,
      etag: '"v8"'
    })
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    const editor = await screen.findByLabelText("HTML source")
    const storageWrite = vi.spyOn(Object.getPrototypeOf(sessionStorage), "setItem")
      .mockImplementation(() => {
        throw new DOMException("quota", "QuotaExceededError")
      })

    fireEvent.change(editor, { target: { value: EDITED } })
    expect(await screen.findByText(/Recovery unavailable/i)).toBeVisible()
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    storageWrite.mockRestore()
    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() =>
      expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Saved")
    )
    expect(screen.queryByText(/Recovery unavailable/i)).not.toBeInTheDocument()
  })

  it.each([
    ["loading", "Checking standalone HTML access"],
    ["error", "Standalone HTML access could not be confirmed"],
    ["auth_required", "Current standalone HTML access requires authentication"],
    ["forbidden", "This account cannot read standalone HTML presentations"]
  ])("makes zero detail or recovery reads while capability status is %s", async (status, guardText) => {
    const retry = vi.fn()
    mocks.slidesCapabilities = {
      ...readyCapabilities,
      status,
      canReadStandalone: status === "error",
      retry
    }
    const storageRead = vi.spyOn(Object.getPrototypeOf(sessionStorage), "getItem")
    const { StandaloneHtmlWorkspace } = await loadWorkspace()

    render(<StandaloneHtmlWorkspace presentationId="html-1" />)

    expect(await screen.findByText(new RegExp(guardText, "i"))).toBeVisible()
    expect(mocks.getPresentation).not.toHaveBeenCalled()
    expect(storageRead).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    expect(retry).toHaveBeenCalledTimes(1)
  })

  it("keeps unsupported read authority source-free and retries capability discovery", async () => {
    const retry = vi.fn()
    mocks.slidesCapabilities = {
      ...readyCapabilities,
      status: "ready",
      canReadStandalone: false,
      canDraftStandalone: false,
      canEditStandalone: false,
      retry
    }
    const storageRead = vi.spyOn(Object.getPrototypeOf(sessionStorage), "getItem")
    const { StandaloneHtmlWorkspace } = await loadWorkspace()

    render(<StandaloneHtmlWorkspace presentationId="html-1" />)

    expect(await screen.findByText(/does not support reading standalone HTML presentations/i)).toBeVisible()
    expect(mocks.getPresentation).not.toHaveBeenCalled()
    expect(storageRead).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    expect(retry).toHaveBeenCalledTimes(1)
  })

  it("enforces edit authority in the mounted editor and inside the shared download handler", async () => {
    mocks.slidesCapabilities = {
      ...readyCapabilities,
      canEditStandalone: false,
      canDraftStandalone: false
    }
    const { StandaloneHtmlWorkspace } = await loadWorkspace()

    render(<StandaloneHtmlWorkspace presentationId="html-1" />)

    const editor = await screen.findByLabelText("HTML source")
    await waitFor(() => expect(mocks.editorProps?.options?.readOnly).toBe(true))
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()
    const download = screen.getByRole("button", { name: "Download current draft" }) as HTMLButtonElement
    expect(download).toBeDisabled()
    fireEvent.change(editor, { target: { value: EDITED } })
    await act(async () => Promise.resolve())
    expect(editor).toHaveValue(SOURCE)
    download.disabled = false
    fireEvent.click(download)
    expect(mocks.downloadStandaloneHtmlDraft).not.toHaveBeenCalled()
    expect(editor).toHaveValue(SOURCE)
  })

  it("enforces draft authority on current, recovered, and conflict download actions", async () => {
    const recovery = await loadRecovery()
    const source = await vi.importActual<Record<string, any>>(["..", "standalone-html-source"].join("/"))
    const recovered = await source.validateStandaloneHtmlSource(RECOVERED)
    recovery.writeStandaloneHtmlRecovery(
      sessionStorage,
      recovery.createPresentationPrincipalScope("https://tldw.example", "42"),
      {
        presentationId: "html-1",
        baseEtag: '"v6"',
        baseDigest: "f".repeat(64),
        acceptedSource: recovered,
        updatedAt: Date.now()
      }
    )
    const conflict = Object.assign(new Error("source-free conflict"), { status: 412 })
    mocks.saveStandaloneHtmlSource.mockRejectedValueOnce(conflict)
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    const view = render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    fireEvent.change(await screen.findByLabelText("HTML source"), { target: { value: EDITED } })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Conflict"))

    mocks.slidesCapabilities = { ...readyCapabilities, canDraftStandalone: false }
    view.rerender(<StandaloneHtmlWorkspace presentationId="html-1" />)

    for (const name of [
      "Download current draft",
      "Download recovered draft",
      "Download my draft"
    ]) {
      const button = screen.getByRole("button", { name }) as HTMLButtonElement
      expect(button).toBeDisabled()
      button.disabled = false
      fireEvent.click(button)
    }
    expect(mocks.downloadStandaloneHtmlDraft).not.toHaveBeenCalled()
  })

  it("uses an explicit strong-ETag save, never autosaves, and announces dirty/saving/saved states", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    let resolveSave: ((value: any) => void) | null = null
    mocks.saveStandaloneHtmlSource.mockReturnValue(
      new Promise((resolve) => {
        resolveSave = resolve
      })
    )

    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    const editor = await screen.findByLabelText("HTML source")
    fireEvent.change(editor, { target: { value: EDITED } })

    await waitFor(() =>
      expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Not saved")
    )
    await new Promise((resolve) => setTimeout(resolve, 25))
    expect(mocks.saveStandaloneHtmlSource).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    expect(await screen.findByText("Saving")).toBeVisible()
    expect(mocks.saveStandaloneHtmlSource).toHaveBeenCalledWith(
      "html-1",
      EDITED,
      expect.objectContaining({ ifMatch: '"v7"', abortSignal: expect.any(AbortSignal) })
    )

    resolveSave?.({
      record: detail({
        title: "Edited",
        version: 8,
        html_document: EDITED,
        html_sha256: EDITED_DIGEST,
        html_bytes: 36
      }).record,
      etag: '"v8"'
    })
    expect(await screen.findByText("Saved")).toBeVisible()
    expect(screen.getByRole("heading", { name: "Edited" })).toBeVisible()
  })

  it("rejects an invalid strong-ETag shape in a save response before rebasing the draft", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    mocks.saveStandaloneHtmlSource.mockResolvedValueOnce({
      record: detail({
        title: "Untrusted save title",
        version: 8,
        html_document: EDITED,
        html_sha256: EDITED_DIGEST,
        html_bytes: 36
      }).record,
      etag: '"v8", W/"v9"'
    })

    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    fireEvent.change(await screen.findByLabelText("HTML source"), { target: { value: EDITED } })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => expect(mocks.getPresentation).toHaveBeenCalledTimes(2))
    expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Not saved")
    expect(screen.getByRole("heading", { name: "Deck" })).toBeVisible()
    expect(screen.queryByRole("heading", { name: "Untrusted save title" })).not.toBeInTheDocument()
  })

  it("adopts the saved A title, ETag, and base while preserving a newer visible B draft", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    let resolveFirstSave: ((value: any) => void) | null = null
    mocks.saveStandaloneHtmlSource
      .mockReturnValueOnce(new Promise((resolve) => { resolveFirstSave = resolve }))
      .mockResolvedValueOnce({
        record: detail({
          title: "Second saved title",
          version: 9,
          html_document: SECOND_EDIT,
          html_sha256: SECOND_EDIT_DIGEST,
          html_bytes: 47
        }).record,
        etag: '"v9"'
      })

    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    const editor = await screen.findByLabelText("HTML source")
    fireEvent.change(editor, { target: { value: EDITED } })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(screen.getByText("Saving")).toBeVisible())
    fireEvent.change(editor, { target: { value: SECOND_EDIT } })
    await waitFor(() => expect(editor).toHaveValue(SECOND_EDIT))

    resolveFirstSave?.({
      record: detail({
        title: "Server title for saved A",
        version: 8,
        html_document: EDITED,
        html_sha256: EDITED_DIGEST,
        html_bytes: 36
      }).record,
      etag: '"v8"'
    })

    expect(await screen.findByRole("heading", { name: "Server title for saved A" })).toBeVisible()
    expect(editor).toHaveValue(SECOND_EDIT)
    expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Not saved")
    const stored = JSON.parse(sessionStorage.getItem(sessionStorage.key(0)!)!)
    expect(stored).toEqual(expect.objectContaining({
      baseEtag: '"v8"',
      baseDigest: EDITED_DIGEST,
      source: SECOND_EDIT
    }))

    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(mocks.saveStandaloneHtmlSource).toHaveBeenLastCalledWith(
      "html-1",
      SECOND_EDIT,
      expect.objectContaining({ ifMatch: '"v8"' })
    ))
  })

  it("keeps pending candidate C available to pagehide while saved A rebases accepted B", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    let resolveSave: ((value: any) => void) | null = null
    mocks.saveStandaloneHtmlSource.mockReturnValueOnce(
      new Promise((resolve) => { resolveSave = resolve })
    )

    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    const editor = await screen.findByLabelText("HTML source")
    fireEvent.change(editor, { target: { value: EDITED } })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(screen.getByText("Saving")).toBeVisible())
    fireEvent.change(editor, { target: { value: SECOND_EDIT } })
    await waitFor(() => expect(editor).toHaveValue(SECOND_EDIT))

    const pendingDigest = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(THIRD_EDIT)
    )
    const savedDigest = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(EDITED)
    )
    let resolvePendingDigest: ((value: ArrayBuffer) => void) | null = null
    vi.spyOn(crypto.subtle, "digest")
      .mockReturnValueOnce(new Promise<ArrayBuffer>((resolve) => { resolvePendingDigest = resolve }))
      .mockResolvedValueOnce(savedDigest)

    act(() => mocks.editorProps?.onChange?.(THIRD_EDIT))
    resolveSave?.({
      record: detail({
        title: "Server title for saved A with pending C",
        version: 8,
        html_document: EDITED,
        html_sha256: EDITED_DIGEST,
        html_bytes: 36
      }).record,
      etag: '"v8"'
    })

    expect(
      await screen.findByRole("heading", { name: "Server title for saved A with pending C" })
    ).toBeVisible()
    act(() => window.dispatchEvent(new PageTransitionEvent("pagehide", { persisted: true })))

    const stored = JSON.parse(sessionStorage.getItem(sessionStorage.key(0)!)!)
    expect(stored).toEqual(expect.objectContaining({
      baseEtag: '"v8"',
      baseDigest: EDITED_DIGEST,
      source: THIRD_EDIT
    }))
    resolvePendingDigest?.(pendingDigest)
    await act(async () => Promise.resolve())
  })

  it("keeps a pending empty candidate authoritative when saved A returns before its digest", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    let resolveSave: ((value: any) => void) | null = null
    mocks.saveStandaloneHtmlSource.mockReturnValueOnce(
      new Promise((resolve) => { resolveSave = resolve })
    )
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    const editor = await screen.findByLabelText("HTML source")
    await waitFor(() => expect(mocks.editorProps).not.toBeNull())
    act(() => mocks.editorProps?.onChange?.(EDITED))
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(screen.getByText("Saving")).toBeVisible())
    let resolveEmptyDigest: ((value: ArrayBuffer) => void) | null = null
    vi.spyOn(crypto.subtle, "digest").mockReturnValueOnce(
      new Promise<ArrayBuffer>((resolve) => { resolveEmptyDigest = resolve })
    )

    act(() => mocks.editorProps?.onChange?.(""))
    resolveSave?.({
      record: detail({
        title: "Saved A while empty is pending",
        version: 8,
        html_document: EDITED,
        html_sha256: EDITED_DIGEST,
        html_bytes: 36
      }).record,
      etag: '"v8"'
    })

    expect(await screen.findByRole("heading", { name: "Saved A while empty is pending" })).toBeVisible()
    expect(editor).toHaveValue("")
    expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Not saved")
    act(() => window.dispatchEvent(new PageTransitionEvent("pagehide", { persisted: true })))
    const stored = JSON.parse(sessionStorage.getItem(sessionStorage.key(0)!)!)
    expect(stored).toEqual(expect.objectContaining({
      baseEtag: '"v8"',
      baseDigest: EDITED_DIGEST,
      source: ""
    }))

    resolveEmptyDigest?.(new ArrayBuffer(32))
    await act(async () => Promise.resolve())
  })

  it("reconciles saved A after an ambiguous response without replacing a newer B draft", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    let resolveReconciliation: ((value: any) => void) | null = null
    mocks.saveStandaloneHtmlSource
      .mockRejectedValueOnce(new TypeError("lost response"))
      .mockResolvedValueOnce({
        record: detail({
          title: "Second saved title",
          version: 9,
          html_document: SECOND_EDIT,
          html_sha256: SECOND_EDIT_DIGEST,
          html_bytes: 47
        }).record,
        etag: '"v9"'
      })
    mocks.getPresentation
      .mockResolvedValueOnce(detail())
      .mockReturnValueOnce(new Promise((resolve) => { resolveReconciliation = resolve }))

    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    const editor = await screen.findByLabelText("HTML source")
    fireEvent.change(editor, { target: { value: EDITED } })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(mocks.getPresentation).toHaveBeenCalledTimes(2))
    fireEvent.change(editor, { target: { value: SECOND_EDIT } })
    await waitFor(() => expect(editor).toHaveValue(SECOND_EDIT))

    resolveReconciliation?.({
      ...detail({
        title: "Reconciled server title for A",
        version: 8,
        html_document: EDITED,
        html_sha256: EDITED_DIGEST,
        html_bytes: 36
      }),
      etag: '"reconciled-v8"'
    })

    expect(await screen.findByRole("heading", { name: "Reconciled server title for A" })).toBeVisible()
    expect(editor).toHaveValue(SECOND_EDIT)
    expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Not saved")
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(mocks.saveStandaloneHtmlSource).toHaveBeenLastCalledWith(
      "html-1",
      SECOND_EDIT,
      expect.objectContaining({ ifMatch: '"reconciled-v8"' })
    ))
  })

  it("rejects an invalid strong ETag during ambiguous-response reconciliation", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    mocks.saveStandaloneHtmlSource.mockRejectedValueOnce(new TypeError("lost response"))
    mocks.getPresentation
      .mockResolvedValueOnce(detail())
      .mockResolvedValueOnce({
        ...detail({
          title: "Untrusted reconciliation title",
          version: 8,
          html_document: EDITED,
          html_sha256: EDITED_DIGEST,
          html_bytes: 36
        }),
        etag: '"v8"W/"tail"'
      })

    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    fireEvent.change(await screen.findByLabelText("HTML source"), { target: { value: EDITED } })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => expect(mocks.getPresentation).toHaveBeenCalledTimes(2))
    expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Not saved")
    expect(screen.getByRole("heading", { name: "Deck" })).toBeVisible()
    expect(screen.queryByRole("heading", { name: "Untrusted reconciliation title" })).not.toBeInTheDocument()
  })

  it("preserves the local draft and exposes all three explicit choices after a 412, including a second overwrite race", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    const conflict = Object.assign(new Error("source-free conflict"), {
      status: 412,
      details: { error_code: "presentation_version_conflict" }
    })
    mocks.saveStandaloneHtmlSource.mockRejectedValue(conflict)
    mocks.getPresentation
      .mockResolvedValueOnce(detail())
      .mockResolvedValueOnce({ ...detail({ version: 8 }), etag: '"v8"' })

    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    fireEvent.change(await screen.findByLabelText("HTML source"), { target: { value: EDITED } })
    await waitFor(() =>
      expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Not saved")
    )
    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() =>
      expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Conflict")
    )
    expect(screen.getByLabelText("HTML source")).toHaveValue(EDITED)
    expect(screen.getByRole("button", { name: "Discard my changes and load server version" })).toBeVisible()
    expect(screen.getByRole("button", { name: "Overwrite server with my draft" })).toBeVisible()
    expect(screen.getByRole("button", { name: "Download my draft" })).toBeVisible()

    fireEvent.click(screen.getByRole("button", { name: "Overwrite server with my draft" }))
    await waitFor(() => expect(mocks.getPresentation).toHaveBeenCalledTimes(2))
    expect(await screen.findByText(/Confirm replacing the current server version/i)).toBeVisible()
    fireEvent.click(screen.getByRole("button", { name: "Confirm overwrite" }))

    await waitFor(() =>
      expect(mocks.saveStandaloneHtmlSource).toHaveBeenLastCalledWith(
        "html-1",
        EDITED,
        expect.objectContaining({ ifMatch: '"v8"' })
      )
    )
    await waitFor(() =>
      expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Conflict")
    )
    expect(screen.getByLabelText("HTML source")).toHaveValue(EDITED)
  })

  it.each(["overwrite-preflight-first", "discard-refresh-first"] as const)(
    "lets the newer discard action win when %s resolves first",
    async (responseOrder) => {
      const { StandaloneHtmlWorkspace } = await loadWorkspace()
      const conflict = Object.assign(new Error("source-free conflict"), { status: 412 })
      let resolveOverwritePreflight: ((value: any) => void) | null = null
      let resolveDiscardRefresh: ((value: any) => void) | null = null
      mocks.saveStandaloneHtmlSource.mockRejectedValueOnce(conflict)
      mocks.getPresentation
        .mockResolvedValueOnce(detail())
        .mockReturnValueOnce(new Promise((resolve) => { resolveOverwritePreflight = resolve }))
        .mockReturnValueOnce(new Promise((resolve) => { resolveDiscardRefresh = resolve }))

      render(<StandaloneHtmlWorkspace presentationId="html-1" />)
      fireEvent.change(await screen.findByLabelText("HTML source"), { target: { value: EDITED } })
      await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
      fireEvent.click(screen.getByRole("button", { name: "Save" }))
      await screen.findByRole("region", { name: "Save conflict" })

      fireEvent.click(screen.getByRole("button", {
        name: "Discard my changes and load server version"
      }))
      const confirmDiscard = await screen.findByRole("button", {
        name: "Confirm discard and load server version"
      })
      fireEvent.click(screen.getByRole("button", { name: "Overwrite server with my draft" }))
      await waitFor(() => expect(mocks.getPresentation).toHaveBeenCalledTimes(2))
      fireEvent.click(confirmDiscard)
      await waitFor(() => expect(mocks.getPresentation).toHaveBeenCalledTimes(3))
      const overwriteSignal = mocks.getPresentation.mock.calls[1][1].abortSignal as AbortSignal
      const discardSignal = mocks.getPresentation.mock.calls[2][1].abortSignal as AbortSignal
      let staleOverwriteConfirmationPublished = false
      const overwriteFresh = { ...detail({ title: "Overwrite preflight", version: 8 }), etag: '"v8"' }
      const discardFresh = { ...detail({ title: "Discard winner", version: 9 }), etag: '"v9"' }

      if (responseOrder === "overwrite-preflight-first") {
        await act(async () => {
          resolveOverwritePreflight?.(overwriteFresh)
          await Promise.resolve()
        })
        staleOverwriteConfirmationPublished = Boolean(
          screen.queryByRole("button", { name: "Confirm overwrite" })
        )
        await act(async () => {
          resolveDiscardRefresh?.(discardFresh)
          await Promise.resolve()
        })
      } else {
        await act(async () => {
          resolveDiscardRefresh?.(discardFresh)
          await Promise.resolve()
        })
        expect(await screen.findByRole("heading", { name: "Discard winner" })).toBeVisible()
        await act(async () => {
          resolveOverwritePreflight?.(overwriteFresh)
          await Promise.resolve()
        })
      }

      expect(await screen.findByRole("heading", { name: "Discard winner" })).toBeVisible()
      expect(overwriteSignal.aborted).toBe(true)
      expect(discardSignal.aborted).toBe(false)
      expect(staleOverwriteConfirmationPublished).toBe(false)
      expect(screen.queryByRole("button", { name: "Confirm overwrite" })).not.toBeInTheDocument()
      expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Saved")
      expect(screen.getByLabelText("HTML source")).toHaveValue(SOURCE)
    }
  )

  it.each(["overwrite-save-first", "discard-refresh-first"] as const)(
    "lets the newer overwrite action win when %s resolves first",
    async (responseOrder) => {
      const { StandaloneHtmlWorkspace } = await loadWorkspace()
      const conflict = Object.assign(new Error("source-free conflict"), { status: 412 })
      let resolveOverwriteSave: ((value: any) => void) | null = null
      let resolveDiscardRefresh: ((value: any) => void) | null = null
      mocks.saveStandaloneHtmlSource
        .mockRejectedValueOnce(conflict)
        .mockReturnValueOnce(new Promise((resolve) => { resolveOverwriteSave = resolve }))
      mocks.getPresentation
        .mockResolvedValueOnce(detail())
        .mockResolvedValueOnce({ ...detail({ title: "Overwrite preflight", version: 8 }), etag: '"v8"' })
        .mockReturnValueOnce(new Promise((resolve) => { resolveDiscardRefresh = resolve }))

      render(<StandaloneHtmlWorkspace presentationId="html-1" />)
      fireEvent.change(await screen.findByLabelText("HTML source"), { target: { value: EDITED } })
      await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
      fireEvent.click(screen.getByRole("button", { name: "Save" }))
      await screen.findByRole("region", { name: "Save conflict" })

      fireEvent.click(screen.getByRole("button", {
        name: "Discard my changes and load server version"
      }))
      const confirmDiscard = await screen.findByRole("button", {
        name: "Confirm discard and load server version"
      })
      fireEvent.click(screen.getByRole("button", { name: "Overwrite server with my draft" }))
      const confirmOverwrite = await screen.findByRole("button", { name: "Confirm overwrite" })
      fireEvent.click(confirmDiscard)
      await waitFor(() => expect(mocks.getPresentation).toHaveBeenCalledTimes(3))
      const discardSignal = mocks.getPresentation.mock.calls[2][1].abortSignal as AbortSignal
      fireEvent.click(confirmOverwrite)
      await waitFor(() => expect(mocks.saveStandaloneHtmlSource).toHaveBeenCalledTimes(2))
      const overwriteSignal = mocks.saveStandaloneHtmlSource.mock.calls[1][2].abortSignal as AbortSignal
      const overwriteSaved = {
        record: detail({
          title: "Overwrite winner",
          version: 9,
          html_document: EDITED,
          html_sha256: EDITED_DIGEST,
          html_bytes: 36
        }).record,
        etag: '"v9"'
      }
      const discardFresh = { ...detail({ title: "Discard loser", version: 9 }), etag: '"v9"' }

      if (responseOrder === "overwrite-save-first") {
        await act(async () => {
          resolveOverwriteSave?.(overwriteSaved)
          await Promise.resolve()
        })
        expect(await screen.findByRole("heading", { name: "Overwrite winner" })).toBeVisible()
        await act(async () => {
          resolveDiscardRefresh?.(discardFresh)
          await Promise.resolve()
        })
      } else {
        await act(async () => {
          resolveDiscardRefresh?.(discardFresh)
          await Promise.resolve()
        })
        await act(async () => {
          resolveOverwriteSave?.(overwriteSaved)
          await Promise.resolve()
        })
      }

      expect(await screen.findByRole("heading", { name: "Overwrite winner" })).toBeVisible()
      expect(discardSignal.aborted).toBe(true)
      expect(overwriteSignal.aborted).toBe(false)
      expect(screen.queryByRole("heading", { name: "Discard loser" })).not.toBeInTheDocument()
      expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Saved")
      expect(screen.getByLabelText("HTML source")).toHaveValue(EDITED)
    }
  )

  it("keeps conflict choices authoritative when a newer candidate finishes validation after the save returns 412", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    const conflict = Object.assign(new Error("source-free conflict"), { status: 412 })
    let rejectSave: ((reason: unknown) => void) | null = null
    mocks.saveStandaloneHtmlSource.mockReturnValueOnce(
      new Promise((_resolve, reject) => { rejectSave = reject })
    )
    const secondDigest = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(SECOND_EDIT)
    )

    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    const editor = await screen.findByLabelText("HTML source")
    fireEvent.change(editor, { target: { value: EDITED } })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await screen.findByText("Saving")

    let resolveSecondDigest: ((value: ArrayBuffer) => void) | null = null
    vi.spyOn(crypto.subtle, "digest").mockReturnValueOnce(
      new Promise<ArrayBuffer>((resolve) => { resolveSecondDigest = resolve })
    )
    fireEvent.change(editor, { target: { value: SECOND_EDIT } })
    await waitFor(() => expect(editor).toHaveValue(SECOND_EDIT))

    rejectSave?.(conflict)
    await waitFor(() =>
      expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Conflict")
    )
    resolveSecondDigest?.(secondDigest)

    await waitFor(() => expect(editor).toHaveValue(SECOND_EDIT))
    expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Conflict")
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Overwrite server with my draft" })).toBeVisible()
    expect(screen.getByRole("button", { name: "Discard my changes and load server version" })).toBeVisible()
    expect(screen.getByRole("button", { name: "Download my draft" })).toBeVisible()
    expect(sessionStorage.length).toBe(1)
  })

  it("reconciles an ambiguous overwrite response when the exact saved digest is on the server", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    const conflict = Object.assign(new Error("source-free conflict"), { status: 412 })
    mocks.saveStandaloneHtmlSource
      .mockRejectedValueOnce(conflict)
      .mockRejectedValueOnce(new TypeError("overwrite response lost"))
    mocks.getPresentation
      .mockResolvedValueOnce(detail())
      .mockResolvedValueOnce({ ...detail({ version: 8 }), etag: '"v8"' })
      .mockResolvedValueOnce({
        ...detail({
          title: "Reconciled overwrite",
          version: 9,
          html_document: EDITED,
          html_sha256: EDITED_DIGEST,
          html_bytes: 36
        }),
        etag: '"v9"'
      })

    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    fireEvent.change(await screen.findByLabelText("HTML source"), { target: { value: EDITED } })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await screen.findByRole("region", { name: "Save conflict" })
    fireEvent.click(screen.getByRole("button", { name: "Overwrite server with my draft" }))
    fireEvent.click(await screen.findByRole("button", { name: "Confirm overwrite" }))

    expect(await screen.findByRole("heading", { name: "Reconciled overwrite" })).toBeVisible()
    expect(mocks.getPresentation).toHaveBeenCalledTimes(3)
    expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Saved")
    expect(screen.getByLabelText("HTML source")).toHaveValue(EDITED)
  })

  it("preserves the local draft and recovery when overwrite reconciliation finds a different digest", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    const conflict = Object.assign(new Error("source-free conflict"), { status: 412 })
    mocks.saveStandaloneHtmlSource
      .mockRejectedValueOnce(conflict)
      .mockRejectedValueOnce(new TypeError("overwrite response lost"))
    mocks.getPresentation
      .mockResolvedValueOnce(detail())
      .mockResolvedValueOnce({ ...detail({ version: 8 }), etag: '"v8"' })
      .mockResolvedValueOnce({ ...detail({ title: "Different server source", version: 9 }), etag: '"v9"' })

    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    fireEvent.change(await screen.findByLabelText("HTML source"), { target: { value: EDITED } })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await screen.findByRole("region", { name: "Save conflict" })
    fireEvent.click(screen.getByRole("button", { name: "Overwrite server with my draft" }))
    fireEvent.click(await screen.findByRole("button", { name: "Confirm overwrite" }))

    await waitFor(() => expect(mocks.getPresentation).toHaveBeenCalledTimes(3))
    expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Not saved")
    expect(screen.getByLabelText("HTML source")).toHaveValue(EDITED)
    expect(
      Array.from({ length: sessionStorage.length }, (_, index) =>
        sessionStorage.getItem(sessionStorage.key(index)!)
      ).join("\n")
    ).toContain(EDITED)
  })

  it("does not process a late overwrite reconciliation response after scope mismatch", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    const conflict = Object.assign(new Error("source-free conflict"), { status: 412 })
    let resolveReconciliation: ((value: any) => void) | null = null
    mocks.saveStandaloneHtmlSource
      .mockRejectedValueOnce(conflict)
      .mockRejectedValueOnce(new TypeError("overwrite response lost"))
    mocks.getPresentation
      .mockResolvedValueOnce(detail())
      .mockResolvedValueOnce({ ...detail({ version: 8 }), etag: '"v8"' })
      .mockReturnValueOnce(new Promise((resolve) => { resolveReconciliation = resolve }))

    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    fireEvent.change(await screen.findByLabelText("HTML source"), { target: { value: EDITED } })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await screen.findByRole("region", { name: "Save conflict" })
    fireEvent.click(screen.getByRole("button", { name: "Overwrite server with my draft" }))
    fireEvent.click(await screen.findByRole("button", { name: "Confirm overwrite" }))
    await waitFor(() => expect(mocks.getPresentation).toHaveBeenCalledTimes(3))
    const sentinel = "<!doctype html><title>stale overwrite reconciliation sentinel</title>"
    const observed = installSourceDigestSentinel(sentinel)

    act(() => window.dispatchEvent(new CustomEvent("tldw:slides-scope-mismatch")))
    resolveReconciliation?.(detail({
      html_document: sentinel,
      html_sha256: "0".repeat(64),
      html_bytes: new TextEncoder().encode(sentinel).byteLength
    }))
    await act(async () => Promise.resolve())

    expect(observed).toEqual([])
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
  })

  it("fetches a fresh ETag before confirmation and preserves edits during refresh and after confirmation", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    const conflict = Object.assign(new Error("source-free conflict"), { status: 412 })
    let resolveFresh: ((value: any) => void) | null = null
    let resolveOverwrite: ((value: any) => void) | null = null
    mocks.saveStandaloneHtmlSource
      .mockRejectedValueOnce(conflict)
      .mockReturnValueOnce(new Promise((resolve) => { resolveOverwrite = resolve }))
    mocks.getPresentation
      .mockResolvedValueOnce(detail())
      .mockReturnValueOnce(new Promise((resolve) => { resolveFresh = resolve }))

    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    const editor = await screen.findByLabelText("HTML source")
    fireEvent.change(editor, { target: { value: EDITED } })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Conflict"))
    fireEvent.click(screen.getByRole("button", { name: "Overwrite server with my draft" }))
    await waitFor(() => expect(mocks.getPresentation).toHaveBeenCalledTimes(2))
    expect(screen.queryByRole("button", { name: "Confirm overwrite" })).not.toBeInTheDocument()

    fireEvent.change(editor, { target: { value: SECOND_EDIT } })
    await waitFor(() => expect(editor).toHaveValue(SECOND_EDIT))
    resolveFresh?.({ ...detail({ version: 8 }), etag: '"v8"' })
    fireEvent.click(await screen.findByRole("button", { name: "Confirm overwrite" }))

    await waitFor(() => expect(mocks.saveStandaloneHtmlSource).toHaveBeenLastCalledWith(
      "html-1",
      SECOND_EDIT,
      expect.objectContaining({ ifMatch: '"v8"' })
    ))
    fireEvent.change(editor, { target: { value: THIRD_EDIT } })
    await waitFor(() => expect(editor).toHaveValue(THIRD_EDIT))
    resolveOverwrite?.({
      record: detail({
        title: "Overwrite saved B",
        version: 9,
        html_document: SECOND_EDIT,
        html_sha256: SECOND_EDIT_DIGEST,
        html_bytes: 47
      }).record,
      etag: '"v9"'
    })

    expect(await screen.findByRole("heading", { name: "Overwrite saved B" })).toBeVisible()
    expect(editor).toHaveValue(THIRD_EDIT)
    expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Not saved")
  })

  it("rejects an invalid strong ETag from the fresh overwrite read before confirmation", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    const conflict = Object.assign(new Error("source-free conflict"), { status: 412 })
    mocks.saveStandaloneHtmlSource.mockRejectedValueOnce(conflict)
    mocks.getPresentation
      .mockResolvedValueOnce(detail())
      .mockResolvedValueOnce({ ...detail({ version: 8 }), etag: '"v8" "v9"' })

    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    fireEvent.change(await screen.findByLabelText("HTML source"), { target: { value: EDITED } })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await screen.findByRole("region", { name: "Save conflict" })
    fireEvent.click(screen.getByRole("button", { name: "Overwrite server with my draft" }))

    expect(
      await screen.findByText("The current server version could not be verified. Your draft is preserved.")
    ).toBeVisible()
    expect(screen.queryByRole("button", { name: "Confirm overwrite" })).not.toBeInTheDocument()
    expect(screen.getByLabelText("HTML source")).toHaveValue(EDITED)
  })

  it("retains post-confirm edits and Conflict when a captured overwrite loses a second race", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    const conflict = Object.assign(new Error("source-free conflict"), { status: 412 })
    let resolveFresh: ((value: any) => void) | null = null
    let rejectOverwrite: ((reason: unknown) => void) | null = null
    mocks.saveStandaloneHtmlSource
      .mockRejectedValueOnce(conflict)
      .mockReturnValueOnce(new Promise((_resolve, reject) => { rejectOverwrite = reject }))
    mocks.getPresentation
      .mockResolvedValueOnce(detail())
      .mockReturnValueOnce(new Promise((resolve) => { resolveFresh = resolve }))

    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    const editor = await screen.findByLabelText("HTML source")
    fireEvent.change(editor, { target: { value: EDITED } })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Conflict"))
    fireEvent.click(screen.getByRole("button", { name: "Overwrite server with my draft" }))
    await waitFor(() => expect(mocks.getPresentation).toHaveBeenCalledTimes(2))
    resolveFresh?.({ ...detail({ version: 8 }), etag: '"v8"' })
    fireEvent.click(await screen.findByRole("button", { name: "Confirm overwrite" }))

    await waitFor(() => expect(mocks.saveStandaloneHtmlSource).toHaveBeenLastCalledWith(
      "html-1",
      EDITED,
      expect.objectContaining({ ifMatch: '"v8"' })
    ))
    fireEvent.change(editor, { target: { value: SECOND_EDIT } })
    await waitFor(() => expect(editor).toHaveValue(SECOND_EDIT))
    rejectOverwrite?.(conflict)
    await waitFor(() => expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Conflict"))
    expect(editor).toHaveValue(SECOND_EDIT)
  })

  it("cannot save, download, or confirm overwrite from an older accepted source while a newer valid candidate is pending", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    const conflict = Object.assign(new Error("source-free conflict"), { status: 412 })
    mocks.saveStandaloneHtmlSource.mockRejectedValueOnce(conflict)
    mocks.getPresentation
      .mockResolvedValueOnce(detail())
      .mockResolvedValueOnce({ ...detail({ version: 8 }), etag: '"v8"' })

    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    const editor = await screen.findByLabelText("HTML source")
    fireEvent.change(editor, { target: { value: EDITED } })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Conflict"))
    fireEvent.click(screen.getByRole("button", { name: "Overwrite server with my draft" }))
    const confirm = (await screen.findByRole("button", {
      name: "Confirm overwrite"
    })) as HTMLButtonElement

    const digest = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(THIRD_EDIT)
    )
    let resolveDigest: ((value: ArrayBuffer) => void) | null = null
    vi.spyOn(crypto.subtle, "digest").mockReturnValueOnce(
      new Promise<ArrayBuffer>((resolve) => { resolveDigest = resolve })
    )
    act(() => mocks.editorProps?.onChange?.(THIRD_EDIT))

    const save = screen.getByRole("button", { name: "Save" }) as HTMLButtonElement
    const currentDownload = screen.getByRole("button", {
      name: "Download current draft"
    }) as HTMLButtonElement
    const conflictDownload = screen.getByRole("button", {
      name: "Download my draft"
    }) as HTMLButtonElement
    await waitFor(() => {
      expect(save).toBeDisabled()
      expect(currentDownload).toBeDisabled()
      expect(conflictDownload).toBeDisabled()
      expect(confirm).toBeDisabled()
    })

    act(() => {
      for (const button of [currentDownload, conflictDownload, save, confirm]) {
        button.disabled = false
        button.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }))
      }
    })
    await act(async () => Promise.resolve())
    expect(mocks.downloadStandaloneHtmlDraft).not.toHaveBeenCalled()
    expect(mocks.saveStandaloneHtmlSource).toHaveBeenCalledTimes(1)

    act(() => mocks.editorProps?.onChange?.("invalid\u0000candidate"))

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Save" })).toBeEnabled()
      expect(screen.getByRole("button", { name: "Download current draft" })).toBeEnabled()
      expect(screen.getByRole("button", { name: "Download my draft" })).toBeEnabled()
      expect(screen.getByRole("button", { name: "Confirm overwrite" })).toBeEnabled()
    })
    resolveDigest?.(digest)
    await act(async () => Promise.resolve())
    expect(editor).toHaveValue(EDITED)
    expect(mocks.saveStandaloneHtmlSource).toHaveBeenCalledTimes(1)
  })

  it("does not discard edits made after discard confirmation while its fresh-detail GET is pending", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    const conflict = Object.assign(new Error("source-free conflict"), { status: 412 })
    let resolveFresh: ((value: any) => void) | null = null
    mocks.saveStandaloneHtmlSource.mockRejectedValueOnce(conflict)
    mocks.getPresentation
      .mockResolvedValueOnce(detail())
      .mockReturnValueOnce(new Promise((resolve) => { resolveFresh = resolve }))

    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    const editor = await screen.findByLabelText("HTML source")
    fireEvent.change(editor, { target: { value: EDITED } })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Conflict"))
    fireEvent.click(screen.getByRole("button", { name: "Discard my changes and load server version" }))
    fireEvent.click(await screen.findByRole("button", { name: "Confirm discard and load server version" }))
    await waitFor(() => expect(mocks.getPresentation).toHaveBeenCalledTimes(2))
    fireEvent.change(editor, { target: { value: SECOND_EDIT } })
    await waitFor(() => expect(editor).toHaveValue(SECOND_EDIT))

    resolveFresh?.({ ...detail({ title: "Fresh server", version: 8 }), etag: '"v8"' })

    await waitFor(() => expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Conflict"))
    expect(editor).toHaveValue(SECOND_EDIT)
    expect(screen.queryByRole("heading", { name: "Fresh server" })).not.toBeInTheDocument()
  })

  it("does not discard a preflight-valid edit whose digest is pending during the fresh-detail GET", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    const conflict = Object.assign(new Error("source-free conflict"), { status: 412 })
    let resolveFresh: ((value: any) => void) | null = null
    mocks.saveStandaloneHtmlSource.mockRejectedValueOnce(conflict)
    mocks.getPresentation
      .mockResolvedValueOnce(detail())
      .mockReturnValueOnce(new Promise((resolve) => { resolveFresh = resolve }))

    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    const editor = await screen.findByLabelText("HTML source")
    fireEvent.change(editor, { target: { value: EDITED } })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Conflict"))
    fireEvent.click(screen.getByRole("button", { name: "Discard my changes and load server version" }))
    fireEvent.click(await screen.findByRole("button", { name: "Confirm discard and load server version" }))
    await waitFor(() => expect(mocks.getPresentation).toHaveBeenCalledTimes(2))

    const pendingDigest = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(THIRD_EDIT)
    )
    let resolvePendingDigest: ((value: ArrayBuffer) => void) | null = null
    vi.spyOn(crypto.subtle, "digest").mockReturnValueOnce(
      new Promise<ArrayBuffer>((resolve) => { resolvePendingDigest = resolve })
    )
    act(() => mocks.editorProps?.onChange?.(THIRD_EDIT))

    resolveFresh?.({ ...detail({ title: "Fresh server", version: 8 }), etag: '"v8"' })
    expect(
      await screen.findByText(/draft changed while the server version was loading/i)
    ).toBeVisible()
    expect(screen.queryByRole("heading", { name: "Fresh server" })).not.toBeInTheDocument()

    resolvePendingDigest?.(pendingDigest)
    await waitFor(() => expect(editor).toHaveValue(THIRD_EDIT))
    expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Conflict")
    expect(screen.getByRole("button", { name: "Overwrite server with my draft" })).toBeVisible()
  })

  it("does not digest a discard refresh response returned after a scope mismatch", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    const conflict = Object.assign(new Error("source-free conflict"), { status: 412 })
    let resolveFresh: ((value: any) => void) | null = null
    mocks.saveStandaloneHtmlSource.mockRejectedValueOnce(conflict)
    mocks.getPresentation
      .mockResolvedValueOnce(detail())
      .mockReturnValueOnce(new Promise((resolve) => { resolveFresh = resolve }))

    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    fireEvent.change(await screen.findByLabelText("HTML source"), { target: { value: EDITED } })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() =>
      expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Conflict")
    )
    fireEvent.click(screen.getByRole("button", { name: "Discard my changes and load server version" }))
    fireEvent.click(await screen.findByRole("button", { name: "Confirm discard and load server version" }))
    await waitFor(() => expect(mocks.getPresentation).toHaveBeenCalledTimes(2))
    const sentinel = "<!doctype html><title>stale discard refresh sentinel</title>"
    const observed = installSourceDigestSentinel(sentinel)

    act(() => window.dispatchEvent(new CustomEvent("tldw:slides-scope-mismatch")))
    resolveFresh?.(detail({
      html_document: sentinel,
      html_sha256: "0".repeat(64),
      html_bytes: new TextEncoder().encode(sentinel).byteLength
    }))
    await act(async () => Promise.resolve())

    expect(observed).toEqual([])
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
    expect(document.body.textContent).not.toContain(sentinel)
  })

  it("aborts an in-flight conflict refresh before logout can process returned source", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    const conflict = Object.assign(new Error("source-free conflict"), { status: 412 })
    let resolveRefresh: ((value: any) => void) | null = null
    mocks.saveStandaloneHtmlSource.mockRejectedValueOnce(conflict)
    mocks.getPresentation
      .mockResolvedValueOnce(detail())
      .mockReturnValueOnce(
        new Promise((resolve) => {
          resolveRefresh = resolve
        })
      )

    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    fireEvent.change(await screen.findByLabelText("HTML source"), { target: { value: EDITED } })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() =>
      expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Conflict")
    )
    fireEvent.click(screen.getByRole("button", { name: "Overwrite server with my draft" }))
    await waitFor(() => expect(mocks.getPresentation).toHaveBeenCalledTimes(2))
    const refreshSignal = mocks.getPresentation.mock.calls[1][1].abortSignal as AbortSignal
    const sentinel = "<!doctype html><title>stale overwrite refresh sentinel</title>"
    const observed = installSourceDigestSentinel(sentinel)

    act(() =>
      window.dispatchEvent(
        new CustomEvent("tldw:auth-principal-changed", { detail: { kind: "logout" } })
      )
    )

    expect(refreshSignal.aborted).toBe(true)
    expect(document.body.textContent).not.toContain(EDITED)
    resolveRefresh?.(detail({
      html_document: sentinel,
      html_sha256: "0".repeat(64),
      html_bytes: new TextEncoder().encode(sentinel).byteLength
    }))
    await act(async () => Promise.resolve())
    expect(observed).toEqual([])
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
  })

  it("synchronously scrubs loaded source, recovery, editor, and outline on a slides scope mismatch", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    fireEvent.change(await screen.findByLabelText("HTML source"), { target: { value: EDITED } })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    expect(sessionStorage.length).toBe(1)

    act(() => {
      window.dispatchEvent(new CustomEvent("tldw:slides-scope-mismatch"))
      expect(mocks.monacoModelDispose).toHaveBeenCalled()
      expect(mocks.monacoEditorDispose).toHaveBeenCalled()
      expect(mocks.outlineTerminate).toHaveBeenCalled()
      expect(sessionStorage.length).toBe(0)
    })

    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
    expect(document.body.textContent).not.toContain(EDITED)
    expect(screen.getByText(/Current server and account could not be confirmed/i)).toBeVisible()
  })

  it("removes recovery source from aggregate refs before a mismatch callback returns", async () => {
    const recovery = await loadRecovery()
    const source = await vi.importActual<Record<string, any>>(
      ["..", "standalone-html-source"].join("/")
    )
    const recovered = await source.validateStandaloneHtmlSource(RECOVERED)
    recovery.writeStandaloneHtmlRecovery(
      sessionStorage,
      recovery.createPresentationPrincipalScope("https://tldw.example", "42"),
      {
        presentationId: "html-1",
        baseEtag: '"v6"',
        baseDigest: "f".repeat(64),
        acceptedSource: recovered,
        updatedAt: Date.now()
      }
    )
    const realUseRef = React.useRef
    let aggregateRef: React.MutableRefObject<Record<string, any>> | null = null
    vi.spyOn(React, "useRef").mockImplementation(((initialValue: unknown) => {
      const ref = realUseRef(initialValue) as React.MutableRefObject<any>
      if (
        initialValue &&
        typeof initialValue === "object" &&
        Object.keys(initialValue as Record<string, unknown>).sort().join(",") ===
          "message,recovery,saveStatus,title"
      ) {
        aggregateRef = ref
      }
      return ref
    }) as any)
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    await screen.findByRole("region", { name: "Recovered draft" })
    expect(JSON.stringify(aggregateRef?.current)).toContain(RECOVERED)

    let synchronousSnapshot = ""
    const observeAfterWorkspace = () => {
      synchronousSnapshot = JSON.stringify(aggregateRef?.current)
    }
    window.addEventListener("tldw:slides-scope-mismatch", observeAfterWorkspace)
    try {
      act(() => window.dispatchEvent(new CustomEvent("tldw:slides-scope-mismatch")))
    } finally {
      window.removeEventListener("tldw:slides-scope-mismatch", observeAfterWorkspace)
    }

    expect(synchronousSnapshot).not.toContain(RECOVERED)
    expect(synchronousSnapshot).not.toContain("html_document")
  })

  it("aborts and fences an in-flight detail response on a slides scope mismatch", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    let resolveDetail: ((value: any) => void) | null = null
    mocks.getPresentation.mockReturnValueOnce(
      new Promise((resolve) => { resolveDetail = resolve })
    )
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    await waitFor(() => expect(mocks.getPresentation).toHaveBeenCalledTimes(1))
    const signal = mocks.getPresentation.mock.calls[0][1].abortSignal as AbortSignal

    act(() => window.dispatchEvent(new CustomEvent("tldw:slides-scope-mismatch")))

    expect(signal.aborted).toBe(true)
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
    resolveDetail?.(detail({ html_document: "late private detail" }))
    await act(async () => Promise.resolve())
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
    expect(document.body.textContent).not.toContain("late private detail")
  })

  it("aborts and fences an in-flight save response on a slides scope mismatch", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    let resolveSave: ((value: any) => void) | null = null
    mocks.saveStandaloneHtmlSource.mockReturnValueOnce(
      new Promise((resolve) => { resolveSave = resolve })
    )
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    fireEvent.change(await screen.findByLabelText("HTML source"), { target: { value: EDITED } })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(mocks.saveStandaloneHtmlSource).toHaveBeenCalledTimes(1))
    const signal = mocks.saveStandaloneHtmlSource.mock.calls[0][2].abortSignal as AbortSignal
    const sentinel = "<!doctype html><title>stale save response sentinel</title>"
    const observed = installSourceDigestSentinel(sentinel)

    act(() => window.dispatchEvent(new CustomEvent("tldw:slides-scope-mismatch")))

    expect(signal.aborted).toBe(true)
    expect(sessionStorage.length).toBe(0)
    resolveSave?.({
      record: detail({
        title: "Late saved title",
        version: 8,
        html_document: sentinel,
        html_sha256: "0".repeat(64),
        html_bytes: new TextEncoder().encode(sentinel).byteLength
      }).record,
      etag: '"v8"'
    })
    await act(async () => Promise.resolve())
    expect(observed).toEqual([])
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
    expect(screen.queryByRole("heading", { name: "Late saved title" })).not.toBeInTheDocument()
  })

  it("does not publish a late save failure when scope mismatch aborts ambiguous-response reconciliation", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    let resolveReconciliation: ((value: any) => void) | null = null
    mocks.saveStandaloneHtmlSource.mockRejectedValueOnce(new TypeError("lost response"))
    mocks.getPresentation
      .mockResolvedValueOnce(detail())
      .mockReturnValueOnce(
        new Promise((resolve) => { resolveReconciliation = resolve })
      )

    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    fireEvent.change(await screen.findByLabelText("HTML source"), {
      target: { value: EDITED }
    })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(mocks.getPresentation).toHaveBeenCalledTimes(2))
    const reconciliationSignal = mocks.getPresentation.mock.calls[1][1]
      .abortSignal as AbortSignal
    const sentinel = "<!doctype html><title>stale reconciliation sentinel</title>"
    const observed = installSourceDigestSentinel(sentinel)

    act(() => window.dispatchEvent(new CustomEvent("tldw:slides-scope-mismatch")))

    expect(reconciliationSignal.aborted).toBe(true)
    expect(screen.getByText(/Current server and account could not be confirmed/i)).toBeVisible()

    await act(async () => {
      resolveReconciliation?.(detail({
        html_document: sentinel,
        html_sha256: "0".repeat(64),
        html_bytes: new TextEncoder().encode(sentinel).byteLength
      }))
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(observed).toEqual([])
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
    expect(screen.queryByText(/Save could not be confirmed/i)).not.toBeInTheDocument()
  })

  it("aborts and fences an in-flight download on a slides scope mismatch", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    let resolveDownload: ((value: Uint8Array) => void) | null = null
    mocks.downloadStandaloneHtmlDraft.mockReturnValueOnce(
      new Promise((resolve) => { resolveDownload = resolve })
    )
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    await screen.findByLabelText("HTML source")
    fireEvent.click(screen.getByRole("button", { name: "Download current draft" }))
    await waitFor(() => expect(mocks.downloadStandaloneHtmlDraft).toHaveBeenCalledTimes(1))
    const signal = mocks.downloadStandaloneHtmlDraft.mock.calls[0][2].abortSignal as AbortSignal

    act(() => window.dispatchEvent(new CustomEvent("tldw:slides-scope-mismatch")))

    expect(signal.aborted).toBe(true)
    resolveDownload?.(new TextEncoder().encode(SOURCE))
    await act(async () => Promise.resolve())
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
    expect(document.body.textContent).not.toContain("Download could not be prepared")
  })

  it.each(["model", "editor", "outline", "download"] as const)(
    "continues every security fence when the owned %s disposer throws",
    async (throwingOwner) => {
      const outlineModule = await vi.importActual<Record<string, any>>(
        ["..", "standalone-html-outline-client"].join("/")
      )
      const downloadModule = await vi.importActual<Record<string, any>>(
        ["..", "standalone-html-download"].join("/")
      )
      if (throwingOwner === "model") {
        mocks.monacoModelDispose.mockImplementationOnce(() => {
          throw new Error("model cleanup unavailable")
        })
      } else if (throwingOwner === "editor") {
        mocks.monacoEditorDispose.mockImplementationOnce(() => {
          throw new Error("editor cleanup unavailable")
        })
      } else if (throwingOwner === "outline") {
        const original = outlineModule.StandaloneHtmlOutlineController.prototype.dispose
        vi.spyOn(
          outlineModule.StandaloneHtmlOutlineController.prototype,
          "dispose"
        ).mockImplementationOnce(function (this: unknown) {
          original.call(this)
          throw new Error("outline cleanup unavailable")
        })
      } else {
        const original = downloadModule.StandaloneHtmlDownloadManager.prototype.dispose
        vi.spyOn(
          downloadModule.StandaloneHtmlDownloadManager.prototype,
          "dispose"
        ).mockImplementationOnce(function (this: unknown) {
          original.call(this)
          throw new Error("download cleanup unavailable")
        })
      }
      mocks.saveStandaloneHtmlSource.mockReturnValueOnce(new Promise(() => undefined))
      mocks.downloadStandaloneHtmlDraft.mockReturnValueOnce(new Promise(() => undefined))
      const { StandaloneHtmlWorkspace } = await loadWorkspace()
      render(<StandaloneHtmlWorkspace presentationId="html-1" />)
      fireEvent.change(await screen.findByLabelText("HTML source"), {
        target: { value: EDITED }
      })
      await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
      fireEvent.click(screen.getByRole("button", { name: "Save" }))
      fireEvent.click(screen.getByRole("button", { name: "Download current draft" }))
      await waitFor(() => {
        expect(mocks.saveStandaloneHtmlSource).toHaveBeenCalledTimes(1)
        expect(mocks.downloadStandaloneHtmlDraft).toHaveBeenCalledTimes(1)
      })
      const loadSignal = mocks.getPresentation.mock.calls[0][1].abortSignal as AbortSignal
      const saveSignal = mocks.saveStandaloneHtmlSource.mock.calls[0][2].abortSignal as AbortSignal
      const downloadSignal = mocks.downloadStandaloneHtmlDraft.mock.calls[0][2]
        .abortSignal as AbortSignal

      expect(() => {
        act(() => window.dispatchEvent(new CustomEvent("tldw:slides-scope-mismatch")))
      }).not.toThrow()

      expect(loadSignal.aborted).toBe(true)
      expect(saveSignal.aborted).toBe(true)
      expect(downloadSignal.aborted).toBe(true)
      expect(mocks.monacoModelDispose).toHaveBeenCalled()
      expect(mocks.monacoEditorDispose).toHaveBeenCalled()
      expect(mocks.outlineTerminate).toHaveBeenCalled()
      expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
      expect(document.body.textContent).not.toContain(EDITED)
      expect(screen.getByText(/Current server and account could not be confirmed/i)).toBeVisible()
    }
  )

  it("fences a malformed save response before deferred reconciliation crosses an authority boundary", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    let malformedReads = 0
    const malformed = detail()
    Object.defineProperty(malformed.record, "html_document", {
      configurable: true,
      get: () => {
        malformedReads += 1
        throw new Error("malformed source-bearing save response")
      }
    })
    let resolveReconciliation: ((value: any) => void) | null = null
    mocks.saveStandaloneHtmlSource.mockResolvedValueOnce(malformed)
    mocks.getPresentation
      .mockResolvedValueOnce(detail())
      .mockReturnValueOnce(new Promise((resolve) => { resolveReconciliation = resolve }))
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    fireEvent.change(await screen.findByLabelText("HTML source"), {
      target: { value: EDITED }
    })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(mocks.getPresentation).toHaveBeenCalledTimes(2))
    const reconciliationSignal = mocks.getPresentation.mock.calls[1][1]
      .abortSignal as AbortSignal
    const sentinel = "<!doctype html><title>malformed reconciliation sentinel</title>"
    const observed = installSourceDigestSentinel(sentinel)

    act(() => window.dispatchEvent(new CustomEvent("tldw:slides-scope-mismatch")))
    expect(reconciliationSignal.aborted).toBe(true)
    await act(async () => {
      resolveReconciliation?.(detail({
        html_document: sentinel,
        html_sha256: "0".repeat(64),
        html_bytes: new TextEncoder().encode(sentinel).byteLength
      }))
      await Promise.resolve()
    })

    expect(malformedReads).toBe(1)
    expect(observed).toEqual([])
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
    expect(document.body.textContent).not.toContain(sentinel)
  })

  it("reconciles a lost save response only when owner ID and canonical digest match, then adopts server title and ETag", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    mocks.saveStandaloneHtmlSource.mockRejectedValueOnce(new TypeError("network unavailable"))
    mocks.getPresentation
      .mockResolvedValueOnce(detail())
      .mockResolvedValueOnce({
        ...detail({
          title: "Server-derived edited title",
          version: 8,
          html_document: EDITED,
          html_sha256: EDITED_DIGEST,
          html_bytes: 36
        }),
        etag: '"opaque-v8"'
      })

    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    fireEvent.change(await screen.findByLabelText("HTML source"), { target: { value: EDITED } })
    await waitFor(() =>
      expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Not saved")
    )
    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => expect(mocks.saveStandaloneHtmlSource).toHaveBeenCalledTimes(1))
    expect(await screen.findByRole("heading", { name: "Server-derived edited title" })).toBeVisible()
    expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Saved")
    expect(screen.getByLabelText("HTML source")).toHaveValue(EDITED)
  })

  it("never autoapplies a divergent recovery record and offers restore, download, and confirmed discard", async () => {
    const recovery = await loadRecovery()
    const accepted = await (await vi.importActual<Record<string, any>>(["..", "standalone-html-source"].join("/"))).validateStandaloneHtmlSource(RECOVERED)
    const scope = recovery.createPresentationPrincipalScope("https://tldw.example", "42")
    const stored = recovery.writeStandaloneHtmlRecovery(sessionStorage, scope, {
      presentationId: "html-1",
      baseEtag: '"v6"',
      baseDigest: "f".repeat(64),
      acceptedSource: accepted,
      updatedAt: Date.now()
    })
    expect(stored.ok).toBe(true)
    const { StandaloneHtmlWorkspace } = await loadWorkspace()

    render(<StandaloneHtmlWorkspace presentationId="html-1" />)

    expect(await screen.findByLabelText("HTML source")).toHaveValue(SOURCE)
    const recoveryPanel = screen.getByRole("region", { name: "Recovered draft" })
    expect(within(recoveryPanel).getByRole("button", { name: "Restore recovered draft" })).toBeVisible()
    expect(within(recoveryPanel).getByRole("button", { name: "Download recovered draft" })).toBeVisible()
    expect(within(recoveryPanel).getByRole("button", { name: "Discard recovered draft" })).toBeVisible()

    fireEvent.click(within(recoveryPanel).getByRole("button", { name: "Restore recovered draft" }))
    expect(screen.getByLabelText("HTML source")).toHaveValue(RECOVERED)
    expect(screen.getByText("Not saved")).toBeVisible()

    fireEvent.click(screen.getByRole("button", { name: "Discard recovered draft" }))
    expect(screen.getByText(/Confirm discarding the recovered draft/i)).toBeVisible()
    fireEvent.click(screen.getByRole("button", { name: "Confirm discard recovered draft" }))
    expect(screen.queryByRole("region", { name: "Recovered draft" })).not.toBeInTheDocument()
  })

  it("keeps a recovered draft visible and raises the bounded warning when confirmed removal fails", async () => {
    const recovery = await loadRecovery()
    const source = await vi.importActual<Record<string, any>>(["..", "standalone-html-source"].join("/"))
    const accepted = await source.validateStandaloneHtmlSource(RECOVERED)
    const scope = recovery.createPresentationPrincipalScope("https://tldw.example", "42")
    recovery.writeStandaloneHtmlRecovery(sessionStorage, scope, {
      presentationId: "html-1",
      baseEtag: '"v6"',
      baseDigest: "f".repeat(64),
      acceptedSource: accepted,
      updatedAt: Date.now()
    })
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    const panel = await screen.findByRole("region", { name: "Recovered draft" })
    vi.spyOn(Object.getPrototypeOf(sessionStorage), "removeItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError")
    })

    fireEvent.click(within(panel).getByRole("button", { name: "Discard recovered draft" }))
    fireEvent.click(screen.getByRole("button", { name: "Confirm discard recovered draft" }))

    expect(screen.getByRole("region", { name: "Recovered draft" })).toBeVisible()
    expect(screen.getByText(/Recovery unavailable/i)).toBeVisible()
  })

  it("reports recovery cleanup failure after a confirmed save instead of silently claiming cleanup", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    mocks.saveStandaloneHtmlSource.mockResolvedValueOnce({
      record: detail({
        title: "Edited",
        version: 8,
        html_document: EDITED,
        html_sha256: EDITED_DIGEST,
        html_bytes: 36
      }).record,
      etag: '"v8"'
    })
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    fireEvent.change(await screen.findByLabelText("HTML source"), { target: { value: EDITED } })
    await waitFor(() => expect(sessionStorage.length).toBe(1))
    vi.spyOn(Object.getPrototypeOf(sessionStorage), "removeItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError")
    })

    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    expect(await screen.findByText(/Recovery unavailable/i)).toBeVisible()
    expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Saved")
    expect(sessionStorage.length).toBe(1)
  })

  it("flushes the last accepted keystroke synchronously on pagehide, disposes source memory, and reauthenticates before pageshow restore", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    fireEvent.change(await screen.findByLabelText("HTML source"), { target: { value: EDITED } })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())

    act(() => {
      window.dispatchEvent(new PageTransitionEvent("pagehide", { persisted: true }))
      expect(mocks.monacoModelDispose).toHaveBeenCalledTimes(1)
      expect(mocks.monacoEditorDispose).toHaveBeenCalledTimes(1)
    })

    const saved = Array.from({ length: sessionStorage.length }, (_, index) =>
      sessionStorage.getItem(sessionStorage.key(index)!)
    ).join("\n")
    expect(saved).toContain(EDITED)
    expect(document.body.textContent).not.toContain(EDITED)

    const callsBeforeRestore = mocks.getCurrentUser.mock.calls.length
    act(() => window.dispatchEvent(new PageTransitionEvent("pageshow", { persisted: true })))
    await waitFor(() => expect(mocks.getCurrentUser.mock.calls.length).toBeGreaterThan(callsBeforeRestore))
    expect(await screen.findByLabelText("HTML source")).toHaveValue(SOURCE)
    expect(screen.getByRole("region", { name: "Recovered draft" })).toBeVisible()
  })

  it("quarantines across same-scope pageshow, focus, visibility, and config reauthentication without refetching or overwriting", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    fireEvent.change(await screen.findByLabelText("HTML source"), { target: { value: EDITED } })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    const initialDetailCalls = mocks.getPresentation.mock.calls.length
    const visibilityDescriptor = Object.getOwnPropertyDescriptor(document, "visibilityState")
    Object.defineProperty(document, "visibilityState", { configurable: true, value: "visible" })

    try {
      for (const dispatch of [
        () => window.dispatchEvent(new PageTransitionEvent("pageshow", { persisted: true })),
        () => window.dispatchEvent(new Event("focus")),
        () => document.dispatchEvent(new Event("visibilitychange")),
        () => window.dispatchEvent(new Event("tldw:config-updated"))
      ]) {
        const scopeCalls = mocks.getCurrentUser.mock.calls.length
        act(() => { dispatch() })
        expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
        await waitFor(() => expect(mocks.getCurrentUser.mock.calls.length).toBeGreaterThan(scopeCalls))
        expect(await screen.findByLabelText("HTML source")).toHaveValue(EDITED)
        expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Not saved")
        expect(mocks.getPresentation).toHaveBeenCalledTimes(initialDetailCalls)
      }
    } finally {
      if (visibilityDescriptor) Object.defineProperty(document, "visibilityState", visibilityDescriptor)
    }
  })

  it("keeps pending dirty authority guarded while same-scope reauthentication is deferred, then flushes quarantine on pagehide", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    const pendingDigest = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(SECOND_EDIT)
    )
    let resolveDigest: ((value: ArrayBuffer) => void) | null = null
    const view = render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    await screen.findByLabelText("HTML source")
    vi.spyOn(crypto.subtle, "digest").mockReturnValueOnce(
      new Promise<ArrayBuffer>((resolve) => { resolveDigest = resolve })
    )
    act(() => mocks.editorProps?.onChange?.(SECOND_EDIT))
    expect(screen.getByLabelText("HTML source")).toHaveValue(SECOND_EDIT)
    let resolveConfig: ((value: any) => void) | null = null
    mocks.getConfig.mockReturnValueOnce(new Promise((resolve) => { resolveConfig = resolve }))
    const promptCalls = mocks.usePrompt.mock.calls.length

    act(() => window.dispatchEvent(new Event("focus")))

    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
    expect(screen.getByText(/Confirming current server and account/i)).toBeVisible()
    expect(mocks.promptActive).toBe(true)
    expect(mocks.usePrompt.mock.calls.length).toBeGreaterThan(promptCalls)
    expect(mocks.usePrompt).toHaveBeenLastCalledWith(expect.objectContaining({ when: true }))
    const unload = new Event("beforeunload", { cancelable: true }) as BeforeUnloadEvent
    window.dispatchEvent(unload)
    expect(unload.defaultPrevented).toBe(true)

    act(() => window.dispatchEvent(new PageTransitionEvent("pagehide", { persisted: true })))
    const stored = JSON.parse(sessionStorage.getItem(sessionStorage.key(0)!)!)
    expect(stored).toEqual(expect.objectContaining({ source: SECOND_EDIT }))
    expect(document.body.textContent).not.toContain(SECOND_EDIT)

    resolveConfig?.({ serverUrl: "https://TLDW.Example/path" })
    resolveDigest?.(pendingDigest)
    await act(async () => Promise.resolve())
    view.unmount()
  })

  it("scrubs quarantined source and keeps a bounded warning when pagehide recovery persistence fails", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    await screen.findByLabelText("HTML source")
    let resolveDigest: ((value: ArrayBuffer) => void) | null = null
    vi.spyOn(crypto.subtle, "digest").mockReturnValueOnce(
      new Promise<ArrayBuffer>((resolve) => { resolveDigest = resolve })
    )
    act(() => mocks.editorProps?.onChange?.(SECOND_EDIT))
    mocks.getConfig.mockReturnValueOnce(new Promise(() => undefined))
    act(() => window.dispatchEvent(new Event("focus")))
    vi.spyOn(Object.getPrototypeOf(sessionStorage), "setItem").mockImplementation(() => {
      throw new DOMException("quota", "QuotaExceededError")
    })

    expect(() => {
      act(() => window.dispatchEvent(new PageTransitionEvent("pagehide", { persisted: true })))
    }).not.toThrow()

    expect(document.body.textContent).not.toContain(SECOND_EDIT)
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
    const warning = screen.getByText(/Recovery unavailable/i)
    expect(warning).toBeVisible()
    expect(warning.textContent?.length).toBeLessThanOrEqual(100)
    resolveDigest?.(new ArrayBuffer(32))
    await act(async () => Promise.resolve())
  })

  it("keeps the dirty route prompt active in the offline shell", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    const view = render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    fireEvent.change(await screen.findByLabelText("HTML source"), {
      target: { value: EDITED }
    })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    mocks.online = false

    view.rerender(<StandaloneHtmlWorkspace presentationId="html-1" />)

    expect(screen.getByText(/Server is offline/i)).toBeVisible()
    expect(mocks.promptActive).toBe(true)
    expect(mocks.usePrompt).toHaveBeenLastCalledWith(expect.objectContaining({ when: true }))
    const unload = new Event("beforeunload", { cancelable: true }) as BeforeUnloadEvent
    window.dispatchEvent(unload)
    expect(unload.defaultPrevented).toBe(true)
  })

  it("flushes a synchronously preflight-valid latest editor candidate when pagehide beats SHA-256", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    await screen.findByLabelText("HTML source")
    const digest = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(SECOND_EDIT)
    )
    let resolveDigest: ((value: ArrayBuffer) => void) | null = null
    vi.spyOn(crypto.subtle, "digest").mockReturnValueOnce(
      new Promise<ArrayBuffer>((resolve) => { resolveDigest = resolve })
    )

    act(() => {
      mocks.editorProps?.onChange?.(SECOND_EDIT)
      window.dispatchEvent(new PageTransitionEvent("pagehide", { persisted: true }))
    })

    const saved = Array.from({ length: sessionStorage.length }, (_, index) =>
      sessionStorage.getItem(sessionStorage.key(index)!)
    ).join("\n")
    expect(saved).toContain(SECOND_EDIT)
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
    resolveDigest?.(digest)
    await act(async () => Promise.resolve())
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
  })

  it("flushes an exact digest-pending candidate before a confirmed SPA route unmount", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    const view = render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    await screen.findByLabelText("HTML source")
    const digest = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(THIRD_EDIT)
    )
    let resolveDigest: ((value: ArrayBuffer) => void) | null = null
    vi.spyOn(crypto.subtle, "digest").mockReturnValueOnce(
      new Promise<ArrayBuffer>((resolve) => { resolveDigest = resolve })
    )

    act(() => mocks.editorProps?.onChange?.(THIRD_EDIT))
    expect(mocks.editorProps?.value).toBe(THIRD_EDIT)

    view.unmount()

    const saved = Array.from({ length: sessionStorage.length }, (_, index) =>
      sessionStorage.getItem(sessionStorage.key(index)!)
    ).join("\n")
    expect(saved).toContain(THIRD_EDIT)
    resolveDigest?.(digest)
    await act(async () => Promise.resolve())
  })

  it.each([
    ["slides scope mismatch", () => new CustomEvent("tldw:slides-scope-mismatch")],
    [
      "logout",
      () => new CustomEvent("tldw:auth-principal-changed", { detail: { kind: "logout" } })
    ]
  ])("never rewrites old-scope recovery when %s is followed by unmount", async (_case, eventFactory) => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    const view = render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    fireEvent.change(await screen.findByLabelText("HTML source"), {
      target: { value: EDITED }
    })
    await waitFor(() => expect(sessionStorage.length).toBe(1))
    const setItem = vi.spyOn(Object.getPrototypeOf(sessionStorage), "setItem")

    act(() => window.dispatchEvent(eventFactory()))

    expect(sessionStorage.length).toBe(0)
    setItem.mockClear()
    expect(() => view.unmount()).not.toThrow()
    expect(setItem).not.toHaveBeenCalled()
  })

  it("writes a pagehide candidate exactly once when unmount follows", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    const view = render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    await screen.findByLabelText("HTML source")
    let resolveDigest: ((value: ArrayBuffer) => void) | null = null
    vi.spyOn(crypto.subtle, "digest").mockReturnValueOnce(
      new Promise<ArrayBuffer>((resolve) => { resolveDigest = resolve })
    )
    const setItem = vi.spyOn(Object.getPrototypeOf(sessionStorage), "setItem")

    act(() => {
      mocks.editorProps?.onChange?.(SECOND_EDIT)
      window.dispatchEvent(new PageTransitionEvent("pagehide", { persisted: true }))
    })
    view.unmount()

    expect(
      setItem.mock.calls.filter(([, value]) => String(value).includes(SECOND_EDIT))
    ).toHaveLength(1)
    resolveDigest?.(new ArrayBuffer(32))
    await act(async () => Promise.resolve())
  })

  it("flushes the latest pending candidate from same-scope quarantine on unmount", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    const view = render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    await screen.findByLabelText("HTML source")
    let resolveDigest: ((value: ArrayBuffer) => void) | null = null
    vi.spyOn(crypto.subtle, "digest").mockReturnValueOnce(
      new Promise<ArrayBuffer>((resolve) => { resolveDigest = resolve })
    )
    mocks.getConfig.mockReturnValueOnce(new Promise(() => undefined))

    act(() => {
      mocks.editorProps?.onChange?.(THIRD_EDIT)
      window.dispatchEvent(new Event("focus"))
    })

    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
    expect(() => view.unmount()).not.toThrow()
    const saved = Array.from({ length: sessionStorage.length }, (_, index) =>
      sessionStorage.getItem(sessionStorage.key(index)!)
    ).join("\n")
    expect(saved).toContain(THIRD_EDIT)
    resolveDigest?.(new ArrayBuffer(32))
    await act(async () => Promise.resolve())
  })

  it.each(["getter", "quota"])(
    "continues unmount disposal when recovery storage %s access fails",
    async (failure) => {
      const { StandaloneHtmlWorkspace } = await loadWorkspace()
      const view = render(<StandaloneHtmlWorkspace presentationId="html-1" />)
      await screen.findByLabelText("HTML source")
      vi.spyOn(crypto.subtle, "digest").mockReturnValueOnce(new Promise(() => undefined))
      act(() => mocks.editorProps?.onChange?.(SECOND_EDIT))
      mocks.monacoModelDispose.mockClear()
      mocks.monacoEditorDispose.mockClear()
      mocks.outlineTerminate.mockClear()
      const storagePrototype = Object.getPrototypeOf(sessionStorage)
      let restoreStorage = () => undefined
      if (failure === "getter") {
        restoreStorage = installThrowingSessionStorageGetter()
      } else {
        vi.spyOn(storagePrototype, "setItem").mockImplementation(() => {
          throw new DOMException("quota", "QuotaExceededError")
        })
      }

      try {
        expect(() => view.unmount()).not.toThrow()
        expect(mocks.monacoModelDispose).toHaveBeenCalled()
        expect(mocks.monacoEditorDispose).toHaveBeenCalled()
        expect(mocks.outlineTerminate).toHaveBeenCalled()
      } finally {
        restoreStorage()
      }
    }
  )

  it("preserves a pending empty draft through same-scope reauthentication", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    const editor = await screen.findByLabelText("HTML source")
    let resolveDigest: ((value: ArrayBuffer) => void) | null = null
    vi.spyOn(crypto.subtle, "digest").mockReturnValueOnce(
      new Promise<ArrayBuffer>((resolve) => { resolveDigest = resolve })
    )

    act(() => mocks.editorProps?.onChange?.(""))
    expect(editor).toHaveValue("")
    const scopeCalls = mocks.getCurrentUser.mock.calls.length
    act(() => window.dispatchEvent(new Event("focus")))

    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
    await waitFor(() => expect(mocks.getCurrentUser.mock.calls.length).toBeGreaterThan(scopeCalls))
    expect(await screen.findByLabelText("HTML source")).toHaveValue("")
    expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Not saved")

    resolveDigest?.(new ArrayBuffer(32))
    await act(async () => Promise.resolve())
  })

  it("persists a divergent pending empty draft synchronously on pagehide", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    await screen.findByLabelText("HTML source")
    let resolveDigest: ((value: ArrayBuffer) => void) | null = null
    vi.spyOn(crypto.subtle, "digest").mockReturnValueOnce(
      new Promise<ArrayBuffer>((resolve) => { resolveDigest = resolve })
    )

    act(() => {
      mocks.editorProps?.onChange?.("")
      window.dispatchEvent(new PageTransitionEvent("pagehide", { persisted: true }))
    })

    expect(sessionStorage.length).toBe(1)
    const stored = JSON.parse(sessionStorage.getItem(sessionStorage.key(0)!)!)
    expect(stored).toEqual(expect.objectContaining({ source: "" }))
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()

    resolveDigest?.(new ArrayBuffer(32))
    await act(async () => Promise.resolve())
  })

  it("clears stale recovery when pending empty exactly matches an empty saved base", async () => {
    mocks.getPresentation.mockResolvedValueOnce(detail({
      html_document: "",
      html_sha256: EMPTY_DIGEST,
      html_bytes: 0
    }))
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    await screen.findByLabelText("HTML source")
    act(() => mocks.editorProps?.onChange?.(EDITED))
    await waitFor(() => expect(sessionStorage.length).toBe(1))
    let resolveDigest: ((value: ArrayBuffer) => void) | null = null
    vi.spyOn(crypto.subtle, "digest").mockReturnValueOnce(
      new Promise<ArrayBuffer>((resolve) => { resolveDigest = resolve })
    )

    act(() => {
      mocks.editorProps?.onChange?.("")
      window.dispatchEvent(new PageTransitionEvent("pagehide", { persisted: true }))
    })

    expect(sessionStorage.length).toBe(0)
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()

    resolveDigest?.(new ArrayBuffer(32))
    await act(async () => Promise.resolve())
  })

  it("clears stale recovery when a pending pagehide candidate exactly reverts to the saved base", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    await screen.findByLabelText("HTML source")
    act(() => mocks.editorProps?.onChange?.(EDITED))
    await waitFor(() => expect(sessionStorage.length).toBe(1))
    const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(SOURCE))
    let resolveDigest: ((value: ArrayBuffer) => void) | null = null
    vi.spyOn(crypto.subtle, "digest").mockReturnValueOnce(
      new Promise<ArrayBuffer>((resolve) => { resolveDigest = resolve })
    )

    act(() => {
      mocks.editorProps?.onChange?.(SOURCE)
      window.dispatchEvent(new PageTransitionEvent("pagehide", { persisted: true }))
    })

    expect(sessionStorage.length).toBe(0)
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
    act(() => window.dispatchEvent(new PageTransitionEvent("pageshow", { persisted: true })))
    expect(await screen.findByLabelText("HTML source")).toHaveValue(SOURCE)
    expect(screen.queryByRole("region", { name: "Recovered draft" })).not.toBeInTheDocument()

    resolveDigest?.(digest)
    await act(async () => Promise.resolve())
  })

  it("still scrubs and suppresses stale recovery when pagehide reversion cleanup fails", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    await screen.findByLabelText("HTML source")
    act(() => mocks.editorProps?.onChange?.(EDITED))
    await waitFor(() => expect(sessionStorage.length).toBe(1))
    const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(SOURCE))
    let resolveDigest: ((value: ArrayBuffer) => void) | null = null
    vi.spyOn(crypto.subtle, "digest").mockReturnValueOnce(
      new Promise<ArrayBuffer>((resolve) => { resolveDigest = resolve })
    )
    vi.spyOn(Object.getPrototypeOf(sessionStorage), "removeItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError")
    })

    act(() => mocks.editorProps?.onChange?.(SOURCE))
    expect(() => {
      act(() => window.dispatchEvent(new PageTransitionEvent("pagehide", { persisted: true })))
    }).not.toThrow()

    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
    expect(document.body.textContent).not.toContain(EDITED)
    act(() => window.dispatchEvent(new PageTransitionEvent("pageshow", { persisted: true })))
    expect(await screen.findByLabelText("HTML source")).toHaveValue(SOURCE)
    expect(screen.queryByRole("region", { name: "Recovered draft" })).not.toBeInTheDocument()
    expect(screen.getByText(/Recovery unavailable/i)).toBeVisible()

    resolveDigest?.(digest)
    await act(async () => Promise.resolve())
  })

  it("raises the bounded recovery warning when pagehide cannot persist the latest candidate", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    await screen.findByLabelText("HTML source")
    const digest = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(SECOND_EDIT)
    )
    let resolveDigest: ((value: ArrayBuffer) => void) | null = null
    vi.spyOn(crypto.subtle, "digest").mockReturnValueOnce(
      new Promise<ArrayBuffer>((resolve) => { resolveDigest = resolve })
    )
    vi.spyOn(Object.getPrototypeOf(sessionStorage), "setItem").mockImplementation(() => {
      throw new DOMException("quota", "QuotaExceededError")
    })

    act(() => {
      mocks.editorProps?.onChange?.(SECOND_EDIT)
      window.dispatchEvent(new PageTransitionEvent("pagehide", { persisted: true }))
    })
    act(() => window.dispatchEvent(new PageTransitionEvent("pageshow", { persisted: true })))

    expect(await screen.findByText(/Recovery unavailable/i)).toBeVisible()
    resolveDigest?.(digest)
    await act(async () => Promise.resolve())
  })

  it("preserves the only in-memory draft and recovery warning through same-scope restoration after quota failure", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    await screen.findByLabelText("HTML source")
    vi.spyOn(Object.getPrototypeOf(sessionStorage), "setItem").mockImplementation(() => {
      throw new DOMException("quota", "QuotaExceededError")
    })
    fireEvent.change(screen.getByLabelText("HTML source"), { target: { value: EDITED } })
    expect(await screen.findByText(/Recovery unavailable/i)).toBeVisible()

    act(() => window.dispatchEvent(new Event("focus")))

    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
    expect(await screen.findByLabelText("HTML source")).toHaveValue(EDITED)
    expect(screen.getByText(/Recovery unavailable/i)).toBeVisible()
    const unload = new Event("beforeunload", { cancelable: true }) as BeforeUnloadEvent
    window.dispatchEvent(unload)
    expect(unload.defaultPrevented).toBe(true)
  })

  it.each(["focus reauthentication", "capability loading"])(
    "keeps the newest pending candidate authoritative across same-scope %s",
    async (boundary) => {
      const originalDigest = crypto.subtle.digest.bind(crypto.subtle)
      const [secondDigest, thirdDigest] = await Promise.all([
        originalDigest("SHA-256", new TextEncoder().encode(SECOND_EDIT)),
        originalDigest("SHA-256", new TextEncoder().encode(THIRD_EDIT))
      ])
      let resolveOriginalSecond: ((value: ArrayBuffer) => void) | null = null
      let resolveRestoredSecond: ((value: ArrayBuffer) => void) | null = null
      let resolveThird: ((value: ArrayBuffer) => void) | null = null
      const { StandaloneHtmlWorkspace } = await loadWorkspace()
      const view = render(<StandaloneHtmlWorkspace presentationId="html-1" />)
      await screen.findByLabelText("HTML source")
      const digest = vi.spyOn(crypto.subtle, "digest")
        .mockReturnValueOnce(new Promise<ArrayBuffer>((resolve) => { resolveOriginalSecond = resolve }))
        .mockReturnValueOnce(new Promise<ArrayBuffer>((resolve) => { resolveRestoredSecond = resolve }))
        .mockReturnValueOnce(new Promise<ArrayBuffer>((resolve) => { resolveThird = resolve }))

      act(() => mocks.editorProps?.onChange?.(SECOND_EDIT))
      expect(screen.getByLabelText("HTML source")).toHaveValue(SECOND_EDIT)

      if (boundary === "focus reauthentication") {
        act(() => window.dispatchEvent(new Event("focus")))
      } else {
        mocks.slidesCapabilities = { ...readyCapabilities, status: "loading" }
        view.rerender(<StandaloneHtmlWorkspace presentationId="html-1" />)
        expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
        mocks.slidesCapabilities = readyCapabilities
        view.rerender(<StandaloneHtmlWorkspace presentationId="html-1" />)
      }

      await waitFor(() => expect(digest).toHaveBeenCalledTimes(2))
      expect(await screen.findByLabelText("HTML source")).toHaveValue(SECOND_EDIT)
      expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()
      expect(screen.getByRole("button", { name: "Download current draft" })).toBeDisabled()

      resolveOriginalSecond?.(secondDigest)
      await act(async () => Promise.resolve())
      act(() => mocks.editorProps?.onChange?.(THIRD_EDIT))
      await waitFor(() => expect(digest).toHaveBeenCalledTimes(3))
      expect(screen.getByLabelText("HTML source")).toHaveValue(THIRD_EDIT)

      resolveRestoredSecond?.(secondDigest)
      await act(async () => Promise.resolve())
      expect(screen.getByLabelText("HTML source")).toHaveValue(THIRD_EDIT)
      expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()
      expect(screen.getByRole("button", { name: "Download current draft" })).toBeDisabled()

      resolveThird?.(thirdDigest)
      await waitFor(() => expect(screen.getByLabelText("HTML source")).toHaveValue(THIRD_EDIT))
      expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Not saved")
      expect(screen.getByRole("button", { name: "Save" })).toBeEnabled()
      expect(screen.getByRole("button", { name: "Download current draft" })).toBeEnabled()
    }
  )

  it("synchronously fences and clears source, matching recovery, and in-flight requests on logout", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    fireEvent.change(await screen.findByLabelText("HTML source"), { target: { value: EDITED } })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    const loadSignal = mocks.getPresentation.mock.calls[0][1].abortSignal as AbortSignal

    act(() =>
      window.dispatchEvent(
        new CustomEvent("tldw:auth-principal-changed", { detail: { kind: "logout" } })
      )
    )

    expect(document.body.textContent).not.toContain(EDITED)
    expect(sessionStorage.length).toBe(0)
    expect(loadSignal.aborted).toBe(true)
    expect(screen.getByText(/Current server and account could not be confirmed/i)).toBeVisible()
  })

  it.each([
    [
      "slides scope mismatch",
      () => new CustomEvent("tldw:slides-scope-mismatch")
    ],
    [
      "logout",
      () => new CustomEvent("tldw:auth-principal-changed", { detail: { kind: "logout" } })
    ]
  ])("scrubs synchronously on %s even when the sessionStorage getter throws", async (_case, eventFactory) => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    await screen.findByLabelText("HTML source")
    act(() => mocks.editorProps?.onChange?.(EDITED))
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    const restoreStorage = installThrowingSessionStorageGetter()

    try {
      expect(() => {
        act(() => window.dispatchEvent(eventFactory()))
      }).not.toThrow()

      expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
      expect(document.body.textContent).not.toContain(EDITED)
      expect(mocks.monacoModelDispose).toHaveBeenCalled()
      expect(mocks.monacoEditorDispose).toHaveBeenCalled()
      expect(mocks.outlineTerminate).toHaveBeenCalled()
      expect(screen.getByText(/Current server and account could not be confirmed/i)).toBeVisible()

      fireEvent.click(screen.getByRole("button", { name: "Retry" }))
      expect(await screen.findByLabelText("HTML source")).toHaveValue(SOURCE)
      const warning = screen.getByText(/Recovery unavailable/i)
      expect(warning).toBeVisible()
      expect(warning.textContent?.length).toBeLessThanOrEqual(100)
    } finally {
      restoreStorage()
    }
  })

  it("scrubs synchronously on pagehide when recovery storage acquisition throws", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    await screen.findByLabelText("HTML source")
    act(() => mocks.editorProps?.onChange?.(EDITED))
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    const restoreStorage = installThrowingSessionStorageGetter()

    try {
      expect(() => {
        act(() => window.dispatchEvent(new PageTransitionEvent("pagehide", { persisted: true })))
      }).not.toThrow()

      expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
      expect(document.body.textContent).not.toContain(EDITED)
      expect(mocks.monacoModelDispose).toHaveBeenCalled()
      expect(mocks.monacoEditorDispose).toHaveBeenCalled()
      expect(mocks.outlineTerminate).toHaveBeenCalled()

      act(() => window.dispatchEvent(new PageTransitionEvent("pageshow", { persisted: true })))
      expect(await screen.findByLabelText("HTML source")).toHaveValue(SOURCE)
      const warning = screen.getByText(/Recovery unavailable/i)
      expect(warning).toBeVisible()
      expect(warning.textContent?.length).toBeLessThanOrEqual(100)
    } finally {
      restoreStorage()
    }
  })

  it("clears old-origin recovery when a configuration boundary reauthenticates to a different server", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    fireEvent.change(await screen.findByLabelText("HTML source"), { target: { value: EDITED } })
    await waitFor(() =>
      expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Not saved")
    )
    expect(sessionStorage.length).toBe(1)
    mocks.getConfig.mockResolvedValue({ serverUrl: "https://other.example/path" })

    act(() => window.dispatchEvent(new Event("tldw:config-updated")))

    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
    await waitFor(() => expect(sessionStorage.length).toBe(0))
  })

  it("keeps failed old-scope cleanup tracked and warned after a successful new-scope write", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    await screen.findByLabelText("HTML source")
    act(() => mocks.editorProps?.onChange?.(EDITED))
    await waitFor(() => expect(sessionStorage.length).toBe(1))
    const storagePrototype = Object.getPrototypeOf(sessionStorage)
    const realRemoveItem = storagePrototype.removeItem
    const oldOriginFragment = encodeURIComponent("https://tldw.example")
    const removeItem = vi.spyOn(storagePrototype, "removeItem").mockImplementation(function (
      this: Storage,
      key: string
    ) {
      if (key.includes(oldOriginFragment)) {
        throw new DOMException("old scope cleanup blocked", "SecurityError")
      }
      return realRemoveItem.call(this, key)
    })
    mocks.getConfig.mockResolvedValue({ serverUrl: "https://other.example/path" })

    act(() => window.dispatchEvent(new Event("tldw:config-updated")))

    expect(await screen.findByLabelText("HTML source")).toHaveValue(SOURCE)
    expect(screen.getByText(/Recovery unavailable/i)).toBeVisible()
    act(() => mocks.editorProps?.onChange?.(SECOND_EDIT))
    await waitFor(() => expect(screen.getByLabelText("HTML source")).toHaveValue(SECOND_EDIT))

    expect(screen.getByText(/Recovery unavailable/i)).toBeVisible()
    expect(
      removeItem.mock.calls.filter(([key]) => String(key).includes(oldOriginFragment)).length
    ).toBeGreaterThanOrEqual(2)
  })

  it("keeps a newly written same-key draft when an older cleanup obligation later retries", async () => {
    mocks.saveStandaloneHtmlSource.mockResolvedValueOnce({
      record: detail({
        title: "Edited",
        version: 8,
        html_document: EDITED,
        html_sha256: EDITED_DIGEST,
        html_bytes: 36
      }).record,
      etag: '"v8"'
    })
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    const view = render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    fireEvent.change(await screen.findByLabelText("HTML source"), { target: { value: EDITED } })
    await waitFor(() => expect(sessionStorage.length).toBe(1))
    const storagePrototype = Object.getPrototypeOf(sessionStorage)
    const blockedRemove = vi.spyOn(storagePrototype, "removeItem").mockImplementation(() => {
      throw new DOMException("cleanup blocked", "SecurityError")
    })

    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() =>
      expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Saved")
    )
    expect(screen.getByText(/Recovery unavailable/i)).toBeVisible()
    fireEvent.change(screen.getByLabelText("HTML source"), { target: { value: SECOND_EDIT } })
    await waitFor(() => expect(screen.getByLabelText("HTML source")).toHaveValue(SECOND_EDIT))
    const beforeReload = Array.from({ length: sessionStorage.length }, (_, index) =>
      sessionStorage.getItem(sessionStorage.key(index)!)
    ).join("\n")
    expect(beforeReload).toContain(SECOND_EDIT)

    blockedRemove.mockRestore()
    mocks.slidesCapabilities = {
      ...readyCapabilities,
      canReadStandalone: false,
      canDraftStandalone: false,
      canEditStandalone: false
    }
    view.rerender(<StandaloneHtmlWorkspace presentationId="html-1" />)
    await waitFor(() => expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument())
    mocks.slidesCapabilities = readyCapabilities
    view.rerender(<StandaloneHtmlWorkspace presentationId="html-1" />)

    expect(await screen.findByLabelText("HTML source")).toHaveValue(SOURCE)
    expect(await screen.findByRole("region", { name: "Recovered draft" })).toBeVisible()
    const afterReload = Array.from({ length: sessionStorage.length }, (_, index) =>
      sessionStorage.getItem(sessionStorage.key(index)!)
    ).join("\n")
    expect(afterReload).toContain(SECOND_EDIT)
  })

  it("does not read or offer an unresolved same-key recovery until its cleanup succeeds", async () => {
    const recovery = await loadRecovery()
    const source = await vi.importActual<Record<string, any>>(
      ["..", "standalone-html-source"].join("/")
    )
    const accepted = await source.validateStandaloneHtmlSource(RECOVERED)
    const scope = recovery.createPresentationPrincipalScope("https://tldw.example", "42")
    recovery.writeStandaloneHtmlRecovery(sessionStorage, scope, {
      presentationId: "html-1",
      baseEtag: '"v6"',
      baseDigest: "f".repeat(64),
      acceptedSource: accepted,
      updatedAt: Date.now()
    })
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    const view = render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    const recoveryPanel = await screen.findByRole("region", { name: "Recovered draft" })
    const blockedRemove = vi.spyOn(Object.getPrototypeOf(sessionStorage), "removeItem")
      .mockImplementation(() => {
        throw new DOMException("cleanup blocked", "SecurityError")
      })
    fireEvent.click(within(recoveryPanel).getByRole("button", { name: "Discard recovered draft" }))
    fireEvent.click(screen.getByRole("button", { name: "Confirm discard recovered draft" }))
    expect(screen.getByText(/Recovery unavailable/i)).toBeVisible()

    mocks.slidesCapabilities = {
      ...readyCapabilities,
      canReadStandalone: false,
      canDraftStandalone: false,
      canEditStandalone: false
    }
    view.rerender(<StandaloneHtmlWorkspace presentationId="html-1" />)
    mocks.slidesCapabilities = readyCapabilities
    view.rerender(<StandaloneHtmlWorkspace presentationId="html-1" />)

    expect(await screen.findByLabelText("HTML source")).toHaveValue(SOURCE)
    expect(screen.queryByRole("region", { name: "Recovered draft" })).not.toBeInTheDocument()
    expect(screen.getByText(/Recovery unavailable/i)).toBeVisible()

    blockedRemove.mockRestore()
    mocks.slidesCapabilities = {
      ...readyCapabilities,
      canReadStandalone: false,
      canDraftStandalone: false,
      canEditStandalone: false
    }
    view.rerender(<StandaloneHtmlWorkspace presentationId="html-1" />)
    mocks.slidesCapabilities = readyCapabilities
    view.rerender(<StandaloneHtmlWorkspace presentationId="html-1" />)

    expect(await screen.findByLabelText("HTML source")).toHaveValue(SOURCE)
    expect(screen.queryByRole("region", { name: "Recovered draft" })).not.toBeInTheDocument()
    expect(screen.queryByText(/Recovery unavailable/i)).not.toBeInTheDocument()
  })

  it("warns before unload and requires an inline confirmation before Back discards dirty memory", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    fireEvent.change(await screen.findByLabelText("HTML source"), { target: { value: EDITED } })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    const event = new Event("beforeunload", { cancelable: true }) as BeforeUnloadEvent

    window.dispatchEvent(event)
    expect(event.defaultPrevented).toBe(true)

    fireEvent.click(screen.getByRole("button", { name: "Back to presentations" }))
    expect(screen.getByText(/Leave without saving/i)).toBeVisible()
    expect(mocks.navigate).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole("button", { name: "Leave presentation" }))
    await waitFor(() =>
      expect(mocks.usePrompt).toHaveBeenLastCalledWith(expect.objectContaining({ when: false }))
    )
    expect(mocks.navigate).toHaveBeenCalledTimes(1)
    expect(mocks.navigate).toHaveBeenCalledWith("/presentation-studio")
  })

  it("registers the shared route prompt with the exact dirty-state warning", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    fireEvent.change(await screen.findByLabelText("HTML source"), { target: { value: EDITED } })

    await waitFor(() =>
      expect(mocks.usePrompt).toHaveBeenLastCalledWith({
        when: true,
        message: "Leave without saving? Your local draft is preserved only in this tab."
      })
    )
  })

  it("blocks navigation and source-consuming actions as soon as a valid candidate awaits its digest", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    const editor = await screen.findByLabelText("HTML source")
    const digest = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(SECOND_EDIT)
    )
    let resolveDigest: ((value: ArrayBuffer) => void) | null = null
    vi.spyOn(crypto.subtle, "digest").mockReturnValueOnce(
      new Promise<ArrayBuffer>((resolve) => { resolveDigest = resolve })
    )

    const unload = new Event("beforeunload", { cancelable: true }) as BeforeUnloadEvent
    act(() => {
      mocks.editorProps?.onChange?.(SECOND_EDIT)
      window.dispatchEvent(unload)
    })

    expect(unload.defaultPrevented).toBe(true)
    await waitFor(() =>
      expect(mocks.usePrompt).toHaveBeenLastCalledWith(expect.objectContaining({ when: true }))
    )
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Download current draft" })).toBeDisabled()
    fireEvent.click(screen.getByRole("button", { name: "Back to presentations" }))
    expect(screen.getByText(/Leave without saving/i)).toBeVisible()
    expect(mocks.navigate).not.toHaveBeenCalled()

    const cleanUnload = new Event("beforeunload", { cancelable: true }) as BeforeUnloadEvent
    act(() => {
      mocks.editorProps?.onChange?.("invalid\u0000candidate")
      window.dispatchEvent(cleanUnload)
    })

    expect(cleanUnload.defaultPrevented).toBe(false)
    await waitFor(() =>
      expect(mocks.usePrompt).toHaveBeenLastCalledWith(expect.objectContaining({ when: false }))
    )
    expect(screen.getByRole("button", { name: "Download current draft" })).toBeEnabled()
    resolveDigest?.(digest)
    await act(async () => Promise.resolve())
    expect(editor).toHaveValue(SOURCE)
  })

  it("keeps validator-unavailable access inert and read/recovery-only", async () => {
    mocks.slidesCapabilities = {
      ...readyCapabilities,
      status: "validator_unavailable",
      reason: "validator_unavailable",
      canEditStandalone: false
    }
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    render(<StandaloneHtmlWorkspace presentationId="html-1" />)

    expect(await screen.findByLabelText("HTML source")).toHaveValue(SOURCE)
    expect(screen.getByText("Saving is unavailable")).toBeVisible()
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Download current draft" })).toBeEnabled()
  })

  it("quarantines a dirty draft during live capability loading and restores it without a server overwrite", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    let resolveSave: ((value: any) => void) | null = null
    mocks.saveStandaloneHtmlSource.mockReturnValueOnce(
      new Promise((resolve) => { resolveSave = resolve })
    )
    const view = render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    fireEvent.change(await screen.findByLabelText("HTML source"), { target: { value: EDITED } })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(mocks.saveStandaloneHtmlSource).toHaveBeenCalledTimes(1))
    const saveSignal = mocks.saveStandaloneHtmlSource.mock.calls[0][2].abortSignal as AbortSignal

    mocks.slidesCapabilities = { ...readyCapabilities, status: "loading" }
    view.rerender(<StandaloneHtmlWorkspace presentationId="html-1" />)

    expect(saveSignal.aborted).toBe(true)
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
    expect(screen.getByText(/Checking standalone HTML access/i)).toBeVisible()
    mocks.slidesCapabilities = readyCapabilities
    view.rerender(<StandaloneHtmlWorkspace presentationId="html-1" />)

    expect(await screen.findByLabelText("HTML source")).toHaveValue(EDITED)
    expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Not saved")
    expect(mocks.getPresentation).toHaveBeenCalledTimes(1)
    resolveSave?.(detail())
  })

  it("aborts draft download immediately when draft authority is revoked and gates stale handlers", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    let resolveDownload: ((value: Uint8Array) => void) | null = null
    mocks.downloadStandaloneHtmlDraft.mockReturnValueOnce(
      new Promise((resolve) => { resolveDownload = resolve })
    )
    const view = render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    await screen.findByLabelText("HTML source")
    const download = screen.getByRole("button", { name: "Download current draft" })
    fireEvent.click(download)
    await waitFor(() => expect(mocks.downloadStandaloneHtmlDraft).toHaveBeenCalledTimes(1))
    const signal = mocks.downloadStandaloneHtmlDraft.mock.calls[0][2].abortSignal as AbortSignal

    mocks.slidesCapabilities = { ...readyCapabilities, canDraftStandalone: false }
    view.rerender(<StandaloneHtmlWorkspace presentationId="html-1" />)

    expect(signal.aborted).toBe(true)
    const gated = screen.getByRole("button", { name: "Download current draft" }) as HTMLButtonElement
    expect(gated).toBeDisabled()
    gated.disabled = false
    fireEvent.click(gated)
    expect(mocks.downloadStandaloneHtmlDraft).toHaveBeenCalledTimes(1)
    resolveDownload?.(new TextEncoder().encode(SOURCE))
    await act(async () => Promise.resolve())
    expect(document.body.textContent).not.toContain("Download could not be prepared")
  })

  it("scrubs mounted source and fences late work when read authority is definitively revoked", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    let resolveDownload: ((value: Uint8Array) => void) | null = null
    mocks.downloadStandaloneHtmlDraft.mockReturnValueOnce(
      new Promise((resolve) => { resolveDownload = resolve })
    )
    const view = render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    await screen.findByLabelText("HTML source")
    fireEvent.click(screen.getByRole("button", { name: "Download current draft" }))
    await waitFor(() => expect(mocks.downloadStandaloneHtmlDraft).toHaveBeenCalledTimes(1))
    const signal = mocks.downloadStandaloneHtmlDraft.mock.calls[0][2].abortSignal as AbortSignal

    mocks.slidesCapabilities = {
      ...readyCapabilities,
      canReadStandalone: false,
      canDraftStandalone: false,
      canEditStandalone: false
    }
    view.rerender(<StandaloneHtmlWorkspace presentationId="html-1" />)

    expect(signal.aborted).toBe(true)
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
    expect(screen.getByText(/does not support reading standalone HTML presentations/i)).toBeVisible()
    resolveDownload?.(new TextEncoder().encode(SOURCE))
    await act(async () => Promise.resolve())
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
  })

  it("aborts save and makes the editor read-only when edit authority is revoked", async () => {
    const { StandaloneHtmlWorkspace } = await loadWorkspace()
    let resolveSave: ((value: any) => void) | null = null
    mocks.saveStandaloneHtmlSource.mockReturnValueOnce(
      new Promise((resolve) => { resolveSave = resolve })
    )
    const view = render(<StandaloneHtmlWorkspace presentationId="html-1" />)
    fireEvent.change(await screen.findByLabelText("HTML source"), { target: { value: EDITED } })
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(mocks.saveStandaloneHtmlSource).toHaveBeenCalledTimes(1))
    const signal = mocks.saveStandaloneHtmlSource.mock.calls[0][2].abortSignal as AbortSignal

    mocks.slidesCapabilities = { ...readyCapabilities, canEditStandalone: false }
    view.rerender(<StandaloneHtmlWorkspace presentationId="html-1" />)

    expect(signal.aborted).toBe(true)
    expect(await screen.findByLabelText("HTML source")).toHaveValue(EDITED)
    await waitFor(() => expect(mocks.editorProps?.options?.readOnly).toBe(true))
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()
    expect(screen.getByTestId("standalone-html-save-status")).toHaveTextContent("Not saved")
    resolveSave?.(detail())
  })
})

describe("kind-first Presentation Studio dispatch", () => {
  beforeEach(() => {
    mocks.online = true
    mocks.runtimeExtension = true
    mocks.getPresentationMetadata.mockReset().mockResolvedValue({
      record: { id: "html-1", content_kind: "standalone_html" },
      etag: null
    })
    mocks.getPresentation.mockReset()
    mocks.listVisualStyles.mockReset().mockResolvedValue([])
  })

  it("keeps an extension runtime source-free after standalone metadata dispatch", async () => {
    const { PresentationStudioPage } = await vi.importActual<Record<string, any>>(
      ["..", "PresentationStudioPage"].join("/")
    )

    render(<PresentationStudioPage mode="detail" projectId="html-1" />)

    expect(await screen.findByText("Standalone HTML editing is available only in the WebUI.")).toBeVisible()
    expect(mocks.getPresentationMetadata).toHaveBeenCalledWith("html-1")
    expect(mocks.getPresentation).not.toHaveBeenCalled()
    expect(mocks.listVisualStyles).not.toHaveBeenCalled()
  })

  it("keeps an extension source-free when legacy metadata and capability probes are unavailable", async () => {
    const missing = Object.assign(new Error("missing"), { status: 404 })
    mocks.getPresentationMetadata.mockRejectedValueOnce(missing)
    mocks.getSlidesCapabilities.mockRejectedValueOnce(missing)
    const { PresentationStudioPage } = await vi.importActual<Record<string, any>>(
      ["..", "PresentationStudioPage"].join("/")
    )

    render(<PresentationStudioPage mode="detail" projectId="html-1" />)

    expect(await screen.findByText("Presentation metadata is unavailable")).toBeVisible()
    expect(mocks.getPresentation).not.toHaveBeenCalled()
    expect(mocks.listVisualStyles).not.toHaveBeenCalled()
  })
})
