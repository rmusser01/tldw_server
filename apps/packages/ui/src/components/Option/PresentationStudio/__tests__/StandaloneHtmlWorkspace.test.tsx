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
  editorProps: null as Record<string, any> | null,
  monacoEditorDispose: vi.fn(),
  monacoModelDispose: vi.fn()
}))

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom")
  return { ...actual, useNavigate: () => mocks.navigate }
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
  terminate = vi.fn()
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
    mocks.editorProps = null
    mocks.monacoEditorDispose.mockReset()
    mocks.monacoModelDispose.mockReset()
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
    const localWrite = vi.spyOn(Storage.prototype, "setItem")
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
    expect(screen.getByText("Safe outline — text only; code never runs in Studio")).toBeVisible()
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

  it("aborts an in-flight conflict refresh before logout can return source", async () => {
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
    fireEvent.click(await screen.findByRole("button", { name: "Confirm overwrite" }))
    await waitFor(() => expect(mocks.getPresentation).toHaveBeenCalledTimes(2))
    const refreshSignal = mocks.getPresentation.mock.calls[1][1].abortSignal as AbortSignal

    act(() =>
      window.dispatchEvent(
        new CustomEvent("tldw:auth-principal-changed", { detail: { kind: "logout" } })
      )
    )

    expect(refreshSignal.aborted).toBe(true)
    expect(document.body.textContent).not.toContain(EDITED)
    resolveRefresh?.(detail({ html_document: "private response after logout" }))
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
    expect(mocks.navigate).toHaveBeenCalledWith("/presentation-studio")
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
