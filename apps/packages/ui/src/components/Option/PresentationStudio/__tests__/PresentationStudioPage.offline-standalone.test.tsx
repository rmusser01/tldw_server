import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { Link, useLocation } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { MemoryRouterWithFuture } from "@/entries/shared/router-utils"
import { usePresentationStudioStore } from "@/store/presentation-studio"
import { PresentationStudioPage } from "../PresentationStudioPage"

const mocks = vi.hoisted(() => ({
  online: true,
  getConfig: vi.fn(),
  getCurrentUser: vi.fn(),
  getPresentationMetadata: vi.fn(),
  getPresentation: vi.fn(),
  listVisualStyles: vi.fn(),
  saveStandaloneHtmlSource: vi.fn(),
  downloadStandaloneHtmlDraft: vi.fn(),
  serverCapabilities: null as any,
  slidesCapabilities: null as any,
  editorProps: null as Record<string, any> | null
}))

const readySlidesCapabilities = {
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

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => mocks.online
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => mocks.serverCapabilities
}))

vi.mock("@/hooks/useSlidesCapabilities", () => ({
  useSlidesCapabilities: () => mocks.slidesCapabilities
}))

vi.mock("@/utils/browser-runtime", () => ({
  isExtensionRuntime: () => false,
  getBrowserRuntime: () => null
}))

vi.mock("@/services/tldw/TldwApiClient", async () => {
  const actual = await vi.importActual<Record<string, any>>("@/services/tldw/TldwApiClient")
  return {
    ...actual,
    tldwClient: {
      ...actual.tldwClient,
      getConfig: (...args: any[]) => mocks.getConfig(...args),
      getPresentationMetadata: (...args: any[]) => mocks.getPresentationMetadata(...args),
      getPresentation: (...args: any[]) => mocks.getPresentation(...args),
      listVisualStyles: (...args: any[]) => mocks.listVisualStyles(...args),
      saveStandaloneHtmlSource: (...args: any[]) => mocks.saveStandaloneHtmlSource(...args),
      downloadStandaloneHtmlDraft: (...args: any[]) => mocks.downloadStandaloneHtmlDraft(...args)
    }
  }
})

vi.mock("@/services/tldw/TldwAuth", async () => {
  const actual = await vi.importActual<Record<string, any>>("@/services/tldw/TldwAuth")
  return {
    ...actual,
    tldwAuth: {
      ...actual.tldwAuth,
      getCurrentUser: (...args: any[]) => mocks.getCurrentUser(...args)
    }
  }
})

vi.mock("@monaco-editor/react", async () => {
  const ReactModule = await import("react")
  const Monaco = (props: Record<string, any>) => {
    mocks.editorProps = props
    ReactModule.useEffect(() => {
      const model = { dispose: vi.fn() }
      props.onMount?.(
        {
          getModel: () => model,
          getDomNode: () => document.querySelector("[data-routed-monaco]"),
          getValue: () => props.value,
          setValue: vi.fn(),
          updateOptions: vi.fn(),
          dispose: vi.fn()
        },
        { languages: {} }
      )
    }, [props])
    return (
      <textarea
        data-routed-monaco
        id={props.wrapperProps?.id}
        aria-label="HTML source"
        value={props.value}
        onChange={(event) => props.onChange?.(event.currentTarget.value)}
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

const SOURCE = "<!doctype html><title>Deck</title>"
const SOURCE_DIGEST = "860887583dae29d0a221e3c9315a092fc6b271dd5d11cbe6e89be21a5260223d"
const EDITED = "<!doctype html><title>Edited</title>"
const SECOND_EDIT = "<!doctype html><title>Deferred routed edit</title>"

const detail = () => ({
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
    generation_provenance: { source_kind: "prompt", provider: "test", model: "model" }
  },
  etag: '"v7"'
})

const structuredDetail = () => ({
  record: {
    id: "html-1",
    content_kind: "structured_slides",
    title: "New authority structured deck",
    description: null,
    theme: "black",
    visual_style_id: null,
    visual_style_scope: null,
    visual_style_name: null,
    visual_style_version: null,
    visual_style_snapshot: null,
    slides: [{
      order: 0,
      layout: "title",
      title: "New authority slide",
      content: "",
      speaker_notes: "",
      metadata: {
        studio: {
          slideId: "new-authority-slide",
          audio: { status: "missing" },
          image: { status: "missing" }
        }
      }
    }],
    studio_data: { origin: "blank" },
    created_at: "2026-07-15T00:00:00Z",
    last_modified: "2026-07-15T00:00:02Z",
    deleted: false,
    client_id: "84",
    version: 1
  },
  etag: '"structured-v1"'
})

const RoutedStandalonePage: React.FC<{ revision: number }> = () => {
  const location = useLocation()
  return (
    <>
      <Link to="/outside">Leave presentation route</Link>
      <output aria-label="Current routed page">{location.pathname}</output>
      <PresentationStudioPage mode="detail" projectId="html-1" />
    </>
  )
}

const renderRoutedPage = () => {
  let revision = 0
  const view = render(
    <MemoryRouterWithFuture>
      <RoutedStandalonePage revision={revision} />
    </MemoryRouterWithFuture>
  )
  return {
    ...view,
    refresh: () => {
      revision += 1
      view.rerender(
        <MemoryRouterWithFuture>
          <RoutedStandalonePage revision={revision} />
        </MemoryRouterWithFuture>
      )
    }
  }
}

const editSource = async (source: string) => {
  await waitFor(() => expect(mocks.editorProps?.onChange).toBeTypeOf("function"))
  act(() => mocks.editorProps?.onChange?.(source))
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

const persistedRecoverySource = (): string | null => {
  const key = sessionStorage.key(0)
  if (!key) return null
  const raw = sessionStorage.getItem(key)
  return raw ? (JSON.parse(raw) as { source?: string }).source ?? null : null
}

describe("PresentationStudioPage routed standalone offline lifecycle", () => {
  beforeEach(() => {
    sessionStorage.clear()
    localStorage.clear()
    usePresentationStudioStore.getState().reset()
    mocks.online = true
    mocks.getConfig.mockReset().mockResolvedValue({ serverUrl: "https://TLDW.Example/path" })
    mocks.getCurrentUser.mockReset().mockResolvedValue({ id: 42, username: "owner", is_active: true })
    mocks.getPresentationMetadata.mockReset().mockResolvedValue({
      record: { id: "html-1", content_kind: "standalone_html" },
      etag: null
    })
    mocks.getPresentation.mockReset().mockResolvedValue(detail())
    mocks.listVisualStyles.mockReset().mockResolvedValue([])
    mocks.saveStandaloneHtmlSource.mockReset()
    mocks.downloadStandaloneHtmlDraft.mockReset().mockResolvedValue(new TextEncoder().encode(SOURCE))
    mocks.serverCapabilities = {
      loading: false,
      capabilities: {
        hasSlides: true,
        hasPresentationStudio: true,
        hasPresentationRender: true
      }
    }
    mocks.slidesCapabilities = readySlidesCapabilities
    mocks.editorProps = null
    Object.defineProperty(globalThis, "Worker", {
      configurable: true,
      writable: true,
      value: ImmediateOutlineWorker
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("keeps accepted dirty authority mounted and route-guarded through the standalone offline shell", async () => {
    const user = userEvent.setup()
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false)
    const view = renderRoutedPage()
    const editor = await screen.findByLabelText("HTML source")
    vi.spyOn(Object.getPrototypeOf(sessionStorage), "setItem").mockImplementation(() => {
      throw new DOMException("quota", "QuotaExceededError")
    })

    await editSource(EDITED)
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    expect(screen.getByText(/Recovery unavailable/i)).toBeVisible()
    expect(mocks.getPresentation).toHaveBeenCalledTimes(1)

    mocks.online = false
    view.refresh()

    expect(screen.getByText("Server is offline. Your in-memory draft has not been sent.")).toBeVisible()
    expect(screen.getByText(/Recovery unavailable/i)).toBeVisible()
    await user.click(screen.getByRole("link", { name: "Leave presentation route" }))
    expect(confirm).toHaveBeenCalledWith(
      "Leave without saving? Your local draft is preserved only in this tab."
    )
    expect(screen.getByRole("status", { name: "Current routed page" })).toHaveTextContent("/")

    mocks.online = true
    view.refresh()
    expect(await screen.findByLabelText("HTML source")).toHaveValue(EDITED)
    expect(mocks.getPresentation).toHaveBeenCalledTimes(1)
  })

  it("restores and revalidates the exact deferred candidate after routed offline recovery fails", async () => {
    const view = renderRoutedPage()
    const editor = await screen.findByLabelText("HTML source")
    vi.spyOn(Object.getPrototypeOf(sessionStorage), "setItem").mockImplementation(() => {
      throw new DOMException("quota", "QuotaExceededError")
    })
    await editSource(EDITED)
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
    expect(screen.getByText(/Recovery unavailable/i)).toBeVisible()

    const digestResult = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(SECOND_EDIT)
    )
    let resolveRetiredDigest: ((value: ArrayBuffer) => void) | null = null
    const digest = vi.spyOn(crypto.subtle, "digest")
      .mockReturnValueOnce(new Promise<ArrayBuffer>((resolve) => { resolveRetiredDigest = resolve }))
      .mockResolvedValueOnce(digestResult)
    act(() => mocks.editorProps?.onChange?.(SECOND_EDIT))
    expect(screen.getByLabelText("HTML source")).toHaveValue(SECOND_EDIT)
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()

    mocks.online = false
    view.refresh()
    expect(screen.getByText("Server is offline. Your in-memory draft has not been sent.")).toBeVisible()

    mocks.online = true
    view.refresh()
    expect(await screen.findByLabelText("HTML source")).toHaveValue(SECOND_EDIT)
    await waitFor(() => expect(digest).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(screen.getByRole("button", { name: "Save" })).toBeEnabled())
    expect(screen.getByLabelText("HTML source")).toHaveValue(SECOND_EDIT)
    expect(mocks.getPresentation).toHaveBeenCalledTimes(1)

    resolveRetiredDigest?.(digestResult)
    await act(async () => Promise.resolve())
    expect(screen.getByLabelText("HTML source")).toHaveValue(SECOND_EDIT)
  })

  it("retains the keyed workspace and exact pending draft through same-scope metadata reauthentication", async () => {
    const user = userEvent.setup()
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false)
    const view = renderRoutedPage()
    const editor = await screen.findByLabelText("HTML source")
    vi.spyOn(Object.getPrototypeOf(sessionStorage), "setItem").mockImplementation(() => {
      throw new DOMException("quota", "QuotaExceededError")
    })
    await editSource(EDITED)
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())

    const digestResult = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(SECOND_EDIT)
    )
    let resolveRetiredDigest: ((value: ArrayBuffer) => void) | null = null
    const digest = vi.spyOn(crypto.subtle, "digest")
      .mockReturnValueOnce(new Promise<ArrayBuffer>((resolve) => { resolveRetiredDigest = resolve }))
      .mockResolvedValueOnce(digestResult)
    act(() => mocks.editorProps?.onChange?.(SECOND_EDIT))
    expect(screen.getByLabelText("HTML source")).toHaveValue(SECOND_EDIT)

    let resolveConfig: ((value: { serverUrl: string }) => void) | null = null
    let resolveMetadata: ((value: any) => void) | null = null
    mocks.getConfig.mockReturnValueOnce(
      new Promise((resolve) => { resolveConfig = resolve })
    )
    mocks.getPresentationMetadata.mockReturnValueOnce(
      new Promise((resolve) => { resolveMetadata = resolve })
    )
    fireEvent(window, new Event("tldw:config-updated"))

    expect(await screen.findByText("Confirming current server and account…")).toBeVisible()
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
    expect(screen.getByText(/Recovery unavailable/i)).toBeVisible()
    await user.click(screen.getByRole("link", { name: "Leave presentation route" }))
    expect(confirm).toHaveBeenCalledTimes(1)
    expect(screen.getByRole("status", { name: "Current routed page" })).toHaveTextContent("/")
    expect(mocks.getPresentationMetadata).toHaveBeenCalledTimes(2)
    expect(mocks.getPresentation).toHaveBeenCalledTimes(1)

    resolveConfig?.({ serverUrl: "https://TLDW.Example/path" })
    expect(await screen.findByLabelText("HTML source")).toHaveValue(SECOND_EDIT)
    await waitFor(() => expect(digest).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(screen.getByRole("button", { name: "Save" })).toBeEnabled())
    expect(mocks.getPresentation).toHaveBeenCalledTimes(1)

    resolveMetadata?.({
      record: { id: "html-1", content_kind: "standalone_html" },
      etag: null
    })
    await waitFor(() => expect(mocks.getPresentationMetadata).toHaveBeenCalledTimes(2))
    expect(screen.getByLabelText("HTML source")).toHaveValue(SECOND_EDIT)
    expect(mocks.getPresentation).toHaveBeenCalledTimes(1)

    resolveRetiredDigest?.(digestResult)
    await act(async () => Promise.resolve())
    expect(screen.getByLabelText("HTML source")).toHaveValue(SECOND_EDIT)
    view.unmount()
  })

  it("buffers an immediate structured result until the retained workspace settles same-scope reauthentication", async () => {
    renderRoutedPage()
    const editor = await screen.findByLabelText("HTML source")
    await editSource(EDITED)
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())

    let resolveConfig: ((value: { serverUrl: string }) => void) | null = null
    mocks.getConfig.mockReturnValueOnce(
      new Promise((resolve) => { resolveConfig = resolve })
    )
    mocks.getPresentationMetadata.mockResolvedValueOnce({
      record: { id: "html-1", content_kind: "structured_slides" },
      etag: null
    })
    mocks.getPresentation.mockResolvedValueOnce(structuredDetail())

    fireEvent(window, new Event("tldw:config-updated"))

    await waitFor(() => expect(mocks.getPresentation).toHaveBeenCalledTimes(2))
    expect(screen.getByText("Confirming current server and account…")).toBeVisible()
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
    expect(screen.queryByText(/New authority structured deck/)).not.toBeInTheDocument()
    expect(usePresentationStudioStore.getState().title).not.toBe("New authority structured deck")

    resolveConfig?.({ serverUrl: "https://TLDW.Example/path" })

    expect(await screen.findByText(/New authority structured deck/)).toBeVisible()
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
  })

  it.each([
    ["an accepted edit hits a storage getter failure before structured metadata resolves", "accepted", "getter", "structured"],
    ["a digest-pending edit hits a storage write failure before metadata rejects", "pending", "setItem", "error"]
  ])("revokes cached release authority when %s", async (
    _scenario,
    candidateKind,
    failureKind,
    outcomeKind
  ) => {
    const user = userEvent.setup()
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false)
    renderRoutedPage()
    await screen.findByLabelText("HTML source")

    let resolveConfig: ((value: { serverUrl: string }) => void) | null = null
    let resolveMetadata: ((value: any) => void) | null = null
    let rejectMetadata: ((reason?: unknown) => void) | null = null
    mocks.getConfig.mockReturnValueOnce(
      new Promise((resolve) => { resolveConfig = resolve })
    )
    mocks.getPresentationMetadata.mockReturnValueOnce(
      new Promise((resolve, reject) => {
        resolveMetadata = resolve
        rejectMetadata = reject
      })
    )
    fireEvent(window, new Event("tldw:config-updated"))

    expect(await screen.findByText("Confirming current server and account…")).toBeVisible()
    resolveConfig?.({ serverUrl: "https://TLDW.Example/path" })
    expect(await screen.findByLabelText("HTML source")).toHaveValue(SOURCE)
    expect(mocks.getPresentationMetadata).toHaveBeenCalledTimes(2)

    const restoreStorage = failureKind === "getter"
      ? installThrowingSessionStorageGetter()
      : (() => {
          const setItem = vi.spyOn(Object.getPrototypeOf(sessionStorage), "setItem")
            .mockImplementation(() => {
              throw new DOMException("quota", "QuotaExceededError")
            })
          return () => setItem.mockRestore()
        })()
    let newestCandidate = EDITED
    let resolveCandidateDigest: ((value: ArrayBuffer) => void) | null = null
    let candidateDigest: ArrayBuffer | null = null

    try {
      if (candidateKind === "pending") {
        newestCandidate = SECOND_EDIT
        candidateDigest = await crypto.subtle.digest(
          "SHA-256",
          new TextEncoder().encode(newestCandidate)
        )
        vi.spyOn(crypto.subtle, "digest").mockReturnValueOnce(
          new Promise<ArrayBuffer>((resolve) => { resolveCandidateDigest = resolve })
        )
        act(() => mocks.editorProps?.onChange?.(newestCandidate))
        expect(screen.getByLabelText("HTML source")).toHaveValue(newestCandidate)
        expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()
      } else {
        await editSource(newestCandidate)
        await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
      }

      if (outcomeKind === "structured") {
        mocks.getPresentation.mockResolvedValueOnce(structuredDetail())
        resolveMetadata?.({
          record: { id: "html-1", content_kind: "structured_slides" },
          etag: null
        })
        await waitFor(() => expect(mocks.getPresentation).toHaveBeenCalledTimes(2))
      } else {
        rejectMetadata?.(new Error("Delayed authority metadata failed."))
      }

      expect(await screen.findByText(/Recovery unavailable/i)).toBeVisible()
      expect(screen.getByText("Confirming current server and account…")).toBeVisible()
      expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
      expect(screen.queryByText(/New authority structured deck/)).not.toBeInTheDocument()
      expect(screen.queryByText("Delayed authority metadata failed.")).not.toBeInTheDocument()
      await user.click(screen.getByRole("link", { name: "Leave presentation route" }))
      expect(confirm).toHaveBeenCalledWith(
        "Leave without saving? Your local draft is preserved only in this tab."
      )
      expect(screen.getByRole("status", { name: "Current routed page" })).toHaveTextContent("/")
    } finally {
      restoreStorage()
    }

    mocks.getConfig.mockResolvedValueOnce({ serverUrl: "https://TLDW.Example/path" })
    fireEvent(window, new Event("focus"))

    if (outcomeKind === "structured") {
      expect(await screen.findByText(/New authority structured deck/)).toBeVisible()
    } else {
      expect(await screen.findByText("Delayed authority metadata failed.")).toBeVisible()
    }
    expect(persistedRecoverySource()).toBe(newestCandidate)
    if (resolveCandidateDigest && candidateDigest) resolveCandidateDigest(candidateDigest)
  })

  it.each([
    ["capability discovery errors", { ...readySlidesCapabilities, status: "error", canReadStandalone: false }],
    ["standalone read is unavailable", {
      ...readySlidesCapabilities,
      status: "ready",
      canReadStandalone: false,
      canDraftStandalone: false,
      canEditStandalone: false
    }]
  ])("settles a buffered structured handoff when %s after same-scope reauthentication", async (
    _scenario,
    settledCapabilities
  ) => {
    const view = renderRoutedPage()
    await screen.findByLabelText("HTML source")

    mocks.slidesCapabilities = { ...readySlidesCapabilities, status: "loading" }
    view.refresh()
    expect(await screen.findByText(/Checking standalone HTML access/i)).toBeVisible()

    let resolveConfig: ((value: { serverUrl: string }) => void) | null = null
    mocks.getConfig.mockReturnValueOnce(
      new Promise((resolve) => { resolveConfig = resolve })
    )
    mocks.getPresentationMetadata.mockResolvedValueOnce({
      record: { id: "html-1", content_kind: "structured_slides" },
      etag: null
    })
    mocks.getPresentation.mockResolvedValueOnce(structuredDetail())

    fireEvent(window, new Event("tldw:config-updated"))
    await waitFor(() => expect(mocks.getPresentation).toHaveBeenCalledTimes(2))
    expect(screen.queryByText(/New authority structured deck/)).not.toBeInTheDocument()

    resolveConfig?.({ serverUrl: "https://TLDW.Example/path" })
    mocks.slidesCapabilities = settledCapabilities
    view.refresh()

    expect(await screen.findByText(/New authority structured deck/)).toBeVisible()
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
  })

  it.each(["getter", "setItem"])(
    "keeps quarantined dirty authority guarded when standalone read denial follows a storage %s failure",
    async (failureKind) => {
      const user = userEvent.setup()
      const confirm = vi.spyOn(window, "confirm").mockReturnValue(false)
      const view = renderRoutedPage()
      await screen.findByLabelText("HTML source")
      await editSource(EDITED)
      await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())

      const digestResult = await crypto.subtle.digest(
        "SHA-256",
        new TextEncoder().encode(SECOND_EDIT)
      )
      let resolvePendingDigest: ((value: ArrayBuffer) => void) | null = null
      vi.spyOn(crypto.subtle, "digest").mockReturnValueOnce(
        new Promise<ArrayBuffer>((resolve) => { resolvePendingDigest = resolve })
      )
      act(() => mocks.editorProps?.onChange?.(SECOND_EDIT))
      expect(screen.getByLabelText("HTML source")).toHaveValue(SECOND_EDIT)
      expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()

      const restoreStorage = failureKind === "getter"
        ? installThrowingSessionStorageGetter()
        : (() => {
            const setItem = vi.spyOn(Object.getPrototypeOf(sessionStorage), "setItem")
              .mockImplementation(() => {
                throw new DOMException("quota", "QuotaExceededError")
              })
            return () => setItem.mockRestore()
          })()

      try {
        mocks.slidesCapabilities = { ...readySlidesCapabilities, status: "loading" }
        view.refresh()
        expect(await screen.findByText(/Checking standalone HTML access/i)).toBeVisible()

        mocks.getPresentationMetadata.mockResolvedValueOnce({
          record: { id: "html-1", content_kind: "structured_slides" },
          etag: null
        })
        mocks.getPresentation.mockResolvedValueOnce(structuredDetail())
        fireEvent(window, new Event("tldw:config-updated"))

        await waitFor(() => expect(mocks.getPresentation).toHaveBeenCalledTimes(2))
        expect(await screen.findByText(/Recovery unavailable/i)).toBeVisible()
        expect(screen.getByText("Confirming current server and account…")).toBeVisible()
        expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
        expect(screen.queryByText(/New authority structured deck/)).not.toBeInTheDocument()

        mocks.slidesCapabilities = {
          ...readySlidesCapabilities,
          status: "ready",
          canReadStandalone: false,
          canDraftStandalone: false,
          canEditStandalone: false
        }
        view.refresh()

        expect(screen.getByText("Confirming current server and account…")).toBeVisible()
        expect(screen.getByText(/Recovery unavailable/i)).toBeVisible()
        expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
        expect(screen.queryByText(/New authority structured deck/)).not.toBeInTheDocument()
        await user.click(screen.getByRole("link", { name: "Leave presentation route" }))
        expect(confirm).toHaveBeenCalledWith(
          "Leave without saving? Your local draft is preserved only in this tab."
        )
        expect(screen.getByRole("status", { name: "Current routed page" })).toHaveTextContent("/")
      } finally {
        restoreStorage()
      }

      mocks.getConfig.mockResolvedValueOnce({ serverUrl: "https://TLDW.Example/path" })
      fireEvent(window, new Event("focus"))

      expect(await screen.findByText(/New authority structured deck/)).toBeVisible()
      expect(persistedRecoverySource()).toBe(SECOND_EDIT)
      expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
      resolvePendingDigest?.(digestResult)
    }
  )

  it("preserves same-scope quarantine when buffered metadata remains standalone across a capability error", async () => {
    const view = renderRoutedPage()
    await screen.findByLabelText("HTML source")
    vi.spyOn(Object.getPrototypeOf(sessionStorage), "setItem").mockImplementation(() => {
      throw new DOMException("quota", "QuotaExceededError")
    })
    await editSource(EDITED)
    await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())

    mocks.slidesCapabilities = { ...readySlidesCapabilities, status: "loading" }
    view.refresh()
    expect(await screen.findByText(/Checking standalone HTML access/i)).toBeVisible()

    let resolveConfig: ((value: { serverUrl: string }) => void) | null = null
    mocks.getConfig.mockReturnValueOnce(
      new Promise((resolve) => { resolveConfig = resolve })
    )
    mocks.getPresentationMetadata.mockResolvedValueOnce({
      record: { id: "html-1", content_kind: "standalone_html" },
      etag: null
    })
    fireEvent(window, new Event("tldw:config-updated"))
    await waitFor(() => expect(mocks.getPresentationMetadata).toHaveBeenCalledTimes(2))

    resolveConfig?.({ serverUrl: "https://TLDW.Example/path" })
    mocks.slidesCapabilities = {
      ...readySlidesCapabilities,
      status: "error",
      canReadStandalone: false
    }
    view.refresh()
    expect(await screen.findByText(/Standalone HTML access could not be confirmed/i)).toBeVisible()

    mocks.slidesCapabilities = readySlidesCapabilities
    view.refresh()
    expect(await screen.findByLabelText("HTML source")).toHaveValue(EDITED)
    expect(mocks.getPresentation).toHaveBeenCalledTimes(1)
    expect(screen.getByText(/Recovery unavailable/i)).toBeVisible()
  })

  it("publishes an authority-refresh error only after the same-scope workspace settles", async () => {
    renderRoutedPage()
    await screen.findByLabelText("HTML source")

    let resolveConfig: ((value: { serverUrl: string }) => void) | null = null
    mocks.getConfig.mockReturnValueOnce(
      new Promise((resolve) => { resolveConfig = resolve })
    )
    mocks.getPresentationMetadata.mockRejectedValueOnce(
      new Error("Metadata authority refresh failed.")
    )
    fireEvent(window, new Event("tldw:config-updated"))

    await waitFor(() => expect(mocks.getPresentationMetadata).toHaveBeenCalledTimes(2))
    expect(screen.getByText("Confirming current server and account…")).toBeVisible()
    expect(screen.queryByText("Metadata authority refresh failed.")).not.toBeInTheDocument()

    resolveConfig?.({ serverUrl: "https://TLDW.Example/path" })

    expect(await screen.findByText("Metadata authority refresh failed.")).toBeVisible()
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
  })

  it("keeps an identity-refresh error buffered until failed old-scope cleanup succeeds", async () => {
    renderRoutedPage()
    await screen.findByLabelText("HTML source")
    await editSource(EDITED)
    await waitFor(() => expect(sessionStorage.length).toBe(1))

    const removeItem = vi.spyOn(Object.getPrototypeOf(sessionStorage), "removeItem")
      .mockImplementation(() => {
        throw new DOMException("blocked", "SecurityError")
      })
    let resolveConfig: ((value: { serverUrl: string }) => void) | null = null
    mocks.getConfig.mockReturnValueOnce(
      new Promise((resolve) => { resolveConfig = resolve })
    )
    mocks.getPresentationMetadata.mockRejectedValueOnce(
      new Error("New authority metadata failed.")
    )
    fireEvent(window, new Event("tldw:config-updated"))
    await waitFor(() => expect(mocks.getPresentationMetadata).toHaveBeenCalledTimes(2))

    resolveConfig?.({ serverUrl: "https://other.example/path" })
    expect(await screen.findByText(/Recovery unavailable/i)).toBeVisible()
    expect(screen.getByText("Confirming current server and account…")).toBeVisible()
    expect(screen.queryByText("New authority metadata failed.")).not.toBeInTheDocument()
    expect(sessionStorage.length).toBe(1)

    removeItem.mockRestore()
    mocks.getConfig.mockResolvedValueOnce({ serverUrl: "https://other.example/path" })
    fireEvent(window, new Event("focus"))

    await waitFor(() => expect(sessionStorage.length).toBe(0))
    expect(await screen.findByText("New authority metadata failed.")).toBeVisible()
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
  })

  it.each(["getter", "setItem"])(
    "keeps a dirty new-kind handoff guarded until the exact draft survives a sessionStorage %s failure",
    async (failureKind) => {
      const user = userEvent.setup()
      const confirm = vi.spyOn(window, "confirm").mockReturnValue(false)
      renderRoutedPage()
      await screen.findByLabelText("HTML source")

      const restoreStorage = failureKind === "getter"
        ? installThrowingSessionStorageGetter()
        : (() => {
            const setItem = vi.spyOn(Object.getPrototypeOf(sessionStorage), "setItem")
              .mockImplementation(() => {
                throw new DOMException("quota", "QuotaExceededError")
              })
            return () => setItem.mockRestore()
          })()
      try {
        await editSource(EDITED)
        await waitFor(() => expect(screen.getByText("Not saved")).toBeVisible())
        expect(screen.getByText(/Recovery unavailable/i)).toBeVisible()

        let resolveConfig: ((value: { serverUrl: string }) => void) | null = null
        mocks.getConfig.mockReturnValueOnce(
          new Promise((resolve) => { resolveConfig = resolve })
        )
        mocks.getPresentationMetadata.mockResolvedValueOnce({
          record: { id: "html-1", content_kind: "structured_slides" },
          etag: null
        })
        mocks.getPresentation.mockResolvedValueOnce(structuredDetail())
        fireEvent(window, new Event("tldw:config-updated"))
        await waitFor(() => expect(mocks.getPresentation).toHaveBeenCalledTimes(2))

        resolveConfig?.({ serverUrl: "https://TLDW.Example/path" })
        expect(await screen.findByText("Confirming current server and account…")).toBeVisible()
        expect(screen.getByText(/Recovery unavailable/i)).toBeVisible()
        expect(screen.queryByText(/New authority structured deck/)).not.toBeInTheDocument()
        await user.click(screen.getByRole("link", { name: "Leave presentation route" }))
        expect(confirm).toHaveBeenCalledTimes(1)
        expect(screen.getByRole("status", { name: "Current routed page" })).toHaveTextContent("/")
      } finally {
        restoreStorage()
      }

      mocks.getConfig.mockResolvedValueOnce({ serverUrl: "https://TLDW.Example/path" })
      fireEvent(window, new Event("focus"))

      expect(await screen.findByText(/New authority structured deck/)).toBeVisible()
      expect(persistedRecoverySource()).toBe(EDITED)
      expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
    }
  )

  it("persists the exact digest-pending candidate before a same-scope structured handoff", async () => {
    renderRoutedPage()
    await screen.findByLabelText("HTML source")

    vi.spyOn(crypto.subtle, "digest").mockReturnValueOnce(new Promise<ArrayBuffer>(() => {}))
    act(() => mocks.editorProps?.onChange?.(SECOND_EDIT))
    expect(screen.getByLabelText("HTML source")).toHaveValue(SECOND_EDIT)
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()

    mocks.getPresentationMetadata.mockResolvedValueOnce({
      record: { id: "html-1", content_kind: "structured_slides" },
      etag: null
    })
    mocks.getPresentation.mockResolvedValueOnce(structuredDetail())
    fireEvent(window, new Event("tldw:config-updated"))

    expect(await screen.findByText(/New authority structured deck/)).toBeVisible()
    expect(persistedRecoverySource()).toBe(SECOND_EDIT)
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
  })

  it("does not fetch new-scope HTML while exact standalone metadata remains pending", async () => {
    renderRoutedPage()
    await screen.findByLabelText("HTML source")
    await editSource(EDITED)
    await waitFor(() => expect(sessionStorage.length).toBe(1))

    let resolveMetadata: ((value: any) => void) | null = null
    mocks.getConfig.mockResolvedValueOnce({ serverUrl: "https://other.example/path" })
    mocks.getPresentationMetadata.mockReturnValueOnce(
      new Promise((resolve) => { resolveMetadata = resolve })
    )
    fireEvent(window, new Event("tldw:config-updated"))

    await waitFor(() => expect(sessionStorage.length).toBe(0))
    expect(screen.getByText("Confirming current server and account…")).toBeVisible()
    expect(mocks.getPresentation).toHaveBeenCalledTimes(1)
    await act(async () => Promise.resolve())
    expect(mocks.getPresentation).toHaveBeenCalledTimes(1)

    resolveMetadata?.({
      record: { id: "html-1", content_kind: "standalone_html" },
      etag: null
    })

    await waitFor(() => expect(mocks.getPresentation).toHaveBeenCalledTimes(2))
    expect(await screen.findByLabelText("HTML source")).toHaveValue(SOURCE)
  })

  it("does not release an immediate structured handoff before a switched principal resolves", async () => {
    renderRoutedPage()
    await screen.findByLabelText("HTML source")

    let resolveUser: ((value: { id: number; username: string; is_active: boolean }) => void) | null = null
    mocks.getCurrentUser.mockReturnValueOnce(
      new Promise((resolve) => { resolveUser = resolve })
    )
    mocks.getPresentationMetadata.mockResolvedValueOnce({
      record: { id: "html-1", content_kind: "structured_slides" },
      etag: null
    })
    mocks.getPresentation.mockResolvedValueOnce(structuredDetail())
    fireEvent(window, new CustomEvent("tldw:auth-principal-changed", {
      detail: { kind: "switch" }
    }))

    await waitFor(() => expect(mocks.getPresentation).toHaveBeenCalledTimes(2))
    expect(screen.getByText("Confirming current server and account…")).toBeVisible()
    expect(screen.queryByText(/New authority structured deck/)).not.toBeInTheDocument()

    resolveUser?.({ id: 84, username: "other", is_active: true })

    expect(await screen.findByText(/New authority structured deck/)).toBeVisible()
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
  })

  it("keeps an immediate unsupported handoff guarded after direct scope mismatch until reauthentication", async () => {
    renderRoutedPage()
    await screen.findByLabelText("HTML source")
    await editSource(EDITED)
    await waitFor(() => expect(sessionStorage.length).toBe(1))

    mocks.getPresentationMetadata.mockResolvedValueOnce({
      record: {
        id: "html-1",
        content_kind: "unsupported",
        unsupported_content_kind: "future_canvas",
        read_only: true
      },
      etag: null
    })
    fireEvent(window, new Event("tldw:slides-scope-mismatch"))

    await waitFor(() => expect(mocks.getPresentationMetadata).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(sessionStorage.length).toBe(0))
    expect(screen.getByText(/Current server and account could not be confirmed/i)).toBeVisible()
    expect(screen.queryByText("Unsupported presentation kind")).not.toBeInTheDocument()
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()

    mocks.getConfig.mockResolvedValueOnce({ serverUrl: "https://TLDW.Example/path" })
    mocks.getCurrentUser.mockResolvedValueOnce({ id: 42, username: "owner", is_active: true })
    fireEvent(window, new Event("focus"))

    expect(await screen.findByText("Unsupported presentation kind")).toBeVisible()
    expect(screen.getByText("future_canvas")).toBeVisible()
  })

  it.each([
    [
      "origin",
      () => {
        let resolveIdentity: ((value: { serverUrl: string }) => void) | null = null
        mocks.getConfig.mockReturnValueOnce(
          new Promise((resolve) => { resolveIdentity = resolve })
        )
        return () => resolveIdentity?.({ serverUrl: "https://other.example/path" })
      }
    ],
    [
      "principal",
      () => {
        let resolveIdentity: ((value: { id: number; username: string; is_active: boolean }) => void) | null = null
        mocks.getCurrentUser.mockReturnValueOnce(
          new Promise((resolve) => { resolveIdentity = resolve })
        )
        return () => resolveIdentity?.({ id: 84, username: "other", is_active: true })
      }
    ]
  ])("buffers immediate unsupported metadata until the workspace scrubs the old %s scope", async (
    _boundary,
    deferIdentity
  ) => {
    renderRoutedPage()
    const editor = await screen.findByLabelText("HTML source")
    await editSource(EDITED)
    await waitFor(() => expect(sessionStorage.length).toBe(1))

    const resolveIdentity = deferIdentity()
    mocks.getPresentationMetadata.mockResolvedValueOnce({
      record: {
        id: "html-1",
        content_kind: "unsupported",
        unsupported_content_kind: "future_canvas",
        read_only: true
      },
      etag: null
    })
    fireEvent(window, new Event("tldw:config-updated"))

    await waitFor(() => expect(mocks.getPresentationMetadata).toHaveBeenCalledTimes(2))
    expect(screen.getByText("Confirming current server and account…")).toBeVisible()
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
    expect(screen.queryByText("Unsupported presentation kind")).not.toBeInTheDocument()
    expect(sessionStorage.length).toBe(1)
    expect(mocks.getPresentation).toHaveBeenCalledTimes(1)

    resolveIdentity()

    await waitFor(() => expect(sessionStorage.length).toBe(0))
    expect(await screen.findByText("Unsupported presentation kind")).toBeVisible()
    expect(screen.getByText("future_canvas")).toBeVisible()
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
  })

  it("keeps the guarded standalone owner mounted ahead of the global capability-unavailable shell", async () => {
    const view = renderRoutedPage()
    const editor = await screen.findByLabelText("HTML source")
    await editSource(EDITED)
    await waitFor(() => expect(sessionStorage.length).toBe(1))

    let resolveConfig: ((value: { serverUrl: string }) => void) | null = null
    mocks.getConfig.mockReturnValueOnce(
      new Promise((resolve) => { resolveConfig = resolve })
    )
    mocks.getPresentationMetadata.mockResolvedValueOnce({
      record: {
        id: "html-1",
        content_kind: "unsupported",
        unsupported_content_kind: "future_canvas",
        read_only: true
      },
      etag: null
    })
    mocks.serverCapabilities = {
      loading: false,
      capabilities: {
        hasSlides: false,
        hasPresentationStudio: false,
        hasPresentationRender: false
      }
    }

    fireEvent(window, new Event("tldw:config-updated"))
    view.refresh()

    await waitFor(() => expect(mocks.getPresentationMetadata).toHaveBeenCalledTimes(2))
    expect(screen.getByText("Confirming current server and account…")).toBeVisible()
    expect(screen.queryByText("Presentation Studio is not available on this server.")).not.toBeInTheDocument()
    expect(screen.queryByText("Unsupported presentation kind")).not.toBeInTheDocument()

    resolveConfig?.({ serverUrl: "https://TLDW.Example/path" })

    expect(await screen.findByText("Presentation Studio is not available on this server.")).toBeVisible()
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
  })

  it("does not evict the scrub owner when old-scope recovery removal fails", async () => {
    renderRoutedPage()
    const editor = await screen.findByLabelText("HTML source")
    await editSource(EDITED)
    await waitFor(() => expect(sessionStorage.length).toBe(1))

    const removeItem = vi.spyOn(Object.getPrototypeOf(sessionStorage), "removeItem")
      .mockImplementation(() => {
        throw new DOMException("blocked", "SecurityError")
      })
    let resolveConfig: ((value: { serverUrl: string }) => void) | null = null
    mocks.getConfig.mockReturnValueOnce(
      new Promise((resolve) => { resolveConfig = resolve })
    )
    mocks.getPresentationMetadata.mockResolvedValueOnce({
      record: {
        id: "html-1",
        content_kind: "unsupported",
        unsupported_content_kind: "future_canvas",
        read_only: true
      },
      etag: null
    })

    fireEvent(window, new Event("tldw:config-updated"))
    await waitFor(() => expect(mocks.getPresentationMetadata).toHaveBeenCalledTimes(2))
    resolveConfig?.({ serverUrl: "https://other.example/path" })

    expect(await screen.findByText(/Recovery unavailable/i)).toBeVisible()
    expect(screen.getByText("Confirming current server and account…")).toBeVisible()
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
    expect(screen.queryByText("Unsupported presentation kind")).not.toBeInTheDocument()
    expect(sessionStorage.length).toBe(1)

    removeItem.mockRestore()
    mocks.getConfig.mockResolvedValueOnce({ serverUrl: "https://other.example/path" })
    fireEvent(window, new Event("focus"))

    await waitFor(() => expect(sessionStorage.length).toBe(0))
    expect(await screen.findByText("Unsupported presentation kind")).toBeVisible()
    expect(screen.getByText("future_canvas")).toBeVisible()
  })

  it.each([
    [
      "origin",
      () => {
        mocks.getConfig.mockResolvedValueOnce({ serverUrl: "https://other.example/path" })
        return new Event("tldw:config-updated")
      }
    ],
    [
      "principal",
      () => {
        mocks.getCurrentUser.mockResolvedValueOnce({ id: 84, username: "other", is_active: true })
        return new CustomEvent("tldw:auth-principal-changed", {
          detail: { kind: "subject_changed" }
        })
      }
    ]
  ])("scrubs %s-scoped source and recovery before exact metadata changes the surface", async (
    _boundary,
    createBoundary
  ) => {
    const editor = await (async () => {
      renderRoutedPage()
      return screen.findByLabelText("HTML source")
    })()
    await editSource(EDITED)
    await waitFor(() => expect(sessionStorage.length).toBe(1))

    let resolveMetadata: ((value: any) => void) | null = null
    mocks.getPresentationMetadata.mockReturnValueOnce(
      new Promise((resolve) => { resolveMetadata = resolve })
    )
    mocks.getPresentation.mockResolvedValueOnce(structuredDetail())
    fireEvent(window, createBoundary())

    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
    expect(document.body.textContent).not.toContain(EDITED)
    expect(await screen.findByText("Confirming current server and account…")).toBeVisible()
    await waitFor(() => expect(sessionStorage.length).toBe(0))
    expect(mocks.getPresentationMetadata).toHaveBeenCalledTimes(2)
    expect(mocks.getPresentation).toHaveBeenCalledTimes(1)
    expect(screen.queryByText("New authority slide")).not.toBeInTheDocument()

    resolveMetadata?.({
      record: { id: "html-1", content_kind: "structured_slides" },
      etag: null
    })
    expect(await screen.findByText(/New authority structured deck/)).toBeVisible()
    expect(mocks.getPresentation).toHaveBeenCalledTimes(2)
    expect(screen.queryByLabelText("HTML source")).not.toBeInTheDocument()
  })
})
