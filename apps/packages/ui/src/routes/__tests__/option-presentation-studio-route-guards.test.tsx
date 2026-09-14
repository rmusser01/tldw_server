import React from "react"
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor
} from "@testing-library/react"
import {
  createMemoryRouter,
  MemoryRouter,
  Navigate,
  RouterProvider,
  Route,
  Routes
} from "react-router-dom"
import {
  afterAll,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi
} from "vitest"

import OptionPresentationStudio from "../option-presentation-studio"
import { getRouteMetadata } from "../route-metadata"

type Deferred<T> = {
  promise: Promise<T>
  resolve: (value: T) => void
  reject: (error: unknown) => void
}

const deferred = <T,>(): Deferred<T> => {
  let resolve!: (value: T) => void
  let reject!: (error: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, resolve, reject }
}

const mocks = vi.hoisted(() => ({
  useServerOnline: vi.fn(),
  useServerCapabilities: vi.fn(),
  useConnectionState: vi.fn(),
  listPresentations: vi.fn(),
  getPresentationMetadata: vi.fn(),
  getPresentation: vi.fn(),
  listPresentationVersions: vi.fn(),
  getPresentationVersionContent: vi.fn(),
  restorePresentationVersion: vi.fn(),
  getConfig: vi.fn(),
  createPresentation: vi.fn(),
  patchPresentation: vi.fn(),
  saveStandaloneHtmlSource: vi.fn(),
  downloadStandaloneHtmlDraft: vi.fn(),
  downloadStandaloneHtmlPresentation: vi.fn(),
  exportPresentation: vi.fn(),
  submitPresentationRenderJob: vi.fn(),
  getPresentationRenderJob: vi.fn(),
  listPresentationRenderArtifacts: vi.fn(),
  chromeRuntimeSendMessage: vi.fn(),
  browserRuntimeSendMessage: vi.fn(),
  chromeStorageLocalSet: vi.fn(),
  chromeStorageSyncSet: vi.fn(),
  chromeStorageSessionSet: vi.fn(),
  browserStorageLocalSet: vi.fn(),
  browserStorageSyncSet: vi.fn(),
  browserStorageSessionSet: vi.fn()
}))

vi.mock("@/components/Layouts/Layout", () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="option-layout">{children}</div>
  )
}))

vi.mock("@/components/Common/RouteErrorBoundary", () => ({
  RouteErrorBoundary: ({
    routeId,
    children
  }: {
    routeId: string
    children: React.ReactNode
  }) => <div data-testid={`route-boundary-${routeId}`}>{children}</div>
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => mocks.useServerOnline()
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => mocks.useServerCapabilities()
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => mocks.useConnectionState()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback: string) => fallback
  })
}))

vi.mock("@/libs/get-screenshot", () => ({
  getScreenshotFromCurrentTab: vi.fn().mockResolvedValue({
    success: false,
    error: "Screenshot unavailable in test."
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    listPresentations: (...args: unknown[]) => mocks.listPresentations(...args),
    getPresentationMetadata: (...args: unknown[]) =>
      mocks.getPresentationMetadata(...args),
    getPresentation: (...args: unknown[]) => mocks.getPresentation(...args),
    listPresentationVersions: (...args: unknown[]) =>
      mocks.listPresentationVersions(...args),
    getPresentationVersionContent: (...args: unknown[]) =>
      mocks.getPresentationVersionContent(...args),
    restorePresentationVersion: (...args: unknown[]) =>
      mocks.restorePresentationVersion(...args),
    getConfig: (...args: unknown[]) => mocks.getConfig(...args),
    createPresentation: (...args: unknown[]) => mocks.createPresentation(...args),
    patchPresentation: (...args: unknown[]) => mocks.patchPresentation(...args),
    saveStandaloneHtmlSource: (...args: unknown[]) =>
      mocks.saveStandaloneHtmlSource(...args),
    downloadStandaloneHtmlDraft: (...args: unknown[]) =>
      mocks.downloadStandaloneHtmlDraft(...args),
    downloadStandaloneHtmlPresentation: (...args: unknown[]) =>
      mocks.downloadStandaloneHtmlPresentation(...args),
    exportPresentation: (...args: unknown[]) => mocks.exportPresentation(...args),
    submitPresentationRenderJob: (...args: unknown[]) =>
      mocks.submitPresentationRenderJob(...args),
    getPresentationRenderJob: (...args: unknown[]) =>
      mocks.getPresentationRenderJob(...args),
    listPresentationRenderArtifacts: (...args: unknown[]) =>
      mocks.listPresentationRenderArtifacts(...args)
  }
}))

const baseSummary = {
  id: "html-1",
  title: "Architecture briefing",
  description: "A bounded source-free summary.",
  theme: "black",
  created_at: "2026-08-01T00:00:00Z",
  last_modified: "2026-08-02T00:00:00Z",
  deleted: false,
  version: 1,
  provenance: {
    source_kind: "prompt",
    provider: "openai",
    model: "gpt-5"
  }
}

const standaloneSummary = {
  ...baseSummary,
  content_kind: "standalone_html" as const,
  html_slide_count: 7,
  html_bytes: 12_345
}

const structuredSummary = {
  ...baseSummary,
  id: "structured-1",
  title: "Structured quarterly review",
  content_kind: "structured_slides" as const,
  slide_count: 6
}

const unsupportedSummary = {
  ...baseSummary,
  id: "future-1",
  title: "Future presentation",
  content_kind: "unsupported" as const,
  unsupported_content_kind: "immersive_canvas",
  read_only: true as const
}

const emptyPage = {
  presentations: [],
  total: 0,
  limit: 25,
  offset: 0,
  pagination: {
    mode: "offset",
    limit: 25,
    offset: 0,
    total: 0,
    has_more: false,
    next_offset: null
  },
  has_more: false,
  next_offset: null
}

const sourceBearingClientMocks = () => [
  mocks.getPresentation,
  mocks.listPresentationVersions,
  mocks.getPresentationVersionContent,
  mocks.restorePresentationVersion,
  mocks.patchPresentation,
  mocks.saveStandaloneHtmlSource,
  mocks.downloadStandaloneHtmlDraft,
  mocks.downloadStandaloneHtmlPresentation,
  mocks.exportPresentation,
  mocks.submitPresentationRenderJob,
  mocks.getPresentationRenderJob,
  mocks.listPresentationRenderArtifacts
]

const extensionRuntimeMessageMocks = () => [
  mocks.chromeRuntimeSendMessage,
  mocks.browserRuntimeSendMessage
]

const extensionStorageWriteMocks = () => [
  mocks.chromeStorageLocalSet,
  mocks.chromeStorageSyncSet,
  mocks.chromeStorageSessionSet,
  mocks.browserStorageLocalSet,
  mocks.browserStorageSyncSet,
  mocks.browserStorageSessionSet
]

const loadProjectPanel = async () => {
  const module = await vi.importActual<Record<string, unknown>>(
    "@/components/Option/PresentationStudio/ExtensionStartPanel"
  )
  const component = module.ExtensionPresentationProjectPanel
  if (typeof component !== "function") {
    throw new Error("ExtensionPresentationProjectPanel is not implemented")
  }
  return component as React.ComponentType<{ structuredDetail: React.ReactNode }>
}

const renderProjectPanel = async (
  path: string,
  structuredDetail: React.ReactNode = (
    <h1 data-testid="structured-detail">Structured presentation editor</h1>
  )
) => {
  const ExtensionPresentationProjectPanel = await loadProjectPanel()
  const router = createMemoryRouter(
    [
      {
        path: "/presentation-studio/:projectId",
        element: (
          <ExtensionPresentationProjectPanel structuredDetail={structuredDetail} />
        )
      }
    ],
    { initialEntries: [path] }
  )
  const view = render(<RouterProvider router={router} />)
  return { ...view, router }
}

let routeRegistry: typeof import("../route-registry")
let originalChromeDescriptor: PropertyDescriptor | undefined
let originalBrowserDescriptor: PropertyDescriptor | undefined

beforeAll(async () => {
  originalChromeDescriptor = Object.getOwnPropertyDescriptor(globalThis, "chrome")
  originalBrowserDescriptor = Object.getOwnPropertyDescriptor(globalThis, "browser")
  Object.defineProperty(globalThis, "chrome", {
    configurable: true,
    value: {
      runtime: {
        id: "task16-chrome-extension-test",
        sendMessage: mocks.chromeRuntimeSendMessage
      },
      storage: {
        local: { set: mocks.chromeStorageLocalSet },
        sync: { set: mocks.chromeStorageSyncSet },
        session: { set: mocks.chromeStorageSessionSet }
      }
    }
  })
  Object.defineProperty(globalThis, "browser", {
    configurable: true,
    value: {
      runtime: {
        id: "task16-extension-test",
        sendMessage: mocks.browserRuntimeSendMessage
      },
      storage: {
        local: { set: mocks.browserStorageLocalSet },
        sync: { set: mocks.browserStorageSyncSet },
        session: { set: mocks.browserStorageSessionSet }
      }
    }
  })
  routeRegistry = await import("../route-registry")
})

afterAll(() => {
  if (originalChromeDescriptor) {
    Object.defineProperty(globalThis, "chrome", originalChromeDescriptor)
  } else {
    Reflect.deleteProperty(globalThis, "chrome")
  }
  if (originalBrowserDescriptor) {
    Object.defineProperty(globalThis, "browser", originalBrowserDescriptor)
  } else {
    Reflect.deleteProperty(globalThis, "browser")
  }
})

describe("presentation studio option route guards", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.useServerOnline.mockReturnValue(true)
    mocks.useServerCapabilities.mockReturnValue({
      loading: false,
      capabilities: {
        hasSlides: true,
        hasPresentationStudio: true,
        hasPresentationRender: true
      }
    })
    mocks.useConnectionState.mockReturnValue({
      serverUrl: "http://127.0.0.1:8000"
    })
    mocks.listPresentations.mockResolvedValue(emptyPage)
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user"
    })
  })

  it("blocks the route when presentation studio is unsupported", () => {
    mocks.useServerCapabilities.mockReturnValue({
      loading: false,
      capabilities: {
        hasSlides: true,
        hasPresentationStudio: false,
        hasPresentationRender: false
      }
    })

    render(
      <MemoryRouter>
        <OptionPresentationStudio />
      </MemoryRouter>
    )

    expect(
      screen.getByText("Presentation Studio is not available on this server.")
    ).toBeInTheDocument()
  })

  it("renders the editor shell when presentation studio is supported", () => {
    render(
      <MemoryRouter>
        <OptionPresentationStudio />
      </MemoryRouter>
    )

    expect(screen.getByTestId("route-boundary-presentation-studio")).toBeVisible()
    expect(screen.getByText("Presentation Studio")).toBeInTheDocument()
  })

  it("registers the extension index, quick-start, reserved new redirect, and metadata resolver in safe order", () => {
    const presentationRoutes = routeRegistry.ROUTE_DEFINITIONS.filter((route) =>
      route.path.startsWith("/presentation-studio")
    )
    const paths = presentationRoutes.map((route) => route.path)
    expect(paths).toEqual([
      "/presentation-studio",
      "/presentation-studio/new",
      "/presentation-studio/start",
      "/presentation-studio/:projectId"
    ])

    const newRoute = presentationRoutes.find(
      (route) => route.path === "/presentation-studio/new"
    )
    const detailRoute = presentationRoutes.find(
      (route) => route.path === "/presentation-studio/:projectId"
    )
    expect(newRoute?.element.type).toBe(Navigate)
    expect(newRoute?.element.props).toMatchObject({
      to: "/presentation-studio/start",
      replace: true
    })
    expect(detailRoute?.element.props.structuredDetail).toBeTruthy()
    expect(paths.indexOf("/presentation-studio/start")).toBeLessThan(
      paths.indexOf("/presentation-studio/:projectId")
    )
  })

  it("reserves literal new without requesting metadata or importing an editor route", async () => {
    const newRoute = routeRegistry.ROUTE_DEFINITIONS.find(
      (route) => route.path === "/presentation-studio/new"
    )
    expect(newRoute).toBeDefined()

    render(
      <MemoryRouter initialEntries={["/presentation-studio/new"]}>
        <Routes>
          <Route path="/presentation-studio/new" element={newRoute?.element} />
          <Route
            path="/presentation-studio/start"
            element={<h1>Presentation Studio Quick Start</h1>}
          />
        </Routes>
      </MemoryRouter>
    )

    expect(
      await screen.findByRole("heading", {
        name: "Presentation Studio Quick Start"
      })
    ).toBeVisible()
    expect(mocks.getPresentationMetadata).not.toHaveBeenCalled()
    expect(mocks.getPresentation).not.toHaveBeenCalled()
  })

  it("declares exact WebUI and extension availability for every Presentation Studio route", () => {
    expect(getRouteMetadata("/presentation-studio")?.availability).toEqual([
      "web",
      "extension_options"
    ])
    expect(getRouteMetadata("/presentation-studio/new")?.availability).toEqual([
      "web"
    ])
    expect(getRouteMetadata("/presentation-studio/start")?.availability).toEqual([
      "extension_options"
    ])
    expect(getRouteMetadata("/presentation-studio/start")?.smoke).toBe("manual")
    expect(
      getRouteMetadata("/presentation-studio/:projectId")?.availability
    ).toEqual(["web", "extension_options"])
    expect(getRouteMetadata("/presentation-studio/html-1")).toBeUndefined()
  })

  it("loads standalone HTML direct links through metadata only and renders bounded provenance", async () => {
    mocks.getPresentationMetadata.mockResolvedValue({
      record: standaloneSummary,
      etag: null
    })

    await renderProjectPanel("/presentation-studio/html-1")

    expect(
      await screen.findByRole("heading", { name: "Architecture briefing" })
    ).toBeVisible()
    expect(screen.getByText("Standalone HTML + JavaScript")).toBeVisible()
    expect(screen.getByText("Prompt")).toBeVisible()
    expect(screen.getByText("openai")).toBeVisible()
    expect(screen.getByText("gpt-5")).toBeVisible()
    expect(screen.getByRole("button", { name: "Open in WebUI" })).toBeEnabled()
    expect(mocks.getPresentationMetadata).toHaveBeenCalledTimes(1)
    expect(mocks.getPresentationMetadata).toHaveBeenCalledWith("html-1")
    for (const sourceClient of sourceBearingClientMocks()) {
      expect(sourceClient).not.toHaveBeenCalled()
    }
    for (const storageWrite of extensionStorageWriteMocks()) {
      expect(storageWrite).not.toHaveBeenCalled()
    }
    for (const runtimeMessage of extensionRuntimeMessageMocks()) {
      expect(runtimeMessage).not.toHaveBeenCalled()
    }
  })

  it("uses the same metadata-only handoff for an unknown future kind", async () => {
    mocks.getPresentationMetadata.mockResolvedValue({
      record: unsupportedSummary,
      etag: null
    })

    await renderProjectPanel("/presentation-studio/future-1")

    expect(
      await screen.findByRole("heading", { name: "Future presentation" })
    ).toBeVisible()
    expect(screen.getByText("Unknown kind: immersive_canvas")).toBeVisible()
    expect(screen.getByText(/read only in this extension/i)).toBeVisible()
    expect(screen.getByRole("button", { name: "Open in WebUI" })).toBeEnabled()
    for (const sourceClient of sourceBearingClientMocks()) {
      expect(sourceClient).not.toHaveBeenCalled()
    }
  })

  it("mounts the existing structured detail only after exact structured metadata", async () => {
    mocks.getPresentationMetadata.mockResolvedValue({
      record: structuredSummary,
      etag: null
    })

    await renderProjectPanel("/presentation-studio/structured-1")

    expect(await screen.findByTestId("structured-detail")).toBeVisible()
    expect(mocks.getPresentationMetadata).toHaveBeenCalledWith("structured-1")
    expect(screen.queryByRole("button", { name: "Open in WebUI" })).toBeNull()
    for (const sourceClient of sourceBearingClientMocks()) {
      expect(sourceClient).not.toHaveBeenCalled()
    }
  })

  it("keeps the prior kind fenced across HTML-to-structured and structured-to-HTML route changes", async () => {
    const staleHtml = deferred<{ record: typeof standaloneSummary; etag: null }>()
    mocks.getPresentationMetadata.mockImplementation((id: string) => {
      if (id === "html-1") return staleHtml.promise
      if (id === "structured-1") {
        return Promise.resolve({ record: structuredSummary, etag: null })
      }
      return Promise.resolve({
        record: { ...standaloneSummary, id: "html-2", title: "Second HTML deck" },
        etag: null
      })
    })

    const { router } = await renderProjectPanel("/presentation-studio/html-1")
    await act(async () => {
      await router.navigate("/presentation-studio/structured-1")
    })
    expect(await screen.findByTestId("structured-detail")).toBeVisible()

    staleHtml.resolve({ record: standaloneSummary, etag: null })
    await act(async () => {
      await staleHtml.promise
    })
    expect(screen.queryByText("Architecture briefing")).toBeNull()
    expect(screen.getByTestId("structured-detail")).toBeVisible()

    await act(async () => {
      await router.navigate("/presentation-studio/html-2")
    })
    expect(
      await screen.findByRole("heading", { name: "Second HTML deck" })
    ).toBeVisible()
    expect(screen.queryByTestId("structured-detail")).toBeNull()
  })

  it.each([
    ["mismatched id", { ...standaloneSummary, id: "other-id" }],
    ["blank title", { ...standaloneSummary, title: "   " }],
    ["oversized title", { ...standaloneSummary, title: "x".repeat(513) }],
    ["C0 control", { ...standaloneSummary, title: "Bad\u0001title" }],
    ["C1 control", { ...standaloneSummary, title: "Bad\u0085title" }],
    ["bidi control", { ...standaloneSummary, title: "Bad\u202Etitle" }],
    [
      "oversized description",
      { ...standaloneSummary, description: "d".repeat(2_049) }
    ],
    [
      "oversized source kind",
      {
        ...standaloneSummary,
        provenance: {
          ...standaloneSummary.provenance,
          source_kind: "s".repeat(257)
        }
      }
    ],
    [
      "oversized provider",
      {
        ...standaloneSummary,
        provenance: {
          ...standaloneSummary.provenance,
          provider: "p".repeat(257)
        }
      }
    ],
    [
      "oversized model",
      {
        ...standaloneSummary,
        provenance: {
          ...standaloneSummary.provenance,
          model: "m".repeat(257)
        }
      }
    ],
    [
      "lone surrogate",
      {
        ...standaloneSummary,
        provenance: { ...standaloneSummary.provenance, model: "bad\uD800" }
      }
    ],
    ["nonfinite byte count", { ...standaloneSummary, html_bytes: Number.POSITIVE_INFINITY }],
    ["negative byte count", { ...standaloneSummary, html_bytes: -1 }],
    ["negative slide count", { ...standaloneSummary, html_slide_count: -1 }],
    ["fractional slide count", { ...standaloneSummary, html_slide_count: 1.5 }],
    [
      "negative structured count",
      {
        ...structuredSummary,
        id: "html-1",
        slide_count: -1
      }
    ],
    [
      "nonfinite structured count",
      {
        ...structuredSummary,
        id: "html-1",
        slide_count: Number.NaN
      }
    ],
    [
      "blank unknown kind",
      { ...unsupportedSummary, id: "html-1", unsupported_content_kind: " " }
    ],
    [
      "oversized unknown kind",
      {
        ...unsupportedSummary,
        id: "html-1",
        unsupported_content_kind: "k".repeat(257)
      }
    ]
  ])("fails closed before state/DOM for malformed metadata: %s", async (_case, record) => {
    mocks.getPresentationMetadata.mockResolvedValue({ record, etag: null })

    await renderProjectPanel("/presentation-studio/html-1")

    expect(
      await screen.findByText("Presentation metadata could not be verified")
    ).toBeVisible()
    expect(screen.queryByRole("button", { name: "Open in WebUI" })).toBeNull()
    expect(screen.queryByText(String(record.title))).toBeNull()
    for (const sourceClient of sourceBearingClientMocks()) {
      expect(sourceClient).not.toHaveBeenCalled()
    }
  })

  it.each([
    ["blank", "  "],
    ["dot segment", "%2E"],
    ["parent dot segment", "%2E%2E"],
    ["oversized", "i".repeat(257)],
    ["C0 control", "bad\u0001id"],
    ["C1 control", "bad\u0085id"],
    ["bidi control", "bad\u2066id"],
    ["lone surrogate", "bad\uD800id"]
  ])("rejects a %s route ID before metadata transport", async (_case, routeId) => {
    await renderProjectPanel(`/presentation-studio/${routeId}`)

    expect(
      await screen.findByText("Presentation metadata could not be verified")
    ).toBeVisible()
    expect(mocks.getPresentationMetadata).not.toHaveBeenCalled()
  })

  it("counts Unicode scalars rather than UTF-16 code units at exact metadata caps", async () => {
    const projectId = "i".repeat(256)
    const exactScalarTitle = "😀".repeat(512)
    mocks.getPresentationMetadata.mockResolvedValue({
      record: {
        ...standaloneSummary,
        id: projectId,
        title: exactScalarTitle,
        description: "d".repeat(2_048),
        provenance: {
          source_kind: "s".repeat(256),
          provider: "p".repeat(256),
          model: "m".repeat(256)
        }
      },
      etag: null
    })

    await renderProjectPanel(
      `/presentation-studio/${encodeURIComponent(projectId)}`
    )

    expect(
      await screen.findByRole("heading", { name: exactScalarTitle })
    ).toBeVisible()
  })

  it.each([
    ["null result", null],
    ["array result", []],
    ["empty result", {}],
    ["null record", { record: null, etag: null }],
    ["array record", { record: [], etag: null }],
    [
      "array provenance",
      { record: { ...standaloneSummary, provenance: [] }, etag: null }
    ],
    [
      "non-string content kind",
      { record: { ...standaloneSummary, content_kind: [] }, etag: null }
    ]
  ])("fails closed for a malformed metadata envelope: %s", async (_case, response) => {
    mocks.getPresentationMetadata.mockResolvedValue(response)

    await renderProjectPanel("/presentation-studio/html-1")

    expect(
      await screen.findByText("Presentation metadata could not be verified")
    ).toBeVisible()
    expect(screen.queryByRole("button", { name: "Open in WebUI" })).toBeNull()
  })

  it("shows accessible loading, offline, capability-loading, unsupported, and retry states without prohibited calls", async () => {
    const pending = deferred<{ record: typeof standaloneSummary; etag: null }>()
    mocks.getPresentationMetadata.mockReturnValue(pending.promise)
    const loadingView = await renderProjectPanel("/presentation-studio/html-1")
    expect(screen.getByRole("status", { name: /loading presentation metadata/i })).toBeVisible()
    loadingView.unmount()

    mocks.useServerOnline.mockReturnValue(false)
    const offlineView = await renderProjectPanel("/presentation-studio/html-1")
    expect(screen.getByText("Presentation handoff is offline")).toBeVisible()
    offlineView.unmount()

    mocks.useServerOnline.mockReturnValue(true)
    mocks.useServerCapabilities.mockReturnValue({ loading: true, capabilities: null })
    const capabilityLoadingView = await renderProjectPanel(
      "/presentation-studio/html-1"
    )
    expect(screen.getByText("Checking Presentation Studio availability")).toBeVisible()
    capabilityLoadingView.unmount()

    mocks.useServerCapabilities.mockReturnValue({
      loading: false,
      capabilities: { hasPresentationStudio: false }
    })
    const unsupportedView = await renderProjectPanel("/presentation-studio/html-1")
    expect(screen.getByText("Presentation Studio is not available")).toBeVisible()
    unsupportedView.unmount()

    mocks.useServerCapabilities.mockReturnValue({
      loading: false,
      capabilities: { hasPresentationStudio: true }
    })
    mocks.getPresentationMetadata
      .mockRejectedValueOnce(new Error("secret transport details"))
      .mockResolvedValueOnce({ record: standaloneSummary, etag: null })
    await renderProjectPanel("/presentation-studio/html-1")
    expect(await screen.findByText("Presentation metadata could not load")).toBeVisible()
    expect(screen.queryByText("secret transport details")).toBeNull()
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    expect(
      await screen.findByRole("heading", { name: "Architecture briefing" })
    ).toBeVisible()

    for (const sourceClient of sourceBearingClientMocks()) {
      expect(sourceClient).not.toHaveBeenCalled()
    }
  })

  it.each([
    "tldw:config-updated",
    "tldw:auth-principal-changed",
    "tldw:slides-scope-mismatch"
  ])("retires a stale metadata response across %s", async (eventName) => {
    const first = deferred<{ record: typeof standaloneSummary; etag: null }>()
    mocks.getPresentationMetadata
      .mockReturnValueOnce(first.promise)
      .mockResolvedValueOnce({
        record: { ...standaloneSummary, title: "Fresh authority metadata" },
        etag: null
      })
    await renderProjectPanel("/presentation-studio/html-1")

    act(() => {
      window.dispatchEvent(new CustomEvent(eventName))
    })
    first.resolve({ record: standaloneSummary, etag: null })

    expect(
      await screen.findByRole("heading", { name: "Fresh authority metadata" })
    ).toBeVisible()
    expect(screen.queryByText("Architecture briefing")).toBeNull()
    expect(mocks.getPresentationMetadata).toHaveBeenCalledTimes(2)
  })

  it("does not publish or navigate from a response after unmount", async () => {
    const metadata = deferred<{ record: typeof standaloneSummary; etag: null }>()
    mocks.getPresentationMetadata.mockReturnValue(metadata.promise)
    const open = vi.spyOn(window, "open").mockReturnValue(null)
    const view = await renderProjectPanel("/presentation-studio/html-1")
    view.unmount()

    metadata.resolve({ record: standaloneSummary, etag: null })
    await act(async () => {
      await metadata.promise
    })

    expect(open).not.toHaveBeenCalled()
    open.mockRestore()
  })

  it("builds the handoff only from click-time canonical config and the encoded trusted route ID", async () => {
    const projectId = "deck /? # source-looking"
    const record = {
      ...standaloneSummary,
      id: projectId,
      title: "https://evil.example/metadata-path",
      provenance: {
        source_kind: "evil.example/path",
        provider: "https-provider",
        model: "model?redirect=evil"
      }
    }
    mocks.getPresentationMetadata.mockResolvedValue({ record, etag: null })
    mocks.getConfig.mockResolvedValue({
      serverUrl: "https://api-attacker.example.invalid:8000/source",
      webUiUrl: "https://user:pass@webui.example.test/tldw/sub/?ignored=1#ignored",
      authMode: "single-user"
    })
    const open = vi.spyOn(window, "open").mockReturnValue(null)

    await renderProjectPanel(
      `/presentation-studio/${encodeURIComponent(projectId)}`
    )
    fireEvent.click(await screen.findByRole("button", { name: "Open in WebUI" }))

    await waitFor(() => {
      expect(open).toHaveBeenCalledWith(
        `https://webui.example.test/tldw/sub/presentation-studio/${encodeURIComponent(projectId)}`,
        "_blank",
        "noopener,noreferrer"
      )
    })
    expect(open.mock.calls[0]?.[0]).not.toContain("evil.example")
    expect(open.mock.calls[0]?.[0]).not.toContain("api-attacker")
    open.mockRestore()
  })

  it("uses canonical server-port inference when no explicit WebUI alias is configured", async () => {
    mocks.getPresentationMetadata.mockResolvedValue({
      record: standaloneSummary,
      etag: null
    })
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://user:pass@127.0.0.1:8000/api/v1?ignored=1#ignored",
      authMode: "single-user"
    })
    const open = vi.spyOn(window, "open").mockReturnValue(null)

    await renderProjectPanel("/presentation-studio/html-1")
    fireEvent.click(await screen.findByRole("button", { name: "Open in WebUI" }))

    await waitFor(() => {
      expect(open).toHaveBeenCalledWith(
        "http://127.0.0.1:8080/presentation-studio/html-1",
        "_blank",
        "noopener,noreferrer"
      )
    })
    open.mockRestore()
  })

  it.each([
    [
      "legacy alias",
      {
        serverUrl: "http://127.0.0.1:8000",
        webuiUrl: "https://legacy-webui.example.test/legacy/base/?ignored=1#ignored",
        authMode: "single-user"
      },
      "https://legacy-webui.example.test/legacy/base/presentation-studio/html-1"
    ],
    [
      "legacy fallback after invalid preferred alias",
      {
        serverUrl: "http://127.0.0.1:8000",
        webUiUrl: "javascript:alert(1)",
        webuiUrl: "https://legacy-webui.example.test/fallback/",
        authMode: "single-user"
      },
      "https://legacy-webui.example.test/fallback/presentation-studio/html-1"
    ],
    [
      "preferred alias when both are valid",
      {
        serverUrl: "http://127.0.0.1:8000",
        webUiUrl: "https://preferred.example.test/base/",
        webuiUrl: "https://legacy-webui.example.test/ignored/",
        authMode: "single-user"
      },
      "https://preferred.example.test/base/presentation-studio/html-1"
    ]
  ])("honors canonical WebUI config priority: %s", async (_case, config, expected) => {
    mocks.getPresentationMetadata.mockResolvedValue({
      record: standaloneSummary,
      etag: null
    })
    mocks.getConfig.mockResolvedValue(config)
    const open = vi.spyOn(window, "open").mockReturnValue(null)

    await renderProjectPanel("/presentation-studio/html-1")
    fireEvent.click(await screen.findByRole("button", { name: "Open in WebUI" }))

    await waitFor(() => {
      expect(open).toHaveBeenCalledWith(
        expected,
        "_blank",
        "noopener,noreferrer"
      )
    })
    open.mockRestore()
  })

  it.each([
    ["missing", null],
    [
      "all invalid",
      {
        serverUrl: "file:///tmp/api",
        webUiUrl: "javascript:alert(1)",
        webuiUrl: "not a URL",
        authMode: "single-user"
      }
    ]
  ])("fails closed when canonical WebUI configuration is %s", async (_case, config) => {
    mocks.getPresentationMetadata.mockResolvedValue({
      record: standaloneSummary,
      etag: null
    })
    mocks.getConfig.mockResolvedValue(config)
    const open = vi.spyOn(window, "open").mockReturnValue(null)

    await renderProjectPanel("/presentation-studio/html-1")
    fireEvent.click(await screen.findByRole("button", { name: "Open in WebUI" }))

    expect(
      await screen.findByText("A valid WebUI address is not configured")
    ).toBeVisible()
    expect(open).not.toHaveBeenCalled()
    open.mockRestore()
  })

  it("fences a click-time config result after authority and unmount retirement", async () => {
    mocks.getPresentationMetadata.mockResolvedValue({
      record: standaloneSummary,
      etag: null
    })
    const config = deferred<Record<string, unknown>>()
    mocks.getConfig.mockReturnValue(config.promise)
    const open = vi.spyOn(window, "open").mockReturnValue(null)
    const view = await renderProjectPanel("/presentation-studio/html-1")
    fireEvent.click(await screen.findByRole("button", { name: "Open in WebUI" }))

    act(() => {
      window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
    })
    view.unmount()
    config.resolve({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user"
    })
    await act(async () => {
      await config.promise
    })

    expect(open).not.toHaveBeenCalled()
    open.mockRestore()
  })

  it("cannot open an old ready handoff clicked in the same task as authority retirement", async () => {
    mocks.getPresentationMetadata.mockResolvedValue({
      record: standaloneSummary,
      etag: null
    })
    const config = deferred<Record<string, unknown>>()
    mocks.getConfig.mockReturnValue(config.promise)
    const open = vi.spyOn(window, "open").mockReturnValue(null)

    await renderProjectPanel("/presentation-studio/html-1")
    const oldButton = await screen.findByRole("button", {
      name: "Open in WebUI"
    })

    await act(async () => {
      window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
      fireEvent.click(oldButton)
      config.resolve({
        serverUrl: "http://127.0.0.1:8000",
        webUiUrl: "https://webui.example.test/current/",
        authMode: "single-user"
      })
      await config.promise
      await Promise.resolve()
    })

    expect(open).not.toHaveBeenCalled()
    open.mockRestore()
  })

  it.each([
    "tldw:config-updated",
    "tldw:auth-principal-changed",
    "tldw:slides-scope-mismatch"
  ])("does not open a structured quick-start create response after %s", async (eventName) => {
    const created = deferred<{ id: string }>()
    mocks.createPresentation.mockReturnValue(created.promise)
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user"
    })
    const open = vi.spyOn(window, "open").mockReturnValue(null)
    const module = await vi.importActual<
      typeof import("@/components/Option/PresentationStudio/ExtensionStartPanel")
    >("@/components/Option/PresentationStudio/ExtensionStartPanel")
    render(<module.ExtensionStartPanel />)

    fireEvent.click(
      screen.getByRole("button", { name: "Create blank project" })
    )
    await waitFor(() => expect(mocks.createPresentation).toHaveBeenCalledTimes(1))
    act(() => {
      window.dispatchEvent(new CustomEvent(eventName))
    })
    created.resolve({ id: "retired-project" })
    await act(async () => {
      await created.promise
    })

    expect(open).not.toHaveBeenCalled()
    open.mockRestore()
  })

  it("does not open a structured quick-start create response after unmount", async () => {
    const created = deferred<{ id: string }>()
    mocks.createPresentation.mockReturnValue(created.promise)
    const open = vi.spyOn(window, "open").mockReturnValue(null)
    const module = await vi.importActual<
      typeof import("@/components/Option/PresentationStudio/ExtensionStartPanel")
    >("@/components/Option/PresentationStudio/ExtensionStartPanel")
    const view = render(<module.ExtensionStartPanel />)

    fireEvent.click(
      screen.getByRole("button", { name: "Create blank project" })
    )
    await waitFor(() => expect(mocks.createPresentation).toHaveBeenCalledTimes(1))
    view.unmount()
    created.resolve({ id: "retired-project" })
    await act(async () => {
      await created.promise
    })

    expect(open).not.toHaveBeenCalled()
    open.mockRestore()
  })

  it("rechecks click-time config and authority after structured creation resolves", async () => {
    const config = deferred<Record<string, unknown>>()
    mocks.createPresentation.mockResolvedValue({ id: "created-project" })
    mocks.getConfig.mockReturnValue(config.promise)
    const open = vi.spyOn(window, "open").mockReturnValue(null)
    const module = await vi.importActual<
      typeof import("@/components/Option/PresentationStudio/ExtensionStartPanel")
    >("@/components/Option/PresentationStudio/ExtensionStartPanel")
    render(<module.ExtensionStartPanel />)

    fireEvent.click(
      screen.getByRole("button", { name: "Create blank project" })
    )
    await waitFor(() => expect(mocks.getConfig).toHaveBeenCalledTimes(1))
    act(() => {
      window.dispatchEvent(new CustomEvent("tldw:slides-scope-mismatch"))
    })
    config.resolve({
      serverUrl: "http://127.0.0.1:8000",
      webUiUrl: "https://webui.example.test/current/",
      authMode: "single-user"
    })
    await act(async () => {
      await config.promise
    })

    expect(open).not.toHaveBeenCalled()
    open.mockRestore()
  })

  it("opens a structured quick-start result with the current canonical WebUI config", async () => {
    mocks.createPresentation.mockResolvedValue({ id: "created-project" })
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      webuiUrl: "https://webui.example.test/current/base/",
      authMode: "single-user"
    })
    const open = vi.spyOn(window, "open").mockReturnValue(null)
    const module = await vi.importActual<
      typeof import("@/components/Option/PresentationStudio/ExtensionStartPanel")
    >("@/components/Option/PresentationStudio/ExtensionStartPanel")
    render(<module.ExtensionStartPanel />)

    fireEvent.click(
      screen.getByRole("button", { name: "Create blank project" })
    )

    await waitFor(() => {
      expect(open).toHaveBeenCalledWith(
        "https://webui.example.test/current/base/presentation-studio/created-project",
        "_blank",
        "noopener,noreferrer"
      )
    })
    open.mockRestore()
  })

  it.each([
    "tldw:config-updated",
    "tldw:auth-principal-changed",
    "tldw:slides-scope-mismatch"
  ])(
    "does not publish an old index response after %s",
    async (eventName) => {
      const oldRequest = deferred<typeof emptyPage>()
      mocks.listPresentations
        .mockReturnValueOnce(oldRequest.promise)
        .mockResolvedValueOnce(emptyPage)

      render(
        <MemoryRouter>
          <OptionPresentationStudio />
        </MemoryRouter>
      )
      act(() => {
        window.dispatchEvent(new CustomEvent(eventName))
      })
      oldRequest.resolve({
        ...emptyPage,
        presentations: [standaloneSummary] as never[],
        total: 1,
        pagination: { ...emptyPage.pagination, total: 1 }
      })

      await waitFor(() => expect(mocks.listPresentations).toHaveBeenCalledTimes(2))
      expect(screen.queryByText("Architecture briefing")).toBeNull()
    }
  )

  it("does not publish an index response after unmount", async () => {
    const oldRequest = deferred<typeof emptyPage>()
    mocks.listPresentations.mockReturnValue(oldRequest.promise)
    const view = render(
      <MemoryRouter>
        <OptionPresentationStudio />
      </MemoryRouter>
    )
    await waitFor(() => expect(mocks.listPresentations).toHaveBeenCalledTimes(1))
    view.unmount()

    oldRequest.resolve({
      ...emptyPage,
      presentations: [standaloneSummary] as never[],
      total: 1,
      pagination: { ...emptyPage.pagination, total: 1 }
    })
    await act(async () => {
      await oldRequest.promise
    })

    expect(screen.queryByText("Architecture briefing")).toBeNull()
  })
})
