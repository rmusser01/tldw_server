import { act, fireEvent, render as renderUI, screen, waitFor } from "@testing-library/react"
import { HashRouter, MemoryRouter, Route, Routes } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useWorkspaceStore } from "@/store/workspace"
import { hydrateWorkspaceFromServer } from "@/store/workspace-api"
import { serverWorkspacePayload } from "@/store/__tests__/workspace-activation.fixtures"
import { buildChatSurfaceScopeKeyFromConfig } from "@/services/chat-surface-scope"
import { ActivatedLocalWorkspace, ResearchWorkspaceRouteGate } from "../ResearchWorkspaceRouteGate"

// External API/account boundaries only are doubled; activation and the store are real.
const boundary = vi.hoisted(() => ({
  location: { search: "?workspace=server-research", key: "first", hash: "" },
  scope: { scopeKey: "owner-a", config: { serverUrl: "https://owner.test", authMode: "multi-user" as const }, userId: 42 },
  resolve: vi.fn(),
  request: vi.fn(),
  captures: vi.fn(),
  changed: null as null | ((invalidated: boolean) => void),
  getWorkspace: vi.fn(), getWorkspaceSources: vi.fn(), getWorkspaceArtifacts: vi.fn(), getWorkspaceNotes: vi.fn()
}))
vi.mock("react-router-dom", async importOriginal => ({
  ...await importOriginal<typeof import("react-router-dom")>(),
  useLocation: () => boundary.location
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: boundary }))
vi.mock("@/services/background-proxy", () => ({ bgRequest: boundary.request }))
vi.mock("@/utils/research-workspace-prefill", async importOriginal => ({
  ...await importOriginal<typeof import("@/utils/research-workspace-prefill")>(),
  readResearchWebCaptures: boundary.captures
}))
vi.mock("@/services/service-prompts", () => ({ resolveServicePromptScope: boundary.resolve }))
vi.mock("@/services/chat-account-boundary", () => ({
  watchChatAccountChanges: (changed: (invalidated: boolean) => void) => {
    boundary.changed = changed
    return () => { boundary.changed = null }
  }
}))
vi.mock("../index", () => ({ ResearchWorkspace: () => <div data-testid="activated-body" /> }))

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}

const render = (ui: React.ReactNode) => renderUI(ui, { wrapper: MemoryRouter })

describe("canonical route activation (unit regression doubles)", () => {
  afterEach(() => vi.restoreAllMocks())
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    useWorkspaceStore.getState().reset()
    useWorkspaceStore.setState({ storeHydrated: true, savedWorkspaces: [], workspaceSnapshots: {} })
    useWorkspaceStore.getState().initializeWorkspace("Outgoing")
    useWorkspaceStore.getState().updateNoteContent("Dirty draft")
    boundary.location = { search: "?workspace=server-research", key: "first", hash: "" }
    boundary.scope.scopeKey = "owner-a"
    boundary.scope.userId = 42
    boundary.captures.mockReset().mockResolvedValue([])
    boundary.resolve.mockImplementation(async () => boundary.scope)
    boundary.request.mockResolvedValue({ notes: [] })
    const payload = serverWorkspacePayload()
    boundary.getWorkspace.mockResolvedValue(payload.metadata)
    boundary.getWorkspaceSources.mockResolvedValue(payload.sources)
    boundary.getWorkspaceArtifacts.mockResolvedValue(payload.artifacts)
    boundary.getWorkspaceNotes.mockResolvedValue(payload.notes)
  })

  const captureFixture = async (refused: boolean, cached = false) => {
    const pin = {
      clipId: "route-clip", requestedUrl: "https://article.test/canonical",
      capturedAt: "2026-10-07T00:00:00Z", contentSha256: "digest", refreshOf: null,
      mediaId: 101, versionNumber: 1, versionUuid: "route-version-one"
    }
    const payload = serverWorkspacePayload()
    payload.sources[0] = { ...payload.sources[0], id: "web-clipper:route-clip", url: pin.requestedUrl, selected: true }
    boundary.getWorkspaceSources.mockResolvedValue(payload.sources)
    const owner = buildChatSurfaceScopeKeyFromConfig(boundary.scope.config, { userId: 42 })
    boundary.captures.mockImplementation(async (scope: string, workspace: string) =>
      scope === owner && workspace === "server-research" ? [{ pin }] : [])
    if (refused) {
      const local = await hydrateWorkspaceFromServer("server-research", { requireComplete: true, fetch: async () => payload })
      local.sources[0].webCapture = pin
      expect(useWorkspaceStore.getState().installServerWorkspace(local, {
        scopeKey: "owner-a", expectedWorkspaceId: useWorkspaceStore.getState().workspaceId
      })).toBe(true)
      useWorkspaceStore.getState().setSourceStatusById("web-clipper:route-clip", "error", "Head moved", undefined,
        { statusReason: "capture_head_changed", retryEligible: false })
      if (cached) useWorkspaceStore.getState().createNewWorkspace("Outgoing after refusal")
    }
    return { pin, payload, owner }
  }

  it("hydrates owner-bound capture pins only on canonical membership before activation", async () => {
    const { pin, owner } = await captureFixture(false)
    render(<ResearchWorkspaceRouteGate />)
    expect(await screen.findByTestId("activated-body")).toBeVisible()
    expect(boundary.captures).toHaveBeenCalledWith(owner, "server-research")
    expect(useWorkspaceStore.getState().sources).toMatchObject([{ id: "web-clipper:route-clip", webCapture: pin }])
    expect(useWorkspaceStore.getState().getSelectedMediaIds()).toEqual([101])
  })

  it.each(["active", "cached"])("retains exact-pin refusal from an owned %s canonical cache", async display => {
    const { pin } = await captureFixture(true, display === "cached")
    render(<ResearchWorkspaceRouteGate />)
    expect(await screen.findByTestId("activated-body")).toBeVisible()
    expect(useWorkspaceStore.getState().sources[0]).toMatchObject({
      webCapture: pin, status: "error", statusMessage: "Head moved",
      statusDetails: { statusReason: "capture_head_changed", retryEligible: false }
    })
    expect(useWorkspaceStore.getState().selectedSourceIds).toEqual([])
    expect(useWorkspaceStore.getState().getSelectedMediaIds()).toEqual([])
  })

  it.each(["pin", "removed", "foreign-owner", "missing-record", "legacy", "media", "url", "capture-read-failure"])("does not grant capture authority across %s boundaries", async change => {
    const { pin, payload } = await captureFixture(true)
    const retained = useWorkspaceStore.getState()
    if (change === "pin") boundary.captures.mockResolvedValue([{ pin: { ...pin, versionNumber: 2, versionUuid: "route-version-two" } }])
    if (change === "removed") boundary.getWorkspaceSources.mockResolvedValue([])
    if (change === "foreign-owner") { boundary.scope.userId = 43; boundary.scope.scopeKey = "owner-b" }
    if (change === "missing-record") boundary.captures.mockResolvedValue([])
    if (change === "media") boundary.getWorkspaceSources.mockResolvedValue([{ ...payload.sources[0], media_id: 202 }])
    if (change === "url") boundary.getWorkspaceSources.mockResolvedValue([{ ...payload.sources[0], url: "https://article.test/other" }])
    if (change === "capture-read-failure") boundary.captures.mockRejectedValue(new Error("Capture storage unavailable"))
    if (change === "legacy") {
      useWorkspaceStore.setState({ serverWorkspace: null })
      useWorkspaceStore.getState().saveCurrentWorkspace()
    }
    render(<ResearchWorkspaceRouteGate />)
    if (change === "foreign-owner" || change === "legacy" || change === "capture-read-failure") {
      expect(await screen.findByRole("alert")).toBeVisible()
      expect(useWorkspaceStore.getState().sources).toEqual(retained.sources)
      expect(screen.queryByTestId("activated-body")).toBeNull()
    } else {
      expect(await screen.findByTestId("activated-body")).toBeVisible()
      const sources = useWorkspaceStore.getState().sources
      if (change === "removed") expect(sources).toEqual([])
      else {
        expect(sources[0].status).toBe("ready")
        expect(sources[0].mediaId).toBe(change === "media" ? 202 : payload.sources[0].media_id)
        if (change === "pin") expect(sources[0].webCapture?.versionNumber).toBe(2)
        if (["missing-record", "media", "url"].includes(change)) expect(sources[0].webCapture).toBeUndefined()
      }
    }
  })

  it.each(["account-aba", "workspace-aba", "unmount"])("does not install after %s during capture hydration", async change => {
    const { pin } = await captureFixture(false)
    const capture = deferred<Array<{ pin: typeof pin }>>()
    boundary.captures.mockReturnValue(capture.promise)
    const outgoing = useWorkspaceStore.getState().workspaceId
    const view = render(<ResearchWorkspaceRouteGate />)
    await waitFor(() => expect(boundary.captures).toHaveBeenCalled())
    act(() => {
      if (change === "account-aba") { boundary.changed?.(true); boundary.scope.scopeKey = "owner-a" }
      if (change === "workspace-aba") {
        useWorkspaceStore.getState().createNewWorkspace("Intervening")
        useWorkspaceStore.getState().switchWorkspace(outgoing)
      }
      if (change === "unmount") view.unmount()
    })
    await act(async () => capture.resolve([{ pin }]))
    expect(useWorkspaceStore.getState().workspaceId).toBe(outgoing)
    expect(useWorkspaceStore.getState().workspaceSnapshots["server-research"]).toBeUndefined()
    expect(screen.queryByTestId("activated-body")).toBeNull()
    if (change !== "unmount") expect(screen.getByRole("alert")).toBeVisible()
  })

  it("keeps the body unmounted until every scoped read completes", async () => {
    const notes = deferred<ReturnType<typeof serverWorkspacePayload>["notes"]>()
    boundary.getWorkspaceNotes.mockReturnValue(notes.promise)
    const outgoing = useWorkspaceStore.getState().workspaceId
    render(<ResearchWorkspaceRouteGate />)
    await waitFor(() => expect(boundary.getWorkspaceNotes).toHaveBeenCalled())
    expect(screen.queryByTestId("activated-body")).toBeNull()
    expect(useWorkspaceStore.getState().workspaceId).toBe(outgoing)
    act(() => { useWorkspaceStore.getState().updateNoteContent("Latest dirty draft") })
    await act(async () => notes.resolve(serverWorkspacePayload().notes))
    expect(await screen.findByTestId("activated-body")).toBeVisible()
    expect(useWorkspaceStore.getState().workspaceId).toBe("server-research")
    expect(useWorkspaceStore.getState().workspaceSnapshots[outgoing].currentNote.content).toBe("Latest dirty draft")
    for (const method of [boundary.getWorkspace, boundary.getWorkspaceSources, boundary.getWorkspaceArtifacts, boundary.getWorkspaceNotes]) {
      expect(method).toHaveBeenCalledWith("server-research", expect.objectContaining({
        signal: expect.any(AbortSignal), requestScope: { config: boundary.scope.config, userId: 42 }
      }))
    }
  })

  it("keeps supplied chat content unmounted until qualified canonical activation", async () => {
    const notes = deferred<ReturnType<typeof serverWorkspacePayload>["notes"]>()
    boundary.getWorkspaceNotes.mockReturnValue(notes.promise)
    render(<ActivatedLocalWorkspace workspaceId="server-research" webClip={false}>
      <div data-testid="chat-content" />
    </ActivatedLocalWorkspace>)
    await waitFor(() => expect(boundary.getWorkspaceNotes).toHaveBeenCalled())
    expect(screen.getByRole("status")).toHaveTextContent("Loading workspace")
    expect(screen.queryByTestId("chat-content")).toBeNull()
    await act(async () => notes.resolve(serverWorkspacePayload().notes))
    expect(await screen.findByTestId("chat-content")).toBeVisible()
    expect(screen.queryByTestId("activated-body")).toBeNull()
    expect(useWorkspaceStore.getState().serverWorkspace?.scopeKey).toBe("owner-a")
  })

  it.each(["getWorkspace", "getWorkspaceSources", "getWorkspaceArtifacts", "getWorkspaceNotes"] as const)("fails closed when %s fails", async method => {
    boundary[method].mockRejectedValue(new Error("Required read failed"))
    const outgoing = useWorkspaceStore.getState().workspaceId
    render(<ResearchWorkspaceRouteGate />)
    expect(await screen.findByRole("alert")).toBeVisible()
    expect(screen.queryByTestId("activated-body")).toBeNull()
    expect(useWorkspaceStore.getState().workspaceId).toBe(outgoing)
    expect(useWorkspaceStore.getState().currentNote.content).toBe("Dirty draft")
  })

  it.each(["fresh", "persisted-local"])("offers manager navigation after missing %s canonical activation without changing local data", async kind => {
    if (kind === "fresh") {
      useWorkspaceStore.getState().reset()
      useWorkspaceStore.setState({ storeHydrated: true, savedWorkspaces: [], workspaceSnapshots: {} })
      useWorkspaceStore.getState().initializeWorkspace("Fresh local")
    }
    const before = useWorkspaceStore.getState()
    boundary.getWorkspace.mockRejectedValue(Object.assign(new Error("Workspace not found"), { status: 404 }))
    render(<ActivatedLocalWorkspace workspaceId={before.workspaceId!} webClip={false}>
      <div data-testid="chat-content" />
    </ActivatedLocalWorkspace>)
    expect(await screen.findByRole("alert")).toBeVisible()
    expect(screen.getByRole("link", { name: "Open workspaces" })).toHaveAttribute("href", "/workspaces")
    expect(screen.getByRole("button", { name: "Retry workspace" })).toBeEnabled()
    expect(screen.queryByTestId("chat-content")).toBeNull()
    expect(useWorkspaceStore.getState().workspaceId).toBe(before.workspaceId)
    expect(useWorkspaceStore.getState().currentNote).toEqual(before.currentNote)
    expect(useWorkspaceStore.getState().workspaceSnapshots).toEqual(before.workspaceSnapshots)
  })

  it.each(["hash", "memory"])("navigates to the manager through the real %s router after failed activation", async kind => {
    const before = useWorkspaceStore.getState()
    boundary.getWorkspace.mockRejectedValue(new Error("Workspace not found"))
    const routes = <Routes>
      <Route path="/chat-workspace" element={<ActivatedLocalWorkspace workspaceId={before.workspaceId!} webClip={false} />} />
      <Route path="/workspaces" element={<div data-testid="workspace-manager" />} />
    </Routes>
    if (kind === "hash") {
      window.location.hash = "/chat-workspace"
      renderUI(<HashRouter>{routes}</HashRouter>)
    } else renderUI(<MemoryRouter initialEntries={["/chat-workspace"]}>{routes}</MemoryRouter>)
    expect(await screen.findByRole("alert")).toBeVisible()
    fireEvent.click(screen.getByRole("link", { name: "Open workspaces" }))
    expect(await screen.findByTestId("workspace-manager")).toBeVisible()
    expect(useWorkspaceStore.getState().workspaceId).toBe(before.workspaceId)
    expect(useWorkspaceStore.getState().currentNote).toEqual(before.currentNote)
  })

  it.each(["unmount", "account-aba", "workspace-aba", "route-replacement"])("ignores late responses after %s", async change => {
    const notes = deferred<ReturnType<typeof serverWorkspacePayload>["notes"]>()
    boundary.getWorkspaceNotes.mockReturnValue(notes.promise)
    const outgoing = useWorkspaceStore.getState().workspaceId
    const view = render(<ResearchWorkspaceRouteGate />)
    await waitFor(() => expect(boundary.getWorkspaceNotes).toHaveBeenCalled())
    const options = boundary.getWorkspaceNotes.mock.calls[0][1]
    act(() => {
      if (change === "unmount") view.unmount()
      if (change === "account-aba") { boundary.changed?.(true); boundary.scope.scopeKey = "owner-a" }
      if (change === "workspace-aba") {
        useWorkspaceStore.getState().createNewWorkspace("Intervening")
        useWorkspaceStore.getState().switchWorkspace(outgoing)
      }
      if (change === "route-replacement") {
        boundary.location = { search: "?workspace=other", key: "second", hash: "" }
        boundary.getWorkspace.mockRejectedValue(new Error("Replacement unavailable"))
        view.rerender(<ResearchWorkspaceRouteGate />)
      }
    })
    await act(async () => notes.resolve(serverWorkspacePayload().notes))
    expect(options.signal.aborted).toBe(true)
    expect(useWorkspaceStore.getState().workspaceId).toBe(outgoing)
    expect(useWorkspaceStore.getState().workspaceSnapshots["server-research"]).toBeUndefined()
    if (change !== "unmount") expect(screen.getByRole("alert")).toBeVisible()
  })

  it.each(["principal", "auth-source"])("terminates pending when final %s verification differs", async change => {
    boundary.resolve.mockResolvedValueOnce({ ...boundary.scope }).mockResolvedValueOnce(change === "principal"
      ? { ...boundary.scope, scopeKey: "owner-b" }
      : { ...boundary.scope, config: { ...boundary.scope.config, authSource: "cookie-session" } })
    const outgoing = useWorkspaceStore.getState().workspaceId
    render(<ResearchWorkspaceRouteGate />)
    expect(await screen.findByRole("alert")).toBeVisible()
    expect(useWorkspaceStore.getState().workspaceId).toBe(outgoing)
    expect(screen.queryByTestId("activated-body")).toBeNull()
  })

  it.each(["unmount", "account-aba", "workspace-aba"])("ignores a resolver that fulfills after %s", async change => {
    const scope = deferred<typeof boundary.scope>()
    boundary.resolve.mockReturnValueOnce(scope.promise)
    const outgoing = useWorkspaceStore.getState().workspaceId
    const view = render(<ResearchWorkspaceRouteGate />)
    await waitFor(() => expect(boundary.resolve).toHaveBeenCalled())
    const signal = boundary.resolve.mock.calls[0][0].signal
    act(() => {
      if (change === "unmount") view.unmount()
      if (change === "account-aba") boundary.changed?.(true)
      if (change === "workspace-aba") {
        useWorkspaceStore.getState().createNewWorkspace("Intervening")
        useWorkspaceStore.getState().switchWorkspace(outgoing)
      }
    })
    await act(async () => scope.resolve(boundary.scope))
    expect(signal.aborted).toBe(true)
    expect(boundary.getWorkspace).not.toHaveBeenCalled()
    expect(useWorkspaceStore.getState().workspaceId).toBe(outgoing)
    expect(screen.queryByTestId("activated-body")).toBeNull()
    if (change !== "unmount") expect(screen.getByRole("alert")).toBeVisible()
  })

  it("keeps an already-active WebClip draft without replaying its stored snapshot", async () => {
    const outgoing = useWorkspaceStore.getState().workspaceId
    boundary.location.search = `?workspace=${outgoing}&agent_task_handoff=web_clip`
    render(<ResearchWorkspaceRouteGate />)
    expect(await screen.findByTestId("activated-body")).toBeVisible()
    expect(useWorkspaceStore.getState().currentNote.content).toBe("Dirty draft")
    expect(boundary.getWorkspace).not.toHaveBeenCalled()
  })

  it.each(["active", "cached"])("rejects an initially foreign-owner %s canonical WebClip target", async target => {
    const hydrated = await hydrateWorkspaceFromServer("server-research", { fetch: async () => serverWorkspacePayload() })
    useWorkspaceStore.getState().installServerWorkspace(hydrated, {
      scopeKey: "owner-a", expectedWorkspaceId: useWorkspaceStore.getState().workspaceId
    })
    useWorkspaceStore.getState().updateNoteContent("Private owner A draft")
    if (target === "cached") useWorkspaceStore.getState().createNewWorkspace("Owner B outgoing")
    const outgoing = useWorkspaceStore.getState().workspaceId
    boundary.scope.scopeKey = "owner-b"
    boundary.location.search = "?workspace=server-research&agent_task_handoff=web_clip"
    render(<ResearchWorkspaceRouteGate />)
    expect(await screen.findByRole("alert")).toBeVisible()
    expect(screen.queryByTestId("activated-body")).toBeNull()
    expect(useWorkspaceStore.getState().workspaceId).toBe(outgoing)
    const retained = target === "active" ? useWorkspaceStore.getState() : useWorkspaceStore.getState().workspaceSnapshots["server-research"]
    expect(retained.serverWorkspace?.scopeKey).toBe("owner-a")
    expect(retained.currentNote.content).toBe("Private owner A draft")
    expect(boundary.resolve).toHaveBeenCalledTimes(2)
    for (const method of [boundary.getWorkspace, boundary.getWorkspaceSources, boundary.getWorkspaceArtifacts, boundary.getWorkspaceNotes]) {
      expect(method).toHaveBeenCalledWith("server-research", expect.objectContaining({
        signal: expect.any(AbortSignal), requestScope: { config: boundary.scope.config, userId: 42 }
      }))
    }
  })

  it("rehydrates an owned active canonical WebClip target instead of using the local shortcut", async () => {
    const hydrated = await hydrateWorkspaceFromServer("server-research", { fetch: async () => serverWorkspacePayload() })
    useWorkspaceStore.getState().installServerWorkspace(hydrated, {
      scopeKey: "owner-a", expectedWorkspaceId: useWorkspaceStore.getState().workspaceId
    })
    useWorkspaceStore.getState().updateNoteContent("Owned draft")
    const notes = deferred<ReturnType<typeof serverWorkspacePayload>["notes"]>()
    boundary.getWorkspaceNotes.mockReturnValue(notes.promise)
    boundary.location.search = "?workspace=server-research&agent_task_handoff=web_clip"
    render(<ResearchWorkspaceRouteGate />)
    await waitFor(() => expect(boundary.getWorkspaceNotes).toHaveBeenCalled())
    expect(screen.queryByTestId("activated-body")).toBeNull()
    await act(async () => notes.resolve(serverWorkspacePayload().notes))
    expect(await screen.findByTestId("activated-body")).toBeVisible()
    expect(useWorkspaceStore.getState().currentNote.content).toBe("Owned draft")
    expect(boundary.resolve).toHaveBeenCalledTimes(2)
  })

  it.each(["missing", "archived-only"])("fails closed for an unavailable %s WebClip target", async kind => {
    const outgoing = useWorkspaceStore.getState().workspaceId
    if (kind === "archived-only") {
      useWorkspaceStore.getState().createNewWorkspace("Archived")
      const target = useWorkspaceStore.getState().workspaceId
      useWorkspaceStore.getState().archiveWorkspace(target)
      const snapshots = { ...useWorkspaceStore.getState().workspaceSnapshots }
      delete snapshots[target]
      useWorkspaceStore.setState({ workspaceSnapshots: snapshots })
      boundary.location.search = `?workspace=${target}&agent_task_handoff=web_clip`
    } else boundary.location.search = "?workspace=missing&agent_task_handoff=web_clip"
    boundary.getWorkspace.mockRejectedValue(new Error("Unavailable"))
    render(<ResearchWorkspaceRouteGate />)
    expect(await screen.findByRole("alert")).toBeVisible()
    expect(useWorkspaceStore.getState().workspaceId).toBe(outgoing)
    expect(screen.queryByTestId("activated-body")).toBeNull()
  })

  it("does not mark a failed local WebClip switch ready", async () => {
    useWorkspaceStore.getState().createNewWorkspace("Local clip")
    const local = useWorkspaceStore.getState().workspaceId
    useWorkspaceStore.getState().createNewWorkspace("Other")
    const other = useWorkspaceStore.getState().workspaceId
    vi.spyOn(useWorkspaceStore.getState(), "switchWorkspace").mockImplementation(() => {})
    boundary.location.search = `?workspace=${local}&agent_task_handoff=web_clip`
    render(<ResearchWorkspaceRouteGate />)
    expect(await screen.findByRole("alert")).toBeVisible()
    expect(useWorkspaceStore.getState().workspaceId).toBe(other)
    expect(screen.queryByTestId("activated-body")).toBeNull()
  })

  it("retries transient failure with a new captured request", async () => {
    boundary.getWorkspace.mockRejectedValueOnce(new Error("Transient"))
    render(<ResearchWorkspaceRouteGate />)
    await screen.findByRole("alert")
    const firstSignal = boundary.getWorkspace.mock.calls[0][1].signal
    fireEvent.click(screen.getByRole("button", { name: "Retry workspace" }))
    expect(await screen.findByTestId("activated-body")).toBeVisible()
    expect(firstSignal.aborted).toBe(true)
    expect(boundary.getWorkspace.mock.calls[1][1].signal).not.toBe(firstSignal)
  })

  it("consumes a hash activation query without changing shared search precedence", async () => {
    boundary.location = { search: "", hash: "#/research-workspace?workspace=server-research", key: "hash" }
    render(<ResearchWorkspaceRouteGate />)
    expect(await screen.findByTestId("activated-body")).toBeVisible()
    expect(useWorkspaceStore.getState().workspaceId).toBe("server-research")
  })

  it("waits for local persistence hydration before staging server reads", async () => {
    useWorkspaceStore.setState({ storeHydrated: false })
    render(<ResearchWorkspaceRouteGate />)
    await act(async () => {})
    expect(boundary.getWorkspace).not.toHaveBeenCalled()
    expect(screen.queryByTestId("activated-body")).toBeNull()
    act(() => { useWorkspaceStore.setState({ storeHydrated: true }) })
    expect(await screen.findByTestId("activated-body")).toBeVisible()
  })

  it("keeps an existing local WebClip handoff on its local switch path", async () => {
    useWorkspaceStore.getState().createNewWorkspace("Local clip")
    const local = useWorkspaceStore.getState().workspaceId
    useWorkspaceStore.getState().createNewWorkspace("Other")
    const other = useWorkspaceStore.getState().workspaceId
    boundary.location.search = `?workspace=${local}&agent_task_handoff=web_clip`
    render(<ResearchWorkspaceRouteGate />)
    expect(await screen.findByTestId("activated-body")).toBeVisible()
    expect(useWorkspaceStore.getState().workspaceId).toBe(local)
    expect(useWorkspaceStore.getState().workspaceSnapshots[other]).toBeDefined()
    expect(boundary.getWorkspace).not.toHaveBeenCalled()
  })
})
