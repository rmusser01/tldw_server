import { beforeEach, describe, expect, it } from "vitest"
import { useWorkspaceStore } from "../workspace"
import { hydrateWorkspaceFromServer } from "../workspace-api"
import { serverWorkspaceMetadata, serverWorkspacePayload } from "./workspace-activation.fixtures"

describe("canonical server workspace activation (unit regression)", () => {
  beforeEach(() => {
    localStorage.clear()
    useWorkspaceStore.getState().reset()
    useWorkspaceStore.setState({ storeHydrated: true, savedWorkspaces: [], workspaceSnapshots: {} })
  })

  it("retains fetched metadata with the existing source/artifact projection", async () => {
    const hydrated = await hydrateWorkspaceFromServer("server-research", {
      fetch: async () => serverWorkspacePayload(), requireComplete: true
    })
    expect(hydrated.metadata).toMatchObject(serverWorkspaceMetadata)
    expect(hydrated.selectedSourceIds).toEqual([])
    expect(hydrated.artifacts[0].content).toBe("Report body")
    expect(hydrated.notes[0].workspace_id).toBe("server-research")
  })

  it.each(["sources", "artifacts", "notes"] as const)("rejects incomplete required %s before activation", async resource => {
    const payload = serverWorkspacePayload()
    delete payload[resource]
    await expect(hydrateWorkspaceFromServer("server-research", {
      fetch: async () => payload, requireComplete: true
    })).rejects.toThrow(/incomplete/i)
  })

  it("rejects a foreign resource even when the requested metadata matches", async () => {
    const payload = serverWorkspacePayload()
    payload.notes![0].workspace_id = "foreign"
    await expect(hydrateWorkspaceFromServer("server-research", {
      fetch: async () => payload, requireComplete: true
    })).rejects.toThrow(/workspace/i)
  })

  it("installs all fetched data while preserving the latest outgoing dirty draft", async () => {
    const store = useWorkspaceStore.getState()
    const outgoingId = store.initializeWorkspace("Outgoing")
    store.updateNoteContent("Unsaved outgoing draft")
    const hydrated = await hydrateWorkspaceFromServer("server-research", { fetch: async () => serverWorkspacePayload() })
    const install = useWorkspaceStore.getState().installServerWorkspace
    expect(install).toBeTypeOf("function")
    expect(install(hydrated, { scopeKey: "owner-a", expectedWorkspaceId: outgoingId })).toBe(true)
    const active = useWorkspaceStore.getState()
    expect(active.workspaceId).toBe("server-research")
    expect(active.sources[0].mediaId).toBe(101)
    expect(active.studyMaterialsPolicy).toBe("general")
    expect(active.assistantDefaults?.assistantId).toBe("persona-1")
    expect(active.serverWorkspace?.notes[0].content).toBe("Canonical body")
    expect(active.workspaceSnapshots[outgoingId].currentNote).toMatchObject({ content: "Unsaved outgoing draft", isDirty: true })
    active.switchWorkspace(outgoingId)
    expect(useWorkspaceStore.getState().currentNote.content).toBe("Unsaved outgoing draft")
    expect(useWorkspaceStore.getState().serverWorkspace).toBeNull()
  })

  it("refuses installation after an intervening active workspace selection", async () => {
    const first = useWorkspaceStore.getState().initializeWorkspace("First")
    useWorkspaceStore.getState().createNewWorkspace("Second")
    const second = useWorkspaceStore.getState().workspaceId
    const hydrated = await hydrateWorkspaceFromServer("server-research", { fetch: async () => serverWorkspacePayload() })
    const install = useWorkspaceStore.getState().installServerWorkspace
    expect(install).toBeTypeOf("function")
    expect(install(hydrated, { scopeKey: "owner-a", expectedWorkspaceId: first })).toBe(false)
    expect(useWorkspaceStore.getState().workspaceId).toBe(second)
    expect(useWorkspaceStore.getState().workspaceSnapshots["server-research"]).toBeUndefined()
  })

  it.each(["owner", "workspace", "numeric-id", "empty-id"])("rejects a restored UUID note with mismatched %s authority", async mismatch => {
    const hydrated = await hydrateWorkspaceFromServer("server-research", { fetch: async () => serverWorkspacePayload() })
    hydrated.currentNote = {
      id: mismatch === "numeric-id" ? 7 : mismatch === "empty-id" ? "" : "12345678-1234-4123-8123-123456789abc",
      title: "Scoped note", content: "Private note", keywords: [], isDirty: false,
      serverWorkspaceId: mismatch === "workspace" ? "foreign" : "server-research",
      serverScopeKey: mismatch === "owner" ? "owner-b" : "owner-a",
    }
    const before = useWorkspaceStore.getState()
    expect(before.installServerWorkspace(hydrated, { scopeKey: "owner-a", expectedWorkspaceId: "" })).toBe(false)
    expect(useWorkspaceStore.getState()).toBe(before)
  })

  it.each(["dirty", "title", "content", "keywords"])("preserves an empty-ID %s note rather than automatically adopting it", async kind => {
    useWorkspaceStore.getState().setCurrentNote({
      title: kind === "title" ? "Retained title" : "",
      content: kind === "content" ? "Retained content" : "",
      keywords: kind === "keywords" ? ["retained-keyword"] : [],
      isDirty: kind === "dirty",
    })
    const before = useWorkspaceStore.getState()
    const hydrated = await hydrateWorkspaceFromServer("server-research", { fetch: async () => serverWorkspacePayload() })
    expect(before.installServerWorkspace(hydrated, { scopeKey: "owner-a", expectedWorkspaceId: "" })).toBe(false)
    expect(useWorkspaceStore.getState()).toBe(before)
  })

  it("retains local pane state and native chat reference when refreshing an owned target", async () => {
    const hydrated = await hydrateWorkspaceFromServer("server-research", { fetch: async () => serverWorkspacePayload() })
    useWorkspaceStore.getState().installServerWorkspace(hydrated, { scopeKey: "owner-a", expectedWorkspaceId: useWorkspaceStore.getState().workspaceId })
    useWorkspaceStore.setState({ leftPaneCollapsed: true, workspaceChatReferenceId: "native-reference" })
    useWorkspaceStore.getState().saveWorkspaceChatSession("server-research", {
      messages: [{ isBot: false, name: "User", role: "user", message: "Kept turn", sources: [], id: "local-turn" }],
      history: [{ role: "user", content: "Kept turn" }], historyId: "history-reference", serverChatId: "native-chat"
    })
    const chatSessions = useWorkspaceStore.getState().workspaceChatSessions
    useWorkspaceStore.getState().updateNoteContent("Same target draft")
    useWorkspaceStore.getState().installServerWorkspace(hydrated, { scopeKey: "owner-a", expectedWorkspaceId: "server-research" })
    expect(useWorkspaceStore.getState()).toMatchObject({ leftPaneCollapsed: true, workspaceChatReferenceId: "native-reference",
      currentNote: { content: "Same target draft", isDirty: true } })
    expect(useWorkspaceStore.getState().workspaceChatSessions).toBe(chatSessions)
    expect(useWorkspaceStore.getState().getWorkspaceChatSession("server-research")).toMatchObject({
      serverChatId: "native-chat", historyId: "history-reference", messages: [{ id: "local-turn", message: "Kept turn" }]
    })
  })

  it("does not relabel a dirty canonical draft as a different owner", async () => {
    const hydrated = await hydrateWorkspaceFromServer("server-research", { fetch: async () => serverWorkspacePayload() })
    useWorkspaceStore.getState().installServerWorkspace(hydrated, { scopeKey: "owner-a", expectedWorkspaceId: useWorkspaceStore.getState().workspaceId })
    useWorkspaceStore.getState().setCurrentNote({ title: "Private draft", content: "Owner A", keywords: [], isDirty: true,
      serverWorkspaceId: "server-research", serverScopeKey: "owner-a" })
    useWorkspaceStore.setState({ notes: "Owner A notes", workspaceBanner: {
      ...useWorkspaceStore.getState().workspaceBanner, image: "data:image/png;base64,owner-a"
    } })
    const previous = useWorkspaceStore.getState()
    expect(useWorkspaceStore.getState().installServerWorkspace(hydrated, { scopeKey: "owner-b", expectedWorkspaceId: "server-research" })).toBe(false)
    expect(useWorkspaceStore.getState()).toBe(previous)
    expect(useWorkspaceStore.getState().serverWorkspace.scopeKey).toBe("owner-a")
    expect(useWorkspaceStore.getState().currentNote.content).toBe("Owner A")
  })

  it("fails closed rather than assigning an unowned legacy target draft to a principal", async () => {
    useWorkspaceStore.getState().loadWorkspace({ id: "server-research", name: "Legacy", tag: "workspace:legacy", createdAt: new Date(), updatedAt: new Date() })
    useWorkspaceStore.getState().updateNoteContent("Unowned local draft")
    const hydrated = await hydrateWorkspaceFromServer("server-research", { fetch: async () => serverWorkspacePayload() })
    expect(useWorkspaceStore.getState().installServerWorkspace(hydrated, { scopeKey: "owner-a", expectedWorkspaceId: "server-research" })).toBe(false)
    expect(useWorkspaceStore.getState().serverWorkspace).toBeNull()
    expect(useWorkspaceStore.getState().currentNote.content).toBe("Unowned local draft")
  })

  it.each(["canonical", "local"])("atomically restores %s provenance after archive switches to the other kind", async kind => {
    const local = useWorkspaceStore.getState().initializeWorkspace("Local")
    const hydrated = await hydrateWorkspaceFromServer("server-research", { fetch: async () => serverWorkspacePayload() })
    useWorkspaceStore.getState().installServerWorkspace(hydrated, { scopeKey: "owner-a", expectedWorkspaceId: local })
    if (kind === "local") useWorkspaceStore.getState().switchWorkspace(local)
    useWorkspaceStore.getState().updateNoteContent("Restored draft")
    const target = useWorkspaceStore.getState().workspaceId
    const snapshot = useWorkspaceStore.getState().captureUndoSnapshot()
    useWorkspaceStore.getState().archiveWorkspace(target)
    expect(useWorkspaceStore.getState().workspaceId).not.toBe(target)
    const transitions: { id: string; scope: string | null }[] = []
    const stop = useWorkspaceStore.subscribe(state => {
      transitions.push({ id: state.workspaceId, scope: state.serverWorkspace?.scopeKey ?? null })
    })
    try { useWorkspaceStore.getState().restoreUndoSnapshot(snapshot) } finally { stop() }
    expect(transitions).toEqual([{ id: target, scope: kind === "canonical" ? "owner-a" : null }])
    expect(useWorkspaceStore.getState().currentNote.content).toBe("Restored draft")
    if (kind === "canonical") expect(useWorkspaceStore.getState().serverWorkspace?.notes[0].content).toBe("Canonical body")
    else expect(useWorkspaceStore.getState().serverWorkspace).toBeNull()
  })
})
