import { beforeEach, describe, expect, it } from "vitest"
import { createWorkspaceStorage, useWorkspaceStore, WORKSPACE_STORAGE_SPLIT_KEY_FLAG_STORAGE_KEY,
  WORKSPACE_STORAGE_INDEXEDDB_FLAG_STORAGE_KEY } from "../workspace"
import { WORKSPACE_STORAGE_KEY } from "../workspace-events"
import { hydrateWorkspaceFromServer } from "../workspace-api"
import { serverWorkspacePayload } from "./workspace-activation.fixtures"

const id = "server-research"
const receiptKey = `tldw:research-workspace:migration:tombstone:${id}`
const markDeleted = () => localStorage.setItem(receiptKey, JSON.stringify({
  legacyWorkspaceId: id, serverWorkspaceId: id, migrationId: "migration", contentRetained: false,
  serverScopeKey: "owner-a", deletedAt: "2026-10-03T00:00:00Z"
}))
const payload = () => {
  const options = useWorkspaceStore.persist.getOptions()
  return JSON.stringify({ state: options.partialize!(useWorkspaceStore.getState()), version: options.version })
}
const snapshotKey = `${WORKSPACE_STORAGE_KEY}:workspace:${id}:snapshot`
const chatKey = `${WORKSPACE_STORAGE_KEY}:workspace:${id}:chat`

beforeEach(async () => {
  localStorage.clear()
  useWorkspaceStore.getState().reset()
  const local = await hydrateWorkspaceFromServer(id, { requireComplete: true, fetch: async () => serverWorkspacePayload() })
  expect(useWorkspaceStore.getState().installServerWorkspace(local, { scopeKey: "owner-a", expectedWorkspaceId: "" })).toBe(true)
  useWorkspaceStore.getState().updateNoteContent("Retained private draft")
  useWorkspaceStore.getState().saveWorkspaceChatSession(id, { messages: [], history: [], historyId: "history", serverChatId: "chat" })
  useWorkspaceStore.getState().saveWorkspaceChatSession(`${id}::conversation`, { messages: [], history: [], historyId: "other-history", serverChatId: "other-chat" })
  useWorkspaceStore.getState().saveCurrentWorkspace()
})

describe.each(["split", "monolithic"])("tombstone union with %s storage", mode => {
  const storage = () => {
    localStorage.setItem(WORKSPACE_STORAGE_SPLIT_KEY_FLAG_STORAGE_KEY, mode === "split" ? "1" : "0")
    localStorage.setItem(WORKSPACE_STORAGE_INDEXEDDB_FLAG_STORAGE_KEY, "0")
    return createWorkspaceStorage()
  }

  it.each(["legacy", "pin-only", "note-only", "wrong-workspace", "missing-owner"])("suppresses %s caches on read and repeated write, not just active selection", async kind => {
    if (kind === "wrong-workspace") {
      const canonical = useWorkspaceStore.getState().serverWorkspace!
      useWorkspaceStore.setState({ serverWorkspace: { ...canonical, metadata: { ...canonical.metadata, id: "foreign" } } })
    } else useWorkspaceStore.setState({ serverWorkspace: null })
    if (kind === "note-only") useWorkspaceStore.getState().setCurrentNote({
      ...useWorkspaceStore.getState().currentNote, serverWorkspaceId: id, serverScopeKey: "owner-a"
    })
    if (kind === "pin-only") useWorkspaceStore.setState({ sources: [{ ...useWorkspaceStore.getState().sources[0], webCapture: {
      clipId: "pin", mediaId: 101, versionNumber: 1, versionUuid: "version", contentSha256: "digest",
      requestedUrl: "https://article.test", capturedAt: "2026-10-07T00:00:00Z", refreshOf: null
    } }] })
    useWorkspaceStore.getState().saveCurrentWorkspace()
    const candidate = JSON.parse(payload())
    if (kind === "missing-owner") candidate.state.workspaceSnapshots[id].serverWorkspace = { metadata: { id, workspace_profile: "research" } }
    candidate.state.archivedWorkspaces = [{ ...candidate.state.savedWorkspaces[0], id }]
    const raw = JSON.stringify(candidate)
    const adapter = storage()
    await adapter.setItem(WORKSPACE_STORAGE_KEY, raw)
    markDeleted()
    const marker = localStorage.getItem(receiptKey)
    const read = JSON.parse((await adapter.getItem(WORKSPACE_STORAGE_KEY))!)
    expect(read.state.workspaceId).not.toBe(id)
    expect(read.state.workspaceSnapshots[id]).toBeUndefined()
    expect(read.state.workspaceChatSessions[id]).toBeUndefined()
    expect(read.state.workspaceChatSessions[`${id}::conversation`]).toBeUndefined()
    expect(read.state.savedWorkspaces.some((item: { id: string }) => item.id === id)).toBe(false)
    expect(read.state.archivedWorkspaces.some((item: { id: string }) => item.id === id)).toBe(false)
    await adapter.setItem(WORKSPACE_STORAGE_KEY, raw)
    await adapter.setItem(WORKSPACE_STORAGE_KEY, raw)
    expect(localStorage.getItem(WORKSPACE_STORAGE_KEY)).not.toContain(id)
    expect(localStorage.getItem(snapshotKey)).toBeNull()
    expect(localStorage.getItem(chatKey)).toBeNull()
    expect(localStorage.getItem(`${WORKSPACE_STORAGE_KEY}:workspace:${encodeURIComponent(`${id}::conversation`)}:chat`)).toBeNull()
    expect(localStorage.getItem(receiptKey)).toBe(marker)
  })

  it("preserves installed canonical ownership, draft, membership and chat despite a same-ID legacy receipt", async () => {
    const raw = payload()
    const adapter = storage()
    await adapter.setItem(WORKSPACE_STORAGE_KEY, raw)
    markDeleted()
    await adapter.setItem(WORKSPACE_STORAGE_KEY, raw)
    const read = JSON.parse((await adapter.getItem(WORKSPACE_STORAGE_KEY))!)
    expect(read.state.workspaceId).toBe(id)
    expect(read.state.workspaceSnapshots[id]).toMatchObject({
      serverWorkspace: { scopeKey: "owner-a", metadata: { id, workspace_profile: "research" } },
      currentNote: { content: "Retained private draft", isDirty: true },
      sources: [{ id: "server-source", mediaId: 101 }]
    })
    expect(read.state.workspaceChatSessions[id].serverChatId).toBe("chat")
    expect(read.state.workspaceChatSessions[`${id}::conversation`].serverChatId).toBe("other-chat")
  })

  it.each(["missing-server", "missing-time", "invalid-time", "copies-retained"])("does not treat a %s receipt as deletion authority for a retained draft", async kind => {
    useWorkspaceStore.setState({ serverWorkspace: null })
    useWorkspaceStore.getState().saveCurrentWorkspace()
    const raw = payload()
    const adapter = storage()
    await adapter.setItem(WORKSPACE_STORAGE_KEY, raw)
    markDeleted()
    const marker = JSON.parse(localStorage.getItem(receiptKey)!)
    if (kind === "missing-server") delete marker.serverWorkspaceId
    if (kind === "missing-time") delete marker.deletedAt
    if (kind === "invalid-time") marker.deletedAt = "invalid"
    if (kind === "copies-retained") marker.contentRetained = true
    localStorage.setItem(receiptKey, JSON.stringify(marker))
    await adapter.setItem(WORKSPACE_STORAGE_KEY, raw)
    const read = JSON.parse((await adapter.getItem(WORKSPACE_STORAGE_KEY))!)
    expect(read.state.workspaceId).toBe(id)
    expect(read.state.workspaceSnapshots[id].currentNote.content).toBe("Retained private draft")
  })
})
