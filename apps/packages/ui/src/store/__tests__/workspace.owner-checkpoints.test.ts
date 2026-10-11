import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { HistorySelectionReference } from "@/hooks/chat/useHistorySelection"
import { WORKSPACE_STORAGE_KEY } from "@/store/workspace-events"
import type {
  WorkspaceChatSession,
  WorkspaceChatSessionQualification
} from "../workspace"

const reference: HistorySelectionReference = {
  profile_id: "profile-a",
  client_session_id: "client-a",
  owner_key: "native-history-owner-a",
  conversation_id: "server-chat-a",
  pending_confirmation_reference: {
    client_session_id: "pending-client-a",
    view_session_id: "view-a",
    projection_id: "projection-a"
  }
}
const message = {
  isBot: false,
  name: "You",
  message: "Retained message",
  sources: []
}
const captureStorage = () =>
  Object.fromEntries(
    Array.from({ length: localStorage.length }, (_, index) => {
      const key = localStorage.key(index)!
      return [key, localStorage.getItem(key)]
    })
  )

describe.each([
  ["monolithic", "0"],
  ["split", "1"]
])("workspace owner checkpoints (%s storage)", (_mode, splitFlag) => {
  let store: typeof import("../workspace").useWorkspaceStore
  let qualification: WorkspaceChatSessionQualification
  let sessionKey: string

  const makeSession = (): WorkspaceChatSession => ({
    messages: [{ ...message }],
    history: [{ role: "user", content: "Retained message" }],
    historyId: "local-mirror-a",
    serverChatId: "server-chat-a",
    checkpoint: {
      version: 1,
      ...qualification,
      historySelectionReference: structuredClone(reference),
      draft: "Unsent draft A"
    }
  })

  const expectReadOnlyRejection = (
    key = sessionKey,
    expected = qualification
  ) => {
    const sessionsBefore = store.getState().workspaceChatSessions
    const storageBefore = captureStorage()
    const setItem = vi.spyOn(Object.getPrototypeOf(localStorage), "setItem")
    const removeItem = vi.spyOn(Object.getPrototypeOf(localStorage), "removeItem")

    expect(store.getState().getWorkspaceChatSession(key, expected)).toBeNull()
    expect(store.getState().workspaceChatSessions).toBe(sessionsBefore)
    expect(captureStorage()).toEqual(storageBefore)
    expect(setItem).not.toHaveBeenCalled()
    expect(removeItem).not.toHaveBeenCalled()
  }

  beforeEach(async () => {
    vi.resetModules()
    localStorage.clear()
    localStorage.setItem(
      "tldw:feature-rollout:workspace_split_storage_v1:enabled", splitFlag
    )
    localStorage.setItem(
      "tldw:feature-rollout:workspace_indexeddb_offload_v1:enabled", "0"
    )
    store = (await import("../workspace")).useWorkspaceStore
    store.getState().initializeWorkspace("Checkpoint workspace")
    qualification = {
      ownerKey: "mirror-target-account-a",
      workspaceId: store.getState().workspaceId,
      referenceId: "reference-a"
    }
    sessionKey = `${qualification.workspaceId}::reference-a`
  })

  afterEach(() => {
    vi.restoreAllMocks()
    localStorage.clear()
  })

  it("reads a qualified exact record without conflating mirror and H1 owner keys", () => {
    const session = makeSession()
    store.setState({ workspaceChatSessions: { [sessionKey]: session } })

    expect(store.getState().getWorkspaceChatSession(sessionKey, qualification)).toEqual(session)
  })

  it("never falls back to a base record for qualified reads but retains legacy fallback", () => {
    const session = makeSession()
    store.setState({
      workspaceChatSessions: { [qualification.workspaceId]: session }
    })

    expectReadOnlyRejection()
    expect(store.getState().getWorkspaceChatSession(sessionKey)?.historyId).toBe("local-mirror-a")
  })

  it("rejects unqualified legacy rows without adopting or removing them", () => {
    const { checkpoint: _checkpoint, ...legacy } = makeSession()
    store.setState({ workspaceChatSessions: { [sessionKey]: legacy } })

    expectReadOnlyRejection()
    expect(store.getState().getWorkspaceChatSession(sessionKey)).toEqual(legacy)
  })

  it.each([
    ["version", { version: 2 }],
    ["owner", { ownerKey: "mirror-target-account-b" }],
    ["workspace", { workspaceId: "other-workspace" }],
    ["reference", { referenceId: "reference-b" }],
    ["draft", { draft: null }]
  ])("rejects invalid checkpoint %s without writes", (_name, changes) => {
    const session = makeSession()
    session.checkpoint = { ...session.checkpoint, ...changes } as WorkspaceChatSession["checkpoint"]
    store.setState({ workspaceChatSessions: { [sessionKey]: session } })

    expectReadOnlyRejection()
  })

  it.each([
    ["account/target", { ownerKey: "mirror-target-account-b" }],
    ["workspace", { workspaceId: "other-workspace" }],
    ["reference", { referenceId: "reference-b" }],
    ["empty owner", { ownerKey: "" }]
  ])("rejects a changed qualification %s", (_name, changes) => {
    store.setState({ workspaceChatSessions: { [sessionKey]: makeSession() } })

    expectReadOnlyRejection(sessionKey, { ...qualification, ...changes })
  })

  it("rejects a key that disagrees with otherwise matching checkpoint metadata", () => {
    store.setState({
      workspaceChatSessions: { [qualification.workspaceId]: makeSession() }
    })

    expectReadOnlyRejection(qualification.workspaceId)
  })

  it.each([
    ["missing field", { owner_key: "native-history-owner-a", conversation_id: "server-chat-a" }],
    ["empty owner", { ...reference, owner_key: "" }],
    ["extra field", { ...reference, unexpected: true }],
    ["invalid pending pointer", { ...reference, pending_confirmation_reference: { client_session_id: "x" } }],
    ["extra pending field", { ...reference, pending_confirmation_reference: { ...reference.pending_confirmation_reference, extra: true } }],
    ["wrong conversation", { ...reference, conversation_id: "other-chat" }],
    ["local mirror for native chat", { ...reference, conversation_id: "local-mirror-a" }],
    ["handoff discriminator", { ...reference, owner_kind: "native" }],
    ["missing reference", undefined]
  ])("rejects invalid H1 %s without writes", (_name, invalidReference) => {
    const session = makeSession()
    session.checkpoint!.historySelectionReference = invalidReference as HistorySelectionReference
    store.setState({ workspaceChatSessions: { [sessionKey]: session } })

    expectReadOnlyRejection()
  })

  it("accepts a local H1 reference only for its existing local history ID", () => {
    const session = makeSession()
    session.serverChatId = null
    session.checkpoint!.historySelectionReference = {
      ...reference, owner_key: "local-history-owner-a", conversation_id: "local-mirror-a"
    }
    store.setState({ workspaceChatSessions: { [sessionKey]: session } })

    expect(store.getState().getWorkspaceChatSession(sessionKey, qualification)).toEqual(session)
  })

  it.each([
    ["messages", { messages: [{ ...message }] }],
    ["history", { history: [{ role: "user", content: "Orphan history" }] }],
    ["history ID", { historyId: "orphan-history" }],
    ["server ID", { serverChatId: "orphan-server" }]
  ])("rejects null H1 with existing %s", (_name, existing) => {
    const session = makeSession()
    store.setState({ workspaceChatSessions: {
      [sessionKey]: {
        ...session, messages: [], history: [], historyId: null, serverChatId: null,
        ...existing,
        checkpoint: { ...session.checkpoint!, historySelectionReference: null }
      } as WorkspaceChatSession
    } })

    expectReadOnlyRejection()
  })

  it("keeps checkpoint and nested H1 clones independent on save and read", () => {
    const session = makeSession()
    store.getState().saveWorkspaceChatSession(sessionKey, session)
    session.checkpoint!.draft = "Changed caller draft"
    session.checkpoint!.historySelectionReference!.pending_confirmation_reference!.projection_id = "changed-caller"
    const saved = store.getState().getWorkspaceChatSession(sessionKey, qualification)!

    expect(saved?.checkpoint?.draft).toBe("Unsent draft A")
    expect(saved?.checkpoint?.historySelectionReference?.pending_confirmation_reference?.projection_id).toBe("projection-a")
    saved.checkpoint!.historySelectionReference!.pending_confirmation_reference!.projection_id = "changed-reader"
    expect(store.getState().getWorkspaceChatSession(sessionKey, qualification)?.checkpoint?.historySelectionReference?.pending_confirmation_reference?.projection_id).toBe("projection-a")
  })

  it("round-trips checkpoint metadata with bounded messages and no duplicate persisted history", async () => {
    const session = makeSession()
    session.messages = Array.from({ length: 300 }, (_, index) => ({ ...message, message: `Message ${index + 1}` }))
    store.getState().saveWorkspaceChatSession(sessionKey, session)
    const persisted = await store.persist.getOptions().storage!.getItem(WORKSPACE_STORAGE_KEY)
    const persistedSession = persisted!.state.workspaceChatSessions[sessionKey]

    expect(persistedSession.checkpoint).toEqual(session.checkpoint)
    expect(persistedSession.messages).toHaveLength(250)
    expect(persistedSession.messages[0].message).toBe("Message 51")
    expect(persistedSession).not.toHaveProperty("history")
    await store.persist.rehydrate()
    const restored = store.getState().getWorkspaceChatSession(sessionKey, qualification)
    expect(restored?.checkpoint).toEqual(session.checkpoint)
    expect(restored?.messages).toHaveLength(250)
  })

  it("retains an empty draft-only checkpoint with null H1 through normalization", async () => {
    const session: WorkspaceChatSession = {
      messages: [], history: [], historyId: null, serverChatId: null,
      checkpoint: { version: 1, ...qualification, historySelectionReference: null, draft: "Draft before first send" }
    }
    store.getState().saveWorkspaceChatSession(sessionKey, session)
    await store.persist.rehydrate()

    expect(store.getState().getWorkspaceChatSession(sessionKey, qualification)).toEqual(session)
  })

  it("retains mismatched checkpoint copies across normalization without granting qualified reads", async () => {
    const session = makeSession()
    session.checkpoint!.ownerKey = "mirror-target-account-b"
    store.getState().saveWorkspaceChatSession(sessionKey, session)
    await store.persist.rehydrate()

    expectReadOnlyRejection()
    expect(store.getState().getWorkspaceChatSession(sessionKey)?.checkpoint).toEqual(session.checkpoint)
  })
})
