// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from "vitest"

import { buildQueuedRequest } from "@/utils/chat-request-queue"
import type { ChatDraftHandle } from "@/utils/chat-draft-address"
import { usePlaygroundSessionStore } from "../playground-session"

describe("playground-session-store", () => {
  beforeEach(() => {
    usePlaygroundSessionStore.getState().clearSession()
  })

  it("treats a fresh queue-only session as valid", () => {
    usePlaygroundSessionStore.getState().saveSession({
      scopeKey: "scope:a",
      queuedMessages: [buildQueuedRequest({ promptText: "Run this later" })]
    })

    expect(usePlaygroundSessionStore.getState().isSessionValid()).toBe(true)
  })

  it("persists a valid draft-only handle, preserving the Temporary bit and excluding runtime intent", () => {
    const chatDraftHandle: ChatDraftHandle = { ownerHash: "a".repeat(64), address: { kind: "fresh", id: "new", scope: { type: "global" }, temporary: true } }
    usePlaygroundSessionStore.getState().saveSession({ scopeKey: "scope:a", chatDraftHandle })
    expect(usePlaygroundSessionStore.getState().isSessionValid("scope:a")).toBe(true)
    expect(JSON.parse(localStorage.getItem("tldw-playground-session")!).state.chatDraftHandle).toEqual(chatDraftHandle)
    usePlaygroundSessionStore.getState().clearSession()
    expect(usePlaygroundSessionStore.getState().chatDraftHandle).toBeNull()
  })

  it.each([
    { ownerHash: "credentials", address: { kind: "fresh", id: "new", scope: { type: "global" }, temporary: false } },
    { ownerHash: "a".repeat(64), address: { kind: "fresh", id: "new", scope: { type: "workspace", workspaceId: "" }, temporary: false } }
  ])("does not treat a malformed draft-only handle as a saved session (%j)", chatDraftHandle => {
    usePlaygroundSessionStore.getState().saveSession({ chatDraftHandle: chatDraftHandle as unknown as ChatDraftHandle })
    expect(usePlaygroundSessionStore.getState().isSessionValid()).toBe(false)
  })

  it("keeps deliberate server selection transient and clears it with account state", () => {
    usePlaygroundSessionStore.getState().requestServerChatSelection("chat-a")
    const first = usePlaygroundSessionStore.getState().serverChatSelectionIntent
    usePlaygroundSessionStore.getState().requestServerChatSelection("chat-a")
    expect(usePlaygroundSessionStore.getState().serverChatSelectionIntent).not.toBe(first)
    expect(JSON.parse(localStorage.getItem("tldw-playground-session")!).state)
      .not.toHaveProperty("serverChatSelectionIntent")
    usePlaygroundSessionStore.getState().clearSession()
    expect(usePlaygroundSessionStore.getState().serverChatSelectionIntent).toBeNull()
  })

  it("retiring a completed selection cannot clear its replacement intent", () => {
    const store = usePlaygroundSessionStore.getState()
    store.requestServerChatSelection("B")
    const completed = usePlaygroundSessionStore.getState().serverChatSelectionIntent
    store.requestServerChatSelection("B")
    const replacement = usePlaygroundSessionStore.getState().serverChatSelectionIntent
    store.completeServerChatSelection(completed)
    expect(usePlaygroundSessionStore.getState().serverChatSelectionIntent).toBe(replacement)
    store.completeServerChatSelection(replacement)
    expect(usePlaygroundSessionStore.getState().serverChatSelectionIntent).toBeNull()
  })

  it("rejects persisted sessions when the expected scope key changes", () => {
    usePlaygroundSessionStore.getState().saveSession({
      scopeKey: "scope:a",
      queuedMessages: [buildQueuedRequest({ promptText: "Run this later" })]
    })

    expect(usePlaygroundSessionStore.getState().isSessionValid("scope:a")).toBe(
      true
    )
    expect(usePlaygroundSessionStore.getState().isSessionValid("scope:b")).toBe(
      false
    )
  })

  it("records each accepted source intent without resetting its revision on session clear", () => {
    const initial = usePlaygroundSessionStore.getState().sourceSelectionRevision
    usePlaygroundSessionStore.getState().markSourceSelectionIntent()
    usePlaygroundSessionStore.getState().markSourceSelectionIntent()
    usePlaygroundSessionStore.getState().clearSession()
    expect(usePlaygroundSessionStore.getState().sourceSelectionRevision).toBe(initial + 2)
  })

  it("keeps source intent revision out of the persisted session payload", () => {
    usePlaygroundSessionStore.getState().markSourceSelectionIntent()
    usePlaygroundSessionStore.getState().saveSession({ scopeKey: "scope:a", ragMediaIds: [42] })
    const saved = JSON.parse(localStorage.getItem("tldw-playground-session")!)
    expect(saved.state.ragMediaIds).toEqual([42])
    expect(saved.state).not.toHaveProperty("sourceSelectionRevision")
    expect(saved.state).not.toHaveProperty("markSourceSelectionIntent")
  })
})

it("stores only an initialization address and clears it with the session", () => {
  const reference = { profile_id: "p", client_session_id: "a", owner_key: "owner", conversation_id: "chat" }
  usePlaygroundSessionStore.getState().saveSession({ historyId: "chat", historySelectionReference: reference })
  expect(usePlaygroundSessionStore.getState().historySelectionReference).toEqual(reference)
  usePlaygroundSessionStore.getState().clearSession()
  expect(usePlaygroundSessionStore.getState().historySelectionReference).toBeNull()
})


it.each([false, true])(
  "keeps only a valid Temporary session eligible beyond ordinary bookmark expiry (%s)",
  (temporary) => {
    const chatDraftHandle: ChatDraftHandle = {
      ownerHash: "a".repeat(64),
      address: {
        kind: "fresh",
        id: "age",
        scope: { type: "global" },
        temporary
      }
    }
    usePlaygroundSessionStore
      .getState()
      .saveSession({ scopeKey: "scope:a", chatDraftHandle })
    usePlaygroundSessionStore.setState({
      lastUpdated: Date.now() - 48 * 60 * 60 * 1000
    })
    expect(usePlaygroundSessionStore.getState().isSessionStale()).toBe(
      !temporary
    )
    expect(usePlaygroundSessionStore.getState().isSessionValid("scope:a")).toBe(
      temporary
    )
    expect(usePlaygroundSessionStore.getState().isSessionValid("scope:b")).toBe(
      false
    )
    usePlaygroundSessionStore.getState().clearSession()
    expect(usePlaygroundSessionStore.getState().isSessionValid("scope:a")).toBe(
      false
    )
  }
)
