// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from "vitest"

import { buildQueuedRequest } from "@/utils/chat-request-queue"
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
