import { afterEach, describe, expect, it } from "vitest"

import { useACPSessionsStore } from "@/store/acp-sessions"

const resetStore = () => {
  useACPSessionsStore.getState().reset()
}

describe("ACP sessions persistence (partialize)", () => {
  afterEach(() => {
    resetStore()
  })

  it("excludes sensitive and transient fields from persisted state", () => {
    const state = useACPSessionsStore.getState()
    const sessionId = state.createSession({
      cwd: "/workspace/my-project",
      name: "Test Session",
    })

    // Set sensitive metadata on the session
    state.updateSessionMetadata(sessionId, {
      sshWsUrl: "wss://sandbox.example.com/ws",
      sshUser: "dev-user",
      sandboxSessionId: "sandbox-abc-123",
      sandboxRunId: "run-xyz-789",
    })

    // Verify the live session has the metadata set
    const liveSession = useACPSessionsStore.getState().getSession(sessionId)
    expect(liveSession?.sshWsUrl).toBe("wss://sandbox.example.com/ws")
    expect(liveSession?.sshUser).toBe("dev-user")
    expect(liveSession?.sandboxSessionId).toBe("sandbox-abc-123")
    expect(liveSession?.sandboxRunId).toBe("run-xyz-789")

    // Access the partialize function from the persist middleware
    const persistOptions = useACPSessionsStore.persist.getOptions()
    const partialize = persistOptions.partialize!
    const persisted = partialize(useACPSessionsStore.getState())

    const persistedSession = persisted.sessions[sessionId]

    // Sensitive fields must be nulled out
    expect(persistedSession.sshWsUrl).toBeNull()
    expect(persistedSession.sshUser).toBeNull()
    expect(persistedSession.sandboxSessionId).toBeNull()
    expect(persistedSession.sandboxRunId).toBeNull()

    // Transient state must be reset
    expect(persistedSession.updates).toEqual([])
    expect(persistedSession.state).toBe("disconnected")
    expect(persistedSession.pendingPermissions).toEqual([])

    // Metadata that should survive persistence
    expect(persistedSession.cwd).toBe("/workspace/my-project")
    expect(persistedSession.name).toBe("Test Session")
    expect(persistedSession.id).toBe(sessionId)
  })
})
