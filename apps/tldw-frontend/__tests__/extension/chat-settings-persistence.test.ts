import { afterEach, beforeEach, expect, it, vi } from "vitest"

vi.mock("@/db/dexie/schema", async () => ({
  db: (await import("@/hooks/chat/__tests__/local-history-fixture")).memory
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {} }))
const key = "chatSettings:local:history-1"
const required = {
  schemaVersion: 2,
  updatedAt: "2026-09-17T00:00:00Z",
  assistantOverlay: { kind: "persona", id: "required-persona" }
}
let backing: Storage
beforeEach(async () => {
  vi.resetModules()
  backing = window.localStorage
  backing.clear()
  const { seedLocalFork, memory } =
    await import("@/hooks/chat/__tests__/local-history-fixture")
  vi.doMock("@/db/dexie/schema", () => ({ db: memory }))
  await seedLocalFork(false)
})
afterEach(() => {
  vi.restoreAllMocks()
  backing.clear()
})
const attemptFork = async () => {
  const fork = await import("@/db/dexie/branch")
  const { getLocalHistoryOwner } = await import("@/db/dexie/history-selection")
  const owner = await getLocalHistoryOwner("history-1")
  try {
    const input = await fork.captureLocalForkSelection(owner, {
      owner_key: owner.owner_key,
      conversation_id: "history-1",
      view_session_id: "view",
      selection_revision: 1,
      interpretation: { kind: "parent_graph_v1" },
      cursor: { kind: "after_message", message_id: "a1" }
    })
    const request: any = {
      operation_id: crypto.randomUUID(),
      owner_key: owner.owner_key,
      destination_owner_key: owner.owner_key,
      input
    }
    request.request_digest = fork.forkRequestDigest(request)
    return await fork.commitLocalFork(await fork.prepareLocalFork(request))
  } catch (error) {
    return {
      state: "rejected",
      code: error instanceof Error ? error.message : String(error)
    }
  }
}
it.each([
  undefined,
  { initialized: true, revision: "settled", pending: [] },
  { initialized: false, revision: "pending", pending: ["live-writer"] }
])(
  "rejects inaccessible persistence without changing guard %j",
  async (guard) => {
    const { memory } =
      await import("@/hooks/chat/__tests__/local-history-fixture")
    if (guard)
      await memory.chatHistories.update("history-1", {
        local_settings_guard: guard
      })
    backing.setItem(key, JSON.stringify(required))
    vi.spyOn(window, "localStorage", "get").mockImplementation(() => {
      throw new DOMException("Access denied", "SecurityError")
    })
    if (guard?.initialized) {
      const settings = await import("@/services/chat-settings")
      await settings.chatSettingsStorageForKey("local:history-1").set(key, null)
    }
    expect(await attemptFork()).toMatchObject({
      state: "rejected",
      code: "fork_chat_settings_unavailable"
    })
    expect(memory.chatHistories.rows.size).toBe(1)
    expect(
      (await memory.chatHistories.get("history-1")).local_settings_guard
    ).toEqual(guard)
    expect(JSON.parse(backing.getItem(key)!)).toEqual(required)
  }
)
it("does not certify a fallback instance after global storage recovers", async () => {
  const getter = vi
    .spyOn(window, "localStorage", "get")
    .mockImplementation(() => {
      throw new Error("denied")
    })
  await import("@/services/chat-settings")
  getter.mockRestore()
  backing.setItem(key, JSON.stringify(required))
  expect(await attemptFork()).toMatchObject({
    state: "rejected",
    code: "fork_chat_settings_unavailable"
  })
})
it.each([false, true])(
  "allows persistent empty baseline, initialized=%s",
  async (initialized) => {
    const { memory } =
      await import("@/hooks/chat/__tests__/local-history-fixture")
    if (initialized) {
      backing.setItem(key, "null")
      await memory.chatHistories.update("history-1", {
        local_settings_guard: {
          initialized: true,
          revision: "settled",
          pending: []
        }
      })
    }
    expect(await attemptFork()).toMatchObject({ state: "committed" })
    expect(backing.getItem(key)).toBe("null")
  }
)
it("preserves required persistent settings and rejects a plain fork", async () => {
  backing.setItem(key, JSON.stringify(required))
  expect(await attemptFork()).toMatchObject({
    state: "rejected",
    code: "unsupported_fork_chat_settings"
  })
  expect(JSON.parse(backing.getItem(key)!)).toEqual(required)
})
it("does not certify ordinary local writes through fallback", async () => {
  const { memory } =
    await import("@/hooks/chat/__tests__/local-history-fixture")
  vi.spyOn(window, "localStorage", "get").mockImplementation(() => {
    throw new Error("denied")
  })
  vi.spyOn(console, "error").mockImplementation(() => {})
  const settings = await import("@/services/chat-settings")
  expect(
    await settings.saveChatSettingsForKey("local:history-1", required as any)
  ).toBe(false)
  expect(
    (await memory.chatHistories.get("history-1")).local_settings_guard
  ).toBeUndefined()
})
