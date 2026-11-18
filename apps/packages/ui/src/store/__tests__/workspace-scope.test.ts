import { describe, it, expect } from "vitest"
import { sanitizeImportedChatSession } from "@/store/workspace-bundle"
import { validateCachedServerChatId } from "@/store/workspace-sync-contract"

describe("workspace import safety", () => {
  it("strips serverChatId during workspace import", () => {
    const session = {
      serverChatId: "server-123",
      messages: [],
      history: [],
    }
    const sanitized = sanitizeImportedChatSession(session)
    expect(sanitized.serverChatId).toBeNull()
  })

  it("detaches foreign local history identity while preserving portable conversation content", () => {
    const session = {
      serverChatId: "server-123",
      messages: [{ role: "user", content: "hi" }],
      history: [{ role: "user", content: "hi" }, { role: "assistant", content: "Cited answer [1]" }],
      historyId: "hist-1",
    }
    const sanitized = sanitizeImportedChatSession(session)
    expect(sanitized.messages).toEqual([{ role: "user", content: "hi" }])
    expect(sanitized.historyId).toBeNull()
    expect(sanitized.history).toEqual(session.history)
    expect(sanitized.serverChatId).toBeNull()
  })
})

describe("validateCachedServerChatId", () => {
  it("clears cached serverChatId when scope validation fails", () => {
    const validated = validateCachedServerChatId({
      cachedId: "server-123",
      serverScope: { scope_type: "global", workspace_id: null },
      expectedScope: { type: "workspace", workspaceId: "ws-a" },
    })
    expect(validated).toBeNull()
  })

  it("preserves cached serverChatId when scope matches (workspace)", () => {
    const validated = validateCachedServerChatId({
      cachedId: "server-123",
      serverScope: { scope_type: "workspace", workspace_id: "ws-a" },
      expectedScope: { type: "workspace", workspaceId: "ws-a" },
    })
    expect(validated).toBe("server-123")
  })

  it("preserves cached serverChatId when scope matches (global)", () => {
    const validated = validateCachedServerChatId({
      cachedId: "server-456",
      serverScope: { scope_type: "global", workspace_id: null },
      expectedScope: { type: "global" },
    })
    expect(validated).toBe("server-456")
  })

  it("returns null when cachedId is null", () => {
    const validated = validateCachedServerChatId({
      cachedId: null,
      serverScope: { scope_type: "global", workspace_id: null },
      expectedScope: { type: "global" },
    })
    expect(validated).toBeNull()
  })

  it("returns null when serverScope is null", () => {
    const validated = validateCachedServerChatId({
      cachedId: "server-123",
      serverScope: null,
      expectedScope: { type: "global" },
    })
    expect(validated).toBeNull()
  })

  it("rejects workspace chat when expected scope is different workspace", () => {
    const validated = validateCachedServerChatId({
      cachedId: "server-123",
      serverScope: { scope_type: "workspace", workspace_id: "ws-a" },
      expectedScope: { type: "workspace", workspaceId: "ws-b" },
    })
    expect(validated).toBeNull()
  })
})
